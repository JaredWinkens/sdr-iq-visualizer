# dashboard/callbacks.py

import numpy as np
import plotly.graph_objs as go
from dash import Input, Output, callback_context
from collections import deque
import logging

from sdr.streamer import sdr_streamer

logger = logging.getLogger(__name__)

# Keep waterfall history (last 100 frames)
waterfall_data = deque(maxlen=100)
waterfall_freqs = None


def register_callbacks(app):
    """Attach all callbacks to the Dash app."""

    # ---- Connect SDR ----
    @app.callback(
        Output('status-div', 'children'),
        [Input('connect-btn', 'n_clicks'),
         Input('reconnect-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def connect_sdr(connect_clicks, reconnect_clicks):
        ctx = callback_context
        if not ctx.triggered:
            return ""
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'connect-btn' and connect_clicks > 0:
            if sdr_streamer.connect():
                logger.info(f"Connected to SDR at {sdr_streamer.uri}")
                return "SDR Connected Successfully!"
            else:
                logger.error("Failed to connect to SDR. Check connection and try again.")
                return "Failed to connect to SDR. Check console for details."
        elif button_id == 'reconnect-btn' and reconnect_clicks > 0:
            # Stop streaming if it's running
            if sdr_streamer.running:
                sdr_streamer.stop_streaming()
            
            if sdr_streamer.reconnect():
                logger.info(f"Reconnected to SDR at {sdr_streamer.uri}")
                return "SDR Reconnected Successfully!"
            else:
                logger.error("Failed to reconnect to SDR. Check connection and try again.")
                return "Failed to reconnect to SDR. Check console for details."
        
        return ""

    # ---- Start / Stop Streaming ----
    @app.callback(
        Output('interval-component', 'disabled'),
        [Input('start-btn', 'n_clicks'),
         Input('stop-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def control_streaming(start_clicks, stop_clicks):
        ctx = callback_context
        if not ctx.triggered:
            return True

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'start-btn' and start_clicks > 0:
            if sdr_streamer.start_streaming():
                return False  # enable interval updates
        elif button_id == 'stop-btn' and stop_clicks > 0:
            sdr_streamer.stop_streaming()
            return True  # disable interval updates

        return True

    # ---- Update Graphs ----
    @app.callback(
        [Output('time-domain-graph', 'figure'),
         Output('freq-domain-graph', 'figure'),
         Output('waterfall-graph', 'figure')],
        Input('interval-component', 'n_intervals'),
        prevent_initial_call=True
    )
    def update_graphs(n):
        global waterfall_data, waterfall_freqs

        # Get the newest SDR data
        data = sdr_streamer.get_latest_data()

        if data is None:
            empty = go.Figure()
            return empty, empty, empty

        samples = data['samples']
        freqs = data['freqs']
        power_db = data['power_db']

        # --- Time Domain Plot ---
        time_samples = np.arange(len(samples)) / data['sample_rate'] * 1000  # ms
        time_fig = go.Figure()
        time_fig.add_trace(go.Scatter(
            x=time_samples,
            y=np.real(samples),
            mode='lines',
            name='I (Real)',
            line=dict(color='blue')
        ))
        time_fig.add_trace(go.Scatter(
            x=time_samples,
            y=np.imag(samples),
            mode='lines',
            name='Q (Imag)',
            line=dict(color='red')
        ))
        time_fig.update_layout(
            title='Time Domain - I/Q Samples',
            xaxis_title='Time (ms)',
            yaxis_title='Amplitude',
            showlegend=True
        )

        # --- Frequency Domain Plot ---
        freq_fig = go.Figure()
        freq_fig.add_trace(go.Scatter(
            x=freqs / 1e6,
            y=power_db,
            mode='lines',
            name='Power Spectrum',
            line=dict(color='green')
        ))
        freq_fig.update_layout(
            title='Frequency Domain - Power Spectrum',
            xaxis_title='Frequency (MHz)',
            yaxis_title='Power (dB)',
            showlegend=False
        )

        # --- Waterfall Plot ---
        waterfall_data.append(power_db)
        if waterfall_freqs is None:
            waterfall_freqs = freqs / 1e6

        waterfall_fig = go.Figure()
        if len(waterfall_data) > 1:
            waterfall_array = np.array(waterfall_data)
            waterfall_fig.add_trace(go.Heatmap(
                z=waterfall_array,
                x=waterfall_freqs,
                y=list(range(len(waterfall_data))),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Power (dB)")
            ))
        waterfall_fig.update_layout(
            title='Waterfall Plot - Spectrogram',
            xaxis_title='Frequency (MHz)',
            yaxis_title='Time (frames)',
            height=400
        )

        return time_fig, freq_fig, waterfall_fig
