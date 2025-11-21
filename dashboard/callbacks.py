# dashboard/callbacks.py

import numpy as np
import plotly.graph_objs as go
from dash import Input, Output, callback_context, no_update, dcc
from collections import deque
import logging
import json
import io
import zipfile
from datetime import datetime
import dash_bootstrap_components as dbc
from sdr.streamer import sdr_streamer
from processing.classifier import classify_signal_advanced

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
         Output('waterfall-graph', 'figure'),
         Output('constellation-plot', 'figure'),
         Output('pause-status', 'children'),
         Output('classification-text', 'children')],
        [Input('interval-component', 'n_intervals'),
         Input('pause-toggle', 'value')],
        prevent_initial_call=True
    )
    def update_graphs(n, pause_value):
        global waterfall_data, waterfall_freqs

        # Handle pause
        paused = 1 in (pause_value or [])
        if paused:
            return no_update, no_update, no_update, no_update, "Paused", no_update

        # Get the newest SDR data
        data = sdr_streamer.get_latest_data()

        if data is None:
            empty = go.Figure()
            return empty, empty, empty, empty, "Waiting for data...", ""

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
        # Peak detection (use scipy if available, fallback to simple method)
        peaks_idx = []
        try:
            from scipy.signal import find_peaks
            # dynamic distance to avoid clutter; prominence for meaningful peaks
            dist = max(5, len(power_db)//200)
            peaks_idx, _ = find_peaks(power_db, distance=dist, prominence=3)
        except Exception:
            # Fallback simple local maxima above median+5dB
            med = float(np.median(power_db))
            for i in range(1, len(power_db)-1):
                if power_db[i] > power_db[i-1] and power_db[i] > power_db[i+1] and power_db[i] > med + 5:
                    peaks_idx.append(i)
        if len(peaks_idx):
            freq_fig.add_trace(go.Scatter(
                x=(freqs[peaks_idx] / 1e6),
                y=power_db[peaks_idx],
                mode='markers',
                name='Peaks',
                marker=dict(color='orange', size=8, symbol='x')
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

        # Constellation diagram
        constellation_fig = go.Figure()
        # Randomly subsample if too many points for performance
        if len(samples) > 2000:
            idx = np.random.choice(len(samples), 2000, replace=False)
            plot_real = np.real(samples)[idx]
            plot_imag = np.imag(samples)[idx]
        else:
            plot_real = np.real(samples)
            plot_imag = np.imag(samples)
        constellation_fig.add_trace(go.Scatter(
            x=plot_real,
            y=plot_imag,
            mode='markers',
            marker=dict(size=4, color='purple', opacity=0.5),
            name='Constellation'
        ))
        constellation_fig.update_layout(
            title='Constellation Diagram',
            xaxis_title='In-Phase (I)',
            yaxis_title='Quadrature (Q)',
            showlegend=False,
            height=400,
            xaxis=dict(scaleanchor="y", scaleratio=1)  # Square aspect ratio
        )

        # Classification
        try:
            res = classify_signal_advanced(freqs, power_db)
            label = res.get('label', 'Unknown')
            conf = res.get('confidence', 0.0)
            feats = res.get('features', {})
            bw20 = feats.get('bandwidth_hz_20db', 0.0) / 1e6
            snr = feats.get('snr_db', 0.0)
            flat = feats.get('spectral_flatness', 0.0)
            kurt = feats.get('spectral_kurtosis', 0.0)
            peaks = feats.get('peak_count', 0)
            explain = res.get('explanation', '')
            cls_text = (
                f"Detected: {label} (conf {conf:.2f}) — OBW20={bw20:.2f}MHz SNR={snr:.1f}dB | Flat {flat:.2f} | Kurt {kurt:.2f} | Peaks {peaks}"
                f"\n{explain}"  # newline instead of HTML span for cleaner rendering
            )
        except Exception as e:
            cls_text = f"Classification unavailable: {e}"

        return time_fig, freq_fig, waterfall_fig, constellation_fig, "", cls_text
    
    
    # ---- Download Sample of Live Feed ----
    @app.callback(
        Output('sample-download-location', 'data'),
        Output('alert-div', 'children', allow_duplicate=True),
        Input('download-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def download_sample_data(n_clicks):
        """
        Download a sample of live SDR data in SigMF format.
        Creates both .sigmf-data (binary IQ) and .sigmf-meta (JSON metadata)
        packaged together in a zip file for easy sharing and archival.
        """
        if n_clicks == 0:
            return no_update, no_update
        
        # Get current data from stream
        data = sdr_streamer.get_latest_data()
        
        if data is None:
            logger.warning("No data available for download")
            return no_update, dbc.Alert(
                                    "No data available for download. Please start streaming.",
                                    id="alert-auto",
                                    is_open=True,
                                    duration=4000,
                                    color='danger',
                                    fade=True
                                ),
        
        samples = data['samples']
        sample_rate = data['sample_rate']
        center_freq = data['center_freq']
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"sdr_sample_{timestamp}"
        
        # Create metadata dictionary (SigMF format)
        metadata = {
            "global": {
                "core:datatype": "cf32_le",  # complex float32 little-endian
                "core:sample_rate": int(sample_rate),
                "core:version": "1.0.0",
                "core:description": "SDR live stream sample from IQ Visualizer",
                "core:author": "SDR IQ Visualizer Dashboard",
                "core:recorder": "PlutoSDR via pyadi-iio",
                "core:hw": f"PlutoSDR @ {sdr_streamer.uri}",
                "core:license": "CC0-1.0"
            },
            "captures": [
                {
                    "core:sample_start": 0,
                    "core:frequency": int(center_freq),
                    "core:datetime": datetime.utcnow().isoformat() + 'Z'
                }
            ],
            "annotations": []
        }
        
        # Convert samples to binary format (complex64 = cf32)
        if samples.dtype != np.complex64:
            samples = samples.astype(np.complex64)
        
        binary_data = samples.tobytes()
        metadata_json = json.dumps(metadata, indent=2)
        
        # Create a zip file containing both .sigmf-data and .sigmf-meta
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add binary data file
            zip_file.writestr(f"{base_filename}.sigmf-data", binary_data)
            # Add metadata JSON file
            zip_file.writestr(f"{base_filename}.sigmf-meta", metadata_json)
            # Add a README for user convenience
            readme_content = f"""SigMF Recording from SDR IQ Visualizer
            =====================================

            Timestamp: {timestamp}
            Sample Rate: {sample_rate/1e6:.2f} MHz
            Center Frequency: {center_freq/1e6:.2f} MHz
            Number of Samples: {len(samples)}
            Duration: {len(samples)/sample_rate:.3f} seconds

            Files:
            - {base_filename}.sigmf-data: Binary IQ samples (complex float32)
            - {base_filename}.sigmf-meta: SigMF metadata (JSON)

            To use with GNU Radio, inspectrum, or other tools:
            1. Extract both files to the same directory
            2. Keep the base filename the same
            3. Open the .sigmf-data file in your SDR tool

            SigMF Format: https://github.com/gnuradio/SigMF
            """
            zip_file.writestr("README.txt", readme_content)
        
        zip_buffer.seek(0)
        
        logger.info(f"Generated SigMF download: {base_filename}.zip ({len(samples)} samples)")
        
        return dcc.send_bytes(
            zip_buffer.getvalue(),
            filename=f"{base_filename}_sigmf.zip"
        ), no_update


