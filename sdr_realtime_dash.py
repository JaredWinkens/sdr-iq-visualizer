import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import numpy as np
import adi
import threading
import queue
import time
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SDRDataStreamer:
    """Class to handle SDR data streaming in a separate thread"""
    
    def __init__(self, uri="ip:192.168.2.1", sample_rate=1000000, center_freq=2400000000, 
                 rx_lo=2400000000, rx_rf_bandwidth=4000000, rx_buffer_size=2**12):
        """
        Initialize SDR parameters
        
        Args:
            uri: SDR device URI (default for PlutoSDR)
            sample_rate: Sample rate in Hz
            center_freq: Center frequency in Hz
            rx_lo: RX LO frequency in Hz
            rx_rf_bandwidth: RX RF bandwidth in Hz
            rx_buffer_size: Buffer size for RX
        """
        self.uri = uri
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.rx_lo = rx_lo
        self.rx_rf_bandwidth = rx_rf_bandwidth
        self.rx_buffer_size = rx_buffer_size
        
        self.sdr = None
        self.data_queue = queue.Queue(maxsize=100)  # Limit queue size to prevent memory issues
        self.running = False
        self.thread = None
        
    def connect(self):
        """Connect to the SDR device"""
        try:
            # Try to connect to PlutoSDR first, fallback to other devices
            self.sdr = adi.Pluto(uri=self.uri)
            
            # Configure RX parameters
            self.sdr.sample_rate = int(self.sample_rate)
            self.sdr.rx_rf_bandwidth = int(self.rx_rf_bandwidth)
            self.sdr.rx_lo = int(self.rx_lo)
            self.sdr.rx_buffer_size = self.rx_buffer_size
            
            logger.info(f"Connected to SDR at {self.uri}")
            logger.info(f"Sample Rate: {self.sdr.sample_rate}")
            logger.info(f"Center Frequency: {self.sdr.rx_lo}")
            logger.info(f"Bandwidth: {self.sdr.rx_rf_bandwidth}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to SDR: {e}")
            return False
    
    def start_streaming(self):
        """Start streaming data in a separate thread"""
        if not self.sdr:
            logger.error("SDR not connected")
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._stream_data)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Started SDR data streaming")
        return True
    
    def stop_streaming(self):
        """Stop streaming data"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        logger.info("Stopped SDR data streaming")
    
    def _stream_data(self):
        """Internal method to continuously read SDR data"""
        while self.running:
            try:
                # Read samples from SDR
                samples = self.sdr.rx()
                
                # Calculate FFT for frequency domain representation
                fft_data = np.fft.fftshift(np.fft.fft(samples))
                freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/self.sample_rate))
                freqs = freqs + self.center_freq  # Shift to actual frequencies
                
                # Calculate power spectrum (dB)
                power_db = 20 * np.log10(np.abs(fft_data) + 1e-10)
                
                # Prepare data for plotting
                plot_data = {
                    'time': time.time(),
                    'samples': samples,
                    'freqs': freqs,
                    'power_db': power_db,
                    'sample_rate': self.sample_rate,
                    'center_freq': self.center_freq
                }
                
                # Add to queue (non-blocking)
                try:
                    self.data_queue.put_nowait(plot_data)
                except queue.Full:
                    # Remove oldest data if queue is full
                    try:
                        self.data_queue.get_nowait()
                        self.data_queue.put_nowait(plot_data)
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                logger.error(f"Error reading SDR data: {e}")
                time.sleep(0.1)  # Brief pause before retrying
    
    def get_latest_data(self):
        """Get the latest data from the queue"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

# Global SDR streamer instance
sdr_streamer = SDRDataStreamer()

# Initialize Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Real-time SDR Data Visualization", style={'textAlign': 'center'}),
    
    html.Div([
        html.Button('Connect SDR', id='connect-btn', n_clicks=0, 
                   style={'margin': '10px', 'padding': '10px'}),
        html.Button('Start Streaming', id='start-btn', n_clicks=0, 
                   style={'margin': '10px', 'padding': '10px'}),
        html.Button('Stop Streaming', id='stop-btn', n_clicks=0, 
                   style={'margin': '10px', 'padding': '10px'}),
        html.Div(id='status-div', style={'margin': '10px', 'fontWeight': 'bold'})
    ], style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.H3("Time Domain (I/Q Samples)"),
            dcc.Graph(id='time-domain-graph')
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("Frequency Domain (Power Spectrum)"),
            dcc.Graph(id='freq-domain-graph')
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),
    
    html.Div([
        html.H3("Waterfall Plot (Spectrogram)"),
        dcc.Graph(id='waterfall-graph')
    ]),
    
    # Interval component for updating graphs
    dcc.Interval(
        id='interval-component',
        interval=300,  # Update every 300ms
        n_intervals=0,
        disabled=True
    )
])

# Global variables for waterfall plot
waterfall_data = deque(maxlen=100)  # Store last 100 FFT frames
waterfall_freqs = None

# Callback for connect button
@app.callback(
    Output('status-div', 'children'),
    Input('connect-btn', 'n_clicks'),
    prevent_initial_call=True
)
def connect_sdr(n_clicks):
    if n_clicks > 0:
        if sdr_streamer.connect():
            return "SDR Connected Successfully!"
        else:
            return "Failed to connect to SDR. Check connection and try again."
    return ""

# Callback for start button
@app.callback(
    Output('interval-component', 'disabled'),
    [Input('start-btn', 'n_clicks'),
     Input('stop-btn', 'n_clicks')],
    prevent_initial_call=True
)
def control_streaming(start_clicks, stop_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-btn' and start_clicks > 0:
        if sdr_streamer.start_streaming():
            return False  # Enable interval (start updating)
    elif button_id == 'stop-btn' and stop_clicks > 0:
        sdr_streamer.stop_streaming()
        return True  # Disable interval (stop updating)
    
    return True

# Callback for updating graphs
@app.callback(
    [Output('time-domain-graph', 'figure'),
     Output('freq-domain-graph', 'figure'),
     Output('waterfall-graph', 'figure')],
    Input('interval-component', 'n_intervals'),
    prevent_initial_call=True
)
def update_graphs(n):
    global waterfall_data, waterfall_freqs
    
    # Get latest data
    data = sdr_streamer.get_latest_data()
    
    if data is None:
        # Return empty figures if no data
        empty_fig = go.Figure()
        return empty_fig, empty_fig, empty_fig
    
    samples = data['samples']
    freqs = data['freqs']
    power_db = data['power_db']
    
    # Time domain plot (I/Q samples)
    time_samples = np.arange(len(samples)) / data['sample_rate'] * 1000  # Convert to ms
    
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
        name='Q (Imaginary)',
        line=dict(color='red')
    ))
    time_fig.update_layout(
        title='Time Domain - I/Q Samples',
        xaxis_title='Time (ms)',
        yaxis_title='Amplitude',
        showlegend=True
    )
    
    # Frequency domain plot (Power Spectrum)
    freq_fig = go.Figure()
    freq_fig.add_trace(go.Scatter(
        x=freqs / 1e6,  # Convert to MHz
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
    
    # Update waterfall data
    waterfall_data.append(power_db)
    if waterfall_freqs is None:
        waterfall_freqs = freqs / 1e6  # Convert to MHz
    
    # Waterfall plot (Spectrogram)
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

if __name__ == '__main__':
    print("Starting SDR Real-time Visualization Dashboard")
    print("Open your browser and go to: http://127.0.0.1:8050")
    print("\nInstructions:")
    print("1. Click 'Connect SDR' to establish connection")
    print("2. Click 'Start Streaming' to begin data acquisition")
    print("3. Click 'Stop Streaming' to stop data acquisition")
    print("\nNote: Make sure your SDR device is connected and accessible")
    
    app.run(debug=True, host='127.0.0.1', port=8050)