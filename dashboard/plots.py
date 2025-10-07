import numpy as np
import plotly.graph_objs as go

def plot_time(samples, sample_rate):
    t = np.arange(len(samples)) / sample_rate * 1000  # ms
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=np.real(samples), mode='lines', name='I', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=t, y=np.imag(samples), mode='lines', name='Q', line=dict(color='red')))
    fig.update_layout(title="Time Domain", xaxis_title="Time (ms)", yaxis_title="Amplitude")
    return fig

def plot_freq(freqs, power_db):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs/1e6, y=power_db, mode='lines', name='Power', line=dict(color='green')))
    fig.update_layout(title="Frequency Domain", xaxis_title="Frequency (MHz)", yaxis_title="Power (dB)")
    return fig

def plot_waterfall(power_db, freqs, history, max_frames=100):
    history.append(power_db)
    if len(history) > max_frames:
        history.pop(0)

    fig = go.Figure()
    if history:
        z = np.array(history)
        fig.add_trace(go.Heatmap(
            z=z,
            x=freqs/1e6,
            y=list(range(len(history))),
            colorscale='Viridis',
            colorbar=dict(title="Power (dB)")
        ))
    fig.update_layout(title="Waterfall (Spectrogram)", xaxis_title="Frequency (MHz)", yaxis_title="Frames")
    return fig
