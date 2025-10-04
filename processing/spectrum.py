import numpy as np

def compute_spectrum(samples, sample_rate, center_freq):
    """Compute FFT power spectrum."""
    fft_data = np.fft.fftshift(np.fft.fft(samples))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/sample_rate))
    freqs += center_freq
    power_db = 20 * np.log10(np.abs(fft_data) + 1e-10)
    return freqs, power_db
