import numpy as np

def classify_signal(freqs, power_db):
    """Very simple heuristic classifier (stub)."""
    threshold = np.max(power_db) - 20
    occupied = freqs[power_db > threshold]
    bw = occupied[-1] - occupied[0] if len(occupied) else 0

    if bw < 3e6:
        return "Bluetooth (narrowband burst)"
    elif bw > 15e6:
        return "Wi-Fi (wideband OFDM)"
    else:
        return "Unknown"
