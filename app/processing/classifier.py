import numpy as np
from collections import Counter, deque

# Maintain a short rolling history for temporal smoothing
_CLASS_HISTORY = deque(maxlen=12)  # about last ~12 frames
_CONF_HISTORY = deque(maxlen=12)

"""Advanced signal classification utilities.

Provides two entry points:
    - classify_signal_simple(freqs, power_db): legacy string label
    - classify_signal_advanced(freqs, power_db): rich dict with features and explanation
"""

def classify_signal_simple(freqs, power_db):
    if len(freqs) == 0:
        return "No Data"
    threshold = np.max(power_db) - 20
    mask = power_db > threshold
    if not np.any(mask):
        return "Noise"
    occupied = freqs[mask]
    bw = occupied[-1] - occupied[0]
    if bw < 3e6:
        return "Narrowband"
    elif bw > 15e6:
        return "Wideband"
    return "Unknown"

def classify_signal_advanced(freqs, power_db):
    """Return dict(label, confidence, features{...}, explanation:str).

    Features include:
      - bandwidth_hz_3db, bandwidth_hz_10db, bandwidth_hz_20db
      - snr_db (peak vs robust noise floor)
      - spectral_flatness (0..1)
      - spectral_kurtosis
      - peak_count
      - peak_spacing_std_hz (0 for <2 peaks)
    """
    if len(freqs) == 0 or len(power_db) == 0:
        return {"label": "No Data", "confidence": 0.0, "features": {}, "explanation": "No spectrum data"}

    # Core measurements
    noise_floor_db = _estimate_noise_floor(power_db)
    snr_db = float(np.max(power_db) - noise_floor_db)
    bw3 = _occupied_bandwidth(freqs, power_db, drop_db=3)
    bw10 = _occupied_bandwidth(freqs, power_db, drop_db=10)
    bw20 = _occupied_bandwidth(freqs, power_db, drop_db=20)
    sfm = _spectral_flatness(power_db)
    kurt = _spectral_kurtosis(power_db)
    # Adaptive threshold: median + max(5dB, 0.1*SNR) to reduce noisy peak proliferation
    adaptive_thr = max(noise_floor_db + 5.0, np.max(power_db) - 0.9 * snr_db + 5.0)
    peak_idx = _find_peaks(power_db, threshold_db=adaptive_thr, min_distance_bins=max(3, len(power_db)//300))
    peak_count = int(len(peak_idx))
    spacing_std = _peak_spacing_std(freqs, peak_idx)
    peak_density = peak_count / max(len(power_db), 1)
    mid_freq = float((freqs[0] + freqs[-1]) / 2.0)

    label = "Unknown"
    confidence = 0.25
    reasons = []

    # Primary rules
    span_hz = float(freqs[-1] - freqs[0]) if len(freqs) else 1.0
    occ_ratio = bw20 / span_hz if span_hz > 0 else 0.0

    # Rule blocks (ordered)
    if snr_db < 3:
        label = "Low SNR / Noise"
        confidence = 0.45
        reasons.append(f"Low SNR ({snr_db:.1f} dB) below 3 dB threshold")
    elif sfm > 0.85 and snr_db < 8 and occ_ratio > 0.5:
        label = "Broadband Noise / Hash"
        confidence = 0.55
        reasons.append(f"High spectral flatness ({sfm:.2f}) with moderate SNR and broad occupancy ({occ_ratio:.2f})")
    elif peak_count == 1 and bw20 < 60e3 and sfm < 0.4:
        label = "CW Carrier"
        confidence = 0.8 if snr_db > 6 else 0.6
        reasons.append(f"Single strong peak, OBW20 {bw20/1e3:.0f} kHz, flatness {sfm:.2f}")
    elif 2 <= peak_count <= 4 and bw20 < 600e3 and sfm < 0.55:
        label = "Multitone / FSK-like"
        confidence = 0.7 if snr_db > 6 else 0.55
        reasons.append(f"Few peaks ({peak_count}) with narrow OBW20 {bw20/1e3:.0f} kHz and low flatness {sfm:.2f}")
    elif 88e6 <= mid_freq <= 108e6 and 110e3 <= bw20 <= 300e3 and 0.15 < sfm < 0.6 and snr_db > 8:
        label = "FM Broadcast (candidate)"
        confidence = 0.78
        reasons.append("In FM band with plausible OBW and features")
    elif bw20 > 10e6 and 0.25 < sfm < 0.9 and peak_density > 0.02 and spacing_std / max(bw20,1) < 0.12:
        label = "Wideband OFDM / Multi-carrier"
        confidence = 0.82 if peak_count > 20 else 0.7
        reasons.append(f"Wide OBW {bw20/1e6:.1f} MHz with many peaks ({peak_count}) and regular spacing")
    elif bw20 < 600e3 and snr_db > 4:
        # Narrowband generic bucket
        if peak_count <= 2 and sfm < 0.5:
            label = "Narrowband (voice)"
            confidence = 0.65
            reasons.append("Narrow OBW with few peaks and low flatness (voice-like)")
        elif peak_count > 4:
            label = "Channelized Narrowband"
            confidence = 0.6
            reasons.append("Narrow OBW with multiple peaks (channelized)")
        else:
            label = "Narrowband"
            confidence = 0.55
            reasons.append("Narrow OBW with moderate features")
    elif occ_ratio > 0.6 and snr_db > 6 and peak_density < 0.01 and (0.4 < sfm < 0.8):
        # Fallback structured wideband
        label = "Wideband Structured"
        confidence = 0.55
        reasons.append("High occupancy with structured spectrum (not noise)")

    # Fallback: avoid "Unknown" if we have strong SNR; classify generically
    if label == "Unknown":
        if snr_db > 10 and bw20 < 1e6:
            label = "Narrowband (generic)"
            confidence = max(confidence, 0.5)
            reasons.append("Fallback: strong SNR and narrow OBW")
        elif snr_db > 10 and bw20 > 5e6:
            label = "Wideband (generic)"
            confidence = max(confidence, 0.5)
            reasons.append("Fallback: strong SNR and wide OBW")

    # Temporal smoothing
    _CLASS_HISTORY.append(label)
    _CONF_HISTORY.append(confidence)
    counts = Counter(_CLASS_HISTORY)
    most_label, most_count = counts.most_common(1)[0]
    stability = most_count / len(_CLASS_HISTORY)
    if stability >= 0.5 and most_label != label:
        # Smooth jump suppression: adopt stable label, blend confidence
        smoothed_conf = (np.mean([c for l, c in zip(_CLASS_HISTORY, _CONF_HISTORY) if l == most_label]) + confidence) / 2
        label = most_label
        confidence = min(0.95, max(confidence, smoothed_conf + 0.05 * stability))
        reasons.append(f"Temporal smoothing applied; adopting stable label '{most_label}' (stability {stability:.2f})")
    else:
        # Slight boost for stability of current label occurrences
        label_occ = counts[label]
        confidence = min(0.95, confidence + 0.05 * (label_occ / len(_CLASS_HISTORY)))

    explanation = (
        f"SNR={snr_db:.1f} dB | peaks={peak_count} (density {peak_density:.3f}) | flat={sfm:.2f} | kurt={kurt:.2f} "
        f"| OBW20={bw20/1e6:.2f} MHz (OBW3={bw3/1e6:.3f} MHz) | spacingσ={spacing_std/1e3:.1f} kHz | stability={stability if 'stability' in locals() else 0:.2f}"
    )

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "features": {
            "bandwidth_hz_3db": float(bw3),
            "bandwidth_hz_10db": float(bw10),
            "bandwidth_hz_20db": float(bw20),
            "snr_db": float(round(snr_db, 2)),
            "spectral_flatness": float(round(sfm, 3)),
            "spectral_kurtosis": float(round(kurt, 3)),
            "peak_count": int(peak_count),
            "peak_spacing_std_hz": float(spacing_std)
        },
        "explanation": explanation,
        "reasons": reasons
    }

def _occupied_bandwidth(freqs, power_db, drop_db=20):
    peak = np.max(power_db)
    thr = peak - float(drop_db)
    mask = power_db >= thr
    if not np.any(mask):
        return 0.0
    occ = freqs[mask]
    return float(occ[-1] - occ[0])

def _estimate_snr(power_db):
    if len(power_db) < 4:
        return 0.0
    signal_lvl = np.percentile(power_db, 95)
    noise_lvl = np.median(power_db)
    return signal_lvl - noise_lvl

def _estimate_noise_floor(power_db):
    # Robust: 20th percentile approximates noise floor in mixed spectra
    return float(np.percentile(power_db, 20))

def _spectral_flatness(power_db):
    # Convert dB to linear power
    p = np.power(10.0, np.asarray(power_db, dtype=float) / 10.0)
    p = np.clip(p, 1e-15, None)
    geo_mean = float(np.exp(np.mean(np.log(p))))
    arith_mean = float(np.mean(p))
    return float(np.clip(geo_mean / arith_mean, 0.0, 1.0))

def _spectral_kurtosis(power_db):
    x = np.asarray(power_db, dtype=float)
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < 1e-9:
        return 0.0
    z4 = np.mean(((x - mu) / sigma) ** 4)
    return float(z4)

def _find_peaks(power_db, threshold_db, min_distance_bins=5):
    x = np.asarray(power_db, dtype=float)
    n = len(x)
    if n < 3:
        return []
    peaks = []
    last_idx = -min_distance_bins
    for i in range(1, n - 1):
        if x[i] > threshold_db and x[i] > x[i - 1] and x[i] > x[i + 1]:
            if i - last_idx >= min_distance_bins:
                peaks.append(i)
                last_idx = i
    return peaks

def _peak_spacing_std(freqs, peak_idx):
    if len(peak_idx) < 3:
        return 0.0
    pf = np.asarray(freqs)[peak_idx]
    diffs = np.diff(pf)
    return float(np.std(diffs))
