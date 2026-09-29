"""RF impairment augmentation for IQ training data.

Implements the robustness upgrade pipeline from the masters project proposal
(Phase 2 — "Data augmentation with controlled impairments").

Each function takes a complex numpy array (or a (2, N) float32 array of
[I, Q] rows) and returns the same shape with the impairment applied.

AugmentationPipeline wraps them into a callable that can be dropped
directly into RadioMLDataset / SigMFDataset as a ``transform``.

Supported impairments
---------------------
- AWGN          : additive white Gaussian noise at a target SNR
- FrequencyOffset: constant carrier frequency offset (normalised to sample rate)
- PhaseShift    : constant phase rotation
- AmplitudeScale: random amplitude scaling (channel fading proxy)
- DCOffset      : small DC bias on I and/or Q (hardware imperfection)

Usage
-----
from app.processing.augmentation import AugmentationPipeline
from app.processing.ml_classifier import RadioMLDataset, train

pipeline = AugmentationPipeline(
    snr_db_range=(-5, 15),   # inject AWGN so effective SNR lands in this range
    freq_offset_range=(-0.05, 0.05),  # ±5% of sample rate
    phase_shift_range=(-3.14159, 3.14159),
    amplitude_scale_range=(0.8, 1.2),
    dc_offset_std=0.01,
    p_apply=0.8,             # probability of applying the full pipeline per sample
)

dataset = RadioMLDataset("RML2016.10a_dict.pkl", transform=pipeline)
train(dataset, out_path="models/augmented_cnn.pt", epochs=20)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Individual impairment functions
# All operate on complex64 numpy arrays of shape (N,)
# ---------------------------------------------------------------------------

def awgn(iq: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise to reach a target SNR.

    Parameters
    ----------
    iq:
        Complex IQ samples, shape (N,).
    snr_db:
        Desired signal-to-noise ratio in dB.  The signal power is measured
        from the input; noise is scaled to achieve the requested SNR.

    Returns
    -------
    Noisy complex array, same shape and dtype as input.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    signal_power = float(np.mean(np.abs(iq) ** 2))
    if signal_power < 1e-12:
        return iq  # silence — nothing to add noise to
    noise_power = np.float32(signal_power / (10.0 ** (snr_db / 10.0)))
    std = np.float32(np.sqrt(noise_power / 2.0))
    noise = (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq))).astype(np.complex64) * std
    return iq + noise


def frequency_offset(iq: np.ndarray, offset_norm: float) -> np.ndarray:
    """Apply a constant carrier frequency offset.

    Parameters
    ----------
    iq:
        Complex IQ samples, shape (N,).
    offset_norm:
        Normalised frequency offset as a fraction of the sample rate.
        e.g. 0.05 → 5% of Fs.  Typical realistic range: ±0.05.

    Returns
    -------
    Frequency-shifted complex array.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    n = np.arange(len(iq), dtype=np.float32)
    phasor = np.exp(1j * 2.0 * np.pi * offset_norm * n).astype(np.complex64)
    return iq * phasor


def phase_shift(iq: np.ndarray, theta: float) -> np.ndarray:
    """Apply a constant phase rotation.

    Parameters
    ----------
    iq:
        Complex IQ samples, shape (N,).
    theta:
        Phase offset in radians.  A uniform random draw from [−π, π]
        covers all phase ambiguities.

    Returns
    -------
    Phase-rotated complex array.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    return iq * np.exp(1j * float(theta)).astype(np.complex64)


def amplitude_scale(iq: np.ndarray, scale: float) -> np.ndarray:
    """Multiply by a scalar amplitude factor (simple fading proxy).

    Parameters
    ----------
    iq:
        Complex IQ samples, shape (N,).
    scale:
        Multiplicative gain.  Values in [0.8, 1.2] represent mild fading.

    Returns
    -------
    Scaled complex array.
    """
    return np.asarray(iq, dtype=np.complex64) * np.float32(scale)


def dc_offset(iq: np.ndarray, i_bias: float, q_bias: float) -> np.ndarray:
    """Add a small DC bias to I and Q channels (hardware imperfection).

    Parameters
    ----------
    iq:
        Complex IQ samples, shape (N,).
    i_bias, q_bias:
        Bias values added to real and imaginary parts respectively.
        Typical magnitude: ≤ 0.02 × RMS signal amplitude.

    Returns
    -------
    Biased complex array.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    return iq + np.complex64(i_bias + 1j * q_bias)


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def _to_complex(x: np.ndarray) -> np.ndarray:
    """Accept either (N,) complex or (2, N) float [I, Q] → complex64 (N,)."""
    if np.iscomplexobj(x):
        return x.astype(np.complex64)
    if x.ndim == 2 and x.shape[0] == 2:
        return (x[0] + 1j * x[1]).astype(np.complex64)
    raise ValueError(f"Unexpected IQ shape: {x.shape}")


def _to_iq2d(x: np.ndarray) -> np.ndarray:
    """Convert complex (N,) → float32 (2, N) [I, Q] rows."""
    return np.stack([x.real, x.imag], axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class AugmentationPipeline:
    """Randomised RF impairment pipeline usable as a dataset transform.

    Each call draws fresh random parameters within the specified ranges and
    applies the enabled impairments in order:
        AWGN → frequency offset → phase shift → amplitude scale → DC offset

    Parameters
    ----------
    snr_db_range:
        (lo, hi) range for the injected AWGN SNR in dB.
        Set to None to skip AWGN.
    freq_offset_range:
        (lo, hi) range for the normalised frequency offset.
        Set to None to skip.
    phase_shift_range:
        (lo, hi) range for the phase rotation in radians.
        Defaults to (−π, π) which covers full rotation ambiguity.
        Set to None to skip.
    amplitude_scale_range:
        (lo, hi) range for the amplitude multiplier.
        Set to None to skip.
    dc_offset_std:
        Standard deviation of the Gaussian used to draw I/Q biases.
        Set to 0 or None to skip.
    p_apply:
        Probability of applying the full pipeline to any given sample.
        1.0 = always augment; 0.5 = augment half the time.
    seed:
        Optional fixed seed for reproducible augmentation (useful in tests).
    """

    snr_db_range: Optional[Tuple[float, float]] = (-5.0, 15.0)
    freq_offset_range: Optional[Tuple[float, float]] = (-0.05, 0.05)
    phase_shift_range: Optional[Tuple[float, float]] = (-np.pi, np.pi)
    amplitude_scale_range: Optional[Tuple[float, float]] = (0.8, 1.2)
    dc_offset_std: Optional[float] = 0.01
    p_apply: float = 0.8
    seed: Optional[int] = None

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the pipeline to one IQ frame.

        Parameters
        ----------
        x:
            Either complex64 of shape (N,) or float32 of shape (2, N).

        Returns
        -------
        Augmented array in the same format as the input.
        """
        was_2d = (not np.iscomplexobj(x)) and x.ndim == 2
        iq = _to_complex(x)

        if self._rng.random() > self.p_apply:
            # Skip augmentation this sample — return original format
            return _to_iq2d(iq) if was_2d else iq

        # AWGN
        if self.snr_db_range is not None:
            lo, hi = self.snr_db_range
            snr = float(self._rng.uniform(lo, hi))
            iq = awgn(iq, snr)

        # Frequency offset
        if self.freq_offset_range is not None:
            lo, hi = self.freq_offset_range
            offset = float(self._rng.uniform(lo, hi))
            iq = frequency_offset(iq, offset)

        # Phase shift
        if self.phase_shift_range is not None:
            lo, hi = self.phase_shift_range
            theta = float(self._rng.uniform(lo, hi))
            iq = phase_shift(iq, theta)

        # Amplitude scale
        if self.amplitude_scale_range is not None:
            lo, hi = self.amplitude_scale_range
            scale = float(self._rng.uniform(lo, hi))
            iq = amplitude_scale(iq, scale)

        # DC offset
        if self.dc_offset_std and self.dc_offset_std > 0:
            i_bias = float(self._rng.normal(0.0, self.dc_offset_std))
            q_bias = float(self._rng.normal(0.0, self.dc_offset_std))
            iq = dc_offset(iq, i_bias, q_bias)

        return _to_iq2d(iq) if was_2d else iq

    # ------------------------------------------------------------------
    # Convenience: describe current config for logging / reproducibility
    # ------------------------------------------------------------------

    def describe(self) -> dict:
        return {
            "snr_db_range": self.snr_db_range,
            "freq_offset_range": self.freq_offset_range,
            "phase_shift_range": self.phase_shift_range,
            "amplitude_scale_range": self.amplitude_scale_range,
            "dc_offset_std": self.dc_offset_std,
            "p_apply": self.p_apply,
        }
