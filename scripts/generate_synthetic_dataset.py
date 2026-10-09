#!/usr/bin/env python3
"""Generate a synthetic labeled IQ dataset for modulation classification.

Produces AM-DSB, FM, BPSK, and QPSK signals at multiple SNR levels and saves
them in RadioML 2016.10a-compatible pickle format so train_classifier.py and
run_experiment.py work without needing the real RadioML dataset downloaded.

Each signal is generated from first principles using standard digital
communications math — no external SDR hardware required.

Usage
-----
# Default: 1000 examples per (modulation, SNR) pair, SNR -20 to 18 dB step 2
python scripts/generate_synthetic_dataset.py --out data/synthetic_rml.pkl

# Quick smoke-test size
python scripts/generate_synthetic_dataset.py \
    --out data/synthetic_rml.pkl \
    --n-per-class 100 \
    --snr-min -10 --snr-max 10 --snr-step 5

Output format
-------------
dict keyed by (modulation_str, snr_int) → ndarray shape (N, 2, 128)
  axis 0: example index
  axis 1: [I, Q] channels
  axis 2: 128 samples per window
Compatible with RadioMLDataset in app/processing/ml_classifier.py.
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

WINDOW = 128        # samples per frame — matches WINDOW_SIZE in ml_classifier.py
FS = 1.0            # normalised sample rate (1 = Nyquist)
FC = 0.1            # normalised carrier frequency for all modulations


# ---------------------------------------------------------------------------
# Signal generators — each returns complex64 array of length WINDOW
# ---------------------------------------------------------------------------

def _am_dsb(rng: np.random.Generator) -> np.ndarray:
    """AM-DSB: x(t) = (1 + m·msg(t)) · cos(2π·fc·t)
    Modulation index m drawn uniformly from [0.5, 1.0].
    Message is a low-frequency sinusoid with random phase.
    """
    t = np.arange(WINDOW, dtype=np.float32)
    m = rng.uniform(0.5, 1.0)
    fm = rng.uniform(0.01, 0.05)          # message frequency (normalised)
    phi_msg = rng.uniform(0, 2 * np.pi)
    msg = np.cos(2 * np.pi * fm * t + phi_msg).astype(np.float32)

    phi_c = rng.uniform(0, 2 * np.pi)
    carrier = np.exp(1j * (2 * np.pi * FC * t + phi_c)).astype(np.complex64)
    envelope = (1.0 + m * msg).astype(np.float32)
    return (envelope * carrier).astype(np.complex64)


def _fm(rng: np.random.Generator) -> np.ndarray:
    """FM: phase is the integral of the message signal.
    Frequency deviation kf drawn from [0.1, 0.3] of normalised Fs.
    """
    t = np.arange(WINDOW, dtype=np.float32)
    kf = rng.uniform(0.1, 0.3)
    fm = rng.uniform(0.01, 0.05)
    phi_msg = rng.uniform(0, 2 * np.pi)
    msg = np.cos(2 * np.pi * fm * t + phi_msg).astype(np.float32)

    # Instantaneous phase = 2π·fc·t + 2π·kf·cumsum(msg)/Fs
    inst_phase = 2 * np.pi * FC * t + 2 * np.pi * kf * np.cumsum(msg)
    phi_c = rng.uniform(0, 2 * np.pi)
    return np.exp(1j * (inst_phase + phi_c)).astype(np.complex64)


def _bpsk(rng: np.random.Generator) -> np.ndarray:
    """BPSK: each symbol ∈ {-1, +1}, pulse-shaped with raised cosine,
    then upconverted to fc.  Oversampling factor sps drawn from {4, 8}.
    """
    sps = int(rng.choice([4, 8]))
    n_sym = WINDOW // sps + 2           # a few extra to trim cleanly
    bits = rng.choice([-1.0, 1.0], size=n_sym).astype(np.float32)

    # Simple rectangular pulse (upsample)
    baseband = np.repeat(bits, sps)[:WINDOW].astype(np.float32)

    t = np.arange(WINDOW, dtype=np.float32)
    phi_c = rng.uniform(0, 2 * np.pi)
    carrier = np.exp(1j * (2 * np.pi * FC * t + phi_c)).astype(np.complex64)
    return (baseband * carrier).astype(np.complex64)


def _qpsk(rng: np.random.Generator) -> np.ndarray:
    """QPSK: symbols from {±1 ±j}/√2, rectangular pulse, upconverted.
    Oversampling factor sps drawn from {4, 8}.
    """
    sps = int(rng.choice([4, 8]))
    n_sym = WINDOW // sps + 2
    # Gray-coded QPSK constellation
    constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2)
    symbols = rng.choice(constellation, size=n_sym)

    baseband = np.repeat(symbols, sps)[:WINDOW].astype(np.complex64)

    t = np.arange(WINDOW, dtype=np.float32)
    phi_c = rng.uniform(0, 2 * np.pi)
    carrier = np.exp(1j * (2 * np.pi * FC * t + phi_c)).astype(np.complex64)
    return (baseband * carrier).astype(np.complex64)


GENERATORS = {
    "AM-DSB": _am_dsb,
    "FM":     _fm,
    "BPSK":   _bpsk,
    "QPSK":   _qpsk,
}


# ---------------------------------------------------------------------------
# AWGN at exact SNR
# ---------------------------------------------------------------------------

def _add_awgn(iq: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = float(np.mean(np.abs(iq) ** 2))
    if signal_power < 1e-12:
        return iq
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    std = np.sqrt(noise_power / 2.0)
    noise = (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))).astype(np.complex64)
    return (iq + noise * np.float32(std)).astype(np.complex64)


# ---------------------------------------------------------------------------
# Normalise to unit power
# ---------------------------------------------------------------------------

def _normalise(iq: np.ndarray) -> np.ndarray:
    power = float(np.mean(np.abs(iq) ** 2))
    if power < 1e-12:
        return iq
    return (iq / np.sqrt(power)).astype(np.complex64)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate(
    n_per_class: int,
    snr_values: list[int],
    seed: int = 0,
) -> dict:
    """Return a RadioML-format dict ready to pickle.

    Returns
    -------
    dict[(modulation, snr)] → ndarray shape (n_per_class, 2, WINDOW)
    """
    rng = np.random.default_rng(seed)
    dataset = {}
    total = len(GENERATORS) * len(snr_values) * n_per_class

    logger.info(
        "Generating %d frames: %d modulations × %d SNR levels × %d examples",
        total, len(GENERATORS), len(snr_values), n_per_class,
    )

    for mod, gen_fn in GENERATORS.items():
        for snr in snr_values:
            frames = []
            for _ in range(n_per_class):
                iq = gen_fn(rng)
                iq = _normalise(iq)
                iq = _add_awgn(iq, snr, rng)
                # Store as (2, WINDOW) float32: row 0 = I, row 1 = Q
                frame = np.stack([iq.real, iq.imag], axis=0).astype(np.float32)
                frames.append(frame)
            dataset[(mod, snr)] = np.stack(frames)   # (N, 2, WINDOW)
            logger.info("  %-8s  SNR %+3d dB  → %d frames", mod, snr, n_per_class)

    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Generate synthetic IQ dataset (RadioML-compatible pickle)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--out", default="data/synthetic_rml.pkl",
                   help="Output pickle path (default: data/synthetic_rml.pkl)")
    p.add_argument("--n-per-class", type=int, default=1000,
                   help="Examples per (modulation, SNR) pair (default: 1000)")
    p.add_argument("--snr-min", type=int, default=-20)
    p.add_argument("--snr-max", type=int, default=18)
    p.add_argument("--snr-step", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    snr_values = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    dataset = generate(args.n_per_class, snr_values, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pickle.dumps(dataset))

    n_total = sum(v.shape[0] for v in dataset.values())
    logger.info("Saved %d total frames to %s", n_total, out_path)
    logger.info(
        "File size: %.1f MB", out_path.stat().st_size / 1e6
    )
    logger.info(
        "Use with: python scripts/run_experiment.py --data %s", out_path
    )


if __name__ == "__main__":
    main()
