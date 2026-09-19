#!/usr/bin/env python3
"""Convenience wrapper for training the baseline CNN classifier.

Usage examples
--------------
# From project root — RadioML 2016.10a, all SNRs
python scripts/train_classifier.py \
    --data /path/to/RML2016.10a_dict.pkl \
    --epochs 20

# Low-SNR slice only (-10 to 0 dB) — useful for auditing baseline weakness
python scripts/train_classifier.py \
    --data /path/to/RML2016.10a_dict.pkl \
    --snr-min -10 --snr-max 0 \
    --epochs 20 \
    --out models/baseline_cnn_lowsnr.pt

# SigMF recordings with 'core:label' annotations
python scripts/train_classifier.py \
    --source sigmf \
    --data "recordings/am.sigmf recordings/fm.sigmf" \
    --epochs 30

Downloading RadioML 2016.10a
-----------------------------
The dataset is freely available at https://www.deepsig.ai/datasets
File: RML2016.10a.tar.bz2 (~55 MB unpacked)
Expected labels in baseline filter: AM-DSB, FM, BPSK, QPSK
"""

import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.processing.ml_classifier import main

if __name__ == "__main__":
    main()
