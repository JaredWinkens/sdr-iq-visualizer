#!/usr/bin/env python3
"""Before/after robustness experiment runner.

Trains two models on the same RadioML dataset and compares them:
  1. Baseline — no augmentation
  2. Augmented — RF impairment pipeline applied during training

Results are printed as a side-by-side table and saved to a JSON file
so they can be referenced in the written report.

Usage
-----
# Full experiment (20 epochs each, all SNRs)
python scripts/run_experiment.py --data /path/to/RML2016.10a_dict.pkl

# Quick smoke-test run (2 epochs, narrow SNR slice)
python scripts/run_experiment.py --data /path/to/RML2016.10a_dict.pkl \
    --epochs 2 --snr-min -10 --snr-max 10 --out-dir results/smoke

# Evaluate existing checkpoints without retraining
python scripts/run_experiment.py --data /path/to/RML2016.10a_dict.pkl \
    --baseline-model models/baseline_cnn.pt \
    --augmented-model models/augmented_cnn.pt \
    --eval-only
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.processing.augmentation import AugmentationPipeline
from app.processing.ml_classifier import (
    RadioMLDataset,
    BASELINE_LABELS,
    evaluate,
    load_model,
    train,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# SNR buckets that match the proposal's evaluation protocol
SNR_BUCKETS = [
    ("< 0 dB",   lambda s: s < 0),
    ("0–5 dB",   lambda s: 0 <= s < 5),
    ("5–10 dB",  lambda s: 5 <= s < 10),
    ("≥ 10 dB",  lambda s: s >= 10),
]


def _load_dataset(pkl_path, snr_min, snr_max, transform=None):
    ds = RadioMLDataset(pkl_path, snr_range=(snr_min, snr_max), transform=transform)
    # Attach SNR array for per-bucket evaluation
    import pickle
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    snrs = []
    for (mod, snr), data in raw.items():
        if mod not in BASELINE_LABELS:
            continue
        if not (snr_min <= snr <= snr_max):
            continue
        snrs.extend([snr] * len(data))
    ds.snrs = np.array(snrs, dtype=int)
    return ds


def _evaluate_with_snr_buckets(model, dataset):
    """Run evaluate() then re-bucket using the proposal's bucket labels."""
    results = evaluate(model, dataset)

    if hasattr(dataset, "snrs"):
        import torch
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                all_preds.append(model(x).argmax(dim=1).numpy())
                all_labels.append(y.numpy())
        preds  = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        snrs   = dataset.snrs

        from app.processing.ml_classifier import _confusion_matrix, _macro_f1, NUM_CLASSES
        bucket_f1 = {}
        for name, cond in SNR_BUCKETS:
            mask = np.array([cond(s) for s in snrs])
            if mask.sum() == 0:
                continue
            cm = _confusion_matrix(labels[mask], preds[mask], NUM_CLASSES)
            bucket_f1[name] = round(_macro_f1(cm), 4)
        results["snr_buckets"] = bucket_f1

    return results


def _print_comparison(baseline, augmented):
    print("\n" + "=" * 62)
    print("  ROBUSTNESS EXPERIMENT — BEFORE vs AFTER")
    print("=" * 62)
    print(f"  {'Metric':<28} {'Baseline':>12} {'Augmented':>12}")
    print("-" * 62)
    print(f"  {'Macro F1':<28} {baseline['macro_f1']:>12.4f} {augmented['macro_f1']:>12.4f}")
    print(f"  {'Balanced Accuracy':<28} {baseline['balanced_acc']:>12.4f} {augmented['balanced_acc']:>12.4f}")
    print()
    print("  Per-class recall:")
    for lbl in BASELINE_LABELS:
        b = baseline["per_class_recall"].get(lbl, 0)
        a = augmented["per_class_recall"].get(lbl, 0)
        delta = a - b
        arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else " ")
        print(f"    {lbl:<26} {b:>12.4f} {a:>12.4f}  {arrow}{abs(delta):.4f}")
    print()
    if "snr_buckets" in baseline and "snr_buckets" in augmented:
        print("  Macro F1 by SNR bucket:")
        all_buckets = sorted(
            set(baseline["snr_buckets"]) | set(augmented["snr_buckets"])
        )
        for bucket in all_buckets:
            b = baseline["snr_buckets"].get(bucket, float("nan"))
            a = augmented["snr_buckets"].get(bucket, float("nan"))
            delta = a - b if not (np.isnan(a) or np.isnan(b)) else float("nan")
            arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else " ")
            print(f"    {bucket:<26} {b:>12.4f} {a:>12.4f}  {arrow}{abs(delta):.4f}")
    print("=" * 62)


def main():
    p = argparse.ArgumentParser(
        description="Before/after robustness experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data", required=True, help="RadioML 2016.10a pickle path")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--snr-min", type=int, default=-20)
    p.add_argument("--snr-max", type=int, default=18)
    p.add_argument("--out-dir", default="results", help="Directory for checkpoints and results JSON")
    p.add_argument("--baseline-model", help="Skip baseline training; load from this path")
    p.add_argument("--augmented-model", help="Skip augmented training; load from this path")
    p.add_argument("--eval-only", action="store_true", help="Requires both --baseline-model and --augmented-model")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_ckpt  = args.baseline_model  or str(out_dir / "baseline_cnn.pt")
    augmented_ckpt = args.augmented_model or str(out_dir / "augmented_cnn.pt")

    # ── Baseline ──────────────────────────────────────────────────────────────
    if not args.eval_only:
        logger.info("Loading dataset (no augmentation) …")
        baseline_ds = _load_dataset(args.data, args.snr_min, args.snr_max, transform=None)

        logger.info("Training BASELINE model …")
        train(
            baseline_ds,
            out_path=baseline_ckpt,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )

    logger.info("Evaluating baseline …")
    baseline_model, _ = load_model(baseline_ckpt)
    eval_ds = _load_dataset(args.data, args.snr_min, args.snr_max, transform=None)
    baseline_results = _evaluate_with_snr_buckets(baseline_model, eval_ds)

    # ── Augmented ─────────────────────────────────────────────────────────────
    if not args.eval_only:
        pipeline = AugmentationPipeline(
            snr_db_range=(-5, 15),
            freq_offset_range=(-0.05, 0.05),
            phase_shift_range=(-np.pi, np.pi),
            amplitude_scale_range=(0.8, 1.2),
            dc_offset_std=0.01,
            p_apply=0.9,
            seed=args.seed,
        )
        logger.info("Augmentation config: %s", pipeline.describe())

        logger.info("Loading dataset (with augmentation) …")
        augmented_ds = _load_dataset(args.data, args.snr_min, args.snr_max, transform=pipeline)

        logger.info("Training AUGMENTED model …")
        train(
            augmented_ds,
            out_path=augmented_ckpt,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )

    logger.info("Evaluating augmented model …")
    augmented_model, _ = load_model(augmented_ckpt)
    augmented_results = _evaluate_with_snr_buckets(augmented_model, eval_ds)

    # ── Report ────────────────────────────────────────────────────────────────
    _print_comparison(baseline_results, augmented_results)

    # Save JSON for the report
    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "epochs": args.epochs,
            "snr_range": [args.snr_min, args.snr_max],
            "seed": args.seed,
            "labels": BASELINE_LABELS,
        },
        "baseline":  {k: v.tolist() if hasattr(v, "tolist") else v
                      for k, v in baseline_results.items()},
        "augmented": {k: v.tolist() if hasattr(v, "tolist") else v
                      for k, v in augmented_results.items()},
    }
    result_path = out_dir / "experiment_results.json"
    result_path.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", result_path)


if __name__ == "__main__":
    main()
