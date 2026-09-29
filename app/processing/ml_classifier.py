"""Baseline ML modulation classifier for the SDR IQ platform.

Architecture: 1D CNN over interleaved IQ samples (standard AMC approach).

Supports two data sources:
  1. RadioML 2016.10a/b — pickle format from DeepSig
     https://www.deepsig.ai/datasets
  2. SigMF recordings with 'core:label' annotations

Label set (baseline): AM-DSB, FM, BPSK, QPSK
  - Expand only after baseline evaluation is complete (proposal §Phase 1)

Typical usage
-------------
# Train from RadioML pickle
python -m app.processing.ml_classifier \
    --data path/to/RML2016.10a_dict.pkl \
    --source radioml \
    --epochs 20 \
    --out models/baseline_cnn.pt

# Evaluate a saved model
python -m app.processing.ml_classifier \
    --model models/baseline_cnn.pt \
    --data path/to/RML2016.10a_dict.pkl \
    --source radioml \
    --eval-only

# Predict on a SigMF file (inference)
python -m app.processing.ml_classifier \
    --model models/baseline_cnn.pt \
    --sigmf path/to/recording \
    --predict
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label registry — single source of truth
# ---------------------------------------------------------------------------

BASELINE_LABELS = ["AM-DSB", "FM", "BPSK", "QPSK"]
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(BASELINE_LABELS)}
IDX_TO_LABEL = {i: lbl for lbl, i in LABEL_TO_IDX.items()}
NUM_CLASSES = len(BASELINE_LABELS)

# IQ window fed to the network (number of complex samples → 2× floats)
WINDOW_SIZE = 128  # 128 complex samples → 256-element input vector


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class RadioMLDataset(Dataset):
    """Loads the RadioML 2016.10a/b pickle.

    The pickle is a dict keyed by (modulation_str, snr_int) → ndarray
    of shape (N, 2, 128) where axis-1 is [I, Q].

    Only labels present in ``label_filter`` are kept.  If ``snr_range`` is
    given as (lo, hi) only entries with lo <= snr <= hi are kept.
    """

    def __init__(
        self,
        pkl_path: str,
        label_filter: list[str] = None,
        snr_range: tuple[int, int] = None,
        transform=None,
    ):
        if label_filter is None:
            label_filter = BASELINE_LABELS

        with open(pkl_path, "rb") as f:
            raw = pickle.load(f, encoding="latin1")

        samples, labels = [], []
        for (mod, snr), data in raw.items():
            if mod not in label_filter:
                continue
            if snr_range is not None and not (snr_range[0] <= snr <= snr_range[1]):
                continue
            for frame in data:  # frame shape: (2, 128)
                samples.append(frame.astype(np.float32))
                labels.append(LABEL_TO_IDX[mod])

        self.samples = np.stack(samples)   # (N, 2, 128)
        self.labels = np.array(labels, dtype=np.int64)
        self.transform = transform
        logger.info(
            "RadioMLDataset: %d samples | labels: %s",
            len(self.labels),
            {IDX_TO_LABEL[i]: int((self.labels == i).sum()) for i in range(NUM_CLASSES)},
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.samples[idx]
        if self.transform is not None:
            x = self.transform(x)
        return torch.from_numpy(x), self.labels[idx]


class SigMFDataset(Dataset):
    """Dataset built from SigMF recordings that carry 'core:label' annotations.

    Each annotation's sample_start/sample_count window is sliced into
    non-overlapping WINDOW_SIZE chunks and labelled with its annotation label.
    Only annotations whose label appears in ``label_filter`` are used.
    """

    def __init__(self, paths: list[str], label_filter: list[str] = None, transform=None):
        if label_filter is None:
            label_filter = BASELINE_LABELS

        try:
            from sigmf.sigmffile import fromfile
        except ImportError as exc:
            raise ImportError(
                "sigmf library is required for SigMF datasets. "
                "Install with: pip install sigmf"
            ) from exc

        samples, labels = [], []
        for path in paths:
            base = str(path).replace(".sigmf-data", "").replace(".sigmf-meta", "")
            sf = fromfile(base)
            iq = sf.read_samples()  # complex64 array

            for ann in sf.get_annotations():
                lbl = ann.get("core:label", "")
                if lbl not in label_filter:
                    continue
                start = ann.get("core:sample_start", 0)
                count = ann.get("core:sample_count", len(iq) - start)
                chunk = iq[start: start + count]

                # Slice into WINDOW_SIZE windows
                n_windows = len(chunk) // WINDOW_SIZE
                for w in range(n_windows):
                    seg = chunk[w * WINDOW_SIZE: (w + 1) * WINDOW_SIZE]
                    iq_2d = np.stack(
                        [seg.real.astype(np.float32), seg.imag.astype(np.float32)],
                        axis=0,
                    )  # shape (2, WINDOW_SIZE)
                    samples.append(iq_2d)
                    labels.append(LABEL_TO_IDX[lbl])

        if not samples:
            raise ValueError(
                "No usable samples found. Check annotation labels and label_filter."
            )

        self.samples = np.stack(samples)
        self.labels = np.array(labels, dtype=np.int64)
        self.transform = transform
        logger.info(
            "SigMFDataset: %d samples from %d file(s)", len(self.labels), len(paths)
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.samples[idx]
        if self.transform is not None:
            x = self.transform(x)
        return torch.from_numpy(x), self.labels[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BaselineCNN(nn.Module):
    """Lightweight 1D CNN for modulation classification.

    Input:  (batch, 2, WINDOW_SIZE)  — 2 channels: I and Q
    Output: (batch, NUM_CLASSES)     — raw logits

    Design notes
    ------------
    - Three conv blocks with increasing filter counts capture local IQ patterns
      at multiple scales (fine timing → coarser spectral shape).
    - Global average pooling keeps the parameter count low and gives the
      network translation invariance over the window.
    - Dropout(0.5) on the FC layer is the primary regulariser for Phase 1;
      weight decay is applied by the optimiser.
    - Architecture is intentionally compact so training works on CPU for
      quick iteration without a GPU.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, window: int = WINDOW_SIZE):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — fine-grained IQ structure
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),   # → length window/2

            # Block 2 — mid-range patterns
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),   # → length window/4

            # Block 3 — coarse envelope / spectral shape
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),       # → (batch, 256, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """Return a sampler that up-samples minority classes each epoch.

    Works with both plain datasets (expose ``.labels``) and
    ``torch.utils.data.Subset`` wrappers produced by ``random_split``.
    """
    from torch.utils.data import Subset

    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = dataset.indices
        if hasattr(base, "labels"):
            labels = base.labels[np.array(indices)]
        else:
            labels = np.array([base[i][1] for i in indices])
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    class_counts = np.where(class_counts == 0, 1.0, class_counts)
    weights = 1.0 / class_counts
    sample_weights = torch.tensor(weights[labels], dtype=torch.float)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def train(
    dataset: Dataset,
    out_path: str,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.15,
    seed: int = 42,
    use_weighted_sampler: bool = True,
):
    """Train BaselineCNN and save the checkpoint.

    Returns
    -------
    dict with keys 'train_loss', 'val_loss', 'val_acc' — lists over epochs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # Train / val split
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    sampler = _make_weighted_sampler(train_ds) if use_weighted_sampler else None
    # WeightedRandomSampler replaces shuffle; use shuffle=True only when no sampler
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = BaselineCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)
        train_loss = running_loss / n_train
        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * len(y)
                correct += (logits.argmax(dim=1) == y).sum().item()
        val_loss /= n_val
        val_acc = correct / n_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            "Epoch %02d/%02d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
            epoch, epochs, train_loss, val_loss, val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save_checkpoint(model, out_path, epoch, val_acc)
            logger.info("  ↑ saved best checkpoint (val_acc=%.4f)", val_acc)

    logger.info("Training complete. Best val_acc=%.4f", best_val_acc)
    return history


def _save_checkpoint(model: nn.Module, path: str, epoch: int, val_acc: float):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "window_size": WINDOW_SIZE,
            "labels": BASELINE_LABELS,
            "epoch": epoch,
            "val_acc": val_acc,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, dataset: Dataset, batch_size: int = 256) -> dict:
    """Return per-class metrics, confusion matrix, and SNR-bucket breakdown.

    ``dataset`` must expose a ``.labels`` attribute (numpy int64 array).
    For SNR-bucket breakdown, dataset must also expose ``.snrs`` (int array
    parallel to ``.labels``); if absent, SNR breakdown is skipped.

    Returns
    -------
    dict with:
      - macro_f1       : float
      - balanced_acc   : float
      - per_class_recall: dict  label → recall
      - confusion_matrix: np.ndarray  (NUM_CLASSES × NUM_CLASSES)
      - snr_buckets    : dict  bucket_str → macro_f1  (if .snrs available)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    cm = _confusion_matrix(labels, preds, NUM_CLASSES)
    per_class_recall = {}
    recalls = []
    for i in range(NUM_CLASSES):
        total = cm[i].sum()
        recall = float(cm[i, i] / total) if total > 0 else 0.0
        per_class_recall[IDX_TO_LABEL[i]] = round(recall, 4)
        recalls.append(recall)

    balanced_acc = float(np.mean(recalls))
    macro_f1 = _macro_f1(cm)

    results = {
        "macro_f1": round(macro_f1, 4),
        "balanced_acc": round(balanced_acc, 4),
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm,
    }

    # SNR-bucket breakdown (requires dataset.snrs)
    if hasattr(dataset, "snrs"):
        snrs = dataset.snrs
        buckets = [
            ("snr_lt0", lambda s: s < 0),
            ("snr_0_5", lambda s: 0 <= s < 5),
            ("snr_5_10", lambda s: 5 <= s < 10),
            ("snr_gte10", lambda s: s >= 10),
        ]
        snr_results = {}
        for name, cond in buckets:
            mask = np.array([cond(s) for s in snrs])
            if mask.sum() == 0:
                continue
            b_cm = _confusion_matrix(labels[mask], preds[mask], NUM_CLASSES)
            snr_results[name] = round(_macro_f1(b_cm), 4)
        results["snr_buckets"] = snr_results

    return results


def _confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _macro_f1(cm: np.ndarray) -> float:
    f1s = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


# ---------------------------------------------------------------------------
# Inference helpers (used by the dashboard)
# ---------------------------------------------------------------------------

def load_model(model_path: str) -> tuple[nn.Module, list[str]]:
    """Load a saved checkpoint.  Returns (model, label_list)."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    model = BaselineCNN(num_classes=ckpt["num_classes"], window=ckpt["window_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["labels"]


def predict_iq(model: nn.Module, iq: np.ndarray, labels: list[str]) -> dict:
    """Classify a raw complex IQ array.

    Takes the first ``WINDOW_SIZE`` samples.  Returns
    ``{"label": str, "confidence": float, "probabilities": {label: prob}}``.

    Parameters
    ----------
    iq:
        Complex numpy array of length >= WINDOW_SIZE.
    """
    if len(iq) < WINDOW_SIZE:
        return {"label": "Insufficient data", "confidence": 0.0, "probabilities": {}}

    window = iq[:WINDOW_SIZE]
    x = np.stack([window.real, window.imag], axis=0).astype(np.float32)  # (2, W)
    x = torch.from_numpy(x).unsqueeze(0)  # (1, 2, W)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()

    best = int(np.argmax(probs))
    return {
        "label": labels[best],
        "confidence": round(float(probs[best]), 4),
        "probabilities": {lbl: round(float(p), 4) for lbl, p in zip(labels, probs)},
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description="Baseline CNN modulation classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data", help="Path to RadioML pickle or SigMF file list (space-separated)")
    p.add_argument("--source", choices=["radioml", "sigmf"], default="radioml")
    p.add_argument("--model", help="Path to saved .pt checkpoint")
    p.add_argument("--out", default="models/baseline_cnn.pt", help="Output checkpoint path")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--snr-min", type=int, default=-20, help="Min SNR to include (RadioML)")
    p.add_argument("--snr-max", type=int, default=18, help="Max SNR to include (RadioML)")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--sigmf", help="SigMF recording path for inference (--predict)")
    p.add_argument("--predict", action="store_true")
    return p


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args()

    # --- Inference mode ---
    if args.predict:
        if not args.model or not args.sigmf:
            raise SystemExit("--predict requires --model and --sigmf")
        try:
            from sigmf.sigmffile import fromfile
        except ImportError:
            raise SystemExit("pip install sigmf  # required for --predict")
        model, labels = load_model(args.model)
        base = args.sigmf.replace(".sigmf-data", "").replace(".sigmf-meta", "")
        iq = fromfile(base).read_samples()
        result = predict_iq(model, iq, labels)
        print(f"Prediction: {result['label']}  (confidence {result['confidence']:.2%})")
        print("Per-class probabilities:")
        for lbl, prob in result["probabilities"].items():
            print(f"  {lbl:10s} {prob:.4f}")
        return

    # --- Load dataset ---
    if not args.data:
        raise SystemExit("--data is required unless using --predict")

    if args.source == "radioml":
        dataset = RadioMLDataset(
            args.data,
            snr_range=(args.snr_min, args.snr_max),
        )
        # Attach SNR array for eval bucketing
        with open(args.data, "rb") as f:
            raw = pickle.load(f, encoding="latin1")
        snr_list = []
        for (mod, snr), data in raw.items():
            if mod not in BASELINE_LABELS:
                continue
            if not (args.snr_min <= snr <= args.snr_max):
                continue
            snr_list.extend([snr] * len(data))
        dataset.snrs = np.array(snr_list, dtype=int)
    else:
        paths = args.data.split()
        dataset = SigMFDataset(paths)

    # --- Train or eval ---
    if args.eval_only:
        if not args.model:
            raise SystemExit("--eval-only requires --model")
        model, labels = load_model(args.model)
        results = evaluate(model, dataset)
    else:
        history = train(dataset, out_path=args.out, epochs=args.epochs,
                        batch_size=args.batch_size, lr=args.lr)
        model, labels = load_model(args.out)
        results = evaluate(model, dataset)

    print("\n=== Evaluation Results ===")
    print(f"Macro F1:       {results['macro_f1']:.4f}")
    print(f"Balanced Acc:   {results['balanced_acc']:.4f}")
    print("\nPer-class recall:")
    for lbl, rec in results["per_class_recall"].items():
        print(f"  {lbl:10s} {rec:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("            " + "  ".join(f"{l:10s}" for l in BASELINE_LABELS))
    for i, row in enumerate(results["confusion_matrix"]):
        print(f"  {BASELINE_LABELS[i]:10s}" + "  ".join(f"{v:10d}" for v in row))
    if "snr_buckets" in results:
        print("\nMacro F1 by SNR bucket:")
        for bucket, f1 in results["snr_buckets"].items():
            print(f"  {bucket:12s} {f1:.4f}")


if __name__ == "__main__":
    main()
