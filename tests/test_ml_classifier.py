"""Unit tests for the baseline CNN modulation classifier.

These tests are designed to run without a GPU and without the RadioML
dataset — they use synthetic IQ data to exercise all code paths.
"""

import unittest
import numpy as np
import torch

from app.processing.ml_classifier import (
    BaselineCNN,
    RadioMLDataset,
    SigMFDataset,
    evaluate,
    load_model,
    predict_iq,
    train,
    _confusion_matrix,
    _macro_f1,
    BASELINE_LABELS,
    IDX_TO_LABEL,
    NUM_CLASSES,
    WINDOW_SIZE,
)
from torch.utils.data import TensorDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(n_per_class: int = 64, seed: int = 0):
    """Return a TensorDataset with random IQ frames and balanced labels."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((NUM_CLASSES * n_per_class, 2, WINDOW_SIZE)).astype(np.float32)
    y = np.repeat(np.arange(NUM_CLASSES), n_per_class).astype(np.int64)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    ds.labels = y          # expose .labels so evaluate() can use it
    return ds


def _make_synthetic_radioml_pkl(path, n_per_class: int = 32, seed: int = 1):
    """Write a minimal RadioML-format pickle to ``path``."""
    import pickle
    rng = np.random.default_rng(seed)
    data = {}
    for lbl in BASELINE_LABELS:
        for snr in [-10, 0, 10]:
            data[(lbl, snr)] = rng.standard_normal(
                (n_per_class, 2, WINDOW_SIZE)
            ).astype(np.float32)
    path.write_bytes(pickle.dumps(data))
    return data


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------

class TestBaselineCNN(unittest.TestCase):

    def test_output_shape(self):
        """Forward pass produces (batch, NUM_CLASSES) logits."""
        model = BaselineCNN()
        x = torch.randn(8, 2, WINDOW_SIZE)
        out = model(x)
        self.assertEqual(out.shape, (8, NUM_CLASSES))

    def test_single_sample(self):
        model = BaselineCNN()
        x = torch.randn(1, 2, WINDOW_SIZE)
        out = model(x)
        self.assertEqual(out.shape, (1, NUM_CLASSES))

    def test_no_nan_in_output(self):
        model = BaselineCNN()
        x = torch.randn(16, 2, WINDOW_SIZE)
        out = model(x)
        self.assertFalse(torch.isnan(out).any(), "NaN in model output")

    def test_parameter_count_reasonable(self):
        """Model should be lightweight — under 500k parameters."""
        model = BaselineCNN()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertLess(n_params, 500_000, f"Model has {n_params} params — too large for baseline")


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestRadioMLDataset(unittest.TestCase):

    def setUp(self):
        import tempfile, pathlib
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.pkl_path = self.tmp / "rml.pkl"
        _make_synthetic_radioml_pkl(self.pkl_path)

    def test_loads_correct_classes(self):
        ds = RadioMLDataset(str(self.pkl_path))
        unique_labels = set(ds.labels.tolist())
        self.assertEqual(unique_labels, set(range(NUM_CLASSES)))

    def test_sample_shape(self):
        ds = RadioMLDataset(str(self.pkl_path))
        x, y = ds[0]
        self.assertEqual(x.shape, (2, WINDOW_SIZE))
        self.assertIsInstance(y, (int, np.integer))

    def test_snr_filter(self):
        """Only samples with SNR in [0, 10] should be loaded."""
        ds_all = RadioMLDataset(str(self.pkl_path))
        ds_filtered = RadioMLDataset(str(self.pkl_path), snr_range=(0, 10))
        # Filtered dataset should have exactly 2/3 of samples (SNRs 0 and 10 only)
        self.assertEqual(len(ds_filtered) * 3, len(ds_all) * 2)

    def test_label_filter(self):
        ds = RadioMLDataset(str(self.pkl_path), label_filter=["AM-DSB", "FM"])
        unique_labels = set(ds.labels.tolist())
        self.assertEqual(len(unique_labels), 2)

    def test_len_positive(self):
        ds = RadioMLDataset(str(self.pkl_path))
        self.assertGreater(len(ds), 0)


# ---------------------------------------------------------------------------
# Metric helper tests
# ---------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):

    def test_confusion_matrix_perfect(self):
        y = np.array([0, 1, 2, 3])
        cm = _confusion_matrix(y, y, NUM_CLASSES)
        np.testing.assert_array_equal(cm, np.eye(NUM_CLASSES, dtype=int))

    def test_confusion_matrix_all_wrong(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([1, 2, 3, 0])
        cm = _confusion_matrix(y_true, y_pred, NUM_CLASSES)
        # Diagonal should be all zeros
        self.assertEqual(np.trace(cm), 0)

    def test_macro_f1_perfect(self):
        cm = np.eye(NUM_CLASSES, dtype=int) * 100
        self.assertAlmostEqual(_macro_f1(cm), 1.0)

    def test_macro_f1_all_wrong(self):
        # All predictions on class 0 → other classes have 0 recall/precision
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        cm[:, 0] = 10  # everything predicted as class 0
        self.assertLess(_macro_f1(cm), 0.5)

    def test_macro_f1_range(self):
        rng = np.random.default_rng(7)
        cm = rng.integers(0, 100, size=(NUM_CLASSES, NUM_CLASSES))
        f1 = _macro_f1(cm)
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)


# ---------------------------------------------------------------------------
# Training smoke test (tiny dataset, 1 epoch)
# ---------------------------------------------------------------------------

class TestTraining(unittest.TestCase):

    def test_train_runs_and_saves(self):
        """Training 1 epoch on synthetic data should produce a valid checkpoint."""
        import tempfile, pathlib
        tmp = pathlib.Path(tempfile.mkdtemp())
        out = str(tmp / "test_model.pt")

        ds = _make_synthetic_dataset(n_per_class=16)
        history = train(ds, out_path=out, epochs=1, batch_size=32, val_split=0.2)

        self.assertTrue(pathlib.Path(out).exists(), "Checkpoint file not created")
        self.assertEqual(len(history["train_loss"]), 1)
        self.assertFalse(np.isnan(history["train_loss"][0]), "NaN training loss")

    def test_load_and_predict(self):
        """A trained checkpoint should produce valid predictions."""
        import tempfile, pathlib
        tmp = pathlib.Path(tempfile.mkdtemp())
        out = str(tmp / "test_model.pt")

        ds = _make_synthetic_dataset(n_per_class=16)
        train(ds, out_path=out, epochs=1, batch_size=32, val_split=0.2)

        model, labels = load_model(out)
        self.assertEqual(labels, BASELINE_LABELS)

        rng = np.random.default_rng(42)
        iq = (rng.standard_normal(WINDOW_SIZE) + 1j * rng.standard_normal(WINDOW_SIZE)).astype(np.complex64)
        result = predict_iq(model, iq, labels)

        self.assertIn("label", result)
        self.assertIn(result["label"], BASELINE_LABELS)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        prob_sum = sum(result["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)


# ---------------------------------------------------------------------------
# Inference edge-case tests
# ---------------------------------------------------------------------------

class TestInference(unittest.TestCase):

    def _get_model(self):
        """Quick untrained model for inference path tests."""
        model = BaselineCNN()
        model.eval()
        return model

    def test_predict_iq_too_short(self):
        model = self._get_model()
        iq = np.zeros(WINDOW_SIZE - 1, dtype=np.complex64)
        result = predict_iq(model, iq, BASELINE_LABELS)
        self.assertEqual(result["label"], "Insufficient data")
        self.assertEqual(result["confidence"], 0.0)

    def test_predict_iq_exact_window(self):
        model = self._get_model()
        iq = np.random.randn(WINDOW_SIZE).astype(np.float32) + 1j * np.random.randn(WINDOW_SIZE).astype(np.float32)
        result = predict_iq(model, iq, BASELINE_LABELS)
        self.assertIn(result["label"], BASELINE_LABELS)

    def test_predict_iq_longer_than_window(self):
        """Only first WINDOW_SIZE samples should be used — no error."""
        model = self._get_model()
        iq = np.random.randn(WINDOW_SIZE * 4).astype(np.float32) + 1j * np.random.randn(WINDOW_SIZE * 4).astype(np.float32)
        result = predict_iq(model, iq, BASELINE_LABELS)
        self.assertIn(result["label"], BASELINE_LABELS)

    def test_evaluate_returns_required_keys(self):
        model = BaselineCNN()
        ds = _make_synthetic_dataset(n_per_class=8)
        results = evaluate(model, ds)
        for key in ("macro_f1", "balanced_acc", "per_class_recall", "confusion_matrix"):
            self.assertIn(key, results)
        self.assertEqual(results["confusion_matrix"].shape, (NUM_CLASSES, NUM_CLASSES))

    def test_evaluate_confusion_matrix_sum(self):
        """Confusion matrix row-sums should equal the true label counts."""
        model = BaselineCNN()
        ds = _make_synthetic_dataset(n_per_class=8)
        results = evaluate(model, ds)
        cm = results["confusion_matrix"]
        for i in range(NUM_CLASSES):
            expected = int((ds.labels == i).sum())
            self.assertEqual(int(cm[i].sum()), expected)


if __name__ == "__main__":
    unittest.main()
