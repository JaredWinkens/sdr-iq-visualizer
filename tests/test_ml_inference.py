"""Unit tests for the dashboard ML inference wrapper."""

import sys
import types
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_fake_model():
    """Return a mock that behaves like a loaded BaselineCNN."""
    import torch
    model = MagicMock()
    model.eval.return_value = model
    # Simulate logits: QPSK (index 3) wins with high confidence
    logits = torch.tensor([[0.1, 0.1, 0.1, 5.0]])
    model.return_value = logits
    return model


def _synthetic_iq(n=256, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


class TestClassifyIQNoModel(unittest.TestCase):
    """Behaviour when no model file exists on disk."""

    def setUp(self):
        # Reset module state before each test
        import app.processing.ml_inference as m
        m._model = None
        m._labels = None
        m._load_attempted = False

    def test_returns_dict_when_no_model(self):
        with patch("pathlib.Path.exists", return_value=False):
            from app.processing.ml_inference import classify_iq
            result = classify_iq(_synthetic_iq())
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertIn("source", result)

    def test_source_is_none_when_no_model(self):
        with patch("pathlib.Path.exists", return_value=False):
            from app.processing.ml_inference import classify_iq
            result = classify_iq(_synthetic_iq())
        self.assertEqual(result["source"], "none")

    def test_short_signal_returns_safely(self):
        with patch("pathlib.Path.exists", return_value=False):
            from app.processing.ml_inference import classify_iq
            result = classify_iq(_synthetic_iq(n=10))
        self.assertIsInstance(result, dict)

    def test_model_loaded_false_when_no_file(self):
        with patch("pathlib.Path.exists", return_value=False):
            import app.processing.ml_inference as m
            m._try_load_model()
            from app.processing.ml_inference import model_loaded
            self.assertFalse(model_loaded())


class TestClassifyIQWithModel(unittest.TestCase):
    """Behaviour when a model is injected directly into module state."""

    def setUp(self):
        import app.processing.ml_inference as m
        m._model = None
        m._labels = None
        m._load_attempted = False

    def _inject_model(self):
        import app.processing.ml_inference as m
        from app.processing.ml_classifier import BASELINE_LABELS
        m._model = _make_fake_model()
        m._labels = BASELINE_LABELS
        m._load_attempted = True

    def test_returns_cnn_source(self):
        self._inject_model()
        from app.processing.ml_inference import classify_iq
        result = classify_iq(_synthetic_iq())
        self.assertEqual(result["source"], "cnn")

    def test_label_is_string(self):
        self._inject_model()
        from app.processing.ml_inference import classify_iq
        result = classify_iq(_synthetic_iq())
        self.assertIsInstance(result["label"], str)

    def test_confidence_in_range(self):
        self._inject_model()
        from app.processing.ml_inference import classify_iq
        result = classify_iq(_synthetic_iq())
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_probabilities_sum_to_one(self):
        self._inject_model()
        from app.processing.ml_inference import classify_iq
        result = classify_iq(_synthetic_iq())
        probs = result.get("probabilities", {})
        self.assertGreater(len(probs), 0)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=3)

    def test_all_labels_in_probabilities(self):
        self._inject_model()
        from app.processing.ml_inference import classify_iq
        from app.processing.ml_classifier import BASELINE_LABELS
        result = classify_iq(_synthetic_iq())
        for lbl in BASELINE_LABELS:
            self.assertIn(lbl, result["probabilities"])

    def test_model_loaded_true(self):
        self._inject_model()
        from app.processing.ml_inference import model_loaded
        self.assertTrue(model_loaded())

    def test_reload_resets_state(self):
        self._inject_model()
        import app.processing.ml_inference as m
        with patch("pathlib.Path.exists", return_value=False):
            m.reload_model()
        self.assertFalse(m.model_loaded())

    def test_no_nan_in_probabilities(self):
        self._inject_model()
        from app.processing.ml_inference import classify_iq
        result = classify_iq(_synthetic_iq())
        for v in result["probabilities"].values():
            self.assertFalse(np.isnan(v))


if __name__ == "__main__":
    unittest.main()
