"""Singleton model loader for live dashboard inference.

Keeps one model instance in memory for the lifetime of the Dash process so
every callback invocation does not reload from disk.

Usage (dashboard callbacks):
    from app.processing.ml_inference import classify_iq

    result = classify_iq(samples)   # samples: complex numpy array
    # result = {"label": "FM", "confidence": 0.91, "probabilities": {...},
    #           "source": "cnn"}
    # or, when no model is available:
    # result = {"label": "No model loaded", "confidence": 0.0, "source": "none"}

The model path is read from the environment variable ML_MODEL_PATH, falling
back to "models/baseline_cnn.pt" relative to the project root.  Set the
variable before launching the dashboard to point at a trained checkpoint:

    ML_MODEL_PATH=models/baseline_cnn.pt python main.py
"""

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state - loaded once on first call
# ---------------------------------------------------------------------------

_model = None
_labels = None
_load_attempted = False


def _try_load_model() -> bool:
    """Attempt to load the CNN from disk.  Returns True on success."""
    global _model, _labels, _load_attempted
    _load_attempted = True

    model_path = os.environ.get("ML_MODEL_PATH", "models/baseline_cnn.pt")
    path = Path(model_path)

    if not path.exists():
        logger.warning(
            "ML model not found at %s. "
            "Train one with: python scripts/run_experiment.py --data <pkl> "
            "or set ML_MODEL_PATH to point at an existing checkpoint.",
            path,
        )
        return False

    try:
        from app.processing.ml_classifier import load_model, BASELINE_LABELS
        model, meta = load_model(str(path))
        model.eval()
        _model = model
        _labels = BASELINE_LABELS
        logger.info("ML classifier loaded from %s (epoch %s)", path, meta.get("epoch", "?"))
        return True
    except Exception as exc:
        logger.error("Failed to load ML model: %s", exc)
        return False


def classify_iq(samples: np.ndarray) -> dict:
    """Classify a complex IQ array using the CNN.

    Parameters
    ----------
    samples:
        Complex numpy array of raw IQ samples (length >= 128).

    Returns
    -------
    dict with keys:
        label       : str   - predicted modulation type
        confidence  : float - softmax probability of top class
        probabilities: dict - per-class probabilities
        source      : str   - "cnn" | "none"
    """
    global _model, _labels, _load_attempted

    # Lazy load on first call
    if not _load_attempted:
        _try_load_model()

    if _model is None:
        return {
            "label": "No model loaded",
            "confidence": 0.0,
            "probabilities": {},
            "source": "none",
        }

    try:
        from app.processing.ml_classifier import predict_iq
        result = predict_iq(_model, samples, _labels)
        result["source"] = "cnn"
        return result
    except Exception as exc:
        logger.error("Inference error: %s", exc)
        return {
            "label": "Inference error",
            "confidence": 0.0,
            "probabilities": {},
            "source": "none",
        }


def model_loaded() -> bool:
    """Return True if a model is currently in memory."""
    return _model is not None


def reload_model() -> bool:
    """Force a reload from disk (e.g. after training completes)."""
    global _model, _labels, _load_attempted
    _model = None
    _labels = None
    _load_attempted = False
    return _try_load_model()
