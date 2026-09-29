"""Unit tests for the RF impairment augmentation pipeline."""

import unittest
import numpy as np

from app.processing.augmentation import (
    awgn,
    frequency_offset,
    phase_shift,
    amplitude_scale,
    dc_offset,
    AugmentationPipeline,
    _to_complex,
    _to_iq2d,
)
from app.processing.ml_classifier import WINDOW_SIZE


def _synthetic_iq(n=WINDOW_SIZE, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def _synthetic_2d(n=WINDOW_SIZE, seed=0):
    iq = _synthetic_iq(n, seed)
    return _to_iq2d(iq)


class TestFormatHelpers(unittest.TestCase):

    def test_complex_passthrough(self):
        iq = _synthetic_iq()
        out = _to_complex(iq)
        self.assertEqual(out.dtype, np.complex64)
        np.testing.assert_array_equal(out, iq.astype(np.complex64))

    def test_2d_to_complex(self):
        x = _synthetic_2d()
        out = _to_complex(x)
        self.assertEqual(out.shape, (WINDOW_SIZE,))
        self.assertTrue(np.iscomplexobj(out))

    def test_complex_to_2d(self):
        iq = _synthetic_iq()
        out = _to_iq2d(iq)
        self.assertEqual(out.shape, (2, WINDOW_SIZE))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out[0], iq.real, atol=1e-6)
        np.testing.assert_allclose(out[1], iq.imag, atol=1e-6)

    def test_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            _to_complex(np.zeros((3, WINDOW_SIZE), dtype=np.float32))


class TestAWGN(unittest.TestCase):

    def test_shape_preserved(self):
        iq = _synthetic_iq()
        out = awgn(iq, snr_db=10.0)
        self.assertEqual(out.shape, iq.shape)
        self.assertEqual(out.dtype, np.complex64)

    def test_noise_added(self):
        iq = _synthetic_iq()
        out = awgn(iq, snr_db=0.0)
        # Output must differ from input
        self.assertFalse(np.allclose(out, iq))

    def test_high_snr_close_to_original(self):
        """At very high SNR the added noise should be negligible."""
        iq = _synthetic_iq()
        out = awgn(iq, snr_db=60.0)
        np.testing.assert_allclose(np.abs(out), np.abs(iq), rtol=0.01)

    def test_silence_unchanged(self):
        iq = np.zeros(WINDOW_SIZE, dtype=np.complex64)
        out = awgn(iq, snr_db=10.0)
        np.testing.assert_array_equal(out, iq)

    def test_snr_approximately_achieved(self):
        """Measured SNR after adding noise should be within ±3 dB of target."""
        rng = np.random.default_rng(7)
        iq = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(np.complex64)
        target = 10.0
        noisy = awgn(iq, snr_db=target)
        noise = noisy - iq
        signal_power = float(np.mean(np.abs(iq) ** 2))
        noise_power = float(np.mean(np.abs(noise) ** 2))
        measured_snr = 10 * np.log10(signal_power / (noise_power + 1e-30))
        self.assertAlmostEqual(measured_snr, target, delta=3.0)


class TestFrequencyOffset(unittest.TestCase):

    def test_shape_preserved(self):
        iq = _synthetic_iq()
        out = frequency_offset(iq, 0.05)
        self.assertEqual(out.shape, iq.shape)
        self.assertEqual(out.dtype, np.complex64)

    def test_magnitude_unchanged(self):
        """Frequency shift must not change per-sample magnitude."""
        iq = _synthetic_iq()
        out = frequency_offset(iq, 0.1)
        np.testing.assert_allclose(np.abs(out), np.abs(iq), atol=1e-5)

    def test_zero_offset_identity(self):
        iq = _synthetic_iq()
        out = frequency_offset(iq, 0.0)
        np.testing.assert_allclose(out, iq, atol=1e-6)


class TestPhaseShift(unittest.TestCase):

    def test_shape_preserved(self):
        iq = _synthetic_iq()
        out = phase_shift(iq, np.pi / 4)
        self.assertEqual(out.shape, iq.shape)

    def test_magnitude_unchanged(self):
        iq = _synthetic_iq()
        out = phase_shift(iq, 1.23)
        np.testing.assert_allclose(np.abs(out), np.abs(iq), atol=1e-5)

    def test_zero_phase_identity(self):
        iq = _synthetic_iq()
        out = phase_shift(iq, 0.0)
        np.testing.assert_allclose(out, iq, atol=1e-6)

    def test_pi_rotation(self):
        """180° rotation should negate the signal."""
        iq = _synthetic_iq()
        out = phase_shift(iq, np.pi)
        np.testing.assert_allclose(out.real, -iq.real, atol=1e-5)
        np.testing.assert_allclose(out.imag, -iq.imag, atol=1e-5)


class TestAmplitudeScale(unittest.TestCase):

    def test_shape_preserved(self):
        iq = _synthetic_iq()
        out = amplitude_scale(iq, 0.9)
        self.assertEqual(out.shape, iq.shape)

    def test_scale_applied(self):
        iq = _synthetic_iq()
        out = amplitude_scale(iq, 2.0)
        np.testing.assert_allclose(np.abs(out), np.abs(iq) * 2.0, atol=1e-5)

    def test_unity_scale_identity(self):
        iq = _synthetic_iq()
        out = amplitude_scale(iq, 1.0)
        np.testing.assert_allclose(out, iq, atol=1e-6)


class TestDCOffset(unittest.TestCase):

    def test_shape_preserved(self):
        iq = _synthetic_iq()
        out = dc_offset(iq, 0.01, -0.01)
        self.assertEqual(out.shape, iq.shape)

    def test_bias_applied(self):
        iq = _synthetic_iq()
        out = dc_offset(iq, 0.1, 0.2)
        np.testing.assert_allclose(out.real - iq.real, 0.1, atol=1e-6)
        np.testing.assert_allclose(out.imag - iq.imag, 0.2, atol=1e-6)

    def test_zero_bias_identity(self):
        iq = _synthetic_iq()
        out = dc_offset(iq, 0.0, 0.0)
        np.testing.assert_array_equal(out, iq)


class TestAugmentationPipeline(unittest.TestCase):

    def _pipeline(self, **kwargs):
        return AugmentationPipeline(seed=42, **kwargs)

    def test_complex_input_complex_output(self):
        p = self._pipeline(p_apply=1.0)
        iq = _synthetic_iq()
        out = p(iq)
        self.assertTrue(np.iscomplexobj(out))
        self.assertEqual(out.shape, iq.shape)

    def test_2d_input_2d_output(self):
        p = self._pipeline(p_apply=1.0)
        x = _synthetic_2d()
        out = p(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, np.float32)

    def test_p_apply_zero_returns_original(self):
        """p_apply=0 means never augment — output should equal input."""
        p = self._pipeline(p_apply=0.0)
        iq = _synthetic_iq()
        out = p(iq)
        np.testing.assert_array_equal(out, iq)

    def test_p_apply_one_changes_signal(self):
        p = self._pipeline(p_apply=1.0)
        iq = _synthetic_iq()
        out = p(iq)
        self.assertFalse(np.allclose(out, iq))

    def test_all_impairments_disabled(self):
        """With all impairments disabled the output should equal the input."""
        p = AugmentationPipeline(
            snr_db_range=None,
            freq_offset_range=None,
            phase_shift_range=None,
            amplitude_scale_range=None,
            dc_offset_std=None,
            p_apply=1.0,
            seed=0,
        )
        iq = _synthetic_iq()
        out = p(iq)
        np.testing.assert_array_equal(out, iq)

    def test_no_nan_or_inf(self):
        p = self._pipeline(p_apply=1.0)
        for seed in range(20):
            iq = _synthetic_iq(seed=seed)
            out = p(iq)
            self.assertFalse(np.any(np.isnan(out)), f"NaN in output (seed={seed})")
            self.assertFalse(np.any(np.isinf(out)), f"Inf in output (seed={seed})")

    def test_describe_returns_dict(self):
        p = self._pipeline()
        d = p.describe()
        self.assertIn("snr_db_range", d)
        self.assertIn("p_apply", d)

    def test_reproducible_with_seed(self):
        """Two pipelines with the same seed should draw the same scalar params
        (freq offset, phase, amplitude, dc) and thus produce the same output
        when AWGN is disabled (AWGN uses np.random global state separately)."""
        iq = _synthetic_iq()
        kwargs = dict(snr_db_range=None, p_apply=1.0,
                      freq_offset_range=(-0.05, 0.05),
                      phase_shift_range=(-np.pi, np.pi),
                      amplitude_scale_range=(0.8, 1.2),
                      dc_offset_std=0.01)
        out1 = AugmentationPipeline(seed=7, **kwargs)(iq.copy())
        out2 = AugmentationPipeline(seed=7, **kwargs)(iq.copy())
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_differ(self):
        iq = _synthetic_iq()
        out1 = AugmentationPipeline(seed=1, p_apply=1.0)(iq.copy())
        out2 = AugmentationPipeline(seed=2, p_apply=1.0)(iq.copy())
        self.assertFalse(np.allclose(out1, out2))

    def test_integrates_with_radioml_dataset(self):
        """RadioMLDataset should accept a transform and return augmented tensors."""
        import pickle, tempfile, pathlib
        import torch

        # Build minimal fake RadioML pickle
        rng = np.random.default_rng(0)
        from app.processing.ml_classifier import BASELINE_LABELS, WINDOW_SIZE
        pkl = {(lbl, 0): rng.standard_normal((4, 2, WINDOW_SIZE)).astype(np.float32)
               for lbl in BASELINE_LABELS}
        tmp = pathlib.Path(tempfile.mkdtemp()) / "rml.pkl"
        tmp.write_bytes(pickle.dumps(pkl))

        from app.processing.ml_classifier import RadioMLDataset
        pipeline = AugmentationPipeline(seed=99, p_apply=1.0)
        ds = RadioMLDataset(str(tmp), transform=pipeline)

        x, y = ds[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertEqual(x.shape, (2, WINDOW_SIZE))
        # Augmented sample should differ from raw stored sample
        raw = torch.from_numpy(ds.samples[0])
        self.assertFalse(torch.allclose(x, raw))


if __name__ == "__main__":
    unittest.main()
