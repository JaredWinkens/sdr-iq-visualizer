"""Unit tests for synthetic IQ dataset generation."""

import pickle
import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_synthetic_dataset import (
    generate,
    _am_dsb,
    _fm,
    _bpsk,
    _qpsk,
    _add_awgn,
    _normalise,
    WINDOW,
    GENERATORS,
)
from app.processing.ml_classifier import BASELINE_LABELS, RadioMLDataset


def _rng(seed=0):
    return np.random.default_rng(seed)


class TestSignalGenerators(unittest.TestCase):

    def _check_frame(self, iq, name):
        self.assertEqual(iq.shape, (WINDOW,), f"{name}: wrong shape")
        self.assertEqual(iq.dtype, np.complex64, f"{name}: wrong dtype")
        self.assertFalse(np.any(np.isnan(iq)), f"{name}: NaN in output")
        self.assertFalse(np.any(np.isinf(iq)), f"{name}: Inf in output")
        self.assertGreater(float(np.mean(np.abs(iq) ** 2)), 1e-6,
                           f"{name}: signal power near zero")

    def test_am_dsb(self):
        self._check_frame(_am_dsb(_rng(0)), "AM-DSB")

    def test_fm(self):
        self._check_frame(_fm(_rng(1)), "FM")

    def test_bpsk(self):
        self._check_frame(_bpsk(_rng(2)), "BPSK")

    def test_qpsk(self):
        self._check_frame(_qpsk(_rng(3)), "QPSK")

    def test_generators_produce_distinct_signals(self):
        """Different modulations should not be identical."""
        rng = _rng(42)
        signals = {mod: fn(rng) for mod, fn in GENERATORS.items()}
        mods = list(signals.keys())
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                self.assertFalse(
                    np.allclose(signals[mods[i]], signals[mods[j]]),
                    f"{mods[i]} and {mods[j]} produced identical output",
                )

    def test_multiple_calls_differ(self):
        """Same generator called twice with different rng state should differ."""
        rng = _rng(0)
        a = _bpsk(rng)
        b = _bpsk(rng)
        self.assertFalse(np.allclose(a, b))


class TestNormalise(unittest.TestCase):

    def test_unit_power(self):
        rng = _rng(0)
        iq = _am_dsb(rng)
        normed = _normalise(iq)
        power = float(np.mean(np.abs(normed) ** 2))
        self.assertAlmostEqual(power, 1.0, places=4)

    def test_silence_passthrough(self):
        iq = np.zeros(WINDOW, dtype=np.complex64)
        out = _normalise(iq)
        np.testing.assert_array_equal(out, iq)

    def test_shape_preserved(self):
        iq = _fm(_rng(0))
        self.assertEqual(_normalise(iq).shape, iq.shape)


class TestAddAWGN(unittest.TestCase):

    def test_shape_preserved(self):
        rng = _rng(0)
        iq = _normalise(_bpsk(rng))
        out = _add_awgn(iq, 10.0, rng)
        self.assertEqual(out.shape, iq.shape)
        self.assertEqual(out.dtype, np.complex64)

    def test_noise_changes_signal(self):
        rng = _rng(0)
        iq = _normalise(_qpsk(rng))
        out = _add_awgn(iq, 0.0, rng)
        self.assertFalse(np.allclose(out, iq))

    def test_high_snr_close_to_original(self):
        rng = _rng(0)
        iq = _normalise(_bpsk(rng))
        out = _add_awgn(iq, 60.0, rng)
        np.testing.assert_allclose(np.abs(out), np.abs(iq), rtol=0.02)


class TestGenerate(unittest.TestCase):

    def setUp(self):
        self.snr_values = [-10, 0, 10]
        self.n = 8
        self.ds = generate(self.n, self.snr_values, seed=0)

    def test_keys_cover_all_mods_and_snrs(self):
        for mod in BASELINE_LABELS:
            for snr in self.snr_values:
                self.assertIn((mod, snr), self.ds,
                              f"Missing key ({mod}, {snr})")

    def test_frame_shape(self):
        for (mod, snr), arr in self.ds.items():
            self.assertEqual(arr.shape, (self.n, 2, WINDOW),
                             f"Wrong shape for ({mod}, {snr})")

    def test_frame_dtype(self):
        for arr in self.ds.values():
            self.assertEqual(arr.dtype, np.float32)

    def test_no_nan_or_inf(self):
        for (mod, snr), arr in self.ds.items():
            self.assertFalse(np.any(np.isnan(arr)), f"NaN in ({mod}, {snr})")
            self.assertFalse(np.any(np.isinf(arr)), f"Inf in ({mod}, {snr})")

    def test_different_snrs_differ(self):
        """Frames at SNR -10 should on average differ from frames at SNR 10."""
        for mod in BASELINE_LABELS:
            lo = self.ds[(mod, -10)]
            hi = self.ds[(mod, 10)]
            # They won't be identical (different noise)
            self.assertFalse(np.allclose(lo, hi))

    def test_reproducible_with_seed(self):
        ds2 = generate(self.n, self.snr_values, seed=0)
        for key in self.ds:
            np.testing.assert_array_equal(self.ds[key], ds2[key])

    def test_different_seeds_differ(self):
        ds2 = generate(self.n, self.snr_values, seed=99)
        any_diff = any(
            not np.array_equal(self.ds[k], ds2[k]) for k in self.ds
        )
        self.assertTrue(any_diff)

    def test_radioml_dataset_loads_pickle(self):
        """Pickle should be loadable by RadioMLDataset without error."""
        import tempfile
        tmp = Path(tempfile.mkdtemp()) / "synth.pkl"
        tmp.write_bytes(pickle.dumps(self.ds))
        loaded = RadioMLDataset(str(tmp))
        self.assertGreater(len(loaded), 0)
        x, y = loaded[0]
        import torch
        self.assertIsInstance(x, torch.Tensor)
        self.assertEqual(x.shape, (2, WINDOW))


if __name__ == "__main__":
    unittest.main()
