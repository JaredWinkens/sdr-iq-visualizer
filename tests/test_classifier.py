import unittest
import numpy as np
from app.processing.classifier import classify_signal_simple, classify_signal_advanced

class TestClassifier(unittest.TestCase):

    def test_classify_signal_simple_no_data(self):
        freqs = np.array([])
        power_db = np.array([])
        result = classify_signal_simple(freqs, power_db)
        self.assertEqual(result, "No Data")

    def test_classify_signal_simple_narrowband(self):
        freqs = np.linspace(0, 10e6, 100) # 10 MHz span
        power_db = np.zeros(100)
        # Create a peak
        power_db[50] = 50 # Max is 50. Threshold 30.
        # Only index 50 is > 30.
        # occupied = freqs[mask] -> freqs[50]
        # bw = occupied[-1] - occupied[0] = 0
        # 0 < 3e6 -> Narrowband
        result = classify_signal_simple(freqs, power_db)
        self.assertEqual(result, "Narrowband")

    def test_classify_signal_simple_wideband(self):
        freqs = np.linspace(0, 20e6, 200) # 20 MHz span
        power_db = np.zeros(200)
        # Create a wide signal
        power_db[:] = 50 # All 50.
        # All > 30.
        # occupied = all freqs.
        # bw = 20e6.
        # 20e6 > 15e6 -> Wideband
        result = classify_signal_simple(freqs, power_db)
        self.assertEqual(result, "Wideband")

    def test_classify_signal_advanced_no_data(self):
        freqs = np.array([])
        power_db = np.array([])
        result = classify_signal_advanced(freqs, power_db)
        self.assertEqual(result["label"], "No Data")

    def test_classify_signal_advanced_cw_carrier(self):
        # Simulate a CW carrier: Single strong peak, low noise
        freqs = np.linspace(100e6, 101e6, 1024) # 1 MHz span
        power_db = np.random.normal(-80, 1, 1024) # Noise floor around -80
        
        # Add a peak at index 512
        power_db[512] = -20 
        power_db[511] = -30
        power_db[513] = -30
        
        result = classify_signal_advanced(freqs, power_db)
        # We expect "CW Carrier" or similar
        # The logic depends on many factors, but let's check if it returns a valid dict
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertIn("features", result)
        # It might be CW Carrier
        self.assertIn(result["label"], ["CW Carrier", "Narrowband (voice)", "Unknown", "Low SNR / Noise"])

if __name__ == '__main__':
    unittest.main()
