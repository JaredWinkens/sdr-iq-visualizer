import unittest
from unittest.mock import patch, MagicMock
import queue
import time
import sys

# Mock adi before importing streamer
if 'adi' not in sys.modules:
    sys.modules['adi'] = MagicMock()

from app.sdr.streamer import SDRDataStreamer

class TestSDRDataStreamer(unittest.TestCase):

    def setUp(self):
        self.streamer = SDRDataStreamer()
        # Reset the mock for each test
        sys.modules['adi'].reset_mock()
        sys.modules['adi'].Pluto.side_effect = None
        sys.modules['adi'].Pluto.return_value = MagicMock()

    def test_init(self):
        self.assertEqual(self.streamer.uri, "ip:192.168.2.1")
        self.assertEqual(self.streamer.sample_rate, 1_000_000)
        self.assertFalse(self.streamer.connected)
        self.assertIsInstance(self.streamer.data_queue, queue.Queue)

    def test_connect_success(self):
        mock_adi = sys.modules['adi']
        mock_sdr = MagicMock()
        mock_adi.Pluto.return_value = mock_sdr
        
        success = self.streamer.connect()
        
        self.assertTrue(success)
        self.assertTrue(self.streamer.connected)
        self.assertIsNotNone(self.streamer.sdr)
        
        # Verify settings were applied
        self.assertEqual(mock_sdr.sample_rate, 1000000)
        self.assertEqual(mock_sdr.rx_lo, 2400000000)

    def test_connect_failure(self):
        mock_adi = sys.modules['adi']
        mock_adi.Pluto.side_effect = Exception("Connection failed")
        
        success = self.streamer.connect()
        
        self.assertFalse(success)
        self.assertFalse(self.streamer.connected)
        self.assertIsNone(self.streamer.sdr)

    def test_is_connected(self):
        self.streamer.sdr = MagicMock()
        self.streamer.connected = True
        self.assertTrue(self.streamer.is_connected())
        
        self.streamer.sdr = None
        self.assertFalse(self.streamer.is_connected())

    def test_start_stop_streaming(self):
        # Mock connect
        self.streamer.connect()
        
        with patch('threading.Thread') as mock_thread:
            success = self.streamer.start_streaming()
            self.assertTrue(success)
            self.assertTrue(self.streamer.running)
            mock_thread.assert_called_once()
            
            self.streamer.stop_streaming()
            self.assertFalse(self.streamer.running)
            mock_thread.return_value.join.assert_called_once()

    def test_start_streaming_not_connected(self):
        success = self.streamer.start_streaming()
        self.assertFalse(success)

if __name__ == '__main__':
    unittest.main()
