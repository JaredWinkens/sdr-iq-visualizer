import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import importlib

# Mock dotenv before importing settings
sys.modules['dotenv'] = MagicMock()

from app.config import settings

class TestSettings(unittest.TestCase):

    @patch.dict(os.environ, {'DASH_HOST': '1.2.3.4', 'DASH_PORT': '9090'})
    def test_dash_config_env_vars(self):
        # Reload the module to pick up env vars
        importlib.reload(settings)
        self.assertEqual(settings.DASH_CONFIGS['host'], '1.2.3.4')
        self.assertEqual(settings.DASH_CONFIGS['port'], 9090)

    @patch.dict(os.environ, {}, clear=True)
    def test_dash_config_defaults(self):
        # Reload the module to pick up defaults
        importlib.reload(settings)
        self.assertEqual(settings.DASH_CONFIGS['host'], '0.0.0.0')
        self.assertEqual(settings.DASH_CONFIGS['port'], 8050)

    def test_sdr_configs_structure(self):
        self.assertIn('pluto', settings.SDR_CONFIGS)
        self.assertIn('usrp', settings.SDR_CONFIGS)
        self.assertIn('local', settings.SDR_CONFIGS)
        
        pluto = settings.SDR_CONFIGS['pluto']
        self.assertIn('uri', pluto)
        self.assertIn('sample_rate', pluto)

if __name__ == '__main__':
    unittest.main()
