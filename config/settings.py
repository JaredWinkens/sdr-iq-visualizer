from dotenv import load_dotenv
import os
load_dotenv()

# Dash App Configuration
DASH_CONFIGS = {
    'host': "0.0.0.0",
    'port': 8050
}

CHATBOT_CONFIG = {
    'api_key': os.getenv('GOOGLE_API_KEY'),
    'model': "gemini-2.5-flash"
}

# SDR Device Configuration
SDR_CONFIGS = {
    'pluto': {
        'uri': 'ip:192.168.2.1',  # Default PlutoSDR IP
        'sample_rate': 1000000,   # 1 MSPS
        'center_freq': 2400000000,  # 2.4 GHz
        'rx_lo': 2400000000,
        'rx_rf_bandwidth': 4000000,  # 4 MHz
        'rx_buffer_size': 2**12
    },
    'usrp': {
        'uri': 'ip:192.168.10.2',  # USRP device IP
        'sample_rate': 1000000,
        'center_freq': 100000000,  # 100 MHz
        'rx_lo': 100000000,
        'rx_rf_bandwidth': 1000000,
        'rx_buffer_size': 2**12
    },
    'local': {
        'uri': 'local:',  # Local device
        'sample_rate': 1000000,
        'center_freq': 915000000,  # 915 MHz
        'rx_lo': 915000000,
        'rx_rf_bandwidth': 2000000,
        'rx_buffer_size': 2**12
    }
}

# Display Configuration
DISPLAY_CONFIG = {
    'update_interval_ms': 100,  # Graph update interval
    'waterfall_history': 100,   # Number of FFT frames to keep
    'max_queue_size': 100       # Maximum data queue size
}

# Frequency bands of interest (in Hz)
FREQUENCY_BANDS = {
    'ISM_2.4GHz': (2400000000, 2500000000),
    'ISM_915MHz': (902000000, 928000000),
    'FM_Radio': (88000000, 108000000),
    'Amateur_2m': (144000000, 148000000),
    'Amateur_70cm': (420000000, 450000000)
}