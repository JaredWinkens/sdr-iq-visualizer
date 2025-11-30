# run_dashboard.py

import logging
import os
from dash import dash
import app.dashboard.layout as dashboard_layout
import app.dashboard.callbacks as dashboard_callbacks
import app.chatbot.callbacks as chatbot_callbacks
import dash_bootstrap_components as dbc
from app.config.settings import DASH_CONFIGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the app
stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP]
app = dash.Dash(
    __name__,
    external_stylesheets=stylesheets,
    suppress_callback_exceptions=True,
)

# Attach layout
app.layout = dashboard_layout.layout

# Register callbacks
dashboard_callbacks.register_callbacks(app)
chatbot_callbacks.register_callbacks(app)

host = DASH_CONFIGS['host']
port = DASH_CONFIGS['port']
debug_flag = os.getenv('DASH_DEBUG', 'false').lower() == 'true'

# Print startup messages
print("Starting SDR Real-time Visualization Dashboard")
print(f"Open your browser and go to: http://localhost:{port}")
print("\nInstructions:")
print("1. Click 'Connect SDR' to establish connection")
print("2. Click 'Reconnect SDR' to reconnect if connection is lost")
print("3. Click 'Start Streaming' to begin data acquisition")
print("4. Click 'Stop Streaming' to stop data acquisition")
print("\nEnvironment:")
print(f"HOST={host} PORT={port} DEBUG={'on' if debug_flag else 'off'}")
print("\nNote: Ensure your SDR device (e.g., PlutoSDR) is reachable from the container/network")

# Run server
if __name__ == "__main__":
    app.run(debug=debug_flag, dev_tools_hot_reload=debug_flag, host=host, port=port)
