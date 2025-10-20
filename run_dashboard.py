# run_dashboard.py

import logging
from dash import dash
import dashboard.layout as dashboard_layout
import dashboard.callbacks as dashboard_callbacks
import chatbot.callbacks as chatbot_callbacks
import dash_bootstrap_components as dbc
from config.settings import DASH_CONFIGS

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

# Print startup messages
print("Starting SDR Real-time Visualization Dashboard")
print("Open your browser and go to: http://127.0.0.1:8050")
print("\nInstructions:")
print("1. Click 'Connect SDR' to establish connection")
print("2. Click 'Reconnect SDR' to reconnect if connection is lost")
print("3. Click 'Start Streaming' to begin data acquisition")
print("4. Click 'Stop Streaming' to stop data acquisition")
print("\nNote: Make sure your SDR device is connected and accessible")

# Run server
if __name__ == "__main__":
    app.run(debug=True, host=DASH_CONFIGS['host'], port=DASH_CONFIGS['port'])
