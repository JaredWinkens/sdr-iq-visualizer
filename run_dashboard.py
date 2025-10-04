# run_dashboard.py

import logging
from dash import Dash
from dash import html
import dashboard.layout as dashboard_layout
import dashboard.callbacks as dashboard_callbacks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the app
app = Dash(__name__)

# Attach layout
app.layout = dashboard_layout.layout

# Register callbacks
dashboard_callbacks.register_callbacks(app)

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
    app.run(debug=True, host="127.0.0.1", port=8050)
