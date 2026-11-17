"""
Standalone Remote ID page. Run from the project root:

    REMOTE_ID_CAPTURE_DIR=~/CS591/data python remote_id_app.py

then open http://127.0.0.1:8051
"""

import dash
import dash_bootstrap_components as dbc
from dash import html

from app.remote_id.panel import remote_id_layout, register_remote_id_callbacks

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
)


def serve_layout():
    # a function, so reloading the page re-lists the capture files
    return dbc.Container(
        [html.H2("Remote ID detection", className="my-4"), remote_id_layout()],
        fluid=True,
    )


app.layout = serve_layout
register_remote_id_callbacks(app)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8051, debug=False)
