import dash
from dash import html
import dash_bootstrap_components as dbc
from config import HOST, PORT
from layouts.homepage import load_homepage_layout

stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(
    __name__,
    external_stylesheets=stylesheets,
    suppress_callback_exceptions=True,
)

app.layout = load_homepage_layout()

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=True)