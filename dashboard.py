import dash
from dash import html
import dash_bootstrap_components as dbc
from config import DASH_CONFIGS
from layouts.homepage import load_homepage_layout

stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP]

app = dash.Dash(
    __name__,
    external_stylesheets=stylesheets,
    suppress_callback_exceptions=True,
)

app.layout = load_homepage_layout()

if __name__ == '__main__':
    app.run(host=DASH_CONFIGS['host'], port=DASH_CONFIGS['port'], debug=True)