
from dash import dcc, html
import dash_bootstrap_components as dbc
from chatbot.layout import load_mini_chatbot_layout

layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [html.H1("Real-time SDR Data Visualization", className="my-4")],
                className="d-flex justify-content-center"
            ),
        ),
        dbc.Row(
            html.Div([
                dbc.Button('Connect SDR', id='connect-btn', n_clicks=0,
                            style={'margin': '10px', 'padding': '10px'}),
                dbc.Button('Reconnect SDR', id='reconnect-btn', n_clicks=0,
                            style={'margin': '10px', 'padding': '10px'}),
                dbc.Button('Start Streaming', id='start-btn', n_clicks=0,
                            style={'margin': '10px', 'padding': '10px'}),
                dbc.Button('Stop Streaming', id='stop-btn', n_clicks=0,
                            style={'margin': '10px', 'padding': '10px'}),
                dbc.Checklist(
                    options=[{"label": " Pause Updates", "value": 1}],
                    value=[],
                    id="pause-toggle",
                    switch=True,
                    inline=True,
                    style={'display': 'inline-block', 'margin': '10px'}
                ),
                html.Span(id='pause-status', style={'margin': '10px', 'color': '#666'}),
                html.Div(id='status-div', style={'margin': '10px', 'fontWeight': 'bold'})
            ], style={'textAlign': 'center'}),
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Card(
                    dbc.CardBody(
                        [
                            html.H3("Time Domain (I/Q Samples)"),
                            dcc.Graph(id='time-domain-graph')
                        ]
                    ),
                    class_name="home-page-card",  # Add margin-bottom for spacing between cards
                ), md=6, class_name='home-page-column'),  # Takes 6 of 12 columns on medium screens and up
                dbc.Col(dbc.Card(
                    dbc.CardBody(
                        [
                            html.H3("Frequency Domain (Power Spectrum)"),
                            dcc.Graph(id='freq-domain-graph')
                        ]
                    ),
                    class_name="home-page-card",  # Add margin-bottom for spacing between cards
                ), md=6, class_name='home-page-column'),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Card(
                    dbc.CardBody(
                        [
                            html.H3("Waterfall Plot (Spectrogram)"),
                            dcc.Graph(id='waterfall-graph')
                        ]
                    ),
                    class_name="home-page-card",  # Add margin-bottom for spacing between cards
                ), md=6, class_name='home-page-column'),  # Takes 6 of 12 columns on medium screens and up
                dbc.Col(dbc.Card(
                    dbc.CardBody(
                        [
                            html.H3("I/Q Constellation Diagram"),
                            dcc.Graph(id='constellation-plot'),
                        ]
                    ),
                    class_name="home-page-card",  # Add margin-bottom for spacing between cards
                ), md=6, class_name='home-page-column'),
            ]
        ),
        load_mini_chatbot_layout("Hi! I am an interactive chatbot design to analyze radio frequency data and provide useful insights based on that data."),
        # Interval for updating graphs
        dcc.Interval(
            id='interval-component',
            interval=300,  # update every 300ms
            n_intervals=0,
            disabled=True
        )
    ],
    fluid=True,
)