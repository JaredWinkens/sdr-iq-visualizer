
from dash import dcc, html
import dash_bootstrap_components as dbc
from chatbot.layout import load_mini_chatbot_layout

layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [html.H1("Real-time SDR Data Analysis", className="my-4")],
                className="d-flex justify-content-center"
            ),
        ),
        dbc.Row(
            html.Div([
                dcc.Download(id='sample-download-location'),
                
                dbc.Button([html.I(className='bi bi-plug-fill me-2'),'Connect SDR'], id='connect-btn', n_clicks=0,
                            style={'margin': '10px', 'padding': '10px'}),
                dbc.Button([html.I(className='bi bi-arrow-clockwise me-2'),'Reconnect SDR'], id='reconnect-btn', n_clicks=0,
                            style={'margin': '10px', 'padding': '10px'}),
                dbc.Button([html.I(className='bi bi-play-fill me-2'),'Start Streaming'], id='start-btn', n_clicks=0,
                            style={'margin': '10px', 'padding': '10px'}),
                dbc.Button([html.I(className='bi bi-stop-fill me-2'),'Stop Streaming'], id='stop-btn', n_clicks=0,
                            style={'margin': '10px', 'padding': '10px'}),
                dbc.Button([html.I(className='bi bi-download me-2'),'Download Sample'], id='download-btn', n_clicks=0,
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
                    dbc.CardBody([
                        html.H3("Signal Classification"),
                        html.Div("No Data to Display", id='classification-text', style={'fontSize': '16px', 'whiteSpace': 'pre-wrap'})
                    ]),
                    class_name="home-page-card",
                ), md=12)
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Card(
                    dbc.CardBody(
                        [
                            html.H3("Time Domain (I/Q Samples)"),
                            dcc.Graph(
                                id='time-domain-graph',
                                figure={
                                    'data': [],
                                    'layout': {
                                        'xaxis': {"visible": False},
                                        'yaxis': {"visible": False},
                                        'annotations': [
                                            {
                                                "text": 'No Data to Display',
                                                "xref": "paper",
                                                "yref": "paper",
                                                "showarrow": False,
                                                "font": {"size": 28, "color": "gray"},
                                            }
                                        ],
                                    }
                                    
                                }
                            )
                        ]
                    ),
                    class_name="home-page-card",  # Add margin-bottom for spacing between cards
                ), md=6, class_name='home-page-column'),  # Takes 6 of 12 columns on medium screens and up
                dbc.Col(dbc.Card(
                    dbc.CardBody(
                        [
                            html.H3("Frequency Domain (Power Spectrum)"),
                            dcc.Graph(
                                id='freq-domain-graph',
                                figure={
                                    'data': [],
                                    'layout': {
                                        'xaxis': {"visible": False},
                                        'yaxis': {"visible": False},
                                        'annotations': [
                                            {
                                                "text": 'No Data to Display',
                                                "xref": "paper",
                                                "yref": "paper",
                                                "showarrow": False,
                                                "font": {"size": 28, "color": "gray"},
                                            }
                                        ],
                                    }
                                    
                                }
                            )
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
                            dcc.Graph(
                                id='waterfall-graph',
                                figure={
                                    'data': [],
                                    'layout': {
                                        'xaxis': {"visible": False},
                                        'yaxis': {"visible": False},
                                        'annotations': [
                                            {
                                                "text": 'No Data to Display',
                                                "xref": "paper",
                                                "yref": "paper",
                                                "showarrow": False,
                                                "font": {"size": 28, "color": "gray"},
                                            }
                                        ],
                                    }
                                    
                                }
                            )
                        ]
                    ),
                    class_name="home-page-card",  # Add margin-bottom for spacing between cards
                ), md=6, class_name='home-page-column'),  # Takes 6 of 12 columns on medium screens and up
                dbc.Col(dbc.Card(
                    dbc.CardBody(
                        [
                            html.H3("I/Q Constellation Diagram"),
                            dcc.Graph(
                                id='constellation-plot',
                                figure={
                                    'data': [],
                                    'layout': {
                                        'xaxis': {"visible": False},
                                        'yaxis': {"visible": False},
                                        'annotations': [
                                            {
                                                "text": 'No Data to Display',
                                                "xref": "paper",
                                                "yref": "paper",
                                                "showarrow": False,
                                                "font": {"size": 28, "color": "gray"},
                                            }
                                        ],
                                    }
                                    
                                }
                            ),
                        ]
                    ),
                    class_name="home-page-card",  # Add margin-bottom for spacing between cards
                ), md=6, class_name='home-page-column'),
            ]
        ),
        
        html.Div(id='alert-div'),
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