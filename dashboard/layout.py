
from dash import dcc, html

layout = html.Div([
    html.H1("Real-time SDR Data Visualization", style={'textAlign': 'center'}),

    html.Div([
        html.Button('Connect SDR', id='connect-btn', n_clicks=0,
                    style={'margin': '10px', 'padding': '10px'}),
        html.Button('Reconnect SDR', id='reconnect-btn', n_clicks=0,
                    style={'margin': '10px', 'padding': '10px'}),
        html.Button('Start Streaming', id='start-btn', n_clicks=0,
                    style={'margin': '10px', 'padding': '10px'}),
        html.Button('Stop Streaming', id='stop-btn', n_clicks=0,
                    style={'margin': '10px', 'padding': '10px'}),
        html.Div(id='status-div', style={'margin': '10px', 'fontWeight': 'bold'})
    ], style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.H3("Time Domain (I/Q Samples)"),
            dcc.Graph(id='time-domain-graph')
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            html.H3("Frequency Domain (Power Spectrum)"),
            dcc.Graph(id='freq-domain-graph')
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),

    html.Div([
        html.H3("Waterfall Plot (Spectrogram)"),
        dcc.Graph(id='waterfall-graph')
    ]),

    # Interval for updating graphs
    dcc.Interval(
        id='interval-component',
        interval=300,  # update every 300ms
        n_intervals=0,
        disabled=True
    )
])
