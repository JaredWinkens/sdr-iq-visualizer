"""
Remote ID panel for the SDR IQ visualizer (replay mode).

Lets you pick a saved IQ capture (.npy), run it through the Remote ID
decoder from the CS591 independent study, and see the detected devices
in a table and on a map. Live scanning from the Pluto is not wired up
yet; the decoder needs about 4 Msps and captures of a second or more,
which the existing streaming path does not provide.

Needs the decoder installed in this environment:

    pip install -e ~/CS591

and, to point the panel at the captures:

    REMOTE_ID_CAPTURE_DIR=~/CS591/data python remote_id_app.py

A position of exactly 0, 0 is the spec's "unknown" value, so a device
without a real position appears in the table but never on the map.
"""

import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html
import dash_bootstrap_components as dbc

from remote_id_detector import detect_remote_id, merge_by_device

SAMPLE_RATE = 4_000_000.0
CAPTURE_DIR = Path(os.environ.get("REMOTE_ID_CAPTURE_DIR", "data")).expanduser()

TABLE_COLUMNS = [
    {"name": "Address", "id": "address"},
    {"name": "UAS ID", "id": "uas_id"},
    {"name": "Operator ID", "id": "operator_id"},
    {"name": "Position", "id": "position"},
    {"name": "Altitude (m)", "id": "altitude"},
    {"name": "Speed (m/s)", "id": "speed"},
    {"name": "Packets", "id": "seen"},
]


def list_capture_files(capture_dir: Path = None) -> list:
    capture_dir = CAPTURE_DIR if capture_dir is None else Path(capture_dir)
    if not capture_dir.is_dir():
        return []
    return sorted(p.name for p in capture_dir.glob("*.npy"))


def record_to_row(rec) -> dict:
    return {
        "address": rec.adv_address,
        "uas_id": rec.uas_id or "",
        "operator_id": rec.operator_id or "",
        "position": (f"{rec.latitude:.6f}, {rec.longitude:.6f}"
                     if rec.has_position else "unknown"),
        "altitude": "" if rec.altitude_geo_m is None else f"{rec.altitude_geo_m:.1f}",
        "speed": "" if rec.speed_mps is None else f"{rec.speed_mps:.1f}",
        "seen": rec.n_detections,
    }


def build_map_figure(records) -> go.Figure:
    """Markers for devices with a real position, and for known operator positions."""
    fig = go.Figure()
    drones = [r for r in records if r.has_position]
    operators = [r for r in records
                 if r.operator_latitude is not None and r.operator_longitude is not None]
    if drones:
        fig.add_trace(go.Scattermap(
            lat=[r.latitude for r in drones], lon=[r.longitude for r in drones],
            mode="markers+text", name="Aircraft",
            text=[r.uas_id or r.adv_address for r in drones], textposition="top right",
            marker={"size": 14}))
    if operators:
        fig.add_trace(go.Scattermap(
            lat=[r.operator_latitude for r in operators],
            lon=[r.operator_longitude for r in operators],
            mode="markers", name="Operator", marker={"size": 10}))
    points = [(r.latitude, r.longitude) for r in drones] + \
             [(r.operator_latitude, r.operator_longitude) for r in operators]
    if points:
        center = {"lat": sum(p[0] for p in points) / len(points),
                  "lon": sum(p[1] for p in points) / len(points)}
        zoom = 14
    else:
        center, zoom = {"lat": 20.0, "lon": 0.0}, 1
    fig.update_layout(
        map={"style": "open-street-map", "center": center, "zoom": zoom},
        margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=380,
        legend={"orientation": "h"},
    )
    return fig


def run_scan(filename, channel, capture_dir: Path = None):
    """Decode one capture. Returns (summary text, table rows, map figure)."""
    capture_dir = CAPTURE_DIR if capture_dir is None else Path(capture_dir)
    empty = build_map_figure([])
    if not filename or filename not in list_capture_files(capture_dir):
        return "Choose a capture file first.", [], empty
    try:
        iq = np.load(capture_dir / filename)
        detections = detect_remote_id(iq, sample_rate=SAMPLE_RATE,
                                      channel_index=int(channel or 37))
    except Exception as exc:  # shown to the user rather than crashing the callback
        return f"Could not decode {filename}: {exc}", [], empty
    records = merge_by_device(detections)
    n_located = sum(1 for r in records if r.has_position)
    summary = (f"{filename}: {len(detections)} Remote ID packet(s) from "
               f"{len(records)} device(s); {n_located} with a position.")
    if records and not n_located:
        summary += " Devices with no GPS fix report an unknown position, so none are on the map."
    return summary, [record_to_row(r) for r in records], build_map_figure(records)


def remote_id_layout():
    files = list_capture_files()
    default = "rid37.npy" if "rid37.npy" in files else (files[0] if files else None)
    return dbc.Card(dbc.CardBody([
        html.H4("Remote ID detections", className="card-title"),
        html.P("Replays a saved capture through the decoder. Live scanning from the "
               "Pluto is not connected yet.", className="text-muted"),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id="rid-file", options=files, value=default,
                                 placeholder="Select a capture (.npy)", clearable=False), md=6),
            dbc.Col(dcc.Dropdown(id="rid-channel", value=37, clearable=False,
                                 options=[{"label": f"Channel {c}", "value": c}
                                          for c in (37, 38, 39)]), md=3),
            dbc.Col(dbc.Button("Decode capture", id="rid-scan-btn", color="primary"), md=3),
        ], className="mb-3"),
        dcc.Loading(html.Div([
            html.Div(id="rid-summary", className="mb-2"),
            dash_table.DataTable(
                id="rid-table", columns=TABLE_COLUMNS, data=[], page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "6px"}),
            dcc.Graph(id="rid-map", figure=build_map_figure([]),
                      config={"displaylogo": False}),
        ])),
    ]), className="mb-4")


def register_remote_id_callbacks(app):
    @app.callback(
        Output("rid-summary", "children"),
        Output("rid-table", "data"),
        Output("rid-map", "figure"),
        Input("rid-scan-btn", "n_clicks"),
        State("rid-file", "value"),
        State("rid-channel", "value"),
        prevent_initial_call=True,
    )
    def _scan(n_clicks, filename, channel):
        return run_scan(filename, channel)
