import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("remote_id_detector", reason="install the decoder first: pip install -e ~/CS591")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from remote_id_detector import DroneRecord  # noqa: E402
from app.remote_id import panel  # noqa: E402


def _record(**kw):
    base = dict(adv_address="AA:BB:CC:DD:EE:FF", uas_id="TEST123", operator_id="OP-1",
                n_detections=3)
    base.update(kw)
    return DroneRecord(**base)


def test_row_for_device_without_position():
    row = panel.record_to_row(_record(altitude_geo_m=117.5))
    assert row["position"] == "unknown"
    assert row["altitude"] == "117.5"
    assert row["speed"] == ""
    assert row["seen"] == 3


def test_row_for_device_with_position():
    row = panel.record_to_row(_record(latitude=42.8864, longitude=-78.8784, speed_mps=12.5))
    assert row["position"] == "42.886400, -78.878400"
    assert row["speed"] == "12.5"


def test_map_has_no_markers_without_a_position():
    fig = panel.build_map_figure([_record()])
    assert len(fig.data) == 0


def test_map_marks_aircraft_and_operator():
    rec = _record(latitude=42.8864, longitude=-78.8784,
                  operator_latitude=42.8860, operator_longitude=-78.8790)
    fig = panel.build_map_figure([rec])
    assert [t.name for t in fig.data] == ["Aircraft", "Operator"]
    assert fig.layout.map.zoom == 14


def test_scan_with_nothing_selected_or_missing_file(tmp_path):
    summary, rows, fig = panel.run_scan(None, 37, capture_dir=tmp_path)
    assert "Choose a capture" in summary and rows == []
    summary, rows, fig = panel.run_scan("nope.npy", 37, capture_dir=tmp_path)
    assert "Choose a capture" in summary and rows == []


def test_scan_rejects_names_outside_the_capture_dir(tmp_path):
    # only files that are really in the capture directory can be opened
    summary, rows, _ = panel.run_scan("../secret.npy", 37, capture_dir=tmp_path)
    assert "Choose a capture" in summary and rows == []


def test_scan_reports_decode_errors_instead_of_crashing(tmp_path):
    np.save(tmp_path / "bad.npy", np.zeros(10, dtype=np.complex64))
    summary, rows, _ = panel.run_scan("bad.npy", 37, capture_dir=tmp_path)
    assert isinstance(summary, str) and rows == []


def test_layout_builds():
    assert panel.remote_id_layout() is not None
