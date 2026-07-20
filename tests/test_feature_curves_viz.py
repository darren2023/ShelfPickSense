"""Feature curve visualization tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_feature_curves_viz_generates_html(tmp_path: Path):
    from analysis.feature_curves_viz import build_feature_curves_payload, render_feature_curves_html, write_feature_curves_html

    features_dir = tmp_path / "features"
    features_dir.mkdir()
    frame_path = features_dir / "frame_features.parquet"
    pd.DataFrame(
        [
            {
                "record_id": "record_001",
                "frame_idx": 1,
                "person_track_id": 1,
                "shelf_code": "S1",
                "is_picking": False,
                "spatial.any_wrist_inside_box": 0.0,
                "temporal.consecutive_hit_3": 0.0,
            },
            {
                "record_id": "record_001",
                "frame_idx": 2,
                "person_track_id": 1,
                "shelf_code": "S1",
                "is_picking": True,
                "spatial.any_wrist_inside_box": 1.0,
                "temporal.consecutive_hit_3": 1.0,
            },
        ]
    ).to_parquet(frame_path, index=False)

    payload = build_feature_curves_payload(frame_path)
    assert payload["row_count"] == 2
    assert payload["person_track_ids"] == ["1"]
    assert payload["shelf_codes"] == ["S1"]
    assert "spatial.any_wrist_inside_box" in payload["features"]

    html = render_feature_curves_html(payload)
    assert "canvas" in html
    assert "person_track_id" in html
    assert "Zoom +" in html
    assert "resetView" in html
    assert "wheel" in html
    assert "Click a curve point" in html
    assert "inspectNearestPoint" in html
    assert "spatial.any_wrist_inside_box" in html

    out = write_feature_curves_html(features_dir)
    assert out.is_file()
    assert "Feature Curves" in out.read_text(encoding="utf-8")
