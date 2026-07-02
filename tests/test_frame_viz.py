"""帧可视化测试。"""

from __future__ import annotations

from pathlib import Path

from fixtures import make_fixture_record


def test_viz_frames_generates_html(tmp_path: Path):
    from analysis.frame_viz import build_viz_payload, render_viz_html, write_frame_viz_html
    from analysis.records import load_record

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    record = load_record(fixture_dir)
    payload = build_viz_payload(record, max_frames=3)
    assert len(payload["frames"]) == 3
    assert payload["boxes"]
    html = render_viz_html(payload)
    assert "canvas" in html
    assert "record_001" in html

    out = write_frame_viz_html(fixture_dir, tmp_path / "viz.html", max_frames=5)
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "skeleton_edges" in text or "DATA" in text
