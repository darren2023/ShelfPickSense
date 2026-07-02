"""标注坐标缩放测试。"""

from __future__ import annotations

from pathlib import Path

from fixtures import make_fixture_record


def test_infer_frame_size_uses_pose_aspect_ratio(tmp_path: Path):
    from analysis.records import load_record, resolve_infer_frame_size

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    record = load_record(fixture_dir)
    # fixture annotation 640x480 -> infer 640x480
    assert abs(record.infer_width - 640.0) < 0.01
    assert abs(record.infer_height - 480.0) < 0.01


def test_infer_frame_size_from_manifest(tmp_path: Path):
    import json

    from analysis.records import load_record

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    manifest = {"infer_width": 852, "infer_height": 480}
    (fixture_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    record = load_record(fixture_dir)
    assert record.infer_width == 852.0
    assert record.infer_height == 480.0


def test_build_box_index_prefers_normalized_polygon():
    from analysis.annotation import build_box_index

    annotation = {
        "annotation_size": {"width": 640, "height": 360},
        "shelves": [
            {
                "shelf_code": "S1",
                "boxes": [
                    {
                        "box_id": "A1",
                        "video_polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
                        "video_polygon_norm": [[0.5, 0.25], [0.75, 0.25], [0.75, 0.5], [0.5, 0.5]],
                    }
                ],
            }
        ],
    }
    index = build_box_index(annotation, infer_w=852, infer_h=480)
    poly = index["S1:A1"].polygon
    assert abs(poly[0][0] - 426.0) < 0.01
    assert abs(poly[0][1] - 120.0) < 0.01


def test_record_001_picking_frame_wrist_inside_box():
    data_dir = Path("data/data28/Train/record_001")
    if not data_dir.is_dir():
        return

    from analysis.annotation import build_box_index
    from analysis.features.spatial import point_in_polygon
    from analysis.records import load_record

    record = load_record(data_dir)
    assert record.infer_width == 852.0
    assert record.infer_height == 480.0

    box_index = build_box_index(record.annotation, infer_w=record.infer_width, infer_h=record.infer_height)
    frame = record.frame_index()[685].persons[0]
    left_wrist = frame["keypoints"][9]
    poly = box_index["81:1005"].polygon
    assert point_in_polygon(left_wrist[0], left_wrist[1], poly)
