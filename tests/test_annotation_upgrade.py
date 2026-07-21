"""Annotation schema upgrade tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fixtures import _skeleton_row


def _write_legacy_annotation_record(record_dir: Path) -> Path:
    record_dir.mkdir(parents=True, exist_ok=True)
    annotation = {
        "shelf_corners": [[10, 10], [200, 10], [200, 200], [10, 200]],
        "annotation_size": {"width": 640, "height": 360},
        "grid_shape": [4, 4],
        "boxes": [
            {
                "box_id": "3081",
                "layer": 1,
                "column": 1,
                "video_polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
                "video_polygon_norm": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]],
            }
        ],
        "source_info": {"camera_name": "MAP_18", "shelf_code": "MAP_18"},
    }
    (record_dir / "annotation.json").write_text(json.dumps(annotation, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([_skeleton_row(1, left_wrist=(150, 150), right_wrist=(160, 155))]).to_parquet(
        record_dir / "skeleton.parquet",
        index=False,
    )
    return record_dir


def test_upgrade_annotation_payload_wraps_legacy_boxes():
    from analysis.annotation import build_box_index
    from analysis.annotation_upgrade import upgrade_annotation_payload

    annotation = {
        "annotation_size": {"width": 640, "height": 360},
        "grid_shape": [4, 4],
        "shelf_corners": [[10, 10], [200, 10], [200, 200], [10, 200]],
        "boxes": [{"box_id": "3081", "video_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}],
        "source_info": {"shelf_code": "MAP_18"},
    }

    upgraded, changed, shelf_code, box_count = upgrade_annotation_payload(annotation)

    assert changed is True
    assert shelf_code == "MAP_18"
    assert box_count == 1
    assert "boxes" not in upgraded
    assert "source_info" not in upgraded
    assert upgraded["shelves"][0]["shelf_code"] == "MAP_18"
    assert upgraded["shelves"][0]["grid_shape"] == [4, 4]
    assert upgraded["shelves"][0]["boxes"][0]["box_id"] == "3081"
    assert build_box_index(upgraded, infer_w=640, infer_h=360)["MAP_18:3081"].shelf_code == "MAP_18"


def test_upgrade_record_annotation_writes_v2_file(tmp_path: Path):
    from analysis.annotation_upgrade import upgrade_record_annotation
    from analysis.records import load_record

    record_dir = _write_legacy_annotation_record(tmp_path / "record_026")

    stats = upgrade_record_annotation(record_dir)

    output_path = Path(stats.output_path)
    assert output_path.name == "annotation_v2.json"
    assert stats.upgraded is True
    assert stats.shelf_code == "MAP_18"
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert "source_info" not in data
    assert data["shelves"][0]["shelf_code"] == "MAP_18"
    assert data["shelves"][0]["boxes"][0]["box_id"] == "3081"
    assert (record_dir / "annotation.json").is_file()
    record = load_record(record_dir, validate_review_schema=False)
    assert "MAP_18:3081" in record.box_index


def test_upgrade_record_annotation_in_place_creates_backup(tmp_path: Path):
    from analysis.annotation_upgrade import upgrade_record_annotation

    record_dir = _write_legacy_annotation_record(tmp_path / "record_026")

    stats = upgrade_record_annotation(record_dir, in_place=True)

    assert Path(stats.output_path) == record_dir.resolve() / "annotation.json"
    assert (record_dir / "annotation.json.bak").is_file()
    data = json.loads((record_dir / "annotation.json").read_text(encoding="utf-8"))
    assert "shelves" in data
    assert "boxes" not in data


def test_upgrade_annotation_payload_keeps_existing_shelves():
    from analysis.annotation_upgrade import upgrade_annotation_payload

    annotation = {
        "annotation_size": {"width": 640, "height": 360},
        "shelves": [{"shelf_code": "S1", "boxes": [{"box_id": "A1"}]}],
    }

    upgraded, changed, shelf_code, box_count = upgrade_annotation_payload(annotation)

    assert changed is False
    assert upgraded == annotation
    assert shelf_code == "S1"
    assert box_count == 1
