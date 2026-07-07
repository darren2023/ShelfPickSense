"""标注坐标缩放与 infer 对齐测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fixtures import make_fixture_record


def test_infer_frame_size_uses_pose_aspect_ratio(tmp_path: Path):
    from analysis.records import load_record

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    record = load_record(fixture_dir)
    # fixture annotation 640x480 -> infer 640x480
    assert abs(record.infer_width - 640.0) < 0.01
    assert abs(record.infer_height - 480.0) < 0.01


def test_infer_frame_size_from_manifest(tmp_path: Path):
    from analysis.records import load_record

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    manifest = {"infer_width": 852, "infer_height": 480}
    (fixture_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    record = load_record(fixture_dir)
    assert record.infer_width == 852.0
    assert record.infer_height == 480.0


def test_manifest_must_not_use_nested_annotation_size_as_infer(tmp_path: Path):
    """manifest 嵌套 annotation.annotation_size 不得当作 infer（visual-dps 契约）。"""
    from analysis.features.spatial import point_in_polygon
    from analysis.records import load_record

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    # 骨架在 640×480；若误用 640×360 作为 infer，货框 norm 缩放会错位
    manifest = {
        "annotation": {
            "annotation_size": {"width": 640, "height": 360},
            "boxes": [],
        }
    }
    (fixture_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    record = load_record(fixture_dir)
    assert record.infer_width == pytest.approx(640.0)
    assert record.infer_height == pytest.approx(480.0)

    frame = next(f for f in record.frames() if f.frame_idx == 6)
    left_wrist = frame.persons[0]["keypoints"][9]
    poly = record.box_index["S1:A1"].polygon
    assert point_in_polygon(left_wrist[0], left_wrist[1], poly)


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


def test_boxes_already_in_infer_space_no_double_scale():
    """货框已在 infer 空间且 annotation_size 与 infer 一致时不得二次缩放。"""
    from analysis.annotation import build_box_index

    infer_w, infer_h = 852.0, 480.0
    polygon = [[100.0, 80.0], [200.0, 80.0], [200.0, 180.0], [100.0, 180.0]]
    annotation = {
        "annotation_size": {"width": infer_w, "height": infer_h},
        "shelves": [
            {
                "shelf_code": "S1",
                "boxes": [{"box_id": "A1", "video_polygon": polygon}],
            }
        ],
    }
    index = build_box_index(annotation, infer_w=infer_w, infer_h=infer_h)
    assert index["S1:A1"].polygon[0] == pytest.approx((100.0, 80.0))


def test_effective_annotation_size_legacy_override():
    """货框已在 infer 范围但 annotation_size 仍为更大原始画布时，覆盖为 infer。"""
    from analysis.annotation import effective_annotation_size

    annotation = {
        "annotation_size": {"width": 1920, "height": 1080},
        "shelves": [
            {
                "shelf_code": "S1",
                "boxes": [
                    {
                        "box_id": "A1",
                        "video_polygon": [[100, 80], [200, 80], [200, 180], [100, 180]],
                    }
                ],
            }
        ],
    }
    eff_w, eff_h = effective_annotation_size(annotation, infer_w=852, infer_h=480)
    assert eff_w == pytest.approx(852.0)
    assert eff_h == pytest.approx(480.0)


def test_is_norm_polygon_valid_rejects_out_of_range():
    from analysis.annotation import is_norm_polygon_valid

    assert is_norm_polygon_valid([(0.5, 0.5), (0.6, 0.5), (0.6, 0.6)])
    assert not is_norm_polygon_valid([(1.5, 0.5), (0.6, 0.5), (0.6, 0.6)])


def test_scale_polygon_respects_extent_when_exceeds_annotation_size():
    from analysis.annotation import scale_polygon_to_frame

    pts = [(700.0, 400.0), (800.0, 400.0), (800.0, 450.0)]
    scaled = scale_polygon_to_frame(
        pts,
        ann_w=640,
        ann_h=360,
        target_w=852,
        target_h=480,
    )
    max_x = max(p[0] for p in scaled)
    assert max_x <= 852.0 * 1.05


def test_align_infer_corrects_wrong_manifest_infer(tmp_path: Path):
    """manifest 显式 infer 错误且 skeleton 超出范围时，应修正为 pose 管线默认值。"""
    from analysis.records import load_record

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    manifest = {"infer_width": 320, "infer_height": 240}
    (fixture_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    record = load_record(fixture_dir)
    assert record.infer_width == pytest.approx(640.0)
    assert record.infer_height == pytest.approx(480.0)


def test_record_001_picking_frame_wrist_inside_box():
    data_dir = Path("data/data28/Train/record_001")
    if not data_dir.is_dir():
        return

    from analysis.features.spatial import point_in_polygon
    from analysis.records import load_record

    record = load_record(data_dir)
    assert record.infer_width == 852.0
    assert record.infer_height == 480.0

    frame = record.frame_index()[685].persons[0]
    left_wrist = frame["keypoints"][9]
    poly = record.box_index["81:1005"].polygon
    assert point_in_polygon(left_wrist[0], left_wrist[1], poly)


def test_640x360_annotation_resolves_to_852x480_infer(tmp_path: Path):
    """640×360 标注 + 852×480 骨架应对齐到 852×480 infer。"""
    from analysis.records import load_record, resolve_infer_frame_size

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    ann_path = fixture_dir / "annotation.json"
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    ann["annotation_size"] = {"width": 640, "height": 360}
    ann_path.write_text(json.dumps(ann, ensure_ascii=False), encoding="utf-8")

    skeleton = pd.read_parquet(fixture_dir / "skeleton.parquet")
    infer_w, infer_h = resolve_infer_frame_size(fixture_dir, skeleton, ann, record_id="record_001")
    assert infer_w == pytest.approx(852.0)
    assert infer_h == pytest.approx(480.0)

    record = load_record(fixture_dir)
    assert record.infer_width == pytest.approx(852.0)
    assert record.infer_height == pytest.approx(480.0)
