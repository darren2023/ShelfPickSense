"""货框数值布局（层/列/货架侧）测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.box_layout import (
    BoxNumericCode,
    build_box_layout,
    infer_shelf_side_from_x_coords,
    list_box_layout_rows,
    resolve_layout_token,
    token_to_numeric_code,
    tokens_to_numeric_codes,
)


def _box(
    *,
    box_id: str,
    shelf_code: str,
    layer: int,
    column: int,
    cx: float,
    cy: float,
) -> dict:
    """用中心点附近构造矩形 polygon。"""
    half = 8.0
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half
    return {
        "box_id": box_id,
        "shelf_code": shelf_code,
        "layer": layer,
        "column": column,
        "video_polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
    }


def _box_id_for_shelf(shelf_prefix: int, ann_layer: int, ann_col: int) -> str:
    return str(shelf_prefix * 1000 + (ann_layer - 1) * 4 + ann_col)


def _two_shelf_annotation() -> dict:
    """模拟双货架：左 82（x 小），右 81（x 大）；annotation layer 4 在画面下方。"""
    boxes_left: list[dict] = []
    boxes_right: list[dict] = []
    for ann_layer, cy in ((4, 280.0), (3, 220.0), (2, 160.0), (1, 100.0)):
        for ann_col, cx in ((1, 210.0), (2, 190.0), (3, 170.0), (4, 150.0)):
            boxes_left.append(
                _box(
                    box_id=_box_id_for_shelf(2, ann_layer, ann_col),
                    shelf_code="82",
                    layer=ann_layer,
                    column=ann_col,
                    cx=cx,
                    cy=cy,
                )
            )
        for ann_col, cx in ((1, 450.0), (2, 470.0), (3, 490.0), (4, 510.0)):
            boxes_right.append(
                _box(
                    box_id=_box_id_for_shelf(1, ann_layer, ann_col),
                    shelf_code="81",
                    layer=ann_layer,
                    column=ann_col,
                    cx=cx,
                    cy=cy,
                )
            )
    return {
        "annotation_size": {"width": 640, "height": 360},
        "shelves": [
            {"shelf_code": "82", "boxes": boxes_left},
            {"shelf_code": "81", "boxes": boxes_right},
        ],
    }


def test_infer_shelf_side_majority_on_left_half():
    assert infer_shelf_side_from_x_coords([100.0, 120.0, 200.0, 280.0], frame_width=640.0) == 1
    assert infer_shelf_side_from_x_coords([500.0, 520.0, 400.0, 380.0], frame_width=640.0) == 2


def test_infer_shelf_side_tie_breaks_by_mean():
    assert infer_shelf_side_from_x_coords([200.0, 440.0], frame_width=640.0) == 2
    assert infer_shelf_side_from_x_coords([200.0, 300.0], frame_width=640.0) == 1


def test_infer_shelf_side_uses_annotation_midline():
    # 分界线 = 640/2 = 320；3 个在左、1 个在右 -> 左货架
    assert infer_shelf_side_from_x_coords([100.0, 150.0, 310.0, 500.0], frame_width=640.0) == 1


def test_two_shelf_side_layer_column_rules():
    layout = build_box_layout(_two_shelf_annotation(), frame_width=640.0)

    left_bottom_near = layout["82:2013"]
    right_bottom_near = layout["81:1013"]

    assert left_bottom_near.shelf_side == 1
    assert right_bottom_near.shelf_side == 2

    assert left_bottom_near.layer == 1
    assert right_bottom_near.layer == 1

    assert left_bottom_near.column == 1
    assert right_bottom_near.column == 1

    assert layout["82:2016"].column == 4
    assert layout["81:1016"].column == 4


def test_single_shelf_left_vs_right_by_frame_center():
    boxes = [
        _box(box_id="1001", shelf_code="S1", layer=4, column=1, cx=150.0, cy=280.0),
        _box(box_id="1002", shelf_code="S1", layer=4, column=2, cx=120.0, cy=280.0),
    ]
    ann_left = {"annotation_size": {"width": 640, "height": 360}, "shelves": [{"shelf_code": "S1", "boxes": boxes}]}
    layout_left = build_box_layout(ann_left, frame_width=640.0)
    assert layout_left["S1:1001"].shelf_side == 1

    boxes_right = [
        _box(box_id="1001", shelf_code="S1", layer=4, column=1, cx=500.0, cy=280.0),
        _box(box_id="1002", shelf_code="S1", layer=4, column=2, cx=530.0, cy=280.0),
    ]
    ann_right = {"annotation_size": {"width": 640, "height": 360}, "shelves": [{"shelf_code": "S1", "boxes": boxes_right}]}
    layout_right = build_box_layout(ann_right, frame_width=640.0)
    assert layout_right["S1:1001"].shelf_side == 2


def test_resolve_event_review_box_token_without_shelf_prefix():
    ann = _two_shelf_annotation()
    layout = build_box_layout(ann)
    token = resolve_layout_token("Box_2013", layout)
    assert token == "82:2013"
    code = token_to_numeric_code("Box_2013", layout)
    assert code is not None
    assert code.encode() == 111


def test_numeric_encode_format():
    code = BoxNumericCode(shelf_side=2, layer=3, column=4, token="81:1012")
    assert code.encode() == 234
    assert code.as_tuple() == (2, 3, 4)


def test_tokens_to_numeric_codes_skips_unknown():
    ann = _two_shelf_annotation()
    layout = build_box_layout(ann)
    codes = tokens_to_numeric_codes(["Box_2013", "Box_missing"], layout)
    assert codes == [111]


@pytest.mark.skipif(
    not Path("data/data28/Train/record_001/annotation.json").is_file(),
    reason="需要本地 record_001 数据",
)
def test_record_001_layout_matches_geometry():
    ann = json.loads(Path("data/data28/Train/record_001/annotation.json").read_text(encoding="utf-8"))
    layout = build_box_layout(ann)
    left_tokens = sorted(token for token in layout if token.startswith("82:"))
    right_tokens = sorted(token for token in layout if token.startswith("81:"))
    assert left_tokens and right_tokens

    assert layout[left_tokens[0]].shelf_side == 1
    assert layout[right_tokens[0]].shelf_side == 2

    bottom_right = layout["81:1016"] if "81:1016" in layout else layout[right_tokens[-1]]
    assert bottom_right.layer == 1

    code = token_to_numeric_code("Box_1005", layout)
    assert code is not None
    assert code.token == "81:1005"
    assert code.shelf_side == 2
    assert code.layer == 3
    assert code.column == 1
    assert code.encode() == 231


def test_list_box_layout_rows_includes_centroid():
    ann = _two_shelf_annotation()
    rows = list_box_layout_rows(ann)
    assert len(rows) == 32
    bottom_left = next(row for row in rows if row.token == "82:2013")
    assert bottom_left.shelf_side == 1
    assert bottom_left.layer == 1
    assert bottom_left.column == 1
    assert bottom_left.box_code == 111
    assert bottom_left.centroid_x > 0
    assert bottom_left.centroid_y > 0
    assert bottom_left.to_dict()["token"] == "82:2013"


def test_load_record_enriches_box_codes(tmp_path: Path):
    from analysis.records import load_record
    from fixtures import make_fixture_record

    record_dir = make_fixture_record(tmp_path / "record_box_codes")
    review = json.loads((record_dir / "event_review.json").read_text(encoding="utf-8"))
    review["verified_true"][0]["confirmed_box_tokens"] = ["Box_A1"]
    (record_dir / "event_review.json").write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")

    record = load_record(record_dir)
    label = record.labels.frame_labels[6]
    assert label.confirmed_box_tokens == ["S1:A1"]
    assert label.confirmed_box_codes == [record.box_layout["S1:A1"].encode()]
