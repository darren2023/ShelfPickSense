"""从 annotation 推导货框数值编码（货架侧 / 层 / 列），用于建模替代字符串 token。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.annotation import box_collision_token, flatten_annotation_boxes


@dataclass(frozen=True)
class BoxNumericCode:
    """货框数值编码：货架侧 + 层（自下而上 1 起）+ 列（靠摄像头 1 起）。"""

    shelf_side: int
    layer: int
    column: int
    shelf_code: str = ""
    box_id: str = ""
    token: str = ""

    def encode(self) -> int:
        """压缩为单一整数：side*100 + layer*10 + column。"""
        return int(self.shelf_side) * 100 + int(self.layer) * 10 + int(self.column)

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.shelf_side, self.layer, self.column)


@dataclass(frozen=True)
class BoxLayoutRow:
    """货框布局一行记录，含数值编码与标注坐标系下的中心点。"""

    token: str
    shelf_code: str
    box_id: str
    shelf_side: int
    layer: int
    column: int
    box_code: int
    centroid_x: float
    centroid_y: float
    annotation_layer: int | None = None
    annotation_column: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "shelf_code": self.shelf_code,
            "box_id": self.box_id,
            "shelf_side": self.shelf_side,
            "layer": self.layer,
            "column": self.column,
            "box_code": self.box_code,
            "centroid_x": round(self.centroid_x, 2),
            "centroid_y": round(self.centroid_y, 2),
            "annotation_layer": self.annotation_layer,
            "annotation_column": self.annotation_column,
        }


@dataclass(frozen=True)
class ShelfLayoutStats:
    """单个货架的布局统计（基于几何推导的层/列，非 annotation 字段）。"""

    shelf_code: str
    layer_count: int
    column_count_mean: float


def compute_shelf_layout_stats(layout: dict[str, BoxNumericCode]) -> dict[str, ShelfLayoutStats]:
    """按货架统计层数，以及各层列数的平均值。"""
    by_shelf: dict[str, list[BoxNumericCode]] = defaultdict(list)
    for entry in layout.values():
        key = entry.shelf_code or "_default"
        by_shelf[key].append(entry)

    stats: dict[str, ShelfLayoutStats] = {}
    for shelf_code, boxes in by_shelf.items():
        if not boxes:
            continue
        layer_count = max(box.layer for box in boxes)
        columns_by_layer: dict[int, list[int]] = defaultdict(list)
        for box in boxes:
            columns_by_layer[box.layer].append(box.column)
        per_layer_counts = [max(cols) for cols in columns_by_layer.values() if cols]
        column_count_mean = sum(per_layer_counts) / len(per_layer_counts) if per_layer_counts else 0.0
        stats[shelf_code] = ShelfLayoutStats(
            shelf_code=shelf_code,
            layer_count=layer_count,
            column_count_mean=column_count_mean,
        )
    return stats


@dataclass
class _BoxGeom:
    token: str
    shelf_code: str
    box_id: str
    cx: float
    cy: float
    raw_layer: int | None
    raw_column: int | None
    shelf_side: int = 0
    layer: int = 0
    column: int = 0


def _polygon_centroid(box: dict[str, Any]) -> tuple[float, float]:
    pts = box.get("video_polygon") or box.get("video_polygon_norm")
    if not isinstance(pts, list) or len(pts) < 3:
        return 0.0, 0.0
    xs: list[float] = []
    ys: list[float] = []
    for pt in pts:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            xs.append(float(pt[0]))
            ys.append(float(pt[1]))
    if not xs:
        return 0.0, 0.0
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _parse_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_shelf_side_from_x_coords(xs: list[float], *, frame_width: float) -> int:
    """
    根据货框 x 坐标多数落在画面左/右半区判定货架侧。

    分界线为 frame_width / 2；左半区多数 -> 1（左货架），右半区多数 -> 2（右货架）。
    数量持平时用均值相对分界线判定。
    """
    if not xs:
        return 1
    midline = float(frame_width) / 2.0
    left_count = sum(1 for x in xs if x < midline)
    right_count = len(xs) - left_count
    if left_count > right_count:
        return 1
    if right_count > left_count:
        return 2
    mean_x = sum(xs) / len(xs)
    return 1 if mean_x < midline else 2


def _assign_shelf_sides(shelf_groups: dict[str, list[_BoxGeom]], *, frame_width: float) -> None:
    """每个货架：其货框中心点多数落在左/右半区则分别为左/右货架。"""
    for boxes in shelf_groups.values():
        if not boxes:
            continue
        side = infer_shelf_side_from_x_coords([box.cx for box in boxes], frame_width=frame_width)
        for box in boxes:
            box.shelf_side = side


def _assign_layers(boxes: list[_BoxGeom]) -> None:
    """层：画面 y 越大（越靠下）层号越小，最底下为 1 层。"""
    if not boxes:
        return

    groups: dict[int, list[_BoxGeom]] = {}
    for box in boxes:
        key = box.raw_layer if box.raw_layer is not None else 0
        groups.setdefault(key, []).append(box)

    layer_stats: list[tuple[float, list[_BoxGeom]]] = []
    for group_boxes in groups.values():
        mean_cy = sum(b.cy for b in group_boxes) / len(group_boxes)
        layer_stats.append((mean_cy, group_boxes))

    layer_stats.sort(key=lambda item: item[0], reverse=True)
    for layer_idx, (_, group_boxes) in enumerate(layer_stats, start=1):
        for box in group_boxes:
            box.layer = layer_idx


def _assign_columns(boxes: list[_BoxGeom]) -> None:
    """列：同一层内，靠近画面中心/巷道为 1 列。"""
    by_layer: dict[int, list[_BoxGeom]] = {}
    for box in boxes:
        by_layer.setdefault(box.layer, []).append(box)

    for layer_boxes in by_layer.values():
        if not layer_boxes:
            continue
        shelf_side = layer_boxes[0].shelf_side
        # 左货架：x 越大越靠近巷道；右货架：x 越小越靠近巷道
        if shelf_side == 1:
            ordered = sorted(layer_boxes, key=lambda b: b.cx, reverse=True)
        else:
            ordered = sorted(layer_boxes, key=lambda b: b.cx)
        for col_idx, box in enumerate(ordered, start=1):
            box.column = col_idx


def _build_layout_geoms(
    annotation: dict[str, Any],
    *,
    frame_width: float | None = None,
) -> tuple[float, list[_BoxGeom]]:
    ann_size = annotation.get("annotation_size")
    if frame_width is None:
        if isinstance(ann_size, dict) and ann_size.get("width"):
            frame_width = float(ann_size["width"])
        else:
            frame_width = 640.0

    shelf_groups: dict[str, list[_BoxGeom]] = {}
    for raw in flatten_annotation_boxes(annotation):
        token = box_collision_token(raw)
        if not token:
            continue
        cx, cy = _polygon_centroid(raw)
        shelf_code = str(raw.get("shelf_code", "") or "").strip()
        box_id = str(raw.get("box_id", "") or raw.get("id", "") or "").strip()
        geom = _BoxGeom(
            token=token,
            shelf_code=shelf_code,
            box_id=box_id,
            cx=cx,
            cy=cy,
            raw_layer=_parse_int(raw.get("layer")),
            raw_column=_parse_int(raw.get("column")),
        )
        shelf_groups.setdefault(shelf_code or "_default", []).append(geom)

    _assign_shelf_sides(shelf_groups, frame_width=float(frame_width))

    geoms: list[_BoxGeom] = []
    for boxes in shelf_groups.values():
        _assign_layers(boxes)
        _assign_columns(boxes)
        geoms.extend(boxes)
    return float(frame_width), geoms


def build_box_layout(
    annotation: dict[str, Any],
    *,
    frame_width: float | None = None,
) -> dict[str, BoxNumericCode]:
    """
    从 annotation 构建 token -> BoxNumericCode。

    frame_width: 用于货架左右判定（分界线为 width/2），默认取 annotation_size.width 或 640。
    """
    _frame_width, geoms = _build_layout_geoms(annotation, frame_width=frame_width)
    layout: dict[str, BoxNumericCode] = {}
    for box in geoms:
        layout[box.token] = BoxNumericCode(
            shelf_side=box.shelf_side,
            layer=box.layer,
            column=box.column,
            shelf_code=box.shelf_code,
            box_id=box.box_id,
            token=box.token,
        )
    return layout


def list_box_layout_rows(
    annotation: dict[str, Any],
    *,
    frame_width: float | None = None,
    sort: bool = True,
) -> list[BoxLayoutRow]:
    """读取 annotation 并返回各货框的布局与中心点列表。"""
    _frame_width, geoms = _build_layout_geoms(annotation, frame_width=frame_width)
    rows = [
        BoxLayoutRow(
            token=box.token,
            shelf_code=box.shelf_code,
            box_id=box.box_id,
            shelf_side=box.shelf_side,
            layer=box.layer,
            column=box.column,
            box_code=BoxNumericCode(
                shelf_side=box.shelf_side,
                layer=box.layer,
                column=box.column,
            ).encode(),
            centroid_x=box.cx,
            centroid_y=box.cy,
            annotation_layer=box.raw_layer,
            annotation_column=box.raw_column,
        )
        for box in geoms
    ]
    if sort:
        rows.sort(key=lambda row: (row.shelf_side, row.layer, row.column, row.shelf_code, row.box_id))
    return rows


def load_box_layout_rows(annotation_path: Path) -> list[BoxLayoutRow]:
    """从 annotation.json 路径加载并计算货框布局。"""
    from analysis.annotation import load_annotation

    return list_box_layout_rows(load_annotation(annotation_path))


def normalize_box_token(token: str) -> str:
    return str(token or "").strip()


def resolve_layout_token(token: str, layout: dict[str, BoxNumericCode]) -> str:
    """将 event_review 中的 Box_<id> 等形式解析为 layout 中的 canonical token。"""
    token = normalize_box_token(token)
    if not token:
        return ""
    if token in layout:
        return token

    box_id = ""
    if token.startswith("Box_"):
        box_id = token[4:]
    elif ":" in token:
        return token if token in layout else token

    if box_id:
        matches = [t for t, code in layout.items() if code.box_id == box_id]
        if len(matches) == 1:
            return matches[0]

    if not token.startswith("Box_") and ":" not in token:
        matches = [t for t, code in layout.items() if code.box_id == token]
        if len(matches) == 1:
            return matches[0]

    return token


def token_to_numeric_code(token: str, layout: dict[str, BoxNumericCode]) -> BoxNumericCode | None:
    canonical = resolve_layout_token(token, layout)
    return layout.get(canonical)


def tokens_to_numeric_codes(tokens: list[str], layout: dict[str, BoxNumericCode]) -> list[int]:
    codes: list[int] = []
    for token in tokens:
        code = token_to_numeric_code(token, layout)
        if code is not None:
            codes.append(code.encode())
    return codes


def numeric_code_to_token(code: int, layout: dict[str, BoxNumericCode]) -> str | None:
    for token, item in layout.items():
        if item.encode() == int(code):
            return token
    return None
