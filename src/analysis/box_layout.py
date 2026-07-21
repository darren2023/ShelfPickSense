"""从 annotation 推导货框数值编码（货架侧 / 层 / 列），用于建模替代字符串 token。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

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
    """货框布局一行记录，含计算层列与标注坐标系下的中心点。"""

    token: str
    shelf_code: str
    box_id: str
    shelf_side: int
    layer: int
    column: int
    centroid_x: float
    centroid_y: float
    annotation_layer: int | None = None
    annotation_column: int | None = None
    polygon: tuple[tuple[float, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "shelf_code": self.shelf_code,
            "box_id": self.box_id,
            "shelf_side": self.shelf_side,
            "layer": self.layer,
            "column": self.column,
            "centroid_x": round(self.centroid_x, 2),
            "centroid_y": round(self.centroid_y, 2),
            "annotation_layer": self.annotation_layer,
            "annotation_column": self.annotation_column,
            "polygon": [[round(x, 2), round(y, 2)] for x, y in self.polygon],
        }


@dataclass(frozen=True)
class ShelfBottomBounds:
    """当前货架贴地边界的两个端点。"""

    p1_x: float
    p1_y: float
    p2_x: float
    p2_y: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "p1": [round(self.p1_x, 2), round(self.p1_y, 2)],
            "p2": [round(self.p2_x, 2), round(self.p2_y, 2)],
        }


@dataclass(frozen=True)
class ShelfLayoutStats:
    """单个货架的布局统计（基于几何推导的层/列，非 annotation 字段）。"""

    shelf_code: str
    layer_count: int
    column_count: int
    column_count_mean: float


def normalize_layout_layer(layer: int, layer_count: int) -> float:
    """层号归一化到对应层区间中心：(layer - 0.5) / layer_count。"""
    if layer <= 0 or layer_count <= 0:
        return 0.0
    return (float(layer) - 0.5) / float(layer_count)


def normalize_layout_column(column: int, column_count: int) -> float:
    """列号归一化到对应列区间中心：(column - 0.5) / column_count。"""
    if column <= 0 or column_count <= 0:
        return 0.0
    return (float(column) - 0.5) / float(column_count)


def denormalize_layout_layer(layer_norm: float, layer_count: int) -> int:
    if layer_count <= 0 or layer_norm <= 0:
        return 0
    layer = int(float(layer_norm) * layer_count) + 1
    return max(1, min(layer_count, layer))


def denormalize_layout_column(column_norm: float, column_count: int) -> int:
    if column_count <= 0 or column_norm <= 0:
        return 0
    column = int(float(column_norm) * column_count) + 1
    return max(1, min(column_count, column))


def normalized_layout_targets(
    entry: BoxNumericCode,
    stats: ShelfLayoutStats | None,
) -> tuple[float, float]:
    """由货框布局码与货架统计得到归一化监督 (layer_norm, column_norm)。"""
    if stats is None:
        return 0.0, 0.0
    return (
        normalize_layout_layer(entry.layer, stats.layer_count),
        normalize_layout_column(entry.column, stats.column_count),
    )


def record_layout_denorm_bounds(layout: dict[str, BoxNumericCode]) -> tuple[int, int]:
    """记录级层/列上界，用于推理时将归一化预测还原为整数层列。"""
    if not layout:
        return 0, 0
    return max(code.layer for code in layout.values()), max(code.column for code in layout.values())


def compute_shelf_layout_stats(layout: dict[str, BoxNumericCode]) -> dict[str, ShelfLayoutStats]:
    """按货架统计层数、列数（最大值）及各层列数均值。"""
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
        column_count = max(per_layer_counts) if per_layer_counts else 0
        column_count_mean = sum(per_layer_counts) / len(per_layer_counts) if per_layer_counts else 0.0
        stats[shelf_code] = ShelfLayoutStats(
            shelf_code=shelf_code,
            layer_count=layer_count,
            column_count=column_count,
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
    polygon: tuple[tuple[float, float], ...] = ()
    shelf_side: int = 0
    layer: int = 0
    column: int = 0


def _polygon_points_for_layout(box: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    pts = box.get("video_polygon") or box.get("video_polygon_norm")
    if not isinstance(pts, list) or len(pts) < 3:
        return ()
    parsed: list[tuple[float, float]] = []
    for pt in pts:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            parsed.append((float(pt[0]), float(pt[1])))
    return tuple(parsed)


def _polygon_centroid(box: dict[str, Any]) -> tuple[float, float]:
    pts = _polygon_points_for_layout(box)
    if not pts:
        return 0.0, 0.0
    xs: list[float] = []
    ys: list[float] = []
    for x, y in pts:
        xs.append(x)
        ys.append(y)
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
            polygon=_polygon_points_for_layout(raw),
        )
        shelf_groups.setdefault(shelf_code or "_default", []).append(geom)

    _assign_shelf_sides(shelf_groups, frame_width=float(frame_width))

    geoms: list[_BoxGeom] = []
    for boxes in shelf_groups.values():
        _assign_layers(boxes)
        _assign_columns(boxes)
        geoms.extend(boxes)
    return float(frame_width), geoms


def compute_shelf_bottom_bounds_from_rows(rows: list[BoxLayoutRow]) -> dict[str, ShelfBottomBounds]:
    """按货架计算贴地边界端点。

    P1/coord1：右货架取 x 坐标最小的点，左货架取 x 坐标最大的点。Y坐标较大的点。
    P2/coord2：该货架所有货框顶点中 y 坐标最靠下的点。
    """
    by_shelf: dict[str, list[BoxLayoutRow]] = defaultdict(list)
    for row in rows:
        by_shelf[row.shelf_code or "_default"].append(row)

    result: dict[str, ShelfBottomBounds] = {}
    for shelf_code, shelf_rows in by_shelf.items():
        bottom_layer = min((row.layer for row in shelf_rows if row.layer > 0), default=0)
        points = [pt for row in shelf_rows for pt in row.polygon]
        if not points:
            points = [(row.centroid_x, row.centroid_y) for row in shelf_rows]
        if not points:
            continue
        side = next((row.shelf_side for row in shelf_rows), 0)
        coord1 = max(points, key=lambda pt: (pt[1], -pt[0]))
        if side == 2:
            coord2 = min(points, key=lambda pt: (pt[0], -pt[1]))
        else:
            coord2 = max(points, key=lambda pt: (pt[0], pt[1]))
        result[shelf_code] = ShelfBottomBounds(
            p1_x=coord2[0],
            p1_y=coord2[1],
            p2_x=coord1[0],
            p2_y=coord1[1],
        )
    return result


def compute_shelf_bottom_bounds(
    annotation: dict[str, Any],
    *,
    frame_width: float | None = None,
) -> dict[str, ShelfBottomBounds]:
    """从 annotation 计算各货架底层货框坐标范围。"""
    return compute_shelf_bottom_bounds_from_rows(list_box_layout_rows(annotation, frame_width=frame_width))


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
            centroid_x=box.cx,
            centroid_y=box.cy,
            annotation_layer=box.raw_layer,
            annotation_column=box.raw_column,
            polygon=box.polygon,
        )
        for box in geoms
    ]
    if sort:
        rows.sort(key=lambda row: (row.shelf_side, row.layer, row.column, row.shelf_code, row.box_id))
    return rows


def render_box_layout_svg(
    annotation: dict[str, Any],
    *,
    output_path: Path,
    frame_width: float | None = None,
) -> None:
    """将 box_layout 计算结果渲染为 SVG，便于检查层列、编码和底层范围。"""
    ann_size = annotation.get("annotation_size") if isinstance(annotation.get("annotation_size"), dict) else {}
    width = float(frame_width or ann_size.get("width") or 640.0)
    height = float(ann_size.get("height") or 360.0)
    rows = list_box_layout_rows(annotation, frame_width=frame_width)
    bounds = compute_shelf_bottom_bounds_from_rows(rows)
    canvas_w = max(width, 1.0)
    canvas_h = max(height, 1.0)
    colors = {1: "#2563eb", 2: "#dc2626"}

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {canvas_w:.2f} {canvas_h:.2f}" width="{canvas_w:.0f}" height="{canvas_h:.0f}">',
        "<style>",
        "text{font-family:Arial,sans-serif;font-size:8px;dominant-baseline:middle}",
        ".small{font-size:6px}",
        ".range{fill:none;stroke:#16a34a;stroke-width:2;stroke-dasharray:5 3}",
        "</style>",
        f'<rect x="0" y="0" width="{canvas_w:.2f}" height="{canvas_h:.2f}" fill="#ffffff"/>',
        f'<line x1="{canvas_w / 2:.2f}" y1="0" x2="{canvas_w / 2:.2f}" y2="{canvas_h:.2f}" '
        'stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 4"/>',
    ]

    for row in rows:
        if not row.polygon:
            continue
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in row.polygon)
        color = colors.get(row.shelf_side, "#64748b")
        label = f"({row.layer},{row.column})"
        parts.extend(
            [
                f'<polygon points="{pts}" fill="{color}" fill-opacity="0.16" stroke="{color}" stroke-width="1"/>',
                f'<circle cx="{row.centroid_x:.2f}" cy="{row.centroid_y:.2f}" r="2" fill="{color}"/>',
                f'<text x="{row.centroid_x + 3:.2f}" y="{row.centroid_y - 3:.2f}" fill="#111827">'
                f"{escape(label)}</text>",
                f'<text class="small" x="{row.centroid_x + 3:.2f}" y="{row.centroid_y + 6:.2f}" fill="#475569">'
                f"side={row.shelf_side}</text>",
            ]
        )

    for shelf_code, bound in bounds.items():
        x1, y1 = bound.p1_x, bound.p1_y
        x2, y2 = bound.p2_x, bound.p2_y
        label_y = max(8.0, min(y1, y2) - 8.0)
        parts.extend(
            [
                f'<line class="range" x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"/>',
                f'<circle cx="{x1:.2f}" cy="{y1:.2f}" r="3" fill="#16a34a"/>',
                f'<circle cx="{x2:.2f}" cy="{y2:.2f}" r="3" fill="#16a34a"/>',
                f'<text x="{x1 + 5:.2f}" y="{y1 - 5:.2f}" fill="#166534">P1</text>',
                f'<text x="{x2 + 5:.2f}" y="{y2 - 5:.2f}" fill="#166534">P2</text>',
                f'<text x="{min(x1, x2):.2f}" y="{label_y:.2f}" fill="#166534">'
                f"{escape(shelf_code or '_default')} bottom: P1=({x1:.1f},{y1:.1f}) "
                f"P2=({x2:.1f},{y2:.1f})</text>",
            ]
        )

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def load_box_layout_rows(annotation_path: Path) -> list[BoxLayoutRow]:
    """从 annotation.json 路径加载并计算货框布局。"""
    from analysis.annotation import load_annotation

    return list_box_layout_rows(load_annotation(annotation_path))


def resolve_box_tokens_by_layout(
    layout: dict[str, BoxNumericCode],
    *,
    layer: int,
    column: int,
    shelf_side: int | None = None,
) -> list[str]:
    """按几何推导的层/列（及可选货架侧）匹配货框 token。"""
    tokens: list[str] = []
    for token, code in layout.items():
        if int(code.layer) != int(layer) or int(code.column) != int(column):
            continue
        if shelf_side is not None and int(code.shelf_side) != int(shelf_side):
            continue
        tokens.append(token)
    return sorted(tokens)


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
