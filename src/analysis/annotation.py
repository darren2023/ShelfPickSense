"""annotation.json 解析与货框 token 工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 与 visual-dps-datacollect previewLayout / annotation_boxes 一致的量级容差
_POLYGON_INFER_MARGIN = 1.05
_LEGACY_ANN_SIZE_RATIO = 1.15


def flatten_annotation_boxes(config_data: dict[str, Any]) -> list[dict[str, Any]]:
    """解析 shelves[] 或 legacy 顶层 boxes[]。"""
    if not isinstance(config_data, dict):
        return []

    raw_boxes: list[dict[str, Any]] = []
    shelves = config_data.get("shelves")
    if isinstance(shelves, list):
        for shelf in shelves:
            if not isinstance(shelf, dict):
                continue
            shelf_code = str(shelf.get("shelf_code", "") or "").strip()
            boxes = shelf.get("boxes", [])
            if not isinstance(boxes, list):
                continue
            for box in boxes:
                if not isinstance(box, dict):
                    continue
                item = dict(box)
                if shelf_code and not item.get("shelf_code"):
                    item["shelf_code"] = shelf_code
                raw_boxes.append(item)

    if raw_boxes:
        return raw_boxes

    boxes = config_data.get("boxes", [])
    return boxes if isinstance(boxes, list) else []


def box_collision_token(box: dict[str, Any]) -> str:
    """货位唯一标识，与 visual-dps event_engine/box_identity 一致。"""
    if not isinstance(box, dict):
        return ""
    shelf = str(box.get("shelf_code", "") or "").strip()
    box_id = str(box.get("box_id", "") or box.get("id", "") or "").strip()
    if not box_id:
        return ""
    if shelf:
        return f"{shelf}:{box_id}"
    return f"Box_{box_id}"


def annotation_size(annotation: dict[str, Any]) -> tuple[float | None, float | None]:
    ann_size = annotation.get("annotation_size") if isinstance(annotation.get("annotation_size"), dict) else {}
    ann_w = float(ann_size.get("width") or 0) or None
    ann_h = float(ann_size.get("height") or 0) or None
    return ann_w, ann_h


def _parse_polygon_list(raw: Any) -> list[tuple[float, float]]:
    if not isinstance(raw, list) or len(raw) < 3:
        return []
    pts: list[tuple[float, float]] = []
    for pt in raw:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            pts.append((float(pt[0]), float(pt[1])))
    return pts


def polygon_points(box: dict[str, Any]) -> list[tuple[float, float]]:
    """从货框读取 video_polygon 顶点。"""
    return _parse_polygon_list(box.get("video_polygon"))


def normalized_polygon_points(box: dict[str, Any]) -> list[tuple[float, float]]:
    """从货框读取 video_polygon_norm 顶点（相对 annotation_size，0~1）。"""
    return _parse_polygon_list(box.get("video_polygon_norm"))


def polygon_max_extent(pts: list[tuple[float, float]]) -> tuple[float, float]:
    if not pts:
        return 0.0, 0.0
    return max(p[0] for p in pts), max(p[1] for p in pts)


def is_norm_polygon_valid(norm_pts: list[tuple[float, float]]) -> bool:
    """校验 norm 多边形是否在 [0, 1] 范围内（与 visual-dps isNormPolygonValid 一致）。"""
    if len(norm_pts) < 3:
        return False
    for x, y in norm_pts:
        if x < -0.01 or x > 1.01 or y < -0.01 or y > 1.01:
            return False
    return True


def _boxes_polygon_max_extent(boxes: list[dict[str, Any]]) -> tuple[float, float]:
    max_x = 0.0
    max_y = 0.0
    for box in boxes:
        mx, my = polygon_max_extent(polygon_points(box))
        max_x = max(max_x, mx)
        max_y = max(max_y, my)
    return max_x, max_y


def effective_annotation_size(
    annotation: dict[str, Any],
    *,
    infer_w: float,
    infer_h: float,
) -> tuple[float, float]:
    """
    有效标注尺寸：用于货框缩放。

    旧版 pose 内嵌标注中，video_polygon 已在 infer 空间，但 annotation_size 仍为原始标注尺寸；
    此时覆盖为 infer 尺寸，避免二次缩放（与 visual-dps getEffectiveAnnotationSize 一致）。
    """
    ann_w, ann_h = annotation_size(annotation)
    if not ann_w or not ann_h:
        return float(infer_w), float(infer_h)

    boxes = flatten_annotation_boxes(annotation)
    max_x, max_y = _boxes_polygon_max_extent(boxes)
    if (
        max_x <= infer_w * _POLYGON_INFER_MARGIN
        and max_y <= infer_h * _POLYGON_INFER_MARGIN
        and (ann_w > infer_w * _LEGACY_ANN_SIZE_RATIO or ann_h > infer_h * _LEGACY_ANN_SIZE_RATIO)
    ):
        return float(infer_w), float(infer_h)
    return float(ann_w), float(ann_h)


def boxes_already_in_infer_space(
    annotation: dict[str, Any],
    *,
    infer_w: float,
    infer_h: float,
) -> bool:
    """
    货框 video_polygon 是否已在 infer 坐标系内（与 visual-dps getBoxesInInferSpace 一致）。

    条件：有效 annotation_size 与 infer 一致，且所有 polygon 顶点在 infer 范围内。
    """
    boxes = flatten_annotation_boxes(annotation)
    if not boxes:
        return False

    eff_w, eff_h = effective_annotation_size(annotation, infer_w=infer_w, infer_h=infer_h)
    if abs(eff_w - infer_w) > 1.0 or abs(eff_h - infer_h) > 1.0:
        return False

    for box in boxes:
        mx, my = polygon_max_extent(polygon_points(box))
        if mx > infer_w * _POLYGON_INFER_MARGIN or my > infer_h * _POLYGON_INFER_MARGIN:
            return False
    return True


def scale_polygon_to_frame(
    pts: list[tuple[float, float]],
    *,
    ann_w: float | None,
    ann_h: float | None,
    target_w: float,
    target_h: float,
) -> list[tuple[float, float]]:
    """将标注坐标缩放到 infer/骨骼坐标系（与 visual-dps _scale_polygon_to_frame 对齐）。"""
    if not pts:
        return []
    tw, th = float(target_w), float(target_h)
    max_x, max_y = polygon_max_extent(pts)
    if ann_w and ann_h and ann_w > 0 and ann_h > 0:
        sx = tw / float(ann_w) if max_x <= float(ann_w) * _POLYGON_INFER_MARGIN else tw / max_x
        sy = th / float(ann_h) if max_y <= float(ann_h) * _POLYGON_INFER_MARGIN else th / max_y
    elif max_x > 0 and max_y > 0:
        sx, sy = tw / max_x, th / max_y
    else:
        sx = sy = 1.0
    return [(x * sx, y * sy) for x, y in pts]


def infer_polygon_points(
    box: dict[str, Any],
    *,
    infer_w: float,
    infer_h: float,
    ann_w: float | None,
    ann_h: float | None,
    boxes_already_infer: bool = False,
) -> list[tuple[float, float]]:
    """将货框多边形变换到 infer 坐标系，与 skeleton 关键点同一空间。"""
    raw_pts = polygon_points(box)
    if boxes_already_infer and raw_pts:
        return list(raw_pts)

    norm_pts = normalized_polygon_points(box)
    if is_norm_polygon_valid(norm_pts):
        return [(x * infer_w, y * infer_h) for x, y in norm_pts]

    return scale_polygon_to_frame(raw_pts, ann_w=ann_w, ann_h=ann_h, target_w=infer_w, target_h=infer_h)


@dataclass(frozen=True)
class BoxInfo:
    token: str
    shelf_code: str
    box_id: str
    polygon: tuple[tuple[float, float], ...]


def load_annotation(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"annotation.json 根节点必须是 object: {path}")
    return data


def build_box_index(
    annotation: dict[str, Any],
    *,
    infer_w: float,
    infer_h: float,
) -> dict[str, BoxInfo]:
    """构建 token -> BoxInfo 索引（货框 polygon 与 skeleton 同在 infer 空间）。"""
    eff_w, eff_h = effective_annotation_size(annotation, infer_w=infer_w, infer_h=infer_h)
    already_infer = boxes_already_in_infer_space(annotation, infer_w=infer_w, infer_h=infer_h)

    index: dict[str, BoxInfo] = {}
    for raw in flatten_annotation_boxes(annotation):
        token = box_collision_token(raw)
        if not token:
            continue
        scaled = infer_polygon_points(
            raw,
            infer_w=infer_w,
            infer_h=infer_h,
            ann_w=eff_w,
            ann_h=eff_h,
            boxes_already_infer=already_infer,
        )
        if len(scaled) < 3:
            continue
        shelf = str(raw.get("shelf_code", "") or "").strip()
        box_id = str(raw.get("box_id", "") or raw.get("id", "") or "").strip()
        index[token] = BoxInfo(token=token, shelf_code=shelf, box_id=box_id, polygon=tuple(scaled))
    return index
