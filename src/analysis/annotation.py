"""annotation.json 解析与货框 token 工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def scale_polygon_to_frame(
    pts: list[tuple[float, float]],
    *,
    ann_w: float | None,
    ann_h: float | None,
    target_w: float,
    target_h: float,
) -> list[tuple[float, float]]:
    """将标注坐标缩放到推理/骨骼坐标系（保持宽高比一致）。"""
    if not pts:
        return []
    tw, th = float(target_w), float(target_h)
    if ann_w and ann_h and ann_w > 0 and ann_h > 0:
        sx = tw / float(ann_w)
        sy = th / float(ann_h)
    else:
        max_x = max(p[0] for p in pts)
        max_y = max(p[1] for p in pts)
        if max_x > 0 and max_y > 0:
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
) -> list[tuple[float, float]]:
    """将货框多边形变换到推理坐标系。优先使用 video_polygon_norm。"""
    norm_pts = normalized_polygon_points(box)
    if norm_pts:
        return [(x * infer_w, y * infer_h) for x, y in norm_pts]

    raw_pts = polygon_points(box)
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
    """构建 token -> BoxInfo 索引。"""
    ann_w, ann_h = annotation_size(annotation)

    index: dict[str, BoxInfo] = {}
    for raw in flatten_annotation_boxes(annotation):
        token = box_collision_token(raw)
        if not token:
            continue
        scaled = infer_polygon_points(raw, infer_w=infer_w, infer_h=infer_h, ann_w=ann_w, ann_h=ann_h)
        if len(scaled) < 3:
            continue
        shelf = str(raw.get("shelf_code", "") or "").strip()
        box_id = str(raw.get("box_id", "") or raw.get("id", "") or "").strip()
        index[token] = BoxInfo(token=token, shelf_code=shelf, box_id=box_id, polygon=tuple(scaled))
    return index
