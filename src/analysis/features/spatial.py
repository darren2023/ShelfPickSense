"""人体与货框空间关系特征。"""

from __future__ import annotations

import math

from analysis.constants import LEFT_WRIST_IDX, RIGHT_WRIST_IDX
from analysis.features.base import FeatureContext, FeatureExtractor


def point_in_polygon(x: float, y: float, polygon: tuple[tuple[float, float], ...]) -> bool:
    """射线法判断点是否在多边形内。"""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def point_to_segment_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def point_to_polygon_dist(x: float, y: float, polygon: tuple[tuple[float, float], ...]) -> float:
    if point_in_polygon(x, y, polygon):
        return 0.0
    n = len(polygon)
    best = float("inf")
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        best = min(best, point_to_segment_dist(x, y, ax, ay, bx, by))
    return best


def _wrist_points(person: dict) -> list[tuple[float, float, float]]:
    kpts = person.get("keypoints") or []
    pts: list[tuple[float, float, float]] = []
    for idx in (LEFT_WRIST_IDX, RIGHT_WRIST_IDX):
        if idx >= len(kpts):
            continue
        kp = kpts[idx]
        if not isinstance(kp, (list, tuple)) or len(kp) < 3:
            continue
        if float(kp[2]) <= 0.3:
            continue
        pts.append((float(kp[0]), float(kp[1]), float(kp[2])))
    return pts


def _box_centroid(polygon: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return sum(xs) / len(xs), sum(ys) / len(ys)


class BoxSpatialFeatureExtractor(FeatureExtractor):
    name = "spatial"

    def extract_frame(self, ctx: FeatureContext) -> dict[str, float]:
        per_box = self.extract_per_box(ctx)
        if not per_box:
            return {
                "min_wrist_box_dist_norm": 1.0,
                "any_wrist_inside_box": 0.0,
                "boxes_with_wrist_inside": 0.0,
            }
        dists = [feats["wrist_min_dist_norm"] for feats in per_box.values()]
        inside_count = sum(1 for feats in per_box.values() if feats.get("wrist_inside", 0.0) > 0.5)
        return {
            "min_wrist_box_dist_norm": min(dists) if dists else 1.0,
            "any_wrist_inside_box": 1.0 if inside_count > 0 else 0.0,
            "boxes_with_wrist_inside": float(inside_count),
        }

    def extract_per_box(self, ctx: FeatureContext) -> dict[str, dict[str, float]]:
        scale = max(ctx.record.infer_width, ctx.record.infer_height, 1.0)
        wrists: list[tuple[float, float]] = []
        for person in ctx.frame.persons:
            for x, y, _ in _wrist_points(person):
                wrists.append((x, y))

        out: dict[str, dict[str, float]] = {}
        for token, box in ctx.box_index.items():
            if not wrists:
                out[token] = {
                    "wrist_min_dist_norm": 1.0,
                    "wrist_inside": 0.0,
                    "wrist_mean_dist_norm": 1.0,
                    "centroid_dist_norm": 1.0,
                }
                continue
            dists = [point_to_polygon_dist(x, y, box.polygon) for x, y in wrists]
            inside = any(point_in_polygon(x, y, box.polygon) for x, y in wrists)
            cx, cy = _box_centroid(box.polygon)
            centroid_dists = [math.hypot(x - cx, y - cy) for x, y in wrists]
            out[token] = {
                "wrist_min_dist_norm": min(dists) / scale,
                "wrist_inside": 1.0 if inside else 0.0,
                "wrist_mean_dist_norm": (sum(dists) / len(dists)) / scale,
                "centroid_dist_norm": min(centroid_dists) / scale,
            }
        return out
