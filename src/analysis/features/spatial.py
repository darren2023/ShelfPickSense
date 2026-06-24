"""人体与货框空间关系特征。"""

from __future__ import annotations

import math

from analysis.constants import LEFT_ANKLE_IDX, LEFT_WRIST_IDX, RIGHT_ANKLE_IDX, RIGHT_WRIST_IDX
from analysis.features.base import FeatureContext, FeatureExtractor
from analysis.records import FramePersons


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


def collect_keypoints(
    frame: FramePersons,
    indices: tuple[int, ...],
    *,
    min_score: float = 0.3,
) -> list[tuple[float, float, float]]:
    """收集指定关节关键点，返回 (x, y, score) 列表。"""
    pts: list[tuple[float, float, float]] = []
    for person in frame.persons:
        kpts = person.get("keypoints") or []
        for idx in indices:
            if idx >= len(kpts):
                continue
            kp = kpts[idx]
            if not isinstance(kp, (list, tuple)) or len(kp) < 3:
                continue
            if kp[0] is None or kp[1] is None or kp[2] is None:
                continue
            if float(kp[2]) <= min_score:
                continue
            pts.append((float(kp[0]), float(kp[1]), float(kp[2])))
    return pts


def collect_wrist_points(frame: FramePersons) -> list[tuple[float, float, float]]:
    return collect_keypoints(frame, (LEFT_WRIST_IDX, RIGHT_WRIST_IDX))


def collect_ankle_points(frame: FramePersons) -> list[tuple[float, float, float]]:
    return collect_keypoints(frame, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX))


def wrist_hit_box(frame: FramePersons, polygon: tuple[tuple[float, float], ...]) -> bool:
    """手腕是否命中（进入）货框。"""
    wrists = collect_wrist_points(frame)
    return any(point_in_polygon(x, y, polygon) for x, y, _ in wrists)


def _box_centroid(polygon: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _body_part_box_features(
    points: list[tuple[float, float]],
    polygon: tuple[tuple[float, float], ...],
    scale: float,
    *,
    prefix: str,
) -> dict[str, float]:
    if not points:
        return {
            f"{prefix}_min_dist_norm": 1.0,
            f"{prefix}_inside": 0.0,
            f"{prefix}_mean_dist_norm": 1.0,
        }
    dists = [point_to_polygon_dist(x, y, polygon) for x, y in points]
    inside = any(point_in_polygon(x, y, polygon) for x, y in points)
    return {
        f"{prefix}_min_dist_norm": min(dists) / scale,
        f"{prefix}_inside": 1.0 if inside else 0.0,
        f"{prefix}_mean_dist_norm": (sum(dists) / len(dists)) / scale,
    }


class BoxSpatialFeatureExtractor(FeatureExtractor):
    name = "spatial"

    def extract_frame(self, ctx: FeatureContext) -> dict[str, float]:
        per_box = self.extract_per_box(ctx)
        if not per_box:
            return {
                "min_wrist_box_dist_norm": 1.0,
                "any_wrist_inside_box": 0.0,
                "boxes_with_wrist_inside": 0.0,
                "min_foot_box_dist_norm": 1.0,
                "any_foot_inside_box": 0.0,
                "boxes_with_foot_inside": 0.0,
            }
        wrist_dists = [feats["wrist_min_dist_norm"] for feats in per_box.values()]
        wrist_inside_count = sum(1 for feats in per_box.values() if feats.get("wrist_inside", 0.0) > 0.5)
        foot_dists = [feats["foot_min_dist_norm"] for feats in per_box.values()]
        foot_inside_count = sum(1 for feats in per_box.values() if feats.get("foot_inside", 0.0) > 0.5)
        return {
            "min_wrist_box_dist_norm": min(wrist_dists) if wrist_dists else 1.0,
            "any_wrist_inside_box": 1.0 if wrist_inside_count > 0 else 0.0,
            "boxes_with_wrist_inside": float(wrist_inside_count),
            "min_foot_box_dist_norm": min(foot_dists) if foot_dists else 1.0,
            "any_foot_inside_box": 1.0 if foot_inside_count > 0 else 0.0,
            "boxes_with_foot_inside": float(foot_inside_count),
        }

    def extract_per_box(self, ctx: FeatureContext) -> dict[str, dict[str, float]]:
        scale = max(ctx.record.infer_width, ctx.record.infer_height, 1.0)
        wrists = [(x, y) for x, y, _ in collect_wrist_points(ctx.frame)]
        ankles = [(x, y) for x, y, _ in collect_ankle_points(ctx.frame)]

        out: dict[str, dict[str, float]] = {}
        for token, box in ctx.box_index.items():
            feats = _body_part_box_features(wrists, box.polygon, scale, prefix="wrist")
            feats.update(_body_part_box_features(ankles, box.polygon, scale, prefix="foot"))
            if wrists:
                cx, cy = _box_centroid(box.polygon)
                centroid_dists = [math.hypot(x - cx, y - cy) for x, y in wrists]
                feats["centroid_dist_norm"] = min(centroid_dists) / scale
            else:
                feats["centroid_dist_norm"] = 1.0
            out[token] = feats
        return out
