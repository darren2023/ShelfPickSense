"""人体与货框空间关系特征。"""

from __future__ import annotations

import math
from typing import Any

from analysis.constants import LEFT_ANKLE_IDX, LEFT_WRIST_IDX, RIGHT_ANKLE_IDX, RIGHT_WRIST_IDX
from analysis.features.base import FeatureContext, FeatureExtractor
from analysis.features.tracking import (
    LEFT_FOOT,
    LEFT_WRIST,
    MAX_PERSON_SLOTS,
    RIGHT_FOOT,
    RIGHT_WRIST,
    get_keypoint,
    get_side_point,
    person_track_id,
    select_primary_track_id,
    sorted_persons,
)
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


def wrist_hit_box_for_person(person: dict[str, Any], polygon: tuple[tuple[float, float], ...]) -> bool:
    for side in (LEFT_WRIST, RIGHT_WRIST):
        pt = get_side_point(person, side)
        if pt is not None and point_in_polygon(pt[0], pt[1], polygon):
            return True
    return False


def collect_wrist_points(frame: FramePersons) -> list[tuple[float, float, float]]:
    pts: list[tuple[float, float, float]] = []
    for person in frame.persons:
        for idx in (LEFT_WRIST_IDX, RIGHT_WRIST_IDX):
            pt = get_keypoint(person, idx)
            if pt is not None:
                pts.append(pt)
    return pts


def wrist_hit_box(frame: FramePersons, polygon: tuple[tuple[float, float], ...]) -> bool:
    """任意 person 的手腕是否命中货框。"""
    for person in frame.persons:
        if wrist_hit_box_for_person(person, polygon):
            return True
    return False


def _side_min_box_distance(
    person: dict[str, Any],
    side: str,
    ctx: FeatureContext,
    scale: float,
) -> tuple[float, float]:
    """返回到最近货框的归一化距离，以及是否进入任一货框。"""
    pt = get_side_point(person, side)
    if pt is None or not ctx.box_index:
        return 1.0, 0.0
    dists = [point_to_polygon_dist(pt[0], pt[1], box.polygon) for box in ctx.box_index.values()]
    inside = any(point_in_polygon(pt[0], pt[1], box.polygon) for box in ctx.box_index.values())
    return min(dists) / scale, 1.0 if inside else 0.0


def _side_box_distance_for_polygon(
    person: dict[str, Any],
    side: str,
    polygon: tuple[tuple[float, float], ...],
    scale: float,
) -> tuple[float, float]:
    pt = get_side_point(person, side)
    if pt is None:
        return 1.0, 0.0
    dist = point_to_polygon_dist(pt[0], pt[1], polygon)
    inside = point_in_polygon(pt[0], pt[1], polygon)
    return dist / scale, 1.0 if inside else 0.0


def _person_slot_spatial_features(person: dict[str, Any], ctx: FeatureContext, slot: int) -> dict[str, float]:
    scale = max(ctx.record.infer_width, ctx.record.infer_height, 1.0)
    prefix = f"p{slot}"
    track_id = person_track_id(person)
    feats: dict[str, float] = {
        f"{prefix}_track_id": float(track_id or 0),
        f"{prefix}_present": 1.0,
    }
    for side in (LEFT_WRIST, RIGHT_WRIST, LEFT_FOOT, RIGHT_FOOT):
        dist, inside = _side_min_box_distance(person, side, ctx, scale)
        feats[f"{prefix}_{side}_min_box_dist_norm"] = dist
        feats[f"{prefix}_{side}_inside_any_box"] = inside
    return feats


def _box_centroid(polygon: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return sum(xs) / len(xs), sum(ys) / len(ys)


class BoxSpatialFeatureExtractor(FeatureExtractor):
    name = "spatial"

    def extract_frame(self, ctx: FeatureContext) -> dict[str, float]:
        scale = max(ctx.record.infer_width, ctx.record.infer_height, 1.0)
        out: dict[str, float] = {}

        primary_track = select_primary_track_id(ctx)
        primary_person = next((p for p in ctx.frame.persons if person_track_id(p) == primary_track), None)
        if primary_person is None and ctx.frame.persons:
            primary_person = sorted_persons(ctx.frame)[0]

        if primary_person is not None:
            for side in (LEFT_WRIST, RIGHT_WRIST, LEFT_FOOT, RIGHT_FOOT):
                dist, inside = _side_min_box_distance(primary_person, side, ctx, scale)
                out[f"primary_{side}_min_box_dist_norm"] = dist
                out[f"primary_{side}_inside_any_box"] = inside
        else:
            for side in (LEFT_WRIST, RIGHT_WRIST, LEFT_FOOT, RIGHT_FOOT):
                out[f"primary_{side}_min_box_dist_norm"] = 1.0
                out[f"primary_{side}_inside_any_box"] = 0.0

        persons = sorted_persons(ctx.frame)
        for slot in range(MAX_PERSON_SLOTS):
            if slot >= len(persons):
                out.update(_empty_slot_spatial_only(slot))
                continue
            out.update(_person_slot_spatial_features(persons[slot], ctx, slot))

        per_box = self.extract_per_box(ctx)
        if not per_box:
            out.update(
                {
                    "min_wrist_box_dist_norm": 1.0,
                    "any_wrist_inside_box": 0.0,
                    "boxes_with_wrist_inside": 0.0,
                    "min_foot_box_dist_norm": 1.0,
                    "any_foot_inside_box": 0.0,
                    "boxes_with_foot_inside": 0.0,
                }
            )
            return out

        wrist_dists = [feats["wrist_min_dist_norm"] for feats in per_box.values()]
        wrist_inside_count = sum(1 for feats in per_box.values() if feats.get("wrist_inside", 0.0) > 0.5)
        foot_dists = [feats["foot_min_dist_norm"] for feats in per_box.values()]
        foot_inside_count = sum(1 for feats in per_box.values() if feats.get("foot_inside", 0.0) > 0.5)
        out.update(
            {
                "min_wrist_box_dist_norm": min(wrist_dists) if wrist_dists else 1.0,
                "any_wrist_inside_box": 1.0 if wrist_inside_count > 0 else 0.0,
                "boxes_with_wrist_inside": float(wrist_inside_count),
                "min_foot_box_dist_norm": min(foot_dists) if foot_dists else 1.0,
                "any_foot_inside_box": 1.0 if foot_inside_count > 0 else 0.0,
                "boxes_with_foot_inside": float(foot_inside_count),
            }
        )
        return out

    def extract_per_box(self, ctx: FeatureContext) -> dict[str, dict[str, float]]:
        scale = max(ctx.record.infer_width, ctx.record.infer_height, 1.0)
        primary_track = select_primary_track_id(ctx)
        primary_person = next((p for p in ctx.frame.persons if person_track_id(p) == primary_track), None)
        if primary_person is None and ctx.frame.persons:
            primary_person = sorted_persons(ctx.frame)[0]

        out: dict[str, dict[str, float]] = {}
        for token, box in ctx.box_index.items():
            feats: dict[str, float] = {}
            if primary_person is not None:
                for side, prefix in (
                    (LEFT_WRIST, "left_wrist"),
                    (RIGHT_WRIST, "right_wrist"),
                    (LEFT_FOOT, "left_foot"),
                    (RIGHT_FOOT, "right_foot"),
                ):
                    dist, inside = _side_box_distance_for_polygon(primary_person, side, box.polygon, scale)
                    feats[f"{prefix}_dist_norm"] = dist
                    feats[f"{prefix}_inside"] = inside

            wrists = [(x, y) for x, y, _ in collect_wrist_points(ctx.frame)]
            ankles = []
            for person in ctx.frame.persons:
                for side in (LEFT_FOOT, RIGHT_FOOT):
                    pt = get_side_point(person, side)
                    if pt is not None:
                        ankles.append(pt)

            if wrists:
                wrist_dists = [point_to_polygon_dist(x, y, box.polygon) for x, y in wrists]
                feats["wrist_min_dist_norm"] = min(wrist_dists) / scale
                feats["wrist_inside"] = 1.0 if any(point_in_polygon(x, y, box.polygon) for x, y in wrists) else 0.0
                feats["wrist_mean_dist_norm"] = (sum(wrist_dists) / len(wrist_dists)) / scale
                cx, cy = _box_centroid(box.polygon)
                centroid_dists = [math.hypot(x - cx, y - cy) for x, y in wrists]
                feats["centroid_dist_norm"] = min(centroid_dists) / scale
            else:
                feats.update(
                    {
                        "wrist_min_dist_norm": 1.0,
                        "wrist_inside": 0.0,
                        "wrist_mean_dist_norm": 1.0,
                        "centroid_dist_norm": 1.0,
                    }
                )

            if ankles:
                foot_dists = [point_to_polygon_dist(x, y, box.polygon) for x, y in ankles]
                feats["foot_min_dist_norm"] = min(foot_dists) / scale
                feats["foot_inside"] = 1.0 if any(point_in_polygon(x, y, box.polygon) for x, y in ankles) else 0.0
                feats["foot_mean_dist_norm"] = (sum(foot_dists) / len(foot_dists)) / scale
            else:
                feats.update({"foot_min_dist_norm": 1.0, "foot_inside": 0.0, "foot_mean_dist_norm": 1.0})

            out[token] = feats
        return out


def _empty_slot_spatial_only(slot: int) -> dict[str, float]:
    prefix = f"p{slot}"
    feats = {f"{prefix}_track_id": 0.0, f"{prefix}_present": 0.0}
    for side in (LEFT_WRIST, RIGHT_WRIST, LEFT_FOOT, RIGHT_FOOT):
        feats[f"{prefix}_{side}_min_box_dist_norm"] = 1.0
        feats[f"{prefix}_{side}_inside_any_box"] = 0.0
    return feats
