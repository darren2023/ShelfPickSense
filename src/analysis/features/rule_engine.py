"""与 event_engine 规则碰撞检测对齐的离线特征。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from analysis.constants import (
    LEFT_ELBOW_IDX,
    LEFT_SHOULDER_IDX,
    LEFT_WRIST_IDX,
    RIGHT_ELBOW_IDX,
    RIGHT_SHOULDER_IDX,
    RIGHT_WRIST_IDX,
)
from analysis.features.base import FeatureContext, FeatureExtractor
from analysis.features.spatial import point_in_polygon, point_to_polygon_dist, point_to_segment_dist
from analysis.features.tracking import (
    MAX_PERSON_SLOTS,
    find_person_by_track,
    get_keypoint,
    person_track_id,
    select_primary_track_id,
    sorted_persons,
)

RULE_WINDOW_PAIRS = ((3, 6), (3, 7), (5, 7))


@dataclass
class RuleEngineParams:
    """默认与 services.event_engine.collision.CollisionParams 一致。"""

    wrist_conf: float = 0.3
    elbow_conf: float = 0.3
    forearm_extend_ratio: float = 0.4
    boundary_margin_ratio: float = 0.12
    boundary_margin_min_px: float = 8.0
    min_consecutive_frames: int = 3
    window_frames: int = 6


def signed_polygon_distance(x: float, y: float, polygon: tuple[tuple[float, float], ...]) -> float:
    """有符号距离：多边形内为正，外为负（与 cv2.pointPolygonTest measureDist=True 一致）。"""
    if len(polygon) < 3:
        return -1.0
    if point_in_polygon(x, y, polygon):
        best = float("inf")
        n = len(polygon)
        for i in range(n):
            ax, ay = polygon[i]
            bx, by = polygon[(i + 1) % n]
            best = min(best, point_to_segment_dist(x, y, ax, ay, bx, by))
        return best
    return -point_to_polygon_dist(x, y, polygon)


def person_shoulder_width(person: dict) -> float:
    left = get_keypoint(person, LEFT_SHOULDER_IDX, min_score=0.2)
    right = get_keypoint(person, RIGHT_SHOULDER_IDX, min_score=0.2)
    if left and right:
        return math.hypot(left[0] - right[0], left[1] - right[1])
    return 0.0


def collision_margin(person: dict, params: RuleEngineParams) -> float:
    shoulder_width = person_shoulder_width(person)
    return max(params.boundary_margin_min_px, params.boundary_margin_ratio * shoulder_width)


def collect_rule_hand_points(person: dict, params: RuleEngineParams) -> list[tuple[float, float, str]]:
    """腕点 + 肘→腕外推点，与 event_engine 一致。"""
    points: list[tuple[float, float, str]] = []
    for elbow_idx, wrist_idx, wrist_kind, forearm_kind in (
        (LEFT_ELBOW_IDX, LEFT_WRIST_IDX, "wrist_l", "forearm_l"),
        (RIGHT_ELBOW_IDX, RIGHT_WRIST_IDX, "wrist_r", "forearm_r"),
    ):
        wrist = get_keypoint(person, wrist_idx, min_score=params.wrist_conf)
        if wrist is None:
            continue
        wx, wy = wrist[0], wrist[1]
        points.append((wx, wy, wrist_kind))
        elbow = get_keypoint(person, elbow_idx, min_score=params.elbow_conf)
        if elbow is not None:
            ex, ey = elbow[0], elbow[1]
            points.append(
                (
                    wx + params.forearm_extend_ratio * (wx - ex),
                    wy + params.forearm_extend_ratio * (wy - ey),
                    forearm_kind,
                )
            )
    return points


def nearest_box_token_for_point(
    px: float,
    py: float,
    margin: float,
    box_index: dict,
) -> str:
    best_token = ""
    best_dist = margin + 1.0
    for token, box in box_index.items():
        signed = signed_polygon_distance(px, py, box.polygon)
        if signed >= -margin and signed < best_dist:
            best_dist = signed
            best_token = token
    return best_token


def rule_collision_tokens_for_person(
    person: dict,
    ctx: FeatureContext,
    params: RuleEngineParams,
) -> set[str]:
    hand_points = collect_rule_hand_points(person, params)
    if not hand_points:
        return set()
    margin = collision_margin(person, params)
    tokens: set[str] = set()
    for px, py, _kind in hand_points:
        token = nearest_box_token_for_point(px, py, margin, ctx.box_index)
        if token:
            tokens.add(token)
    return tokens


def rule_collision_tokens_for_frame(ctx: FeatureContext, params: RuleEngineParams) -> set[str]:
    tokens: set[str] = set()
    for person in ctx.frame.persons:
        tokens.update(rule_collision_tokens_for_person(person, ctx, params))
    return tokens


def _frame_context(ctx: FeatureContext, offset: int) -> FeatureContext | None:
    frame = ctx.prior_frame(offset)
    if frame is None:
        return None
    return FeatureContext(
        record=ctx.record,
        frame=frame,
        box_index=ctx.box_index,
        box_tokens=ctx.box_tokens,
        frame_index=ctx.frame_index,
    )


def window_hit_count_for_track(
    ctx: FeatureContext,
    track_id: int | None,
    params: RuleEngineParams,
    *,
    box_token: str | None = None,
    window: int | None = None,
) -> int:
    lookback = window if window is not None else params.window_frames
    hits = 0
    for offset in range(lookback):
        hist_ctx = _frame_context(ctx, offset)
        if hist_ctx is None:
            break
        person = find_person_by_track(hist_ctx.frame, track_id)
        if person is None:
            continue
        tokens = rule_collision_tokens_for_person(person, hist_ctx, params)
        if box_token is None:
            if tokens:
                hits += 1
        elif box_token in tokens:
            hits += 1
    return hits


def _box_hand_collision_flags(
    person: dict,
    token: str,
    ctx: FeatureContext,
    params: RuleEngineParams,
) -> tuple[float, float, float, float]:
    box = ctx.box_index.get(token)
    if box is None:
        return 0.0, 0.0, 0.0, -1.0
    margin = collision_margin(person, params)
    scale = max(ctx.record.infer_width, ctx.record.infer_height, 1.0)
    wrist_hit = 0.0
    forearm_hit = 0.0
    best_signed = -1.0
    for px, py, kind in collect_rule_hand_points(person, params):
        signed = signed_polygon_distance(px, py, box.polygon)
        best_signed = max(best_signed, signed)
        if signed >= -margin:
            if kind.startswith("wrist"):
                wrist_hit = 1.0
            elif kind.startswith("forearm"):
                forearm_hit = 1.0
    hand_hit = 1.0 if (wrist_hit > 0.5 or forearm_hit > 0.5) else 0.0
    return wrist_hit, forearm_hit, hand_hit, best_signed / scale


class RuleEngineFeatureExtractor(FeatureExtractor):
    """event_engine 规则碰撞检测对应的离线特征（软边界 + 前臂外推 + M-of-N 滑窗）。"""

    name = "rule"

    def __init__(self, params: RuleEngineParams | None = None) -> None:
        self.params = params or RuleEngineParams()

    def extract_frame(self, ctx: FeatureContext) -> dict[str, float]:
        params = self.params
        active_tokens = rule_collision_tokens_for_frame(ctx, params)
        primary_track = select_primary_track_id(ctx)
        primary_tokens: set[str] = set()
        primary_person = find_person_by_track(ctx.frame, primary_track)
        if primary_person is not None:
            primary_tokens = rule_collision_tokens_for_person(primary_person, ctx, params)

        out: dict[str, float] = {
            "any_collision": 1.0 if active_tokens else 0.0,
            "collision_count": float(len(active_tokens)),
            "primary_any_collision": 1.0 if primary_tokens else 0.0,
            "primary_collision_count": float(len(primary_tokens)),
        }

        for min_hits, window in RULE_WINDOW_PAIRS:
            primary_hits = window_hit_count_for_track(ctx, primary_track, params, window=window)
            out[f"window_hit_{min_hits}_{window}"] = 1.0 if primary_hits >= min_hits else 0.0
            out[f"window_hits_{window}"] = float(primary_hits)

        persons = sorted_persons(ctx.frame)
        for slot in range(MAX_PERSON_SLOTS):
            prefix = f"p{slot}"
            if slot >= len(persons):
                out.update(
                    {
                        f"{prefix}_present": 0.0,
                        f"{prefix}_track_id": 0.0,
                        f"{prefix}_any_collision": 0.0,
                        f"{prefix}_collision_count": 0.0,
                    }
                )
                continue
            person = persons[slot]
            tokens = rule_collision_tokens_for_person(person, ctx, params)
            track_id = person_track_id(person)
            out.update(
                {
                    f"{prefix}_present": 1.0,
                    f"{prefix}_track_id": float(track_id or 0),
                    f"{prefix}_any_collision": 1.0 if tokens else 0.0,
                    f"{prefix}_collision_count": float(len(tokens)),
                }
            )
            for min_hits, window in RULE_WINDOW_PAIRS:
                hits = window_hit_count_for_track(ctx, track_id, params, window=window)
                out[f"{prefix}_window_hit_{min_hits}_{window}"] = 1.0 if hits >= min_hits else 0.0

        max_hits = 0
        for person in ctx.frame.persons:
            max_hits = max(max_hits, window_hit_count_for_track(ctx, person_track_id(person), params))
        out[f"any_track_window_hit_{params.min_consecutive_frames}_{params.window_frames}"] = (
            1.0 if max_hits >= params.min_consecutive_frames else 0.0
        )
        return out

    def extract_per_box(self, ctx: FeatureContext) -> dict[str, dict[str, float]]:
        params = self.params
        primary_track = select_primary_track_id(ctx)
        primary_person = find_person_by_track(ctx.frame, primary_track)
        if primary_person is None and ctx.frame.persons:
            primary_person = sorted_persons(ctx.frame)[0]

        out: dict[str, dict[str, float]] = {}
        for token in ctx.box_tokens:
            feats: dict[str, float] = {
                "frame_collision": 1.0 if token in rule_collision_tokens_for_frame(ctx, params) else 0.0,
            }
            if primary_person is not None:
                wrist_hit, forearm_hit, hand_hit, signed_norm = _box_hand_collision_flags(
                    primary_person, token, ctx, params
                )
                feats.update(
                    {
                        "wrist_collision": wrist_hit,
                        "forearm_collision": forearm_hit,
                        "hand_collision": hand_hit,
                        "max_signed_dist_norm": signed_norm,
                    }
                )
                nearest = ""
                margin = collision_margin(primary_person, params)
                for px, py, _kind in collect_rule_hand_points(primary_person, params):
                    hit = nearest_box_token_for_point(px, py, margin, ctx.box_index)
                    if hit:
                        nearest = hit
                        break
                feats["nearest_collision"] = 1.0 if nearest == token else 0.0
            else:
                feats.update(
                    {
                        "wrist_collision": 0.0,
                        "forearm_collision": 0.0,
                        "hand_collision": 0.0,
                        "max_signed_dist_norm": -1.0,
                        "nearest_collision": 0.0,
                    }
                )

            for min_hits, window in RULE_WINDOW_PAIRS:
                hits = window_hit_count_for_track(ctx, primary_track, params, box_token=token, window=window)
                feats[f"window_hit_{min_hits}_{window}"] = 1.0 if hits >= min_hits else 0.0
                feats[f"window_hits_{window}"] = float(hits)
            out[token] = feats
        return out
