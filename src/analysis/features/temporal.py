"""跨帧时序特征：连续命中、手部移动距离。"""

from __future__ import annotations

import math

from analysis.constants import LEFT_WRIST_IDX, RIGHT_WRIST_IDX
from analysis.features.base import FeatureContext, FeatureExtractor
from analysis.features.spatial import wrist_hit_box
from analysis.records import FramePersons

CONSECUTIVE_HIT_WINDOWS = (3, 5, 7)
HAND_MOVE_OFFSETS = (1, 3, 5, 7)


def consecutive_hit_streak(ctx: FeatureContext, box_token: str, *, max_lookback: int = 7) -> int:
    """统计当前帧起向前的连续手腕命中 streak。"""
    box = ctx.box_index.get(box_token)
    if box is None:
        return 0
    streak = 0
    for offset in range(max_lookback):
        frame = ctx.prior_frame(offset)
        if frame is None:
            break
        if not wrist_hit_box(frame, box.polygon):
            break
        streak += 1
    return streak


def _primary_wrist(frame: FramePersons) -> tuple[float, float] | None:
    """取置信度最高的手腕坐标。"""
    best: tuple[float, float, float] | None = None
    for person in frame.persons:
        kpts = person.get("keypoints") or []
        for idx in (LEFT_WRIST_IDX, RIGHT_WRIST_IDX):
            if idx >= len(kpts):
                continue
            kp = kpts[idx]
            if not isinstance(kp, (list, tuple)) or len(kp) < 3:
                continue
            if kp[0] is None or kp[1] is None or kp[2] is None:
                continue
            score = float(kp[2])
            if score <= 0.3:
                continue
            if best is None or score > best[2]:
                best = (float(kp[0]), float(kp[1]), score)
    if best is None:
        return None
    return best[0], best[1]


def hand_movement_norm(ctx: FeatureContext, offset: int) -> float:
    """跨 offset 帧的手部（手腕）移动距离，按画面尺寸归一化。"""
    if offset <= 0:
        return 0.0
    cur = _primary_wrist(ctx.frame)
    prev_frame = ctx.prior_frame(offset)
    if cur is None or prev_frame is None:
        return 0.0
    prev = _primary_wrist(prev_frame)
    if prev is None:
        return 0.0
    scale = max(ctx.record.infer_width, ctx.record.infer_height, 1.0)
    return math.hypot(cur[0] - prev[0], cur[1] - prev[1]) / scale


class TemporalFeatureExtractor(FeatureExtractor):
    name = "temporal"

    def extract_frame(self, ctx: FeatureContext) -> dict[str, float]:
        out: dict[str, float] = {}
        max_streak = 0
        for token in ctx.box_tokens:
            streak = consecutive_hit_streak(ctx, token)
            max_streak = max(max_streak, streak)
        for window in CONSECUTIVE_HIT_WINDOWS:
            out[f"consecutive_hit_{window}"] = 1.0 if max_streak >= window else 0.0
        for offset in HAND_MOVE_OFFSETS:
            out[f"hand_move_{offset}"] = hand_movement_norm(ctx, offset)
        return out

    def extract_per_box(self, ctx: FeatureContext) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for token in ctx.box_tokens:
            streak = consecutive_hit_streak(ctx, token)
            feats = {f"consecutive_hit_{window}": 1.0 if streak >= window else 0.0 for window in CONSECUTIVE_HIT_WINDOWS}
            out[token] = feats
        return out
