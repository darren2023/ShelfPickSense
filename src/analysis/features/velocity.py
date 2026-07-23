"""关键点速度计算（对齐 visual-dps skeleton_features 约定）。"""

from __future__ import annotations

import math
import statistics

from analysis.features.base import FeatureContext
from analysis.features.tracking import (
    MIN_KEYPOINT_SCORE,
    find_tracked_person_at_frame,
    get_keypoint,
)

# 与 visual-dps skeleton_features 对齐
MAX_VELOCITY_GAP_FRAMES = 2
DEFAULT_VIDEO_FPS = 25.0


def median_xy(values: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not values:
        return None
    xs = sorted(v[0] for v in values)
    ys = sorted(v[1] for v in values)
    mid = len(xs) // 2
    return xs[mid], ys[mid]


def filtered_point_at_samples(
    ctx: FeatureContext,
    *,
    track_id: int | None,
    kpt_idx: int,
    sample_offsets: tuple[int, ...],
) -> tuple[float, float] | None:
    """在采样序列若干步上取关键点坐标中值。"""
    points: list[tuple[float, float]] = []
    for offset in sample_offsets:
        frame = ctx.prior_sample(offset)
        if frame is None:
            continue
        person = find_tracked_person_at_frame(ctx, track_id, frame)
        if person is None:
            continue
        pt = get_keypoint(person, kpt_idx, min_score=MIN_KEYPOINT_SCORE)
        if pt is not None:
            points.append((pt[0], pt[1]))
    return median_xy(points)


def delta_time_sec(cur_frame, prev_frame) -> float:
    dt = float(cur_frame.timestamp_sec) - float(prev_frame.timestamp_sec)
    if dt > 0.0:
        return dt
    gap = abs(int(cur_frame.frame_idx) - int(prev_frame.frame_idx))
    if gap <= 0:
        return 0.0
    return gap / DEFAULT_VIDEO_FPS


def point_speed_px_s(
    ctx: FeatureContext,
    *,
    track_id: int | None,
    kpt_idx: int,
) -> float | None:
    """单关键点标量速度（px/s），3 帧中值滤波 + 与上一采样步差分。"""
    cur_frame = ctx.prior_sample(0)
    prev_frame = ctx.prior_sample(1)
    if cur_frame is None or prev_frame is None:
        return None

    gap = abs(int(cur_frame.frame_idx) - int(prev_frame.frame_idx))
    if gap > MAX_VELOCITY_GAP_FRAMES:
        return None

    cur_xy = filtered_point_at_samples(
        ctx,
        track_id=track_id,
        kpt_idx=kpt_idx,
        sample_offsets=(0, 1, 2),
    )
    prev_xy = filtered_point_at_samples(
        ctx,
        track_id=track_id,
        kpt_idx=kpt_idx,
        sample_offsets=(1, 2, 3),
    )
    if cur_xy is None or prev_xy is None:
        return None

    dt = delta_time_sec(cur_frame, prev_frame)
    if dt <= 0.0:
        return None
    return math.hypot(cur_xy[0] - prev_xy[0], cur_xy[1] - prev_xy[1]) / dt


def aggregate_speed(values: list[float | None], *, mode: str) -> float:
    valid = [v for v in values if v is not None and v >= 0.0]
    if not valid:
        return 0.0
    if mode == "max":
        return float(max(valid))
    return float(statistics.mean(valid))
