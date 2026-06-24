"""跨帧人体跟踪匹配：基于 track_id 关联同一 person。"""

from __future__ import annotations

import math
from typing import Any

from analysis.constants import (
    LEFT_ANKLE_IDX,
    LEFT_SHOULDER_IDX,
    LEFT_WRIST_IDX,
    RIGHT_ANKLE_IDX,
    RIGHT_SHOULDER_IDX,
    RIGHT_WRIST_IDX,
)
from analysis.features.base import FeatureContext
from analysis.records import FramePersons

MIN_KEYPOINT_SCORE = 0.3
MAX_PERSON_SLOTS = 3
CONSECUTIVE_HIT_WINDOWS = (3, 5, 7)
HAND_MOVE_OFFSETS = (1, 3, 5, 7)

LEFT_WRIST = "left_wrist"
RIGHT_WRIST = "right_wrist"
LEFT_FOOT = "left_foot"
RIGHT_FOOT = "right_foot"

SIDE_TO_IDX = {
    LEFT_WRIST: LEFT_WRIST_IDX,
    RIGHT_WRIST: RIGHT_WRIST_IDX,
    LEFT_FOOT: LEFT_ANKLE_IDX,
    RIGHT_FOOT: RIGHT_ANKLE_IDX,
}


def person_track_id(person: dict[str, Any]) -> int | None:
    """读取 person 的 track_id，优先 person_track_id。"""
    for key in ("person_track_id", "track_id"):
        raw = person.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    raw = person.get("person_id")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    return None


def get_keypoint(person: dict[str, Any], idx: int, *, min_score: float = MIN_KEYPOINT_SCORE) -> tuple[float, float, float] | None:
    kpts = person.get("keypoints") or []
    if idx >= len(kpts):
        return None
    kp = kpts[idx]
    if not isinstance(kp, (list, tuple)) or len(kp) < 3:
        return None
    if kp[0] is None or kp[1] is None or kp[2] is None:
        return None
    score = float(kp[2])
    if score <= min_score:
        return None
    return float(kp[0]), float(kp[1]), score


def get_side_point(person: dict[str, Any], side: str) -> tuple[float, float] | None:
    idx = SIDE_TO_IDX.get(side)
    if idx is None:
        return None
    pt = get_keypoint(person, idx)
    if pt is None:
        return None
    return pt[0], pt[1]


def person_anchor(person: dict[str, Any]) -> tuple[float, float] | None:
    kpts = person.get("keypoints") or []
    ls = get_keypoint(person, LEFT_SHOULDER_IDX, min_score=0.2)
    rs = get_keypoint(person, RIGHT_SHOULDER_IDX, min_score=0.2)
    if ls and rs:
        return (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0
    xs: list[float] = []
    ys: list[float] = []
    for kp in kpts if isinstance(kpts, list) else []:
        if isinstance(kp, (list, tuple)) and len(kp) >= 2 and kp[0] is not None and kp[1] is not None:
            xs.append(float(kp[0]))
            ys.append(float(kp[1]))
    if xs:
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return None


def find_person_by_track(frame: FramePersons, track_id: int | None) -> dict[str, Any] | None:
    if track_id is None:
        return None
    for person in frame.persons:
        if person_track_id(person) == track_id:
            return person
    return None


def match_person_by_anchor(frame: FramePersons, anchor: tuple[float, float]) -> dict[str, Any] | None:
    """无 track_id 时，用肩点中心最近邻匹配。"""
    best_person: dict[str, Any] | None = None
    best_dist = float("inf")
    for person in frame.persons:
        pa = person_anchor(person)
        if pa is None:
            continue
        dist = math.hypot(pa[0] - anchor[0], pa[1] - anchor[1])
        if dist < best_dist:
            best_dist = dist
            best_person = person
    return best_person


def sorted_persons(frame: FramePersons) -> list[dict[str, Any]]:
    """按 track_id 稳定排序，最多返回 MAX_PERSON_SLOTS 人。"""
    persons = list(frame.persons)
    persons.sort(key=lambda p: (person_track_id(p) is None, person_track_id(p) or 0, id(p)))
    return persons[:MAX_PERSON_SLOTS]


def select_primary_track_id(ctx: FeatureContext) -> int | None:
    """选择用于跨帧时序特征的主 track：优先与上一帧能匹配且手腕置信度最高者。"""
    current = sorted_persons(ctx.frame)
    if not current:
        return None

    prev_frame = ctx.prior_frame(1)
    candidates: list[tuple[float, int | None]] = []
    for person in current:
        track_id = person_track_id(person)
        score = _wrist_confidence(person)
        if prev_frame is not None:
            matched = find_person_by_track(prev_frame, track_id)
            if matched is None and track_id is None:
                anchor = person_anchor(person)
                if anchor is not None:
                    matched = match_person_by_anchor(prev_frame, anchor)
            if matched is None:
                score *= 0.5
        candidates.append((score, track_id))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def find_tracked_person(frame: FramePersons, track_id: int | None, *, fallback_anchor: tuple[float, float] | None = None) -> dict[str, Any] | None:
    person = find_person_by_track(frame, track_id)
    if person is not None:
        return person
    if fallback_anchor is not None:
        return match_person_by_anchor(frame, fallback_anchor)
    return None


def tracked_person_at_offset(ctx: FeatureContext, track_id: int | None, offset: int) -> dict[str, Any] | None:
    frame = ctx.prior_frame(offset)
    if frame is None:
        return None
    anchor = None
    if offset == 0:
        person = find_person_by_track(frame, track_id)
        return person
    cur_person = find_person_by_track(ctx.frame, track_id)
    if cur_person is not None:
        anchor = person_anchor(cur_person)
    return find_tracked_person(frame, track_id, fallback_anchor=anchor)


def _wrist_confidence(person: dict[str, Any]) -> float:
    scores = []
    for side in (LEFT_WRIST, RIGHT_WRIST):
        idx = SIDE_TO_IDX[side]
        pt = get_keypoint(person, idx)
        if pt is not None:
            scores.append(pt[2])
    return max(scores) if scores else 0.0


def side_movement_norm(
    ctx: FeatureContext,
    *,
    track_id: int | None,
    side: str,
    offset: int,
) -> float:
    """同一 track 的手腕/脚在跨 offset 帧后的位移（归一化）。"""
    if offset <= 0:
        return 0.0
    cur_person = find_person_by_track(ctx.frame, track_id)
    if cur_person is None:
        return 0.0
    cur = get_side_point(cur_person, side)
    if cur is None:
        return 0.0

    prev_person = tracked_person_at_offset(ctx, track_id, offset)
    if prev_person is None:
        return 0.0
    prev = get_side_point(prev_person, side)
    if prev is None:
        return 0.0

    scale = max(ctx.record.infer_width, ctx.record.infer_height, 1.0)
    return math.hypot(cur[0] - prev[0], cur[1] - prev[1]) / scale


def empty_person_slot_features(slot: int) -> dict[str, float]:
    prefix = f"p{slot}"
    feats = {f"{prefix}_track_id": 0.0, f"{prefix}_present": 0.0}
    for window in (3, 5, 7):
        feats[f"{prefix}_consecutive_hit_{window}"] = 0.0
    for side in (LEFT_WRIST, RIGHT_WRIST):
        for offset in HAND_MOVE_OFFSETS:
            feats[f"{prefix}_{side}_move_{offset}"] = 0.0
    return feats
