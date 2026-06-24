"""跨帧时序特征：连续命中、左右手位移（按 track_id 匹配）。"""

from __future__ import annotations

from analysis.features.base import FeatureContext, FeatureExtractor
from analysis.features.spatial import wrist_hit_box_for_person
from analysis.features.tracking import (    CONSECUTIVE_HIT_WINDOWS,
    HAND_MOVE_OFFSETS,
    LEFT_WRIST,
    MAX_PERSON_SLOTS,
    RIGHT_WRIST,
    empty_person_slot_features,
    find_person_by_track,
    person_track_id,
    select_primary_track_id,
    side_movement_norm,
    sorted_persons,
)


def consecutive_hit_streak_for_track(ctx: FeatureContext, track_id: int | None, *, max_lookback: int = 7) -> int:
    """统计同一 track 在当前帧向前连续命中（手腕进入任一货框）的帧数。"""
    streak = 0
    for offset in range(max_lookback):
        frame = ctx.prior_frame(offset)
        if frame is None:
            break
        person = find_person_by_track(frame, track_id)
        if person is None:
            break
        if not _person_hits_any_box(person, ctx):
            break
        streak += 1
    return streak


def _person_hits_any_box(person: dict, ctx: FeatureContext) -> bool:
    for box in ctx.box_index.values():
        if wrist_hit_box_for_person(person, box.polygon):
            return True
    return False


class TemporalFeatureExtractor(FeatureExtractor):
    name = "temporal"

    def extract_frame(self, ctx: FeatureContext) -> dict[str, float]:
        out: dict[str, float] = {}

        primary_track = select_primary_track_id(ctx)
        primary_streak = consecutive_hit_streak_for_track(ctx, primary_track)
        for window in CONSECUTIVE_HIT_WINDOWS:
            out[f"consecutive_hit_{window}"] = 1.0 if primary_streak >= window else 0.0

        for offset in HAND_MOVE_OFFSETS:
            out[f"left_wrist_move_{offset}"] = side_movement_norm(
                ctx, track_id=primary_track, side=LEFT_WRIST, offset=offset
            )
            out[f"right_wrist_move_{offset}"] = side_movement_norm(
                ctx, track_id=primary_track, side=RIGHT_WRIST, offset=offset
            )

        persons = sorted_persons(ctx.frame)
        for slot in range(MAX_PERSON_SLOTS):
            prefix = f"p{slot}"
            if slot >= len(persons):
                out.update(empty_person_slot_features(slot))
                continue

            person = persons[slot]
            track_id = person_track_id(person)
            streak = consecutive_hit_streak_for_track(ctx, track_id)
            out[f"{prefix}_track_id"] = float(track_id or 0)
            out[f"{prefix}_present"] = 1.0
            for window in CONSECUTIVE_HIT_WINDOWS:
                out[f"{prefix}_consecutive_hit_{window}"] = 1.0 if streak >= window else 0.0
            for offset in HAND_MOVE_OFFSETS:
                out[f"{prefix}_left_wrist_move_{offset}"] = side_movement_norm(
                    ctx, track_id=track_id, side=LEFT_WRIST, offset=offset
                )
                out[f"{prefix}_right_wrist_move_{offset}"] = side_movement_norm(
                    ctx, track_id=track_id, side=RIGHT_WRIST, offset=offset
                )

        max_streak = 0
        for person in ctx.frame.persons:
            max_streak = max(max_streak, consecutive_hit_streak_for_track(ctx, person_track_id(person)))
        for window in CONSECUTIVE_HIT_WINDOWS:
            out[f"any_track_consecutive_hit_{window}"] = 1.0 if max_streak >= window else 0.0

        return out
