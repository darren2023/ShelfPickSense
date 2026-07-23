"""聚合速度特征（对齐 visual-dps skeleton_features 聚合列）。"""

from __future__ import annotations

from analysis.constants import (
    LEFT_ANKLE_IDX,
    LEFT_HIP_IDX,
    LEFT_KNEE_IDX,
    LEFT_WRIST_IDX,
    RIGHT_ANKLE_IDX,
    RIGHT_HIP_IDX,
    RIGHT_KNEE_IDX,
    RIGHT_WRIST_IDX,
)
from analysis.features.base import FeatureContext, FeatureExtractor
from analysis.features.tracking import person_track_id
from analysis.features.velocity import aggregate_speed, point_speed_px_s

LOWER_KPT_INDICES = (
    LEFT_HIP_IDX,
    RIGHT_HIP_IDX,
    LEFT_KNEE_IDX,
    RIGHT_KNEE_IDX,
    LEFT_ANKLE_IDX,
    RIGHT_ANKLE_IDX,
)
ANKLE_KPT_INDICES = (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX)
WRIST_KPT_INDICES = (LEFT_WRIST_IDX, RIGHT_WRIST_IDX)


def speed_features_for_person(ctx: FeatureContext) -> dict[str, float]:
    person = ctx.current_person
    if person is None:
        return {
            "ankle_max_speed": 0.0,
            "lower_mean_speed": 0.0,
            "wrist_max_speed": 0.0,
        }

    track_id = ctx.current_track_id
    if track_id is None:
        track_id = person_track_id(person)

    ankle_speeds = [point_speed_px_s(ctx, track_id=track_id, kpt_idx=i) for i in ANKLE_KPT_INDICES]
    wrist_speeds = [point_speed_px_s(ctx, track_id=track_id, kpt_idx=i) for i in WRIST_KPT_INDICES]
    lower_speeds = [point_speed_px_s(ctx, track_id=track_id, kpt_idx=i) for i in LOWER_KPT_INDICES]

    return {
        "ankle_max_speed": aggregate_speed(ankle_speeds, mode="max"),
        "lower_mean_speed": aggregate_speed(lower_speeds, mode="mean"),
        "wrist_max_speed": aggregate_speed(wrist_speeds, mode="max"),
    }


class SpeedFeatureExtractor(FeatureExtractor):
    name = "speed"

    def frame_feature_names(self) -> list[str]:
        return [
            "ankle_max_speed",
            "lower_mean_speed",
            "wrist_max_speed",
        ]

    def extract_frame(self, ctx: FeatureContext) -> dict[str, float]:
        return speed_features_for_person(ctx)
