"""站立/下肢姿态几何特征（对齐 visual-dps skeleton_angles 站立代理）。"""

from __future__ import annotations

import math

from analysis.constants import (
    LEFT_HIP_IDX,
    LEFT_KNEE_IDX,
    LEFT_SHOULDER_IDX,
    RIGHT_HIP_IDX,
    RIGHT_KNEE_IDX,
    RIGHT_SHOULDER_IDX,
)
from analysis.features.base import FeatureContext, FeatureExtractor
from analysis.features.tracking import MIN_KEYPOINT_SCORE, get_keypoint

SHK_TRIPLES = (
    (LEFT_SHOULDER_IDX, LEFT_HIP_IDX, LEFT_KNEE_IDX),
    (RIGHT_SHOULDER_IDX, RIGHT_HIP_IDX, RIGHT_KNEE_IDX),
)


def inner_angle_deg(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> float:
    """顶点 B 处内角（度），180° 为完全伸直。"""
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    la = math.hypot(bax, bay)
    lc = math.hypot(bcx, bcy)
    if la < 1e-6 or lc < 1e-6:
        return 0.0
    cosv = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / (la * lc)))
    return math.degrees(math.acos(cosv))


def shoulder_hip_knee_angle_min(person: dict) -> float:
    """左右肩-髋-膝内角的最小值；站立时通常较大（如 ≥140°）。"""
    angles: list[float] = []
    for shoulder_idx, hip_idx, knee_idx in SHK_TRIPLES:
        s = get_keypoint(person, shoulder_idx, min_score=MIN_KEYPOINT_SCORE)
        h = get_keypoint(person, hip_idx, min_score=MIN_KEYPOINT_SCORE)
        k = get_keypoint(person, knee_idx, min_score=MIN_KEYPOINT_SCORE)
        if s and h and k:
            angles.append(inner_angle_deg(s, h, k))
    return float(min(angles)) if angles else 0.0


def stance_features_for_person(ctx: FeatureContext) -> dict[str, float]:
    person = ctx.current_person
    if person is None:
        return {"shoulder_hip_knee_angle_min": 0.0}
    return {"shoulder_hip_knee_angle_min": shoulder_hip_knee_angle_min(person)}


class StanceFeatureExtractor(FeatureExtractor):
    name = "stance"

    def frame_feature_names(self) -> list[str]:
        return ["shoulder_hip_knee_angle_min"]

    def extract_frame(self, ctx: FeatureContext) -> dict[str, float]:
        return stance_features_for_person(ctx)
