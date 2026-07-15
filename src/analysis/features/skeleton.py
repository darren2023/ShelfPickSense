"""骨骼统计特征。"""

from __future__ import annotations

import math

from analysis.constants import (
    LEFT_ANKLE_IDX,
    LEFT_ELBOW_IDX,
    LEFT_HIP_IDX,
    LEFT_KNEE_IDX,
    LEFT_SHOULDER_IDX,
    LEFT_WRIST_IDX,
    RIGHT_ANKLE_IDX,
    RIGHT_ELBOW_IDX,
    RIGHT_HIP_IDX,
    RIGHT_KNEE_IDX,
    RIGHT_SHOULDER_IDX,
    RIGHT_WRIST_IDX,
)
from analysis.features.base import FeatureContext, FeatureExtractor
from analysis.features.tracking import find_tracked_person_at_frame, person_track_id


MIN_SHAPE_SCORE = 0.2
SHAPE_OFFSETS = (5, 10)
SPREAD_PAIRS = (
    (LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX),
    (LEFT_HIP_IDX, RIGHT_HIP_IDX),
    (LEFT_WRIST_IDX, RIGHT_WRIST_IDX),
    (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX),
    (LEFT_WRIST_IDX, LEFT_HIP_IDX),
    (RIGHT_WRIST_IDX, RIGHT_HIP_IDX),
    (LEFT_SHOULDER_IDX, LEFT_ANKLE_IDX),
    (RIGHT_SHOULDER_IDX, RIGHT_ANKLE_IDX),
)
FOLD_TRIPLES = (
    (LEFT_SHOULDER_IDX, LEFT_ELBOW_IDX, LEFT_WRIST_IDX),
    (RIGHT_SHOULDER_IDX, RIGHT_ELBOW_IDX, RIGHT_WRIST_IDX),
    (LEFT_ELBOW_IDX, LEFT_SHOULDER_IDX, LEFT_HIP_IDX),
    (RIGHT_ELBOW_IDX, RIGHT_SHOULDER_IDX, RIGHT_HIP_IDX),
    (LEFT_SHOULDER_IDX, LEFT_HIP_IDX, LEFT_KNEE_IDX),
    (RIGHT_SHOULDER_IDX, RIGHT_HIP_IDX, RIGHT_KNEE_IDX),
    (LEFT_HIP_IDX, LEFT_KNEE_IDX, LEFT_ANKLE_IDX),
    (RIGHT_HIP_IDX, RIGHT_KNEE_IDX, RIGHT_ANKLE_IDX),
)


def _pt(keypoints: list, idx: int) -> tuple[float, float, float] | None:
    if idx >= len(keypoints):
        return None
    kp = keypoints[idx]
    if not isinstance(kp, (list, tuple)) or len(kp) < 2:
        return None
    if kp[0] is None or kp[1] is None:
        return None
    x, y = float(kp[0]), float(kp[1])
    if x != x or y != y:  # NaN check without math dependency on None
        return None
    score = float(kp[2]) if len(kp) > 2 and kp[2] is not None else 0.0
    return x, y, score


def _valid_shape_pt(keypoints: list, idx: int) -> tuple[float, float, float] | None:
    pt = _pt(keypoints, idx)
    if pt is None or pt[2] < MIN_SHAPE_SCORE:
        return None
    return pt


def _person_anchor(keypoints: list) -> tuple[float, float]:
    ls = _pt(keypoints, LEFT_SHOULDER_IDX)
    rs = _pt(keypoints, RIGHT_SHOULDER_IDX)
    if ls and rs and ls[2] > 0.2 and rs[2] > 0.2:
        return (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0
    xs, ys = [], []
    for kp in keypoints:
        if isinstance(kp, (list, tuple)) and len(kp) >= 2 and kp[0] is not None and kp[1] is not None:
            xs.append(float(kp[0]))
            ys.append(float(kp[1]))
    if xs:
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return 0.0, 0.0


def _bbox_stats(person: dict, infer_w: float, infer_h: float) -> dict[str, float]:
    bbox = person.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return {
            "bbox_center_x_norm": 0.0,
            "bbox_center_y_norm": 0.0,
            "bbox_aspect_norm": 0.0,
            "bbox_area_norm": 0.0,
            "bbox_diag": 0.0,
        }
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    w = max(1e-3, x2 - x1)
    h = max(1e-3, y2 - y1)
    return {
        "bbox_center_x_norm": ((x1 + x2) / 2.0) / max(infer_w, 1.0),
        "bbox_center_y_norm": ((y1 + y2) / 2.0) / max(infer_h, 1.0),
        "bbox_aspect_norm": min(w / h, 2.5) / 2.5,
        "bbox_area_norm": (w * h) / max(infer_w * infer_h, 1.0),
        "bbox_diag": max(30.0, math.hypot(w, h)),
    }


def _torso_angle_norm(keypoints: list) -> float:
    ls = _valid_shape_pt(keypoints, LEFT_SHOULDER_IDX)
    rs = _valid_shape_pt(keypoints, RIGHT_SHOULDER_IDX)
    lh = _valid_shape_pt(keypoints, LEFT_HIP_IDX)
    rh = _valid_shape_pt(keypoints, RIGHT_HIP_IDX)
    if not (ls and rs and lh and rh):
        return 0.0
    sx = (ls[0] + rs[0]) / 2.0
    sy = (ls[1] + rs[1]) / 2.0
    hx = (lh[0] + rh[0]) / 2.0
    hy = (lh[1] + rh[1]) / 2.0
    dx, dy = hx - sx, hy - sy
    if abs(dx) + abs(dy) < 1e-6:
        return 0.0
    return abs(math.degrees(math.atan2(dx, dy))) / 90.0


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> float | None:
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    la = math.hypot(bax, bay)
    lc = math.hypot(bcx, bcy)
    if la < 1e-6 or lc < 1e-6:
        return None
    cosv = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / (la * lc)))
    return math.degrees(math.acos(cosv))


def _shape_features_for_person(person: dict, infer_w: float, infer_h: float) -> dict[str, float]:
    keypoints = person.get("keypoints") or []
    present_points = [
        kp
        for kp in (_pt(keypoints, idx) for idx in range(17))
        if kp is not None
    ]
    all_scores = [
        float(kp[2]) if len(kp) > 2 and kp[2] is not None else 0.0
        for kp in keypoints
        if isinstance(kp, (list, tuple)) and len(kp) >= 3
    ]
    point_ratio = len(present_points) / 17.0
    mean_confidence = sum(all_scores) / 17.0 if all_scores else 0.0
    quality_bad = 1.0 - (0.65 * point_ratio + 0.35 * mean_confidence)

    bbox = _bbox_stats(person, infer_w, infer_h)
    diag = bbox["bbox_diag"]
    distances: list[float] = []
    if diag > 0.0:
        for a_idx, b_idx in SPREAD_PAIRS:
            a = _valid_shape_pt(keypoints, a_idx)
            b = _valid_shape_pt(keypoints, b_idx)
            if a and b:
                distances.append(_dist(a, b) / diag)
    spread = sum(distances) / len(distances) if distances else 0.0
    limb_var = math.sqrt(sum((v - spread) ** 2 for v in distances) / len(distances)) if distances else 0.0

    folds: list[float] = []
    for a_idx, b_idx, c_idx in FOLD_TRIPLES:
        a = _valid_shape_pt(keypoints, a_idx)
        b = _valid_shape_pt(keypoints, b_idx)
        c = _valid_shape_pt(keypoints, c_idx)
        if a and b and c:
            angle = _angle(a, b, c)
            if angle is not None:
                folds.append((180.0 - angle) / 180.0)
    fold = sum(folds) / len(folds) if folds else 0.0

    return {
        "torso_angle_norm": _torso_angle_norm(keypoints),
        "spread": spread,
        "fold": fold,
        "limb_var": limb_var,
        "point_ratio": point_ratio,
        "mean_confidence": mean_confidence,
        "quality_bad": quality_bad,
        "bbox_center_y_norm": bbox["bbox_center_y_norm"],
        "bbox_center_x_norm": bbox["bbox_center_x_norm"],
        "bbox_aspect_norm": bbox["bbox_aspect_norm"],
        "bbox_area_norm": bbox["bbox_area_norm"],
    }


def _tracked_shape_features(ctx: FeatureContext, offset: int) -> dict[str, float] | None:
    frame = ctx.prior_frame(offset)
    if frame is None:
        return None
    track_id = ctx.current_track_id
    if track_id is None and ctx.current_person is not None:
        track_id = person_track_id(ctx.current_person)
    person = find_tracked_person_at_frame(ctx, track_id, frame)
    if person is None:
        return None
    return _shape_features_for_person(person, ctx.record.infer_width, ctx.record.infer_height)


class SkeletonFeatureExtractor(FeatureExtractor):
    name = "skeleton"

    def frame_feature_names(self) -> list[str]:
        return [
            "person_count",
            "wrist_min_score",
            "left_wrist_x_norm",
            "left_wrist_y_norm",
            "right_wrist_x_norm",
            "right_wrist_y_norm",
            "wrist_spread",
            "anchor_x_norm",
            "anchor_y_norm",
            "torso_angle_norm",
            "spread",
            "fold",
            "limb_var",
            "point_ratio",
            "mean_confidence",
            "quality_bad",
            "bbox_center_x_norm",
            "bbox_center_y_norm",
            "bbox_aspect_norm",
            "bbox_area_norm",
            "bbox_delta_y_5",
            "bbox_delta_y_10",
            "bbox_motion_5",
            "bbox_motion_10",
            "torso_angle_5_abs",
            "torso_angle_10_abs",
            "spread_5_abs",
            "spread_10_abs",
            "fold_5_abs",
            "fold_10_abs",
            "limb_var_5_abs",
            "limb_var_10_abs",
        ]

    def extract_frame(self, ctx: FeatureContext) -> dict[str, float]:
        persons = ctx.frame.persons
        out: dict[str, float] = {
            "person_count": float(len(persons)),
        }
        if not persons:
            out.update(
                {
                    "wrist_min_score": 0.0,
                    "left_wrist_x_norm": 0.0,
                    "left_wrist_y_norm": 0.0,
                    "right_wrist_x_norm": 0.0,
                    "right_wrist_y_norm": 0.0,
                    "wrist_spread": 0.0,
                    "anchor_x_norm": 0.0,
                    "anchor_y_norm": 0.0,
                    "torso_angle_norm": 0.0,
                    "spread": 0.0,
                    "fold": 0.0,
                    "limb_var": 0.0,
                    "point_ratio": 0.0,
                    "mean_confidence": 0.0,
                    "quality_bad": 1.0,
                    "bbox_center_x_norm": 0.0,
                    "bbox_center_y_norm": 0.0,
                    "bbox_aspect_norm": 0.0,
                    "bbox_area_norm": 0.0,
                    "bbox_delta_y_5": 0.0,
                    "bbox_delta_y_10": 0.0,
                    "bbox_motion_5": 0.0,
                    "bbox_motion_10": 0.0,
                    "torso_angle_5_abs": 0.0,
                    "torso_angle_10_abs": 0.0,
                    "spread_5_abs": 0.0,
                    "spread_10_abs": 0.0,
                    "fold_5_abs": 0.0,
                    "fold_10_abs": 0.0,
                    "limb_var_5_abs": 0.0,
                    "limb_var_10_abs": 0.0,
                }
            )
            return out

        best_wrist_score = -1.0
        best_feats: dict[str, float] | None = None
        iw = max(ctx.record.infer_width, 1.0)
        ih = max(ctx.record.infer_height, 1.0)

        for person in persons:
            kpts = person.get("keypoints") or []
            lw = _pt(kpts, LEFT_WRIST_IDX)
            rw = _pt(kpts, RIGHT_WRIST_IDX)
            wrist_scores = [p[2] for p in (lw, rw) if p]
            min_score = min((p[2] for p in (lw, rw) if p), default=0.0)
            if min_score <= best_wrist_score:
                continue
            best_wrist_score = min_score
            ax, ay = _person_anchor(kpts)
            lx = lw[0] if lw else 0.0
            ly = lw[1] if lw else 0.0
            rx = rw[0] if rw else 0.0
            ry = rw[1] if rw else 0.0
            spread = math.hypot(rx - lx, ry - ly) if lw and rw else 0.0
            best_feats = {
                "wrist_min_score": float(min(wrist_scores) if wrist_scores else 0.0),
                "left_wrist_x_norm": lx / iw,
                "left_wrist_y_norm": ly / ih,
                "right_wrist_x_norm": rx / iw,
                "right_wrist_y_norm": ry / ih,
                "wrist_spread": spread / max(iw, ih),
                "anchor_x_norm": ax / iw,
                "anchor_y_norm": ay / ih,
            }

        out.update(best_feats or {})
        current_person = ctx.current_person if ctx.current_person is not None else persons[0]
        shape = _shape_features_for_person(current_person, ctx.record.infer_width, ctx.record.infer_height)
        out.update(shape)
        for offset in SHAPE_OFFSETS:
            prev = _tracked_shape_features(ctx, offset)
            out[f"bbox_delta_y_{offset}"] = (
                shape["bbox_center_y_norm"] - prev["bbox_center_y_norm"] if prev is not None else 0.0
            )
            out[f"bbox_motion_{offset}"] = _bbox_motion(shape, prev) if prev is not None else 0.0
            for key, base in (
                ("torso_angle", "torso_angle_norm"),
                ("spread", "spread"),
                ("fold", "fold"),
                ("limb_var", "limb_var"),
            ):
                out[f"{key}_{offset}_abs"] = abs(shape[base] - prev[base]) if prev is not None else 0.0
        return out


def _bbox_motion(current: dict[str, float], previous: dict[str, float] | None) -> float:
    if previous is None:
        return 0.0
    return math.hypot(
        current["bbox_center_x_norm"] - previous["bbox_center_x_norm"],
        current["bbox_center_y_norm"] - previous["bbox_center_y_norm"],
    )
