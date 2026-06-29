"""骨骼统计特征。"""

from __future__ import annotations

import math

from analysis.constants import LEFT_SHOULDER_IDX, LEFT_WRIST_IDX, RIGHT_SHOULDER_IDX, RIGHT_WRIST_IDX
from analysis.features.base import FeatureContext, FeatureExtractor


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


class SkeletonFeatureExtractor(FeatureExtractor):
    name = "skeleton"

    def extract_frame(self, ctx: FeatureContext) -> dict[str, float]:
        persons = ctx.frame.persons
        out: dict[str, float] = {
            "person_count": float(len(persons)),
            "infer_width": float(ctx.record.infer_width),
            "infer_height": float(ctx.record.infer_height),
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
        return out
