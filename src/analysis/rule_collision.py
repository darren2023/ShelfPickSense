"""规则碰撞检测（与线上 event_engine 对齐，无 cv2 依赖）。"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from analysis.annotation import BoxInfo, build_box_index
from analysis.constants import LEFT_HIP_IDX, LEFT_SHOULDER_IDX, RIGHT_HIP_IDX, RIGHT_SHOULDER_IDX
from analysis.features.rule_engine import (
    RuleEngineParams,
    collect_rule_hand_points,
    nearest_box_token_for_point,
)
from analysis.features.tracking import get_keypoint

_TORSO_IDX = (LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX, LEFT_HIP_IDX, RIGHT_HIP_IDX)


@dataclass
class CollisionParams:
    min_consecutive_frames: int = 3
    cooldown_frames: int = 6
    window_frames: int = 6
    wrist_conf: float = 0.3
    elbow_conf: float = 0.3
    forearm_extend_ratio: float = 0.4
    boundary_margin_ratio: float = 0.12
    boundary_margin_min_px: float = 8.0
    track_max_match_dist: float = 220.0
    track_stale_sec: float = 1.2
    per_track_gating: bool = True

    @classmethod
    def from_rule_engine_params(cls, params: RuleEngineParams) -> "CollisionParams":
        return cls(
            min_consecutive_frames=params.min_consecutive_frames,
            window_frames=params.window_frames,
            wrist_conf=params.wrist_conf,
            elbow_conf=params.elbow_conf,
            forearm_extend_ratio=params.forearm_extend_ratio,
            boundary_margin_ratio=params.boundary_margin_ratio,
            boundary_margin_min_px=params.boundary_margin_min_px,
        )


@dataclass
class TrackState:
    x: float
    y: float
    vx: float
    vy: float
    ts_sec: float


class PersonTrackAssigner:
    """恒速卡尔曼式预测 + 全局贪心匹配。"""

    def __init__(self, max_match_dist: float = 220.0, stale_sec: float = 1.2, vel_alpha: float = 0.5):
        self.max_match_dist = float(max_match_dist)
        self.stale_sec = float(stale_sec)
        self.vel_alpha = float(vel_alpha)
        self.next_id = 1
        self.tracks: dict[int, TrackState] = {}

    def _cleanup(self, now_ts: float) -> None:
        dead = [k for k, st in self.tracks.items() if now_ts - st.ts_sec > self.stale_sec]
        for k in dead:
            self.tracks.pop(k, None)

    def _predict(self, st: TrackState, now_ts: float) -> tuple[float, float]:
        dt = max(0.0, now_ts - st.ts_sec)
        return st.x + st.vx * dt, st.y + st.vy * dt

    def assign_batch(self, detections: list[tuple[float, float]], now_ts: float) -> list[int]:
        self._cleanup(now_ts)
        result: list[int | None] = [None] * len(detections)

        preds = {tid: self._predict(st, now_ts) for tid, st in self.tracks.items()}
        pairs: list[tuple[float, int, int]] = []
        for di, (dx, dy) in enumerate(detections):
            for tid, (px, py) in preds.items():
                dist = math.hypot(dx - px, dy - py)
                if dist <= self.max_match_dist:
                    pairs.append((dist, di, tid))
        pairs.sort(key=lambda p: p[0])

        used_det: set[int] = set()
        used_tid: set[int] = set()
        for _dist, di, tid in pairs:
            if di in used_det or tid in used_tid:
                continue
            used_det.add(di)
            used_tid.add(tid)
            result[di] = tid
            self._update_track(tid, detections[di], now_ts)

        for di, tid in enumerate(result):
            if tid is None:
                new_tid = self.next_id
                self.next_id += 1
                dx, dy = detections[di]
                self.tracks[new_tid] = TrackState(x=dx, y=dy, vx=0.0, vy=0.0, ts_sec=now_ts)
                result[di] = new_tid
        return result  # type: ignore[return-value]

    def _update_track(self, tid: int, det: tuple[float, float], now_ts: float) -> None:
        st = self.tracks[tid]
        dt = now_ts - st.ts_sec
        dx, dy = det
        if dt > 1e-3:
            vx = (dx - st.x) / dt
            vy = (dy - st.y) / dt
            st.vx = (1 - self.vel_alpha) * st.vx + self.vel_alpha * vx
            st.vy = (1 - self.vel_alpha) * st.vy + self.vel_alpha * vy
        st.x, st.y, st.ts_sec = dx, dy, now_ts


def _rule_engine_params(params: CollisionParams) -> RuleEngineParams:
    return RuleEngineParams(
        wrist_conf=params.wrist_conf,
        elbow_conf=params.elbow_conf,
        forearm_extend_ratio=params.forearm_extend_ratio,
        boundary_margin_ratio=params.boundary_margin_ratio,
        boundary_margin_min_px=params.boundary_margin_min_px,
        min_consecutive_frames=params.min_consecutive_frames,
        window_frames=params.window_frames,
    )


def _anchor(person: dict) -> tuple[float, float, float]:
    sx = sy = wsum = 0.0
    for idx in _TORSO_IDX:
        kp = get_keypoint(person, idx, min_score=0.0)
        if kp is None:
            continue
        x, y, score = kp
        if score <= 0.0:
            continue
        sx += x * score
        sy += y * score
        wsum += score
    left = get_keypoint(person, LEFT_SHOULDER_IDX, min_score=0.2)
    right = get_keypoint(person, RIGHT_SHOULDER_IDX, min_score=0.2)
    shoulder_width = math.hypot(left[0] - right[0], left[1] - right[1]) if left and right else 0.0
    if wsum > 0:
        return sx / wsum, sy / wsum, shoulder_width
    return 0.0, 0.0, shoulder_width


class RuleCollisionProcessor:
    """逐帧规则碰撞检测 + M-of-N 报警门控。"""

    def __init__(
        self,
        box_index: dict[str, BoxInfo],
        *,
        params: CollisionParams | None = None,
        video_fps: float = 25.0,
    ) -> None:
        self.params = params or CollisionParams()
        self.box_index = box_index
        self.video_fps = max(1.0, float(video_fps))
        self._rule_params = _rule_engine_params(self.params)

        self.person_assigner = PersonTrackAssigner(
            max_match_dist=self.params.track_max_match_dist,
            stale_sec=self.params.track_stale_sec,
        )
        self._hit_history: dict[tuple, deque] = {}
        self._last_alarm_frame: dict[tuple, int] = {}

    def _resolve_now_ts(self, frame_idx: int, timestamp_sec: float | None) -> float:
        if timestamp_sec is not None:
            return float(timestamp_sec)
        return frame_idx / self.video_fps if self.video_fps > 0 else 0.0

    def process_frame(
        self,
        *,
        frame_idx: int,
        persons: list[dict],
        timestamp_sec: float | None = None,
    ) -> dict:
        now_ts = self._resolve_now_ts(frame_idx, timestamp_sec)

        parsed: list[dict] = []
        anchors: list[tuple[float, float]] = []
        for person in persons:
            if not isinstance(person, dict):
                continue
            keypoints = person.get("keypoints") or []
            if len(keypoints) < 11:
                parsed.append({"person": person, "valid": False})
                continue
            ax, ay, shoulder_width = _anchor(person)
            parsed.append(
                {
                    "person": person,
                    "valid": True,
                    "shoulder_width": shoulder_width,
                    "hand_points": collect_rule_hand_points(person, self._rule_params),
                }
            )
            anchors.append((ax, ay))

        track_ids = self.person_assigner.assign_batch(anchors, now_ts) if anchors else []

        current_pairs: set[tuple] = set()
        active_tokens: set[str] = set()
        ai = 0
        for item in parsed:
            if not item["valid"]:
                ai += 1
                continue
            track_id = track_ids[ai] if ai < len(track_ids) else -1
            ai += 1

            hand_points = item["hand_points"]
            if not hand_points:
                continue
            margin = max(
                self.params.boundary_margin_min_px,
                self.params.boundary_margin_ratio * item["shoulder_width"],
            )
            gate_track = track_id if self.params.per_track_gating else None
            for px, py, _kind in hand_points:
                token = nearest_box_token_for_point(px, py, margin, self.box_index)
                if token:
                    active_tokens.add(token)
                    current_pairs.add((gate_track, token))

        alarm_collisions = self._gate(current_pairs, frame_idx)
        return {
            "collisions": sorted(active_tokens),
            "alarm_collisions": alarm_collisions,
            "frame_idx": frame_idx,
        }

    def _gate(self, current_pairs: set[tuple], frame_idx: int) -> list[str]:
        alarm: list[str] = []
        window = self.params.window_frames
        min_hits = self.params.min_consecutive_frames
        for key in set(self._hit_history.keys()) | current_pairs:
            hist = self._hit_history.get(key)
            if hist is None:
                hist = deque(maxlen=window)
                self._hit_history[key] = hist
            hist.append(1 if key in current_pairs else 0)

            window_hits = sum(hist)
            if key not in current_pairs:
                if window_hits == 0:
                    self._hit_history.pop(key, None)
                continue

            if window_hits >= min_hits:
                last_alarm = self._last_alarm_frame.get(key, -(10**9))
                if frame_idx - last_alarm >= self.params.cooldown_frames:
                    alarm.append(key[1])
                    self._last_alarm_frame[key] = frame_idx
        return alarm


def build_box_index_for_record(record) -> dict[str, BoxInfo]:
    return build_box_index(
        record.annotation,
        infer_w=record.infer_width,
        infer_h=record.infer_height,
    )
