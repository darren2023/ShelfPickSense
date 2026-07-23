"""特征提取单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from fixtures import make_fixture_record


@pytest.fixture
def fixture_data_dir(tmp_path: Path) -> Path:
    return make_fixture_record(tmp_path / "record_001")


def _first_frame_group(reg, record, frame):
    from analysis.features.base import FeatureContext

    return reg.extract_frame_feature_groups_from_context(
        FeatureContext.from_record(record, frame, frame_index=record.frame_index())
    )[0]


def test_layout_features_on_picking_frame(fixture_data_dir: Path):
    from analysis.features.registry import default_registry
    from analysis.records import load_record

    record = load_record(fixture_data_dir)
    reg = default_registry()
    frame = next(f for f in record.frames() if f.frame_idx == 6)
    per_box = reg.extract_per_box_features(record, frame)
    a1 = next(pb for pb in per_box if pb.box_token == "S1:A1")

    assert a1.features["layout.shelf_side"] == pytest.approx(1.0)
    assert a1.features["layout.layout_layer"] == pytest.approx(1.0)
    assert a1.features["layout.layout_column"] == pytest.approx(2.0)
    assert a1.features["layout.shelf_layer_count"] == pytest.approx(1.0)
    assert a1.features["layout.shelf_column_count_mean"] == pytest.approx(2.0)


def test_layout_supervision_in_dataset(fixture_data_dir: Path):
    from analysis.dataset import build_dataset
    from analysis.features.registry import default_registry
    from analysis.records import load_record

    record = load_record(fixture_data_dir)
    dataset = build_dataset([record], default_registry())
    frame = next(s for s in dataset.frame_samples if s.frame_idx == 6)
    assert frame.is_picking
    assert frame.target_layout_layer_norm == pytest.approx(0.5)
    assert frame.target_layout_column_norm == pytest.approx(2 / 3)

    target = next(s for s in dataset.box_samples if s.box_token == "S1:A1")
    assert target.is_target
    assert target.target_layout_layer_norm == pytest.approx(0.5)
    assert target.target_layout_column_norm == pytest.approx(2 / 3)


def test_spatial_wrist_and_foot_distance(fixture_data_dir: Path):
    from analysis.features.registry import default_registry
    from analysis.records import load_record

    record = load_record(fixture_data_dir)
    reg = default_registry()
    frame = next(f for f in record.frames() if f.frame_idx == 6)
    per_box = reg.extract_per_box_features(record, frame)
    a1 = next(pb for pb in per_box if pb.box_token == "S1:A1")

    assert a1.features["spatial.wrist_inside"] == pytest.approx(1.0)
    assert a1.features["spatial.left_wrist_inside"] == pytest.approx(1.0)
    assert a1.features["spatial.foot_min_dist_norm"] == pytest.approx(1.0)

    frame_feat = _first_frame_group(reg, record, frame)
    assert frame_feat.features["spatial.p0_present"] == pytest.approx(1.0)
    assert frame_feat.features["spatial.p0_track_id"] == pytest.approx(1.0)
    assert frame_feat.features["spatial.p0_left_wrist_min_box_dist_norm"] == pytest.approx(0.0)
    assert frame_feat.features["spatial.p0_left_foot_min_box_dist_norm"] == pytest.approx(1.0)


def test_frame_feature_groups_are_not_merged_into_default_frame_features():
    import pandas as pd

    from analysis.constants import LEFT_WRIST_IDX, RIGHT_WRIST_IDX
    from analysis.features.base import FeatureContext
    from analysis.features.registry import default_registry
    from analysis.records import FramePersons, RecordData

    def _person(track_id: int, left: tuple[float, float], right: tuple[float, float]) -> dict:
        keypoints: list[list[float | None]] = [[None, None, None] for _ in range(17)]
        keypoints[LEFT_WRIST_IDX] = [left[0], left[1], 0.9]
        keypoints[RIGHT_WRIST_IDX] = [right[0], right[1], 0.8]
        return {"person_track_id": track_id, "keypoints": keypoints}

    record = RecordData(
        record_id="multi_person",
        record_dir=Path("."),
        skeleton=pd.DataFrame(),
        annotation={},
        event_review=None,
        labels=__import__("analysis.labels", fromlist=["RecordLabels"]).RecordLabels(record_id="multi_person"),
        infer_width=640.0,
        infer_height=480.0,
        box_tokens=[],
    )
    frame = FramePersons(
        frame_idx=1,
        timestamp_sec=0.0,
        persons=[
            _person(2, (200.0, 100.0), (240.0, 100.0)),
            _person(1, (100.0, 100.0), (140.0, 100.0)),
        ],
    )

    reg = default_registry()
    groups = reg.extract_frame_feature_groups_from_context(FeatureContext.from_record(record, frame))

    assert len(groups) == 2
    assert "skeleton.infer_width" not in groups[0].features
    assert "skeleton.infer_height" not in groups[0].features
    assert not any(name.startswith("person") for name in groups[0].features)
    assert groups[0].person_track_id == 1
    assert groups[1].person_track_id == 2
    assert "track_id" not in groups[0].features
    assert "track_id" not in groups[1].features
    assert groups[0].features["skeleton.person_count"] == pytest.approx(1.0)
    assert groups[1].features["skeleton.person_count"] == pytest.approx(1.0)
    assert groups[0].features["skeleton.left_wrist_x_norm"] == pytest.approx(100.0 / 640.0)
    assert groups[1].features["skeleton.left_wrist_x_norm"] == pytest.approx(200.0 / 640.0)


def test_skeleton_shape_features_and_window_deltas():
    import math
    import pandas as pd

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
    from analysis.features.base import FeatureContext
    from analysis.features.registry import default_registry
    from analysis.records import FramePersons, RecordData

    def _person(*, hip_shift_x: float, bbox_y_shift: float) -> dict:
        keypoints: list[list[float | None]] = [[None, None, None] for _ in range(17)]
        coords = {
            LEFT_SHOULDER_IDX: (100.0, 100.0),
            RIGHT_SHOULDER_IDX: (140.0, 100.0),
            LEFT_ELBOW_IDX: (90.0, 125.0),
            RIGHT_ELBOW_IDX: (150.0, 125.0),
            LEFT_WRIST_IDX: (80.0, 150.0),
            RIGHT_WRIST_IDX: (160.0, 150.0),
            LEFT_HIP_IDX: (100.0 + hip_shift_x, 200.0),
            RIGHT_HIP_IDX: (140.0 + hip_shift_x, 200.0),
            LEFT_KNEE_IDX: (100.0 + hip_shift_x, 250.0),
            RIGHT_KNEE_IDX: (140.0 + hip_shift_x, 250.0),
            LEFT_ANKLE_IDX: (100.0 + hip_shift_x, 300.0),
            RIGHT_ANKLE_IDX: (140.0 + hip_shift_x, 300.0),
        }
        for idx, (x, y) in coords.items():
            keypoints[idx] = [x, y, 0.9]
        return {
            "person_track_id": 1,
            "keypoints": keypoints,
            "bbox": [70.0, 80.0 + bbox_y_shift, 170.0, 320.0 + bbox_y_shift],
        }

    record = RecordData(
        record_id="shape_test",
        record_dir=Path("."),
        skeleton=pd.DataFrame(),
        annotation={},
        event_review=None,
        labels=__import__("analysis.labels", fromlist=["RecordLabels"]).RecordLabels(record_id="shape_test"),
        infer_width=640.0,
        infer_height=480.0,
        box_tokens=[],
    )
    frames = {
        fi: FramePersons(
            frame_idx=fi,
            timestamp_sec=fi / 25.0,
            persons=[_person(hip_shift_x=60.0 if fi == 1 else 0.0, bbox_y_shift=-10.0 if fi == 1 else 0.0)],
        )
        for fi in range(1, 7)
    }
    sample_frames = [frames[i] for i in range(1, 7)]
    ctx = FeatureContext.from_record(record, frames[6], sample_frames=sample_frames)
    feat = default_registry().extract_frame_feature_groups_from_context(ctx)[0].features

    previous_torso = abs(math.degrees(math.atan2(60.0, 100.0))) / 90.0
    assert feat["skeleton.torso_angle_norm"] == pytest.approx(0.0)
    assert feat["skeleton.torso_angle_5_abs"] == pytest.approx(previous_torso)
    assert feat["skeleton.spread"] > 0.0
    assert feat["skeleton.fold"] >= 0.0
    assert feat["skeleton.limb_var"] >= 0.0
    assert feat["skeleton.point_ratio"] == pytest.approx(12 / 17)
    assert feat["skeleton.mean_confidence"] == pytest.approx(12 * 0.9 / 17)
    assert feat["skeleton.quality_bad"] == pytest.approx(
        1.0 - (0.65 * (12 / 17) + 0.35 * (12 * 0.9 / 17))
    )
    assert feat["skeleton.bbox_center_x_norm"] == pytest.approx(120.0 / 640.0)
    assert feat["skeleton.bbox_center_y_norm"] == pytest.approx(200.0 / 480.0)
    assert feat["skeleton.bbox_aspect_norm"] == pytest.approx((100.0 / 240.0) / 2.5)
    assert feat["skeleton.bbox_area_norm"] == pytest.approx((100.0 * 240.0) / (640.0 * 480.0))
    assert feat["skeleton.bbox_delta_y_5"] == pytest.approx(10.0 / 480.0)
    assert feat["skeleton.bbox_motion_5"] == pytest.approx(10.0 / 480.0)


def test_foot_temporal_movement_features():
    from analysis.constants import LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX
    from analysis.features.base import FeatureContext
    from analysis.features.registry import default_registry
    from analysis.records import FramePersons, RecordData
    from pathlib import Path
    import pandas as pd

    def _person(*, track_id: int, left_ankle: tuple[float, float], right_ankle: tuple[float, float]) -> dict:
        keypoints: list[list[float | None]] = [[None, None, None] for _ in range(17)]
        keypoints[LEFT_ANKLE_IDX] = [left_ankle[0], left_ankle[1], 0.95]
        keypoints[RIGHT_ANKLE_IDX] = [right_ankle[0], right_ankle[1], 0.95]
        return {"person_track_id": track_id, "keypoints": keypoints}

    record = RecordData(
        record_id="foot_test",
        record_dir=Path("."),
        skeleton=pd.DataFrame(),
        annotation={},
        event_review=None,
        labels=__import__("analysis.labels", fromlist=["RecordLabels"]).RecordLabels(record_id="foot_test"),
        infer_width=640.0,
        infer_height=480.0,
        box_tokens=[],
    )
    frames = {
        1: FramePersons(frame_idx=1, timestamp_sec=0.0, persons=[_person(track_id=1, left_ankle=(100.0, 400.0), right_ankle=(140.0, 400.0))]),
        2: FramePersons(frame_idx=2, timestamp_sec=0.04, persons=[_person(track_id=1, left_ankle=(160.0, 400.0), right_ankle=(200.0, 400.0))]),
    }
    ctx = FeatureContext(
        record=record,
        frame=frames[2],
        box_index={},
        box_tokens=[],
        frame_index=frames,
    )
    reg = default_registry()
    feat = reg.extract_frame_feature_groups_from_context(ctx)[0].features

    assert feat["temporal.left_foot_x_norm"] == pytest.approx(160.0 / 640.0)
    assert feat["temporal.foot_avg_x_norm"] == pytest.approx(180.0 / 640.0)
    assert feat["temporal.left_foot_dx_1"] == pytest.approx(60.0 / 640.0)
    assert feat["temporal.left_foot_dy_1"] == pytest.approx(0.0)
    assert feat["temporal.left_foot_dist_1"] == pytest.approx(60.0 / 640.0)
    assert feat["temporal.foot_avg_dist_1"] == pytest.approx(60.0 / 640.0)
    assert feat["temporal.left_foot_dist_2"] == pytest.approx(0.0)


def test_temporal_consecutive_hit_and_hand_move(fixture_data_dir: Path):
    from analysis.features.registry import default_registry
    from analysis.records import load_record

    record = load_record(fixture_data_dir)
    reg = default_registry()

    frame6 = next(f for f in record.frames() if f.frame_idx == 6)
    frame8 = next(f for f in record.frames() if f.frame_idx == 8)

    feat6 = _first_frame_group(reg, record, frame6)
    assert feat6.features["temporal.consecutive_hit_3"] == pytest.approx(0.0)
    assert feat6.features["temporal.left_wrist_move_1"] >= 0.0
    assert feat6.features["temporal.right_wrist_move_1"] >= 0.0

    feat8 = _first_frame_group(reg, record, frame8)
    assert feat8.features["temporal.consecutive_hit_3"] == pytest.approx(1.0)
    assert feat8.features["temporal.consecutive_hit_5"] == pytest.approx(0.0)
    assert feat8.features["temporal.p0_consecutive_hit_3"] == pytest.approx(1.0)


def test_track_id_matches_across_frames(fixture_data_dir: Path):
    from analysis.features.registry import default_registry
    from analysis.records import load_record

    record = load_record(fixture_data_dir)
    reg = default_registry()
    frame7 = next(f for f in record.frames() if f.frame_idx == 7)
    feat7 = _first_frame_group(reg, record, frame7)
    assert feat7.features["temporal.left_wrist_move_1"] == pytest.approx(0.0)
    assert feat7.features["temporal.right_wrist_move_1"] == pytest.approx(0.0)


def test_realtime_temporal_history(tmp_path: Path):
    from analysis.realtime import RealtimePickingPredictor
    from analysis.records import load_record
    from analysis.train import train_model

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    model_dir = tmp_path / "model"
    train_model(fixture_dir, model_dir)

    record = load_record(fixture_dir)
    predictor = RealtimePickingPredictor.from_record_dir(model_dir=model_dir, record_dir=fixture_dir)

    for frame in record.frames():
        if frame.frame_idx > 8:
            break
        pred = predictor.predict_frame(frame.persons, frame_idx=frame.frame_idx)
        assert 0.0 <= pred.picking_prob <= 1.0

    assert len(predictor._frame_history) <= 8


def test_rule_engine_collision_and_window_features(fixture_data_dir: Path):
    from analysis.features.registry import default_registry
    from analysis.records import load_record

    record = load_record(fixture_data_dir)
    reg = default_registry()

    frame6 = next(f for f in record.frames() if f.frame_idx == 6)
    frame8 = next(f for f in record.frames() if f.frame_idx == 8)

    feat6 = _first_frame_group(reg, record, frame6)
    assert feat6.features["rule.any_collision"] == pytest.approx(1.0)
    assert feat6.features["rule.primary_any_collision"] == pytest.approx(1.0)
    assert feat6.features["rule.window_hit_3_6"] == pytest.approx(0.0)

    per_box6 = reg.extract_per_box_features(record, frame6)
    a1 = next(pb for pb in per_box6 if pb.box_token == "S1:A1")
    assert a1.features["rule.hand_collision"] == pytest.approx(1.0)
    assert a1.features["rule.wrist_collision"] == pytest.approx(1.0)
    assert a1.features["rule.frame_collision"] == pytest.approx(1.0)

    feat8 = _first_frame_group(reg, record, frame8)
    assert feat8.features["rule.window_hit_3_6"] == pytest.approx(1.0)
    assert feat8.features["rule.window_hits_6"] >= 3.0


def test_box_hand_collision_flags_respects_hand_points(fixture_data_dir: Path):
    from analysis.features.base import FeatureContext
    from analysis.features.rule_engine import RuleEngineParams, _box_hand_collision_flags
    from analysis.records import load_record

    record = load_record(fixture_data_dir)
    frame = next(f for f in record.frames() if f.frame_idx == 6)
    person = frame.persons[0]
    ctx = FeatureContext.from_record(record, frame)
    params = RuleEngineParams()

    wrist_hit, forearm_hit, hand_hit, _signed = _box_hand_collision_flags(
        person,
        "S1:A1",
        ctx,
        params,
        hand_points=[],
        margin=10.0,
    )
    assert wrist_hit == pytest.approx(0.0)
    assert forearm_hit == pytest.approx(0.0)
    assert hand_hit == pytest.approx(0.0)


def test_cross_frame_features_without_track_id(tmp_path: Path):
    import shutil

    import pandas as pd

    from analysis.features.base import FeatureContext
    from analysis.features.registry import default_registry
    from analysis.features.rule_engine import RuleEngineParams, window_hit_count_for_track
    from analysis.features.temporal import consecutive_hit_streak_for_track
    from analysis.features.tracking import LEFT_WRIST, side_movement_norm
    from analysis.records import load_record

    fixture_dir = make_fixture_record(tmp_path / "record_no_track")
    skeleton = pd.read_parquet(fixture_dir / "skeleton.parquet")
    skeleton = skeleton.drop(columns=["person_track_id"], errors="ignore")
    skeleton.to_parquet(fixture_dir / "skeleton.parquet", index=False)

    record = load_record(fixture_dir)
    frames = record.frame_index()
    ctx8 = FeatureContext.from_record(record, frames[8], frame_index=frames)

    streak = consecutive_hit_streak_for_track(ctx8, None)
    assert streak >= 3

    hits = window_hit_count_for_track(ctx8, None, RuleEngineParams(), window=6)
    assert hits >= 3

    move = side_movement_norm(ctx8, track_id=None, side=LEFT_WRIST, offset=1)
    assert move == pytest.approx(0.0)

    reg = default_registry()
    feat8 = _first_frame_group(reg, record, frames[8])
    assert feat8.features["temporal.consecutive_hit_3"] == pytest.approx(1.0)
    assert feat8.features["rule.window_hit_3_6"] == pytest.approx(1.0)
