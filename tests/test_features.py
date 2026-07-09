"""特征提取单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from fixtures import make_fixture_record


@pytest.fixture
def fixture_data_dir(tmp_path: Path) -> Path:
    return make_fixture_record(tmp_path / "record_001")


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

    frame_feat = reg.extract_frame_features(record, frame)
    assert frame_feat.features["spatial.p0_present"] == pytest.approx(1.0)
    assert frame_feat.features["spatial.p0_track_id"] == pytest.approx(1.0)
    assert frame_feat.features["spatial.p0_left_wrist_min_box_dist_norm"] == pytest.approx(0.0)
    assert frame_feat.features["spatial.p0_left_foot_min_box_dist_norm"] == pytest.approx(1.0)


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
    feat = reg.extract_frame_features_from_context(ctx).features

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

    feat6 = reg.extract_frame_features(record, frame6)
    assert feat6.features["temporal.consecutive_hit_3"] == pytest.approx(0.0)
    assert feat6.features["temporal.left_wrist_move_1"] >= 0.0
    assert feat6.features["temporal.right_wrist_move_1"] >= 0.0

    feat8 = reg.extract_frame_features(record, frame8)
    assert feat8.features["temporal.consecutive_hit_3"] == pytest.approx(1.0)
    assert feat8.features["temporal.consecutive_hit_5"] == pytest.approx(0.0)
    assert feat8.features["temporal.p0_consecutive_hit_3"] == pytest.approx(1.0)


def test_track_id_matches_across_frames(fixture_data_dir: Path):
    from analysis.features.registry import default_registry
    from analysis.records import load_record

    record = load_record(fixture_data_dir)
    reg = default_registry()
    frame7 = next(f for f in record.frames() if f.frame_idx == 7)
    feat7 = reg.extract_frame_features(record, frame7)
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

    feat6 = reg.extract_frame_features(record, frame6)
    assert feat6.features["rule.any_collision"] == pytest.approx(1.0)
    assert feat6.features["rule.primary_any_collision"] == pytest.approx(1.0)
    assert feat6.features["rule.window_hit_3_6"] == pytest.approx(0.0)

    per_box6 = reg.extract_per_box_features(record, frame6)
    a1 = next(pb for pb in per_box6 if pb.box_token == "S1:A1")
    assert a1.features["rule.hand_collision"] == pytest.approx(1.0)
    assert a1.features["rule.wrist_collision"] == pytest.approx(1.0)
    assert a1.features["rule.frame_collision"] == pytest.approx(1.0)

    feat8 = reg.extract_frame_features(record, frame8)
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
    feat8 = reg.extract_frame_features(record, frames[8])
    assert feat8.features["temporal.consecutive_hit_3"] == pytest.approx(1.0)
    assert feat8.features["rule.window_hit_3_6"] == pytest.approx(1.0)
