"""特征提取单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from fixtures import make_fixture_record


@pytest.fixture
def fixture_data_dir(tmp_path: Path) -> Path:
    return make_fixture_record(tmp_path / "record_001")


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
