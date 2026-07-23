"""聚合速度特征单测。"""

from __future__ import annotations

from pathlib import Path

from fixtures import make_fixture_record


def test_speed_feature_names_registered():
    from analysis.features.registry import default_registry

    names = default_registry().frame_feature_names()
    for key in (
        "speed.ankle_max_speed",
        "speed.lower_mean_speed",
        "speed.wrist_max_speed",
    ):
        assert key in names


def test_speed_features_on_fixture(tmp_path: Path):
    from analysis.dataset import stride_frames
    from analysis.features.base import FeatureContext
    from analysis.features.speed import SpeedFeatureExtractor
    from analysis.records import load_record

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    record = load_record(fixture_dir)
    samples = stride_frames(record.frames(), 2)
    ext = SpeedFeatureExtractor()

    feats = None
    for frame in samples[2:]:
        ctx = FeatureContext.from_record(record, frame, sample_frames=samples)
        person = frame.persons[0]
        person_ctx = ctx.for_person(person, person.get("person_track_id", 1))
        row = ext.extract_frame(person_ctx)
        if row["ankle_max_speed"] > 0.0 or row["wrist_max_speed"] > 0.0:
            feats = row
            break

    assert feats is not None
    assert feats["ankle_max_speed"] >= 0.0
    assert feats["lower_mean_speed"] >= 0.0
    assert feats["wrist_max_speed"] >= 0.0
