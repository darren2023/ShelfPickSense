"""数据集构建与过滤测试。"""

from __future__ import annotations

from pathlib import Path

from fixtures import make_fixture_record, make_fixture_record_with_empty_skeleton_frames


def test_frame_has_valid_skeleton(tmp_path: Path):
    from analysis.dataset import frame_has_valid_skeleton
    from analysis.records import load_record

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    record = load_record(fixture_dir)
    frame = record.frames()[0]
    assert frame_has_valid_skeleton(frame) is True


def test_filter_empty_skeleton_frames_before_training(tmp_path: Path):
    from analysis.dataset import build_dataset, filter_empty_skeleton_frames
    from analysis.features.registry import default_registry
    from analysis.records import load_record
    from analysis.train import train_model

    fixture_dir = make_fixture_record_with_empty_skeleton_frames(
        tmp_path / "record_001",
        empty_frame_indices=[11, 12, 13],
    )
    record = load_record(fixture_dir)
    assert len(record.frames()) == 13

    reg = default_registry()
    dataset = build_dataset([record], reg)
    assert dataset.frame_count == 13

    filtered, removed = filter_empty_skeleton_frames(dataset, [record])
    assert removed == 3
    assert filtered.frame_count == 10
    assert filtered.positive_frame_count == 3

    result = train_model(fixture_dir, tmp_path / "model")
    assert result.frame_count == 10
    assert result.skipped_empty_skeleton_frames == 3


def test_keep_empty_skeleton_frames_option(tmp_path: Path):
    from analysis.train import train_model

    fixture_dir = make_fixture_record_with_empty_skeleton_frames(
        tmp_path / "record_001",
        empty_frame_indices=[11, 12],
    )
    result = train_model(
        fixture_dir,
        tmp_path / "model_keep",
        filter_empty_skeleton=False,
    )
    assert result.frame_count == 12
    assert result.skipped_empty_skeleton_frames == 0
