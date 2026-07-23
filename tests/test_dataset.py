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


def test_build_dataset_feature_frame_stride(tmp_path: Path):
    from analysis.dataset import build_dataset
    from analysis.features.registry import default_registry
    from analysis.records import load_record

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    record = load_record(fixture_dir)

    dataset = build_dataset([record], default_registry(), feature_frame_stride=3)

    assert dataset.frame_count == 4
    assert [sample.frame_idx for sample in dataset.frame_samples] == [1, 4, 7, 10]


def test_stride_temporal_lookup_uses_sample_sequence_not_skipped_frames(tmp_path: Path):
    """stride=2 时，move_1 应对齐上一采样帧，不得使用被 stride 跳过的中间帧。"""
    from analysis.dataset import build_dataset
    from analysis.features.registry import default_registry
    from analysis.records import load_record

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    record = load_record(fixture_dir)

    dense = build_dataset([record], default_registry(), feature_frame_stride=1)
    strided = build_dataset([record], default_registry(), feature_frame_stride=2)

    dense_by_idx = {s.frame_idx: s for s in dense.frame_samples}
    strided_by_idx = {s.frame_idx: s for s in strided.frame_samples}

    move_key = "temporal.left_wrist_move_1"
    dense_names = dense.frame_feature_names
    strided_names = strided.frame_feature_names

    # frame 7：stride=2 时上一采样点为 frame 5；全量序列上一帧为 frame 6（取货手腕位置不同）
    dense_move_7 = dense_by_idx[7].x[dense_names.index(move_key)]
    strided_move_7 = strided_by_idx[7].x[strided_names.index(move_key)]
    assert dense_move_7 == 0.0
    assert strided_move_7 > 0.0


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
