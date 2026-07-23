"""评测 stride 与训练采样序列对齐测试。"""

from pathlib import Path

from fixtures import make_fixture_record


def test_evaluate_model_feature_frame_stride_matches_training_sample_count(
    tmp_path: Path,
) -> None:
    from analysis.dataset import build_dataset, stride_frames
    from analysis.evaluation import evaluate_model
    from analysis.features.registry import default_registry
    from analysis.records import load_record
    from analysis.train import train_model

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    record = load_record(fixture_dir)
    source_frame_count = len(record.frames())
    stride = 2
    expected_eval_frames = len(stride_frames(record.frames(), stride))

    model_dir = tmp_path / "model_stride2"
    train_model(
        fixture_dir,
        model_dir,
        feature_frame_stride=stride,
    )
    dataset = build_dataset([record], default_registry(), feature_frame_stride=stride)
    assert dataset.frame_count == expected_eval_frames

    report = evaluate_model(
        model_dir,
        fixture_dir,
        feature_frame_stride=stride,
    )
    assert report.extra["feature_frame_stride"] == stride
    assert report.extra["frame_count"] == expected_eval_frames
    assert report.extra["frame_count"] < source_frame_count


def test_predict_record_stride_uses_sample_sequence_for_temporal_features(
    tmp_path: Path,
) -> None:
    """stride=2 评测时，时序特征应对齐采样序列而非被跳过的中间帧。"""
    from analysis.dataset import build_dataset
    from analysis.evaluation import predict_record
    from analysis.features.registry import default_registry
    from analysis.models import SklearnPickingModel
    from analysis.records import load_record
    from analysis.train import train_model

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    record = load_record(fixture_dir)
    model_dir = tmp_path / "model"
    train_model(fixture_dir, model_dir)
    model = SklearnPickingModel.load(model_dir)

    dense_dataset = build_dataset([record], default_registry(), feature_frame_stride=1)
    strided_dataset = build_dataset([record], default_registry(), feature_frame_stride=2)
    dense_by_idx = {s.frame_idx: s for s in dense_dataset.frame_samples}
    strided_by_idx = {s.frame_idx: s for s in strided_dataset.frame_samples}

    move_key = "temporal.left_wrist_move_1"
    dense_names = dense_dataset.frame_feature_names
    strided_names = strided_dataset.frame_feature_names

    preds = predict_record(model, record, feature_frame_stride=2)
    pred_frame_indices = {p["frame_idx"] for p in preds}
    assert 7 in pred_frame_indices
    assert 6 not in pred_frame_indices

    assert dense_by_idx[7].x[dense_names.index(move_key)] == 0.0
    assert strided_by_idx[7].x[strided_names.index(move_key)] > 0.0
