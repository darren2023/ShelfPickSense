from pathlib import Path

import pytest

from fixtures import make_fixture_record


@pytest.fixture
def fixture_data_dir(tmp_path: Path) -> Path:
    return make_fixture_record(tmp_path / "record_001")


def test_load_and_train_eval(fixture_data_dir: Path, tmp_path: Path):
    import json

    from analysis.evaluation import evaluate_model
    from analysis.records import load_record
    from analysis.train import train_model

    record = load_record(fixture_data_dir)
    assert record.record_id == "record_001"
    assert len(record.box_tokens) == 2
    assert record.labels.frame_labels[6].is_picking
    assert record.labels.frame_labels[6].confirmed_box_tokens == ["S1:A1"]

    model_dir = tmp_path / "model"
    result = train_model(fixture_data_dir, model_dir)
    assert result.frame_count == 10
    assert result.positive_frames == 3

    predictions_path = tmp_path / "predictions.json"
    report = evaluate_model(model_dir, fixture_data_dir, predictions_output_path=predictions_path)
    assert report.picking.f1 >= 0.5
    assert 0.0 <= report.picking.macro_f1 <= 1.0
    assert 0.0 <= report.picking.balanced_accuracy <= 1.0
    assert report.extra["frame_count"] == 10
    assert predictions_path.is_file()
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    assert predictions["prediction_count"] == 10
    first = predictions["predictions"][0]
    assert {"record_id", "frame_idx", "true_is_picking", "pred_is_picking", "picking_prob"} <= set(first)


def test_no_event_review_all_negative(tmp_path: Path):
    from analysis.labels import build_labels_from_event_review
    from analysis.records import load_record
    from fixtures import make_fixture_record

    record_dir = make_fixture_record(tmp_path / "no_review")
    (record_dir / "event_review.json").unlink()

    record = load_record(record_dir)
    assert all(not lbl.is_picking for lbl in record.labels.frame_labels.values())


def test_feature_registry_extensible(fixture_data_dir: Path):
    from analysis.features.base import FeatureContext, FeatureExtractor
    from analysis.features.registry import FeatureRegistry, default_registry
    from analysis.records import load_record

    class DummyExtractor(FeatureExtractor):
        name = "dummy"

        def extract_frame(self, ctx: FeatureContext) -> dict[str, float]:
            return {"value": float(ctx.frame.frame_idx)}

    reg = default_registry()
    reg.register(DummyExtractor())
    record = load_record(fixture_data_dir)
    frame = record.frames()[0]
    feat = reg.extract_frame_features(record, frame)
    assert "dummy.value" in feat.features


def test_benchmark_runs_multiple_models(fixture_data_dir: Path, tmp_path: Path):
    from analysis.benchmark import DEFAULT_MODEL_NAMES, run_benchmark

    assert len(DEFAULT_MODEL_NAMES) > 2

    output_dir = tmp_path / "benchmark"
    result = run_benchmark(
        train_data_dir=fixture_data_dir,
        output_dir=output_dir,
        model_names=["sklearn_rf", "sklearn_logistic"],
        jobs=2,
    )

    assert [r.model_name for r in result.reports] == ["sklearn_rf", "sklearn_logistic"]
    assert len(result.comparison) == 2
    assert "macro_f1" in result.comparison[0]
    assert "negative_f1" in result.comparison[0]
    assert (output_dir / "sklearn_rf" / "eval_report.json").is_file()
    assert (output_dir / "sklearn_rf" / "eval_predictions_record_001.json").is_file()
    assert (output_dir / "sklearn_logistic" / "eval_report.json").is_file()
    assert (output_dir / "sklearn_logistic" / "eval_predictions_record_001.json").is_file()
    assert (output_dir / "benchmark_summary.json").is_file()


def test_benchmark_train_test_dirs_generate_report(tmp_path: Path):
    from analysis.cli import main
    from fixtures import make_fixture_record

    input_dir = tmp_path / "split_data"
    train_dir = make_fixture_record(input_dir / "Train" / "train_record")
    test_dir = make_fixture_record(input_dir / "Test" / "test_record")
    output_dir = tmp_path / "train_test_benchmark"

    ret = main(
        [
            "benchmark",
            "--data-dir",
            str(train_dir),
            "--eval-data-dir",
            str(test_dir),
            "--output",
            str(output_dir),
            "--models",
            "sklearn_rf",
            "sklearn_logistic",
            "--jobs",
            "2",
        ]
    )

    assert ret == 0
    assert (output_dir / "benchmark_summary.json").is_file()
    assert (output_dir / "benchmark_report.md").is_file()
    assert (output_dir / "sklearn_rf" / "train_result.json").is_file()
    assert (output_dir / "sklearn_rf" / "eval_report.json").is_file()
    assert (output_dir / "sklearn_logistic" / "train_result.json").is_file()
    assert (output_dir / "sklearn_logistic" / "eval_report.json").is_file()

    report = (output_dir / "benchmark_report.md").read_text(encoding="utf-8")
    assert "Benchmark 模型训练与评测报告" in report
    assert "## 结论" in report
    assert "Macro-F1" in report


def test_cli_export_features(fixture_data_dir: Path, tmp_path: Path):
    import json

    import pandas as pd

    from analysis.cli import main

    output_dir = tmp_path / "features"
    ret = main(
        [
            "export-features",
            "--data-dir",
            str(fixture_data_dir),
            "--output",
            str(output_dir),
        ]
    )

    assert ret == 0
    frame_path = output_dir / "frame_features.parquet"
    box_path = output_dir / "box_features.parquet"
    meta_path = output_dir / "features_meta.json"
    assert frame_path.is_file()
    assert box_path.is_file()
    assert meta_path.is_file()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["frame_count"] == 10
    assert meta["box_sample_count"] == 6
    assert meta["frame_feature_names"]
    assert meta["box_feature_names"]

    frame_df = pd.read_parquet(frame_path)
    box_df = pd.read_parquet(box_path)
    assert len(frame_df) == 10
    assert len(box_df) == 6
    assert {"record_id", "frame_idx", "is_picking", "confirmed_box_tokens"} <= set(frame_df.columns)
    assert {"record_id", "frame_idx", "box_token", "is_target"} <= set(box_df.columns)


def test_cli_export_features_all_formats(fixture_data_dir: Path, tmp_path: Path):
    import json

    import pandas as pd

    from analysis.cli import main

    output_dir = tmp_path / "features_all"
    ret = main(
        [
            "export-features",
            "--data-dir",
            str(fixture_data_dir),
            "--output",
            str(output_dir),
            "--format",
            "all",
        ]
    )

    assert ret == 0
    for suffix in ("parquet", "csv", "jsonl"):
        assert (output_dir / f"frame_features.{suffix}").is_file()
        assert (output_dir / f"box_features.{suffix}").is_file()

    frame_csv = pd.read_csv(output_dir / "frame_features.csv")
    assert len(frame_csv) == 10
    assert "confirmed_box_tokens" in frame_csv.columns

    first_jsonl = (output_dir / "frame_features.jsonl").read_text(encoding="utf-8").splitlines()[0]
    first_row = json.loads(first_jsonl)
    assert {"record_id", "frame_idx", "is_picking", "confirmed_box_tokens"} <= set(first_row)

    meta = json.loads((output_dir / "features_meta.json").read_text(encoding="utf-8"))
    assert meta["output_format"] == "all"
    assert set(meta["output_files"]) == {"parquet", "csv", "jsonl"}


def test_cli_analyze_features(fixture_data_dir: Path, tmp_path: Path):
    import json

    import pandas as pd

    from analysis.cli import main

    output_dir = tmp_path / "correlations"
    ret = main(
        [
            "analyze-features",
            "--data-dir",
            str(fixture_data_dir),
            "--output",
            str(output_dir),
            "--threshold",
            "0.0",
            "--top-n",
            "10",
        ]
    )

    assert ret == 0
    expected_files = [
        "frame_feature_samples.csv",
        "box_feature_samples.csv",
        "frame_feature_correlation.csv",
        "frame_target_correlation.csv",
        "frame_high_correlation_pairs.csv",
        "frame_pca_explained_variance.csv",
        "frame_pca_loadings.csv",
        "frame_pca_projection.csv",
        "frame_low_value_constant_features.csv",
        "frame_low_value_low_target_correlation.csv",
        "frame_low_value_redundant_pairs.csv",
        "box_feature_correlation.csv",
        "box_target_correlation.csv",
        "box_high_correlation_pairs.csv",
        "box_pca_explained_variance.csv",
        "box_pca_loadings.csv",
        "box_pca_projection.csv",
        "box_low_value_constant_features.csv",
        "box_low_value_low_target_correlation.csv",
        "box_low_value_redundant_pairs.csv",
        "correlation_report.md",
        "correlation_summary.json",
    ]
    for filename in expected_files:
        assert (output_dir / filename).is_file()
    for filename in [
        "frame_target_correlation_top.svg",
        "box_target_correlation_top.svg",
        "frame_feature_correlation_heatmap.svg",
        "box_feature_correlation_heatmap.svg",
        "frame_high_correlation_pairs.svg",
        "box_high_correlation_pairs.svg",
        "frame_pca_explained_variance.svg",
        "box_pca_explained_variance.svg",
        "frame_pca_scatter.svg",
        "box_pca_scatter.svg",
    ]:
        assert (output_dir / "figures" / filename).is_file()

    summary = json.loads((output_dir / "correlation_summary.json").read_text(encoding="utf-8"))
    assert summary["frame_count"] == 10
    assert summary["box_sample_count"] == 6
    assert "frame_target_correlation" in summary["outputs"]
    assert "box_target_correlation" in summary["outputs"]
    assert "frame_pca_explained_variance" in summary["outputs"]
    assert "box_pca_projection" in summary["outputs"]
    assert "frame_low_value_constant_features" in summary["outputs"]
    assert "box_low_value_redundant_pairs" in summary["outputs"]
    assert "report" in summary["outputs"]

    frame_target = pd.read_csv(output_dir / "frame_target_correlation.csv")
    box_target = pd.read_csv(output_dir / "box_target_correlation.csv")
    assert {"feature", "correlation", "abs_correlation", "non_null_count"} <= set(frame_target.columns)
    assert {"feature", "correlation", "abs_correlation", "non_null_count"} <= set(box_target.columns)
    report = (output_dir / "correlation_report.md").read_text(encoding="utf-8")
    assert "![帧级特征与 is_picking 的相关性]" in report
    assert "![货框特征与 is_target 的相关性]" in report
    assert "主成分分析 PCA" in report
    assert "低价值/冗余特征提示" in report
    assert "frame_pca_scatter.svg" in report


def test_cli_analyze_exported_features(fixture_data_dir: Path, tmp_path: Path):
    import json

    from analysis.cli import main

    features_dir = tmp_path / "features"
    export_ret = main(
        [
            "export-features",
            "--data-dir",
            str(fixture_data_dir),
            "--output",
            str(features_dir),
            "--format",
            "csv",
        ]
    )
    assert export_ret == 0

    output_dir = tmp_path / "correlations_from_features"
    analyze_ret = main(
        [
            "analyze-features",
            "--features-dir",
            str(features_dir),
            "--output",
            str(output_dir),
        ]
    )

    assert analyze_ret == 0
    summary = json.loads((output_dir / "correlation_summary.json").read_text(encoding="utf-8"))
    assert summary["input_source"] == str(features_dir)
    assert summary["frame_count"] == 10
    assert summary["box_sample_count"] == 6
    assert (output_dir / "frame_target_correlation.csv").is_file()
    assert (output_dir / "box_target_correlation.csv").is_file()
    assert (output_dir / "frame_pca_projection.csv").is_file()
    assert (output_dir / "box_pca_projection.csv").is_file()
    assert (output_dir / "correlation_report.md").is_file()


def test_realtime_predict_frame(fixture_data_dir: Path, tmp_path: Path):
    from analysis.realtime import RealtimePickingPredictor
    from analysis.records import load_record
    from analysis.train import train_model

    model_dir = tmp_path / "model"
    train_model(fixture_data_dir, model_dir)
    record = load_record(fixture_data_dir)
    frame = record.frames()[5]

    predictor = RealtimePickingPredictor(record_id="record_001")
    predictor.load_model(model_dir)
    predictor.set_infer_size(record.infer_width, record.infer_height)
    predictor.annotation = record.annotation
    pred = predictor.predict_frame(
        frame.persons,
        frame_idx=frame.frame_idx,
        timestamp_sec=frame.timestamp_sec,
    )

    assert pred.record_id == "record_001"
    assert pred.frame_idx == frame.frame_idx
    assert 0.0 <= pred.picking_prob <= 1.0
    assert isinstance(pred.predicted_box_tokens, list)


def test_picking_macro_f1_metrics():
    from analysis.evaluation import compute_picking_metrics

    metrics = compute_picking_metrics(
        y_true=[True, True, False, False],
        y_pred=[True, False, False, True],
    )

    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.negative_f1 == pytest.approx(0.5)
    assert metrics.macro_f1 == pytest.approx(0.5)
    assert metrics.balanced_accuracy == pytest.approx(0.5)
