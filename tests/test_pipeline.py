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
    ctx = FeatureContext.from_record(record, frame, frame_index=record.frame_index())
    feat = reg.extract_frame_feature_groups_from_context(ctx)[0]
    assert "dummy.value" in feat.features


def test_two_stage_layout_prediction(fixture_data_dir: Path, tmp_path: Path):
    import json

    from analysis.dataset import load_dataset
    from analysis.models import SklearnPickingModel, create_model
    from analysis.records import load_record

    dataset = load_dataset(fixture_data_dir)
    record = load_record(fixture_data_dir)
    model = create_model("sklearn_rf")
    model.fit(dataset)

    pick_frame = next(s for s in dataset.frame_samples if s.is_picking)
    pred = model.predict_frame(
        pick_frame.x,
        record_id=pick_frame.record_id,
        frame_idx=pick_frame.frame_idx,
        box_layout=record.box_layout,
    )
    assert pred.is_picking
    assert pred.predicted_layout_layer_norm == pytest.approx(
        pick_frame.target_layout_layer_norm, abs=0.05
    )
    assert pred.predicted_layout_column_norm == pytest.approx(
        pick_frame.target_layout_column_norm, abs=0.05
    )
    assert pred.predicted_layout_layer == 1
    assert pred.predicted_layout_column == 2
    assert pred.predicted_box_tokens

    model_dir = tmp_path / "layout_model"
    model.save(model_dir)
    loaded = SklearnPickingModel.load(model_dir)
    assert (model_dir / "layout_layer_reg.pkl").is_file()
    assert (model_dir / "layout_column_reg.pkl").is_file()
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["stage1_target"] == "is_picking"
    assert meta["stage2_targets"] == ["target_layout_layer_norm", "target_layout_column_norm"]


def test_supported_sklearn_models_can_train_predict_and_load(fixture_data_dir: Path, tmp_path: Path):
    from analysis.dataset import load_dataset
    from analysis.models import SUPPORTED_MODEL_NAMES, SklearnPickingModel, create_model

    dataset = load_dataset(fixture_data_dir)
    sample = dataset.frame_samples[0]

    for model_name in SUPPORTED_MODEL_NAMES:
        if model_name == "lightgbm":
            continue
        model = create_model(model_name)
        model.fit(dataset)
        prediction = model.predict_frame(
            sample.x,
            record_id=sample.record_id,
            frame_idx=sample.frame_idx,
        )
        assert prediction.record_id == sample.record_id
        assert 0.0 <= prediction.picking_prob <= 1.0

        model_dir = tmp_path / model_name
        model.save(model_dir)
        loaded = SklearnPickingModel.load(model_dir)
        loaded_prediction = loaded.predict_frame(
            sample.x,
            record_id=sample.record_id,
            frame_idx=sample.frame_idx,
        )
        assert 0.0 <= loaded_prediction.picking_prob <= 1.0


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
    assert len(result.comparison) == 3
    assert result.comparison[0]["model_name"] == "rule_collision"
    assert result.comparison[0].get("is_baseline") is True
    ml_rows = [row for row in result.comparison if not row.get("is_baseline")]
    assert len(ml_rows) == 2
    assert "macro_f1" in ml_rows[0]
    assert "beats_baseline" in ml_rows[0]
    assert "negative_f1" in ml_rows[0]
    assert result.baseline_report is not None
    assert (output_dir / "rule_collision" / "eval_report.json").is_file()
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


def test_cli_benchmark_features_runs_multiple_feature_sets(tmp_path: Path):
    import json

    from analysis.cli import main
    from fixtures import make_fixture_record

    train_dir = make_fixture_record(tmp_path / "Train" / "train_record")
    test_dir = make_fixture_record(tmp_path / "Test" / "test_record")
    selected_frame = ["skeleton.person_count", "spatial.any_wrist_inside_box"]
    selected_box = ["spatial.wrist_min_dist_norm", "spatial.wrist_inside"]
    plan_path = tmp_path / "feature_benchmark_plan.json"
    output_dir = tmp_path / "feature_benchmark"
    plan_path.write_text(
        json.dumps(
            {
                "train_data_dir": str(train_dir),
                "eval_data_dir": str(test_dir),
                "output_dir": str(output_dir),
                "models": ["sklearn_rf", "sklearn_logistic"],
                "jobs": 2,
                "feature_frame_stride": 2,
                "feature_sets": [
                    {"name": "all_features"},
                    {
                        "name": "selected",
                        "frame_features": selected_frame,
                        "box_features": selected_box,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    ret = main(["benchmark-features", "--plan", str(plan_path)])

    assert ret == 0
    run_dirs = [path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("run_")]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "feature_benchmark_summary.json").is_file()
    assert (run_dir / "feature_benchmark_report.md").is_file()
    assert (run_dir / "feature_benchmark_plan.json").is_file()
    assert (run_dir / "all_features" / "benchmark_summary.json").is_file()
    assert (run_dir / "selected" / "benchmark_summary.json").is_file()
    assert (run_dir / "rule_collision" / "eval_report.json").is_file()
    assert not (output_dir / "feature_benchmark_summary.json").exists()

    summary = json.loads((run_dir / "feature_benchmark_summary.json").read_text(encoding="utf-8"))
    assert len(summary["sets"]) == 2
    assert {item["name"] for item in summary["sets"]} == {"all_features", "selected"}
    saved_plan = json.loads((run_dir / "feature_benchmark_plan.json").read_text(encoding="utf-8"))
    assert saved_plan["train_data_dir"] == str(train_dir.resolve())
    assert saved_plan["eval_data_dir"] == str(test_dir.resolve())
    assert saved_plan["output_dir"] == str(output_dir.resolve())
    assert saved_plan["run_output_dir"] == str(run_dir.resolve())
    assert saved_plan["models"] == ["sklearn_rf", "sklearn_logistic"]
    assert saved_plan["jobs"] == 2
    assert saved_plan["feature_frame_stride"] == 2
    assert [item["name"] for item in saved_plan["feature_sets"]] == ["all_features", "selected"]
    assert summary["feature_frame_stride"] == 2
    for item in summary["sets"]:
        assert item["benchmark"]["baseline_report"]["model_name"] == "rule_collision"
        assert (
            item["benchmark"]["baseline_report"]["extra"]["source"]
            == "box_human_det/services/event_engine/collision.py"
        )
        assert item["benchmark"]["comparison"][0]["model_name"] == "rule_collision"
        assert item["benchmark"]["feature_cache_path"]
        assert item["benchmark"]["feature_cache_hit"] is False
        assert item["benchmark"]["feature_frame_stride"] == 2
        assert item["benchmark"]["feature_dataset_seconds"] >= 0.0
        assert item["benchmark"]["train_results"][0]["frame_count"] == 5
        assert set(item["benchmark"]["model_timings"]) == {"sklearn_rf", "sklearn_logistic"}
        for timing in item["benchmark"]["model_timings"].values():
            assert timing["fit_seconds"] >= 0.0
            assert timing["eval_seconds"] >= 0.0
            assert timing["total_seconds"] >= 0.0
    assert summary["output_dir"] == str(run_dir.resolve())
    report = (run_dir / "feature_benchmark_report.md").read_text(encoding="utf-8")
    assert "多特征配置 Benchmark 对比报告" in report
    assert "各特征配置最佳模型汇总" in report
    assert "各特征配置模型明细" in report
    assert "各特征配置最佳模型" in report
    assert "全局推荐" in report
    assert "Macro-F1" in report
    assert "sklearn_rf" in report
    assert "all_features" in report
    assert "selected" in report
    assert len(list((output_dir / "feature_cache").glob("*.npz"))) == 2

    ret = main(["benchmark-features", "--plan", str(plan_path)])

    assert ret == 0
    run_dirs = sorted(path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("run_"))
    assert len(run_dirs) == 2
    second_run_dir = run_dirs[-1]
    second_summary = json.loads(
        (second_run_dir / "feature_benchmark_summary.json").read_text(encoding="utf-8")
    )
    assert all(item["benchmark"]["feature_cache_hit"] is True for item in second_summary["sets"])
    assert all(item["benchmark"]["feature_frame_stride"] == 2 for item in second_summary["sets"])


def test_benchmark_features_auto_split_without_eval_data_dir(tmp_path: Path):
    import json

    from analysis.cli import main
    from fixtures import make_fixture_record

    data_dir = tmp_path / "Records"
    make_fixture_record(data_dir / "record_001")
    make_fixture_record(data_dir / "record_002")
    make_fixture_record(data_dir / "record_003")
    output_dir = tmp_path / "feature_benchmark"
    plan_path = tmp_path / "feature_benchmark_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "train_data_dir": str(data_dir),
                "output_dir": str(output_dir),
                "models": ["sklearn_rf"],
                "jobs": 1,
                "train_split_ratio": 0.67,
                "feature_sets": [{"name": "all_features"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    ret = main(["benchmark-features", "--plan", str(plan_path)])

    assert ret == 0
    run_dirs = [path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("run_")]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    summary = json.loads((run_dir / "feature_benchmark_summary.json").read_text(encoding="utf-8"))
    saved_plan = json.loads((run_dir / "feature_benchmark_plan.json").read_text(encoding="utf-8"))
    assert saved_plan["eval_data_dir"] is None
    assert saved_plan["auto_split"] is True
    assert saved_plan["actual_train_record_ids"] == ["record_001", "record_002"]
    assert saved_plan["actual_eval_record_ids"] == ["record_003"]
    assert summary["auto_split"] is True
    assert summary["train_split_ratio"] == 0.67
    assert summary["train_record_ids"] == ["record_001", "record_002"]
    assert summary["eval_record_ids"] == ["record_003"]
    benchmark = summary["sets"][0]["benchmark"]
    assert benchmark["train_record_ids"] == ["record_001", "record_002"]
    assert benchmark["eval_record_ids"] == ["record_003"]
    assert benchmark["train_results"][0]["record_ids"] == ["record_001", "record_002"]
    assert benchmark["baseline_report"]["record_ids"] == ["record_003"]
    assert benchmark["feature_cache_hit"] is False


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
    meta_path = output_dir / "features_meta.json"
    assert frame_path.is_file()
    assert not (output_dir / "box_features.parquet").exists()
    assert meta_path.is_file()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["frame_count"] == 10
    assert meta["box_sample_count"] == 0
    assert meta["frame_feature_names"]
    assert meta["box_feature_names"] == []

    frame_df = pd.read_parquet(frame_path)
    assert len(frame_df) == 10
    assert {"record_id", "frame_idx", "is_picking"} <= set(frame_df.columns)
    assert {"target_layout_layer_norm", "target_layout_column_norm", "target_layout_shelf_side"} <= set(frame_df.columns)


def test_cli_export_features_defaults_output_to_record_dir(fixture_data_dir: Path):
    import json

    from analysis.cli import main

    ret = main(
        [
            "export-features",
            "--data-dir",
            str(fixture_data_dir),
        ]
    )

    assert ret == 0
    assert (fixture_data_dir / "frame_features.parquet").is_file()
    assert not (fixture_data_dir / "box_features.parquet").exists()
    meta_path = fixture_data_dir / "features_meta.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["output_dir"] == str(fixture_data_dir)


def test_cli_export_features_parallel_records(tmp_path: Path):
    import json

    import pandas as pd

    from analysis.cli import main
    from fixtures import make_fixture_record

    data_dir = tmp_path / "records"
    make_fixture_record(data_dir / "record_001")
    make_fixture_record(data_dir / "record_002")
    output_dir = tmp_path / "parallel_features"

    ret = main(
        [
            "export-features",
            "--data-dir",
            str(data_dir),
            "--output",
            str(output_dir),
            "--format",
            "csv",
            "--feature-jobs",
            "2",
        ]
    )

    assert ret == 0
    meta = json.loads((output_dir / "features_meta.json").read_text(encoding="utf-8"))
    assert meta["frame_count"] == 20
    frame_df = pd.read_csv(output_dir / "frame_features.csv")
    assert set(frame_df["record_id"]) == {"record_001", "record_002"}


def test_cli_feature_config_filters_export_and_train(fixture_data_dir: Path, tmp_path: Path):
    import json

    import pandas as pd

    from analysis.cli import main

    config_path = tmp_path / "feature_config.json"
    selected_frame = ["skeleton.person_count", "spatial.any_wrist_inside_box"]
    selected_box = ["spatial.wrist_min_dist_norm", "spatial.wrist_inside"]
    config_path.write_text(
        json.dumps(
            {
                "frame_features": selected_frame,
                "box_features": selected_box,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    features_dir = tmp_path / "selected_features"
    ret = main(
        [
            "export-features",
            "--data-dir",
            str(fixture_data_dir),
            "--output",
            str(features_dir),
            "--format",
            "csv",
            "--feature-config",
            str(config_path),
        ]
    )

    assert ret == 0
    meta = json.loads((features_dir / "features_meta.json").read_text(encoding="utf-8"))
    assert meta["frame_feature_names"] == selected_frame
    assert meta["box_feature_names"] == []
    assert meta["feature_selection"]["frame_features"] == selected_frame

    frame_df = pd.read_csv(features_dir / "frame_features.csv")
    assert set(selected_frame) <= set(frame_df.columns)
    assert "skeleton.infer_width" not in frame_df.columns
    assert not (features_dir / "box_features.csv").exists()

    model_dir = tmp_path / "selected_model"
    ret = main(
        [
            "train",
            "--data-dir",
            str(fixture_data_dir),
            "--output",
            str(model_dir),
            "--feature-config",
            str(config_path),
        ]
    )

    assert ret == 0
    model_meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    assert model_meta["frame_feature_names"] == selected_frame
    assert model_meta["box_feature_names"] == []


def test_cli_feature_config_expands_feature_groups(fixture_data_dir: Path, tmp_path: Path):
    import json

    import pandas as pd

    from analysis.cli import main

    config_path = tmp_path / "feature_groups.json"
    config_path.write_text(
        json.dumps(
            {
                "frame_features": ["skeleton"],
                "box_features": ["layout"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    features_dir = tmp_path / "group_features"
    ret = main(
        [
            "export-features",
            "--data-dir",
            str(fixture_data_dir),
            "--output",
            str(features_dir),
            "--format",
            "csv",
            "--feature-config",
            str(config_path),
        ]
    )

    assert ret == 0
    meta = json.loads((features_dir / "features_meta.json").read_text(encoding="utf-8"))
    assert meta["frame_feature_names"]
    assert all(name.startswith("skeleton.") for name in meta["frame_feature_names"])
    assert meta["box_feature_names"] == []

    frame_df = pd.read_csv(features_dir / "frame_features.csv")
    feature_columns = set(meta["frame_feature_names"])
    assert feature_columns <= set(frame_df.columns)
    assert "spatial.any_wrist_inside_box" not in frame_df.columns


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
        assert not (output_dir / f"box_features.{suffix}").exists()

    frame_csv = pd.read_csv(output_dir / "frame_features.csv")
    assert len(frame_csv) == 10
    assert "target_layout_layer_norm" in frame_csv.columns

    first_jsonl = (output_dir / "frame_features.jsonl").read_text(encoding="utf-8").splitlines()[0]
    first_row = json.loads(first_jsonl)
    assert {"record_id", "frame_idx", "is_picking", "target_layout_layer_norm"} <= set(first_row)

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
    assert summary["box_sample_count"] == 0
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


def test_cli_analyze_features_with_single_frame_feature(fixture_data_dir: Path, tmp_path: Path):
    import json

    import pandas as pd

    from analysis.cli import main

    config_path = tmp_path / "single_frame_feature.json"
    config_path.write_text(
        json.dumps({"frame_features": ["rule.p0_any_collision"]}, ensure_ascii=False),
        encoding="utf-8",
    )

    output_dir = tmp_path / "correlations_single_feature"
    ret = main(
        [
            "analyze-features",
            "--data-dir",
            str(fixture_data_dir),
            "--output",
            str(output_dir),
            "--feature-config",
            str(config_path),
        ]
    )

    assert ret == 0
    frame_loadings = pd.read_csv(output_dir / "frame_pca_loadings.csv")
    assert list(frame_loadings.columns) == ["feature", "PC1"]
    assert (output_dir / "correlation_report.md").is_file()


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
    assert summary["box_sample_count"] == 0
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
