"""多特征 benchmark 报告测试。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from analysis.benchmark import BenchmarkResult
from analysis.feature_benchmark import FeatureBenchmarkBatchResult, FeatureBenchmarkSetResult, _write_batch_report
from analysis.train import TrainResult


def test_write_batch_report_includes_per_feature_model_tables(tmp_path: Path):
    comparison = [
        {
            "model_name": "sklearn_rf",
            "macro_f1": 0.82,
            "balanced_accuracy": 0.80,
            "picking_f1": 0.78,
            "picking_recall": 0.76,
            "picking_precision": 0.80,
            "box_micro_f1": 0.70,
            "box_exact_match": 0.65,
        },
        {
            "model_name": "sklearn_logistic",
            "macro_f1": 0.75,
            "balanced_accuracy": 0.74,
            "picking_f1": 0.72,
            "picking_recall": 0.70,
            "picking_precision": 0.74,
            "box_micro_f1": 0.68,
            "box_exact_match": 0.60,
        },
    ]
    benchmark = BenchmarkResult(
        train_data_dir="/train",
        eval_data_dir="/test",
        output_dir="/out/set_a",
        model_names=["sklearn_rf", "sklearn_logistic"],
        train_results=[
            TrainResult(
                model_name="sklearn_rf",
                model_path="/out/set_a/sklearn_rf",
                data_dir="/train",
                record_ids=["r1"],
                frame_count=10,
                positive_frames=3,
                box_samples=6,
                trained_at=datetime.now(timezone.utc).isoformat(),
            )
        ],
        reports=[],
        comparison=comparison,
        benchmarked_at=datetime.now(timezone.utc).isoformat(),
    )
    batch = FeatureBenchmarkBatchResult(
        train_data_dir="/train",
        eval_data_dir="/test",
        output_dir=str(tmp_path),
        model_names=["sklearn_rf", "sklearn_logistic"],
        sets=[
            FeatureBenchmarkSetResult(
                name="consecutive_hit_3",
                output_dir=str(tmp_path / "consecutive_hit_3"),
                feature_selection={"frame_features": ["temporal.consecutive_hit_3"], "box_features": None},
                best_model="sklearn_rf",
                best_macro_f1=0.82,
                benchmark=benchmark,
            )
        ],
        benchmarked_at=datetime.now(timezone.utc).isoformat(),
        report_path="",
        summary_path="",
    )

    report_path = _write_batch_report(batch, tmp_path)
    report = report_path.read_text(encoding="utf-8")

    assert "## 各特征配置最佳模型汇总" in report
    assert "## 各特征配置模型明细" in report
    assert "### consecutive_hit_3" in report
    assert "temporal.consecutive_hit_3" in report
    assert "| sklearn_rf |" in report
    assert "| sklearn_logistic |" in report
    assert "**本组推荐**：`sklearn_rf`" in report
    assert "### 各特征配置最佳模型" in report
    assert "### 全局推荐" in report
