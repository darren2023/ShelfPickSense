"""批量训练、评测与模型对比。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis.evaluation import ModelEvaluation, compare_reports, evaluate_model, save_report
from analysis.train import TrainResult, train_model


DEFAULT_MODEL_NAMES = ["sklearn_rf", "sklearn_logistic"]


@dataclass
class BenchmarkResult:
    train_data_dir: str
    eval_data_dir: str
    output_dir: str
    model_names: list[str]
    train_results: list[TrainResult]
    reports: list[ModelEvaluation]
    comparison: list[dict[str, Any]]
    benchmarked_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_data_dir": self.train_data_dir,
            "eval_data_dir": self.eval_data_dir,
            "output_dir": self.output_dir,
            "model_names": self.model_names,
            "train_results": [r.to_dict() for r in self.train_results],
            "reports": [r.to_dict() for r in self.reports],
            "comparison": self.comparison,
            "benchmarked_at": self.benchmarked_at,
        }


def run_benchmark(
    *,
    train_data_dir: Path,
    output_dir: Path,
    model_names: list[str] | None = None,
    eval_data_dir: Path | None = None,
) -> BenchmarkResult:
    """批量训练多个模型，并在同一评测集上生成对比结果。"""
    train_data_dir = Path(train_data_dir)
    eval_data_dir = Path(eval_data_dir) if eval_data_dir else train_data_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(model_names or DEFAULT_MODEL_NAMES)
    train_results: list[TrainResult] = []
    reports: list[ModelEvaluation] = []

    for model_name in names:
        model_dir = output_dir / model_name
        train_result = train_model(train_data_dir, model_dir, model_name=model_name)
        report = evaluate_model(model_dir, eval_data_dir)
        save_report(report, model_dir / "eval_report.json")

        train_results.append(train_result)
        reports.append(report)

    comparison = compare_reports(reports)
    result = BenchmarkResult(
        train_data_dir=str(train_data_dir.resolve()),
        eval_data_dir=str(eval_data_dir.resolve()),
        output_dir=str(output_dir.resolve()),
        model_names=names,
        train_results=train_results,
        reports=reports,
        comparison=comparison,
        benchmarked_at=datetime.now(timezone.utc).isoformat(),
    )
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result
