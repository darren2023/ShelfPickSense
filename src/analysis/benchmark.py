"""批量训练、评测与模型对比。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from analysis.dataset import build_dataset
from analysis.evaluation import Evaluator, ModelEvaluation, compare_reports, save_report
from analysis.features.registry import default_registry
from analysis.models import SUPPORTED_MODEL_NAMES
from analysis.records import load_all_records
from analysis.train import TrainResult, train_model_from_dataset


DEFAULT_MODEL_NAMES = list(SUPPORTED_MODEL_NAMES)


def _safe_filename_part(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text or "").strip())
    return safe.strip("_") or "unknown"


def prediction_filename_for_records(eval_records) -> str:
    """生成带评测记录名的预测结果文件名。"""
    record_ids = [_safe_filename_part(getattr(record, "record_id", "")) for record in eval_records]
    if not record_ids:
        return "eval_predictions_unknown.json"
    if len(record_ids) == 1:
        return f"eval_predictions_{record_ids[0]}.json"
    joined = "__".join(record_ids[:3])
    if len(record_ids) > 3:
        joined = f"{joined}__and_{len(record_ids) - 3}_more"
    return f"eval_predictions_{len(record_ids)}records_{joined}.json"


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
    jobs: int = 1,
) -> BenchmarkResult:
    """批量训练多个模型，并在同一评测集上生成对比结果。"""
    train_data_dir = Path(train_data_dir)
    eval_data_dir = Path(eval_data_dir) if eval_data_dir else train_data_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(model_names or DEFAULT_MODEL_NAMES)
    workers = max(1, int(jobs or 1))
    logger.info(
        "准备运行 benchmark: models={}, workers={}, train_data={}, eval_data={}, output={}",
        names,
        min(workers, len(names)),
        train_data_dir,
        eval_data_dir,
        output_dir,
    )
    registry = default_registry()
    logger.info("benchmark 加载训练数据: {}", train_data_dir)
    train_records = load_all_records(train_data_dir)
    logger.info("benchmark 构建训练数据集")
    train_dataset = build_dataset(train_records, registry)
    logger.info(
        "benchmark 训练数据集就绪: records={}, frames={}, positive_frames={}, box_samples={}",
        len(train_records),
        train_dataset.frame_count,
        train_dataset.positive_frame_count,
        len(train_dataset.box_samples),
    )

    if train_data_dir.resolve() == eval_data_dir.resolve():
        eval_records = train_records
        logger.info("benchmark 复用训练记录作为评测记录")
    else:
        logger.info("benchmark 加载评测数据: {}", eval_data_dir)
        eval_records = load_all_records(eval_data_dir)
    evaluator = Evaluator(eval_records, registry=registry)
    predictions_filename = prediction_filename_for_records(eval_records)

    def _run_one(model_name: str) -> tuple[str, TrainResult, ModelEvaluation]:
        model_dir = output_dir / model_name
        try:
            logger.info("benchmark 子任务开始: model={}, output={}", model_name, model_dir)
            train_result, model = train_model_from_dataset(
                train_dataset,
                records=train_records,
                data_dir=train_data_dir,
                output_dir=model_dir,
                model_name=model_name,
            )
            report = evaluator.evaluate(
                model,
                data_dir=str(eval_data_dir.resolve()),
                predictions_output_path=model_dir / predictions_filename,
            )
            save_report(report, model_dir / "eval_report.json")
            logger.info(
                "benchmark 子任务完成: model={}, picking_f1={:.4f}, box_f1={:.4f}",
                model_name,
                report.picking.f1,
                report.box.micro_f1,
            )
            return model_name, train_result, report
        except Exception:
            logger.exception("benchmark 子任务失败: model={}", model_name)
            raise

    results_by_name: dict[str, tuple[TrainResult, ModelEvaluation]] = {}
    if workers == 1 or len(names) <= 1:
        for model_name in names:
            name, train_result, report = _run_one(model_name)
            results_by_name[name] = (train_result, report)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(names))) as executor:
            futures = {executor.submit(_run_one, model_name): model_name for model_name in names}
            for future in as_completed(futures):
                name, train_result, report = future.result()
                results_by_name[name] = (train_result, report)

    train_results = [results_by_name[name][0] for name in names]
    reports = [results_by_name[name][1] for name in names]

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
    logger.info("benchmark 汇总报告已保存: {}", output_dir / "benchmark_summary.json")
    return result
