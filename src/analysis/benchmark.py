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
from analysis.features.selection import FeatureSelection
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
    feature_selection: FeatureSelection | None = None,
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
    train_dataset = build_dataset(train_records, registry, feature_selection=feature_selection)
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
    report_path = _write_benchmark_report(result, output_dir)
    logger.info("benchmark 汇总报告已保存: {}", output_dir / "benchmark_summary.json")
    logger.info("benchmark Markdown 报告已保存: {}", report_path)
    return result


def _fmt(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _comparison_markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "无模型结果。\n"
    columns = [
        "model_name",
        "macro_f1",
        "balanced_accuracy",
        "picking_f1",
        "picking_recall",
        "picking_precision",
        "box_micro_f1",
        "box_exact_match",
    ]
    labels = {
        "model_name": "模型",
        "macro_f1": "Macro-F1",
        "balanced_accuracy": "Balanced Acc",
        "picking_f1": "取货 F1",
        "picking_recall": "取货 Recall",
        "picking_precision": "取货 Precision",
        "box_micro_f1": "货框 Micro-F1",
        "box_exact_match": "货框精确匹配",
    }
    lines = [
        "| " + " | ".join(labels[c] for c in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            values.append(str(value) if col == "model_name" else _fmt(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _recommendation(comparison: list[dict[str, Any]]) -> tuple[str, str]:
    if not comparison:
        return "", "没有可用模型结果，无法给出推荐。"
    best = comparison[0]
    best_model = str(best["model_name"])
    reason = (
        f"推荐模型 `{best_model}`。它在 Test 集上的 Macro-F1 为 {_fmt(best.get('macro_f1'))}，"
        f"Balanced Accuracy 为 {_fmt(best.get('balanced_accuracy'))}，"
        f"取货 Recall 为 {_fmt(best.get('picking_recall'))}，"
        f"货框 Micro-F1 为 {_fmt(best.get('box_micro_f1'))}。"
    )
    if len(comparison) > 1:
        second = comparison[1]
        delta = float(best.get("macro_f1", 0.0) or 0.0) - float(second.get("macro_f1", 0.0) or 0.0)
        if delta < 0.01:
            reason += (
                f" 但它与第二名 `{second['model_name']}` 的 Macro-F1 差距只有 {_fmt(delta)}，"
                "建议结合推理速度、稳定性和业务偏好再做最终选择。"
            )
    return best_model, reason


def _write_benchmark_report(result: BenchmarkResult, output_dir: Path) -> Path:
    best_model, recommendation = _recommendation(result.comparison)
    train = result.train_results[0] if result.train_results else None
    positive_rate = (train.positive_frames / train.frame_count) if train and train.frame_count else 0.0
    report_path = output_dir / "benchmark_report.md"
    lines = [
        "# Benchmark 模型训练与评测报告",
        "",
        "## 数据与任务",
        "",
        f"- 训练目录：`{result.train_data_dir}`",
        f"- 评测目录：`{result.eval_data_dir}`",
        f"- 输出目录：`{result.output_dir}`",
        f"- 参与模型：`{', '.join(result.model_names)}`",
        "",
        "## 训练数据概览",
        "",
    ]
    if train:
        lines.extend(
            [
                f"- 训练记录数：`{len(train.record_ids)}`",
                f"- 训练帧数：`{train.frame_count}`",
                f"- 正样本帧数：`{train.positive_frames}`",
                f"- 正样本比例：`{_fmt(positive_rate)}`",
                f"- 货框训练样本数：`{train.box_samples}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 评测集模型对比",
            "",
            _comparison_markdown_table(result.comparison),
            "",
            "## 结论",
            "",
            recommendation,
            "",
            "## 输出文件",
            "",
            "- `benchmark_summary.json`：完整训练、测试与对比结果。",
            "- `<model>/train_result.json`：单模型训练结果。",
            "- `<model>/eval_report.json`：单模型 Test 集评测报告。",
            "- `<model>/eval_predictions_*.json`：单模型 Test 集逐帧预测结果。",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
