"""按多组特征配置批量运行 benchmark。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from analysis.benchmark import (
    DEFAULT_MODEL_NAMES,
    BenchmarkResult,
    _comparison_markdown_table,
    _fmt,
    _write_benchmark_report,
    run_benchmark,
)
from analysis.train import TrainResult
from analysis.features.selection import FeatureSelection, load_feature_selection
from analysis.models import SUPPORTED_MODEL_NAMES


@dataclass(frozen=True)
class FeatureBenchmarkSetSpec:
    name: str
    feature_config: str = ""
    frame_features: list[str] | None = None
    box_features: list[str] | None = None


@dataclass
class FeatureBenchmarkPlan:
    train_data_dir: Path
    output_dir: Path
    eval_data_dir: Path | None = None
    model_names: list[str] = field(default_factory=lambda: list(DEFAULT_MODEL_NAMES))
    jobs: int = 8
    sets: list[FeatureBenchmarkSetSpec] = field(default_factory=list)
    source_path: str = ""


@dataclass
class FeatureBenchmarkSetResult:
    name: str
    output_dir: str
    feature_selection: dict[str, Any] | None
    best_model: str
    best_macro_f1: float
    benchmark: BenchmarkResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_dir": self.output_dir,
            "feature_selection": self.feature_selection,
            "best_model": self.best_model,
            "best_macro_f1": self.best_macro_f1,
            "benchmark": self.benchmark.to_dict(),
        }


@dataclass
class FeatureBenchmarkBatchResult:
    train_data_dir: str
    eval_data_dir: str
    output_dir: str
    model_names: list[str]
    sets: list[FeatureBenchmarkSetResult]
    benchmarked_at: str
    report_path: str
    summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_data_dir": self.train_data_dir,
            "eval_data_dir": self.eval_data_dir,
            "output_dir": self.output_dir,
            "model_names": self.model_names,
            "sets": [item.to_dict() for item in self.sets],
            "benchmarked_at": self.benchmarked_at,
            "report_path": self.report_path,
            "summary_path": self.summary_path,
        }


def _safe_dir_name(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text or "").strip())
    return safe.strip("_") or "feature_set"


def _optional_string_list(data: dict[str, Any], key: str) -> list[str] | None:
    if key not in data:
        return None
    value = data[key]
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} 必须是字符串列表")
    return list(dict.fromkeys(value))


def _parse_set_spec(data: dict[str, Any], index: int) -> FeatureBenchmarkSetSpec:
    if not isinstance(data, dict):
        raise ValueError(f"feature_sets[{index}] 必须是对象")
    name = str(data.get("name") or "").strip()
    if not name:
        raise ValueError(f"feature_sets[{index}] 缺少 name")
    feature_config = str(data.get("feature_config") or data.get("feature_config_path") or "").strip()
    return FeatureBenchmarkSetSpec(
        name=name,
        feature_config=feature_config,
        frame_features=_optional_string_list(data, "frame_features") or _optional_string_list(data, "frame"),
        box_features=_optional_string_list(data, "box_features") or _optional_string_list(data, "box"),
    )


def load_feature_benchmark_plan(path: str | Path) -> FeatureBenchmarkPlan:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("批量 benchmark 配置必须是 JSON 对象")

    train_data_dir = str(data.get("train_data_dir") or data.get("data_dir") or "").strip()
    if not train_data_dir:
        raise ValueError("配置缺少 train_data_dir 或 data_dir")
    output_dir = str(data.get("output_dir") or data.get("output") or "").strip()
    if not output_dir:
        raise ValueError("配置缺少 output_dir 或 output")

    eval_data_dir = str(data.get("eval_data_dir") or "").strip() or None
    model_names = _optional_string_list(data, "models") or _optional_string_list(data, "model_names")
    if model_names:
        unknown = [name for name in model_names if name not in SUPPORTED_MODEL_NAMES]
        if unknown:
            raise ValueError(f"未知模型: {', '.join(unknown)}")

    raw_sets = data.get("feature_sets") or data.get("sets") or []
    if not isinstance(raw_sets, list) or not raw_sets:
        raise ValueError("配置缺少 feature_sets 或 sets")

    return FeatureBenchmarkPlan(
        train_data_dir=Path(train_data_dir),
        eval_data_dir=Path(eval_data_dir) if eval_data_dir else None,
        output_dir=Path(output_dir),
        model_names=list(model_names or DEFAULT_MODEL_NAMES),
        jobs=int(data.get("jobs") or 8),
        sets=[_parse_set_spec(item, index) for index, item in enumerate(raw_sets)],
        source_path=str(config_path.resolve()),
    )


def resolve_feature_selection(spec: FeatureBenchmarkSetSpec, *, base_dir: Path) -> FeatureSelection | None:
    if spec.feature_config:
        config_path = _resolve_config_path(spec.feature_config, base_dir=base_dir)
        return load_feature_selection(config_path)
    if spec.frame_features is not None or spec.box_features is not None:
        return FeatureSelection(
            frame_features=spec.frame_features,
            box_features=spec.box_features,
        )
    return None


def _resolve_config_path(path: str, *, base_dir: Path) -> Path:
    """解析特征配置文件路径，兼容相对项目根目录与相对 plan 文件目录两种写法。"""
    config_path = Path(path)
    if config_path.is_absolute():
        if not config_path.is_file():
            raise FileNotFoundError(f"特征配置文件不存在: {config_path}")
        return config_path

    candidates = [config_path, base_dir / config_path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(item.resolve()) for item in candidates)
    raise FileNotFoundError(f"特征配置文件不存在: {path}（已尝试: {tried}）")


def _best_from_benchmark(result: BenchmarkResult) -> tuple[str, float]:
    if not result.comparison:
        return "", 0.0
    best = result.comparison[0]
    return str(best.get("model_name") or ""), float(best.get("macro_f1") or 0.0)


def _feature_names_block(selection: dict[str, Any] | None) -> list[str]:
    if not selection:
        return ["- 帧级特征：全部", "- 货框特征：全部"]
    frame_names = selection.get("frame_features")
    box_names = selection.get("box_features")
    lines = []
    if frame_names is None:
        lines.append("- 帧级特征：全部")
    else:
        lines.append(f"- 帧级特征（{len(frame_names)}）：`{', '.join(frame_names)}`")
    if box_names is None:
        lines.append("- 货框特征：全部")
    else:
        lines.append(f"- 货框特征（{len(box_names)}）：`{', '.join(box_names)}`")
    if selection.get("source_path"):
        lines.append(f"- 特征配置文件：`{selection['source_path']}`")
    return lines


def _best_model_summary_row(item: FeatureBenchmarkSetResult) -> dict[str, Any]:
    if not item.benchmark.comparison:
        return {}
    for row in item.benchmark.comparison:
        if row.get("model_name") == item.best_model:
            return row
    return item.benchmark.comparison[0]


def _best_model_bullet(item: FeatureBenchmarkSetResult) -> str:
    row = _best_model_summary_row(item)
    if not row:
        return f"- **{item.name}**：无可用模型结果。"
    return (
        f"- **{item.name}**：推荐 `{item.best_model}`，"
        f"Macro-F1={_fmt(row.get('macro_f1'))}，"
        f"Balanced Acc={_fmt(row.get('balanced_accuracy'))}，"
        f"取货 F1={_fmt(row.get('picking_f1'))}，"
        f"取货 Recall={_fmt(row.get('picking_recall'))}，"
        f"货框 Micro-F1={_fmt(row.get('box_micro_f1'))}。"
    )


def _best_models_summary_table(batch: FeatureBenchmarkBatchResult) -> str:
    columns = [
        ("name", "特征配置"),
        ("best_model", "最佳模型"),
        ("macro_f1", "Macro-F1"),
        ("balanced_accuracy", "Balanced Acc"),
        ("picking_f1", "取货 F1"),
        ("picking_recall", "取货 Recall"),
        ("picking_precision", "取货 Precision"),
        ("box_micro_f1", "货框 Micro-F1"),
        ("box_exact_match", "货框精确匹配"),
    ]
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    ranked = sorted(batch.sets, key=lambda item: item.best_macro_f1, reverse=True)
    for item in ranked:
        row = _best_model_summary_row(item)
        values = []
        for key, _ in columns:
            if key == "name":
                values.append(item.name)
            elif key == "best_model":
                values.append(item.best_model or "-")
            else:
                values.append(_fmt(row.get(key)))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _write_batch_report(batch: FeatureBenchmarkBatchResult, output_dir: Path) -> Path:
    report_path = output_dir / "feature_benchmark_report.md"
    lines = [
        "# 多特征配置 Benchmark 对比报告",
        "",
        "## 任务概览",
        "",
        f"- 训练目录：`{batch.train_data_dir}`",
        f"- 评测目录：`{batch.eval_data_dir}`",
        f"- 输出目录：`{batch.output_dir}`",
        f"- 参与模型：`{', '.join(batch.model_names)}`",
        f"- 特征配置组数：`{len(batch.sets)}`",
        "",
        "## 各特征配置最佳模型汇总",
        "",
        _best_models_summary_table(batch),
        "",
        "## 各特征配置模型明细",
        "",
    ]

    for item in batch.sets:
        selection = item.feature_selection
        lines.extend([f"### {item.name}", ""])
        lines.extend(_feature_names_block(selection))
        lines.append(f"- 输出目录：`{item.output_dir}`")
        lines.append("")
        lines.append(_comparison_markdown_table(item.benchmark.comparison).rstrip())
        lines.append("")
        row = _best_model_summary_row(item)
        if row:
            lines.append(
                f"**本组推荐**：`{item.best_model}`（Macro-F1={_fmt(row.get('macro_f1'))}，"
                f"Balanced Acc={_fmt(row.get('balanced_accuracy'))}，"
                f"取货 Recall={_fmt(row.get('picking_recall'))}，"
                f"货框 Micro-F1={_fmt(row.get('box_micro_f1'))}）"
            )
        else:
            lines.append("**本组推荐**：无可用模型结果。")
        lines.extend(["", "---", ""])

    if batch.sets:
        best_item = max(batch.sets, key=lambda item: item.best_macro_f1)
        best_row = _best_model_summary_row(best_item)
        lines.extend(
            [
                "## 结论",
                "",
                "### 各特征配置最佳模型",
                "",
            ]
        )
        lines.extend(_best_model_bullet(item) for item in batch.sets)
        lines.extend(
            [
                "",
                "### 全局推荐",
                "",
                (
                    f"在当前全部特征配置对比中，**`{best_item.name}`** 下的 **`{best_item.best_model}`** 表现最好，"
                    f"Macro-F1 为 {_fmt(best_row.get('macro_f1'))}，"
                    f"Balanced Accuracy 为 {_fmt(best_row.get('balanced_accuracy'))}，"
                    f"取货 Recall 为 {_fmt(best_row.get('picking_recall'))}，"
                    f"货框 Micro-F1 为 {_fmt(best_row.get('box_micro_f1'))}。"
                ),
                "",
                "各特征配置目录内仍保留完整的 `benchmark_report.md` 与单模型评测结果，便于继续下钻分析。",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _benchmark_result_from_dict(data: dict[str, Any]) -> BenchmarkResult:
    train_results = [TrainResult(**item) for item in data.get("train_results") or []]
    return BenchmarkResult(
        train_data_dir=str(data.get("train_data_dir") or ""),
        eval_data_dir=str(data.get("eval_data_dir") or ""),
        output_dir=str(data.get("output_dir") or ""),
        model_names=list(data.get("model_names") or []),
        train_results=train_results,
        reports=[],
        comparison=list(data.get("comparison") or []),
        benchmarked_at=str(data.get("benchmarked_at") or ""),
    )


def _set_result_from_dict(data: dict[str, Any]) -> FeatureBenchmarkSetResult:
    benchmark = _benchmark_result_from_dict(data["benchmark"])
    best_model, best_macro_f1 = _best_from_benchmark(benchmark)
    return FeatureBenchmarkSetResult(
        name=str(data["name"]),
        output_dir=str(data["output_dir"]),
        feature_selection=data.get("feature_selection"),
        best_model=str(data.get("best_model") or best_model),
        best_macro_f1=float(data.get("best_macro_f1") or best_macro_f1),
        benchmark=benchmark,
    )


def load_feature_benchmark_batch_result(output_dir: Path) -> FeatureBenchmarkBatchResult:
    summary_path = Path(output_dir) / "feature_benchmark_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"未找到汇总结果: {summary_path}")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    sets = [_set_result_from_dict(item) for item in data.get("sets") or []]
    if not sets:
        raise ValueError(f"汇总结果为空: {summary_path}")
    return FeatureBenchmarkBatchResult(
        train_data_dir=str(data.get("train_data_dir") or ""),
        eval_data_dir=str(data.get("eval_data_dir") or ""),
        output_dir=str(Path(output_dir).resolve()),
        model_names=list(data.get("model_names") or []),
        sets=sets,
        benchmarked_at=str(data.get("benchmarked_at") or ""),
        report_path=str(data.get("report_path") or ""),
        summary_path=str(summary_path.resolve()),
    )


def _load_batch_result_from_plan(plan: FeatureBenchmarkPlan) -> FeatureBenchmarkBatchResult:
    output_dir = Path(plan.output_dir)
    base_dir = Path(plan.source_path).parent if plan.source_path else Path.cwd()
    eval_data_dir = plan.eval_data_dir or plan.train_data_dir
    set_results: list[FeatureBenchmarkSetResult] = []
    for spec in plan.sets:
        set_output = output_dir / _safe_dir_name(spec.name)
        summary_path = set_output / "benchmark_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"未找到特征配置 benchmark 结果: {summary_path}")
        benchmark = _benchmark_result_from_dict(json.loads(summary_path.read_text(encoding="utf-8")))
        feature_selection = resolve_feature_selection(spec, base_dir=base_dir)
        best_model, best_macro_f1 = _best_from_benchmark(benchmark)
        set_results.append(
            FeatureBenchmarkSetResult(
                name=spec.name,
                output_dir=str(set_output.resolve()),
                feature_selection=feature_selection.to_dict() if feature_selection else None,
                best_model=best_model,
                best_macro_f1=best_macro_f1,
                benchmark=benchmark,
            )
        )
    return FeatureBenchmarkBatchResult(
        train_data_dir=str(plan.train_data_dir.resolve()),
        eval_data_dir=str(eval_data_dir.resolve()),
        output_dir=str(output_dir.resolve()),
        model_names=list(plan.model_names),
        sets=set_results,
        benchmarked_at=datetime.now(timezone.utc).isoformat(),
        report_path="",
        summary_path=str((output_dir / "feature_benchmark_summary.json").resolve()),
    )


def regenerate_feature_benchmark_report(plan: FeatureBenchmarkPlan) -> FeatureBenchmarkBatchResult:
    """基于已有 benchmark 结果重新生成 Markdown 报告，不重新训练或评测模型。"""
    output_dir = Path(plan.output_dir)
    summary_path = output_dir / "feature_benchmark_summary.json"
    if summary_path.is_file():
        batch = load_feature_benchmark_batch_result(output_dir)
    else:
        logger.info("未找到 feature_benchmark_summary.json，改为从各特征配置目录加载 benchmark_summary.json")
        batch = _load_batch_result_from_plan(plan)

    for item in batch.sets:
        _write_benchmark_report(item.benchmark, Path(item.output_dir))
    report_path = _write_batch_report(batch, output_dir)
    batch.report_path = str(report_path.resolve())
    batch.summary_path = str(summary_path.resolve())
    summary_path.write_text(json.dumps(batch.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("多特征 benchmark 报告已重新生成: {}", report_path)
    return batch


def run_feature_benchmarks(plan: FeatureBenchmarkPlan) -> FeatureBenchmarkBatchResult:
    output_dir = Path(plan.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(plan.source_path).parent if plan.source_path else Path.cwd()
    eval_data_dir = plan.eval_data_dir or plan.train_data_dir

    set_results: list[FeatureBenchmarkSetResult] = []
    for spec in plan.sets:
        feature_selection = resolve_feature_selection(spec, base_dir=base_dir)
        set_output = output_dir / _safe_dir_name(spec.name)
        logger.info(
            "开始特征配置 benchmark: name={}, output={}, feature_config={}",
            spec.name,
            set_output,
            feature_selection.source_path if feature_selection else "all_features",
        )
        benchmark = run_benchmark(
            train_data_dir=plan.train_data_dir,
            eval_data_dir=eval_data_dir,
            output_dir=set_output,
            model_names=plan.model_names,
            jobs=plan.jobs,
            feature_selection=feature_selection,
        )
        best_model, best_macro_f1 = _best_from_benchmark(benchmark)
        set_results.append(
            FeatureBenchmarkSetResult(
                name=spec.name,
                output_dir=str(set_output.resolve()),
                feature_selection=feature_selection.to_dict() if feature_selection else None,
                best_model=best_model,
                best_macro_f1=best_macro_f1,
                benchmark=benchmark,
            )
        )

    summary_path = output_dir / "feature_benchmark_summary.json"
    batch = FeatureBenchmarkBatchResult(
        train_data_dir=str(plan.train_data_dir.resolve()),
        eval_data_dir=str(eval_data_dir.resolve()),
        output_dir=str(output_dir.resolve()),
        model_names=list(plan.model_names),
        sets=set_results,
        benchmarked_at=datetime.now(timezone.utc).isoformat(),
        report_path="",
        summary_path=str(summary_path.resolve()),
    )
    report_path = _write_batch_report(batch, output_dir)
    batch.report_path = str(report_path.resolve())
    summary_path.write_text(json.dumps(batch.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("多特征 benchmark 汇总已保存: {}", summary_path)
    logger.info("多特征 benchmark 报告已保存: {}", report_path)
    return batch
