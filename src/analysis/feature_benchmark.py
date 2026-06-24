"""按多组特征配置批量运行 benchmark。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from analysis.benchmark import DEFAULT_MODEL_NAMES, BenchmarkResult, _fmt, run_benchmark
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
        "## 各特征配置最佳模型",
        "",
        "| 特征配置 | 最佳模型 | Macro-F1 | 帧级特征数 | 货框特征数 | 输出目录 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in batch.sets:
        selection = item.feature_selection or {}
        frame_count = len(selection.get("frame_features") or [])
        box_count = len(selection.get("box_features") or [])
        if selection.get("frame_features") is None:
            frame_count = "全部"
        if selection.get("box_features") is None:
            box_count = "全部"
        lines.append(
            "| "
            + " | ".join(
                [
                    item.name,
                    item.best_model or "-",
                    _fmt(item.best_macro_f1),
                    str(frame_count),
                    str(box_count),
                    f"`{item.output_dir}`",
                ]
            )
            + " |"
        )

    if batch.sets:
        best_item = max(batch.sets, key=lambda item: item.best_macro_f1)
        lines.extend(
            [
                "",
                "## 结论",
                "",
                (
                    f"在当前对比中，特征配置 `{best_item.name}` 下的最佳模型为 `{best_item.best_model}`，"
                    f"Macro-F1 为 {_fmt(best_item.best_macro_f1)}。"
                ),
                "",
                "各特征配置目录内仍保留完整的 `benchmark_report.md` 与单模型评测结果，便于继续下钻分析。",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


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
