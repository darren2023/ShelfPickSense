"""评测指标与评测器。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from analysis.features.registry import FeatureRegistry, default_registry
from analysis.models import PickingModel, SklearnPickingModel
from analysis.records import FramePersons, RecordData, load_all_records


@dataclass
class PickingMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    support_positive: int
    support_negative: int
    tp: int
    fp: int
    fn: int
    tn: int
    negative_precision: float = 0.0
    negative_recall: float = 0.0
    negative_f1: float = 0.0
    macro_f1: float = 0.0
    balanced_accuracy: float = 0.0


@dataclass
class BoxMetrics:
    """仅在真值为取货且含 confirmed_box_tokens 的帧上评测。"""

    frame_count: int
    exact_match_ratio: float
    any_hit_ratio: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    tp: int
    fp: int
    fn: int


@dataclass
class ModelEvaluation:
    model_name: str
    data_dir: str
    record_ids: list[str]
    picking: PickingMetrics
    box: BoxMetrics
    evaluated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "data_dir": self.data_dir,
            "record_ids": self.record_ids,
            "picking": asdict(self.picking),
            "box": asdict(self.box),
            "evaluated_at": self.evaluated_at,
            "extra": self.extra,
        }


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def compute_picking_metrics(y_true: list[bool], y_pred: list[bool]) -> PickingMetrics:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    negative_precision = _safe_div(tn, tn + fn)
    negative_recall = _safe_div(tn, tn + fp)
    negative_f1 = _safe_div(2 * negative_precision * negative_recall, negative_precision + negative_recall)
    macro_f1 = (f1 + negative_f1) / 2.0
    balanced_accuracy = (recall + negative_recall) / 2.0
    accuracy = _safe_div(tp + tn, len(y_true))
    return PickingMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        support_positive=sum(y_true),
        support_negative=len(y_true) - sum(y_true),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        negative_precision=negative_precision,
        negative_recall=negative_recall,
        negative_f1=negative_f1,
        macro_f1=macro_f1,
        balanced_accuracy=balanced_accuracy,
    )


def compute_box_metrics(
    true_tokens_list: list[set[str]],
    pred_tokens_list: list[set[str]],
) -> BoxMetrics:
    if not true_tokens_list:
        return BoxMetrics(
            frame_count=0,
            exact_match_ratio=0.0,
            any_hit_ratio=0.0,
            micro_precision=0.0,
            micro_recall=0.0,
            micro_f1=0.0,
            tp=0,
            fp=0,
            fn=0,
        )

    exact = 0
    any_hit = 0
    tp = fp = fn = 0
    for true_tokens, pred_tokens in zip(true_tokens_list, pred_tokens_list):
        if true_tokens == pred_tokens:
            exact += 1
        if true_tokens & pred_tokens:
            any_hit += 1
        tp += len(true_tokens & pred_tokens)
        fp += len(pred_tokens - true_tokens)
        fn += len(true_tokens - pred_tokens)

    micro_p = _safe_div(tp, tp + fp)
    micro_r = _safe_div(tp, tp + fn)
    micro_f1 = _safe_div(2 * micro_p * micro_r, micro_p + micro_r)
    n = len(true_tokens_list)
    return BoxMetrics(
        frame_count=n,
        exact_match_ratio=_safe_div(exact, n),
        any_hit_ratio=_safe_div(any_hit, n),
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f1=micro_f1,
        tp=tp,
        fp=fp,
        fn=fn,
    )


def predict_record(
    model: PickingModel,
    record: RecordData,
    registry: FeatureRegistry | None = None,
    frames: list[FramePersons] | None = None,
) -> list[dict[str, Any]]:
    reg = registry or default_registry()
    assert isinstance(model, SklearnPickingModel)
    results: list[dict[str, Any]] = []

    for frame in frames if frames is not None else record.frames():
        frame_feat = reg.extract_frame_features(record, frame)
        x = frame_feat.to_vector(model.frame_feature_names)
        pred = model.predict_frame(x, record_id=record.record_id, frame_idx=frame.frame_idx)

        box_tokens: list[str] = []
        if pred.is_picking and model.box_clf is not None and model.box_feature_names:
            per_box = reg.extract_per_box_features(record, frame)
            box_inputs = [(pb.box_token, pb.to_vector(model.box_feature_names)) for pb in per_box]
            box_tokens = model.predict_boxes_for_frame(box_inputs)

        results.append(
            {
                "record_id": record.record_id,
                "frame_idx": frame.frame_idx,
                "is_picking": pred.is_picking,
                "picking_prob": pred.picking_prob,
                "predicted_box_tokens": box_tokens,
            }
        )
    return results


class Evaluator:
    """对多个模型在同一数据集上评测，便于增量数据后重复评测。"""

    def __init__(
        self,
        records: list[RecordData],
        registry: FeatureRegistry | None = None,
    ) -> None:
        self.records = records
        self.registry = registry or default_registry()
        logger.debug("初始化评测器: records={}", len(records))

    def evaluate(
        self,
        model: PickingModel,
        *,
        data_dir: str,
        predictions_output_path: Path | None = None,
    ) -> ModelEvaluation:
        logger.info("开始评测: model={}, records={}", getattr(model, "name", model.__class__.__name__), len(self.records))
        y_true: list[bool] = []
        y_pred: list[bool] = []
        true_boxes: list[set[str]] = []
        pred_boxes: list[set[str]] = []
        prediction_rows: list[dict[str, Any]] = []

        for record in self.records:
            frames = record.frames()
            logger.debug("评测记录: record_id={}, frames={}", record.record_id, len(frames))
            preds = predict_record(model, record, self.registry, frames=frames)
            pred_by_frame = {p["frame_idx"]: p for p in preds}

            for frame in frames:
                label = record.labels.label_for(frame.frame_idx)
                pred = pred_by_frame.get(frame.frame_idx, {})
                true_is_picking = label.is_picking
                pred_is_picking = bool(pred.get("is_picking"))
                true_box_tokens = list(label.confirmed_box_tokens)
                pred_box_tokens = list(pred.get("predicted_box_tokens") or [])
                y_true.append(true_is_picking)
                y_pred.append(pred_is_picking)

                prediction_rows.append(
                    {
                        "record_id": record.record_id,
                        "frame_idx": frame.frame_idx,
                        "true_is_picking": true_is_picking,
                        "pred_is_picking": pred_is_picking,
                        "picking_prob": float(pred.get("picking_prob") or 0.0),
                        "true_box_tokens": true_box_tokens,
                        "predicted_box_tokens": pred_box_tokens,
                        "box_exact_match": set(true_box_tokens) == set(pred_box_tokens),
                    }
                )

                if label.is_picking and label.confirmed_box_tokens:
                    true_boxes.append(set(label.confirmed_box_tokens))
                    pred_boxes.append(set(pred.get("predicted_box_tokens") or []))

        picking = compute_picking_metrics(y_true, y_pred)
        box = compute_box_metrics(true_boxes, pred_boxes)
        logger.info(
            "评测指标: model={}, macro_f1={:.4f}, picking_f1={:.4f}, recall={:.4f}, precision={:.4f}, box_f1={:.4f}",
            getattr(model, "name", model.__class__.__name__),
            picking.macro_f1,
            picking.f1,
            picking.recall,
            picking.precision,
            box.micro_f1,
        )
        report = ModelEvaluation(
            model_name=getattr(model, "name", model.__class__.__name__),
            data_dir=data_dir,
            record_ids=[r.record_id for r in self.records],
            picking=picking,
            box=box,
            extra={
                "frame_count": len(y_true),
                "positive_frames": sum(y_true),
                "box_eval_frames": len(true_boxes),
            },
        )
        if predictions_output_path is not None:
            save_predictions(prediction_rows, predictions_output_path)
            report.extra["predictions_path"] = str(Path(predictions_output_path).resolve())
        return report


def evaluate_model(
    model_path: Path,
    data_dir: Path,
    *,
    registry: FeatureRegistry | None = None,
    predictions_output_path: Path | None = None,
) -> ModelEvaluation:
    logger.info("加载评测数据: {}", data_dir)
    records = load_all_records(data_dir)
    logger.info("加载模型: {}", model_path)
    model = SklearnPickingModel.load(model_path)
    evaluator = Evaluator(records, registry=registry)
    return evaluator.evaluate(
        model,
        data_dir=str(data_dir.resolve()),
        predictions_output_path=predictions_output_path,
    )


def save_predictions(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "prediction_count": len(rows),
        "predictions": rows,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("预测结果已保存: {}", output_path)
    return output_path


def save_report(report: ModelEvaluation, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("评测报告已保存: {}", output_path)
    return output_path


def _report_to_comparison_row(report: ModelEvaluation) -> dict[str, Any]:
    return {
        "model_name": report.model_name,
        "macro_f1": report.picking.macro_f1,
        "balanced_accuracy": report.picking.balanced_accuracy,
        "picking_f1": report.picking.f1,
        "negative_f1": report.picking.negative_f1,
        "picking_recall": report.picking.recall,
        "picking_precision": report.picking.precision,
        "box_exact_match": report.box.exact_match_ratio,
        "box_micro_f1": report.box.micro_f1,
        "evaluated_at": report.evaluated_at,
    }


def compare_reports(reports: list[ModelEvaluation]) -> list[dict[str, Any]]:
    """按 Macro-F1 排序的模型对比摘要。"""
    rows = [_report_to_comparison_row(r) for r in reports]
    return sorted(rows, key=lambda x: x["macro_f1"], reverse=True)


def compare_reports_with_baseline(
    reports: list[ModelEvaluation],
    baseline: ModelEvaluation | None,
) -> list[dict[str, Any]]:
    """在模型对比中加入规则基线，并标注是否超过基线。"""
    ml_rows = compare_reports(reports)
    if baseline is None:
        return ml_rows

    baseline_macro = float(baseline.picking.macro_f1)
    baseline_row = _report_to_comparison_row(baseline)
    baseline_row["is_baseline"] = True
    baseline_row["beats_baseline"] = None
    baseline_row["macro_f1_delta"] = 0.0

    for row in ml_rows:
        delta = float(row["macro_f1"]) - baseline_macro
        row["is_baseline"] = False
        row["baseline_macro_f1"] = baseline_macro
        row["macro_f1_delta"] = delta
        row["beats_baseline"] = delta > 0.0

    return [baseline_row, *ml_rows]
