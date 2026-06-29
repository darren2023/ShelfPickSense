"""规则碰撞基线评测（用于对比 ML 模型是否超过规则方法）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from analysis.evaluation import (
    ModelEvaluation,
    compute_box_metrics,
    compute_picking_metrics,
    save_predictions,
    save_report,
)
from analysis.records import RecordData
from analysis.rule_collision import CollisionParams, RuleCollisionProcessor, build_box_index_for_record

RULE_BASELINE_NAME = "rule_baseline"


def predict_record_with_rules(
    record: RecordData,
    *,
    params: CollisionParams | None = None,
) -> list[dict[str, Any]]:
    box_index = build_box_index_for_record(record)
    processor = RuleCollisionProcessor(box_index, params=params)
    results: list[dict[str, Any]] = []

    for frame in record.frames():
        output = processor.process_frame(
            frame_idx=frame.frame_idx,
            persons=frame.persons,
            timestamp_sec=frame.timestamp_sec,
        )
        alarm_tokens = list(output.get("alarm_collisions") or [])
        collision_tokens = list(output.get("collisions") or [])
        is_picking = bool(alarm_tokens)
        results.append(
            {
                "record_id": record.record_id,
                "frame_idx": frame.frame_idx,
                "is_picking": is_picking,
                "picking_prob": 1.0 if is_picking else 0.0,
                "predicted_box_tokens": alarm_tokens if is_picking else collision_tokens,
                "rule_collisions": collision_tokens,
                "rule_alarm_collisions": alarm_tokens,
            }
        )
    return results


def evaluate_rule_baseline(
    records: list[RecordData],
    *,
    data_dir: str,
    params: CollisionParams | None = None,
    predictions_output_path: Path | None = None,
) -> ModelEvaluation:
    logger.info("开始规则基线评测: records={}", len(records))
    y_true: list[bool] = []
    y_pred: list[bool] = []
    true_boxes: list[set[str]] = []
    pred_boxes: list[set[str]] = []
    prediction_rows: list[dict[str, Any]] = []

    for record in records:
        preds = predict_record_with_rules(record, params=params)
        pred_by_frame = {p["frame_idx"]: p for p in preds}

        for frame in record.frames():
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
                    "rule_collisions": list(pred.get("rule_collisions") or []),
                    "rule_alarm_collisions": list(pred.get("rule_alarm_collisions") or []),
                }
            )

            if label.is_picking and label.confirmed_box_tokens:
                true_boxes.append(set(label.confirmed_box_tokens))
                pred_boxes.append(set(pred_box_tokens))

    picking = compute_picking_metrics(y_true, y_pred)
    box = compute_box_metrics(true_boxes, pred_boxes)
    logger.info(
        "规则基线指标: macro_f1={:.4f}, picking_f1={:.4f}, box_f1={:.4f}",
        picking.macro_f1,
        picking.f1,
        box.micro_f1,
    )
    report = ModelEvaluation(
        model_name=RULE_BASELINE_NAME,
        data_dir=data_dir,
        record_ids=[r.record_id for r in records],
        picking=picking,
        box=box,
        extra={
            "frame_count": len(y_true),
            "positive_frames": sum(y_true),
            "box_eval_frames": len(true_boxes),
            "kind": "rule_baseline",
        },
    )
    if predictions_output_path is not None:
        save_predictions(prediction_rows, predictions_output_path)
        report.extra["predictions_path"] = str(Path(predictions_output_path).resolve())
    return report


def run_rule_baseline(
    records: list[RecordData],
    *,
    data_dir: str,
    output_dir: Path,
    params: CollisionParams | None = None,
    predictions_filename: str = "eval_predictions_rule_baseline.json",
) -> ModelEvaluation:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate_rule_baseline(
        records,
        data_dir=data_dir,
        params=params,
        predictions_output_path=output_dir / predictions_filename,
    )
    save_report(report, output_dir / "eval_report.json")
    return report
