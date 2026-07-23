"""规则碰撞基线评测（用于对比 ML 模型是否超过规则方法）。"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from loguru import logger

from analysis.annotation import BoxInfo, build_box_index
from analysis.evaluation import (
    ModelEvaluation,
    compute_box_metrics,
    compute_picking_metrics,
    save_predictions,
    save_report,
)
from analysis.records import RecordData

RULE_COLLISION_BASELINE_NAME = "rule_collision"


def _build_box_index_for_record(record: RecordData) -> dict[str, BoxInfo]:
    return build_box_index(
        record.annotation,
        infer_w=record.infer_width,
        infer_h=record.infer_height,
    )


def _load_external_collision_processor():
    box_human_det_path = Path(__file__).resolve().parents[3] / "box_human_det"
    if str(box_human_det_path) not in sys.path:
        sys.path.insert(0, str(box_human_det_path))
    logger.info("已添加到 sys.path: {}", box_human_det_path)

    try:
        from services.event_engine.collision import CollisionProcessor
    except ImportError as exc:
        raise RuntimeError(f"无法导入外部 collision.py: {exc}") from exc
    return CollisionProcessor


def _external_collision_boxes(record: RecordData) -> list[dict[str, Any]]:
    import numpy as np

    if not record.box_index:
        record.box_index = _build_box_index_for_record(record)

    boxes: list[dict[str, Any]] = []
    for _token, box_info in record.box_index.items():
        polygon = np.array(box_info.polygon, dtype=np.float32).reshape(-1, 1, 2)
        boxes.append(
            {
                "token": box_info.token,
                "shelf_code": box_info.shelf_code,
                "box_id": box_info.box_id,
                "orig_contour": polygon,
            }
        )
    return boxes


def predict_record_with_external_collision(
    record: RecordData,
    *,
    collision_processor_cls: Any | None = None,
    video_fps: float = 15.0,
    alarm_min_consecutive_frames: int = 3,
    alarm_cooldown_frames: int = 0,
    pose_frame_interval: int = 1,
) -> list[dict[str, Any]]:
    """使用 box_human_det/services/event_engine/collision.py 生成逐帧预测。"""
    CollisionProcessor = collision_processor_cls or _load_external_collision_processor()
    processor = CollisionProcessor(
        _external_collision_boxes(record),
        alarm_min_consecutive_frames=alarm_min_consecutive_frames,
        alarm_cooldown_frames=alarm_cooldown_frames,
        video_fps=video_fps,
    )

    results: list[dict[str, Any]] = []
    for frame in record.frames():
        is_pose_processed = not (pose_frame_interval > 1 and (frame.frame_idx - 1) % pose_frame_interval != 0)
        if not is_pose_processed:
            output = {"collisions": [], "alarm_collisions": []}
        else:
            output = processor.process(
                {
                    "frame_idx": frame.frame_idx,
                    "persons": frame.persons,
                }
            )
        alarm_tokens = list(output.get("alarm_collisions") or [])
        collision_tokens = list(output.get("collisions") or [])
        is_picking = bool(alarm_tokens)
        results.append(
            {
                "record_id": record.record_id,
                "frame_idx": frame.frame_idx,
                "is_pose_processed": is_pose_processed,
                "is_picking": is_picking,
                "picking_prob": 1.0 if is_picking else 0.0,
                "predicted_box_tokens": alarm_tokens if is_picking else collision_tokens,
                "collision_collisions": collision_tokens,
                "collision_alarm_collisions": alarm_tokens,
            }
        )
    return results


def evaluate_external_collision_baseline(
    records: list[RecordData],
    *,
    data_dir: str,
    video_fps: float = 15.0,
    alarm_min_consecutive_frames: int = 3,
    alarm_cooldown_frames: int = 0,
    pose_frame_interval: int = 1,
    model_name: str = RULE_COLLISION_BASELINE_NAME,
    predictions_output_path: Path | None = None,
) -> ModelEvaluation:
    logger.info("开始外部 collision.py 基线评测: records={}", len(records))
    y_true: list[bool] = []
    y_pred: list[bool] = []
    true_boxes: list[set[str]] = []
    pred_boxes: list[set[str]] = []
    prediction_rows: list[dict[str, Any]] = []
    CollisionProcessor = _load_external_collision_processor()
    source_frame_count = 0
    skipped_pose_frame_count = 0

    for record in records:
        preds = predict_record_with_external_collision(
            record,
            collision_processor_cls=CollisionProcessor,
            video_fps=video_fps,
            alarm_min_consecutive_frames=alarm_min_consecutive_frames,
            alarm_cooldown_frames=alarm_cooldown_frames,
            pose_frame_interval=pose_frame_interval,
        )
        pred_by_frame = {p["frame_idx"]: p for p in preds}

        for frame in record.frames():
            source_frame_count += 1
            label = record.labels.label_for(frame.frame_idx)
            pred = pred_by_frame.get(frame.frame_idx, {})
            if not pred.get("is_pose_processed", True):
                skipped_pose_frame_count += 1
                continue
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
                    "collision_collisions": list(pred.get("collision_collisions") or []),
                    "collision_alarm_collisions": list(pred.get("collision_alarm_collisions") or []),
                }
            )

            if label.is_picking and label.confirmed_box_tokens:
                true_boxes.append(set(label.confirmed_box_tokens))
                pred_boxes.append(set(pred_box_tokens))

    picking = compute_picking_metrics(y_true, y_pred)
    box = compute_box_metrics(true_boxes, pred_boxes)
    logger.info(
        "外部 collision.py 基线指标: macro_f1={:.4f}, picking_f1={:.4f}, box_f1={:.4f}",
        picking.macro_f1,
        picking.f1,
        box.micro_f1,
    )
    report = ModelEvaluation(
        model_name=model_name,
        data_dir=data_dir,
        record_ids=[r.record_id for r in records],
        picking=picking,
        box=box,
        extra={
            "frame_count": len(y_true),
            "source_frame_count": source_frame_count,
            "skipped_pose_frame_count": skipped_pose_frame_count,
            "positive_frames": sum(y_true),
            "box_eval_frames": len(true_boxes),
            "kind": model_name,
            "source": "box_human_det/services/event_engine/collision.py",
            "video_fps": float(video_fps),
            "alarm_min_consecutive_frames": int(alarm_min_consecutive_frames),
            "alarm_cooldown_frames": int(alarm_cooldown_frames),
            "pose_frame_interval": int(pose_frame_interval),
        },
    )
    if predictions_output_path is not None:
        save_predictions(prediction_rows, predictions_output_path)
        report.extra["predictions_path"] = str(Path(predictions_output_path).resolve())
    return report


def run_external_collision_baseline(
    records: list[RecordData],
    *,
    data_dir: str,
    output_dir: Path,
    video_fps: float = 15.0,
    alarm_min_consecutive_frames: int = 3,
    alarm_cooldown_frames: int = 0,
    pose_frame_interval: int = 1,
    model_name: str = RULE_COLLISION_BASELINE_NAME,
    predictions_filename: str = "eval_predictions_rule_collision.json",
) -> ModelEvaluation:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate_external_collision_baseline(
        records,
        data_dir=data_dir,
        video_fps=video_fps,
        alarm_min_consecutive_frames=alarm_min_consecutive_frames,
        alarm_cooldown_frames=alarm_cooldown_frames,
        pose_frame_interval=pose_frame_interval,
        model_name=model_name,
        predictions_output_path=output_dir / predictions_filename,
    )
    save_report(report, output_dir / "eval_report.json")
    return report
