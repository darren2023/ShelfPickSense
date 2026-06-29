"""规则碰撞基线评测（用于对比 ML 模型是否超过规则方法）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from analysis.annotation import build_box_index, load_annotation
from analysis.evaluation import (
    ModelEvaluation,
    compute_box_metrics,
    compute_picking_metrics,
    save_predictions,
    save_report,
)
from analysis.records import FramePersons, RecordData
from analysis.rule_collision import CollisionParams, RuleCollisionProcessor, build_box_index_for_record

RULE_BASELINE_NAME = "rule_baseline"


@dataclass
class RealtimeRulePrediction:
    record_id: str
    frame_idx: int
    is_picking: bool
    picking_prob: float
    predicted_box_tokens: list[str]
    rule_collisions: list[str]
    rule_alarm_collisions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _frame_from_skeleton_data(
    skeleton_data: dict[str, Any] | list[dict[str, Any]] | FramePersons,
    *,
    frame_idx: int | None,
    timestamp_sec: float | None,
) -> FramePersons:
    if isinstance(skeleton_data, FramePersons):
        return FramePersons(
            frame_idx=int(frame_idx if frame_idx is not None else skeleton_data.frame_idx),
            timestamp_sec=float(timestamp_sec if timestamp_sec is not None else skeleton_data.timestamp_sec),
            persons=skeleton_data.persons,
        )
    if isinstance(skeleton_data, list):
        return FramePersons(
            frame_idx=int(frame_idx or 0),
            timestamp_sec=float(timestamp_sec or 0.0),
            persons=skeleton_data,
        )

    try:
        resolved_frame_idx = int(
            frame_idx
            if frame_idx is not None
            else skeleton_data.get("frame_idx") or skeleton_data.get("source_frame_idx") or 0
        )
    except (TypeError, ValueError):
        resolved_frame_idx = 0
    try:
        resolved_timestamp = float(
            timestamp_sec if timestamp_sec is not None else skeleton_data.get("timestamp_sec") or 0.0
        )
    except (TypeError, ValueError):
        resolved_timestamp = 0.0

    persons = skeleton_data.get("persons") or skeleton_data.get("skeletons")
    if persons is None and "keypoints" in skeleton_data:
        persons = [skeleton_data]
    if not isinstance(persons, list):
        persons = []
    return FramePersons(
        frame_idx=resolved_frame_idx,
        timestamp_sec=resolved_timestamp,
        persons=persons,
    )


def _prediction_from_processor_output(
    *,
    record_id: str,
    frame_idx: int,
    output: dict[str, Any],
) -> RealtimeRulePrediction:
    alarm_tokens = list(output.get("alarm_collisions") or [])
    collision_tokens = list(output.get("collisions") or [])
    is_picking = bool(alarm_tokens)
    return RealtimeRulePrediction(
        record_id=record_id,
        frame_idx=frame_idx,
        is_picking=is_picking,
        picking_prob=1.0 if is_picking else 0.0,
        predicted_box_tokens=alarm_tokens if is_picking else collision_tokens,
        rule_collisions=collision_tokens,
        rule_alarm_collisions=alarm_tokens,
    )


class RealtimeRulePredictor:
    """逐帧规则碰撞推理，用法对齐 `RealtimePickingPredictor`。"""

    def __init__(
        self,
        *,
        annotation: dict[str, Any] | None = None,
        annotation_path: Path | None = None,
        infer_width: float | None = None,
        infer_height: float | None = None,
        record_id: str = "realtime",
        video_fps: float = 25.0,
        params: CollisionParams | None = None,
    ) -> None:
        self.record_id = record_id
        self.video_fps = float(video_fps)
        self.params = params
        self.infer_width = float(infer_width or 0.0)
        self.infer_height = float(infer_height or 0.0)
        self._annotation: dict[str, Any] = {}
        self._processor: RuleCollisionProcessor | None = None

        if annotation is not None:
            self.annotation = annotation
        elif annotation_path is not None:
            self.load_annotation(annotation_path)

    @property
    def annotation(self) -> dict[str, Any]:
        return self._annotation

    @annotation.setter
    def annotation(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise ValueError("annotation 必须是 dict")
        self._annotation = data
        self._rebuild_processor()

    def set_infer_size(self, infer_width: float, infer_height: float) -> None:
        self.infer_width = float(infer_width)
        self.infer_height = float(infer_height)
        self._rebuild_processor()

    def load_annotation(
        self,
        annotation_path: Path,
        *,
        infer_width: float | None = None,
        infer_height: float | None = None,
    ) -> None:
        if infer_width is not None and infer_height is not None:
            self.set_infer_size(infer_width, infer_height)
        self.annotation = load_annotation(Path(annotation_path))

    @classmethod
    def from_record_dir(
        cls,
        record_dir: Path,
        *,
        infer_width: float | None = None,
        infer_height: float | None = None,
        video_fps: float = 25.0,
        params: CollisionParams | None = None,
    ) -> RealtimeRulePredictor:
        from analysis.records import load_record

        record = load_record(Path(record_dir))
        predictor = cls(
            infer_width=float(infer_width if infer_width is not None else record.infer_width),
            infer_height=float(infer_height if infer_height is not None else record.infer_height),
            record_id=record.record_id,
            video_fps=video_fps,
            params=params,
        )
        predictor.annotation = record.annotation
        return predictor

    def predict_frame(
        self,
        skeleton_data: dict[str, Any] | list[dict[str, Any]] | FramePersons,
        *,
        frame_idx: int | None = None,
        timestamp_sec: float | None = None,
    ) -> RealtimeRulePrediction:
        if self._processor is None:
            raise RuntimeError("annotation 尚未设置，请赋值 predictor.annotation 或调用 load_annotation()")
        frame = _frame_from_skeleton_data(
            skeleton_data,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
        )
        output = self._processor.process_frame(
            frame_idx=frame.frame_idx,
            persons=frame.persons,
            timestamp_sec=frame.timestamp_sec,
        )
        return _prediction_from_processor_output(
            record_id=self.record_id,
            frame_idx=frame.frame_idx,
            output=output,
        )

    def _rebuild_processor(self) -> None:
        if not self._annotation or self.infer_width <= 0 or self.infer_height <= 0:
            self._processor = None
            return
        box_index = build_box_index(
            self._annotation,
            infer_w=self.infer_width,
            infer_h=self.infer_height,
        )
        self._processor = RuleCollisionProcessor(
            box_index,
            params=self.params,
            video_fps=self.video_fps,
        )
        logger.debug("规则推理货框索引已更新: boxes={}", len(box_index))


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
        results.append(
            _prediction_from_processor_output(
                record_id=record.record_id,
                frame_idx=frame.frame_idx,
                output=output,
            ).to_dict()
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
