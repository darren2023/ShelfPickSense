"""实时逐帧推理。

面向外部应用集成：
- `RealtimePickingPredictor()` 可以先空初始化；
- 模型通过 `load_model()` 按需加载；
- 标注可以通过 `predictor.annotation = data` 赋值，或 `load_annotation()` 从文件加载；
- `predict_frame()` 直接接收当前帧骨架数据。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from analysis.annotation import BoxInfo, build_box_index, load_annotation
from analysis.features.base import FeatureContext
from analysis.features.registry import FeatureRegistry, default_registry
from analysis.labels import RecordLabels
from analysis.models import SklearnPickingModel
from analysis.records import FramePersons, RecordData


@dataclass
class RealtimePrediction:
    record_id: str
    frame_idx: int
    is_picking: bool
    picking_prob: float
    predicted_box_tokens: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RealtimePickingPredictor:
    """逐帧取货推理器。

    推荐外部集成方式：

    ```python
    predictor = RealtimePickingPredictor(record_id="camera_01")
    predictor.load_model("models/rf")
    predictor.set_infer_size(640, 480)
    predictor.annotation = annotation_dict

    pred = predictor.predict_frame(
        skeleton_persons,
        frame_idx=123,
        timestamp_sec=4.92,
    )
    ```

    `predict_frame()` 的骨架输入支持：
    - `list[dict]`：当前帧 persons 列表；
    - `dict`：包含 `persons` / `skeletons` / 单个人体 `keypoints`；
    - `FramePersons`。
    """

    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        annotation: dict[str, Any] | None = None,
        annotation_path: Path | None = None,
        infer_width: float | None = None,
        infer_height: float | None = None,
        record_id: str = "realtime",
        registry: FeatureRegistry | None = None,
    ) -> None:
        self.registry = registry or default_registry()
        self.model: SklearnPickingModel | None = None
        self._annotation: dict[str, Any] = {}
        self.infer_width = float(infer_width or 0.0)
        self.infer_height = float(infer_height or 0.0)
        self.box_index: dict[str, BoxInfo] = {}
        self.box_tokens: list[str] = []
        self._frame_history: dict[int, FramePersons] = {}
        self.record = self._make_record(record_id)

        if model_dir is not None:
            self.load_model(model_dir)
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
        self.record.annotation = self._annotation
        self._rebuild_box_index()

    def load_model(self, model_dir: Path) -> None:
        self.model = SklearnPickingModel.load(Path(model_dir))
        logger.info("实时推理模型已加载: {}", model_dir)

    def set_infer_size(self, infer_width: float, infer_height: float) -> None:
        self.infer_width = float(infer_width)
        self.infer_height = float(infer_height)
        self.record.infer_width = self.infer_width
        self.record.infer_height = self.infer_height
        self._rebuild_box_index()

    def set_annotation(
        self,
        annotation: dict[str, Any],
        *,
        infer_width: float | None = None,
        infer_height: float | None = None,
    ) -> None:
        if infer_width is not None and infer_height is not None:
            self.set_infer_size(infer_width, infer_height)
        self.annotation = annotation

    def load_annotation(
        self,
        annotation_path: Path,
        *,
        infer_width: float | None = None,
        infer_height: float | None = None,
    ) -> None:
        self.set_annotation(
            load_annotation(Path(annotation_path)),
            infer_width=infer_width,
            infer_height=infer_height,
        )

    @classmethod
    def from_record_dir(
        cls,
        *,
        model_dir: Path | None = None,
        record_dir: Path,
        infer_width: float | None = None,
        infer_height: float | None = None,
        registry: FeatureRegistry | None = None,
    ) -> RealtimePickingPredictor:
        """从已有记录目录初始化，便于使用其 annotation.json 和坐标尺寸。"""
        from analysis.records import load_record

        record = load_record(Path(record_dir))
        predictor = cls(
            model_dir=model_dir,
            infer_width=float(infer_width if infer_width is not None else record.infer_width),
            infer_height=float(infer_height if infer_height is not None else record.infer_height),
            record_id=record.record_id,
            registry=registry,
        )
        predictor.annotation = record.annotation
        return predictor

    def predict_frame(
        self,
        skeleton_data: dict[str, Any] | list[dict[str, Any]] | FramePersons,
        *,
        frame_idx: int | None = None,
        timestamp_sec: float | None = None,
    ) -> RealtimePrediction:
        if self.model is None:
            raise RuntimeError("模型尚未加载，请先调用 load_model()")
        if not self.annotation:
            raise RuntimeError("annotation 尚未设置，请赋值 predictor.annotation 或调用 load_annotation()")
        if self.infer_width <= 0 or self.infer_height <= 0:
            raise RuntimeError("推理尺寸尚未设置，请调用 set_infer_size()")

        frame_data = self._normalize_frame(
            skeleton_data,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
        )
        self._remember_frame(frame_data)
        ctx = FeatureContext(
            record=self.record,
            frame=frame_data,
            box_index=self.box_index,
            box_tokens=self.box_tokens,
            frame_index=self._frame_history,
        )
        frame_features = self.registry.extract_frame_features_from_context(ctx)
        x = frame_features.to_vector(self.model.frame_feature_names)
        pred = self.model.predict_frame(
            x,
            record_id=self.record.record_id,
            frame_idx=frame_data.frame_idx,
        )

        predicted_box_tokens: list[str] = []
        if pred.is_picking and self.model.box_clf is not None and self.model.box_feature_names:
            per_box = self.registry.extract_per_box_features_from_context(ctx)
            box_inputs = [(pb.box_token, pb.to_vector(self.model.box_feature_names)) for pb in per_box]
            predicted_box_tokens = self.model.predict_boxes_for_frame(box_inputs)

        return RealtimePrediction(
            record_id=self.record.record_id,
            frame_idx=frame_data.frame_idx,
            is_picking=pred.is_picking,
            picking_prob=pred.picking_prob,
            predicted_box_tokens=predicted_box_tokens,
        )

    def _remember_frame(self, frame: FramePersons, *, keep_back: int = 7) -> None:
        self._frame_history[frame.frame_idx] = frame
        min_idx = frame.frame_idx - keep_back
        self._frame_history = {idx: fr for idx, fr in self._frame_history.items() if idx >= min_idx}

    def _make_record(self, record_id: str) -> RecordData:
        return RecordData(
            record_id=record_id,
            record_dir=Path("."),
            skeleton=pd.DataFrame(),
            annotation=self._annotation,
            event_review=None,
            labels=RecordLabels(record_id=record_id),
            infer_width=self.infer_width,
            infer_height=self.infer_height,
            box_tokens=[],
        )

    def _rebuild_box_index(self) -> None:
        if not self._annotation or self.infer_width <= 0 or self.infer_height <= 0:
            self.box_index = {}
            self.box_tokens = []
        else:
            self.box_index = build_box_index(
                self._annotation,
                infer_w=self.infer_width,
                infer_h=self.infer_height,
            )
            self.box_tokens = sorted(self.box_index.keys())
        self.record.box_tokens = self.box_tokens
        logger.debug("实时推理货框索引已更新: boxes={}", len(self.box_tokens))

    def _normalize_frame(
        self,
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
