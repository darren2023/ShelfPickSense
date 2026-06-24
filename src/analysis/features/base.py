"""特征提取器基类与上下文。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from analysis.annotation import BoxInfo, build_box_index
from analysis.records import FramePersons, RecordData


@dataclass
class FeatureContext:
    """单帧特征提取上下文。"""

    record: RecordData
    frame: FramePersons
    box_index: dict[str, BoxInfo]
    box_tokens: list[str]
    frame_index: dict[int, FramePersons] = field(default_factory=dict)

    def prior_frame(self, offset: int) -> FramePersons | None:
        """按帧序号偏移取历史帧。offset=0 为当前帧，offset=1 为上一帧。"""
        if offset < 0:
            return None
        if offset == 0:
            return self.frame
        return self.frame_index.get(self.frame.frame_idx - offset)

    @classmethod
    def from_record(
        cls,
        record: RecordData,
        frame: FramePersons,
        *,
        frame_index: dict[int, FramePersons] | None = None,
    ) -> FeatureContext:
        box_index = build_box_index(
            record.annotation,
            infer_w=record.infer_width,
            infer_h=record.infer_height,
        )
        idx = frame_index if frame_index is not None else record.frame_index()
        return cls(
            record=record,
            frame=frame,
            box_index=box_index,
            box_tokens=sorted(box_index.keys()),
            frame_index=idx,
        )


@dataclass
class FeatureSet:
    """一帧的全局特征向量（用于取货检测）。"""

    record_id: str
    frame_idx: int
    features: dict[str, float] = field(default_factory=dict)

    def names(self) -> list[str]:
        return sorted(self.features.keys())

    def to_vector(self, feature_names: list[str] | None = None) -> np.ndarray:
        names = feature_names or self.names()
        return np.array([float(self.features.get(n, 0.0)) for n in names], dtype=np.float64)


@dataclass
class PerBoxFeatureSet:
    """单帧、单货框的特征（用于货框分类）。"""

    record_id: str
    frame_idx: int
    box_token: str
    features: dict[str, float] = field(default_factory=dict)

    def to_vector(self, feature_names: list[str]) -> np.ndarray:
        return np.array([float(self.features.get(n, 0.0)) for n in feature_names], dtype=np.float64)


class FeatureExtractor(ABC):
    """特征提取器插件接口。"""

    name: str = "base"

    @abstractmethod
    def extract_frame(self, ctx: FeatureContext) -> dict[str, float]:
        """提取帧级（全局）特征。"""

    def extract_per_box(self, ctx: FeatureContext) -> dict[str, dict[str, float]]:
        """提取每货框特征，默认无。"""
        return {}
