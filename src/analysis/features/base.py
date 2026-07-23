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
    sample_frames: list[FramePersons] = field(default_factory=list)
    current_person: dict[str, Any] | None = None
    current_track_id: int | None = None
    _sample_index: dict[int, int] = field(default_factory=dict, repr=False, compare=False)
    _rule_hit_cache: dict[tuple[int | None, int, str | None], int] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.sample_frames and self.frame_index:
            self.sample_frames = [self.frame_index[i] for i in sorted(self.frame_index.keys())]
        if self.sample_frames and not self._sample_index:
            self._sample_index = {f.frame_idx: i for i, f in enumerate(self.sample_frames)}

    @staticmethod
    def _sample_index_from(frames: list[FramePersons]) -> dict[int, int]:
        return {f.frame_idx: i for i, f in enumerate(frames)}

    def prior_sample(self, k: int) -> FramePersons | None:
        """按采样序列回退 k 步。k=0 为当前帧，k=1 为序列中上一采样帧。"""
        if k < 0:
            return None
        if k == 0:
            return self.frame
        if not self.sample_frames:
            return None
        idx = self._sample_index.get(self.frame.frame_idx)
        if idx is None or idx - k < 0:
            return None
        return self.sample_frames[idx - k]

    def prior_frame(self, offset: int) -> FramePersons | None:
        """时序 lookup：有序列时按 prior_sample，否则按 frame_idx-offset（兼容旧路径）。"""
        if offset < 0:
            return None
        if offset == 0:
            return self.frame
        if self.sample_frames:
            return self.prior_sample(offset)
        return self.frame_index.get(self.frame.frame_idx - offset)

    @classmethod
    def from_record(
        cls,
        record: RecordData,
        frame: FramePersons,
        *,
        frame_index: dict[int, FramePersons] | None = None,
        sample_frames: list[FramePersons] | None = None,
    ) -> FeatureContext:
        box_index = record.box_index if record.box_index else build_box_index(
            record.annotation,
            infer_w=record.infer_width,
            infer_h=record.infer_height,
        )
        idx = frame_index if frame_index is not None else record.frame_index()
        if sample_frames is not None:
            samples = list(sample_frames)
        elif idx:
            samples = [idx[i] for i in sorted(idx.keys())]
        else:
            samples = record.frames()
        tokens = record.box_tokens if record.box_tokens else sorted(box_index.keys())
        return cls(
            record=record,
            frame=frame,
            box_index=box_index,
            box_tokens=tokens,
            frame_index=idx,
            sample_frames=samples,
            _sample_index=cls._sample_index_from(samples),
        )

    def for_person(self, person: dict[str, Any], track_id: int | None) -> "FeatureContext":
        return FeatureContext(
            record=self.record,
            frame=FramePersons(
                frame_idx=self.frame.frame_idx,
                timestamp_sec=self.frame.timestamp_sec,
                persons=[person],
            ),
            box_index=self.box_index,
            box_tokens=self.box_tokens,
            frame_index=self.frame_index,
            sample_frames=self.sample_frames,
            _sample_index=self._sample_index,
            current_person=person,
            current_track_id=track_id,
        )


@dataclass
class FeatureSet:
    """一帧的全局特征向量（用于取货检测）。"""

    record_id: str
    frame_idx: int
    person_track_id: int | None = None
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

    def frame_feature_names(self) -> list[str]:
        """返回该提取器稳定输出的帧级特征名（不含提取器前缀）。"""
        return []

    def extract_per_box(self, ctx: FeatureContext) -> dict[str, dict[str, float]]:
        """提取每货框特征，默认无。"""
        return {}

    def per_box_feature_names(self) -> list[str]:
        """返回该提取器稳定输出的货框级特征名（不含提取器前缀）。"""
        return []
