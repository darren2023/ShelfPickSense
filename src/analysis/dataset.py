"""训练/评测数据集构建。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.constants import LEFT_SHOULDER_IDX, LEFT_WRIST_IDX, RIGHT_SHOULDER_IDX, RIGHT_WRIST_IDX
from analysis.features.registry import FeatureRegistry, default_registry
from analysis.features.selection import FeatureSelection
from analysis.features.tracking import MIN_KEYPOINT_SCORE, get_keypoint
from analysis.records import FramePersons, RecordData, load_all_records

_SKELETON_PROBE_INDICES = (
    LEFT_SHOULDER_IDX,
    RIGHT_SHOULDER_IDX,
    LEFT_WRIST_IDX,
    RIGHT_WRIST_IDX,
)


@dataclass
class FrameSample:
    record_id: str
    frame_idx: int
    x: np.ndarray
    is_picking: bool
    confirmed_box_tokens: list[str]


@dataclass
class BoxSample:
    record_id: str
    frame_idx: int
    box_token: str
    x: np.ndarray
    is_target: bool


@dataclass
class Dataset:
    frame_samples: list[FrameSample]
    box_samples: list[BoxSample]
    frame_feature_names: list[str]
    box_feature_names: list[str]

    @property
    def frame_count(self) -> int:
        return len(self.frame_samples)

    @property
    def positive_frame_count(self) -> int:
        return sum(1 for s in self.frame_samples if s.is_picking)


def frame_has_valid_skeleton(
    frame: FramePersons,
    *,
    min_score: float = MIN_KEYPOINT_SCORE,
) -> bool:
    """帧内是否检测到可用骨架（至少一个置信度达标的躯干/手腕关键点）。"""
    if not frame.persons:
        return False
    for person in frame.persons:
        for idx in _SKELETON_PROBE_INDICES:
            if get_keypoint(person, idx, min_score=min_score) is not None:
                return True
    return False


def skeleton_frame_keys(records: list[RecordData]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for record in records:
        for frame in record.frames():
            if frame_has_valid_skeleton(frame):
                keys.add((record.record_id, frame.frame_idx))
    return keys


def filter_empty_skeleton_frames(
    dataset: Dataset,
    records: list[RecordData],
) -> tuple[Dataset, int]:
    """特征提取后、训练前过滤无骨架帧，降低负样本占比。"""
    valid_keys = skeleton_frame_keys(records)
    kept_frames = [
        sample
        for sample in dataset.frame_samples
        if (sample.record_id, sample.frame_idx) in valid_keys
    ]
    kept_box = [
        sample
        for sample in dataset.box_samples
        if (sample.record_id, sample.frame_idx) in valid_keys
    ]
    removed = len(dataset.frame_samples) - len(kept_frames)
    return (
        Dataset(
            frame_samples=kept_frames,
            box_samples=kept_box,
            frame_feature_names=list(dataset.frame_feature_names),
            box_feature_names=list(dataset.box_feature_names),
        ),
        removed,
    )
def build_dataset(
    records: list[RecordData],
    registry: FeatureRegistry | None = None,
    feature_selection: FeatureSelection | None = None,
    *,
    filter_empty_skeleton: bool = False,
) -> Dataset:
    reg = registry or default_registry()
    frame_samples: list[FrameSample] = []
    box_samples: list[BoxSample] = []
    frame_feature_names: list[str] = []
    box_feature_names: list[str] = []

    for record in records:
        if not frame_feature_names:
            frame_feature_names = reg.frame_feature_names(record)
            if feature_selection:
                frame_feature_names = feature_selection.select_frame(frame_feature_names)
        if not box_feature_names:
            box_feature_names = reg.per_box_feature_names(record)
            if feature_selection:
                box_feature_names = feature_selection.select_box(box_feature_names)

        for frame in record.frames():
            label = record.labels.label_for(frame.frame_idx)
            frame_feat = reg.extract_frame_features(record, frame)
            frame_samples.append(
                FrameSample(
                    record_id=record.record_id,
                    frame_idx=frame.frame_idx,
                    x=frame_feat.to_vector(frame_feature_names),
                    is_picking=label.is_picking,
                    confirmed_box_tokens=list(label.confirmed_box_tokens),
                )
            )

            if not label.is_picking:
                continue

            confirmed = set(label.confirmed_box_tokens)
            for pb in reg.extract_per_box_features(record, frame):
                box_samples.append(
                    BoxSample(
                        record_id=record.record_id,
                        frame_idx=frame.frame_idx,
                        box_token=pb.box_token,
                        x=pb.to_vector(box_feature_names),
                        is_target=pb.box_token in confirmed,
                    )
                )

    dataset = Dataset(
        frame_samples=frame_samples,
        box_samples=box_samples,
        frame_feature_names=frame_feature_names,
        box_feature_names=box_feature_names,
    )
    if filter_empty_skeleton:
        dataset, _ = filter_empty_skeleton_frames(dataset, records)
    return dataset


def load_dataset(
    data_dir,
    feature_selection: FeatureSelection | None = None,
    *,
    filter_empty_skeleton: bool = False,
) -> Dataset:
    from pathlib import Path

    records = load_all_records(Path(data_dir))
    return build_dataset(
        records,
        feature_selection=feature_selection,
        filter_empty_skeleton=filter_empty_skeleton,
    )
