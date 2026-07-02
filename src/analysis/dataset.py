"""训练/评测数据集构建。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from analysis.constants import LEFT_SHOULDER_IDX, LEFT_WRIST_IDX, RIGHT_SHOULDER_IDX, RIGHT_WRIST_IDX
from analysis.features.base import FeatureContext
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
    confirmed_box_codes: list[int]


@dataclass
class BoxSample:
    record_id: str
    frame_idx: int
    box_token: str
    box_code: int
    box_shelf_side: int
    box_layer: int
    box_column: int
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


def _feature_extract_log_interval(total_frames: int) -> int:
    """根据总帧数选择进度日志间隔。"""
    if total_frames <= 50:
        return 10
    if total_frames <= 500:
        return 50
    if total_frames <= 5000:
        return 200
    return 500


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
    if removed:
        logger.info(
            "过滤无骨架帧: removed={}, kept_frames={}, kept_box_samples={}",
            removed,
            len(kept_frames),
            len(kept_box),
        )
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

    total_frames = sum(len(record.frames()) for record in records)
    log_interval = _feature_extract_log_interval(total_frames)
    logger.info(
        "开始提取特征: records={}, total_frames={}, filter_empty_skeleton={}",
        len(records),
        total_frames,
        filter_empty_skeleton,
    )

    processed_frames = 0
    for record_index, record in enumerate(records, start=1):
        if not frame_feature_names:
            frame_feature_names = reg.frame_feature_names(record)
            if feature_selection:
                frame_feature_names = feature_selection.select_frame(frame_feature_names)
        if not box_feature_names:
            box_feature_names = reg.per_box_feature_names(record)
            if feature_selection:
                box_feature_names = feature_selection.select_box(box_feature_names)

        frames = record.frames()
        frame_index = record.frame_index()
        logger.info(
            "提取特征 [{}/{}]: record={} frames={}",
            record_index,
            len(records),
            record.record_id,
            len(frames),
        )

        for frame in frames:
            ctx = FeatureContext.from_record(record, frame, frame_index=frame_index)
            label = record.labels.label_for(frame.frame_idx)
            frame_feat = reg.extract_frame_features_from_context(ctx)
            frame_samples.append(
                FrameSample(
                    record_id=record.record_id,
                    frame_idx=frame.frame_idx,
                    x=frame_feat.to_vector(frame_feature_names),
                    is_picking=label.is_picking,
                    confirmed_box_tokens=list(label.confirmed_box_tokens),
                    confirmed_box_codes=list(label.confirmed_box_codes),
                )
            )

            if label.is_picking:
                confirmed_codes = set(label.confirmed_box_codes)
                for pb in reg.extract_per_box_features_from_context(ctx):
                    layout_entry = record.box_layout.get(pb.box_token)
                    box_code = layout_entry.encode() if layout_entry else 0
                    box_samples.append(
                        BoxSample(
                            record_id=record.record_id,
                            frame_idx=frame.frame_idx,
                            box_token=pb.box_token,
                            box_code=box_code,
                            box_shelf_side=layout_entry.shelf_side if layout_entry else 0,
                            box_layer=layout_entry.layer if layout_entry else 0,
                            box_column=layout_entry.column if layout_entry else 0,
                            x=pb.to_vector(box_feature_names),
                            is_target=box_code in confirmed_codes,
                        )
                    )

            processed_frames += 1
            if processed_frames == total_frames or processed_frames % log_interval == 0:
                logger.info(
                    "提取特征进度: frames={}/{} ({:.1f}%), box_samples={}",
                    processed_frames,
                    total_frames,
                    100.0 * processed_frames / max(total_frames, 1),
                    len(box_samples),
                )

    dataset = Dataset(
        frame_samples=frame_samples,
        box_samples=box_samples,
        frame_feature_names=frame_feature_names,
        box_feature_names=box_feature_names,
    )
    if filter_empty_skeleton:
        dataset, _ = filter_empty_skeleton_frames(dataset, records)

    logger.info(
        "特征提取完成: frames={}, positive_frames={}, box_samples={}, frame_features={}, box_features={}",
        dataset.frame_count,
        dataset.positive_frame_count,
        len(dataset.box_samples),
        len(dataset.frame_feature_names),
        len(dataset.box_feature_names),
    )
    return dataset


def load_dataset(
    data_dir,
    feature_selection: FeatureSelection | None = None,
    *,
    filter_empty_skeleton: bool = False,
) -> Dataset:
    from pathlib import Path

    records = load_all_records(Path(data_dir))
    logger.info("加载记录完成: records={}", len(records))
    return build_dataset(
        records,
        feature_selection=feature_selection,
        filter_empty_skeleton=filter_empty_skeleton,
    )
