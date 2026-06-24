"""训练/评测数据集构建。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.features.registry import FeatureRegistry, default_registry
from analysis.features.selection import FeatureSelection
from analysis.records import RecordData, load_all_records


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


def build_dataset(
    records: list[RecordData],
    registry: FeatureRegistry | None = None,
    feature_selection: FeatureSelection | None = None,
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

    return Dataset(
        frame_samples=frame_samples,
        box_samples=box_samples,
        frame_feature_names=frame_feature_names,
        box_feature_names=box_feature_names,
    )


def load_dataset(data_dir, feature_selection: FeatureSelection | None = None) -> Dataset:
    from pathlib import Path

    records = load_all_records(Path(data_dir))
    return build_dataset(records, feature_selection=feature_selection)
