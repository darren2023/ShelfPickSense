"""训练/评测数据集构建。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

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


from analysis.box_layout import normalized_layout_targets


def _layout_target_from_label(record: RecordData, label) -> tuple[int, float, float]:
    """从 confirmed 货框取 (shelf_side, layout_layer_norm, layout_column_norm)。"""
    if not label.confirmed_box_tokens:
        return 0, 0.0, 0.0
    entry = record.box_layout.get(label.confirmed_box_tokens[0])
    if entry is None:
        return 0, 0.0, 0.0
    stats = record.shelf_layout_stats.get(entry.shelf_code or "_default")
    layer_norm, column_norm = normalized_layout_targets(entry, stats)
    return entry.shelf_side, layer_norm, column_norm


def _shelf_key(value: str | None) -> str:
    return str(value or "_default")


def _confirmed_tokens_for_shelf(record: RecordData, label, shelf_code: str | None) -> list[str]:
    key = _shelf_key(shelf_code)
    tokens: list[str] = []
    for token in label.confirmed_box_tokens:
        entry = record.box_layout.get(token)
        if entry is not None and _shelf_key(entry.shelf_code) == key:
            tokens.append(token)
    return tokens


def _layout_target_from_tokens(record: RecordData, tokens: list[str]) -> tuple[int, float, float]:
    if not tokens:
        return 0, 0.0, 0.0
    entry = record.box_layout.get(tokens[0])
    if entry is None:
        return 0, 0.0, 0.0
    stats = record.shelf_layout_stats.get(entry.shelf_code or "_default")
    layer_norm, column_norm = normalized_layout_targets(entry, stats)
    return entry.shelf_side, layer_norm, column_norm


def _is_person_pick_label(label, person_track_id: int | None) -> bool:
    track_ids = getattr(label, "picking_person_track_ids", []) or []
    if not track_ids:
        return True
    return person_track_id is not None and int(person_track_id) in set(int(v) for v in track_ids)


@dataclass
class FrameSample:
    record_id: str
    frame_idx: int
    person_track_id: int | None
    shelf_code: str | None
    x: np.ndarray
    is_picking: bool
    target_layout_shelf_side: int = 0
    target_layout_layer_norm: float = 0.0
    target_layout_column_norm: float = 0.0


@dataclass
class BoxSample:
    record_id: str
    frame_idx: int
    box_token: str
    box_code: int
    x: np.ndarray
    is_target: bool
    target_layout_shelf_side: int = 0
    target_layout_layer_norm: float = 0.0
    target_layout_column_norm: float = 0.0


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


def save_dataset(dataset: Dataset, path: Path) -> None:
    """将已提取特征样本序列化为压缩 npz，供后续 benchmark 直接加载。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_x = (
        np.vstack([sample.x for sample in dataset.frame_samples])
        if dataset.frame_samples
        else np.empty((0, len(dataset.frame_feature_names)), dtype=float)
    )
    box_x = (
        np.vstack([sample.x for sample in dataset.box_samples])
        if dataset.box_samples
        else np.empty((0, len(dataset.box_feature_names)), dtype=float)
    )
    np.savez_compressed(
        path,
        frame_feature_names=np.asarray(dataset.frame_feature_names, dtype=str),
        box_feature_names=np.asarray(dataset.box_feature_names, dtype=str),
        frame_record_ids=np.asarray([sample.record_id for sample in dataset.frame_samples], dtype=str),
        frame_indices=np.asarray([sample.frame_idx for sample in dataset.frame_samples], dtype=np.int64),
        frame_person_track_ids=np.asarray(
            [
                -1 if sample.person_track_id is None else int(sample.person_track_id)
                for sample in dataset.frame_samples
            ],
            dtype=np.int64,
        ),
        frame_shelf_codes=np.asarray(
            [sample.shelf_code or "" for sample in dataset.frame_samples],
            dtype=str,
        ),
        frame_x=frame_x,
        frame_is_picking=np.asarray([sample.is_picking for sample in dataset.frame_samples], dtype=bool),
        frame_target_layout_shelf_side=np.asarray(
            [sample.target_layout_shelf_side for sample in dataset.frame_samples],
            dtype=np.int64,
        ),
        frame_target_layout_layer_norm=np.asarray(
            [sample.target_layout_layer_norm for sample in dataset.frame_samples],
            dtype=float,
        ),
        frame_target_layout_column_norm=np.asarray(
            [sample.target_layout_column_norm for sample in dataset.frame_samples],
            dtype=float,
        ),
        box_record_ids=np.asarray([sample.record_id for sample in dataset.box_samples], dtype=str),
        box_indices=np.asarray([sample.frame_idx for sample in dataset.box_samples], dtype=np.int64),
        box_tokens=np.asarray([sample.box_token for sample in dataset.box_samples], dtype=str),
        box_codes=np.asarray([sample.box_code for sample in dataset.box_samples], dtype=np.int64),
        box_x=box_x,
        box_is_target=np.asarray([sample.is_target for sample in dataset.box_samples], dtype=bool),
        box_target_layout_shelf_side=np.asarray(
            [sample.target_layout_shelf_side for sample in dataset.box_samples],
            dtype=np.int64,
        ),
        box_target_layout_layer_norm=np.asarray(
            [sample.target_layout_layer_norm for sample in dataset.box_samples],
            dtype=float,
        ),
        box_target_layout_column_norm=np.asarray(
            [sample.target_layout_column_norm for sample in dataset.box_samples],
            dtype=float,
        ),
    )


def load_serialized_dataset(path: Path) -> Dataset:
    """从 save_dataset 生成的 npz 文件加载特征样本。"""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        frame_feature_names = [str(item) for item in data["frame_feature_names"].tolist()]
        box_feature_names = [str(item) for item in data["box_feature_names"].tolist()]
        frame_x = data["frame_x"]
        box_x = data["box_x"]
        frame_person_track_ids = data["frame_person_track_ids"]
        frame_samples = [
            FrameSample(
                record_id=str(data["frame_record_ids"][idx]),
                frame_idx=int(data["frame_indices"][idx]),
                person_track_id=(
                    None
                    if int(frame_person_track_ids[idx]) < 0
                    else int(frame_person_track_ids[idx])
                ),
                shelf_code=(
                    str(data["frame_shelf_codes"][idx])
                    if "frame_shelf_codes" in data and str(data["frame_shelf_codes"][idx])
                    else None
                ),
                x=np.asarray(frame_x[idx], dtype=float),
                is_picking=bool(data["frame_is_picking"][idx]),
                target_layout_shelf_side=int(data["frame_target_layout_shelf_side"][idx]),
                target_layout_layer_norm=float(data["frame_target_layout_layer_norm"][idx]),
                target_layout_column_norm=float(data["frame_target_layout_column_norm"][idx]),
            )
            for idx in range(len(frame_x))
        ]
        box_samples = [
            BoxSample(
                record_id=str(data["box_record_ids"][idx]),
                frame_idx=int(data["box_indices"][idx]),
                box_token=str(data["box_tokens"][idx]),
                box_code=int(data["box_codes"][idx]),
                x=np.asarray(box_x[idx], dtype=float),
                is_target=bool(data["box_is_target"][idx]),
                target_layout_shelf_side=int(data["box_target_layout_shelf_side"][idx]),
                target_layout_layer_norm=float(data["box_target_layout_layer_norm"][idx]),
                target_layout_column_norm=float(data["box_target_layout_column_norm"][idx]),
            )
            for idx in range(len(box_x))
        ]
    return Dataset(
        frame_samples=frame_samples,
        box_samples=box_samples,
        frame_feature_names=frame_feature_names,
        box_feature_names=box_feature_names,
    )


@dataclass
class _RecordExtractionResult:
    record_index: int
    record_id: str
    frame_count: int
    frame_samples: list[FrameSample]
    box_samples: list[BoxSample]


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


def _stride_frames(frames: list[FramePersons], frame_stride: int) -> list[FramePersons]:
    stride = max(1, int(frame_stride or 1))
    if stride <= 1:
        return frames
    return frames[::stride]


def _extract_record_samples(
    record_index: int,
    record: RecordData,
    registry: FeatureRegistry,
    frame_feature_names: list[str],
    box_feature_names: list[str],
    frame_stride: int,
) -> _RecordExtractionResult:
    frame_samples: list[FrameSample] = []
    box_samples: list[BoxSample] = []
    frames = _stride_frames(record.frames(), frame_stride)
    frame_index = record.frame_index()

    for frame in frames:
        ctx = FeatureContext.from_record(record, frame, frame_index=frame_index)
        label = record.labels.label_for(frame.frame_idx)
        for frame_feat in registry.extract_frame_feature_groups_from_context(ctx):
            shelf_tokens = _confirmed_tokens_for_shelf(record, label, frame_feat.shelf_code)
            target_side, target_layer_norm, target_col_norm = _layout_target_from_tokens(record, shelf_tokens)
            is_shelf_picking = bool(shelf_tokens) and _is_person_pick_label(label, frame_feat.person_track_id)
            frame_samples.append(
                FrameSample(
                    record_id=record.record_id,
                    frame_idx=frame.frame_idx,
                    person_track_id=frame_feat.person_track_id,
                    shelf_code=frame_feat.shelf_code,
                    x=frame_feat.to_vector(frame_feature_names),
                    is_picking=is_shelf_picking,
                    target_layout_shelf_side=target_side if is_shelf_picking else 0,
                    target_layout_layer_norm=target_layer_norm if is_shelf_picking else 0.0,
                    target_layout_column_norm=target_col_norm if is_shelf_picking else 0.0,
                )
            )

        if label.is_picking:
            confirmed_tokens = set(label.confirmed_box_tokens)
            for pb in registry.extract_per_box_features_from_context(ctx):
                layout_entry = record.box_layout.get(pb.box_token)
                box_code = layout_entry.encode() if layout_entry else 0
                box_stats = (
                    record.shelf_layout_stats.get(layout_entry.shelf_code or "_default")
                    if layout_entry
                    else None
                )
                layer_norm, col_norm = (
                    normalized_layout_targets(layout_entry, box_stats)
                    if layout_entry
                    else (0.0, 0.0)
                )
                box_samples.append(
                    BoxSample(
                        record_id=record.record_id,
                        frame_idx=frame.frame_idx,
                        box_token=pb.box_token,
                        box_code=box_code,
                        x=pb.to_vector(box_feature_names),
                        is_target=pb.box_token in confirmed_tokens,
                        target_layout_shelf_side=layout_entry.shelf_side if layout_entry else 0,
                        target_layout_layer_norm=layer_norm,
                        target_layout_column_norm=col_norm,
                    )
                )

    return _RecordExtractionResult(
        record_index=record_index,
        record_id=record.record_id,
        frame_count=len(frames),
        frame_samples=frame_samples,
        box_samples=box_samples,
    )


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
    feature_jobs: int = 1,
    feature_frame_stride: int = 1,
) -> Dataset:
    reg = registry or default_registry()
    frame_samples: list[FrameSample] = []
    box_samples: list[BoxSample] = []
    frame_feature_names: list[str] = []
    box_feature_names: list[str] = []

    frame_stride = max(1, int(feature_frame_stride or 1))
    total_source_frames = sum(len(record.frames()) for record in records)
    total_frames = sum(len(_stride_frames(record.frames(), frame_stride)) for record in records)
    log_interval = _feature_extract_log_interval(total_frames)
    logger.info(
        "开始提取特征: records={}, total_frames={}, source_frames={}, frame_stride={}, filter_empty_skeleton={}",
        len(records),
        total_frames,
        total_source_frames,
        frame_stride,
        filter_empty_skeleton,
    )

    frame_feature_names = reg.frame_feature_names()
    box_feature_names = reg.per_box_schema_feature_names()
    if feature_selection:
        frame_feature_names = feature_selection.select_frame(frame_feature_names)
        box_feature_names = feature_selection.select_box(box_feature_names)
        reg = reg.select_extractors_for_features(
            frame_feature_names=frame_feature_names,
            box_feature_names=box_feature_names,
        )

    workers = max(1, int(feature_jobs or 1))
    use_parallel = workers > 1 and len(records) > 1
    if use_parallel:
        max_workers = min(workers, len(records))
        logger.info("启用多进程特征提取: workers={}, records={}", max_workers, len(records))
        results: list[_RecordExtractionResult] = []
        processed_frames = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _extract_record_samples,
                    record_index,
                    record,
                    reg,
                    frame_feature_names,
                    box_feature_names,
                    frame_stride,
                )
                for record_index, record in enumerate(records, start=1)
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                processed_frames += result.frame_count
                logger.info(
                    "提取特征完成 [{}/{}]: record={} frames={} progress={}/{} ({:.1f}%)",
                    result.record_index,
                    len(records),
                    result.record_id,
                    result.frame_count,
                    processed_frames,
                    total_frames,
                    100.0 * processed_frames / max(total_frames, 1),
                )
        for result in sorted(results, key=lambda item: item.record_index):
            frame_samples.extend(result.frame_samples)
            box_samples.extend(result.box_samples)
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

    processed_frames = 0
    for record_index, record in enumerate(records, start=1):
        frames = _stride_frames(record.frames(), frame_stride)
        frame_index = record.frame_index()
        logger.info(
            "提取特征 [{}/{}]: record={} frames={} frame_stride={}",
            record_index,
            len(records),
            record.record_id,
            len(frames),
            frame_stride,
        )

        for frame in frames:
            ctx = FeatureContext.from_record(record, frame, frame_index=frame_index)
            label = record.labels.label_for(frame.frame_idx)
            for frame_feat in reg.extract_frame_feature_groups_from_context(ctx):
                shelf_tokens = _confirmed_tokens_for_shelf(record, label, frame_feat.shelf_code)
                target_side, target_layer_norm, target_col_norm = _layout_target_from_tokens(record, shelf_tokens)
                is_shelf_picking = bool(shelf_tokens) and _is_person_pick_label(label, frame_feat.person_track_id)
                frame_samples.append(
                    FrameSample(
                        record_id=record.record_id,
                        frame_idx=frame.frame_idx,
                        person_track_id=frame_feat.person_track_id,
                        shelf_code=frame_feat.shelf_code,
                        x=frame_feat.to_vector(frame_feature_names),
                        is_picking=is_shelf_picking,
                        target_layout_shelf_side=target_side if is_shelf_picking else 0,
                        target_layout_layer_norm=target_layer_norm if is_shelf_picking else 0.0,
                        target_layout_column_norm=target_col_norm if is_shelf_picking else 0.0,
                    )
                )

            if label.is_picking:
                confirmed_tokens = set(label.confirmed_box_tokens)
                for pb in reg.extract_per_box_features_from_context(ctx):
                    layout_entry = record.box_layout.get(pb.box_token)
                    box_code = layout_entry.encode() if layout_entry else 0
                    box_stats = (
                        record.shelf_layout_stats.get(layout_entry.shelf_code or "_default")
                        if layout_entry
                        else None
                    )
                    layer_norm, col_norm = (
                        normalized_layout_targets(layout_entry, box_stats)
                        if layout_entry
                        else (0.0, 0.0)
                    )
                    box_samples.append(
                        BoxSample(
                            record_id=record.record_id,
                            frame_idx=frame.frame_idx,
                            box_token=pb.box_token,
                            box_code=box_code,
                            x=pb.to_vector(box_feature_names),
                            is_target=pb.box_token in confirmed_tokens,
                            target_layout_shelf_side=layout_entry.shelf_side if layout_entry else 0,
                            target_layout_layer_norm=layer_norm,
                            target_layout_column_norm=col_norm,
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
    feature_jobs: int = 1,
    feature_frame_stride: int = 1,
) -> Dataset:
    from pathlib import Path

    records = load_all_records(Path(data_dir))
    logger.info("加载记录完成: records={}", len(records))
    return build_dataset(
        records,
        feature_selection=feature_selection,
        filter_empty_skeleton=filter_empty_skeleton,
        feature_jobs=feature_jobs,
        feature_frame_stride=feature_frame_stride,
    )
