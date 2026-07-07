"""记录数据加载：skeleton.parquet + annotation.json + event_review.json。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from analysis.annotation import BoxInfo, annotation_size, build_box_index, load_annotation
from analysis.box_layout import BoxNumericCode, ShelfLayoutStats, build_box_layout, compute_shelf_layout_stats
from analysis.constants import (
    ANNOTATION_FILE,
    DEFAULT_POSE_INFER_HEIGHT,
    DEFAULT_POSE_INFER_WIDTH,
    EVENT_REVIEW_FILE,
    MANIFEST_FILE,
    SKELETON_FILE,
)
from analysis.labels import (
    RecordLabels,
    build_labels_from_event_review,
    enrich_labels_with_box_layout,
    load_event_review,
)

# skeleton 坐标超出声明 infer 的可接受比例（与 visual-dps 1.05 margin 对齐）
_SKELETON_INFER_TOLERANCE = 0.05
# 声明 infer 明显小于 skeleton 范围时视为错误（如 640×360 vs 852×480）
_SKELETON_INFER_TOO_SMALL_RATIO = 0.85


@dataclass
class FramePersons:
    frame_idx: int
    timestamp_sec: float
    persons: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RecordData:
    record_id: str
    record_dir: Path
    skeleton: pd.DataFrame
    annotation: dict[str, Any]
    event_review: dict[str, Any] | None
    labels: RecordLabels
    infer_width: float
    infer_height: float
    box_tokens: list[str]
    box_index: dict[str, BoxInfo] = field(default_factory=dict)
    box_layout: dict[str, BoxNumericCode] = field(default_factory=dict)
    shelf_layout_stats: dict[str, ShelfLayoutStats] = field(default_factory=dict)
    _frames_cache: list[FramePersons] | None = field(default=None, repr=False, compare=False)
    _frame_index_cache: dict[int, FramePersons] | None = field(default=None, repr=False, compare=False)

    def frames(self) -> list[FramePersons]:
        if self._frames_cache is not None:
            return self._frames_cache
        if self.skeleton.empty:
            self._frames_cache = []
            return self._frames_cache
        grouped: list[FramePersons] = []
        for frame_idx, group in self.skeleton.groupby("frame_idx", sort=True):
            fi = int(frame_idx)
            ts = float(group["timestamp_sec"].iloc[0]) if "timestamp_sec" in group.columns else 0.0
            persons = [_row_to_person(row) for _, row in group.iterrows()]
            grouped.append(FramePersons(frame_idx=fi, timestamp_sec=ts, persons=persons))
        self._frames_cache = grouped
        return grouped

    def frame_index(self) -> dict[int, FramePersons]:
        if self._frame_index_cache is not None:
            return self._frame_index_cache
        self._frame_index_cache = {frame.frame_idx: frame for frame in self.frames()}
        return self._frame_index_cache


def _row_to_person(row: pd.Series) -> dict[str, Any]:
    keypoints: list[list[float | None]] = []
    for i in range(17):
        x = row.get(f"kpt_{i}_x")
        y = row.get(f"kpt_{i}_y")
        s = row.get(f"kpt_{i}_score")
        if pd.isna(x) or pd.isna(y):
            keypoints.append([None, None, None])
        else:
            keypoints.append([float(x), float(y), float(s) if not pd.isna(s) else 0.0])

    person: dict[str, Any] = {
        "person_id": int(row.get("person_id") or 0),
        "keypoints": keypoints,
    }
    ptid = row.get("person_track_id")
    if ptid is not None and not pd.isna(ptid):
        person["person_track_id"] = int(ptid)
    bbox_cols = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
    if all(c in row.index for c in bbox_cols):
        bbox = [row[c] for c in bbox_cols]
        if any(not pd.isna(v) for v in bbox):
            person["bbox"] = [float(v) if not pd.isna(v) else 0.0 for v in bbox]
    return person


def _positive_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _infer_from_manifest(record_dir: Path) -> tuple[float, float] | None:
    """仅从 manifest 根级 infer_width/infer_height 读取，不使用 annotation_size 回退。"""
    path = record_dir / MANIFEST_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    infer_w = _positive_float(data.get("infer_width"))
    infer_h = _positive_float(data.get("infer_height"))
    if infer_w and infer_h:
        return infer_w, infer_h
    return None


def _infer_from_skeleton_columns(skeleton: pd.DataFrame) -> tuple[float, float] | None:
    """从 parquet 列 infer_width/infer_height 读取（取首个有效值）。"""
    for col_w, col_h in (("infer_width", "infer_height"),):
        if col_w not in skeleton.columns or col_h not in skeleton.columns:
            continue
        for _, row in skeleton.iterrows():
            infer_w = _positive_float(row.get(col_w))
            infer_h = _positive_float(row.get(col_h))
            if infer_w and infer_h:
                return infer_w, infer_h
    return None


def _infer_from_annotation_meta(annotation: dict[str, Any]) -> tuple[float, float] | None:
    infer_w = _positive_float(annotation.get("infer_width"))
    infer_h = _positive_float(annotation.get("infer_height"))
    if infer_w and infer_h:
        return infer_w, infer_h
    return None


def _infer_from_pose_pipeline(annotation: dict[str, Any]) -> tuple[float, float]:
    """按 pose 采集默认推理尺寸（与 skeleton 坐标系一致）。"""
    ann_w, ann_h = annotation_size(annotation)
    if ann_w and ann_h:
        aw, ah = int(round(ann_w)), int(round(ann_h))
        # 640×360 标注时采集管线固定 852×480（非 round 推算的 853）
        if aw == 640 and ah == 360:
            return DEFAULT_POSE_INFER_WIDTH, DEFAULT_POSE_INFER_HEIGHT
        infer_h = DEFAULT_POSE_INFER_HEIGHT
        infer_w = round(infer_h * (ann_w / ann_h))
        return float(infer_w), float(infer_h)
    return DEFAULT_POSE_INFER_WIDTH, DEFAULT_POSE_INFER_HEIGHT


def _skeleton_coordinate_extent(skeleton: pd.DataFrame) -> tuple[float, float]:
    """从 skeleton bbox/kpt 分位数估计坐标范围（infer 空间实际占用）。"""
    if skeleton.empty:
        return 0.0, 0.0

    xs: list[float] = []
    ys: list[float] = []

    for col in ("bbox_x2",):
        if col in skeleton.columns:
            vals = pd.to_numeric(skeleton[col], errors="coerce").dropna()
            xs.extend(float(v) for v in vals if v > 0)

    for col in ("bbox_y2",):
        if col in skeleton.columns:
            vals = pd.to_numeric(skeleton[col], errors="coerce").dropna()
            ys.extend(float(v) for v in vals if v > 0)

    for i in range(17):
        col_x = f"kpt_{i}_x"
        col_y = f"kpt_{i}_y"
        if col_x in skeleton.columns:
            vals = pd.to_numeric(skeleton[col_x], errors="coerce").dropna()
            xs.extend(float(v) for v in vals if v > 0)
        if col_y in skeleton.columns:
            vals = pd.to_numeric(skeleton[col_y], errors="coerce").dropna()
            ys.extend(float(v) for v in vals if v > 0)

    if not xs or not ys:
        return 0.0, 0.0

    return float(pd.Series(xs).quantile(0.99)), float(pd.Series(ys).quantile(0.99))


def _snap_infer_size_from_extent(ext_w: float, ext_h: float, annotation: dict[str, Any]) -> tuple[float, float]:
    """将 skeleton 范围对齐到常见 infer 尺寸。"""
    pipeline_w, pipeline_h = _infer_from_pose_pipeline(annotation)
    if ext_w <= pipeline_w * (1.0 + _SKELETON_INFER_TOLERANCE) and ext_h <= pipeline_h * (1.0 + _SKELETON_INFER_TOLERANCE):
        return pipeline_w, pipeline_h
    return math.ceil(ext_w), math.ceil(ext_h)


def _align_infer_with_skeleton(
    infer_w: float,
    infer_h: float,
    skeleton: pd.DataFrame,
    annotation: dict[str, Any],
    *,
    source: str,
    record_id: str,
) -> tuple[float, float]:
    """
    校验并修正 infer 尺寸，确保 skeleton 坐标落在 infer 空间内。

    若元数据 infer 明显小于 skeleton 实际范围（如 manifest 误用 annotation_size），
    则回退到 pose 管线默认或 skeleton 分位数推断。
    """
    sk_w, sk_h = _skeleton_coordinate_extent(skeleton)
    if sk_w <= 0 or sk_h <= 0:
        return infer_w, infer_h

    overflow_w = sk_w > infer_w * (1.0 + _SKELETON_INFER_TOLERANCE)
    overflow_h = sk_h > infer_h * (1.0 + _SKELETON_INFER_TOLERANCE)
    too_small = infer_w < sk_w * _SKELETON_INFER_TOO_SMALL_RATIO or infer_h < sk_h * _SKELETON_INFER_TOO_SMALL_RATIO

    if overflow_w or overflow_h or too_small:
        corrected = _snap_infer_size_from_extent(sk_w, sk_h, annotation)
        logger.warning(
            "记录 {} infer 尺寸 ({:.0f}x{:.0f}, 来源={}) 与 skeleton 范围 ({:.0f}x{:.0f}) 不一致，"
            "已修正为 {:.0f}x{:.0f} 以保证货框与骨架对齐",
            record_id,
            infer_w,
            infer_h,
            source,
            sk_w,
            sk_h,
            corrected[0],
            corrected[1],
        )
        return corrected

    return infer_w, infer_h


def resolve_infer_frame_size(
    record_dir: Path,
    skeleton: pd.DataFrame,
    annotation: dict[str, Any],
    *,
    record_id: str = "",
) -> tuple[float, float]:
    """解析 infer 坐标系尺寸（与 skeleton 关键点、货框 polygon 同一空间）。"""
    rid = record_id or record_dir.name
    source = "pose_pipeline"
    infer_w: float | None = None
    infer_h: float | None = None

    from_manifest = _infer_from_manifest(record_dir)
    if from_manifest:
        infer_w, infer_h = from_manifest
        source = "manifest"

    if infer_w is None:
        from_columns = _infer_from_skeleton_columns(skeleton)
        if from_columns:
            infer_w, infer_h = from_columns
            source = "skeleton_columns"

    if infer_w is None:
        from_annotation = _infer_from_annotation_meta(annotation)
        if from_annotation:
            infer_w, infer_h = from_annotation
            source = "annotation_meta"

    if infer_w is None:
        infer_w, infer_h = _infer_from_pose_pipeline(annotation)

    infer_w, infer_h = _align_infer_with_skeleton(
        infer_w,
        infer_h,
        skeleton,
        annotation,
        source=source,
        record_id=rid,
    )
    return infer_w, infer_h


def _infer_frame_size(skeleton: pd.DataFrame) -> tuple[float, float]:
    """兼容旧调用：无 annotation 时使用 pose 默认推理尺寸。"""
    if skeleton.empty:
        return 640.0, 480.0
    return DEFAULT_POSE_INFER_WIDTH, DEFAULT_POSE_INFER_HEIGHT


def is_record_dir(path: Path) -> bool:
    return path.is_dir() and (path / SKELETON_FILE).is_file() and (path / ANNOTATION_FILE).is_file()


def discover_record_dirs(data_dir: Path) -> list[Path]:
    """发现 data_dir 下所有有效记录目录。"""
    data_dir = Path(data_dir)
    if is_record_dir(data_dir):
        return [data_dir.resolve()]

    found: list[Path] = []
    if not data_dir.is_dir():
        return found
    for child in sorted(data_dir.iterdir()):
        if is_record_dir(child):
            found.append(child.resolve())
    return found


def load_record(record_dir: Path) -> RecordData:
    record_dir = Path(record_dir).resolve()
    if not is_record_dir(record_dir):
        raise FileNotFoundError(
            f"无效记录目录，需包含 {SKELETON_FILE} 与 {ANNOTATION_FILE}: {record_dir}"
        )

    skeleton_path = record_dir / SKELETON_FILE
    skeleton = pd.read_parquet(skeleton_path)
    annotation = load_annotation(record_dir / ANNOTATION_FILE)

    event_review_path = record_dir / EVENT_REVIEW_FILE
    event_review = load_event_review(event_review_path) if event_review_path.is_file() else None

    infer_w, infer_h = resolve_infer_frame_size(
        record_dir,
        skeleton,
        annotation,
        record_id=record_dir.name,
    )

    box_index = build_box_index(annotation, infer_w=infer_w, infer_h=infer_h)
    box_tokens = sorted(box_index.keys())
    ann_w, _ann_h = annotation_size(annotation)
    box_layout = build_box_layout(annotation, frame_width=ann_w)
    shelf_layout_stats = compute_shelf_layout_stats(box_layout)

    frame_indices = sorted(int(v) for v in skeleton["frame_idx"].unique()) if not skeleton.empty else []
    labels = build_labels_from_event_review(
        event_review,
        record_id=record_dir.name,
        all_frame_indices=frame_indices,
    )
    enrich_labels_with_box_layout(labels, box_layout)

    return RecordData(
        record_id=record_dir.name,
        record_dir=record_dir,
        skeleton=skeleton,
        annotation=annotation,
        event_review=event_review,
        labels=labels,
        infer_width=infer_w,
        infer_height=infer_h,
        box_tokens=box_tokens,
        box_index=box_index,
        box_layout=box_layout,
        shelf_layout_stats=shelf_layout_stats,
    )


def load_all_records(data_dir: Path) -> list[RecordData]:
    dirs = discover_record_dirs(data_dir)
    if not dirs:
        raise FileNotFoundError(f"在 {data_dir} 下未找到有效记录目录")
    return [load_record(d) for d in dirs]
