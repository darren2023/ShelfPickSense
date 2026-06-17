"""记录数据加载：skeleton.parquet + annotation.json + event_review.json。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.annotation import build_box_index, load_annotation
from analysis.constants import ANNOTATION_FILE, EVENT_REVIEW_FILE, SKELETON_FILE
from analysis.labels import RecordLabels, build_labels_from_event_review, load_event_review


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

    def frames(self) -> list[FramePersons]:
        if self.skeleton.empty:
            return []
        grouped: list[FramePersons] = []
        for frame_idx, group in self.skeleton.groupby("frame_idx", sort=True):
            fi = int(frame_idx)
            ts = float(group["timestamp_sec"].iloc[0]) if "timestamp_sec" in group.columns else 0.0
            persons = [_row_to_person(row) for _, row in group.iterrows()]
            grouped.append(FramePersons(frame_idx=fi, timestamp_sec=ts, persons=persons))
        return grouped


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


def _infer_frame_size(skeleton: pd.DataFrame) -> tuple[float, float]:
    if skeleton.empty:
        return 640.0, 480.0
    xs: list[float] = []
    ys: list[float] = []
    for i in range(17):
        xcol, ycol = f"kpt_{i}_x", f"kpt_{i}_y"
        if xcol in skeleton.columns:
            xs.extend(float(v) for v in skeleton[xcol].dropna())
        if ycol in skeleton.columns:
            ys.extend(float(v) for v in skeleton[ycol].dropna())
    if not xs or not ys:
        return 640.0, 480.0
    return max(xs) * 1.05, max(ys) * 1.05


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

    infer_w, infer_h = _infer_frame_size(skeleton)
    ann_size = annotation.get("annotation_size") if isinstance(annotation.get("annotation_size"), dict) else {}
    if ann_size.get("width") and ann_size.get("height"):
        # 标注尺寸用于货框多边形缩放
        pass

    box_index = build_box_index(annotation, infer_w=infer_w, infer_h=infer_h)
    box_tokens = sorted(box_index.keys())

    frame_indices = sorted(int(v) for v in skeleton["frame_idx"].unique()) if not skeleton.empty else []
    labels = build_labels_from_event_review(
        event_review,
        record_id=record_dir.name,
        all_frame_indices=frame_indices,
    )

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
    )


def load_all_records(data_dir: Path) -> list[RecordData]:
    dirs = discover_record_dirs(data_dir)
    if not dirs:
        raise FileNotFoundError(f"在 {data_dir} 下未找到有效记录目录")
    return [load_record(d) for d in dirs]
