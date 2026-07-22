"""Analyze event_review_v2.json continuity by picked box."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from analysis.constants import EVENT_REVIEW_V2_FILE
from analysis.records import discover_record_dirs


DEFAULT_REVIEW_V2_ANALYSIS_FILE = "event_review_v2_analysis.json"
DEFAULT_MAX_GROUP_FRAME_GAP = 100


@dataclass
class ReviewV2AnalysisStats:
    record_id: str
    input_path: str
    output_path: str
    event_count: int
    group_count: int
    groups_with_missing_frames: int
    missing_frame_count: int
    duplicate_frame_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"review root must be an object: {path}")
    return data


def _event_frame_idx(event: dict[str, Any]) -> int:
    try:
        return int(event.get("frame_idx") or 0)
    except (TypeError, ValueError):
        return 0


def _event_key(event: dict[str, Any]) -> tuple[str, str]:
    return str(event.get("shelf_code") or "").strip(), str(event.get("box_id") or "").strip()


def _compact_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": str(event.get("event_type") or ""),
        "frame_idx": _event_frame_idx(event),
        "is_pick": bool(event.get("is_pick")),
        "person_track_id": event.get("person_track_id"),
        "shelf_code": str(event.get("shelf_code") or "").strip(),
        "box_id": str(event.get("box_id") or "").strip(),
    }


def _missing_frames(frames: list[int]) -> list[int]:
    if not frames:
        return []
    unique = sorted(set(frames))
    start, end = unique[0], unique[-1]
    present = set(unique)
    return [frame_idx for frame_idx in range(start, end + 1) if frame_idx not in present]


def _duplicate_frames(frames: list[int]) -> list[int]:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for frame_idx in frames:
        if frame_idx in seen:
            duplicates.add(frame_idx)
        seen.add(frame_idx)
    return sorted(duplicates)


def analyze_review_v2_payload(
    review: dict[str, Any],
    *,
    record_id: str = "",
    max_frame_gap: int = DEFAULT_MAX_GROUP_FRAME_GAP,
) -> dict[str, Any]:
    """Sort pick events and group adjacent events by the same shelf/box."""
    if int(review.get("schema") or 0) != 2:
        raise ValueError(f"event_review_v2 for record={record_id or '<unknown>'} must use schema=2")

    raw_events = [event for event in review.get("verified_true") or [] if isinstance(event, dict)]
    sorted_events = sorted(
        (_compact_event(event) for event in raw_events if event.get("is_pick") is True),
        key=lambda event: (int(event["frame_idx"]), str(event["shelf_code"]), str(event["box_id"])),
    )
    skipped_non_pick = len(raw_events) - len(sorted_events)

    groups: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_key: tuple[str, str] | None = None
    previous_frame_idx: int | None = None
    frame_gap = max(1, int(max_frame_gap or DEFAULT_MAX_GROUP_FRAME_GAP))

    for event in sorted_events:
        key = _event_key(event)
        event_frame_idx = int(event["frame_idx"])
        should_start_new_group = (
            current
            and (
                key != current_key
                or (
                    previous_frame_idx is not None
                    and event_frame_idx - previous_frame_idx > frame_gap
                )
            )
        )
        if should_start_new_group:
            groups.append(_build_group(len(groups) + 1, current))
            current = []
        current.append(event)
        current_key = key
        previous_frame_idx = event_frame_idx
    if current:
        groups.append(_build_group(len(groups) + 1, current))

    groups_with_missing = sum(1 for group in groups if group["has_missing_frames"])
    missing_count = sum(len(group["missing_frame_indices"]) for group in groups)
    duplicate_count = sum(len(group["duplicate_frame_indices"]) for group in groups)
    return {
        "record_id": record_id,
        "schema": 1,
        "source_schema": review.get("schema"),
        "max_group_frame_gap": frame_gap,
        "event_count": len(sorted_events),
        "skipped_non_pick_count": skipped_non_pick,
        "group_count": len(groups),
        "groups_with_missing_frames": groups_with_missing,
        "missing_frame_count": missing_count,
        "duplicate_frame_count": duplicate_count,
        "groups": groups,
    }


def _build_group(group_index: int, events: list[dict[str, Any]]) -> dict[str, Any]:
    frames = [int(event["frame_idx"]) for event in events]
    shelf_code, box_id = _event_key(events[0])
    missing = _missing_frames(frames)
    duplicates = _duplicate_frames(frames)
    unique_frames = sorted(set(frames))
    return {
        "group_index": group_index,
        "shelf_code": shelf_code,
        "box_id": box_id,
        "start_frame_idx": unique_frames[0] if unique_frames else 0,
        "end_frame_idx": unique_frames[-1] if unique_frames else 0,
        "event_count": len(events),
        "unique_frame_count": len(unique_frames),
        "expected_frame_count": (unique_frames[-1] - unique_frames[0] + 1) if unique_frames else 0,
        "has_missing_frames": bool(missing),
        "missing_frame_indices": missing,
        "duplicate_frame_indices": duplicates,
        "event_types": sorted({str(event.get("event_type") or "") for event in events if event.get("event_type")}),
        "person_track_ids": sorted(
            {
                int(event["person_track_id"])
                for event in events
                if event.get("person_track_id") is not None
            }
        ),
        "frame_indices": frames,
    }


def analyze_record_review_v2(
    record_dir: Path,
    *,
    review_filename: str = EVENT_REVIEW_V2_FILE,
    output_filename: str = DEFAULT_REVIEW_V2_ANALYSIS_FILE,
    max_frame_gap: int = DEFAULT_MAX_GROUP_FRAME_GAP,
) -> ReviewV2AnalysisStats:
    record_dir = Path(record_dir).resolve()
    input_path = record_dir / review_filename
    if not input_path.is_file():
        raise FileNotFoundError(f"event_review_v2 file not found: {input_path}")
    output_path = record_dir / output_filename
    review = _read_json(input_path)
    payload = analyze_review_v2_payload(
        review,
        record_id=record_dir.name,
        max_frame_gap=max_frame_gap,
    )
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return ReviewV2AnalysisStats(
        record_id=record_dir.name,
        input_path=str(input_path),
        output_path=str(output_path),
        event_count=int(payload["event_count"]),
        group_count=int(payload["group_count"]),
        groups_with_missing_frames=int(payload["groups_with_missing_frames"]),
        missing_frame_count=int(payload["missing_frame_count"]),
        duplicate_frame_count=int(payload["duplicate_frame_count"]),
    )


def analyze_reviews_v2(
    data_dir: Path,
    *,
    review_filename: str = EVENT_REVIEW_V2_FILE,
    output_filename: str = DEFAULT_REVIEW_V2_ANALYSIS_FILE,
    max_frame_gap: int = DEFAULT_MAX_GROUP_FRAME_GAP,
) -> list[ReviewV2AnalysisStats]:
    record_dirs = discover_record_dirs(Path(data_dir))
    if not record_dirs:
        raise FileNotFoundError(f"no valid record directories found under: {data_dir}")
    return [
        analyze_record_review_v2(
            record_dir,
            review_filename=review_filename,
            output_filename=output_filename,
            max_frame_gap=max_frame_gap,
        )
        for record_dir in record_dirs
    ]
