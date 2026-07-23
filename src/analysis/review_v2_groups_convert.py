"""Convert grouped event_review_v2 analysis files back to event_review_v2.json."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from analysis.constants import EVENT_REVIEW_V2_FILE
from analysis.records import discover_record_dirs
from analysis.review_v2_analysis import DEFAULT_REVIEW_V2_ANALYSIS_FILE


@dataclass
class ReviewV2GroupsConvertStats:
    record_id: str
    input_path: str
    output_path: str
    group_count: int
    event_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"grouped review root must be an object: {path}")
    return data


def _as_int(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _group_person_track_id(group: dict[str, Any]) -> int:
    if group.get("person_track_id") is not None:
        return _as_int(group.get("person_track_id"), field_name="person_track_id")
    person_ids = group.get("person_track_ids")
    if not isinstance(person_ids, list) or not person_ids:
        raise ValueError("each group must include person_track_id or non-empty person_track_ids")
    return _as_int(person_ids[0], field_name="person_track_ids[0]")


def _group_event_type(group: dict[str, Any]) -> str:
    event_type = str(group.get("event_type") or "").strip()
    if event_type:
        return event_type
    event_types = group.get("event_types")
    if isinstance(event_types, list) and event_types:
        event_type = str(event_types[0] or "").strip()
        if event_type:
            return event_type
    return "collision"


def convert_review_v2_analysis_payload(analysis: dict[str, Any]) -> dict[str, Any]:
    groups = analysis.get("groups")
    if not isinstance(groups, list):
        raise ValueError("grouped review must include groups list")

    events: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups, start=1):
        if not isinstance(group, dict):
            raise ValueError(f"groups[{group_index - 1}] must be an object")
        shelf_code = str(group.get("shelf_code") or "").strip()
        box_id = str(group.get("box_id") or "").strip()
        if not shelf_code or not box_id:
            raise ValueError(f"group {group_index} must include non-empty shelf_code and box_id")
        frame_indices = group.get("frame_indices")
        if not isinstance(frame_indices, list) or not frame_indices:
            raise ValueError(f"group {group_index} must include non-empty frame_indices")
        person_track_id = _group_person_track_id(group)
        event_type = _group_event_type(group)
        for frame_idx in frame_indices:
            events.append(
                {
                    "event_type": event_type,
                    "frame_idx": _as_int(frame_idx, field_name="frame_idx"),
                    "is_pick": True,
                    "person_track_id": person_track_id,
                    "shelf_code": shelf_code,
                    "box_id": box_id,
                }
            )

    events.sort(key=lambda item: (int(item["frame_idx"]), str(item["shelf_code"]), str(item["box_id"])))
    return {"schema": 2, "verified_true": events}


def convert_record_review_v2_analysis(
    record_dir: Path,
    *,
    input_filename: str = DEFAULT_REVIEW_V2_ANALYSIS_FILE,
    output_filename: str = EVENT_REVIEW_V2_FILE,
) -> ReviewV2GroupsConvertStats:
    record_dir = Path(record_dir).resolve()
    input_path = record_dir / input_filename
    if not input_path.is_file():
        raise FileNotFoundError(f"grouped review file not found: {input_path}")
    output_path = record_dir / output_filename
    analysis = _read_json(input_path)
    payload = convert_review_v2_analysis_payload(analysis)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return ReviewV2GroupsConvertStats(
        record_id=record_dir.name,
        input_path=str(input_path),
        output_path=str(output_path),
        group_count=len(analysis.get("groups") or []),
        event_count=len(payload["verified_true"]),
    )


def convert_reviews_v2_analysis(
    data_dir: Path,
    *,
    input_filename: str = DEFAULT_REVIEW_V2_ANALYSIS_FILE,
    output_filename: str = EVENT_REVIEW_V2_FILE,
) -> list[ReviewV2GroupsConvertStats]:
    record_dirs = discover_record_dirs(Path(data_dir))
    if not record_dirs:
        raise FileNotFoundError(f"no valid record directories found under: {data_dir}")
    return [
        convert_record_review_v2_analysis(
            record_dir,
            input_filename=input_filename,
            output_filename=output_filename,
        )
        for record_dir in record_dirs
    ]
