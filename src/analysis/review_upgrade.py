"""Upgrade legacy event review files with pick person information."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis.box_layout import resolve_layout_token
from analysis.constants import EVENT_REVIEW_FILE
from analysis.features.spatial import point_to_polygon_dist
from analysis.features.tracking import LEFT_WRIST, RIGHT_WRIST, get_side_point, person_anchor, person_track_id
from analysis.records import FramePersons, RecordData, discover_record_dirs, load_record


@dataclass(frozen=True)
class UpgradeStats:
    record_id: str
    input_path: str
    output_path: str
    total_events: int
    upgraded_events: int
    assigned_persons: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "total_events": self.total_events,
            "upgraded_events": self.upgraded_events,
            "assigned_persons": self.assigned_persons,
        }


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _event_tokens(event: dict[str, Any], record: RecordData) -> list[str]:
    raw_values: list[str] = []
    for key in ("confirmed_box_tokens",):
        raw = event.get(key)
        if isinstance(raw, list):
            raw_values.extend(str(item).strip() for item in raw if str(item).strip())
    for key in ("confirmed_box_token",):
        raw = str(event.get(key) or "").strip()
        if raw:
            raw_values.append(raw)
    if not raw_values:
        for key in ("tokens", "box_tokens"):
            raw = event.get(key)
            if isinstance(raw, list):
                raw_values.extend(str(item).strip() for item in raw if str(item).strip())
        for key in ("token", "box_token"):
            raw = str(event.get(key) or "").strip()
            if raw:
                raw_values.append(raw)
    if not raw_values:
        shelf_code = str(event.get("shelf_code") or "").strip()
        box_id = str(event.get("box_id") or "").strip()
        if shelf_code and box_id:
            raw_values.append(f"{shelf_code}:{box_id}")
        elif box_id:
            raw_values.append(box_id)

    tokens: list[str] = []
    for raw in raw_values:
        token = resolve_layout_token(raw, record.box_layout)
        if token in record.box_index:
            tokens.append(token)
    return list(dict.fromkeys(tokens))


def _event_priority(event: dict[str, Any]) -> tuple[int, int]:
    event_type = str(event.get("event_type") or "").strip().lower()
    if event_type == "alarm":
        type_priority = 0
    else:
        type_priority = 1
    has_confirmed = 0 if _has_confirmed_token(event) else 1
    return type_priority, has_confirmed


def _has_confirmed_token(event: dict[str, Any]) -> bool:
    raw = event.get("confirmed_box_tokens")
    if isinstance(raw, list) and any(str(item).strip() for item in raw):
        return True
    return bool(str(event.get("confirmed_box_token") or "").strip())


def _dedupe_events_by_frame(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    passthrough: list[dict[str, Any]] = []
    for event in events:
        try:
            frame_idx = int(event.get("frame_idx") or 0)
        except (TypeError, ValueError):
            passthrough.append(event)
            continue
        by_frame.setdefault(frame_idx, []).append(event)
    deduped = list(passthrough)
    for _frame_idx, frame_events in sorted(by_frame.items()):
        frame_events.sort(key=_event_priority)
        deduped.append(frame_events[0])
    return deduped


def _person_point_candidates(person: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for side in (LEFT_WRIST, RIGHT_WRIST):
        point = get_side_point(person, side)
        if point is not None:
            points.append(point)
    bbox = person.get("bbox")
    if isinstance(bbox, list | tuple) and len(bbox) >= 4:
        try:
            points.append(((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0))
        except (TypeError, ValueError):
            pass
    anchor = person_anchor(person)
    if anchor is not None:
        points.append(anchor)
    return points


def _person_distance_to_tokens(person: dict[str, Any], tokens: list[str], record: RecordData) -> float:
    points = _person_point_candidates(person)
    if not points:
        return float("inf")
    best = float("inf")
    for token in tokens:
        box = record.box_index.get(token)
        if box is None:
            continue
        for x, y in points:
            best = min(best, point_to_polygon_dist(x, y, box.polygon))
    return best


def select_pick_person(
    frame: FramePersons | None,
    tokens: list[str],
    record: RecordData,
) -> tuple[dict[str, Any] | None, str, float | None]:
    """Select the most likely picking person for a token event."""
    if frame is None or not frame.persons:
        return None, "no_person", None
    if len(frame.persons) == 1:
        return frame.persons[0], "single_person", _person_distance_to_tokens(frame.persons[0], tokens, record)

    best_person: dict[str, Any] | None = None
    best_dist = float("inf")
    for person in frame.persons:
        dist = _person_distance_to_tokens(person, tokens, record)
        if dist < best_dist:
            best_dist = dist
            best_person = person
    return best_person, "nearest_to_token", best_dist if math.isfinite(best_dist) else None


def upgrade_review_payload(record: RecordData, review: dict[str, Any]) -> tuple[dict[str, Any], UpgradeStats]:
    frame_index = record.frame_index()
    raw_events = [item for item in review.get("verified_true") or [] if isinstance(item, dict)]
    source_events = _dedupe_events_by_frame(raw_events)
    upgraded_events: list[dict[str, Any]] = []
    assigned = 0

    for event in source_events:
        tokens = _event_tokens(event, record)
        if not tokens:
            continue
        try:
            frame_idx = int(event.get("frame_idx") or 0)
        except (TypeError, ValueError):
            continue
        frame = frame_index.get(frame_idx)
        person, _method, _distance = select_pick_person(frame, tokens, record)
        track_id = person_track_id(person) if person is not None else None
        if track_id is not None:
            assigned += 1
        first_token = tokens[0]
        layout = record.box_layout.get(first_token)
        upgraded = {
            "event_type": event.get("event_type") or "pick",
            "frame_idx": frame_idx,
            "is_pick": True,
        }
        if track_id is not None:
            upgraded["person_track_id"] = track_id
        if layout is not None:
            upgraded["shelf_code"] = layout.shelf_code
            upgraded["box_id"] = layout.box_id
        else:
            upgraded["token"] = first_token
        upgraded_events.append(upgraded)

    output = dict(review)
    output["schema"] = 2
    output["status"] = review.get("status") or "completed"
    output["task"] = {
        "type": "pick_review",
        "version": 1,
        "record_id": record.record_id,
        "source_schema": review.get("schema", 1),
        "upgraded_at": datetime.now(timezone.utc).isoformat(),
        "description": "is_pick + shelf_code/box_id + picking person_track_id when available",
    }
    output["verified_true"] = upgraded_events

    stats = UpgradeStats(
        record_id=record.record_id,
        input_path=str(record.record_dir / EVENT_REVIEW_FILE),
        output_path="",
        total_events=len(raw_events),
        upgraded_events=len(upgraded_events),
        assigned_persons=assigned,
    )
    return output, stats


def upgrade_record_review(
    record_dir: Path,
    *,
    review_filename: str = EVENT_REVIEW_FILE,
    output_path: Path | None = None,
    in_place: bool = False,
    backup: bool = True,
) -> UpgradeStats:
    record = load_record(record_dir, validate_review_schema=False)
    input_path = record.record_dir / review_filename
    if not input_path.is_file():
        raise FileNotFoundError(f"未找到 {EVENT_REVIEW_FILE}: {input_path}")
    review = _read_json(input_path)
    payload, stats = upgrade_review_payload(record, review)

    default_output = record.record_dir / f"{input_path.stem}_v2{input_path.suffix}"
    out = input_path if in_place else (Path(output_path) if output_path else default_output)
    if in_place and backup:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        shutil.copy2(input_path, backup_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return UpgradeStats(
        record_id=stats.record_id,
        input_path=str(input_path),
        output_path=str(out),
        total_events=stats.total_events,
        upgraded_events=stats.upgraded_events,
        assigned_persons=stats.assigned_persons,
    )


def upgrade_reviews(
    data_dir: Path,
    *,
    review_filename: str = EVENT_REVIEW_FILE,
    output_dir: Path | None = None,
    in_place: bool = False,
    backup: bool = True,
) -> list[UpgradeStats]:
    records = discover_record_dirs(data_dir)
    if not records:
        raise FileNotFoundError(f"未找到有效记录目录: {data_dir}")
    results: list[UpgradeStats] = []
    for record_dir in records:
        output_path = None
        if output_dir is not None:
            output_path = Path(output_dir) / Path(record_dir).name / review_filename
        results.append(
            upgrade_record_review(
                record_dir,
                review_filename=review_filename,
                output_path=output_path,
                in_place=in_place,
                backup=backup,
            )
        )
    return results
