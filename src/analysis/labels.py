"""从 event_review.json 构建监督信号。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analysis.box_layout import BoxNumericCode, resolve_layout_token


LEGACY_REVIEW_FIELDS = {
    "confirmed_box_tokens",
    "confirmed_box_token",
    "tokens",
    "box_tokens",
    "box_token",
    "token",
}


class ReviewSchemaError(ValueError):
    """Raised when event_review.json is not in the supported v2 schema."""


def _upgrade_message(record_id: str) -> str:
    return (
        f"event_review.json for record={record_id} uses an unsupported old schema. "
        "Please upgrade review annotations to schema v2 before feature extraction/training "
        "(use the review upgrade script/module, e.g. analysis.review_upgrade)."
    )


def validate_event_review_v2(event_review: dict[str, Any] | None, *, record_id: str) -> None:
    if not event_review:
        return
    if int(event_review.get("schema") or 0) != 2:
        raise ReviewSchemaError(_upgrade_message(record_id))
    for item in event_review.get("verified_true") or []:
        if not isinstance(item, dict):
            continue
        legacy = sorted(LEGACY_REVIEW_FIELDS & set(item))
        if legacy:
            raise ReviewSchemaError(
                f"{_upgrade_message(record_id)} Legacy fields found: {', '.join(legacy)}"
            )
        if item.get("is_pick") is not True:
            raise ReviewSchemaError(
                f"event_review schema v2 for record={record_id} is invalid. "
                "Each verified_true event must include is_pick=true."
            )
        if not str(item.get("shelf_code") or "").strip() or not str(item.get("box_id") or "").strip():
            raise ReviewSchemaError(
                f"event_review schema v2 for record={record_id} is invalid. "
                "Each verified_true event must include non-empty shelf_code and box_id. "
                "Please fix or regenerate event_review_v2.json."
            )
        if item.get("person_track_id") is None:
            raise ReviewSchemaError(
                f"event_review schema v2 for record={record_id} is invalid. "
                "Each verified_true event must include person_track_id."
            )


def extract_confirmed_box_tokens(entry: dict[str, Any]) -> list[str]:
    shelf_code = str(entry.get("shelf_code") or "").strip()
    box_id = str(entry.get("box_id") or "").strip()
    if shelf_code and box_id:
        return [f"{shelf_code}:{box_id}"]
    return []


@dataclass
class FrameLabel:
    frame_idx: int
    is_picking: bool = False
    confirmed_box_tokens: list[str] = field(default_factory=list)
    confirmed_box_codes: list[int] = field(default_factory=list)
    picking_person_track_ids: list[int] = field(default_factory=list)


@dataclass
class RecordLabels:
    """单条记录的帧级标签。"""

    record_id: str
    frame_labels: dict[int, FrameLabel] = field(default_factory=dict)

    def label_for(self, frame_idx: int) -> FrameLabel:
        if frame_idx in self.frame_labels:
            return self.frame_labels[frame_idx]
        return FrameLabel(
            frame_idx=frame_idx,
            is_picking=False,
            confirmed_box_tokens=[],
            confirmed_box_codes=[],
            picking_person_track_ids=[],
        )


def extract_picking_person_track_ids(entry: dict[str, Any]) -> list[int]:
    raw_values: list[Any] = []
    if isinstance(entry.get("person_track_ids"), list):
        raw_values.extend(entry["person_track_ids"])
    if entry.get("person_track_id") is not None:
        raw_values.append(entry.get("person_track_id"))
    person = entry.get("person")
    if isinstance(person, dict) and person.get("person_track_id") is not None:
        raw_values.append(person.get("person_track_id"))

    out: list[int] = []
    for raw in raw_values:
        try:
            out.append(int(raw))
        except (TypeError, ValueError):
            continue
    return list(dict.fromkeys(out))


def load_event_review(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def build_labels_from_event_review(
    event_review: dict[str, Any] | None,
    *,
    record_id: str,
    all_frame_indices: list[int] | None = None,
) -> RecordLabels:
    """
    根据 verified_true 构建监督信号。
    无 event_review 时，所有帧均为非取货。
    """
    labels = RecordLabels(record_id=record_id)
    if all_frame_indices:
        for fi in all_frame_indices:
            labels.frame_labels[fi] = FrameLabel(frame_idx=fi)

    if not event_review:
        return labels
    validate_event_review_v2(event_review, record_id=record_id)

    for item in event_review.get("verified_true") or []:
        if not isinstance(item, dict):
            continue
        try:
            frame_idx = int(item.get("frame_idx") or 0)
        except (TypeError, ValueError):
            continue
        if frame_idx < 0:
            continue
        confirmed = extract_confirmed_box_tokens(item)
        track_ids = extract_picking_person_track_ids(item)
        label = labels.frame_labels.get(frame_idx)
        if label is None:
            label = FrameLabel(frame_idx=frame_idx)
            labels.frame_labels[frame_idx] = label
        label.is_picking = True
        label.confirmed_box_tokens = list(dict.fromkeys([*label.confirmed_box_tokens, *confirmed]))
        label.picking_person_track_ids = list(dict.fromkeys([*label.picking_person_track_ids, *track_ids]))
    return labels


def enrich_labels_with_box_layout(
    labels: RecordLabels,
    layout: dict[str, BoxNumericCode],
) -> None:
    """将 event_review 中的 Box_<id> 等形式解析为 canonical token，并写入数值编码。"""
    for label in labels.frame_labels.values():
        if not label.confirmed_box_tokens:
            label.confirmed_box_codes = []
            continue
        canonical_tokens: list[str] = []
        codes: list[int] = []
        for token in label.confirmed_box_tokens:
            canonical = resolve_layout_token(token, layout)
            entry = layout.get(canonical)
            if entry is None:
                continue
            canonical_tokens.append(canonical)
            codes.append(entry.encode())
        label.confirmed_box_tokens = canonical_tokens
        label.confirmed_box_codes = codes
