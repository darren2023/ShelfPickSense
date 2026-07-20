"""从 event_review.json 构建监督信号。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analysis.box_layout import BoxNumericCode, resolve_layout_token


def extract_confirmed_box_tokens(entry: dict[str, Any]) -> list[str]:
    raw_list = entry.get("confirmed_box_tokens")
    if isinstance(raw_list, list):
        tokens = [str(t).strip() for t in raw_list if str(t).strip()]
        if tokens:
            return tokens
    single = str(entry.get("confirmed_box_token") or "").strip()
    if single:
        return [single]

    raw_list = entry.get("box_tokens")
    if isinstance(raw_list, list):
        tokens = [str(t).strip() for t in raw_list if str(t).strip()]
        if tokens:
            return tokens
    single = str(entry.get("box_token") or "").strip()
    return [single] if single else []


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
