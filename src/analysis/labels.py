"""从 event_review.json 构建监督信号。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def extract_confirmed_box_tokens(entry: dict[str, Any]) -> list[str]:
    raw_list = entry.get("confirmed_box_tokens")
    if isinstance(raw_list, list):
        tokens = [str(t).strip() for t in raw_list if str(t).strip()]
        if tokens:
            return tokens
    single = str(entry.get("confirmed_box_token") or "").strip()
    return [single] if single else []


@dataclass
class FrameLabel:
    frame_idx: int
    is_picking: bool = False
    confirmed_box_tokens: list[str] = field(default_factory=list)


@dataclass
class RecordLabels:
    """单条记录的帧级标签。"""

    record_id: str
    frame_labels: dict[int, FrameLabel] = field(default_factory=dict)

    def label_for(self, frame_idx: int) -> FrameLabel:
        if frame_idx in self.frame_labels:
            return self.frame_labels[frame_idx]
        return FrameLabel(frame_idx=frame_idx, is_picking=False, confirmed_box_tokens=[])


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
        labels.frame_labels[frame_idx] = FrameLabel(
            frame_idx=frame_idx,
            is_picking=True,
            confirmed_box_tokens=confirmed,
        )
    return labels
