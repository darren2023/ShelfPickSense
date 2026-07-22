"""event_review_v2 analysis tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fixtures import _skeleton_row


def _write_record_with_review_v2(record_dir: Path, events: list[dict]) -> Path:
    record_dir.mkdir(parents=True, exist_ok=True)
    annotation = {
        "annotation_size": {"width": 640, "height": 480},
        "shelves": [
            {
                "shelf_code": "S1",
                "boxes": [
                    {"box_id": "1007", "video_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
                    {"box_id": "1008", "video_polygon": [[20, 0], [30, 0], [30, 10], [20, 10]]},
                ],
            }
        ],
    }
    (record_dir / "annotation.json").write_text(json.dumps(annotation, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([_skeleton_row(1, left_wrist=(1, 1), right_wrist=(2, 2))]).to_parquet(
        record_dir / "skeleton.parquet",
        index=False,
    )
    (record_dir / "event_review_v2.json").write_text(
        json.dumps({"schema": 2, "verified_true": events}, ensure_ascii=False),
        encoding="utf-8",
    )
    return record_dir


def _event(frame_idx: int, box_id: str = "1007", event_type: str = "collision") -> dict:
    return {
        "event_type": event_type,
        "frame_idx": frame_idx,
        "is_pick": True,
        "person_track_id": 2,
        "shelf_code": "S1",
        "box_id": box_id,
    }


def test_analyze_review_v2_groups_sorted_events_and_missing_frames():
    from analysis.review_v2_analysis import analyze_review_v2_payload

    payload = analyze_review_v2_payload(
        {
            "schema": 2,
            "verified_true": [
                _event(4),
                _event(1),
                _event(2),
                _event(8, "1008", "alarm"),
                _event(7, "1008"),
                _event(9),
                {"frame_idx": 99, "is_pick": False, "shelf_code": "S1", "box_id": "1007"},
            ],
        },
        record_id="record_001",
    )

    assert [frame for group in payload["groups"] for frame in group["frame_indices"]] == [1, 2, 4, 7, 8, 9]
    assert "sorted_events" not in payload
    assert payload["skipped_non_pick_count"] == 1
    assert payload["group_count"] == 3
    first, second, third = payload["groups"]
    assert first["box_id"] == "1007"
    assert "events" not in first
    assert first["start_frame_idx"] == 1
    assert first["end_frame_idx"] == 4
    assert first["missing_frame_indices"] == [3]
    assert first["has_missing_frames"] is True
    assert second["box_id"] == "1008"
    assert second["frame_indices"] == [7, 8]
    assert third["box_id"] == "1007"
    assert third["frame_indices"] == [9]


def test_analyze_record_review_v2_writes_result_to_record_dir(tmp_path: Path):
    from analysis.review_v2_analysis import analyze_record_review_v2

    record_dir = _write_record_with_review_v2(
        tmp_path / "record_001",
        [_event(1), _event(3), _event(3)],
    )

    stats = analyze_record_review_v2(record_dir)

    output_path = record_dir / "event_review_v2_analysis.json"
    assert Path(stats.output_path) == output_path.resolve()
    assert output_path.is_file()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["groups"][0]["missing_frame_indices"] == [2]
    assert data["groups"][0]["duplicate_frame_indices"] == [3]
    assert stats.missing_frame_count == 1
    assert stats.duplicate_frame_count == 1


def test_analyze_review_v2_splits_same_box_when_frame_gap_exceeds_limit():
    from analysis.review_v2_analysis import analyze_review_v2_payload

    payload = analyze_review_v2_payload(
        {
            "schema": 2,
            "verified_true": [
                _event(1),
                _event(2),
                _event(103),
                _event(104),
            ],
        },
        record_id="record_001",
        max_frame_gap=100,
    )

    assert payload["max_group_frame_gap"] == 100
    assert payload["group_count"] == 2
    assert payload["groups"][0]["frame_indices"] == [1, 2]
    assert payload["groups"][1]["frame_indices"] == [103, 104]
