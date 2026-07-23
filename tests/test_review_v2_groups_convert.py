"""event_review_v2 grouped annotation conversion tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fixtures import _skeleton_row


def test_convert_review_v2_analysis_payload_outputs_minimal_review_v2():
    from analysis.review_v2_groups_convert import convert_review_v2_analysis_payload

    payload = convert_review_v2_analysis_payload(
        {
            "schema": 1,
            "record_id": "record_001",
            "event_count": 99,
            "groups": [
                {
                    "group_index": 1,
                    "shelf_code": "S1",
                    "box_id": "1007",
                    "event_types": ["alarm"],
                    "person_track_ids": [3],
                    "frame_indices": [3, 1, 2],
                    "missing_frame_indices": [],
                },
                {
                    "group_index": 2,
                    "shelf_code": "S1",
                    "box_id": "1008",
                    "event_type": "collision",
                    "person_track_id": 4,
                    "frame_indices": [8],
                },
            ],
        }
    )

    assert set(payload) == {"schema", "verified_true"}
    assert payload["schema"] == 2
    assert payload["verified_true"] == [
        {
            "event_type": "alarm",
            "frame_idx": 1,
            "is_pick": True,
            "person_track_id": 3,
            "shelf_code": "S1",
            "box_id": "1007",
        },
        {
            "event_type": "alarm",
            "frame_idx": 2,
            "is_pick": True,
            "person_track_id": 3,
            "shelf_code": "S1",
            "box_id": "1007",
        },
        {
            "event_type": "alarm",
            "frame_idx": 3,
            "is_pick": True,
            "person_track_id": 3,
            "shelf_code": "S1",
            "box_id": "1007",
        },
        {
            "event_type": "collision",
            "frame_idx": 8,
            "is_pick": True,
            "person_track_id": 4,
            "shelf_code": "S1",
            "box_id": "1008",
        },
    ]
    assert all(set(event) == {"event_type", "frame_idx", "is_pick", "person_track_id", "shelf_code", "box_id"} for event in payload["verified_true"])


def test_convert_record_review_v2_analysis_writes_event_review_v2(tmp_path: Path):
    from analysis.review_v2_groups_convert import convert_record_review_v2_analysis

    record_dir = tmp_path / "record_001"
    record_dir.mkdir(parents=True)
    (record_dir / "annotation.json").write_text(
        json.dumps(
            {
                "annotation_size": {"width": 640, "height": 480},
                "shelves": [{"shelf_code": "S1", "boxes": [{"box_id": "1007", "video_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame([_skeleton_row(1, left_wrist=(1, 1), right_wrist=(2, 2))]).to_parquet(
        record_dir / "skeleton.parquet",
        index=False,
    )
    (record_dir / "event_review_v2_analysis.json").write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "shelf_code": "S1",
                        "box_id": "1007",
                        "person_track_ids": [2],
                        "frame_indices": [1, 2],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    stats = convert_record_review_v2_analysis(record_dir)

    output_path = record_dir / "event_review_v2.json"
    assert Path(stats.output_path) == output_path.resolve()
    assert stats.group_count == 1
    assert stats.event_count == 2
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(data) == {"schema", "verified_true"}
    assert len(data["verified_true"]) == 2
    assert data["verified_true"][0]["event_type"] == "collision"


def test_convert_review_v2_analysis_payload_normalizes_person_track_id_to_int():
    from analysis.review_v2_groups_convert import convert_review_v2_analysis_payload

    payload = convert_review_v2_analysis_payload(
        {
            "groups": [
                {
                    "shelf_code": "S1",
                    "box_id": "1007",
                    "person_track_id": "3",
                    "frame_indices": [1],
                },
                {
                    "shelf_code": "S1",
                    "box_id": "1008",
                    "person_track_ids": ["4"],
                    "frame_indices": [2],
                },
            ]
        }
    )

    assert [type(event["person_track_id"]) for event in payload["verified_true"]] == [int, int]
    assert [event["person_track_id"] for event in payload["verified_true"]] == [3, 4]
