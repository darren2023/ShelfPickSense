"""Review schema upgrade tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _write_multi_person_record(record_dir: Path) -> Path:
    from fixtures import _skeleton_row

    record_dir.mkdir(parents=True, exist_ok=True)
    annotation = {
        "annotation_size": {"width": 640, "height": 480},
        "shelves": [
            {
                "shelf_code": "S1",
                "boxes": [
                    {
                        "box_id": "A1",
                        "shelf_code": "S1",
                        "video_polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
                    }
                ],
            }
        ],
    }
    (record_dir / "annotation.json").write_text(json.dumps(annotation, ensure_ascii=False), encoding="utf-8")
    far = _skeleton_row(1, left_wrist=(500, 400), right_wrist=(510, 405))
    far["person_id"] = 1
    far["person_track_id"] = 1
    near = _skeleton_row(1, left_wrist=(150, 150), right_wrist=(160, 155))
    near["person_id"] = 2
    near["person_track_id"] = 2
    pd.DataFrame([far, near]).to_parquet(record_dir / "skeleton.parquet", index=False)
    review = {
        "schema": 1,
        "status": "completed",
        "verified_true": [
            {
                "event_type": "collision",
                "frame_idx": 1,
                "box_tokens": ["S1:A1"],
            }
        ],
    }
    (record_dir / "event_review.json").write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")
    return record_dir


def test_upgrade_review_assigns_nearest_person(tmp_path: Path):
    from analysis.records import load_record
    from analysis.review_upgrade import upgrade_review_payload

    record = load_record(_write_multi_person_record(tmp_path / "record_001"), validate_review_schema=False)
    upgraded, stats = upgrade_review_payload(record, record.event_review or {})

    event = upgraded["verified_true"][0]
    assert upgraded["schema"] == 2
    assert upgraded["task"]["type"] == "pick_review"
    assert event["is_pick"] is True
    assert "token" not in event
    assert "source_frame_idx" not in event
    assert "tokens" not in event
    assert "box_tokens" not in event
    assert "confirmed_box_tokens" not in event
    assert "person" not in event
    assert "person_id" not in event
    assert event["shelf_code"] == "S1"
    assert event["box_id"] == "A1"
    assert event["person_track_id"] == 2
    assert stats.upgraded_events == 1
    assert stats.assigned_persons == 1


def test_upgrade_review_prefers_alarm_for_duplicate_frame(tmp_path: Path):
    from analysis.records import load_record
    from analysis.review_upgrade import upgrade_review_payload

    record = load_record(_write_multi_person_record(tmp_path / "record_001"), validate_review_schema=False)
    review = {
        "schema": 1,
        "verified_true": [
            {"event_type": "collision", "frame_idx": 1, "box_tokens": ["S1:A1"], "confirmed_box_tokens": ["S1:A1"]},
            {"event_type": "alarm", "frame_idx": 1, "box_tokens": ["S1:A1"], "confirmed_box_tokens": ["S1:A1"]},
        ],
    }

    upgraded, stats = upgrade_review_payload(record, review)

    assert stats.total_events == 2
    assert stats.upgraded_events == 1
    assert upgraded["verified_true"][0]["event_type"] == "alarm"


def test_upgrade_review_uses_confirmed_token_for_person_selection(tmp_path: Path):
    from analysis.records import load_record
    from analysis.review_upgrade import upgrade_review_payload

    record_dir = _write_multi_person_record(tmp_path / "record_001")
    annotation = json.loads((record_dir / "annotation.json").read_text(encoding="utf-8"))
    annotation["shelves"][0]["boxes"].append(
        {
            "box_id": "B1",
            "shelf_code": "S1",
            "video_polygon": [[480, 350], [560, 350], [560, 430], [480, 430]],
        }
    )
    (record_dir / "annotation.json").write_text(json.dumps(annotation, ensure_ascii=False), encoding="utf-8")
    review = {
        "schema": 1,
        "verified_true": [
            {
                "event_type": "collision",
                "frame_idx": 1,
                "box_tokens": ["S1:B1", "S1:A1"],
                "confirmed_box_tokens": ["S1:A1"],
            }
        ],
    }
    record = load_record(record_dir, validate_review_schema=False)

    upgraded, _stats = upgrade_review_payload(record, review)

    event = upgraded["verified_true"][0]
    assert event["box_id"] == "A1"
    assert event["person_track_id"] == 2


def test_person_track_id_limits_positive_frame_samples(tmp_path: Path):
    from analysis.dataset import build_dataset
    from analysis.features.registry import default_registry
    from analysis.records import load_record
    from analysis.review_upgrade import upgrade_record_review

    record_dir = _write_multi_person_record(tmp_path / "record_001")
    upgrade_record_review(record_dir, in_place=True, backup=False)
    record = load_record(record_dir)
    dataset = build_dataset([record], default_registry())

    positives = [sample for sample in dataset.frame_samples if sample.is_picking]
    assert len(positives) == 1
    assert positives[0].person_track_id == 2
    assert positives[0].shelf_code == "S1"
