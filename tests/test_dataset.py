"""数据集构建与过滤测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fixtures import make_fixture_record, make_fixture_record_with_empty_skeleton_frames


def test_frame_has_valid_skeleton(tmp_path: Path):
    from analysis.dataset import frame_has_valid_skeleton
    from analysis.records import load_record

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    record = load_record(fixture_dir)
    frame = record.frames()[0]
    assert frame_has_valid_skeleton(frame) is True


def test_filter_empty_skeleton_frames_before_training(tmp_path: Path):
    from analysis.dataset import build_dataset, filter_empty_skeleton_frames
    from analysis.features.registry import default_registry
    from analysis.records import load_record
    from analysis.train import train_model

    fixture_dir = make_fixture_record_with_empty_skeleton_frames(
        tmp_path / "record_001",
        empty_frame_indices=[11, 12, 13],
    )
    record = load_record(fixture_dir)
    assert len(record.frames()) == 13

    reg = default_registry()
    dataset = build_dataset([record], reg)
    assert dataset.frame_count == 13

    filtered, removed = filter_empty_skeleton_frames(dataset, [record])
    assert removed == 3
    assert filtered.frame_count == 10
    assert filtered.positive_frame_count == 3

    result = train_model(fixture_dir, tmp_path / "model")
    assert result.frame_count == 10
    assert result.skipped_empty_skeleton_frames == 3


def test_build_dataset_feature_frame_stride(tmp_path: Path):
    from analysis.dataset import build_dataset
    from analysis.features.registry import default_registry
    from analysis.records import load_record

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    record = load_record(fixture_dir)

    dataset = build_dataset([record], default_registry(), feature_frame_stride=3)

    assert dataset.frame_count == 4
    assert [sample.frame_idx for sample in dataset.frame_samples] == [1, 4, 7, 10]


def test_build_dataset_extracts_person_by_shelf_samples(tmp_path: Path):
    from analysis.dataset import build_dataset
    from analysis.features.registry import default_registry
    from analysis.records import load_record
    from fixtures import _skeleton_row

    record_dir = tmp_path / "record_001"
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
            },
            {
                "shelf_code": "S2",
                "boxes": [
                    {
                        "box_id": "B1",
                        "shelf_code": "S2",
                        "video_polygon": [[300, 100], [400, 100], [400, 200], [300, 200]],
                    }
                ],
            },
        ],
    }
    (record_dir / "annotation.json").write_text(
        json.dumps(annotation, ensure_ascii=False),
        encoding="utf-8",
    )
    rows = []
    for track_id in (1, 2, 3):
        row = _skeleton_row(1, left_wrist=(150, 150), right_wrist=(160, 155))
        row["person_id"] = track_id
        row["person_track_id"] = track_id
        rows.append(row)
    pd.DataFrame(rows).to_parquet(record_dir / "skeleton.parquet", index=False)
    event_review = {
        "schema": 1,
        "status": "completed",
        "verified_true": [
            {
                "event_type": "collision",
                "frame_idx": 1,
                "confirmed_box_tokens": ["S1:A1"],
            }
        ],
    }
    (record_dir / "event_review.json").write_text(
        json.dumps(event_review, ensure_ascii=False),
        encoding="utf-8",
    )

    dataset = build_dataset([load_record(record_dir)], default_registry())

    assert dataset.frame_count == 6
    assert sorted(sample.person_track_id for sample in dataset.frame_samples) == [1, 1, 2, 2, 3, 3]
    assert sorted(sample.shelf_code for sample in dataset.frame_samples) == ["S1", "S1", "S1", "S2", "S2", "S2"]
    assert sum(sample.is_picking for sample in dataset.frame_samples if sample.shelf_code == "S1") == 3
    assert sum(sample.is_picking for sample in dataset.frame_samples if sample.shelf_code == "S2") == 0


def test_keep_empty_skeleton_frames_option(tmp_path: Path):
    from analysis.train import train_model

    fixture_dir = make_fixture_record_with_empty_skeleton_frames(
        tmp_path / "record_001",
        empty_frame_indices=[11, 12],
    )
    result = train_model(
        fixture_dir,
        tmp_path / "model_keep",
        filter_empty_skeleton=False,
    )
    assert result.frame_count == 12
    assert result.skipped_empty_skeleton_frames == 0
