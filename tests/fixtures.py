"""合成测试数据生成。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def make_fixture_record(output_dir: Path) -> Path:
    """生成一条可训练/评测的合成记录。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation = {
        "annotation_size": {"width": 640, "height": 480},
        "shelves": [
            {
                "shelf_code": "S1",
                "shelf_name": "货架1",
                "boxes": [
                    {
                        "box_id": "A1",
                        "shelf_code": "S1",
                        "video_polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
                    },
                    {
                        "box_id": "A2",
                        "shelf_code": "S1",
                        "video_polygon": [[300, 100], [400, 100], [400, 200], [300, 200]],
                    },
                ],
            }
        ],
    }
    (output_dir / "annotation.json").write_text(
        json.dumps(annotation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows: list[dict] = []
    # 帧 1-5：无取货，手腕远离货框
    for fi in range(1, 6):
        rows.append(_skeleton_row(fi, left_wrist=(10, 10), right_wrist=(15, 12)))
    # 帧 6-8：取货 A1，手腕落在 infer 缩放后的 A1 多边形内
    for fi in range(6, 9):
        rows.append(_skeleton_row(fi, left_wrist=(50, 45), right_wrist=(55, 48)))
    # 帧 9-10：取货 A2
    for fi in range(9, 11):
        rows.append(_skeleton_row(fi, left_wrist=(130, 45), right_wrist=(140, 48)))

    pd.DataFrame(rows).to_parquet(output_dir / "skeleton.parquet", index=False)

    event_review = {
        "schema": 1,
        "status": "completed",
        "verified_true": [
            {
                "event_type": "collision",
                "frame_idx": 6,
                "box_tokens": ["S1:A1"],
                "confirmed_box_tokens": ["S1:A1"],
            },
            {
                "event_type": "collision",
                "frame_idx": 7,
                "box_tokens": ["S1:A1"],
                "confirmed_box_tokens": ["S1:A1"],
            },
            {
                "event_type": "collision",
                "frame_idx": 9,
                "box_tokens": ["S1:A2"],
                "confirmed_box_tokens": ["S1:A2"],
            },
        ],
    }
    (output_dir / "event_review.json").write_text(
        json.dumps(event_review, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir


def _skeleton_row(
    frame_idx: int,
    *,
    left_wrist: tuple[float, float],
    right_wrist: tuple[float, float],
) -> dict:
    row: dict = {
        "frame_idx": frame_idx,
        "source_frame_idx": frame_idx,
        "timestamp_sec": frame_idx / 25.0,
        "person_id": 0,
        "person_track_id": 1,
        "bbox_x1": 40.0,
        "bbox_y1": 40.0,
        "bbox_x2": 420.0,
        "bbox_y2": 420.0,
    }
    for i in range(17):
        row[f"kpt_{i}_x"] = None
        row[f"kpt_{i}_y"] = None
        row[f"kpt_{i}_score"] = None

    lx, ly = left_wrist
    rx, ry = right_wrist
    row["kpt_5_x"], row["kpt_5_y"], row["kpt_5_score"] = 200.0, 120.0, 0.9
    row["kpt_6_x"], row["kpt_6_y"], row["kpt_6_score"] = 240.0, 120.0, 0.9
    row["kpt_9_x"], row["kpt_9_y"], row["kpt_9_score"] = lx, ly, 0.95
    row["kpt_10_x"], row["kpt_10_y"], row["kpt_10_score"] = rx, ry, 0.95
    return row
