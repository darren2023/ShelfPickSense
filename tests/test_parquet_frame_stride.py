"""parquet frame_idx 隔帧检测工具测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fixtures import make_fixture_record
from parquet_stride_check import (
    analyze_frame_indices,
    analyze_path,
    analyze_skeleton_parquet,
    discover_parquet_paths,
    summarize_reports,
)


def _write_skeleton_parquet(path: Path, frame_indices: list[int]) -> None:
    rows = []
    for fi in frame_indices:
        row = {"frame_idx": fi, "source_frame_idx": fi, "timestamp_sec": fi / 25.0, "person_id": 0}
        for i in range(17):
            row[f"kpt_{i}_x"] = 100.0
            row[f"kpt_{i}_y"] = 100.0
            row[f"kpt_{i}_score"] = 0.9
        rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_analyze_frame_indices_dense():
    mode, stride, gaps, is_sub = analyze_frame_indices([1, 2, 3, 4, 5])
    assert mode == "dense"
    assert stride == 1
    assert gaps == [1, 1, 1, 1]
    assert is_sub is False


def test_analyze_frame_indices_regular_stride_2():
    mode, stride, gaps, is_sub = analyze_frame_indices([1, 3, 5, 7, 9])
    assert mode == "regular_stride"
    assert stride == 2
    assert gaps == [2, 2, 2, 2]
    assert is_sub is True


def test_analyze_frame_indices_irregular():
    mode, stride, gaps, is_sub = analyze_frame_indices([1, 2, 4, 7])
    assert mode == "irregular"
    assert stride is None
    assert gaps == [1, 2, 3]
    assert is_sub is True


def test_analyze_frame_indices_empty():
    mode, stride, gaps, is_sub = analyze_frame_indices([])
    assert mode == "empty"
    assert stride is None
    assert gaps == []
    assert is_sub is False


def test_analyze_skeleton_parquet_file(tmp_path: Path):
    parquet = tmp_path / "skeleton.parquet"
    _write_skeleton_parquet(parquet, [1, 3, 5])

    report = analyze_skeleton_parquet(parquet)
    assert report.sampling_mode == "regular_stride"
    assert report.inferred_stride == 2
    assert report.frame_count == 3
    assert report.is_subsampled is True


def test_analyze_record_dir_and_parent_dir(tmp_path: Path):
    record_a = tmp_path / "record_a"
    record_b = tmp_path / "record_b"
    record_a.mkdir()
    record_b.mkdir()
    (record_a / "annotation.json").write_text("{}", encoding="utf-8")
    (record_b / "annotation.json").write_text("{}", encoding="utf-8")
    _write_skeleton_parquet(record_a / "skeleton.parquet", list(range(1, 6)))
    _write_skeleton_parquet(record_b / "skeleton.parquet", [1, 3, 5, 7])

    dense_reports = analyze_path(record_a)
    assert len(dense_reports) == 1
    assert dense_reports[0].sampling_mode == "dense"

    all_reports = analyze_path(tmp_path)
    assert len(all_reports) == 2
    summary = summarize_reports(all_reports)
    assert summary["all_dense"] is False
    assert summary["subsampled_records"] == 1
    assert summary["any_subsampled"] is True


def test_fixture_record_is_dense(tmp_path: Path):
    fixture_dir = make_fixture_record(tmp_path / "record_001")
    report = analyze_skeleton_parquet(fixture_dir / "skeleton.parquet")
    assert report.sampling_mode == "dense"
    assert report.frame_count == 10
    assert report.min_frame_idx == 1
    assert report.max_frame_idx == 10
    assert report.is_subsampled is False


def test_discover_parquet_paths_single_file(tmp_path: Path):
    parquet = tmp_path / "skeleton.parquet"
    _write_skeleton_parquet(parquet, [1, 2])
    assert discover_parquet_paths(parquet) == [parquet]


def test_main_json_output(tmp_path: Path, capsys):
    from parquet_stride_check import main

    record = tmp_path / "record_001"
    record.mkdir()
    (record / "annotation.json").write_text("{}", encoding="utf-8")
    _write_skeleton_parquet(record / "skeleton.parquet", [1, 4, 7])

    code = main([str(record), "--json"])
    assert code == 0
    out = capsys.readouterr().out
    assert "regular_stride" in out
    assert "inferred_stride" in out


def test_main_fail_on_subsampled(tmp_path: Path):
    from parquet_stride_check import main

    record = tmp_path / "record_001"
    record.mkdir()
    (record / "annotation.json").write_text("{}", encoding="utf-8")
    _write_skeleton_parquet(record / "skeleton.parquet", [1, 3, 5])

    assert main([str(record), "--fail-on-subsampled"]) == 1
    assert main([str(record)]) == 0
