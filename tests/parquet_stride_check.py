"""检查 skeleton.parquet 的 frame_idx 是否为隔帧（稀疏/等间隔）采样。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

SKELETON_FILE = "skeleton.parquet"


@dataclass
class FrameSamplingReport:
    """单条 parquet / record 的帧采样分析结果。"""

    path: str
    frame_count: int
    min_frame_idx: int | None
    max_frame_idx: int | None
    gaps: list[int]
    unique_gaps: list[int]
    sampling_mode: str
    """dense=连续逐帧; regular_stride=固定间隔隔帧; irregular=间隔不一致; empty=无帧"""
    inferred_stride: int | None
    """regular_stride 时为 gap 值；dense 为 1；irregular/empty 为 None"""
    is_subsampled: bool
    """True 表示非 dense（隔帧或不规则稀疏）"""

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_frame_indices(frame_indices: list[int]) -> tuple[str, int | None, list[int], bool]:
    """根据有序唯一 frame_idx 推断采样模式。"""
    if not frame_indices:
        return "empty", None, [], False

    sorted_idx = sorted(int(v) for v in frame_indices)
    if len(sorted_idx) == 1:
        return "dense", 1, [], False

    gaps = [sorted_idx[i + 1] - sorted_idx[i] for i in range(len(sorted_idx) - 1)]
    unique = sorted(set(gaps))

    if unique == [1]:
        return "dense", 1, gaps, False

    if len(unique) == 1 and unique[0] > 1:
        return "regular_stride", unique[0], gaps, True

    return "irregular", None, gaps, True


def analyze_skeleton_parquet(parquet_path: Path) -> FrameSamplingReport:
    """读取 skeleton.parquet 并分析 frame_idx 间隔。"""
    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path)
    if df.empty or "frame_idx" not in df.columns:
        indices: list[int] = []
    else:
        indices = [int(v) for v in df["frame_idx"].unique()]

    mode, stride, gaps, is_sub = analyze_frame_indices(indices)
    sorted_idx = sorted(indices)
    return FrameSamplingReport(
        path=str(parquet_path.resolve()),
        frame_count=len(sorted_idx),
        min_frame_idx=sorted_idx[0] if sorted_idx else None,
        max_frame_idx=sorted_idx[-1] if sorted_idx else None,
        gaps=gaps,
        unique_gaps=sorted(set(gaps)),
        sampling_mode=mode,
        inferred_stride=stride,
        is_subsampled=is_sub,
    )


def _is_record_dir(path: Path) -> bool:
    return path.is_dir() and (path / SKELETON_FILE).is_file()


def discover_parquet_paths(data_path: Path) -> list[Path]:
    """接受 skeleton.parquet 文件、record 目录或包含多条 record 的父目录。"""
    data_path = Path(data_path)
    if data_path.is_file() and data_path.name == SKELETON_FILE:
        return [data_path]
    if _is_record_dir(data_path):
        return [data_path / SKELETON_FILE]
    if not data_path.is_dir():
        raise FileNotFoundError(f"路径不存在: {data_path}")

    found = [child / SKELETON_FILE for child in sorted(data_path.iterdir()) if _is_record_dir(child)]
    if not found:
        raise FileNotFoundError(f"在 {data_path} 下未找到含 {SKELETON_FILE} 的记录目录")
    return found


def analyze_path(data_path: Path) -> list[FrameSamplingReport]:
    return [analyze_skeleton_parquet(p) for p in discover_parquet_paths(data_path)]


def summarize_reports(reports: list[FrameSamplingReport]) -> dict:
    modes = {r.sampling_mode for r in reports}
    strides = {r.inferred_stride for r in reports if r.inferred_stride is not None}
    return {
        "record_count": len(reports),
        "sampling_modes": sorted(modes),
        "inferred_strides": sorted(strides),
        "all_dense": len(reports) > 0 and all(r.sampling_mode == "dense" for r in reports),
        "any_subsampled": any(r.is_subsampled for r in reports),
        "subsampled_records": sum(1 for r in reports if r.is_subsampled),
    }


def format_report_line(report: FrameSamplingReport) -> str:
    name = Path(report.path).parent.name
    if report.sampling_mode == "empty":
        return f"{name}: 空 parquet"
    if report.sampling_mode == "dense":
        return (
            f"{name}: 逐帧 (dense), frames={report.frame_count}, "
            f"idx={report.min_frame_idx}..{report.max_frame_idx}"
        )
    if report.sampling_mode == "regular_stride":
        return (
            f"{name}: 隔帧 stride={report.inferred_stride}, frames={report.frame_count}, "
            f"idx={report.min_frame_idx}..{report.max_frame_idx}, gaps={report.unique_gaps}"
        )
    gap_preview = report.unique_gaps[:8]
    suffix = "..." if len(report.unique_gaps) > 8 else ""
    return (
        f"{name}: 不规则间隔 (irregular), frames={report.frame_count}, "
        f"idx={report.min_frame_idx}..{report.max_frame_idx}, unique_gaps={gap_preview}{suffix}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查 skeleton.parquet 的 frame_idx 是否隔帧采样")
    parser.add_argument(
        "path",
        help="skeleton.parquet 路径、单条 record 目录或多 record 父目录",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出完整报告",
    )
    parser.add_argument(
        "--fail-on-subsampled",
        action="store_true",
        help="若存在隔帧/不规则稀疏 record 则以退出码 1 结束（可用于 CI）",
    )
    args = parser.parse_args(argv)

    try:
        reports = analyze_path(Path(args.path))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    summary = summarize_reports(reports)
    if args.json:
        payload = {
            "summary": summary,
            "records": [r.to_dict() for r in reports],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"共 {summary['record_count']} 条 record")
        for report in reports:
            print(format_report_line(report))
        print(
            f"汇总: all_dense={summary['all_dense']}, "
            f"subsampled_records={summary['subsampled_records']}/{summary['record_count']}, "
            f"modes={summary['sampling_modes']}, strides={summary['inferred_strides']}"
        )

    if args.fail_on_subsampled and summary["any_subsampled"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
