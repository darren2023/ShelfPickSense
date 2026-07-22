"""Analyze event_review_v2.json continuity groups."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.constants import EVENT_REVIEW_V2_FILE  # noqa: E402
from analysis.review_v2_analysis import (  # noqa: E402
    DEFAULT_MAX_GROUP_FRAME_GAP,
    DEFAULT_REVIEW_V2_ANALYSIS_FILE,
    analyze_reviews_v2,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sort event_review_v2 events, group adjacent picks by shelf/box, and report missing frames."
    )
    parser.add_argument("--data-dir", required=True, help="record directory or parent directory containing records")
    parser.add_argument("--review-file", default=EVENT_REVIEW_V2_FILE, help="review v2 file name in each record")
    parser.add_argument("--output-file", default=DEFAULT_REVIEW_V2_ANALYSIS_FILE, help="analysis file name written in each record")
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        default=DEFAULT_MAX_GROUP_FRAME_GAP,
        help="start a new group when adjacent events are more than this many frames apart",
    )
    args = parser.parse_args(argv)

    results = analyze_reviews_v2(
        Path(args.data_dir),
        review_filename=args.review_file,
        output_filename=args.output_file,
        max_frame_gap=args.max_frame_gap,
    )
    print(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
