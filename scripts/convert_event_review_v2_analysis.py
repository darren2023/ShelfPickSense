"""Convert grouped event_review_v2_analysis.json files to event_review_v2.json."""

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
from analysis.review_v2_analysis import DEFAULT_REVIEW_V2_ANALYSIS_FILE  # noqa: E402
from analysis.review_v2_groups_convert import convert_reviews_v2_analysis  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert grouped event_review_v2_analysis.json annotations to minimal event_review_v2.json."
    )
    parser.add_argument("--data-dir", required=True, help="record directory or parent directory containing records")
    parser.add_argument("--input-file", default=DEFAULT_REVIEW_V2_ANALYSIS_FILE, help="grouped review file name in each record")
    parser.add_argument("--output-file", default=EVENT_REVIEW_V2_FILE, help="event_review_v2 file name written in each record")
    args = parser.parse_args(argv)

    results = convert_reviews_v2_analysis(
        Path(args.data_dir),
        input_filename=args.input_file,
        output_filename=args.output_file,
    )
    print(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
