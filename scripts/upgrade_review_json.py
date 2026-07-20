"""Upgrade legacy event_review.json files to schema 2 with picking person info."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.review_upgrade import upgrade_reviews  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upgrade event_review.json with token and picking person info.")
    parser.add_argument("--data-dir", required=True, help="record directory or parent directory containing records")
    parser.add_argument("--review-file", default="event_review.json", help="review file name in each record, default: event_review.json")
    parser.add_argument("--output-dir", default="", help="output directory for upgraded files; default writes <review-name>.v2.json in each record")
    parser.add_argument("--in-place", action="store_true", help="overwrite event_review.json")
    parser.add_argument("--no-backup", action="store_true", help="do not create .bak when --in-place is used")
    args = parser.parse_args(argv)

    results = upgrade_reviews(
        Path(args.data_dir),
        review_filename=args.review_file,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        in_place=args.in_place,
        backup=not args.no_backup,
    )
    print(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
