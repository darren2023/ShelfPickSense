"""Upgrade legacy annotation.json files to shelves[] format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.annotation_upgrade import upgrade_annotations  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upgrade annotation.json to shelves[] format.")
    parser.add_argument("--data-dir", required=True, help="record directory or parent directory containing records")
    parser.add_argument("--annotation-file", default="annotation.json", help="annotation file name in each record")
    parser.add_argument("--output-dir", default="", help="output directory; default writes annotation_v2.json in each record")
    parser.add_argument("--in-place", action="store_true", help="overwrite annotation.json")
    parser.add_argument("--no-backup", action="store_true", help="do not create .bak when --in-place is used")
    parser.add_argument("--shelf-code", default="", help="override shelf_code for legacy single-shelf annotations")
    args = parser.parse_args(argv)

    results = upgrade_annotations(
        Path(args.data_dir),
        annotation_filename=args.annotation_file,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        in_place=args.in_place,
        backup=not args.no_backup,
        shelf_code=args.shelf_code,
    )
    print(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
