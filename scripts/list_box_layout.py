"""读取 annotation.json，计算并输出各货框数值布局。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.cli import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main(["box-layout", *sys.argv[1:]]))
