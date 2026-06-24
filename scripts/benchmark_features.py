"""按多组特征配置批量运行 benchmark。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.cli import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main(["benchmark-features", *sys.argv[1:]]))
