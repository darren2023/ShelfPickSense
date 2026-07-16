#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-shelf-pick-inference-service}"
DIST_PATH="${DIST_PATH:-dist}"
WORK_PATH="${WORK_PATH:-build/nuitka}"
INCLUDE_XGBOOST="${INCLUDE_XGBOOST:-0}"
INCLUDE_LIGHTGBM="${INCLUDE_LIGHTGBM:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p "$WORK_PATH"

args=(
  run --with nuitka --with zstandard python -m nuitka
  --standalone
  --onefile
  --assume-yes-for-downloads
  "--output-dir=$DIST_PATH"
  "--output-filename=$NAME"
  --remove-output
  --include-package=analysis
  --include-package=sklearn
  --include-package=joblib
  --nofollow-import-to=cv2
  --nofollow-import-to=xgboost.testing
  --nofollow-import-to=hypothesis
  --nofollow-import-to=pytest
)

if [[ "$INCLUDE_XGBOOST" == "1" ]]; then
  args+=(--include-package=xgboost)
fi
if [[ "$INCLUDE_LIGHTGBM" == "1" ]]; then
  args+=(--include-package=lightgbm)
fi

args+=("scripts/inference_service_entry.py")

uv "${args[@]}"

echo "Built: ${DIST_PATH}/${NAME}"
