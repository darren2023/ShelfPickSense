"""读取 record/skeleton.parquet，逐帧推送给实时推理服务。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.constants import SKELETON_FILE  # noqa: E402


def _json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _json_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: _json_safe_value(value) for key, value in row.items()} for row in rows]


def _post_json(url: str, payload: Any, *, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"请求失败: {exc}") from exc
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"服务响应不是 JSON object: {data}")
    return parsed


def _prediction_url(base_url: str) -> str:
    base = str(base_url).rstrip("/")
    if base.endswith("/predict"):
        return base
    return f"{base}/predict"


def push_record(
    *,
    record_dir: Path,
    url: str,
    start_frame: int = 0,
    max_frames: int = 0,
    timeout: float = 10.0,
    realtime: bool = False,
    fps: float = 25.0,
    output: Path | None = None,
) -> list[dict[str, Any]]:
    skeleton_path = Path(record_dir) / SKELETON_FILE
    if not skeleton_path.is_file():
        raise FileNotFoundError(f"未找到 {SKELETON_FILE}: {skeleton_path}")

    df = pd.read_parquet(skeleton_path)
    if start_frame > 0:
        df = df[df["frame_idx"] >= start_frame]

    predict_url = _prediction_url(url)
    frame_interval = 1.0 / fps if realtime and fps > 0 else 0.0
    responses: list[dict[str, Any]] = []
    output_file = None
    try:
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output_file = output.open("w", encoding="utf-8")

        sent = 0
        for frame_idx, group in df.groupby("frame_idx", sort=True):
            if max_frames > 0 and sent >= max_frames:
                break
            payload = _json_safe_rows(group.to_dict(orient="records"))
            response = _post_json(predict_url, payload, timeout=timeout)
            responses.append(response)
            line = json.dumps(response, ensure_ascii=False)
            print(line)
            if output_file is not None:
                output_file.write(line + "\n")
            sent += 1
            if frame_interval > 0:
                time.sleep(frame_interval)
    finally:
        if output_file is not None:
            output_file.close()
    return responses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="读取 record 并逐帧推送到实时推理服务")
    parser.add_argument("--record-dir", required=True, help="record 目录，需包含 skeleton.parquet")
    parser.add_argument("--url", default="http://127.0.0.1:8765", help="服务地址或 /predict URL")
    parser.add_argument("--start-frame", type=int, default=0, help="起始 frame_idx，0 表示从第一帧开始")
    parser.add_argument("--max-frames", type=int, default=0, help="最多推送帧数，0 表示全部")
    parser.add_argument("--timeout", type=float, default=10.0, help="单次 HTTP 请求超时秒数")
    parser.add_argument("--realtime", action="store_true", help="按 fps 间隔推送")
    parser.add_argument("--fps", type=float, default=25.0, help="realtime 模式推送帧率")
    parser.add_argument("--output", default="", help="可选 JSONL 响应输出路径")
    args = parser.parse_args(argv)

    push_record(
        record_dir=Path(args.record_dir),
        url=args.url,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        timeout=args.timeout,
        realtime=args.realtime,
        fps=args.fps,
        output=Path(args.output) if args.output else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
