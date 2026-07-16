"""可打包的实时推理 HTTP 服务。"""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from loguru import logger

from analysis.annotation import load_annotation
from analysis.realtime import RealtimePickingPredictor


@dataclass
class InferenceServiceConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    model_dir: str = ""
    annotation_path: str = ""
    infer_width: float = 0.0
    infer_height: float = 0.0
    record_id: str = "realtime"
    max_request_bytes: int = 8 * 1024 * 1024

    @classmethod
    def from_file(cls, path: Path) -> "InferenceServiceConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"服务配置必须是 JSON object: {path}")
        config = cls(
            host=str(data.get("host") or cls.host),
            port=int(data.get("port") or cls.port),
            model_dir=str(data.get("model_dir") or ""),
            annotation_path=str(data.get("annotation_path") or ""),
            infer_width=float(data.get("infer_width") or 0.0),
            infer_height=float(data.get("infer_height") or 0.0),
            record_id=str(data.get("record_id") or "realtime"),
            max_request_bytes=int(data.get("max_request_bytes") or cls.max_request_bytes),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.model_dir:
            raise ValueError("配置缺少 model_dir")
        if not self.annotation_path:
            raise ValueError("配置缺少 annotation_path")
        if self.infer_width <= 0 or self.infer_height <= 0:
            raise ValueError("配置 infer_width/infer_height 必须大于 0")


class InferenceService:
    def __init__(self, config: InferenceServiceConfig) -> None:
        self.config = config
        self.predictor = RealtimePickingPredictor(
            model_dir=Path(config.model_dir),
            annotation_path=Path(config.annotation_path),
            infer_width=config.infer_width,
            infer_height=config.infer_height,
            record_id=config.record_id,
        )
        self._lock = threading.RLock()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "record_id": self.predictor.record.record_id,
            "infer_width": self.predictor.infer_width,
            "infer_height": self.predictor.infer_height,
            "box_count": len(self.predictor.box_tokens),
        }

    def predict(self, payload: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
        if isinstance(payload, list):
            first = next((item for item in payload if isinstance(item, dict)), {})
            skeleton_data = payload
            frame_idx = first.get("frame_idx") or first.get("source_frame_idx")
            timestamp_sec = first.get("timestamp_sec")
        elif isinstance(payload, dict):
            if payload.get("record_id"):
                self.predictor.set_record_id(str(payload["record_id"]))
            skeleton_data = payload.get("persons")
            if skeleton_data is None:
                skeleton_data = payload.get("skeletons")
            if skeleton_data is None:
                skeleton_data = payload
            frame_idx = payload.get("frame_idx")
            timestamp_sec = payload.get("timestamp_sec")
        else:
            raise ValueError("请求体必须是 JSON object 或 skeleton 行数组")
        with self._lock:
            pred = self.predictor.predict_frame(
                skeleton_data,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
            )
        return pred.to_dict()

    def update_annotation(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON object")
        annotation = payload.get("annotation")
        if annotation is None and payload.get("annotation_path"):
            annotation = load_annotation(Path(str(payload["annotation_path"])))
        if not isinstance(annotation, dict):
            raise ValueError("请求体缺少 annotation 或 annotation_path")
        infer_width = payload.get("infer_width")
        infer_height = payload.get("infer_height")
        with self._lock:
            self.predictor.set_annotation(
                annotation,
                infer_width=float(infer_width) if infer_width is not None else None,
                infer_height=float(infer_height) if infer_height is not None else None,
            )
            self.predictor.reset_history()
        return self.health()

    def reset(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        with self._lock:
            if isinstance(payload, dict) and payload.get("record_id"):
                self.predictor.set_record_id(str(payload["record_id"]))
            self.predictor.reset_history()
        return {"status": "ok", "record_id": self.predictor.record.record_id}


class _InferenceRequestHandler(BaseHTTPRequestHandler):
    server: "_InferenceHTTPServer"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, self.server.service.health())
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/predict":
                response = self.server.service.predict(payload)
            elif self.path == "/annotation":
                response = self.server.service.update_annotation(payload)
            elif self.path == "/reset":
                response = self.server.service.reset(payload)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._send_json(HTTPStatus.OK, response)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            logger.exception("推理服务请求处理失败: path={}", self.path)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("{} - {}", self.address_string(), fmt % args)

    def _read_json(self) -> dict[str, Any] | list[dict[str, Any]]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > self.server.service.config.max_request_bytes:
            raise ValueError("请求体超过 max_request_bytes")
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, (dict, list)):
            raise ValueError("请求体必须是 JSON object 或 skeleton 行数组")
        if isinstance(data, list) and not all(isinstance(item, dict) for item in data):
            raise ValueError("skeleton 行数组的元素必须是 JSON object")
        return data

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _InferenceHTTPServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], service: InferenceService) -> None:
        super().__init__(address, _InferenceRequestHandler)
        self.service = service


def serve(config: InferenceServiceConfig) -> None:
    service = InferenceService(config)
    server = _InferenceHTTPServer((config.host, config.port), service)
    logger.info("实时推理服务启动: http://{}:{}", config.host, config.port)
    logger.info("接口: GET /health, POST /predict, POST /annotation, POST /reset")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inference-service", description="实时取货推理 HTTP 服务")
    parser.add_argument("--config", required=True, help="服务 JSON 配置路径")
    args = parser.parse_args(argv)
    config = InferenceServiceConfig.from_file(Path(args.config))
    serve(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
