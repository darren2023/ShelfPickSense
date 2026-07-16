from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fixtures import make_fixture_record


def test_push_record_to_inference_service_sends_grouped_frame_rows(tmp_path: Path):
    from scripts.push_record_to_inference_service import push_record

    received: list[list[dict]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            received.append(payload)
            body = json.dumps(
                {
                    "record_id": "test",
                    "frame_idx": payload[0]["frame_idx"],
                    "is_picking": False,
                    "picking_prob": 0.0,
                    "predicted_box_tokens": [],
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _fmt, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        record_dir = make_fixture_record(tmp_path / "record_001")
        responses = push_record(
            record_dir=record_dir,
            url=f"http://127.0.0.1:{server.server_port}",
            max_frames=2,
            timeout=5.0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert [response["frame_idx"] for response in responses] == [1, 2]
    assert len(received) == 2
    assert all(isinstance(payload, list) for payload in received)
    assert received[0][0]["frame_idx"] == 1
    assert "kpt_0_x" in received[0][0]
