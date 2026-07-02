"""前几帧骨架 + 货框标注可视化（生成交互 HTML）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analysis.annotation import build_box_index
from analysis.constants import (
    COCO17_KEYPOINT_NAMES,
    LEFT_ELBOW_IDX,
    LEFT_WRIST_IDX,
    RIGHT_ELBOW_IDX,
    RIGHT_WRIST_IDX,
)
from analysis.features.rule_engine import (
    RuleEngineParams,
    collect_rule_hand_points,
    nearest_box_token_for_point,
    signed_polygon_distance,
)
from analysis.records import RecordData, load_record

SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

_RULE_PARAMS = RuleEngineParams()


def _person_to_dict(person: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "person_id": person.get("person_id", 0),
        "keypoints": person.get("keypoints") or [],
    }
    if person.get("bbox"):
        out["bbox"] = person["bbox"]
    hand_points = collect_rule_hand_points(person, _RULE_PARAMS)
    out["hand_points"] = [{"x": x, "y": y, "kind": kind} for x, y, kind in hand_points]
    return out


def _load_predictions(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    preds: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        preds[int(row["frame_idx"])] = row
    return preds


def build_viz_payload(
    record: RecordData,
    *,
    max_frames: int = 10,
    start_frame: int = 1,
    predictions: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """构建可视化 JSON 数据。"""
    box_index = build_box_index(
        record.annotation,
        infer_w=record.infer_width,
        infer_h=record.infer_height,
    )
    ann_size = record.annotation.get("annotation_size")
    if not isinstance(ann_size, dict):
        ann_size = {}

    boxes = [
        {
            "token": token,
            "polygon": [[x, y] for x, y in box.polygon],
        }
        for token, box in sorted(box_index.items())
    ]

    frames_out: list[dict[str, Any]] = []
    all_frames = record.frames()
    selected = [f for f in all_frames if f.frame_idx >= start_frame][: max(1, max_frames)]

    for frame in selected:
        fi = frame.frame_idx
        frame_data: dict[str, Any] = {
            "frame_idx": fi,
            "timestamp_sec": frame.timestamp_sec,
            "persons": [_person_to_dict(p) for p in frame.persons],
        }

        pred = (predictions or {}).get(fi)
        if pred:
            frame_data["prediction"] = {
                "is_picking": pred.get("is_picking"),
                "rule_collisions": pred.get("rule_collisions") or [],
                "rule_alarm_collisions": pred.get("rule_alarm_collisions") or [],
            }

        # 手腕到碰撞货框的距离（便于调试 margin）
        distances: list[dict[str, Any]] = []
        for person in frame.persons:
            for x, y, kind in collect_rule_hand_points(person, _RULE_PARAMS):
                token = nearest_box_token_for_point(x, y, _RULE_PARAMS.boundary_margin_min_px, box_index)
                if token:
                    signed = signed_polygon_distance(x, y, box_index[token].polygon)
                    distances.append(
                        {
                            "kind": kind,
                            "x": x,
                            "y": y,
                            "box": token,
                            "signed_dist": round(signed, 2),
                            "outside_px": round(max(0.0, -signed), 2),
                        }
                    )
        if distances:
            frame_data["margin_hits"] = distances

        frames_out.append(frame_data)

    return {
        "record_id": record.record_id,
        "infer_width": record.infer_width,
        "infer_height": record.infer_height,
        "annotation_size": ann_size,
        "keypoint_names": list(COCO17_KEYPOINT_NAMES),
        "skeleton_edges": [list(e) for e in SKELETON_EDGES],
        "rule_params": {
            "boundary_margin_min_px": _RULE_PARAMS.boundary_margin_min_px,
            "boundary_margin_ratio": _RULE_PARAMS.boundary_margin_ratio,
            "forearm_extend_ratio": _RULE_PARAMS.forearm_extend_ratio,
            "min_consecutive_frames": _RULE_PARAMS.min_consecutive_frames,
        },
        "boxes": boxes,
        "frames": frames_out,
    }


def render_viz_html(payload: dict[str, Any], *, title: str | None = None) -> str:
    """渲染自包含 HTML 页面。"""
    data_json = json.dumps(payload, ensure_ascii=False)
    page_title = title or f"Frame Viz — {payload.get('record_id', 'record')}"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{page_title}</title>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #1a2332;
      --border: #2d3a4d;
      --text: #e7edf4;
      --muted: #8b9cb3;
      --accent: #5b9fd4;
      --box: rgba(91, 159, 212, 0.15);
      --box-stroke: #5b9fd4;
      --collision: rgba(255, 107, 107, 0.35);
      --collision-stroke: #ff6b6b;
      --alarm: rgba(255, 193, 7, 0.45);
      --skeleton: #7ee787;
      --wrist: #ff7b72;
      --forearm: #ffa657;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }}
    header {{
      padding: 12px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: center;
      background: var(--panel);
    }}
    header h1 {{ margin: 0; font-size: 1.1rem; font-weight: 600; }}
    .meta {{ color: var(--muted); font-size: 0.85rem; }}
    main {{
      display: grid;
      grid-template-columns: 280px 1fr;
      gap: 0;
      min-height: calc(100vh - 56px);
    }}
    aside {{
      border-right: 1px solid var(--border);
      padding: 16px;
      background: var(--panel);
      overflow-y: auto;
    }}
    .section {{ margin-bottom: 20px; }}
    .section h2 {{
      margin: 0 0 10px;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }}
    input[type="range"] {{ width: 100%; }}
    .frame-nav {{
      display: flex;
      gap: 8px;
      margin-top: 8px;
    }}
    button {{
      background: var(--border);
      color: var(--text);
      border: none;
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.85rem;
    }}
    button:hover {{ background: #3d4f66; }}
    .info dl {{
      margin: 0;
      font-size: 0.85rem;
    }}
    .info dt {{ color: var(--muted); margin-top: 8px; }}
    .info dd {{ margin: 2px 0 0; word-break: break-all; }}
    .legend {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 0.8rem;
    }}
    .legend span {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 14px;
      height: 14px;
      border-radius: 3px;
      flex-shrink: 0;
    }}
    .canvas-wrap {{
      padding: 20px;
      overflow: auto;
      display: flex;
      justify-content: center;
      align-items: flex-start;
    }}
    canvas {{
      background: #080c10;
      border: 1px solid var(--border);
      border-radius: 8px;
      max-width: 100%;
      height: auto;
    }}
    .tag {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      margin-right: 4px;
      margin-bottom: 4px;
    }}
    .tag-collision {{ background: rgba(255,107,107,0.25); color: #ff9a9a; }}
    .tag-alarm {{ background: rgba(255,193,7,0.25); color: #ffd666; }}
    .tag-ok {{ background: rgba(126,231,135,0.15); color: #7ee787; }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-right: none; border-bottom: 1px solid var(--border); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1 id="title">{page_title}</h1>
    <span class="meta" id="header-meta"></span>
  </header>
  <main>
    <aside>
      <div class="section">
        <h2>帧导航</h2>
        <label>帧 <strong id="frame-label">1</strong> / <span id="frame-total">1</span></label>
        <input type="range" id="frame-slider" min="0" max="0" value="0" />
        <div class="frame-nav">
          <button type="button" id="btn-prev">上一帧</button>
          <button type="button" id="btn-next">下一帧</button>
        </div>
      </div>
      <div class="section info" id="frame-info"></div>
      <div class="section">
        <h2>图例</h2>
        <div class="legend">
          <span><i class="swatch" style="background:var(--box-stroke)"></i>货框（缩放后）</span>
          <span><i class="swatch" style="background:var(--collision-stroke)"></i>瞬时碰撞货框</span>
          <span><i class="swatch" style="background:#ffc107"></i>报警货框</span>
          <span><i class="swatch" style="background:var(--skeleton)"></i>骨架连线</span>
          <span><i class="swatch" style="background:var(--wrist)"></i>手腕</span>
          <span><i class="swatch" style="background:var(--forearm)"></i>前臂外推点</span>
        </div>
      </div>
    </aside>
    <div class="canvas-wrap">
      <canvas id="canvas"></canvas>
    </div>
  </main>
  <script>
    const DATA = {data_json};

    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const slider = document.getElementById("frame-slider");
    const frameLabel = document.getElementById("frame-label");
    const frameTotal = document.getElementById("frame-total");
    const frameInfo = document.getElementById("frame-info");
    const headerMeta = document.getElementById("header-meta");

    const W = DATA.infer_width;
    const H = DATA.infer_height;
    const MAX_DISPLAY = 960;
    const scale = Math.min(1, MAX_DISPLAY / Math.max(W, H));
    canvas.width = W;
    canvas.height = H;
    canvas.style.width = Math.round(W * scale) + "px";
    canvas.style.height = Math.round(H * scale) + "px";

    headerMeta.textContent =
      `推理坐标 ${{W.toFixed(0)}}×${{H.toFixed(0)}} · 标注 ${{(DATA.annotation_size.width || "?") + "×" + (DATA.annotation_size.height || "?")}} · margin ${{DATA.rule_params.boundary_margin_min_px}}px`;

    let frameIdx = 0;
    slider.max = Math.max(0, DATA.frames.length - 1);
    frameTotal.textContent = DATA.frames.length;

    function drawPolygon(polygon, fill, stroke, lineWidth = 2) {{
      if (!polygon || polygon.length < 3) return;
      ctx.beginPath();
      ctx.moveTo(polygon[0][0], polygon[0][1]);
      for (let i = 1; i < polygon.length; i++) {{
        ctx.lineTo(polygon[i][0], polygon[i][1]);
      }}
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }}

    function drawFrame() {{
      const frame = DATA.frames[frameIdx];
      ctx.clearRect(0, 0, W, H);

      const collisions = new Set((frame.prediction && frame.prediction.rule_collisions) || []);
      const alarms = new Set((frame.prediction && frame.prediction.rule_alarm_collisions) || []);

      for (const box of DATA.boxes) {{
        let fill = "rgba(91,159,212,0.12)";
        let stroke = "#5b9fd4";
        let lw = 1.5;
        if (alarms.has(box.token)) {{
          fill = "rgba(255,193,7,0.35)";
          stroke = "#ffc107";
          lw = 2.5;
        }} else if (collisions.has(box.token)) {{
          fill = "rgba(255,107,107,0.28)";
          stroke = "#ff6b6b";
          lw = 2;
        }}
        drawPolygon(box.polygon, fill, stroke, lw);

        // 货框 token 标签
        const cx = box.polygon.reduce((s, p) => s + p[0], 0) / box.polygon.length;
        const cy = box.polygon.reduce((s, p) => s + p[1], 0) / box.polygon.length;
        ctx.font = "11px Segoe UI, sans-serif";
        ctx.fillStyle = "#8b9cb3";
        ctx.textAlign = "center";
        ctx.fillText(box.token.split(":").pop(), cx, cy);
      }}

      for (const person of frame.persons) {{
        if (person.bbox && person.bbox.length === 4) {{
          const [x1, y1, x2, y2] = person.bbox;
          ctx.strokeStyle = "rgba(139,156,179,0.5)";
          ctx.lineWidth = 1;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        }}

        const kpts = person.keypoints || [];
        for (const [a, b] of DATA.skeleton_edges) {{
          const ka = kpts[a], kb = kpts[b];
          if (!ka || !kb || ka[0] == null || kb[0] == null) continue;
          ctx.beginPath();
          ctx.moveTo(ka[0], ka[1]);
          ctx.lineTo(kb[0], kb[1]);
          ctx.strokeStyle = "#7ee787";
          ctx.lineWidth = 2;
          ctx.stroke();
        }}

        for (let i = 0; i < kpts.length; i++) {{
          const kp = kpts[i];
          if (!kp || kp[0] == null) continue;
          const isWrist = i === {LEFT_WRIST_IDX} || i === {RIGHT_WRIST_IDX};
          ctx.beginPath();
          ctx.arc(kp[0], kp[1], isWrist ? 5 : 3.5, 0, Math.PI * 2);
          ctx.fillStyle = isWrist ? "#ff7b72" : "#7ee787";
          ctx.fill();
          ctx.strokeStyle = "#0f1419";
          ctx.lineWidth = 1;
          ctx.stroke();
        }}

        for (const hp of person.hand_points || []) {{
          ctx.beginPath();
          ctx.arc(hp.x, hp.y, 4, 0, Math.PI * 2);
          ctx.fillStyle = hp.kind.includes("forearm") ? "#ffa657" : "#ff7b72";
          ctx.fill();
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1;
          ctx.stroke();
        }}
      }}

      frameLabel.textContent = frame.frame_idx;
      slider.value = frameIdx;
      renderInfo(frame);
    }}

    function renderInfo(frame) {{
      const pred = frame.prediction;
      let html = "<dl>";
      html += `<dt>帧索引</dt><dd>${{frame.frame_idx}} · t=${{frame.timestamp_sec?.toFixed(2) ?? "?"}}s</dd>`;
      html += `<dt>人数</dt><dd>${{frame.persons.length}}</dd>`;
      if (pred) {{
        html += `<dt>规则预测</dt><dd>`;
        html += pred.is_picking
          ? '<span class="tag tag-alarm">is_picking</span>'
          : '<span class="tag tag-ok">非取货</span>';
        for (const t of pred.rule_collisions || [])
          html += `<span class="tag tag-collision">${{t}}</span>`;
        for (const t of pred.rule_alarm_collisions || [])
          html += `<span class="tag tag-alarm">${{t}}</span>`;
        html += "</dd>";
      }}
      if (frame.margin_hits && frame.margin_hits.length) {{
        html += "<dt>Margin 命中点</dt><dd>";
        for (const h of frame.margin_hits) {{
          html += `<div>${{h.kind}} → ${{h.box}} · 外侧 ${{h.outside_px}}px</div>`;
        }}
        html += "</dd>";
      }}
      html += "</dl>";
      frameInfo.innerHTML = html;
    }}

    document.getElementById("btn-prev").onclick = () => {{
      frameIdx = Math.max(0, frameIdx - 1);
      drawFrame();
    }};
    document.getElementById("btn-next").onclick = () => {{
      frameIdx = Math.min(DATA.frames.length - 1, frameIdx + 1);
      drawFrame();
    }};
    slider.oninput = () => {{
      frameIdx = parseInt(slider.value, 10);
      drawFrame();
    }};

    drawFrame();
  </script>
</body>
</html>
"""


def write_frame_viz_html(
    record_dir: Path,
    output_path: Path,
    *,
    max_frames: int = 10,
    start_frame: int = 1,
    predictions_path: Path | None = None,
) -> Path:
    """从记录目录生成交互可视化 HTML。"""
    record = load_record(record_dir)
    predictions = _load_predictions(predictions_path)
    payload = build_viz_payload(
        record,
        max_frames=max_frames,
        start_frame=start_frame,
        predictions=predictions,
    )
    html = render_viz_html(payload)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
