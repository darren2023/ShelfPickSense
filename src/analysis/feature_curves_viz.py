"""Feature curve visualization from exported parquet feature files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


FRAME_META_COLUMNS = {
    "record_id",
    "frame_idx",
    "person_track_id",
    "shelf_code",
    "is_picking",
    "target_layout_shelf_side",
    "target_layout_layer_norm",
    "target_layout_column_norm",
}


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        out = float(value)
        if math.isfinite(out):
            return int(out) if isinstance(value, int) else out
        return None
    return str(value)


def _feature_names(df: pd.DataFrame) -> list[str]:
    names: list[str] = []
    for column in df.columns:
        if column in FRAME_META_COLUMNS:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            names.append(str(column))
    return names


def build_feature_curves_payload(features_path: Path) -> dict[str, Any]:
    """Load frame feature parquet and build a compact browser payload."""
    features_path = Path(features_path)
    df = pd.read_parquet(features_path)
    if "frame_idx" not in df.columns:
        raise ValueError(f"缺少 frame_idx 列: {features_path}")

    feature_names = _feature_names(df)
    keep_columns = [
        col
        for col in ("record_id", "frame_idx", "person_track_id", "shelf_code", "is_picking")
        if col in df.columns
    ] + feature_names
    view = df[keep_columns].copy()
    view["frame_idx"] = pd.to_numeric(view["frame_idx"], errors="coerce").fillna(0).astype(int)
    sort_columns = [col for col in ("frame_idx", "person_track_id", "shelf_code") if col in view.columns]
    view = view.sort_values(sort_columns, na_position="first")

    rows: list[dict[str, Any]] = []
    for row in view.to_dict(orient="records"):
        rows.append({str(key): _json_value(value) for key, value in row.items()})

    return {
        "source": str(features_path),
        "row_count": len(rows),
        "features": feature_names,
        "records": sorted({str(v) for v in df.get("record_id", pd.Series(dtype=str)).dropna().unique()}),
        "person_track_ids": sorted(
            {str(int(v)) for v in pd.to_numeric(df.get("person_track_id", pd.Series(dtype=float)), errors="coerce").dropna().unique()}
        ),
        "shelf_codes": sorted({str(v) for v in df.get("shelf_code", pd.Series(dtype=str)).dropna().unique() if str(v)}),
        "rows": rows,
    }


def render_feature_curves_html(payload: dict[str, Any], *, title: str = "Feature Curves") -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    safe_title = title.replace("<", "").replace(">", "")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --surface: #ffffff;
      --ink: #1d2733;
      --muted: #687385;
      --line: #d9e0e8;
      --accent: #0f766e;
      --warn: #b45309;
      --active: #e8f5f2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    .app {{
      display: grid;
      grid-template-columns: 320px 1fr;
      min-height: 100vh;
    }}
    aside {{
      border-right: 1px solid var(--line);
      background: var(--surface);
      padding: 18px;
      overflow: auto;
    }}
    main {{
      min-width: 0;
      padding: 18px;
      display: grid;
      grid-template-rows: auto minmax(420px, 1fr) auto;
      gap: 12px;
    }}
    h1 {{
      margin: 0 0 14px;
      font-size: 20px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin: 14px 0 6px;
    }}
    select, input[type="search"] {{
      width: 100%;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 0 10px;
      font: inherit;
      font-size: 13px;
    }}
    .feature-list {{
      margin-top: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      max-height: calc(100vh - 260px);
      overflow: auto;
      background: #fff;
    }}
    .feature-item {{
      display: grid;
      grid-template-columns: 22px 1fr;
      gap: 8px;
      align-items: center;
      padding: 8px 10px;
      border-bottom: 1px solid #eef2f6;
      font-size: 12px;
      line-height: 1.25;
      word-break: break-word;
    }}
    .feature-item:last-child {{ border-bottom: 0; }}
    .feature-item:hover {{ background: var(--active); }}
    .toolbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .summary {{
      color: var(--muted);
      font-size: 13px;
    }}
    .actions {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    button {{
      height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 0 10px;
      cursor: pointer;
    }}
    button:hover {{ border-color: var(--accent); color: var(--accent); }}
    .chart-wrap {{
      position: relative;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      min-height: 420px;
      overflow: hidden;
      cursor: grab;
      touch-action: none;
      user-select: none;
    }}
    .chart-wrap.dragging {{
      cursor: grabbing;
    }}
    canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    .legend {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
      min-height: 24px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .swatch {{
      width: 18px;
      height: 3px;
      border-radius: 2px;
      display: inline-block;
    }}
    .empty {{
      position: absolute;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-size: 14px;
      background: rgba(255,255,255,0.78);
    }}
    .point-info {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface);
      color: var(--muted);
      min-height: 34px;
      padding: 8px 10px;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .point-info strong {{
      color: var(--ink);
      font-weight: 650;
    }}
    @media (max-width: 820px) {{
      .app {{ grid-template-columns: 1fr; }}
      aside {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      main {{ grid-template-rows: auto 52vh auto; }}
      .feature-list {{ max-height: 220px; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>Feature Curves</h1>
      <label for="recordFilter">record_id</label>
      <select id="recordFilter"></select>
      <label for="personFilter">person_track_id</label>
      <select id="personFilter"></select>
      <label for="shelfFilter">shelf_code</label>
      <select id="shelfFilter"></select>
      <label for="featureSearch">features</label>
      <input id="featureSearch" type="search" placeholder="search feature name" />
      <div id="featureList" class="feature-list"></div>
    </aside>
    <main>
      <div class="toolbar">
        <div id="summary" class="summary"></div>
        <div class="actions">
          <button id="zoomIn" type="button">Zoom +</button>
          <button id="zoomOut" type="button">Zoom -</button>
          <button id="resetView" type="button">Reset</button>
          <button id="selectVisible" type="button">Select visible</button>
          <button id="clearFeatures" type="button">Clear</button>
        </div>
      </div>
      <div id="chartWrap" class="chart-wrap">
        <canvas id="chart"></canvas>
        <div id="empty" class="empty">No data to draw</div>
      </div>
      <div id="pointInfo" class="point-info">Click a curve point to inspect frame/value.</div>
      <div id="legend" class="legend"></div>
    </main>
  </div>
  <script>
    const DATA = {data_json};
    const colors = ["#0f766e", "#2563eb", "#b45309", "#be123c", "#7c3aed", "#15803d", "#9333ea", "#ca8a04"];
    const state = {{
      selected: new Set(DATA.features.slice(0, Math.min(5, DATA.features.length))),
      search: "",
      xRange: null,
      drag: null,
      lastFullX: null,
      plot: null,
      inspect: null,
      suppressClick: false
    }};
    const $ = (id) => document.getElementById(id);
    const recordFilter = $("recordFilter");
    const personFilter = $("personFilter");
    const shelfFilter = $("shelfFilter");
    const featureSearch = $("featureSearch");
    const featureList = $("featureList");
    const chartWrap = $("chartWrap");
    const chart = $("chart");
    const empty = $("empty");
    const summary = $("summary");
    const pointInfo = $("pointInfo");
    const legend = $("legend");

    function fillSelect(el, values, allLabel) {{
      el.innerHTML = "";
      const all = document.createElement("option");
      all.value = "";
      all.textContent = allLabel;
      el.appendChild(all);
      values.forEach((value) => {{
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = value;
        el.appendChild(opt);
      }});
    }}

    function filteredRows() {{
      const record = recordFilter.value;
      const person = personFilter.value;
      const shelf = shelfFilter.value;
      return DATA.rows.filter((row) => {{
        if (record && String(row.record_id ?? "") !== record) return false;
        if (person && String(row.person_track_id ?? "") !== person) return false;
        if (shelf && String(row.shelf_code ?? "") !== shelf) return false;
        return true;
      }});
    }}

    function visibleFeatures() {{
      const q = state.search.trim().toLowerCase();
      if (!q) return DATA.features;
      return DATA.features.filter((name) => name.toLowerCase().includes(q));
    }}

    function renderFeatureList() {{
      featureList.innerHTML = "";
      visibleFeatures().forEach((name) => {{
        const label = document.createElement("label");
        label.className = "feature-item";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = state.selected.has(name);
        input.addEventListener("change", () => {{
          if (input.checked) state.selected.add(name);
          else state.selected.delete(name);
          draw();
        }});
        const span = document.createElement("span");
        span.textContent = name;
        label.append(input, span);
        featureList.appendChild(label);
      }});
    }}

    function seriesFor(rows, feature) {{
      const points = [];
      rows.forEach((row) => {{
        const y = Number(row[feature]);
        const x = Number(row.frame_idx);
        if (Number.isFinite(x) && Number.isFinite(y)) points.push({{ x, y }});
      }});
      points.sort((a, b) => a.x - b.x);
      return points;
    }}

    function clampRange(range, full) {{
      if (!range || !full) return null;
      let [minX, maxX] = range;
      const span = Math.max(1, maxX - minX);
      const fullSpan = Math.max(1, full.maxX - full.minX);
      if (span >= fullSpan) return null;
      if (minX < full.minX) {{
        maxX += full.minX - minX;
        minX = full.minX;
      }}
      if (maxX > full.maxX) {{
        minX -= maxX - full.maxX;
        maxX = full.maxX;
      }}
      minX = Math.max(full.minX, minX);
      maxX = Math.min(full.maxX, maxX);
      return [minX, maxX];
    }}

    function zoomX(factor, anchorRatio = 0.5) {{
      if (!state.lastFullX) return;
      const current = state.xRange || [state.lastFullX.minX, state.lastFullX.maxX];
      const span = current[1] - current[0];
      const nextSpan = Math.max(1, span * factor);
      const anchor = current[0] + span * anchorRatio;
      const next = [anchor - nextSpan * anchorRatio, anchor + nextSpan * (1 - anchorRatio)];
      state.xRange = clampRange(next, state.lastFullX);
      draw();
    }}

    function panX(deltaRatio) {{
      if (!state.lastFullX || !state.xRange) return;
      const span = state.xRange[1] - state.xRange[0];
      const delta = span * deltaRatio;
      state.xRange = clampRange([state.xRange[0] + delta, state.xRange[1] + delta], state.lastFullX);
      draw();
    }}

    function formatValue(value) {{
      if (!Number.isFinite(value)) return "";
      const abs = Math.abs(value);
      if (abs >= 1000 || (abs > 0 && abs < 0.001)) return value.toExponential(4);
      return value.toFixed(6).replace(/0+$/, "").replace(/[.]$/, "");
    }}

    function updatePointInfo() {{
      if (!state.inspect) {{
        pointInfo.textContent = "Click a curve point to inspect frame/value.";
        return;
      }}
      pointInfo.textContent = [
        state.inspect.name,
        `frame_idx=${{state.inspect.x}}`,
        `value=${{formatValue(state.inspect.y)}}`
      ].join(" · ");
    }}

    function pointToScreen(point, plot) {{
      const x = plot.pad.left + ((point.x - plot.minX) / (plot.maxX - plot.minX)) * plot.width;
      const y = plot.pad.top + (1 - (point.y - plot.minY) / (plot.maxY - plot.minY)) * plot.height;
      return {{ x, y }};
    }}

    function drawInspectionMarker(ctx) {{
      const plot = state.plot;
      const hit = state.inspect;
      updatePointInfo();
      if (!plot || !hit) return;
      if (!plot.visibleSeries.some((item) => item.name === hit.name)) return;
      if (hit.x < plot.minX || hit.x > plot.maxX || hit.y < plot.minY || hit.y > plot.maxY) return;
      const screen = pointToScreen(hit, plot);
      ctx.save();
      ctx.strokeStyle = hit.color || "#111827";
      ctx.fillStyle = hit.color || "#111827";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(screen.x, plot.pad.top);
      ctx.lineTo(screen.x, plot.pad.top + plot.height);
      ctx.moveTo(plot.pad.left, screen.y);
      ctx.lineTo(plot.pad.left + plot.width, screen.y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(screen.x, screen.y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }}

    function inspectNearestPoint(event) {{
      if (state.suppressClick || !state.plot) {{
        state.suppressClick = false;
        return;
      }}
      const rect = chart.getBoundingClientRect();
      const mouse = {{ x: event.clientX - rect.left, y: event.clientY - rect.top }};
      let best = null;
      for (const item of state.plot.visibleSeries) {{
        for (const point of item.points) {{
          const screen = pointToScreen(point, state.plot);
          const dist = Math.hypot(screen.x - mouse.x, screen.y - mouse.y);
          if (!best || dist < best.dist) {{
            best = {{ dist, name: item.name, color: item.color, x: point.x, y: point.y }};
          }}
        }}
      }}
      if (!best || best.dist > 24) return;
      state.inspect = {{ name: best.name, color: best.color, x: best.x, y: best.y }};
      draw();
    }}

    function draw() {{
      const ctx = chart.getContext("2d");
      const rect = chart.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      chart.width = Math.max(1, Math.floor(rect.width * dpr));
      chart.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);

      const rows = filteredRows();
      const selected = [...state.selected];
      const series = selected.map((name, index) => ({{
        name,
        color: colors[index % colors.length],
        points: seriesFor(rows, name)
      }})).filter((item) => item.points.length);

      summary.textContent = `${{rows.length}} 行 · ${{selected.length}} 个特征`;
      legend.innerHTML = "";
      series.forEach((item) => {{
        const node = document.createElement("span");
        node.innerHTML = `<i class="swatch" style="background:${{item.color}}"></i>${{item.name}}`;
        legend.appendChild(node);
      }});
      empty.style.display = series.length ? "none" : "flex";
      if (!series.length) {{
        state.plot = null;
        updatePointInfo();
        return;
      }}

      const pad = {{ left: 58, right: 18, top: 22, bottom: 44 }};
      const width = rect.width - pad.left - pad.right;
      const height = rect.height - pad.top - pad.bottom;
      const xs = series.flatMap((s) => s.points.map((p) => p.x));
      let fullMinX = Math.min(...xs), fullMaxX = Math.max(...xs);
      if (fullMinX === fullMaxX) {{ fullMinX -= 1; fullMaxX += 1; }}
      state.lastFullX = {{ minX: fullMinX, maxX: fullMaxX }};
      state.xRange = clampRange(state.xRange, state.lastFullX);
      const activeMinX = state.xRange ? state.xRange[0] : fullMinX;
      const activeMaxX = state.xRange ? state.xRange[1] : fullMaxX;
      const visibleSeries = series.map((item) => ({{
        ...item,
        points: item.points.filter((point) => point.x >= activeMinX && point.x <= activeMaxX)
      }})).filter((item) => item.points.length);
      if (!visibleSeries.length) {{
        empty.style.display = "flex";
        state.plot = null;
        updatePointInfo();
        return;
      }}
      const ys = visibleSeries.flatMap((s) => s.points.map((p) => p.y));
      let minX = activeMinX, maxX = activeMaxX;
      let minY = Math.min(...ys), maxY = Math.max(...ys);
      if (minY === maxY) {{ minY -= 1; maxY += 1; }}
      const yPad = (maxY - minY) * 0.08;
      minY -= yPad;
      maxY += yPad;
      const sx = (x) => pad.left + ((x - minX) / (maxX - minX)) * width;
      const sy = (y) => pad.top + (1 - (y - minY) / (maxY - minY)) * height;
      state.plot = {{
        pad,
        width,
        height,
        minX,
        maxX,
        minY,
        maxY,
        visibleSeries
      }};

      ctx.strokeStyle = "#d9e0e8";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, pad.top + height);
      ctx.lineTo(pad.left + width, pad.top + height);
      ctx.stroke();

      ctx.fillStyle = "#687385";
      ctx.font = "12px Segoe UI, Arial";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let i = 0; i <= 4; i++) {{
        const y = minY + ((maxY - minY) * i) / 4;
        const py = sy(y);
        ctx.strokeStyle = "#edf1f5";
        ctx.beginPath();
        ctx.moveTo(pad.left, py);
        ctx.lineTo(pad.left + width, py);
        ctx.stroke();
        ctx.fillText(y.toFixed(3), pad.left - 8, py);
      }}
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let i = 0; i <= 5; i++) {{
        const x = minX + ((maxX - minX) * i) / 5;
        ctx.fillText(Math.round(x), sx(x), pad.top + height + 10);
      }}

      visibleSeries.forEach((item) => {{
        ctx.strokeStyle = item.color;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        item.points.forEach((point, idx) => {{
          const x = sx(point.x);
          const y = sy(point.y);
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }});
        ctx.stroke();
      }});
      drawInspectionMarker(ctx);
    }}

    fillSelect(recordFilter, DATA.records || [], "全部记录");
    fillSelect(personFilter, DATA.person_track_ids || [], "全部人员");
    fillSelect(shelfFilter, DATA.shelf_codes || [], "全部货架");
    renderFeatureList();
    draw();
    [recordFilter, personFilter, shelfFilter].forEach((el) => el.addEventListener("change", draw));
    featureSearch.addEventListener("input", () => {{
      state.search = featureSearch.value;
      renderFeatureList();
    }});
    $("selectVisible").addEventListener("click", () => {{
      visibleFeatures().forEach((name) => state.selected.add(name));
      renderFeatureList();
      draw();
    }});
    $("clearFeatures").addEventListener("click", () => {{
      state.selected.clear();
      renderFeatureList();
      draw();
    }});
    $("zoomIn").addEventListener("click", () => zoomX(0.5));
    $("zoomOut").addEventListener("click", () => zoomX(2.0));
    $("resetView").addEventListener("click", () => {{
      state.xRange = null;
      draw();
    }});
    chart.addEventListener("wheel", (event) => {{
      event.preventDefault();
      const rect = chart.getBoundingClientRect();
      const ratio = rect.width > 0 ? (event.clientX - rect.left) / rect.width : 0.5;
      zoomX(event.deltaY < 0 ? 0.8 : 1.25, Math.min(1, Math.max(0, ratio)));
    }}, {{ passive: false }});
    chart.addEventListener("pointerdown", (event) => {{
      if (!state.lastFullX) return;
      const current = state.xRange || [state.lastFullX.minX, state.lastFullX.maxX];
      state.xRange = current;
      state.drag = {{ x: event.clientX, y: event.clientY, range: [...current], moved: false }};
      chart.setPointerCapture(event.pointerId);
      chartWrap.classList.add("dragging");
    }});
    chart.addEventListener("pointermove", (event) => {{
      if (!state.drag || !state.lastFullX) return;
      const rect = chart.getBoundingClientRect();
      const span = state.drag.range[1] - state.drag.range[0];
      const deltaFrames = -((event.clientX - state.drag.x) / Math.max(1, rect.width)) * span;
      if (Math.hypot(event.clientX - state.drag.x, event.clientY - state.drag.y) > 3) state.drag.moved = true;
      state.xRange = clampRange([state.drag.range[0] + deltaFrames, state.drag.range[1] + deltaFrames], state.lastFullX);
      draw();
    }});
    chart.addEventListener("pointerup", (event) => {{
      state.suppressClick = Boolean(state.drag && state.drag.moved);
      state.drag = null;
      chart.releasePointerCapture(event.pointerId);
      chartWrap.classList.remove("dragging");
    }});
    chart.addEventListener("pointercancel", () => {{
      state.drag = null;
      chartWrap.classList.remove("dragging");
    }});
    chart.addEventListener("click", inspectNearestPoint);
    window.addEventListener("keydown", (event) => {{
      if (event.key === "+" || event.key === "=") zoomX(0.5);
      else if (event.key === "-") zoomX(2.0);
      else if (event.key === "0") {{ state.xRange = null; draw(); }}
      else if (event.key === "ArrowLeft") panX(-0.2);
      else if (event.key === "ArrowRight") panX(0.2);
    }});
    window.addEventListener("resize", draw);
  </script>
</body>
</html>
"""


def write_feature_curves_html(
    features_dir: Path,
    output_path: Path | None = None,
    *,
    frame_features_name: str = "frame_features.parquet",
) -> Path:
    features_dir = Path(features_dir)
    features_path = features_dir / frame_features_name
    if not features_path.is_file():
        raise FileNotFoundError(f"未找到 parquet 帧级特征文件: {features_path}")
    out = Path(output_path) if output_path else features_dir / "feature_curves.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build_feature_curves_payload(features_path)
    out.write_text(render_feature_curves_html(payload), encoding="utf-8")
    return out
