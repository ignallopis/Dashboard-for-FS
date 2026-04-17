"""Video analysis helpers — onboard video synced with telemetry charts.

This module is responsible for everything the Video Analysis sub-tab needs but
that does NOT touch Streamlit:

* **Static-file plumbing** — `ensure_static_videos` symlinks every `videos/<x>.mp4`
  into `src/static/videos/<x>.mp4` so Streamlit's built-in static server can
  serve the file from `<host>/app/static/videos/<x>.mp4` (the only URL an
  `<iframe>` component can fetch from inside Streamlit).
* **Data preparation** — `build_video_payload` flattens a Polars DataFrame into
  a JSON-serialisable dict containing per-lap telemetry arrays (Throttle, Brake,
  Steering, VN_vx, distance, lap-relative time), a global GPS track and the lap
  boundaries the JS layer needs for sync.
* **Component HTML** — `build_video_component_html` returns a self-contained
  HTML+JS+Plotly bundle that runs inside `streamlit.components.v1.html`.  The
  component owns the video element, the four stacked telemetry charts and the
  mini circuit map; it sets up bidirectional sync (video time ↔ chart cursor
  and click-to-seek) and lap-aware chart swapping.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import polars as pl


VIDEO_EXT = ".mp4"
STATIC_REL_DIR = "static/videos"
URL_PREFIX = "app/static/videos"
DEFAULT_OFFSET_S = 0.0
DEFAULT_OFFSET_RANGE_S = 60.0
CHART_DOWNSAMPLE_HZ = 50  # plot 50 Hz instead of raw 100 Hz to keep payload light
GPS_OUTLINE_POINTS = 600  # static circuit outline density


# ── Static file plumbing ──────────────────────────────────────────────────────

def ensure_static_videos(repo_root: Path, script_dir: Path) -> dict[str, str]:
    """Symlink every `<repo_root>/videos/*.mp4` into `<script_dir>/static/videos/`.

    Streamlit serves static assets from a `static/` folder next to the main
    script.  The Video Analysis subtab loads videos from
    `<host>/app/static/videos/<name>.mp4`, so we materialise that directory
    structure here.

    Returns a `{csv_basename: relative_url}` mapping, where `csv_basename` is the
    video filename stem (matched against CSVs by name) and `relative_url` is the
    path to use as `<video src=...>` (resolved against `window.parent.origin`).
    """
    videos_dir = repo_root / "videos"
    static_dir = script_dir / STATIC_REL_DIR
    static_dir.mkdir(parents=True, exist_ok=True)

    available: dict[str, str] = {}
    if not videos_dir.is_dir():
        return available

    for video_path in sorted(videos_dir.glob(f"*{VIDEO_EXT}")):
        link_path = static_dir / video_path.name
        try:
            if link_path.is_symlink() or link_path.exists():
                if link_path.is_symlink() and Path(os.readlink(link_path)) == video_path.resolve():
                    pass  # already correct
                else:
                    link_path.unlink()
                    link_path.symlink_to(video_path.resolve())
            else:
                link_path.symlink_to(video_path.resolve())
        except OSError:
            # Symlink not supported (e.g. exotic filesystem) — skip silently.
            continue
        available[video_path.stem] = f"{URL_PREFIX}/{video_path.name}"
    return available


def video_url_for_csv(
    csv_filename: str,
    available_videos: dict[str, str],
) -> str | None:
    """Look up the video URL for a given CSV filename (matched by stem)."""
    stem = Path(csv_filename).stem
    return available_videos.get(stem)


# ── Data preparation ─────────────────────────────────────────────────────────

_REQUIRED_COLS = (
    "TimeStamp", "laps", "laptime",
    "Throttle", "Brake", "Steering", "VN_vx",
    "VN_latitude", "VN_longitude",
)


def _downsample_indices(n: int, target_hz: int, source_hz: int = 100) -> np.ndarray:
    """Return integer indices that downsample a 100 Hz array to `target_hz`."""
    if target_hz >= source_hz:
        return np.arange(n)
    step = max(1, int(round(source_hz / target_hz)))
    return np.arange(0, n, step)


def _round_floats(values: np.ndarray, decimals: int) -> list[float]:
    """Round + cast to plain Python floats (NaN-safe) for JSON serialisation."""
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), np.round(arr, decimals), np.nan)
    return [None if not np.isfinite(v) else float(v) for v in arr]


def build_video_payload(df: pl.DataFrame) -> dict:
    """Flatten *df* into the JSON payload the JS component consumes.

    The payload is intentionally split into *per-lap* arrays (one chart-window
    per lap as the user demanded) plus a *global* GPS track that the moving
    map dot reads continuously.

    Raises:
        KeyError: if any required column is missing — the caller (dashboard)
            should catch and render a friendly fallback.
    """
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for Video Analysis: {missing}")

    t_global_all = df["TimeStamp"].to_numpy().astype(float)
    laps_all     = df["laps"].to_numpy().astype(float)
    laptime_all  = df["laptime"].to_numpy().astype(float)
    thr_all      = df["Throttle"].to_numpy().astype(float)
    brk_all      = df["Brake"].to_numpy().astype(float)
    str_all      = np.rad2deg(df["Steering"].to_numpy().astype(float))
    vx_all       = df["VN_vx"].to_numpy().astype(float)
    lat_all      = df["VN_latitude"].to_numpy().astype(float)
    lon_all      = df["VN_longitude"].to_numpy().astype(float)

    # ── Per-lap chart arrays ──────────────────────────────────────────────
    lap_ids = sorted({int(v) for v in laps_all if np.isfinite(v) and v > 0})
    laps_payload: list[dict] = []
    for lap_id in lap_ids:
        mask = laps_all == float(lap_id)
        if not mask.any():
            continue
        t_lap_global = t_global_all[mask]
        if t_lap_global.size < 2:
            continue
        order = np.argsort(t_lap_global)
        t_lap_global = t_lap_global[order]
        thr_lap = thr_all[mask][order]
        brk_lap = brk_all[mask][order]
        str_lap = str_all[mask][order]
        vx_lap  = vx_all[mask][order]

        # Lap-relative time (s) and cumulative distance (m, integrated VN_vx).
        t_local = t_lap_global - t_lap_global[0]
        dt = np.diff(t_lap_global, prepend=t_lap_global[0])
        dt = np.where(dt > 0, dt, 0.0)
        # Use the trailing value of vx for the segment to avoid the leading 0.
        distance = np.cumsum(np.where(np.isfinite(vx_lap), vx_lap, 0.0) * dt)

        # Downsample to keep the JSON small without losing visible detail.
        idx = _downsample_indices(len(t_lap_global), CHART_DOWNSAMPLE_HZ)

        laptime_lap = float(np.nanmax(laptime_all[mask])) if mask.any() else float("nan")

        laps_payload.append({
            "lap_id":     int(lap_id),
            "t_start":    float(t_lap_global[0]),
            "t_end":      float(t_lap_global[-1]),
            "laptime_s":  laptime_lap if np.isfinite(laptime_lap) else None,
            "t_global":   _round_floats(t_lap_global[idx], 3),
            "t_lap":      _round_floats(t_local[idx], 3),
            "distance":   _round_floats(distance[idx], 2),
            "throttle":   _round_floats(thr_lap[idx], 2),
            "brake":      _round_floats(brk_lap[idx], 2),
            "steering":   _round_floats(str_lap[idx], 3),
            "vx":         _round_floats(vx_lap[idx], 3),
        })

    # ── Global GPS track for the moving dot ──────────────────────────────
    finite_gps = np.isfinite(lat_all) & np.isfinite(lon_all) & np.isfinite(t_global_all)
    t_gps  = t_global_all[finite_gps]
    lat_g  = lat_all[finite_gps]
    lon_g  = lon_all[finite_gps]
    order = np.argsort(t_gps)
    t_gps, lat_g, lon_g = t_gps[order], lat_g[order], lon_g[order]
    # Downsample the global GPS too — 25 Hz is plenty for a moving dot.
    gps_idx = _downsample_indices(len(t_gps), 25)

    # ── Static circuit outline (downsampled even further) ────────────────
    if len(lat_g) > GPS_OUTLINE_POINTS:
        outline_idx = np.linspace(0, len(lat_g) - 1, GPS_OUTLINE_POINTS).astype(int)
    else:
        outline_idx = np.arange(len(lat_g))

    return {
        "laps": laps_payload,
        "gps": {
            "t_global": _round_floats(t_gps[gps_idx], 3),
            "lat":      _round_floats(lat_g[gps_idx], 7),
            "lon":      _round_floats(lon_g[gps_idx], 7),
        },
        "circuit_outline": {
            "lat": _round_floats(lat_g[outline_idx], 7),
            "lon": _round_floats(lon_g[outline_idx], 7),
        },
    }


# ── Component HTML ───────────────────────────────────────────────────────────

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def build_video_component_html(
    *,
    component_id: str,
    video_url: str | None,
    payload: dict,
    initial_offset_s: float = DEFAULT_OFFSET_S,
    offset_range_s: float = DEFAULT_OFFSET_RANGE_S,
    height_px: int = 720,
    show_video: bool = True,
    layout: str = "horizontal",
) -> str:
    """Return the full HTML string for the synced video+telemetry component.

    `component_id` must be unique per call (one per CSV) so that two side-by-side
    components on the same page do not collide.

    `layout` is either ``"horizontal"`` (video on the left, charts on the right —
    used when only one CSV is loaded) or ``"vertical"`` (video on top, charts
    underneath — used when two CSVs are split half-and-half across the page).
    """
    payload_json = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    has_video = bool(video_url) and show_video
    safe_video_url = json.dumps(video_url) if has_video else "null"

    if not has_video:
        flex_dir, top_basis, bot_basis = "column", "0 0 auto", "1 1 auto"
        video_max_h = "0"
        chart_height_px = max(500, height_px - 240)
        map_height_px = 140
    elif layout == "vertical":
        flex_dir, top_basis, bot_basis = "column", "0 0 45%", "1 1 55%"
        video_max_h = "100%"
        chart_height_px = max(360, int(height_px * 0.38))
        map_height_px = 140
    else:
        flex_dir, top_basis, bot_basis = "row", "0 0 45%", "1 1 55%"
        video_max_h = "60%"
        chart_height_px = max(480, int(height_px * 0.58))
        map_height_px = 160

    media_html = (
        f'<video id="vid_{component_id}" controls preload="metadata"></video>'
        if has_video else ""
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="{_PLOTLY_CDN}"></script>
<style>
  html, body {{ margin:0; padding:0; background:#141417; color:#EBEBEB;
                font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }}
  #wrap_{component_id} {{ display:flex; flex-direction:{flex_dir}; gap:12px;
                          height:{height_px}px; }}
  .pane_left  {{ flex: {top_basis}; display:flex; flex-direction:column; gap:8px;
                 min-height:0; min-width:0; }}
  .pane_right {{ flex: {bot_basis}; display:flex; flex-direction:column; gap:6px;
                 min-width:0; min-height:0; }}
  video {{ width:100%; max-height:{video_max_h}; background:#000;
           border-radius:6px; }}
  .controls {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center;
               font-size:12px; padding:6px 4px; background:#1c1c20;
               border-radius:6px; }}
  .controls label {{ display:inline-flex; align-items:center; gap:4px;
                     color:#bbb; }}
  .controls input[type=range] {{ width:160px; }}
  .lap_info {{ font-size:12px; color:#aaa; padding:4px; }}
  .lap_info b {{ color:#EBEBEB; }}
  .charts_box {{ flex:1 1 auto; min-height:0; }}
  .map_box {{ flex:0 0 {map_height_px}px; min-height:{map_height_px}px; }}
</style>
</head>
<body>
<div id="wrap_{component_id}">
  <div class="pane_left">
    {media_html}
    <div class="controls">
      <label>X-axis
        <select id="xmode_{component_id}">
          <option value="distance">Distance [m]</option>
          <option value="time">Lap time [s]</option>
          <option value="video">Video time [s]</option>
        </select>
      </label>
      <label>Offset [s]
        <input id="offset_{component_id}" type="range"
               min="{-offset_range_s:.2f}" max="{offset_range_s:.2f}"
               step="0.05" value="{initial_offset_s:.2f}">
        <span id="offset_val_{component_id}">{initial_offset_s:+.2f}</span>
      </label>
      <span class="lap_info" id="lap_info_{component_id}">Lap —</span>
    </div>
  </div>
  <div class="pane_right">
    <div class="charts_box" id="charts_{component_id}"></div>
    <div class="map_box"   id="map_{component_id}"></div>
  </div>
</div>

<script>
(function() {{
  const PAYLOAD = {payload_json};
  const VIDEO_URL_REL = {safe_video_url};
  const HAS_VIDEO = {str(has_video).lower()};
  const CID = "{component_id}";

  const SIGNALS = [
    {{ key:"throttle", label:"Throttle [%]",  color:"#73D973", range:[0, 105] }},
    {{ key:"brake",    label:"Brake [%]",     color:"#F27070", range:[0, 105] }},
    {{ key:"steering", label:"Steering [deg]", color:"#F2C94C", range:null }},
    {{ key:"vx",       label:"VN_vx [m/s]",   color:"#4DB3F2", range:null }},
  ];

  let offset = parseFloat(document.getElementById("offset_" + CID).value);
  let xmode  = document.getElementById("xmode_" + CID).value;
  let currentLapIdx = -1;

  // ── Resolve video URL relative to parent origin ─────────────────────
  if (HAS_VIDEO) {{
    const vid = document.getElementById("vid_" + CID);
    let origin = "";
    try {{ origin = window.parent.location.origin; }} catch (e) {{ origin = ""; }}
    vid.src = (origin ? origin + "/" : "") + VIDEO_URL_REL;
  }}

  // ── Helpers ─────────────────────────────────────────────────────────
  function cleanArray(arr) {{
    return arr.map(v => (v === null ? NaN : v));
  }}

  function lapForTime(tg) {{
    for (let i = 0; i < PAYLOAD.laps.length; i++) {{
      const L = PAYLOAD.laps[i];
      if (tg >= L.t_start && tg <= L.t_end) return i;
    }}
    if (PAYLOAD.laps.length === 0) return -1;
    if (tg < PAYLOAD.laps[0].t_start) return 0;
    return PAYLOAD.laps.length - 1;
  }}

  function xArrayForLap(lap, mode) {{
    if (mode === "distance") return cleanArray(lap.distance);
    if (mode === "time")     return cleanArray(lap.t_lap);
    // mode === "video"
    return lap.t_global.map(tg => tg === null ? NaN : tg - offset);
  }}

  function xAxisTitle(mode) {{
    return mode === "distance" ? "Distance [m]"
         : mode === "time"     ? "Lap time [s]"
                               : "Video time [s]";
  }}

  function videoTimeFromXClick(lap, mode, xVal) {{
    // Map a click in the chart back to a video.currentTime.
    if (mode === "video") {{
      return xVal;
    }}
    const arr = (mode === "distance") ? lap.distance : lap.t_lap;
    let bestIdx = 0;
    let bestDiff = Infinity;
    for (let i = 0; i < arr.length; i++) {{
      const v = arr[i];
      if (v === null) continue;
      const d = Math.abs(v - xVal);
      if (d < bestDiff) {{ bestDiff = d; bestIdx = i; }}
    }}
    const tg = lap.t_global[bestIdx];
    return tg === null ? null : (tg - offset);
  }}

  // ── Build the four-row stacked chart ────────────────────────────────
  function buildCharts() {{
    if (PAYLOAD.laps.length === 0) {{
      document.getElementById("charts_" + CID).innerHTML =
        '<div class="no_video">No valid laps in this run.</div>';
      return;
    }}
    const lap = PAYLOAD.laps[0];
    const x = xArrayForLap(lap, xmode);
    const traces = SIGNALS.map((s, i) => ({{
      x: x,
      y: cleanArray(lap[s.key]),
      type: "scattergl",
      mode: "lines",
      line: {{ color: s.color, width: 1.4 }},
      name: s.label,
      xaxis: "x",
      yaxis: "y" + (i === 0 ? "" : (i + 1)),
      hoverinfo: "x+y",
    }}));

    const layout = {{
      paper_bgcolor: "#141417",
      plot_bgcolor:  "#141417",
      font: {{ color: "#EBEBEB", size: 11 }},
      height: {chart_height_px},
      margin: {{ l: 82, r: 18, t: 8, b: 42 }},
      showlegend: false,
      grid: {{ rows: 4, columns: 1, pattern: "independent", ygap: 0.075 }},
      xaxis:  {{
        title: {{ text: xAxisTitle(xmode), standoff: 10 }},
        gridcolor: "#2a2a2e", anchor: "y4", automargin: true,
      }},
      yaxis:  {{
        title: {{ text: SIGNALS[0].label, standoff: 10 }},
        gridcolor: "#2a2a2e", range: SIGNALS[0].range, automargin: true,
      }},
      yaxis2: {{
        title: {{ text: SIGNALS[1].label, standoff: 10 }},
        gridcolor: "#2a2a2e", range: SIGNALS[1].range, automargin: true,
      }},
      yaxis3: {{
        title: {{ text: SIGNALS[2].label, standoff: 10 }},
        gridcolor: "#2a2a2e", automargin: true,
      }},
      yaxis4: {{
        title: {{ text: SIGNALS[3].label, standoff: 10 }},
        gridcolor: "#2a2a2e", automargin: true,
      }},
      shapes: cursorShapes(x[0]),
    }};
    Plotly.newPlot("charts_" + CID, traces, layout, {{
      displayModeBar: false, responsive: true,
    }}).then(gd => {{
      gd.on("plotly_click", evt => {{
        if (!HAS_VIDEO) return;
        if (!evt.points || !evt.points.length) return;
        const lapNow = PAYLOAD.laps[currentLapIdx >= 0 ? currentLapIdx : 0];
        const t = videoTimeFromXClick(lapNow, xmode, evt.points[0].x);
        if (t !== null && isFinite(t)) {{
          const vid = document.getElementById("vid_" + CID);
          vid.currentTime = Math.max(0, t);
        }}
      }});
    }});
    currentLapIdx = 0;
    updateLapInfo();
  }}

  function cursorShapes(x0) {{
    return [1, 2, 3, 4].map(i => ({{
      type: "line",
      xref: "x",
      yref: "y" + (i === 1 ? "" : i) + " domain",
      x0: x0, x1: x0, y0: 0, y1: 1,
      line: {{ color: "#FFFFFF", width: 1, dash: "dot" }},
    }}));
  }}

  function setLap(lapIdx) {{
    if (lapIdx === currentLapIdx) return;
    if (lapIdx < 0 || lapIdx >= PAYLOAD.laps.length) return;
    currentLapIdx = lapIdx;
    const lap = PAYLOAD.laps[lapIdx];
    const x = xArrayForLap(lap, xmode);
    Plotly.restyle("charts_" + CID, {{
      x: SIGNALS.map(() => x),
      y: SIGNALS.map(s => cleanArray(lap[s.key])),
    }});
    Plotly.relayout("charts_" + CID, {{
      "xaxis.title.text": xAxisTitle(xmode),
    }});
    updateLapInfo();
  }}

  function refreshXAxis() {{
    const lap = PAYLOAD.laps[currentLapIdx >= 0 ? currentLapIdx : 0];
    if (!lap) return;
    const x = xArrayForLap(lap, xmode);
    Plotly.restyle("charts_" + CID, {{
      x: SIGNALS.map(() => x),
    }});
    Plotly.relayout("charts_" + CID, {{
      "xaxis.title.text": xAxisTitle(xmode),
    }});
  }}

  function updateLapInfo() {{
    const lap = PAYLOAD.laps[currentLapIdx];
    if (!lap) return;
    const lt = (lap.laptime_s === null) ? "—" : lap.laptime_s.toFixed(2) + " s";
    document.getElementById("lap_info_" + CID).innerHTML =
      "Lap <b>" + lap.lap_id + "</b> — laptime " + lt;
  }}

  function moveCursorTo(xVal) {{
    if (xVal === null || !isFinite(xVal)) return;
    const shapes = cursorShapes(xVal);
    Plotly.relayout("charts_" + CID, {{ shapes: shapes }});
  }}

  function cursorXForTelemetryTime(lap, tg) {{
    if (xmode === "video") return tg - offset;
    if (xmode === "time")  return tg - lap.t_start;
    // distance — bisect in t_global, return interpolated distance
    const tArr = lap.t_global, dArr = lap.distance;
    let lo = 0, hi = tArr.length - 1;
    if (tg <= tArr[0]) return dArr[0];
    if (tg >= tArr[hi]) return dArr[hi];
    while (hi - lo > 1) {{
      const mid = (lo + hi) >> 1;
      if (tArr[mid] <= tg) lo = mid; else hi = mid;
    }}
    const t0 = tArr[lo], t1 = tArr[hi];
    const d0 = dArr[lo], d1 = dArr[hi];
    if (t0 === null || t1 === null || d0 === null || d1 === null) return d0;
    if (t1 === t0) return d0;
    const f = (tg - t0) / (t1 - t0);
    return d0 + f * (d1 - d0);
  }}

  // ── Build map ───────────────────────────────────────────────────────
  function buildMap() {{
    const outline = PAYLOAD.circuit_outline;
    if (!outline.lat.length) return;
    const traceOutline = {{
      x: outline.lon, y: outline.lat,
      type: "scattergl", mode: "lines",
      line: {{ color: "#555", width: 2 }},
      hoverinfo: "skip", name: "track",
    }};
    const traceDot = {{
      x: [outline.lon[0]], y: [outline.lat[0]],
      type: "scattergl", mode: "markers",
      marker: {{ size: 12, color: "#F28C40", line: {{ color: "#fff", width: 1 }} }},
      hoverinfo: "skip", name: "car",
    }};
    Plotly.newPlot("map_" + CID, [traceOutline, traceDot], {{
      paper_bgcolor: "#141417",
      plot_bgcolor:  "#141417",
      font: {{ color: "#EBEBEB" }},
      margin: {{ l: 4, r: 4, t: 4, b: 4 }},
      showlegend: false,
      xaxis: {{ visible: false, scaleanchor: "y" }},
      yaxis: {{ visible: false }},
    }}, {{ displayModeBar: false, responsive: true }});
  }}

  function moveDot(tg) {{
    const g = PAYLOAD.gps;
    const tArr = g.t_global;
    if (tArr.length === 0) return;
    let lo = 0, hi = tArr.length - 1;
    if (tg <= tArr[0]) {{ updateDot(g.lon[0], g.lat[0]); return; }}
    if (tg >= tArr[hi]) {{ updateDot(g.lon[hi], g.lat[hi]); return; }}
    while (hi - lo > 1) {{
      const mid = (lo + hi) >> 1;
      if (tArr[mid] <= tg) lo = mid; else hi = mid;
    }}
    const t0 = tArr[lo], t1 = tArr[hi];
    if (t0 === null || t1 === null || t1 === t0) {{
      updateDot(g.lon[lo], g.lat[lo]); return;
    }}
    const f = (tg - t0) / (t1 - t0);
    updateDot(g.lon[lo] + f * (g.lon[hi] - g.lon[lo]),
              g.lat[lo] + f * (g.lat[hi] - g.lat[lo]));
  }}

  function updateDot(lon, lat) {{
    Plotly.restyle("map_" + CID, {{ x: [[lon]], y: [[lat]] }}, [1]);
  }}

  // ── Sync loop driven by video.timeupdate ────────────────────────────
  function onVideoTime(currentTime) {{
    const tg = currentTime + offset;
    const lapIdx = lapForTime(tg);
    if (lapIdx !== currentLapIdx) setLap(lapIdx);
    const lap = PAYLOAD.laps[currentLapIdx];
    if (!lap) return;
    const xv = cursorXForTelemetryTime(lap, tg);
    moveCursorTo(xv);
    moveDot(tg);
  }}

  // ── Wire UI events ──────────────────────────────────────────────────
  buildCharts();
  buildMap();

  document.getElementById("xmode_" + CID).addEventListener("change", e => {{
    xmode = e.target.value;
    refreshXAxis();
  }});
  const offsetInput = document.getElementById("offset_" + CID);
  offsetInput.addEventListener("input", e => {{
    offset = parseFloat(e.target.value);
    document.getElementById("offset_val_" + CID).textContent =
      (offset >= 0 ? "+" : "") + offset.toFixed(2);
    if (HAS_VIDEO) {{
      onVideoTime(document.getElementById("vid_" + CID).currentTime);
    }}
  }});

  if (HAS_VIDEO) {{
    const vid = document.getElementById("vid_" + CID);
    vid.addEventListener("timeupdate", () => onVideoTime(vid.currentTime));
    vid.addEventListener("seeked",     () => onVideoTime(vid.currentTime));
    vid.addEventListener("loadedmetadata", () => onVideoTime(vid.currentTime));
  }}
}})();
</script>
</body>
</html>
"""
