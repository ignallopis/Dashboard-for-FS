"""Video analysis helpers — onboard video synced with telemetry charts."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import numpy as np
import polars as pl

import src.driver as drv
from utils import cols_to_numpy


VIDEO_EXT = ".mp4"
URL_PREFIX = "/videos"
DEFAULT_OFFSET_S = 0.0
DEFAULT_OFFSET_RANGE_S = 60.0
CHART_DOWNSAMPLE_HZ = 50
MAP_DOWNSAMPLE_HZ = 25
GPS_OUTLINE_POINTS = 600
MAP_PHASE_ORDER = ("ACCELERATING", "BRAKING", "COASTING", "PLAUSIBILITY")
_PHASE_CODE_BY_NAME = {name: idx for idx, name in enumerate(MAP_PHASE_ORDER)}
DEFAULT_SIGNAL_KEYS = ("throttle", "brake", "steering", "vx")
SIGNAL_SLOT_COUNT = 4
_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
_VIDEO_HTTP_SERVER: ThreadingHTTPServer | None = None
_VIDEO_HTTP_ROOT: Path | None = None


@dataclass
class VideoDiagnostics:
    warnings: list[str] = field(default_factory=list)


@dataclass
class VideoServerInfo:
    available_videos: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, VideoDiagnostics] = field(default_factory=dict)
    port: int | None = None
    error: str | None = None


_REQUIRED_COLS = (
    "TimeStamp",
    "laps",
    "laptime",
    "Throttle",
    "Brake",
    "Steering",
    "VN_vx",
    "VN_latitude",
    "VN_longitude",
)

_SIGNAL_SPECS = (
    {
        "key": "throttle",
        "column": "Throttle",
        "label": "Throttle [%]",
        "color": "#73D973",
        "range": [0.0, 105.0],
        "decimals": 1,
        "value_suffix": " %",
    },
    {
        "key": "brake",
        "column": "Brake",
        "label": "Brake [%]",
        "color": "#F27070",
        "range": [0.0, 105.0],
        "decimals": 1,
        "value_suffix": " %",
    },
    {
        "key": "steering",
        "column": "Steering",
        "label": "Steering [deg]",
        "color": "#F2C94C",
        "range": None,
        "decimals": 1,
        "value_suffix": " deg",
        "transform": "rad2deg",
    },
    {
        "key": "vx",
        "column": "VN_vx",
        "label": "VN_vx [m/s]",
        "color": "#4DB3F2",
        "range": None,
        "decimals": 2,
        "value_suffix": " m/s",
    },
    {
        "key": "vy",
        "column": "VN_vy",
        "label": "VN_vy [m/s]",
        "color": "#56CCF2",
        "range": None,
        "decimals": 2,
        "value_suffix": " m/s",
    },
    {
        "key": "ax",
        "columns": ("Filtering_VN_ax", "VN_ax"),
        "label": "Filtering_VN_ax [m/s^2]",
        "fallback_label": "VN_ax [m/s^2]",
        "color": "#F2994A",
        "range": None,
        "decimals": 2,
        "value_suffix": " m/s^2",
    },
    {
        "key": "ay",
        "columns": ("Filtering_VN_ay", "VN_ay"),
        "label": "Filtering_VN_ay [m/s^2]",
        "fallback_label": "VN_ay [m/s^2]",
        "color": "#EB5757",
        "range": None,
        "decimals": 2,
        "value_suffix": " m/s^2",
    },
    {
        "key": "gz",
        "column": "VN_gz",
        "label": "VN_gz",
        "color": "#BB6BD9",
        "range": None,
        "decimals": 3,
        "value_suffix": "",
    },
    {
        "key": "delta",
        "column": "delta",
        "label": "delta",
        "color": "#6FCF97",
        "range": None,
        "decimals": 3,
        "value_suffix": "",
    },
)


class _VideoRequestHandler(BaseHTTPRequestHandler):
    """Serve MP4 files with byte-range support for browser playback."""

    server_version = "CAT17xVideoHTTP/1.0"

    def do_HEAD(self) -> None:
        self._serve_video(send_body=False)

    def do_GET(self) -> None:
        self._serve_video(send_body=True)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_common_headers(
        self,
        *,
        status: int,
        file_size: int,
        content_length: int,
        content_range: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        self.send_header("Access-Control-Allow-Origin", "*")
        if content_range is not None:
            self.send_header("Content-Range", content_range)
        elif status == 416:
            self.send_header("Content-Range", f"bytes */{file_size}")
        self.end_headers()

    def _serve_video(self, *, send_body: bool) -> None:
        root = _VIDEO_HTTP_ROOT
        if root is None:
            self.send_error(503, "Video server is not initialised")
            return

        parsed_path = unquote(urlparse(self.path).path)
        if not parsed_path.startswith(f"{URL_PREFIX}/"):
            self.send_error(404, "Not found")
            return

        filename = Path(parsed_path).name
        if Path(filename).suffix.lower() != VIDEO_EXT:
            self.send_error(404, "Not found")
            return

        video_path = (root / filename).resolve()
        try:
            video_path.relative_to(root)
        except ValueError:
            self.send_error(403, "Forbidden")
            return
        if not video_path.is_file():
            self.send_error(404, "Not found")
            return

        file_size = video_path.stat().st_size
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1
        status = 200

        if range_header:
            unit, _, byte_range = range_header.partition("=")
            if unit.strip().lower() != "bytes":
                self.send_error(416, "Invalid range unit")
                return
            start_text, _, end_text = byte_range.partition("-")
            try:
                if start_text:
                    start = int(start_text)
                    end = int(end_text) if end_text else file_size - 1
                elif end_text:
                    suffix_len = int(end_text)
                    start = max(file_size - suffix_len, 0)
                else:
                    raise ValueError
            except ValueError:
                self._send_common_headers(
                    status=416,
                    file_size=file_size,
                    content_length=0,
                )
                return
            if start < 0 or start >= file_size or end < start:
                self._send_common_headers(
                    status=416,
                    file_size=file_size,
                    content_length=0,
                )
                return
            end = min(end, file_size - 1)
            status = 206

        content_length = end - start + 1
        content_range = f"bytes {start}-{end}/{file_size}" if status == 206 else None
        self._send_common_headers(
            status=status,
            file_size=file_size,
            content_length=content_length,
            content_range=content_range,
        )
        if not send_body:
            return

        with video_path.open("rb") as fh:
            fh.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = fh.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except BrokenPipeError:
                    break
                remaining -= len(chunk)


def _ensure_video_http_server(videos_dir: Path) -> int:
    """Start or reuse the local HTTP server that serves onboard videos."""
    global _VIDEO_HTTP_ROOT, _VIDEO_HTTP_SERVER

    videos_root = videos_dir.resolve()
    if _VIDEO_HTTP_SERVER is not None and _VIDEO_HTTP_ROOT == videos_root:
        return int(_VIDEO_HTTP_SERVER.server_port)

    _VIDEO_HTTP_ROOT = videos_root
    _VIDEO_HTTP_SERVER = ThreadingHTTPServer(("0.0.0.0", 0), _VideoRequestHandler)
    thread = threading.Thread(
        target=_VIDEO_HTTP_SERVER.serve_forever,
        name="cat17x-video-http",
        daemon=True,
    )
    thread.start()
    return int(_VIDEO_HTTP_SERVER.server_port)


def _available_video_urls(repo_root: Path) -> dict[str, str]:
    """Return video URLs for every `<repo_root>/videos/*.mp4` file."""
    videos_dir = repo_root / "videos"
    available: dict[str, str] = {}
    if not videos_dir.is_dir():
        return available

    for video_path in sorted(videos_dir.glob(f"*{VIDEO_EXT}")):
        available[video_path.stem] = f"{URL_PREFIX}/{quote(video_path.name)}"
    return available


def video_url_for_csv(csv_filename: str, available_videos: dict[str, str]) -> str | None:
    """Look up the video URL for a given CSV filename (matched by stem)."""
    return available_videos.get(Path(csv_filename).stem)


def ensure_video_server(repo_root: Path) -> VideoServerInfo:
    """Start the video server and return URLs for available onboard videos."""
    videos_dir = repo_root / "videos"
    try:
        available = _available_video_urls(repo_root)
        port = _ensure_video_http_server(videos_dir) if available else None
    except Exception as exc:
        return VideoServerInfo(error=str(exc))
    return VideoServerInfo(available_videos=available, port=port)


def video_diagnostics_for_csv(
    csv_filename: str,
    diagnostics: dict[str, VideoDiagnostics],
) -> VideoDiagnostics | None:
    """Return diagnostics for a CSV's paired video, or None if not found."""
    return diagnostics.get(Path(csv_filename).stem)


def _payload_lap_ids(laps_all: np.ndarray) -> list[int]:
    """Return complete lap IDs used by video analysis."""
    lap_ids = sorted({int(v) for v in laps_all if np.isfinite(v) and v > 0})
    if len(lap_ids) > 1:
        lap_ids = lap_ids[:-1]
    return lap_ids


def _gps_valid_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Reject invalid GPS samples, including the `(0, 0)` sentinel."""
    finite = np.isfinite(lat) & np.isfinite(lon)
    not_zero_zero = ~((np.abs(lat) < 1e-9) & (np.abs(lon) < 1e-9))
    return finite & not_zero_zero


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


def _signal_column_name(spec: dict[str, object], available_columns: set[str]) -> str | None:
    """Resolve the CSV column used by a configurable video-analysis signal."""
    column = spec.get("column")
    if isinstance(column, str):
        return column if column in available_columns else None

    candidate_columns = spec.get("columns")
    if isinstance(candidate_columns, tuple):
        for candidate in candidate_columns:
            if isinstance(candidate, str) and candidate in available_columns:
                return candidate
    return None


def _signal_values_from_df(df: pl.DataFrame, spec: dict[str, object]) -> np.ndarray:
    """Load one telemetry signal from *df* and apply lightweight transforms."""
    column_name = _signal_column_name(spec, set(df.columns))
    if column_name is None:
        raise KeyError(f"Signal column not found for spec {spec['key']}")
    values = cols_to_numpy(df, [column_name])[column_name]
    if spec.get("transform") == "rad2deg":
        values = np.rad2deg(values)
    return values


def _phase_stats_payload(
    lap_ids: list[int],
    laps_all: np.ndarray,
    laptime_all: np.ndarray,
    throttle_all: np.ndarray,
    brake_all: np.ndarray,
) -> dict[str, object]:
    """Return per-lap and average phase percentages for Video Analysis."""
    lap_rows: list[dict[str, float | int | None]] = []

    for lap_id in lap_ids:
        lap_mask = laps_all == float(lap_id)
        if not lap_mask.any():
            continue
        valid = lap_mask & np.isfinite(throttle_all) & np.isfinite(brake_all)
        if not valid.any():
            continue
        phase = drv._classify_phases(throttle_all[valid], brake_all[valid])
        n_valid = int(valid.sum())
        laptime_s = float(np.nanmax(laptime_all[lap_mask])) if lap_mask.any() else float("nan")
        lap_rows.append(
            {
                "lap_id": int(lap_id),
                "laptime_s": round(laptime_s, 2) if np.isfinite(laptime_s) else None,
                "throttle_pct": round(100.0 * float((phase == "ACCELERATING").sum()) / n_valid, 1),
                "braking_pct": round(100.0 * float((phase == "BRAKING").sum()) / n_valid, 1),
                "coasting_pct": round(100.0 * float((phase == "COASTING").sum()) / n_valid, 1),
                "plausibility_pct": round(100.0 * float((phase == "PLAUSIBILITY").sum()) / n_valid, 1),
            }
        )

    if not lap_rows:
        return {"average": None, "best": None, "laps": {}}

    def _mean(key: str, decimals: int) -> float | None:
        vals = [row[key] for row in lap_rows if row[key] is not None]
        if not vals:
            return None
        return round(float(np.mean(vals)), decimals)

    best_row = min(
        lap_rows,
        key=lambda row: (
            float("inf") if row["laptime_s"] is None else float(row["laptime_s"]),
            int(row["lap_id"]),
        ),
    )

    return {
        "average": {
            "lap_id": "AVG",
            "laptime_s": _mean("laptime_s", 2),
            "throttle_pct": _mean("throttle_pct", 1),
            "braking_pct": _mean("braking_pct", 1),
            "coasting_pct": _mean("coasting_pct", 1),
            "plausibility_pct": _mean("plausibility_pct", 1),
        },
        "best": best_row,
        "laps": {str(int(row["lap_id"])): row for row in lap_rows},
    }


def build_video_payload(
    df: pl.DataFrame,
    base_time_s: float | None = None,
) -> dict:
    """Flatten *df* into the JSON payload the JS component consumes."""
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for Video Analysis: {missing}")

    available_columns = set(df.columns)
    signal_specs = [
        spec
        for spec in _SIGNAL_SPECS
        if _signal_column_name(spec, available_columns) is not None
    ]
    signal_arrays = {
        str(spec["key"]): _signal_values_from_df(df, spec)
        for spec in signal_specs
    }

    cols = cols_to_numpy(df, ["TimeStamp", "laps", "laptime", "VN_latitude", "VN_longitude"])
    t_global_raw = cols["TimeStamp"]
    laps_all = cols["laps"]
    laptime_all = cols["laptime"]
    thr_all = signal_arrays["throttle"]
    brk_all = signal_arrays["brake"]
    vx_all = signal_arrays["vx"]
    lat_all = cols["VN_latitude"]
    lon_all = cols["VN_longitude"]

    signal_library = [
        {
            "key": str(spec["key"]),
            "label": str(
                spec["label"]
                if _signal_column_name(spec, available_columns) == spec.get("columns", (None,))[0]
                else spec.get("fallback_label", spec["label"])
            ),
            "color": str(spec["color"]),
            "range": spec["range"],
            "decimals": int(spec["decimals"]),
            "value_suffix": str(spec["value_suffix"]),
        }
        for spec in signal_specs
    ]
    default_signal_keys = [
        signal_key for signal_key in DEFAULT_SIGNAL_KEYS if signal_key in signal_arrays
    ]
    for spec in signal_specs:
        signal_key = str(spec["key"])
        if signal_key not in default_signal_keys:
            default_signal_keys.append(signal_key)
        if len(default_signal_keys) >= SIGNAL_SLOT_COUNT:
            break
    while len(default_signal_keys) < SIGNAL_SLOT_COUNT:
        default_signal_keys.append("")

    lap_ids = _payload_lap_ids(laps_all)
    phase_stats = _phase_stats_payload(lap_ids, laps_all, laptime_all, thr_all, brk_all)

    if base_time_s is None:
        if lap_ids:
            first_lap_mask = (laps_all == float(lap_ids[0])) & np.isfinite(t_global_raw)
            if first_lap_mask.any():
                base_time_s = float(t_global_raw[first_lap_mask][0])
            else:
                finite_t = t_global_raw[np.isfinite(t_global_raw)]
                base_time_s = float(finite_t.min()) if finite_t.size else 0.0
        else:
            finite_t = t_global_raw[np.isfinite(t_global_raw)]
            base_time_s = float(finite_t.min()) if finite_t.size else 0.0

    t_global_all = t_global_raw - float(base_time_s)

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
        vx_lap = vx_all[mask][order]

        t_local = t_lap_global - t_lap_global[0]
        dt = np.diff(t_lap_global, prepend=t_lap_global[0])
        dt = np.where(dt > 0, dt, 0.0)
        distance = np.cumsum(np.where(np.isfinite(vx_lap), vx_lap, 0.0) * dt)

        laptime_lap = float(np.nanmax(laptime_all[mask])) if mask.any() else float("nan")
        idx = _downsample_indices(len(t_lap_global), CHART_DOWNSAMPLE_HZ)

        map_mask = (
            mask
            & _gps_valid_mask(lat_all, lon_all)
            & np.isfinite(thr_all)
            & np.isfinite(brk_all)
            & np.isfinite(t_global_all)
        )
        if map_mask.any():
            t_map = t_global_all[map_mask]
            lat_map = lat_all[map_mask]
            lon_map = lon_all[map_mask]
            thr_map = thr_all[map_mask]
            brk_map = brk_all[map_mask]
            map_order = np.argsort(t_map)
            t_map = t_map[map_order]
            lat_map = lat_map[map_order]
            lon_map = lon_map[map_order]
            thr_map = thr_map[map_order]
            brk_map = brk_map[map_order]
            phase_names = drv._classify_phases(thr_map, brk_map)
            map_idx = _downsample_indices(len(t_map), MAP_DOWNSAMPLE_HZ)
            phase_codes = [
                int(_PHASE_CODE_BY_NAME[str(phase_name)])
                for phase_name in phase_names[map_idx]
            ]
        else:
            t_map = np.array([], dtype=float)
            lat_map = np.array([], dtype=float)
            lon_map = np.array([], dtype=float)
            map_idx = np.array([], dtype=int)
            phase_codes = []

        lap_payload = {
            "lap_id": int(lap_id),
            "t_start": float(t_lap_global[0]),
            "t_end": float(t_lap_global[-1]),
            "laptime_s": laptime_lap if np.isfinite(laptime_lap) else None,
            "t_global": _round_floats(t_lap_global[idx], 3),
            "t_lap": _round_floats(t_local[idx], 3),
            "distance": _round_floats(distance[idx], 2),
            "map_t_global": _round_floats(t_map[map_idx], 3),
            "map_lat": _round_floats(lat_map[map_idx], 7),
            "map_lon": _round_floats(lon_map[map_idx], 7),
            "map_phase": phase_codes,
        }
        for spec in signal_specs:
            signal_key = str(spec["key"])
            signal_values = signal_arrays[signal_key][mask][order]
            lap_payload[signal_key] = _round_floats(signal_values[idx], int(spec["decimals"]))
        laps_payload.append(lap_payload)

    valid_lap_values = np.asarray(lap_ids, dtype=float)
    finite_gps = (
        np.isfinite(t_global_all)
        & _gps_valid_mask(lat_all, lon_all)
        & np.isfinite(laps_all)
        & np.isin(laps_all, valid_lap_values)
    )
    t_gps = t_global_all[finite_gps]
    lat_g = lat_all[finite_gps]
    lon_g = lon_all[finite_gps]
    order = np.argsort(t_gps)
    t_gps, lat_g, lon_g = t_gps[order], lat_g[order], lon_g[order]

    if len(lat_g) > GPS_OUTLINE_POINTS:
        outline_idx = np.linspace(0, len(lat_g) - 1, GPS_OUTLINE_POINTS).astype(int)
    else:
        outline_idx = np.arange(len(lat_g))

    return {
        "base_time_s": float(base_time_s),
        "laps": laps_payload,
        "phase_stats": phase_stats,
        "signal_library": signal_library,
        "default_signal_keys": default_signal_keys,
        "circuit_outline": {
            "lat": _round_floats(lat_g[outline_idx], 7),
            "lon": _round_floats(lon_g[outline_idx], 7),
        },
    }


def build_video_component_html(
    *,
    component_id: str,
    video_url: str | None,
    video_server_port: int | None = None,
    payload: dict,
    compare_payload: dict | None = None,
    compare_lap_id: int | None = None,
    initial_offset_s: float = DEFAULT_OFFSET_S,
    offset_range_s: float = DEFAULT_OFFSET_RANGE_S,
    height_px: int = 720,
    show_video: bool = True,
    layout: str = "horizontal",
) -> str:
    """Return the full HTML string for the synced video+telemetry component."""
    _ = layout
    payload_json = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    compare_payload_json = (
        json.dumps(compare_payload, ensure_ascii=False, allow_nan=False)
        if compare_payload is not None
        else "null"
    )
    has_video = bool(video_url) and show_video
    safe_video_url = json.dumps(video_url) if has_video else "null"

    media_html = (
        f'<video id="vid_{component_id}" controls preload="metadata" playsinline></video>'
        if has_video else '<div class="media_placeholder">No onboard video loaded.</div>'
    )
    top_row_height_px = max(280, int(height_px * 0.42))
    chart_height_px = max(360, height_px - top_row_height_px - 86)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="{_PLOTLY_CDN}"></script>
<style>
  html, body {{ margin:0; padding:0; background:#141417; color:#EBEBEB;
                font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }}
  #wrap_{component_id} {{ display:flex; flex-direction:column; gap:10px;
                          height:{height_px}px; }}
  .top_row {{ display:grid; grid-template-columns:minmax(0, 1.7fr) minmax(220px, 0.56fr) minmax(0, 0.92fr);
              gap:12px; flex:0 0 {top_row_height_px}px; min-height:0; }}
  .media_pane, .stats_pane, .map_pane {{ display:flex; flex-direction:column; gap:8px;
                                         min-width:0; min-height:0; }}
  video, .media_placeholder {{ width:100%; height:100%; background:#000;
           border-radius:6px; border:1px solid rgba(255,255,255,0.08); }}
  .media_placeholder {{ display:flex; align-items:center; justify-content:center;
                        color:#8A8F98; font-size:14px; }}
  .video_box {{ flex:1 1 auto; min-height:0; }}
  .stats_pane {{ justify-content:flex-start; }}
  .map_box {{ flex:1 1 auto; min-height:0; border:1px solid rgba(255,255,255,0.08);
              border-radius:6px; overflow:hidden; }}
  .phase_stats_box {{ flex:1 1 auto; border:1px solid rgba(255,255,255,0.08); border-radius:8px;
                      overflow:auto; background:linear-gradient(180deg, #1A1D23 0%, #15181D 100%);
                      padding:8px 10px; }}
  .phase_stats_table {{ width:100%; border-collapse:separate; border-spacing:0; table-layout:fixed; font-size:12px; }}
  .phase_stats_table th, .phase_stats_table td {{
    padding:8px 10px; text-align:right; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  }}
  .phase_stats_table thead th {{
    background:#20242B; color:#AEB6C2; font-weight:600; font-size:11px;
    text-transform:uppercase; letter-spacing:0.04em; border-bottom:1px solid rgba(255,255,255,0.08);
  }}
  .phase_stats_table thead th:first-child {{ border-top-left-radius:6px; }}
  .phase_stats_table thead th:last-child {{ border-top-right-radius:6px; }}
  .phase_stats_table td:first-child, .phase_stats_table th:first-child {{ text-align:left; width:40%; }}
  .phase_stats_table th:not(:first-child), .phase_stats_table td:not(:first-child) {{ width:20%; }}
  .phase_stats_table tbody td {{
    color:#F3F5F7; font-variant-numeric: tabular-nums; font-size:13px;
    border-bottom:1px solid rgba(255,255,255,0.06); background:rgba(255,255,255,0.015);
  }}
  .phase_stats_table tbody tr:last-child td {{ border-bottom:none; }}
  .phase_stats_table th.active_col {{ background:#252A33; color:#F3F5F7; }}
  .phase_stats_table th.best_col {{ color:#EBD8FF; }}
  .phase_metric_cell {{ padding-left:12px; }}
  .phase_metric {{ display:flex; align-items:center; gap:8px; min-width:0; }}
  .phase_metric_dot {{ width:8px; height:8px; border-radius:999px; flex:0 0 auto;
                       box-shadow:0 0 0 2px rgba(255,255,255,0.06); }}
  .phase_metric_name {{ color:#F3F5F7; font-weight:500; overflow:hidden; text-overflow:ellipsis; }}
  .phase_stats_table td.best_value {{ color:#DDB8FF; font-weight:600; }}
  .phase_stats_table td.active_value {{ background:rgba(255,255,255,0.03); }}
  .phase_stats_table td.active_best_value {{ color:#DDB8FF; font-weight:600; background:rgba(255,255,255,0.03); }}
  .phase_stats_empty {{ padding:10px 12px; color:#8A8F98; font-size:12px; }}
  .map_toolbar {{ display:flex; align-items:center; justify-content:space-between; gap:10px; }}
  .map_caption {{ font-size:12px; color:#A7AFBA; padding:0 2px; }}
  .map_toolbar label {{ display:inline-flex; align-items:center; gap:4px; color:#bbb; font-size:12px; }}
  .stats_caption {{ font-size:12px; color:#A7AFBA; padding:0 2px; }}
  .controls {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center;
               font-size:12px; padding:6px 4px; background:#1c1c20; border-radius:6px; }}
  .controls label {{ display:inline-flex; align-items:center; gap:4px; color:#bbb; }}
  .controls select, .map_toolbar select {{
    background:#111318; color:#EBEBEB; border:1px solid rgba(255,255,255,0.12);
    border-radius:6px; padding:4px 6px; font-size:12px;
  }}
  .controls button {{
    background:#24262B; color:#EBEBEB; border:1px solid rgba(255,255,255,0.12);
    border-radius:6px; padding:6px 10px; cursor:pointer; font-size:12px;
  }}
  .controls button:hover {{ background:#2D3036; }}
  .controls input[type=range] {{ width:160px; }}
  .signal_slots {{ display:flex; flex-wrap:wrap; gap:8px; align-items:flex-start; }}
  .signal_select {{ display:inline-flex !important; flex-direction:column; align-items:flex-start !important; gap:4px !important; }}
  .signal_select span {{ color:#8A8F98; }}
  .lap_info {{ font-size:12px; color:#aaa; padding:4px; }}
  .lap_info b {{ color:#EBEBEB; }}
  .cursor_help {{ font-size:12px; color:#8A8F98; }}
  .video_status {{ font-size:12px; color:#C9CDD3; padding:0 4px 4px 4px; min-height:18px; }}
  .video_status.error {{ color:#F27070; }}
  .charts_box {{ flex:1 1 auto; min-height:0; border:1px solid rgba(255,255,255,0.08); border-radius:6px; overflow:hidden; }}
  .chart_placeholder {{ width:100%; height:100%; display:flex; align-items:center; justify-content:center; color:#8A8F98; font-size:14px; }}
  @media (max-width: 1300px) {{
    .top_row {{ grid-template-columns:minmax(0, 1fr); flex-basis:auto; }}
    .media_pane, .stats_pane, .map_pane {{ min-height:220px; }}
  }}
</style>
</head>
<body>
<div id="wrap_{component_id}">
  <div class="top_row">
    <div class="media_pane">
      <div class="video_box">
        {media_html}
      </div>
      <div class="video_status" id="video_status_{component_id}"></div>
    </div>
    <div class="stats_pane">
      <div class="stats_caption">Lap phase stats: AVG, best lap, and active lap.</div>
      <div class="phase_stats_box" id="phase_stats_{component_id}"></div>
    </div>
    <div class="map_pane">
      <div class="map_toolbar">
        <div class="map_caption">Current lap map: show either the original lap or the compared lap.</div>
        <label>Map
          <select id="map_source_{component_id}"></select>
        </label>
      </div>
      <div class="map_box" id="map_{component_id}"></div>
    </div>
  </div>
  <div class="controls">
    <label>Lap
      <select id="lap_select_{component_id}"></select>
    </label>
    <label>Offset [s]
      <input id="offset_{component_id}" type="range"
             min="{-offset_range_s:.2f}" max="{offset_range_s:.2f}"
             step="0.05" value="{initial_offset_s:.2f}">
      <span id="offset_val_{component_id}">{initial_offset_s:+.2f}</span>
    </label>
    <label>X-axis
      <select id="x_axis_mode_{component_id}">
        <option value="time">Time</option>
        <option value="distance">Distance</option>
      </select>
    </label>
    <div class="signal_slots" id="signal_slots_{component_id}"></div>
    <button id="reset_zoom_{component_id}" type="button">Reset zoom</button>
    <button id="fullscreen_{component_id}" type="button">Full screen</button>
    <span class="lap_info" id="lap_info_{component_id}">Lap —</span>
    <span class="cursor_help">Left-click or left-drag on the charts to scrub video, cursor, and GPS.</span>
  </div>
  <div class="charts_box" id="charts_{component_id}"></div>
</div>

<script>
(function() {{
  const PAYLOAD = {payload_json};
  const COMPARE_PAYLOAD = {compare_payload_json};
  const COMPARE_LAP_ID = {("null" if compare_lap_id is None else str(int(compare_lap_id)))};
  const COMPARE_COLOR = "#FFFFFF";
  const VIDEO_URL_REL = {safe_video_url};
  const VIDEO_SERVER_PORT = {("null" if video_server_port is None else str(int(video_server_port)))};
  const HAS_VIDEO = {str(has_video).lower()};
  const CID = "{component_id}";
  const HIDDEN_SIGNAL_KEY = "__hidden__";
  const SIGNAL_SLOT_COUNT = {SIGNAL_SLOT_COUNT};
  const DEFAULT_CHART_HEIGHT = {chart_height_px};

  const SIGNAL_LIBRARY = Array.isArray(PAYLOAD.signal_library)
    ? PAYLOAD.signal_library.map(signal => ({{
        key: signal.key,
        label: signal.label,
        color: signal.color || "#4DB3F2",
        range: Array.isArray(signal.range) ? signal.range : null,
        decimals: Number.isFinite(signal.decimals) ? signal.decimals : 2,
        value_suffix: signal.value_suffix || "",
      }}))
    : [];
  const SIGNAL_BY_KEY = Object.fromEntries(SIGNAL_LIBRARY.map(signal => [signal.key, signal]));
  const DEFAULT_SIGNAL_KEYS = Array.isArray(PAYLOAD.default_signal_keys) ? PAYLOAD.default_signal_keys : [];
  const PHASES = [
    {{ code:0, key:"ACCELERATING", label:"Accelerating", color:"{drv.MAP_PHASE_COLORS['ACCELERATING']}" }},
    {{ code:1, key:"BRAKING", label:"Braking", color:"{drv.MAP_PHASE_COLORS['BRAKING']}" }},
    {{ code:2, key:"COASTING", label:"Coasting", color:"{drv.MAP_PHASE_COLORS['COASTING']}" }},
    {{ code:3, key:"PLAUSIBILITY", label:"Plausibility", color:"{drv.MAP_PHASE_COLORS['PLAUSIBILITY']}" }},
  ];
  const PHASE_STATS_COLUMNS = [
    {{ key: "laptime_s", label: "LapTime [s]", color: "rgba(130, 60, 180, 0.92)", decimals: 2 }},
    {{ key: "throttle_pct", label: "Throttle [%]", color: "{drv.MAP_PHASE_COLORS['ACCELERATING']}", decimals: 1 }},
    {{ key: "braking_pct", label: "Braking [%]", color: "{drv.MAP_PHASE_COLORS['BRAKING']}", decimals: 1 }},
    {{ key: "coasting_pct", label: "Coasting [%]", color: "{drv.MAP_PHASE_COLORS['COASTING']}", decimals: 1 }},
    {{ key: "plausibility_pct", label: "Plausability [%]", color: "{drv.MAP_PHASE_COLORS['PLAUSIBILITY']}", textColor: "#111111", decimals: 1 }},
  ];
  const COMPARE_PHASE_COLORS = {{
    ACCELERATING: "#81C784",
    BRAKING: "#FF8A80",
    COASTING: "#B0BEC5",
    PLAUSIBILITY: "#FFD54F",
  }};
  const COMPARE_PHASE_STYLES = {{
    ACCELERATING: {{ dash: "solid", width: 4.8 }},
    BRAKING: {{ dash: "dash", width: 4.8 }},
    COASTING: {{ dash: "dot", width: 4.8 }},
    PLAUSIBILITY: {{ dash: "longdashdot", width: 4.8 }},
  }};
  const fullscreenIcon = {{
    width: 1000,
    height: 1000,
    path: "M60 360V60H360V160H160V360Z M640 60H940V360H840V160H640Z M160 640V840H360V940H60V640Z M840 640V840H640V940H940V640Z",
  }};

  let offset = parseFloat(document.getElementById("offset_" + CID).value);
  let currentLapIdx = -1;
  let currentTelemetryTime = null;
  let currentCursorX = null;
  let chartGd = null;
  let xAxisMode = "time";
  let mapSourceMode = "original";
  let slotSignalKeys = Array.from({{ length: SIGNAL_SLOT_COUNT }}, (_, idx) => {{
    const defaultKey = DEFAULT_SIGNAL_KEYS[idx];
    return SIGNAL_BY_KEY[defaultKey] ? defaultKey : HIDDEN_SIGNAL_KEY;
  }});
  if (!slotSignalKeys.some(signalKey => signalKey !== HIDDEN_SIGNAL_KEY)) {{
    slotSignalKeys = Array.from({{ length: SIGNAL_SLOT_COUNT }}, (_, idx) => {{
      return SIGNAL_LIBRARY[idx] ? SIGNAL_LIBRARY[idx].key : HIDDEN_SIGNAL_KEY;
    }});
  }}

  const statusEl = document.getElementById("video_status_" + CID);
  const lapSelectEl = document.getElementById("lap_select_" + CID);
  const mapSourceEl = document.getElementById("map_source_" + CID);
  const phaseStatsEl = document.getElementById("phase_stats_" + CID);
  const xAxisModeEl = document.getElementById("x_axis_mode_" + CID);
  const signalSlotsEl = document.getElementById("signal_slots_" + CID);
  const chartEl = document.getElementById("charts_" + CID);

  function setStatus(message, isError) {{
    if (!statusEl) return;
    statusEl.textContent = message || "";
    statusEl.classList.toggle("error", Boolean(isError));
  }}

  if (HAS_VIDEO) {{
    const vid = document.getElementById("vid_" + CID);
    let resolvedVideoUrl = VIDEO_URL_REL;
    if (VIDEO_SERVER_PORT !== null) {{
      let hostname = "localhost";
      try {{
        if (window.parent && window.parent.location && window.parent.location.hostname) {{
          hostname = window.parent.location.hostname;
        }}
      }} catch (e) {{}}
      resolvedVideoUrl = "http://" + hostname + ":" + VIDEO_SERVER_PORT + VIDEO_URL_REL;
    }} else {{
      try {{
        if (window.parent && window.parent.location) {{
          resolvedVideoUrl = new URL(VIDEO_URL_REL, window.parent.location.href).toString();
        }}
      }} catch (e) {{}}
    }}
    vid.src = resolvedVideoUrl;
    vid.load();
    setStatus("Loading video metadata...", false);
  }}

  function cleanArray(arr) {{
    return arr.map(v => (v === null ? NaN : v));
  }}

  function valueAt(arr, idx) {{
    const v = arr[idx];
    return (v === null || !isFinite(v)) ? NaN : v;
  }}

  function activeSignals() {{
    return slotSignalKeys
      .map((signalKey, slotIdx) => {{
        const signal = SIGNAL_BY_KEY[signalKey];
        if (!signal) return null;
        return Object.assign({{ slotIdx: slotIdx }}, signal);
      }})
      .filter(Boolean);
  }}

  function renderSignalSelectors() {{
    if (!signalSlotsEl) return;
    signalSlotsEl.innerHTML = "";
    for (let slotIdx = 0; slotIdx < SIGNAL_SLOT_COUNT; slotIdx++) {{
      const labelEl = document.createElement("label");
      labelEl.className = "signal_select";
      const titleEl = document.createElement("span");
      titleEl.textContent = "Plot " + (slotIdx + 1);
      const selectEl = document.createElement("select");
      selectEl.dataset.slot = String(slotIdx);
      const hiddenOption = document.createElement("option");
      hiddenOption.value = HIDDEN_SIGNAL_KEY;
      hiddenOption.textContent = "Hide";
      selectEl.appendChild(hiddenOption);
      SIGNAL_LIBRARY.forEach(signal => {{
        const optionEl = document.createElement("option");
        optionEl.value = signal.key;
        optionEl.textContent = signal.label;
        selectEl.appendChild(optionEl);
      }});
      selectEl.value = SIGNAL_BY_KEY[slotSignalKeys[slotIdx]] ? slotSignalKeys[slotIdx] : HIDDEN_SIGNAL_KEY;
      labelEl.appendChild(titleEl);
      labelEl.appendChild(selectEl);
      signalSlotsEl.appendChild(labelEl);
    }}
  }}

  function lapForTime(tg) {{
    for (let i = 0; i < PAYLOAD.laps.length; i++) {{
      const lap = PAYLOAD.laps[i];
      if (tg >= lap.t_start && tg <= lap.t_end) return i;
    }}
    if (PAYLOAD.laps.length === 0) return -1;
    if (tg < PAYLOAD.laps[0].t_start) return 0;
    return PAYLOAD.laps.length - 1;
  }}

  function selectedCompareLap() {{
    if (!COMPARE_PAYLOAD || !Array.isArray(COMPARE_PAYLOAD.laps)) return null;
    if (COMPARE_LAP_ID === null) return null;
    return COMPARE_PAYLOAD.laps.find(lap => lap.lap_id === COMPARE_LAP_ID) || null;
  }}

  function syncMapSourceSelector() {{
    if (!mapSourceEl) return;
    const compareLap = selectedCompareLap();
    const options = ['<option value="original">Original</option>'];
    if (compareLap) {{
      options.push('<option value="compare">Compared</option>');
    }} else {{
      mapSourceMode = "original";
    }}
    mapSourceEl.innerHTML = options.join("");
    mapSourceEl.value = compareLap && mapSourceMode === "compare" ? "compare" : "original";
  }}

  function interpolateAt(xArr, yArr, xVal) {{
    if (!Array.isArray(xArr) || !Array.isArray(yArr) || !xArr.length || !yArr.length) return NaN;
    const n = Math.min(xArr.length, yArr.length);
    let first = -1;
    let last = -1;
    for (let i = 0; i < n; i++) {{
      if (xArr[i] !== null && yArr[i] !== null && isFinite(xArr[i]) && isFinite(yArr[i])) {{
        if (first < 0) first = i;
        last = i;
      }}
    }}
    if (first < 0) return NaN;
    if (xVal <= xArr[first]) return yArr[first];
    if (xVal >= xArr[last]) return yArr[last];
    for (let i = first; i < last; i++) {{
      const x0 = xArr[i];
      const x1 = xArr[i + 1];
      const y0 = yArr[i];
      const y1 = yArr[i + 1];
      if (x0 === null || x1 === null || y0 === null || y1 === null) continue;
      if (!isFinite(x0) || !isFinite(x1) || !isFinite(y0) || !isFinite(y1)) continue;
      const inSegment = (x0 <= xVal && xVal <= x1) || (x1 <= xVal && xVal <= x0);
      if (!inSegment) continue;
      if (x1 === x0) return y0;
      const f = (xVal - x0) / (x1 - x0);
      return y0 + f * (y1 - y0);
    }}
    return yArr[first];
  }}

  function xArrayForLap(lap) {{
    if (xAxisMode === "distance") {{
      return Array.isArray(lap.distance) ? cleanArray(lap.distance) : [];
    }}
    return lap.t_global.map(tg => tg === null ? NaN : tg - offset);
  }}

  function xArrayForCompareLap(baseLap, compareLap) {{
    if (!baseLap || !compareLap) return [];
    if (xAxisMode === "distance") {{
      return Array.isArray(compareLap.distance) ? cleanArray(compareLap.distance) : [];
    }}
    if (!Array.isArray(compareLap.t_lap)) return [];
    const baseStartX = baseLap.t_start - offset;
    return compareLap.t_lap.map(tl => tl === null ? NaN : baseStartX + tl);
  }}

  function xAxisTitle() {{
    return xAxisMode === "distance" ? "Distance [m]" : "Time [s]";
  }}

  function formatSignalValue(signal, value) {{
    if (!isFinite(value)) return "—";
    const decimals = Number.isFinite(signal.decimals) ? signal.decimals : 2;
    return value.toFixed(decimals) + (signal.value_suffix || "");
  }}

  function axisRef(prefix, idx) {{
    return prefix + (idx === 0 ? "" : String(idx + 1));
  }}

  function axisLayoutName(axisRefValue, prefix) {{
    const ref = axisRefValue || prefix;
    return ref === prefix ? prefix + "axis" : prefix + "axis" + ref.slice(1);
  }}

  function xAxisLayout(idx, totalSignals) {{
    const ax = {{ gridcolor: "#2a2a2e", automargin: true, fixedrange: false }};
    if (idx < totalSignals - 1) {{
      ax.matches = axisRef("x", totalSignals - 1);
      ax.showticklabels = false;
    }} else {{
      ax.title = {{ text: xAxisTitle(), standoff: 10 }};
      ax.showline = true;
      ax.linecolor = "#8A8F98";
      ax.linewidth = 1;
    }}
    return ax;
  }}

  function finiteAbsMax(arr) {{
    let maxAbs = 0;
    for (const v of arr) {{
      if (v === null || !isFinite(v)) continue;
      maxAbs = Math.max(maxAbs, Math.abs(v));
    }}
    return maxAbs;
  }}

  function steeringRangeForLap(lap, compareLap = null) {{
    const steeringValues = Array.isArray(lap.steering) ? lap.steering : [];
    const compareValues = compareLap && Array.isArray(compareLap.steering) ? compareLap.steering : [];
    const bound = Math.max(1, finiteAbsMax(steeringValues) * 1.05, finiteAbsMax(compareValues) * 1.05);
    return [-bound, bound];
  }}

  function yAxisLayout(signal, lap, compareLap = null) {{
    const ax = {{
      title: {{ text: signal.label, standoff: 10 }},
      gridcolor: "#2a2a2e",
      automargin: true,
      autorange: true,
      fixedrange: true,
    }};
    if (signal.key === "steering" && lap) {{
      ax.autorange = false;
      ax.range = steeringRangeForLap(lap, compareLap);
      ax.zeroline = true;
      ax.zerolinecolor = "#8A8F98";
      ax.zerolinewidth = 1;
    }} else if (signal.range) {{
      ax.autorange = false;
      ax.range = signal.range;
    }}
    return ax;
  }}

  function telemetryTimeFromVideoTime(videoTime) {{
    return videoTime + offset;
  }}

  function videoTimeFromXClick(lap, xVal) {{
    if (!lap || xVal === null || !isFinite(xVal)) return null;
    if (xAxisMode === "distance") {{
      const tg = interpolateAt(lap.distance, lap.t_global, xVal);
      return isFinite(tg) ? tg - offset : null;
    }}
    return xVal;
  }}

  function applyVideoTime(videoTime) {{
    const safeTime = Math.max(0, videoTime);
    if (HAS_VIDEO) {{
      const vid = document.getElementById("vid_" + CID);
      vid.currentTime = safeTime;
    }}
    onVideoTime(safeTime);
  }}

  function chartXFromPointerEvent(gd, event) {{
    if (!gd || !gd._fullLayout || !event) return null;
    const fullLayout = gd._fullLayout;
    const xaxis = fullLayout.xaxis;
    const margin = fullLayout.margin || {{}};
    const rect = gd.getBoundingClientRect();
    const width = Number.isFinite(fullLayout.width) ? fullLayout.width : rect.width;
    const height = Number.isFinite(fullLayout.height) ? fullLayout.height : rect.height;
    const leftMargin = Number.isFinite(margin.l) ? margin.l : 0;
    const rightMargin = Number.isFinite(margin.r) ? margin.r : 0;
    const topMargin = Number.isFinite(margin.t) ? margin.t : 0;
    const bottomMargin = Number.isFinite(margin.b) ? margin.b : 0;
    const plotWidth = width - leftMargin - rightMargin;
    const plotHeight = height - topMargin - bottomMargin;
    if (!xaxis || !Array.isArray(xaxis.range) || plotWidth <= 0 || plotHeight <= 0) return null;

    const domain = Array.isArray(xaxis.domain) ? xaxis.domain : [0, 1];
    const plotX = event.clientX - rect.left;
    const plotY = event.clientY - rect.top;
    const x0Px = leftMargin + domain[0] * plotWidth;
    const x1Px = leftMargin + domain[1] * plotWidth;
    const y0Px = topMargin;
    const y1Px = topMargin + plotHeight;
    if (plotX < x0Px || plotX > x1Px || plotY < y0Px || plotY > y1Px) return null;

    const frac = Math.max(0, Math.min(1, (plotX - x0Px) / Math.max(1, x1Px - x0Px)));
    const r0 = Number(xaxis.range[0]);
    const r1 = Number(xaxis.range[1]);
    if (!isFinite(r0) || !isFinite(r1)) return null;
    return r0 + frac * (r1 - r0);
  }}

  function scrubChartToX(xVal) {{
    const lap = PAYLOAD.laps[currentLapIdx >= 0 ? currentLapIdx : 0];
    const videoTime = videoTimeFromXClick(lap, xVal);
    if (videoTime !== null && isFinite(videoTime)) applyVideoTime(videoTime);
  }}

  function populateLapSelector() {{
    lapSelectEl.innerHTML = PAYLOAD.laps
      .map((lap, idx) => {{
        const lt = lap.laptime_s === null ? "—" : lap.laptime_s.toFixed(2) + " s";
        return '<option value="' + idx + '">Lap ' + lap.lap_id + ' (' + lt + ')</option>';
      }})
      .join("");
  }}

  function syncLapSelector() {{
    if (!lapSelectEl || currentLapIdx < 0) return;
    lapSelectEl.value = String(currentLapIdx);
  }}

  function signalYAtTelemetryTime(lap, signal, tg) {{
    const tArr = lap.t_global;
    const yArr = lap[signal.key];
    if (!Array.isArray(tArr) || !Array.isArray(yArr) || !tArr.length || !yArr.length) return NaN;
    let lo = 0;
    let hi = tArr.length - 1;
    if (tg <= tArr[0]) return valueAt(yArr, 0);
    if (tg >= tArr[hi]) return valueAt(yArr, hi);
    while (hi - lo > 1) {{
      const mid = (lo + hi) >> 1;
      if (tArr[mid] <= tg) lo = mid; else hi = mid;
    }}
    const t0 = tArr[lo];
    const t1 = tArr[hi];
    const y0 = valueAt(yArr, lo);
    const y1 = valueAt(yArr, hi);
    if (!isFinite(t0) || !isFinite(t1) || !isFinite(y0) || !isFinite(y1)) return y0;
    if (t1 === t0) return y0;
    const f = (tg - t0) / (t1 - t0);
    return y0 + f * (y1 - y0);
  }}

  function cursorAnnotations(lap, signals, xVal, tg) {{
    if (!isFinite(xVal) || !isFinite(tg)) return [];
    const xValues = xArrayForLap(lap);
    const finiteX = xValues.filter(value => value !== null && isFinite(value));
    const xMin = finiteX.length ? Math.min(...finiteX) : NaN;
    const xMax = finiteX.length ? Math.max(...finiteX) : NaN;
    const xFrac = isFinite(xMin) && isFinite(xMax) && xMax !== xMin
      ? (xVal - xMin) / (xMax - xMin)
      : 0.5;
    const nearRightEdge = xFrac > 0.88;

    function screenYFromData(yref, yVal) {{
      if (!chartGd || !chartGd._fullLayout) return null;
      const axis = chartGd._fullLayout[axisLayoutName(yref, "y")];
      if (!axis || typeof axis.l2p !== "function") return null;
      const pixel = Number(axis.l2p(yVal));
      return isFinite(pixel) ? pixel : null;
    }}

    function annotationOffsets(entries) {{
      const yShifts = Array(entries.length).fill(0);
      const xShifts = Array(entries.length).fill(nearRightEdge ? -6 : 6);
      const minGapPx = 24;
      const xStepPx = 42;
      let cluster = [];

      function applyCluster(items) {{
        if (!items.length) return;
        if (items.length === 1) {{
          yShifts[items[0].idx] = 0;
          return;
        }}
        const center = items.reduce((sum, item) => sum + item.screenY, 0) / items.length;
        const start = center - (minGapPx * (items.length - 1)) / 2;
        items.forEach((item, order) => {{
          yShifts[item.idx] = start + order * minGapPx - item.screenY;
          xShifts[item.idx] = (nearRightEdge ? -6 : 6) + (nearRightEdge ? -1 : 1) * order * xStepPx;
        }});
      }}

      const ordered = entries
        .map((entry, idx) => ({{
          idx,
          yref: entry.yref,
          screenY: screenYFromData(entry.yref, entry.y),
        }}))
        .filter(item => item.screenY !== null && isFinite(item.screenY))
        .sort((a, b) => {{
          if (a.yref !== b.yref) return String(a.yref).localeCompare(String(b.yref));
          return a.screenY - b.screenY;
        }});

      ordered.forEach(item => {{
        if (!cluster.length) {{
          cluster = [item];
          return;
        }}
        const prev = cluster[cluster.length - 1];
        if (item.yref === prev.yref && item.screenY - prev.screenY < minGapPx) {{
          cluster.push(item);
          return;
        }}
        applyCluster(cluster);
        cluster = [item];
      }});
      applyCluster(cluster);
      return {{ yShifts, xShifts }};
    }}

    const entries = signals.map((signal, idx) => {{
      const yVal = signalYAtTelemetryTime(lap, signal, tg);
      const ySafe = isFinite(yVal) ? yVal : 0;
      return {{
        x: xVal,
        y: ySafe,
        xref: axisRef("x", idx),
        yref: axisRef("y", idx),
        text: formatSignalValue(signal, yVal),
        showarrow: false,
        bgcolor: "rgba(20,20,23,0.92)",
        bordercolor: signal.color,
        borderwidth: 1,
        font: {{ color: "#EBEBEB", size: 10 }},
        xanchor: nearRightEdge ? "right" : "left",
        yanchor: "middle",
        xshift: nearRightEdge ? -6 : 6,
      }};
    }});
    const offsets = annotationOffsets(entries);
    return entries.map((entry, idx) => Object.assign({{}}, entry, {{
      xshift: offsets.xShifts[idx],
      yshift: offsets.yShifts[idx],
    }}));
  }}

  function cursorShapes(signals, xVal) {{
    if (!isFinite(xVal)) return [];
    return signals.map((_, idx) => ({{
      type: "line",
      xref: axisRef("x", idx),
      yref: axisRef("y", idx) + " domain",
      x0: xVal,
      x1: xVal,
      y0: 0,
      y1: 1,
      line: {{ color: "#FFFFFF", width: 1, dash: "dot" }},
    }}));
  }}

  function cursorXForTelemetryTime(lap, tg) {{
    if (xAxisMode === "distance") {{
      return interpolateAt(lap.t_global, lap.distance, tg);
    }}
    return tg - offset;
  }}

  function currentChartHeightPx() {{
    const measuredHeight = chartEl.getBoundingClientRect().height;
    if (Number.isFinite(measuredHeight) && measuredHeight > 0) {{
      return Math.max(320, Math.round(measuredHeight));
    }}
    return DEFAULT_CHART_HEIGHT;
  }}

  function chartLayout(lap, signals, xCursor, tg) {{
    const compareLap = selectedCompareLap();
    const layout = {{
      paper_bgcolor: "#141417",
      plot_bgcolor: "#141417",
      font: {{ color: "#EBEBEB", size: 11 }},
      height: currentChartHeightPx(),
      margin: {{ l: 82, r: 18, t: 8, b: 42, autoexpand: false }},
      showlegend: false,
      hovermode: false,
      grid: {{
        rows: signals.length,
        columns: 1,
        pattern: "independent",
        ygap: signals.length > 1 ? 0.075 : 0.02,
      }},
      shapes: cursorShapes(signals, xCursor),
      annotations: cursorAnnotations(lap, signals, xCursor, tg),
      dragmode: "pan",
    }};
    signals.forEach((signal, idx) => {{
      layout[axisRef("xaxis", idx)] = xAxisLayout(idx, signals.length);
      layout[axisRef("yaxis", idx)] = yAxisLayout(signal, lap, compareLap);
    }});
    return layout;
  }}

  function plotConfig() {{
    return {{
      displaylogo: false,
      displayModeBar: true,
      responsive: true,
      scrollZoom: true,
      doubleClick: "reset+autosize",
      modeBarButtonsToRemove: [
        "lasso2d",
        "select2d",
        "toggleSpikelines",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
      ],
      modeBarButtonsToAdd: [{{
        name: "fullscreen",
        title: "Full screen charts",
        icon: fullscreenIcon,
        click: () => toggleFullscreen(document.getElementById("wrap_" + CID)),
      }}],
    }};
  }}

  function attachChartListeners(gd) {{
    if (!gd || gd.dataset.listenersAttached === "1") return;
    gd.on("plotly_click", evt => {{
      if (!evt.points || !evt.points.length) return;
      scrubChartToX(evt.points[0].x);
    }});

    let isScrubbing = false;
    const stopScrub = () => {{
      isScrubbing = false;
      document.removeEventListener("mousemove", onMove, true);
      document.removeEventListener("mouseup", onUp, true);
    }};
    const scrubFromEvent = event => {{
      const xVal = chartXFromPointerEvent(gd, event);
      if (xVal === null || !isFinite(xVal)) return;
      scrubChartToX(xVal);
    }};
    const onMove = event => {{
      if (!isScrubbing) return;
      event.preventDefault();
      event.stopPropagation();
      scrubFromEvent(event);
    }};
    const onUp = event => {{
      if (isScrubbing) {{
        event.preventDefault();
        event.stopPropagation();
      }}
      stopScrub();
    }};
    gd.addEventListener("mousedown", event => {{
      if (event.button !== 0) return;
      const xVal = chartXFromPointerEvent(gd, event);
      if (xVal === null || !isFinite(xVal)) return;
      isScrubbing = true;
      event.preventDefault();
      event.stopPropagation();
      scrubChartToX(xVal);
      document.addEventListener("mousemove", onMove, true);
      document.addEventListener("mouseup", onUp, true);
    }}, true);
    gd.dataset.listenersAttached = "1";
  }}

  function renderCharts(forceNewPlot = false) {{
    if (PAYLOAD.laps.length === 0) {{
      chartEl.innerHTML = '<div class="chart_placeholder">No valid laps in this run.</div>';
      chartGd = null;
      return;
    }}
    populateLapSelector();
    if (currentLapIdx < 0 || currentLapIdx >= PAYLOAD.laps.length) currentLapIdx = 0;
    const lap = PAYLOAD.laps[currentLapIdx];
    const signals = activeSignals();
    if (!signals.length) {{
      chartEl.innerHTML = '<div class="chart_placeholder">Select at least one signal to show telemetry.</div>';
      chartGd = null;
      syncLapSelector();
      updateLapInfo();
      return;
    }}

    const tgRaw = (currentTelemetryTime !== null && isFinite(currentTelemetryTime)) ? currentTelemetryTime : lap.t_start;
    const tg = Math.max(lap.t_start, Math.min(lap.t_end, tgRaw));
    const x = xArrayForLap(lap);
    const xCursor = cursorXForTelemetryTime(lap, tg);
    const compareLap = selectedCompareLap();
    const traces = signals.map((signal, idx) => ({{
      x: x,
      y: cleanArray(lap[signal.key]),
      type: "scattergl",
      mode: "lines",
      line: {{ color: signal.color, width: 1.4 }},
      name: signal.label,
      xaxis: axisRef("x", idx),
      yaxis: axisRef("y", idx),
      hoverinfo: "skip",
      hovertemplate: null,
    }}));
    if (compareLap) {{
      const compareX = xArrayForCompareLap(lap, compareLap);
      signals.forEach((signal, idx) => {{
        if (!Array.isArray(compareLap[signal.key])) return;
        traces.push({{
          x: compareX,
          y: cleanArray(compareLap[signal.key]),
          type: "scattergl",
          mode: "lines",
          line: {{ color: COMPARE_COLOR, width: 1.5 }},
          name: "Compare L" + compareLap.lap_id + " · " + signal.label,
          xaxis: axisRef("x", idx),
          yaxis: axisRef("y", idx),
          hoverinfo: "skip",
          hovertemplate: null,
        }});
      }});
    }}

    if (forceNewPlot && chartGd) {{
      Plotly.purge(chartEl);
      chartEl.innerHTML = "";
      chartGd = null;
    }}
    if (!chartGd) chartEl.innerHTML = "";
    const plotFn = chartGd ? Plotly.react : Plotly.newPlot;
    plotFn(chartEl, traces, chartLayout(lap, signals, xCursor, tg), plotConfig()).then(gd => {{
      chartGd = gd;
      currentTelemetryTime = tg;
      currentCursorX = xCursor;
      attachChartListeners(gd);
      syncLapSelector();
      updateLapInfo();
    }});
  }}

  function setLap(lapIdx, tgOverride = null) {{
    if (lapIdx < 0 || lapIdx >= PAYLOAD.laps.length) return;
    if (lapIdx === currentLapIdx && tgOverride === null) return;
    currentLapIdx = lapIdx;
    const lap = PAYLOAD.laps[lapIdx];
    currentTelemetryTime = (tgOverride !== null && isFinite(tgOverride)) ? tgOverride : lap.t_start;
    currentCursorX = cursorXForTelemetryTime(lap, currentTelemetryTime);
    renderCharts(true);
    updateMapForLap(lap);
    renderPhaseStats();
  }}

  function updateLapInfo() {{
    const lap = PAYLOAD.laps[currentLapIdx];
    if (!lap) return;
    const lt = lap.laptime_s === null ? "—" : lap.laptime_s.toFixed(2) + " s";
    const compareLap = selectedCompareLap();
    const compareText = compareLap
      ? " · compare L" + compareLap.lap_id + " (" + (compareLap.laptime_s === null ? "—" : compareLap.laptime_s.toFixed(2) + " s") + ")"
      : "";
    document.getElementById("lap_info_" + CID).innerHTML =
      "Lap <b>" + lap.lap_id + "</b> — laptime " + lt + compareText;
  }}

  function moveCursorTo(xVal, tg, lap) {{
    currentCursorX = xVal;
    currentTelemetryTime = tg;
    if (!chartGd || !lap || !isFinite(xVal) || !isFinite(tg)) return;
    const signals = activeSignals();
    if (!signals.length) return;
    Plotly.relayout(chartGd, {{
      shapes: cursorShapes(signals, xVal),
      annotations: cursorAnnotations(lap, signals, xVal, tg),
    }});
  }}

  function phaseStatsPayloadForSource() {{
    if (mapSourceMode === "compare" && COMPARE_PAYLOAD && COMPARE_PAYLOAD.phase_stats) {{
      return COMPARE_PAYLOAD.phase_stats;
    }}
    return PAYLOAD.phase_stats || null;
  }}

  function mapSpecForDisplay(baseLap, tgOverride = null) {{
    const baseTime = (tgOverride !== null && isFinite(tgOverride)) ? tgOverride : currentTelemetryTime;
    if (mapSourceMode === "compare") {{
      const compareLap = selectedCompareLap();
      if (compareLap && baseLap) {{
        const baseTg = (baseTime !== null && isFinite(baseTime)) ? baseTime : baseLap.t_start;
        const clampedBaseTg = Math.max(baseLap.t_start, Math.min(baseLap.t_end, baseTg));
        const compareTg = Math.max(
          compareLap.t_start,
          Math.min(compareLap.t_end, compareLap.t_start + (clampedBaseTg - baseLap.t_start)),
        );
        return {{ lap: compareLap, compare: true, tg: compareTg }};
      }}
    }}
    return {{
      lap: baseLap,
      compare: false,
      tg: (baseTime !== null && isFinite(baseTime) && baseLap)
        ? Math.max(baseLap.t_start, Math.min(baseLap.t_end, baseTime))
        : (baseLap ? baseLap.t_start : null),
    }};
  }}

  function phaseStatsLapForSource() {{
    const baseLap = PAYLOAD.laps[currentLapIdx >= 0 ? currentLapIdx : 0];
    const spec = mapSpecForDisplay(baseLap);
    return spec.lap || null;
  }}

  function formatPhaseStatValue(column, value) {{
    if (value === null || value === undefined || !isFinite(value)) return "—";
    const decimals = Number.isFinite(column.decimals) ? column.decimals : 1;
    return Number(value).toFixed(decimals);
  }}

  function renderPhaseStats() {{
    if (!phaseStatsEl) return;
    const statsPayload = phaseStatsPayloadForSource();
    const lap = phaseStatsLapForSource();
    if (!statsPayload || !statsPayload.average || !lap) {{
      phaseStatsEl.innerHTML = '<div class="phase_stats_empty">No lap stats available.</div>';
      return;
    }}
    const avgRow = statsPayload.average;
    const lapRow = statsPayload.laps && statsPayload.laps[String(lap.lap_id)] ? statsPayload.laps[String(lap.lap_id)] : null;
    if (!lapRow) {{
      phaseStatsEl.innerHTML = '<div class="phase_stats_empty">No lap stats available.</div>';
      return;
    }}
    const bestRow = statsPayload.best || null;
    const lapLabel = "L" + String(lap.lap_id);
    const bestLabel = bestRow ? "L" + String(bestRow.lap_id) : null;
    const isBestLap = bestRow && String(bestRow.lap_id) === String(lap.lap_id);
    const headerHtml =
      '<thead><tr><th>Metric</th><th>AVG</th>'
      + (bestRow ? '<th class="best_col">BEST ' + bestLabel + '</th>' : '')
      + '<th class="active_col">' + lapLabel + '</th></tr></thead>';
    const metricCellStyle = column =>
      'box-shadow: inset 3px 0 0 ' + column.color + ';'
      + 'background: linear-gradient(90deg, ' + column.color + ' 0%, rgba(255,255,255,0.015) 78%);';
    const rowHtml = column => {{
      const metricCell =
        '<td class="phase_metric_cell" style="' + metricCellStyle(column) + ';">'
        + '<div class="phase_metric">'
        + '<span class="phase_metric_dot" style="background:' + column.color + ';"></span>'
        + '<span class="phase_metric_name">' + column.label + '</span>'
        + '</div></td>';
      const avgCell = '<td>' + formatPhaseStatValue(column, avgRow[column.key]) + '</td>';
      const bestCell = bestRow
        ? '<td class="best_value">' + formatPhaseStatValue(column, bestRow[column.key]) + '</td>'
        : '';
      const lapCell = '<td class="' + (isBestLap ? 'active_best_value' : 'active_value') + '">'
        + formatPhaseStatValue(column, lapRow[column.key]) + '</td>';
      return '<tr>' + metricCell + avgCell + bestCell + lapCell + '</tr>';
    }};
    phaseStatsEl.innerHTML =
      '<table class="phase_stats_table">'
      + headerHtml
      + '<tbody>'
      + PHASE_STATS_COLUMNS.map(rowHtml).join("")
      + '</tbody></table>';
  }}

  function finiteValues(arr) {{
    return arr.filter(v => v !== null && isFinite(v));
  }}

  function mapRanges(lap) {{
    let lonVals = [];
    let latVals = [];
    if (lap) {{
      lonVals = finiteValues(lap.map_lon);
      latVals = finiteValues(lap.map_lat);
    }}
    if (!lonVals.length || !latVals.length) {{
      const outline = PAYLOAD.circuit_outline || {{ lat: [], lon: [] }};
      lonVals = finiteValues(outline.lon);
      latVals = finiteValues(outline.lat);
    }}
    if (!lonVals.length || !latVals.length) return {{ x: null, y: null }};
    const xMin = Math.min(...lonVals);
    const xMax = Math.max(...lonVals);
    const yMin = Math.min(...latVals);
    const yMax = Math.max(...latVals);
    const xPad = Math.max((xMax - xMin) * 0.04, 1e-6);
    const yPad = Math.max((yMax - yMin) * 0.04, 1e-6);
    return {{ x: [xMin - xPad, xMax + xPad], y: [yMin - yPad, yMax + yPad] }};
  }}

  function phasePoints(lap, phaseCode) {{
    const lon = [];
    const lat = [];
    for (let i = 0; i < lap.map_phase.length; i++) {{
      if (lap.map_phase[i] !== phaseCode) continue;
      const x = lap.map_lon[i];
      const y = lap.map_lat[i];
      if (x === null || y === null || !isFinite(x) || !isFinite(y)) continue;
      lon.push(x);
      lat.push(y);
    }}
    return {{ lon: lon, lat: lat }};
  }}

  function phaseLinePoints(lap, phaseCode) {{
    const lon = [];
    const lat = [];
    let segmentOpen = false;
    for (let i = 0; i < lap.map_phase.length; i++) {{
      const x = lap.map_lon[i];
      const y = lap.map_lat[i];
      const validPoint = x !== null && y !== null && isFinite(x) && isFinite(y);
      if (!validPoint || lap.map_phase[i] !== phaseCode) {{
        if (segmentOpen) {{
          lon.push(null);
          lat.push(null);
          segmentOpen = false;
        }}
        continue;
      }}
      lon.push(x);
      lat.push(y);
      segmentOpen = true;
    }}
    return {{ lon: lon, lat: lat }};
  }}

  function compareMapTraces(compareLap) {{
    if (!compareLap) return [];
    const traces = [];
    PHASES.forEach(phase => {{
      const pts = phaseLinePoints(compareLap, phase.code);
      const style = COMPARE_PHASE_STYLES[phase.key] || {{ dash: "solid", width: 4.8 }};
      const phaseColor = COMPARE_PHASE_COLORS[phase.key] || COMPARE_COLOR;
      traces.push({{
        x: pts.lon,
        y: pts.lat,
        type: "scattergl",
        mode: "lines",
        line: {{ color: phaseColor, width: style.width, dash: style.dash }},
        name: "Compare · " + phase.label,
        legendgroup: "compare_" + phase.key,
        hoverinfo: "skip",
        showlegend: true,
      }});
    }});
    return traces;
  }}

  function mapTracesForLap(lap, compare = false) {{
    const traces = [{{
      x: cleanArray(lap.map_lon),
      y: cleanArray(lap.map_lat),
      type: "scattergl",
      mode: "lines",
      line: {{ color: "#34373D", width: 2 }},
      hoverinfo: "skip",
      name: "Track",
      showlegend: false,
    }}];
    if (compare) {{
      traces.push(...compareMapTraces(lap));
    }} else {{
      PHASES.forEach(phase => {{
        const pts = phasePoints(lap, phase.code);
        traces.push({{
          x: pts.lon,
          y: pts.lat,
          type: "scattergl",
          mode: "markers",
          marker: {{ size: 5, color: phase.color, opacity: 0.95 }},
          name: phase.label,
          legendgroup: phase.key,
          hoverinfo: "skip",
          showlegend: true,
        }});
      }});
    }}
    traces.push({{
      x: [lap.map_lon[0] ?? null],
      y: [lap.map_lat[0] ?? null],
      type: "scattergl",
      mode: "markers",
      marker: {{ size: 12, color: "#F28C40", line: {{ color: "#fff", width: 1 }} }},
      hoverinfo: "skip",
      name: "Car",
      showlegend: false,
    }});
    return traces;
  }}

  function mapLayout(lap) {{
    const ranges = mapRanges(lap);
    return {{
      paper_bgcolor: "#141417",
      plot_bgcolor: "#141417",
      font: {{ color: "#EBEBEB", size: 11 }},
      margin: {{ l: 6, r: 6, t: 6, b: 34 }},
      showlegend: true,
      legend: {{
        orientation: "h",
        x: 0,
        y: -0.08,
        bgcolor: "rgba(20,20,23,0.78)",
        font: {{ size: 10 }},
      }},
      xaxis: {{ visible: false, scaleanchor: "y", range: ranges.x }},
      yaxis: {{ visible: false, range: ranges.y }},
    }};
  }}

  function buildMap() {{
    const baseLap = PAYLOAD.laps[currentLapIdx >= 0 ? currentLapIdx : 0];
    const spec = mapSpecForDisplay(baseLap);
    if (!spec.lap) return;
    Plotly.newPlot("map_" + CID, mapTracesForLap(spec.lap, spec.compare), mapLayout(spec.lap), {{
      displayModeBar: false,
      responsive: true,
    }});
  }}

  function updateMapForLap(baseLap) {{
    const spec = mapSpecForDisplay(baseLap);
    if (!spec.lap) return;
    Plotly.react("map_" + CID, mapTracesForLap(spec.lap, spec.compare), mapLayout(spec.lap), {{
      displayModeBar: false,
      responsive: true,
    }});
  }}

  function moveDot(tg) {{
    const baseLap = PAYLOAD.laps[currentLapIdx];
    const spec = mapSpecForDisplay(baseLap, tg);
    const lap = spec.lap;
    const mapTg = spec.tg;
    if (!lap || mapTg === null || !isFinite(mapTg)) return;
    const tArr = lap.map_t_global;
    if (tArr.length === 0) return;
    let lo = 0;
    let hi = tArr.length - 1;
    if (mapTg <= tArr[0]) {{
      updateDot(lap.map_lon[0], lap.map_lat[0]);
      return;
    }}
    if (mapTg >= tArr[hi]) {{
      updateDot(lap.map_lon[hi], lap.map_lat[hi]);
      return;
    }}
    while (hi - lo > 1) {{
      const mid = (lo + hi) >> 1;
      if (tArr[mid] <= mapTg) lo = mid; else hi = mid;
    }}
    const t0 = tArr[lo];
    const t1 = tArr[hi];
    if (t0 === null || t1 === null || t1 === t0) {{
      updateDot(lap.map_lon[lo], lap.map_lat[lo]);
      return;
    }}
    const f = (mapTg - t0) / (t1 - t0);
    updateDot(
      lap.map_lon[lo] + f * (lap.map_lon[hi] - lap.map_lon[lo]),
      lap.map_lat[lo] + f * (lap.map_lat[hi] - lap.map_lat[lo]),
    );
  }}

  function updateDot(lon, lat) {{
    if (lon === null || lat === null || !isFinite(lon) || !isFinite(lat)) return;
    Plotly.restyle("map_" + CID, {{ x: [[lon]], y: [[lat]] }}, [PHASES.length + 1]);
  }}

  function onVideoTime(currentTime) {{
    const tg = telemetryTimeFromVideoTime(currentTime);
    const lapIdx = lapForTime(tg);
    if (lapIdx !== currentLapIdx) {{
      setLap(lapIdx, tg);
    }}
    const lap = PAYLOAD.laps[currentLapIdx];
    if (!lap) return;
    const xv = cursorXForTelemetryTime(lap, tg);
    moveCursorTo(xv, tg, lap);
    moveDot(tg);
  }}

  function toggleFullscreen(targetEl) {{
    if (!targetEl) return;
    if (document.fullscreenElement) {{
      document.exitFullscreen();
      return;
    }}
    if (targetEl.requestFullscreen) targetEl.requestFullscreen();
  }}

  function resetZoom() {{
    if (!chartGd) return;
    Plotly.relayout(chartGd, {{ "xaxis.autorange": true }});
    const lap = PAYLOAD.laps[currentLapIdx >= 0 ? currentLapIdx : 0];
    const tg = currentTelemetryTime !== null ? currentTelemetryTime : (lap ? lap.t_start : 0);
    if (lap) moveCursorTo(cursorXForTelemetryTime(lap, tg), tg, lap);
  }}

  renderSignalSelectors();
  syncMapSourceSelector();
  renderCharts(true);
  buildMap();
  renderPhaseStats();

  signalSlotsEl.addEventListener("change", e => {{
    const target = e.target;
    if (!target || !target.matches("select[data-slot]")) return;
    const slotIdx = parseInt(target.dataset.slot, 10);
    if (!Number.isInteger(slotIdx) || slotIdx < 0 || slotIdx >= SIGNAL_SLOT_COUNT) return;
    slotSignalKeys[slotIdx] = target.value;
    renderCharts(true);
  }});

  document.getElementById("offset_" + CID).addEventListener("input", e => {{
    offset = parseFloat(e.target.value);
    document.getElementById("offset_val_" + CID).textContent = (offset >= 0 ? "+" : "") + offset.toFixed(2);
    const lap = PAYLOAD.laps[currentLapIdx >= 0 ? currentLapIdx : 0];
    renderCharts(true);
    if (lap) updateMapForLap(lap);
    if (HAS_VIDEO) {{
      const vid = document.getElementById("vid_" + CID);
      onVideoTime(vid.currentTime);
    }}
  }});

  xAxisModeEl.addEventListener("change", e => {{
    xAxisMode = e.target.value === "distance" ? "distance" : "time";
    renderCharts(true);
    const lap = PAYLOAD.laps[currentLapIdx >= 0 ? currentLapIdx : 0];
    if (!lap) return;
    const tg = currentTelemetryTime !== null ? currentTelemetryTime : lap.t_start;
    moveCursorTo(cursorXForTelemetryTime(lap, tg), tg, lap);
  }});

  lapSelectEl.addEventListener("change", e => {{
    const lapIdx = parseInt(e.target.value, 10);
    if (!Number.isInteger(lapIdx) || lapIdx < 0 || lapIdx >= PAYLOAD.laps.length) return;
    const lap = PAYLOAD.laps[lapIdx];
    applyVideoTime(lap.t_start - offset);
  }});

  if (mapSourceEl) {{
    mapSourceEl.addEventListener("change", e => {{
      mapSourceMode = e.target.value === "compare" ? "compare" : "original";
      const lap = PAYLOAD.laps[currentLapIdx >= 0 ? currentLapIdx : 0];
      updateMapForLap(lap);
      moveDot(currentTelemetryTime);
      renderPhaseStats();
    }});
  }}

  document.getElementById("reset_zoom_" + CID).addEventListener("click", resetZoom);
  document.getElementById("fullscreen_" + CID).addEventListener("click", () => {{
    toggleFullscreen(document.getElementById("wrap_" + CID));
  }});

  if (HAS_VIDEO) {{
    const vid = document.getElementById("vid_" + CID);
    vid.addEventListener("timeupdate", () => onVideoTime(vid.currentTime));
    vid.addEventListener("seeked", () => onVideoTime(vid.currentTime));
    vid.addEventListener("loadedmetadata", () => onVideoTime(vid.currentTime));
    vid.addEventListener("error", () => setStatus("Video load failed.", true));
    vid.addEventListener("loadeddata", () => setStatus("", false));
  }}
}})();
</script>
</body>
</html>
"""
