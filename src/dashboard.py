"""CAT17x — Telemetry Dashboard

Entry point:  streamlit run src/dashboard.py

This is the only file that calls st.plotly_chart() or any other st.* rendering
functions.  All src/ modules return go.Figure objects (and kpis dicts) and never
render themselves.
"""
from __future__ import annotations

import copy
import html
import importlib
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pathlib import Path

import numpy as np
import polars as pl
import plotly.graph_objects as go
import streamlit as st

import streamlit.components.v1 as components

import src.powertrain as pt
import src.dynamics as dyn
import src.cornering as corn
import src.tc as tc
import src.tv as tv
import src.rb as rb
import src.driver as drv
import src.gripfactor as gf
import src.skidpad as skidpad
import src.lap_sectors as lsec
import src.lapcount as lapcount
import src.track_map_component as tmc
import src.videoanalysis as va

tv = importlib.reload(tv)
from utils import (
    WHEEL_COLORS,
    available_laps,
    cols_to_numpy,
    enrich_run_df,
    load_data,
    select_laps_df,
    style_metrics_table,
    style_per_lap_table,
)

DATA_DIR = Path(__file__).parent.parent / "data"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RUN_COLORS = ("#4DB3F2", "#F28C40", "#73D973", "#F27070", "#D973D9", "#F2C94C")
TRACE_DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")
TRACE_SYMBOLS = ("circle", "square", "diamond", "triangle-up", "x", "cross")
POTENTIAL_LAP_RUN = "__potential_lap__"
POTENTIAL_LAP_ID = 1
FileSignature = tuple[int, int]
_PL_HASH_FUNCS = {pl.DataFrame: lambda _df: 0}
_TELEMETRY_REQUIRED_HEADER_COLS = {"TimeStamp"}
_TELEMETRY_SIGNAL_HEADER_COLS = {"laps", "VN_vx", "VN_latitude", "VN_longitude"}


# ── Data loading ──────────────────────────────────────────────────────────────

def _file_signature(path: Path) -> FileSignature:
    """Return a cache-busting signature for *path* based on its current stat()."""
    stat = path.stat()
    return (int(stat.st_mtime_ns), int(stat.st_size))


def _is_telemetry_csv(path: Path) -> bool:
    """Fast header-only check to ignore support CSVs stored under data/."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            header = fh.readline().strip().split(",")
    except OSError:
        return False
    header_cols = set(header)
    return (
        _TELEMETRY_REQUIRED_HEADER_COLS.issubset(header_cols)
        and bool(_TELEMETRY_SIGNAL_HEADER_COLS & header_cols)
    )


def _telemetry_csv_paths(data_dir: Path) -> list[Path]:
    """Return dashboard-loadable telemetry CSVs, excluding lookup/support files."""
    return sorted(path for path in data_dir.glob("*.csv") if _is_telemetry_csv(path))


def _video_dir_signature(repo_root: Path) -> tuple[tuple[str, int, int], ...]:
    """Return a lightweight cache token for available onboard videos."""
    videos_dir = repo_root / "videos"
    if not videos_dir.is_dir():
        return ()
    out: list[tuple[str, int, int]] = []
    for path in sorted(videos_dir.glob("*.mp4")):
        try:
            stat = path.stat()
        except OSError:
            continue
        out.append((path.name, int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(out)


@st.cache_resource(show_spinner="Loading run...")
def load_run(path: str, file_signature: FileSignature) -> pl.DataFrame:
    """Load a CSV run through the shared project loader, keeping all laps."""
    _ = file_signature
    return enrich_run_df(load_data(path, complete_laps_only=False))


@st.cache_data(show_spinner=False)
def load_lap_gate(path: str, file_signature: FileSignature) -> dict | None:
    """Load lap-detection gate metadata from the raw CSV for map overlays."""
    _ = file_signature
    return lapcount.lap_detection_gate_from_csv(path)


@st.cache_data(show_spinner=False)
def csv_needs_lap_detection_cached(
    path: str,
    file_signature: FileSignature,
) -> bool:
    """Return whether *path* still needs GPS lap detection."""
    _ = file_signature
    return lapcount.csv_needs_lap_detection(path)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _select_laps_df_cached(
    df: pl.DataFrame,
    run_token: tuple[str, FileSignature],
    lap_ids: tuple[int, ...],
) -> pl.DataFrame:
    """Filter selected laps once per run/lap selection instead of every rerun."""
    _ = run_token
    return select_laps_df(df, list(lap_ids))


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _available_laps_cached(
    df: pl.DataFrame,
    run_token: tuple[str, FileSignature],
) -> tuple[int, ...]:
    """Cached lap IDs for sidebar selectors."""
    _ = run_token
    return tuple(int(lap) for lap in available_laps(df).tolist())


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _lap_laptimes_cached(
    df: pl.DataFrame,
    run_token: tuple[str, FileSignature],
) -> dict[int, float]:
    """Cached laptime labels for selectors."""
    _ = run_token
    return _lap_laptimes(df)


@st.cache_resource(show_spinner=False)
def _video_server_cached(
    repo_root: str,
    video_signature: tuple[tuple[str, int, int], ...],
) -> va.VideoServerInfo:
    """Start/reuse the video HTTP server; refresh only when video files change."""
    _ = video_signature
    return va.ensure_video_server(Path(repo_root))


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _video_payload_cached(
    df: pl.DataFrame,
    run_token: tuple[str, FileSignature],
) -> dict:
    """Build the heavy Video Analysis JSON payload once per raw CSV version."""
    _ = run_token
    return va.build_video_payload(df)


def _clear_data_caches() -> None:
    """Clear cached data derived from CSV contents."""
    cached_funcs = (
        load_run,
        load_lap_gate,
        csv_needs_lap_detection_cached,
        _select_laps_df_cached,
        _available_laps_cached,
        _lap_laptimes_cached,
        _video_server_cached,
        _video_payload_cached,
        _pt_energy_per_lap_fig_cached,
        _pt_power_per_wheel_fig_cached,
        _pt_battery_status_fig_cached,
        _pt_thermal_evolution_fig_cached,
        _dyn_ideal_braking_curve_fig_cached,
        _dyn_decel_envelope_fig_cached,
        _skidpad_fig_cached,
        _dyn_ideal_traction_curve_fig_cached,
        _dyn_accel_envelope_fig_cached,
        _driver_summary_cached,
        _driver_throttle_histogram_fig_cached,
        _driver_full_throttle_time_fig_cached,
        _driver_throttle_speed_fig_cached,
        _driver_braking_effort_fig_cached,
        _driver_brake_application_point_fig_cached,
        _driver_braking_aggressiveness_fig_cached,
        _driver_brake_release_smoothness_fig_cached,
        _driver_steering_smoothness_fig_cached,
        _driver_steering_integral_fig_cached,
        _driver_steering_stability_fig_cached,
        _driver_corner_curvature_fig_cached,
        _driver_circuit_map_fig_cached,
        _driver_circuit_map_stats_cached,
        _driver_lap_time_progression_fig_cached,
        _driver_lap_consistency_stats_cached,
        _driver_lap_time_distribution_fig_cached,
        _driver_cornering_turns_cached,
        _driver_cornering_metrics_cached,
        _driver_fastest_lap_cached,
        _driver_lap_sectors_cached,
        _driver_csv_sector_summary_cached,
        _driver_whole_lap_metrics_cached,
        _driver_potential_lap_cached,
    )
    for func in cached_funcs:
        clear = getattr(func, "clear", None)
        if callable(clear):
            clear()


def _track_zone_mask_from_session(n_points: int) -> tuple[np.ndarray, bool]:
    """Return the current track-zone mask stored in session state."""
    raw = np.asarray(st.session_state.get("_dyn_track_selection_indices", []), dtype=int)
    valid = raw[(raw >= 0) & (raw < n_points)]
    if len(valid) == 0:
        return np.ones(n_points, dtype=bool), False
    mask = np.zeros(n_points, dtype=bool)
    mask[np.unique(valid)] = True
    return mask, True


def _consume_track_component_event(
    track_event: dict[str, object] | None,
    *,
    pool_len: int,
    event_state_key: str,
) -> None:
    """Persist track-map lasso/manual-line interactions from the custom component."""
    if not isinstance(track_event, dict):
        return

    event_id = int(track_event.get("event_id", 0) or 0)
    last_event_id = int(st.session_state.get(event_state_key, -1))
    if event_id <= last_event_id:
        return
    st.session_state[event_state_key] = event_id

    selection_indices = np.asarray(track_event.get("selection_indices", []), dtype=int)
    valid_sel = selection_indices[(selection_indices >= 0) & (selection_indices < pool_len)]
    st.session_state["_dyn_track_selection_indices"] = np.unique(valid_sel).tolist()

    if bool(track_event.get("fullscreen_event", False)):
        st.session_state["_dyn_track_open_fullscreen"] = True
        st.rerun()

    if not bool(track_event.get("line_event", False)):
        return

    line_payload = track_event.get("line")
    if isinstance(line_payload, dict):
        try:
            manual_gate_line = (
                (float(line_payload["x0"]), float(line_payload["y0"])),
                (float(line_payload["x1"]), float(line_payload["y1"])),
            )
        except (KeyError, TypeError, ValueError):
            manual_gate_line = st.session_state.get("_dyn_manual_gate_line")
        else:
            st.session_state["_dyn_manual_gate_line"] = manual_gate_line
    else:
        st.session_state.pop("_dyn_manual_gate_line", None)
    st.rerun()


def _consume_lap_turn_click_event(
    track_event: dict[str, object] | None,
    *,
    all_turn_ids: list[int],
    included_state_key: str,
    event_state_key: str,
) -> bool:
    """Toggle a Lap Analysis turn when the track-map component reports a click."""
    if not isinstance(track_event, dict):
        return False

    event_id = int(track_event.get("event_id", 0) or 0)
    last_event_id = int(st.session_state.get(event_state_key, -1))
    if event_id <= last_event_id:
        return False
    st.session_state[event_state_key] = event_id

    clicked = track_event.get("clicked_turn_id")
    if clicked is None:
        return False
    try:
        turn_id = int(clicked)
    except (TypeError, ValueError):
        return False
    valid_turns = {int(tid) for tid in all_turn_ids}
    if turn_id not in valid_turns:
        return False

    included = {
        int(tid)
        for tid in st.session_state.get(included_state_key, all_turn_ids)
        if int(tid) in valid_turns
    }
    if turn_id in included:
        included.remove(turn_id)
    else:
        included.add(turn_id)
    st.session_state[included_state_key] = sorted(included)
    return True


def _add_manual_gate_line_to_fig(
    fig: go.Figure,
    manual_gate_line: tuple[tuple[float, float], tuple[float, float]] | None,
) -> go.Figure:
    """Overlay the manual finish-line preview on a longitude/latitude figure."""
    if manual_gate_line is None:
        return fig
    (x0, y0), (x1, y1) = manual_gate_line
    fig.add_shape(
        type="line",
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        xref="x",
        yref="y",
        line=dict(color="#4DB3F2", width=3),
    )
    fig.add_trace(go.Scattergl(
        x=[x0, x1],
        y=[y0, y1],
        mode="markers",
        marker=dict(color="#4DB3F2", size=8, symbol="circle"),
        name="Manual finish line",
        hovertemplate="Manual finish line<extra></extra>",
        showlegend=False,
    ))
    return fig


def _lap_gates_from_run_tokens(
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> dict[str, dict]:
    """Load lapcount finish-gate metadata for the selected CSV runs."""
    lap_gates: dict[str, dict] = {}
    for run_name, file_signature, _lap_token in run_tokens:
        try:
            gate = load_lap_gate(str(DATA_DIR / run_name), file_signature)
        except Exception:
            continue
        if not gate:
            continue
        try:
            gate_lon = np.asarray(gate["gate_lon"], dtype=float)
            gate_lat = np.asarray(gate["gate_lat"], dtype=float)
            finish_lon = float(gate["finish_lon"])
            finish_lat = float(gate["finish_lat"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            gate_lon.size == 2
            and gate_lat.size == 2
            and np.all(np.isfinite(gate_lon))
            and np.all(np.isfinite(gate_lat))
            and np.isfinite(finish_lon)
            and np.isfinite(finish_lat)
        ):
            lap_gates[run_name] = gate
    return lap_gates


def _add_lap_detection_gates_to_fig(
    fig: go.Figure,
    lap_gates: dict[str, dict],
) -> go.Figure:
    """Overlay lapcount-detected finish lines on a longitude/latitude figure."""
    if not lap_gates:
        return fig

    multi_run = len(lap_gates) > 1
    for idx, (run_name, gate) in enumerate(lap_gates.items()):
        gate_lon = np.asarray(gate["gate_lon"], dtype=float)
        gate_lat = np.asarray(gate["gate_lat"], dtype=float)
        finish_lon = float(gate["finish_lon"])
        finish_lat = float(gate["finish_lat"])
        gate_half_width_m = float(gate.get("gate_half_width_m", np.nan))
        mode = str(gate.get("lapcount_mode", "circuit"))
        color = RUN_COLORS[idx % len(RUN_COLORS)] if multi_run else "#F2F2F2"
        label = f"Lapcount finish · {Path(run_name).stem}" if multi_run else "Lapcount finish"

        fig.add_trace(go.Scattergl(
            x=gate_lon,
            y=gate_lat,
            mode="lines",
            name=label,
            line=dict(color=color, width=2.5, dash="dash"),
            hovertemplate=(
                f"{label}<br>"
                f"mode={mode}<br>"
                f"half width={gate_half_width_m:.1f} m"
                "<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scattergl(
            x=[finish_lon],
            y=[finish_lat],
            mode="markers",
            marker=dict(
                size=10,
                color=color,
                symbol="x",
                line=dict(color="#FFFFFF", width=1.5),
            ),
            showlegend=False,
            hovertemplate=(
                f"{label} centre"
                f"<br>lon={finish_lon:.6f}"
                f"<br>lat={finish_lat:.6f}<extra></extra>"
            ),
        ))
    return fig


@st.dialog("Lap Analysis Phase Map", width="large")
def _render_lap_phase_fullscreen_dialog(
    phase_fig: go.Figure,
    *,
    turn_ids: list[int],
    included_turns_key: str,
    event_state_key: str,
) -> None:
    """Render the Lap Analysis phase map in a large dialog."""
    st.caption("Click a curve to include/exclude it from Lap Analysis, or draw a line for the finish line.")
    phase_event = tmc.render_track_map_component(
        tmc.serialize_figure(phase_fig),
        height_px=760,
        key=f"drv_lap_phase_map_fullscreen_{event_state_key}",
    )
    _consume_track_component_event(
        phase_event,
        pool_len=0,
        event_state_key=f"{event_state_key}_manual_fullscreen",
    )
    if _consume_lap_turn_click_event(
        phase_event,
        all_turn_ids=turn_ids,
        included_state_key=included_turns_key,
        event_state_key=f"{event_state_key}_fullscreen",
    ):
        st.rerun()


def _peek_persisted_mode_and_gate(
    csv_path: Path,
) -> tuple[str | None, tuple[tuple[float, float], tuple[float, float]] | None, bool]:
    """Read the previously written event mode, centre gate and manual flag.

    Returns (mode_str_or_None, gate_or_None, manual_gate). ``mode_str`` is
    ``None`` for circuit/auto so the caller can fall back to plain auto
    detection. ``manual_gate`` is True when the persisted gate came from a
    user-drawn line (lapcount stores ``min_vel = NaN`` for manual circuit and
    skidpad paths) and must therefore survive a re-detection.
    """
    candidate_cols = (
        "lapcount_mode",
        "lapcount_gate_lon0_deg",
        "lapcount_gate_lat0_deg",
        "lapcount_gate_lon1_deg",
        "lapcount_gate_lat1_deg",
        "lapcount_min_vel_mps",
    )
    try:
        header = pl.read_csv(str(csv_path), n_rows=0).columns
    except Exception:
        return None, None, False
    available = [c for c in candidate_cols if c in header]
    if "lapcount_mode" not in available:
        return None, None, False
    try:
        df = pl.read_csv(str(csv_path), columns=available)
    except Exception:
        return None, None, False
    mode = _current_event_mode_label(df)
    mode_str: str | None = None if mode == "Auto" else mode.lower()
    gate = _stored_gate_line(df)
    manual = False
    if "lapcount_min_vel_mps" in available:
        vals = df["lapcount_min_vel_mps"].drop_nulls()
        if len(vals) > 0:
            try:
                manual = not np.isfinite(float(vals[0]))
            except (TypeError, ValueError):
                manual = False
    return mode_str, gate, manual


def _autodetect_laps(data_dir: Path) -> None:
    """Run lap detection on any CSV in *data_dir* that doesn't have it yet.

    When a CSV already carries a previously chosen event mode (e.g. skidpad
    after a manual override), preserve that mode and the saved gate so a
    lapcount algorithm bump cannot silently demote the session back to
    circuit.
    """
    modified = False
    for path in _telemetry_csv_paths(data_dir):
        try:
            file_signature = _file_signature(path)
            if not csv_needs_lap_detection_cached(str(path), file_signature):
                continue
        except Exception as exc:
            st.sidebar.warning(f"`{path.name}`: cannot inspect — {exc}")
            continue
        persisted_mode, persisted_gate, manual_gate = _peek_persisted_mode_and_gate(path)
        with st.spinner(f"Detecting laps in {path.name}..."):
            try:
                if persisted_mode == "skidpad":
                    # Pass the stored gate (may be None); detect_and_write_laps
                    # falls back to GPS-based auto-estimation when the gate is
                    # missing or stale.
                    n = lapcount.detect_and_write_laps(
                        path, mode="skidpad", gate_line_lonlat=persisted_gate,
                    )
                elif persisted_mode == "acceleration":
                    n = lapcount.detect_and_write_laps(path, mode="acceleration")
                elif manual_gate and persisted_gate is not None:
                    # Circuit run with a user-drawn finish line. Preserve the
                    # manual gate; otherwise auto-fit would silently relocate
                    # the start/finish line on every algorithm version bump.
                    n = lapcount.detect_and_write_laps(
                        path, gate_line_lonlat=persisted_gate,
                    )
                else:
                    n = lapcount.detect_and_write_laps(path)
            except Exception as exc:
                st.sidebar.warning(f"`{path.name}`: lap detection failed — {exc}")
                continue
        modified = True
        if persisted_mode:
            mode_label = persisted_mode
        elif manual_gate and persisted_gate is not None:
            mode_label = "circuit · manual gate"
        else:
            mode_label = "auto"
        if n > 0:
            st.sidebar.info(f"`{path.name}`: detected {n} laps ({mode_label})")
        else:
            st.sidebar.warning(
                f"`{path.name}`: no laps detected from GPS ({mode_label})"
            )
    if modified:
        _clear_data_caches()


_EVENT_MODE_LABELS: tuple[str, ...] = ("Auto", "Acceleration", "Skidpad")
_EVENT_MODE_TO_LABEL: dict[str, str] = {
    "circuit": "Auto",
    "auto": "Auto",
    "acceleration": "Acceleration",
    "accel": "Acceleration",
    "skidpad": "Skidpad",
}


def _current_event_mode_label(raw_df: pl.DataFrame | None) -> str:
    """Return the human-readable event mode currently stored in the CSV."""
    if raw_df is None or "lapcount_mode" not in raw_df.columns:
        return "Auto"
    values = raw_df["lapcount_mode"].drop_nulls()
    if len(values) == 0:
        return "Auto"
    return _EVENT_MODE_TO_LABEL.get(str(values[0]).strip().lower(), "Auto")


def _stored_gate_line(raw_df: pl.DataFrame | None
                      ) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Recover a previously written gate line (lon, lat) from CSV metadata."""
    if raw_df is None:
        return None
    cols = ("lapcount_gate_lon0_deg", "lapcount_gate_lat0_deg",
            "lapcount_gate_lon1_deg", "lapcount_gate_lat1_deg")
    if not all(c in raw_df.columns for c in cols):
        return None
    try:
        lon0 = float(raw_df[cols[0]].drop_nulls()[0])
        lat0 = float(raw_df[cols[1]].drop_nulls()[0])
        lon1 = float(raw_df[cols[2]].drop_nulls()[0])
        lat1 = float(raw_df[cols[3]].drop_nulls()[0])
    except (IndexError, ValueError, TypeError):
        return None
    if not all(np.isfinite(v) for v in (lon0, lat0, lon1, lat1)):
        return None
    return ((lon0, lat0), (lon1, lat1))


def _redetect_with_event_mode(
    csv_path: Path,
    label: str,
    *,
    gate_line: tuple[tuple[float, float], tuple[float, float]] | None,
) -> tuple[bool, str]:
    """Re-run lapcount with the user-selected event mode. Returns (ok, message)."""
    label = (label or "Auto").strip()
    if label == "Auto":
        try:
            n = lapcount.detect_and_write_laps(str(csv_path))
        except Exception as exc:
            return False, f"`{csv_path.name}`: auto detection failed — {exc}"
        return True, f"`{csv_path.name}`: {n} laps (auto)"
    if label == "Acceleration":
        try:
            n = lapcount.detect_and_write_laps(str(csv_path), mode="acceleration")
        except Exception as exc:
            return False, f"`{csv_path.name}`: acceleration detection failed — {exc}"
        return True, f"`{csv_path.name}`: {n} laps (acceleration)"
    if label == "Skidpad":
        # gate_line may be None — detect_and_write_laps falls back to a GPS-
        # based auto-gate when no gate is provided or the supplied gate yields
        # no plausible laps.
        try:
            n = lapcount.detect_and_write_laps(
                str(csv_path), mode="skidpad", gate_line_lonlat=gate_line,
            )
        except Exception as exc:
            return False, f"`{csv_path.name}`: skidpad detection failed — {exc}"
        suffix = "skidpad" if gate_line is not None else "skidpad, auto-gate"
        return True, f"`{csv_path.name}`: {n} laps ({suffix})"
    return False, f"`{csv_path.name}`: unknown mode {label!r}"


def _render_event_mode_selector(
    selected_files: list[str],
    raw_dfs: dict[str, pl.DataFrame],
) -> None:
    """Sidebar selector to override the lapcount event mode per CSV."""
    if not selected_files:
        return
    st.sidebar.divider()
    st.sidebar.markdown("### Event mode")
    st.sidebar.caption(
        "How lapcount segments laps. Skidpad needs a centre-gate line drawn "
        "on a track map (Lap Analysis or Dynamics)."
    )
    pending_changes: list[tuple[Path, str]] = []
    for fname in selected_files:
        raw_df = raw_dfs.get(fname)
        current_label = _current_event_mode_label(raw_df)
        widget_key = f"event_mode_{fname}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = current_label
        elif st.session_state[widget_key] not in _EVENT_MODE_LABELS:
            st.session_state[widget_key] = current_label

        selected_label = st.sidebar.selectbox(
            fname,
            options=_EVENT_MODE_LABELS,
            key=widget_key,
        )
        if selected_label != current_label:
            pending_changes.append((DATA_DIR / fname, selected_label))

    if not pending_changes:
        return

    messages: list[tuple[bool, str]] = []
    any_success = False
    manual_gate = st.session_state.get("_dyn_manual_gate_line")
    with st.spinner("Re-detecting laps with the new event mode..."):
        for csv_path, label in pending_changes:
            # Forward the session manual gate as-is so users can retry
            # detection with a freshly drawn line (including a skidpad
            # centre-gate). Do NOT fall back to the CSV's persisted gate
            # for Skidpad: a gate previously written by circuit/auto runs
            # is a finish-line, not a centre-gate, and would silently feed
            # the wrong geometry to skidpad detection. lapcount's skidpad
            # path itself recovers via GPS auto-estimation when a provided
            # gate yields no plausible laps.
            gate_line = manual_gate
            ok, msg = _redetect_with_event_mode(csv_path, label, gate_line=gate_line)
            messages.append((ok, msg))
            if ok:
                any_success = True
            else:
                st.session_state[f"event_mode_{csv_path.name}"] = (
                    _current_event_mode_label(raw_dfs.get(csv_path.name))
                )
    for ok, msg in messages:
        (st.sidebar.success if ok else st.sidebar.error)(msg)
    if any_success:
        _clear_data_caches()
        st.rerun()


def _fmt(value: float | int, pattern: str) -> str:
    """Format finite numeric values, fallback to n/a."""
    try:
        if not np.isfinite(value):
            return "n/a"
    except TypeError:
        return "n/a"
    return format(value, pattern)


def _select_per_lap_axis(key: str, default: str) -> str:
    """Dashboard selector for per-lap chart x-axis."""
    labels = ["Lap", "Lap time [s]"]
    default_index = 0 if default == "laps" else 1
    choice = st.radio(
        "X-axis",
        options=labels,
        index=default_index,
        horizontal=True,
        key=key,
    )
    return "laps" if choice == "Lap" else "laptime"


def _format_lap_label(lap_id: int) -> str:
    """Human-readable label for sidebar and plot selectors."""
    lap_int = int(lap_id)
    return "Lap 0 (formation)" if lap_int == 0 else f"Lap {lap_int}"




def _lap_laptimes(df: pl.DataFrame) -> dict[int, float]:
    """Return max detected laptime per lap for selector labels."""
    if "laps" not in df.columns or "laptime" not in df.columns:
        return {}
    cols = cols_to_numpy(df, ["laps", "laptime"])
    laps = cols["laps"]
    laptime = cols["laptime"]
    out: dict[int, float] = {}
    for lap_id in available_laps(df).tolist():
        mask = laps == float(lap_id)
        if mask.any() and np.any(np.isfinite(laptime[mask])):
            out[int(lap_id)] = float(np.nanmax(laptime[mask]))
        else:
            out[int(lap_id)] = np.nan
    return out


def _format_lap_with_laptime(lap_id: int, lap_times: dict[int, float]) -> str:
    """Human-readable lap label including laptime when available."""
    base = _format_lap_label(lap_id)
    lt = lap_times.get(int(lap_id), np.nan)
    return f"{base} ({lt:.2f} s)" if np.isfinite(lt) else base


def _lap_signature(df: pl.DataFrame) -> str:
    """Compact signature of the currently selected laps in *df*."""
    return ",".join(str(int(lap)) for lap in available_laps(df))


def _run_color_map(run_names: list[str]) -> dict[str, str]:
    """Stable color per run for multi-run overlays."""
    return {
        run_name: RUN_COLORS[i % len(RUN_COLORS)]
        for i, run_name in enumerate(run_names)
    }


def _wheel_token(trace_name: str) -> str | None:
    """Extract wheel ID from a trace name when present."""
    for wheel in ("FL", "FR", "RL", "RR"):
        if trace_name == wheel or f" {wheel}" in trace_name or trace_name.endswith(f"- {wheel}"):
            return wheel
    return None


def _style_trace_for_run(
    trace: go.BaseTraceType,
    run_name: str,
    run_color: str,
    run_idx: int,
    variant_idx: int,
) -> go.BaseTraceType:
    """Clone *trace* and restyle it for a specific run overlay."""
    out = copy.deepcopy(trace)
    base_name = getattr(out, "name", "") or ""
    out.name = f"{run_name} · {base_name}" if base_name else run_name
    out.legendgroup = run_name

    dash = TRACE_DASHES[variant_idx % len(TRACE_DASHES)]
    symbol = TRACE_SYMBOLS[variant_idx % len(TRACE_SYMBOLS)]
    run_dash = TRACE_DASHES[run_idx % len(TRACE_DASHES)]
    run_symbol = TRACE_SYMBOLS[run_idx % len(TRACE_SYMBOLS)]
    wheel = _wheel_token(base_name)
    trace_color = WHEEL_COLORS[wheel] if wheel is not None else run_color
    trace_type = getattr(out, "type", "") or ""
    run_label = Path(run_name).stem
    is_ideal_share_line = base_name == "ideal (Fz = braking)"
    is_fz_reference_line = base_name == "Fz-proportional reference"
    mode = getattr(out, "mode", "") or ""
    is_wheel_line = (
        wheel is not None
        and trace_type in {"scatter", "scattergl"}
        and "lines" in mode
        and "markers" not in mode
    )

    if hasattr(out, "line") and out.line is not None and not is_ideal_share_line and not is_fz_reference_line:
        out.line.color = trace_color
        if trace_type != "box" and mode != "markers":
            out.line.dash = run_dash if is_wheel_line else dash
    elif hasattr(out, "line") and out.line is not None and is_fz_reference_line:
        out.line.dash = run_dash
    if hasattr(out, "marker") and out.marker is not None:
        out.marker.color = trace_color
        if wheel is not None:
            out.marker.line.color = run_color
            out.marker.line.width = 2
            if trace_type in {"scatter", "scattergl"}:
                out.marker.symbol = run_symbol
        elif getattr(out, "type", "") not in {"bar", "histogram", "barpolar", "bar3d"}:
            out.marker.symbol = symbol
    if hasattr(out, "textfont") and out.textfont is not None and wheel is not None:
        out.textfont.color = run_color

    x_vals = getattr(out, "x", None)
    if (
        wheel is not None
        and x_vals is not None
        and len(x_vals) > 0
        and all(isinstance(x_val, str) for x_val in x_vals)
    ):
        out.x = [f"{run_label} · {str(x_val)}" for x_val in x_vals]
    elif (
        trace_type == "box"
        and x_vals is not None
        and len(x_vals) > 0
        and all(isinstance(x_val, str) for x_val in x_vals)
    ):
        out.x = np.full(len(x_vals), run_label)
    elif (
        base_name == "Lockup threshold"
        and getattr(out, "x", None) is not None
    ):
        out.x = [f"{run_label} · {str(x)}" for x in out.x]
    elif (
        base_name == "Yaw instability threshold"
        and getattr(out, "x", None) is not None
    ):
        out.x = [run_label for _ in out.x]
    elif (
        base_name == "F/R summary"
        and getattr(out, "x", None) is not None
    ):
        out.x = [f"{run_label} · FR" for _ in out.x]
    return out


def _plotly_chart(fig: go.Figure, *args, **kwargs):
    """Render a Plotly figure through the dashboard's single chart wrapper."""
    return st.plotly_chart(fig, *args, **kwargs)


def _render_lap_detail_chart(
    fig: go.Figure,
    *,
    key: str,
    fullscreen_figures: dict[str, go.Figure] | None = None,
    fullscreen_selected: str | None = None,
    fullscreen_track_figure: go.Figure | None = None,
    fullscreen_gg_figures: dict[str, go.Figure] | None = None,
) -> None:
    """Render Lap Analysis detail with Video-Analysis-style crosshair labels."""
    fig_json = fig.to_json()
    fullscreen_payload = {
        str(label): json.loads(corner_fig.to_json())
        for label, corner_fig in (fullscreen_figures or {}).items()
    }
    if not fullscreen_payload:
        fullscreen_payload = {str(fullscreen_selected or "Current corner"): json.loads(fig_json)}
    fullscreen_track_payload = (
        json.loads(fullscreen_track_figure.to_json())
        if fullscreen_track_figure is not None
        else None
    )
    fullscreen_gg_payload = {
        str(label): json.loads(gg_fig.to_json())
        for label, gg_fig in (fullscreen_gg_figures or {}).items()
    }
    height_px = int(fig.layout.height or 900)
    component_id = f"lap_detail_{abs(hash(key))}"
    wrapper_id = f"{component_id}_wrap"
    component_id_js = json.dumps(component_id)
    wrapper_id_js = json.dumps(wrapper_id)
    fullscreen_selected_js = json.dumps(str(fullscreen_selected or next(iter(fullscreen_payload))))
    plotly_cdn_js = json.dumps(va._PLOTLY_CDN)
    escaped_fig_json = html.escape(fig_json, quote=False)
    escaped_fullscreen_json = html.escape(
        json.dumps(fullscreen_payload, ensure_ascii=False, allow_nan=False),
        quote=False,
    )
    escaped_fullscreen_track_json = html.escape(
        json.dumps(fullscreen_track_payload, ensure_ascii=False, allow_nan=False),
        quote=False,
    )
    escaped_fullscreen_gg_json = html.escape(
        json.dumps(fullscreen_gg_payload, ensure_ascii=False, allow_nan=False),
        quote=False,
    )
    component_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src={plotly_cdn_js}></script>
<style>
  html, body {{ margin:0; padding:0; background:#141417; overflow:hidden; }}
  #{wrapper_id} {{
    position:relative; width:100%; height:{height_px}px; background:#141417;
    overflow:hidden; border:1px solid rgba(255,255,255,0.08); border-radius:8px;
  }}
  .lap_detail_shell {{
    width:100%; height:100%; display:grid; grid-template-columns: minmax(0, 1fr);
    grid-template-rows: minmax(0, 1fr);
  }}
  .lap_detail_main {{ min-width:0; min-height:0; }}
  #{component_id} {{ width:100%; height:100%; }}
  .lap_detail_side {{
    display:none; min-width:0; min-height:0; grid-template-rows:minmax(0, 1fr) minmax(0, 1fr);
    gap:14px; padding:52px 12px 12px 0;
  }}
  .lap_detail_side_plot {{
    width:100%; height:100%; min-height:260px; background:#141417;
  }}
  #{wrapper_id}:fullscreen {{
    width:100vw; height:100vh; background:#141417; border-radius:0; border:none;
  }}
  #{wrapper_id}:fullscreen .lap_detail_shell {{
    grid-template-columns: minmax(0, 1.08fr) minmax(520px, 0.92fr);
    gap:16px; padding:12px;
  }}
  #{wrapper_id}:fullscreen .lap_detail_side {{ display:grid; }}
  .lap_detail_toolbar {{
    position:absolute; z-index:20; top:8px; left:10px;
    display:flex; align-items:center; gap:6px; padding:6px 8px;
    background:rgba(20,20,23,0.88); border:1px solid rgba(255,255,255,0.14);
    border-radius:6px; color:#EBEBEB; font:12px -apple-system, "Segoe UI", sans-serif;
  }}
  .lap_detail_toolbar button,
  .lap_detail_toolbar select {{
    background:#111318; color:#EBEBEB; border:1px solid rgba(255,255,255,0.18);
    border-radius:5px; padding:3px 6px; font-size:12px;
  }}
  .lap_detail_toolbar label {{ display:none; align-items:center; gap:6px; }}
  #{wrapper_id}:fullscreen .lap_detail_toolbar label {{ display:flex; }}
</style>
</head>
<body>
<script id="{component_id}_json" type="application/json">{escaped_fig_json}</script>
<script id="{component_id}_fullscreen_json" type="application/json">{escaped_fullscreen_json}</script>
<script id="{component_id}_fullscreen_track_json" type="application/json">{escaped_fullscreen_track_json}</script>
<script id="{component_id}_fullscreen_gg_json" type="application/json">{escaped_fullscreen_gg_json}</script>
<div id="{wrapper_id}">
  <div class="lap_detail_toolbar">
    <button id="{component_id}_fullscreen_btn" type="button">Full screen</button>
    <label>Inspect corner
      <select id="{component_id}_corner_select"></select>
    </label>
  </div>
  <div class="lap_detail_shell">
    <div class="lap_detail_main">
      <div id="{component_id}"></div>
    </div>
    <div class="lap_detail_side">
      <div id="{component_id}_track" class="lap_detail_side_plot"></div>
      <div id="{component_id}_gg" class="lap_detail_side_plot"></div>
    </div>
  </div>
</div>
<script>
(function() {{
  const CID = {component_id_js};
  const WID = {wrapper_id_js};
  const SELECTED_CORNER = {fullscreen_selected_js};
  const wrap = document.getElementById(WID);
  const gd = document.getElementById(CID);
  const trackGd = document.getElementById(CID + "_track");
  const ggGd = document.getElementById(CID + "_gg");
  const cornerSelect = document.getElementById(CID + "_corner_select");
  const fullscreenButton = document.getElementById(CID + "_fullscreen_btn");
  const fullscreenFigures = JSON.parse(document.getElementById(CID + "_fullscreen_json").textContent);
  const fullscreenTrackFigure = JSON.parse(document.getElementById(CID + "_fullscreen_track_json").textContent);
  const fullscreenGgFigures = JSON.parse(document.getElementById(CID + "_fullscreen_gg_json").textContent);
  let fig = JSON.parse(document.getElementById(CID + "_json").textContent);
  let baseShapes = [];
  let baseAnnotations = [];
  let selectedDistanceM = null;
  let currentTrackFig = null;
  let currentGgFig = null;
  let cursorRaf = 0;
  let pendingCursorPoint = null;
  let lastCursorXVal = null;

  const fullscreenIcon = {{
    width: 1000,
    height: 1000,
    path: "M60 360V60H360V160H160V360Z M640 60H940V360H840V160H640Z M160 640V840H360V940H60V640Z M840 640V840H640V940H940V640Z",
  }};
  function targetHeight() {{
    return document.fullscreenElement === wrap ? window.innerHeight - 24 : {height_px};
  }}
  function resizePlotForMode() {{
    Plotly.relayout(gd, {{ height: targetHeight(), autosize: true }});
    if (trackGd && trackGd.data) Plotly.Plots.resize(trackGd);
    if (ggGd && ggGd.data) Plotly.Plots.resize(ggGd);
  }}
  function toggleFullscreen() {{
    if (document.fullscreenElement === wrap) {{
      document.exitFullscreen();
      return;
    }}
    if (wrap.requestFullscreen) wrap.requestFullscreen();
  }}
  const config = {{
    displaylogo: false,
    responsive: true,
    scrollZoom: true,
    doubleClick: "reset+autosize",
    modeBarButtonsToRemove: [
      "lasso2d", "select2d", "toggleSpikelines",
      "hoverClosestCartesian", "hoverCompareCartesian"
    ],
    modeBarButtonsToAdd: [{{
      name: "fullscreen",
      title: "Full screen charts",
      icon: fullscreenIcon,
      click: toggleFullscreen,
    }}],
  }};

  function prepareFigure(nextFig) {{
    fig = JSON.parse(JSON.stringify(nextFig));
    fig.layout = fig.layout || {{}};
    fig.layout.height = targetHeight();
    fig.layout.autosize = true;
    fig.layout.margin = Object.assign({{}}, fig.layout.margin || {{}}, {{ autoexpand: false }});
    fig.layout.showlegend = false;
    fig.layout.hovermode = false;
    fig.layout.dragmode = "pan";
    fig.data = Array.isArray(fig.data) ? fig.data : [];
    fig.data.forEach(trace => {{
      trace.hoverinfo = "skip";
      trace.hovertemplate = null;
    }});
    baseShapes = Array.isArray(fig.layout.shapes) ? fig.layout.shapes.slice() : [];
    baseAnnotations = Array.isArray(fig.layout.annotations) ? fig.layout.annotations.slice() : [];
    pendingCursorPoint = null;
    lastCursorXVal = null;
    if (cursorRaf) {{
      cancelAnimationFrame(cursorRaf);
      cursorRaf = 0;
    }}
  }}

  function populateCornerSelect() {{
    const labels = Object.keys(fullscreenFigures);
    cornerSelect.innerHTML = "";
    labels.forEach(label => {{
      const option = document.createElement("option");
      option.value = label;
      option.textContent = label;
      cornerSelect.appendChild(option);
    }});
    cornerSelect.value = fullscreenFigures[SELECTED_CORNER] ? SELECTED_CORNER : labels[0];
    cornerSelect.disabled = labels.length <= 1;
  }}

  function sidePlotConfig() {{
    return {{
      displaylogo: false,
      responsive: true,
      scrollZoom: true,
      displayModeBar: false,
    }};
  }}

  function customColumn(raw, col) {{
    if (Array.isArray(raw)) {{
      return raw.map(value => Array.isArray(value) ? Number(value[col || 0]) : Number(value));
    }}
    if (!raw || typeof raw !== "object") return [];
    const flat = arrayFrom(raw);
    if (!flat.length) return [];
    const shape = Array.isArray(raw.shape)
      ? raw.shape
      : (typeof raw.shape === "string" ? raw.shape.split(",").map(value => Number(value.trim())) : []);
    const width = shape.length >= 2 ? Number(shape[1]) : 1;
    if (width <= 1) return flat.map(value => Number(value));
    const out = [];
    for (let i = col || 0; i < flat.length; i += width) out.push(Number(flat[i]));
    return out;
  }}

  function nearestPointInTrace(trace, distanceM) {{
    if (!trace || !isFinite(distanceM)) return null;
    if (trace.visible === false || trace.visible === "legendonly") return null;
    const xArr = arrayFrom(trace.x);
    const yArr = arrayFrom(trace.y);
    const sArr = customColumn(trace.customdata, 0);
    const n = Math.min(xArr.length, yArr.length, sArr.length);
    let best = null;
    let bestDiff = Infinity;
    for (let i = 0; i < n; i++) {{
      const x = Number(xArr[i]);
      const y = Number(yArr[i]);
      const s = Number(sArr[i]);
      if (!isFinite(x) || !isFinite(y) || !isFinite(s)) continue;
      const diff = Math.abs(s - distanceM);
      if (diff < bestDiff) {{
        bestDiff = diff;
        best = {{ x, y, s }};
      }}
    }}
    return best;
  }}

  function nearestByDistanceTrace(data, distanceM) {{
    if (!Array.isArray(data) || !isFinite(distanceM)) return null;
    let best = null;
    let bestDiff = Infinity;
    data.forEach(trace => {{
      const point = nearestPointInTrace(trace, distanceM);
      if (!point) return;
      const diff = Math.abs(point.s - distanceM);
      if (diff < bestDiff) {{
        bestDiff = diff;
        best = point;
      }}
    }});
    return best;
  }}

  function nearestByVisibleTrace(data, distanceM) {{
    if (!Array.isArray(data) || !isFinite(distanceM)) return [];
    const points = [];
    data.forEach((trace, idx) => {{
      const point = nearestPointInTrace(trace, distanceM);
      if (!point) return;
      points.push({{
        x: point.x,
        y: point.y,
        s: point.s,
        traceIndex: idx,
        traceName: String(trace.name || ""),
      }});
    }});
    return points;
  }}

  function markerTrace(point, name, color) {{
    if (!point) return null;
    return {{
      x: [point.x],
      y: [point.y],
      mode: "markers",
      name: name,
      showlegend: false,
      hovertemplate: "Selected point<br>Distance %{{customdata[0]:.1f}} m<extra></extra>",
      customdata: [[point.s]],
      marker: {{
        color: color || "#FFFFFF",
        size: 13,
        symbol: "circle",
        line: {{ color: "#111318", width: 2 }},
      }},
    }};
  }}

  function figureWithSelection(baseFig, markerName) {{
    if (!baseFig) return {{ data: [], layout: {{}} }};
    const nextData = Array.isArray(baseFig.data) ? baseFig.data.slice() : [];
    const marker = markerTrace(
      nearestByDistanceTrace(nextData, selectedDistanceM),
      markerName,
      "#FFFFFF",
    );
    if (marker) nextData.push(marker);
    return {{ data: nextData, layout: baseFig.layout || {{}} }};
  }}

  function ggFigureWithSelection(baseFig) {{
    if (!baseFig) return {{ data: [], layout: {{}} }};
    const nextData = Array.isArray(baseFig.data) ? baseFig.data.slice() : [];
    const markerColors = ["#FFFFFF", "#73D973"];
    nearestByVisibleTrace(nextData, selectedDistanceM).forEach((point, idx) => {{
      const markerName = point.traceName
        ? ("Selected GG point · " + point.traceName)
        : ("Selected GG point " + String(idx + 1));
      const marker = markerTrace(
        point,
        markerName,
        markerColors[idx] || "#FFFFFF",
      );
      if (marker) nextData.push(marker);
    }});
    return {{ data: nextData, layout: baseFig.layout || {{}} }};
  }}

  function renderSelectionMarkers() {{
    if (trackGd && currentTrackFig) {{
      const nextTrack = figureWithSelection(currentTrackFig, "Selected car position");
      Plotly.react(trackGd, nextTrack.data, nextTrack.layout, sidePlotConfig()).then(() => {{
        Plotly.Plots.resize(trackGd);
      }});
    }}
    if (ggGd && currentGgFig) {{
      const nextGg = ggFigureWithSelection(currentGgFig);
      Plotly.react(ggGd, nextGg.data, nextGg.layout, sidePlotConfig()).then(() => {{
        Plotly.Plots.resize(ggGd);
      }});
    }}
  }}

  function renderTrackFigure() {{
    if (!trackGd || !fullscreenTrackFigure) return;
    const trackFig = JSON.parse(JSON.stringify(fullscreenTrackFigure));
    trackFig.layout = trackFig.layout || {{}};
    trackFig.layout.autosize = true;
    currentTrackFig = trackFig;
    const nextTrack = figureWithSelection(currentTrackFig, "Selected car position");
    Plotly.react(trackGd, nextTrack.data, nextTrack.layout, sidePlotConfig()).then(() => {{
      Plotly.Plots.resize(trackGd);
    }});
  }}

  function renderGgFigure(label) {{
    if (!ggGd) return;
    const ggFig = fullscreenGgFigures[label];
    if (!ggFig) {{
      Plotly.purge(ggGd);
      ggGd.innerHTML = "";
      return;
    }}
    const nextFig = JSON.parse(JSON.stringify(ggFig));
    nextFig.layout = nextFig.layout || {{}};
    nextFig.layout.autosize = true;
    currentGgFig = nextFig;
    const nextGg = ggFigureWithSelection(currentGgFig);
    Plotly.react(ggGd, nextGg.data, nextGg.layout, sidePlotConfig()).then(() => {{
      Plotly.Plots.resize(ggGd);
    }});
  }}

  function changeCorner(label) {{
    const nextFig = fullscreenFigures[label];
    if (!nextFig) return;
    prepareFigure(nextFig);
    Plotly.react(gd, fig.data, fig.layout, config).then(resizePlotForMode);
    renderGgFigure(label);
  }}

  function arrayFrom(raw) {{
    if (Array.isArray(raw)) return raw;
    if (!raw || typeof raw !== "object" || typeof raw.bdata !== "string") return [];
    const dtype = String(raw.dtype || "");
    let bytes;
    try {{
      const binary = atob(raw.bdata);
      bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    }} catch (e) {{
      return [];
    }}
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    let typed;
    try {{
      if (dtype === "f8") typed = new Float64Array(buffer);
      else if (dtype === "f4") typed = new Float32Array(buffer);
      else if (dtype === "i4") typed = new Int32Array(buffer);
      else if (dtype === "u4") typed = new Uint32Array(buffer);
      else if (dtype === "i2") typed = new Int16Array(buffer);
      else if (dtype === "u2") typed = new Uint16Array(buffer);
      else if (dtype === "i1") typed = new Int8Array(buffer);
      else if (dtype === "u1") typed = new Uint8Array(buffer);
      else if (dtype === "i8") typed = new BigInt64Array(buffer);
      else if (dtype === "u8") typed = new BigUint64Array(buffer);
      else return [];
    }} catch (e) {{
      return [];
    }}
    return Array.from(typed, value => Number(value));
  }}

  function axisLayoutName(axisRef, prefix) {{
    const ref = axisRef || prefix;
    return ref === prefix ? prefix + "axis" : prefix + "axis" + ref.slice(1);
  }}

  function screenYFromData(yref, yVal) {{
    const fullLayout = gd._fullLayout;
    if (!fullLayout) return null;
    const axis = fullLayout[axisLayoutName(yref, "y")];
    if (!axis || typeof axis.l2p !== "function") return null;
    const pixel = Number(axis.l2p(yVal));
    return isFinite(pixel) ? pixel : null;
  }}

  function annotationOffsets(entries, nearRightEdge) {{
    const yShifts = Array(entries.length).fill(0);
    const xShifts = Array(entries.length).fill(nearRightEdge ? -6 : 6);
    const groups = new Map();
    entries.forEach((entry, idx) => {{
      const screenY = screenYFromData(entry.yref, entry.y);
      const key = entry.yref || "y";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push({{ idx, screenY }});
    }});
    const minGapPx = 24;
    const xStepPx = 42;

    function applyCluster(cluster) {{
      if (!cluster.length) return;
      if (cluster.length === 1) {{
        yShifts[cluster[0].idx] = 0;
        return;
      }}
      const center = cluster.reduce((sum, item) => sum + item.screenY, 0) / cluster.length;
      const start = center - (minGapPx * (cluster.length - 1)) / 2;
      cluster.forEach((item, order) => {{
        yShifts[item.idx] = start + order * minGapPx - item.screenY;
        xShifts[item.idx] = (nearRightEdge ? -6 : 6) + (nearRightEdge ? -1 : 1) * order * xStepPx;
      }});
    }}

    groups.forEach(group => {{
      const valid = group
        .filter(item => item.screenY !== null && isFinite(item.screenY))
        .sort((a, b) => a.screenY - b.screenY);
      let cluster = [];
      valid.forEach(item => {{
        if (!cluster.length) {{
          cluster = [item];
          return;
        }}
        if (item.screenY - cluster[cluster.length - 1].screenY < minGapPx) {{
          cluster.push(item);
          return;
        }}
        applyCluster(cluster);
        cluster = [item];
      }});
      applyCluster(cluster);
    }});
    return {{ yShifts, xShifts }};
  }}

  function plotAreaXFromEvent(event) {{
    const fl = gd._fullLayout;
    if (!fl || !fl.xaxis || !Array.isArray(fl.xaxis.range)) return null;
    const rect = gd.getBoundingClientRect();
    const margin = fl.margin || {{}};
    const width = Number.isFinite(fl.width) ? fl.width : rect.width;
    const height = Number.isFinite(fl.height) ? fl.height : rect.height;
    const leftMargin = Number.isFinite(margin.l) ? margin.l : 0;
    const rightMargin = Number.isFinite(margin.r) ? margin.r : 0;
    const topMargin = Number.isFinite(margin.t) ? margin.t : 0;
    const bottomMargin = Number.isFinite(margin.b) ? margin.b : 0;
    const plotWidth = width - leftMargin - rightMargin;
    const plotHeight = height - topMargin - bottomMargin;
    const domain = Array.isArray(fl.xaxis.domain) ? fl.xaxis.domain : [0, 1];
    const x0Px = leftMargin + domain[0] * plotWidth;
    const x1Px = leftMargin + domain[1] * plotWidth;
    const y0Px = topMargin;
    const y1Px = topMargin + plotHeight;
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    if (px < x0Px || px > x1Px || py < y0Px || py > y1Px) return null;
    const frac = Math.max(0, Math.min(1, (px - x0Px) / Math.max(1, x1Px - x0Px)));
    const r0 = Number(fl.xaxis.range[0]);
    const r1 = Number(fl.xaxis.range[1]);
    if (!isFinite(r0) || !isFinite(r1)) return null;
    return r0 + frac * (r1 - r0);
  }}

  function valueAtX(xArr, yArr, xVal) {{
    xArr = arrayFrom(xArr);
    yArr = arrayFrom(yArr);
    if (!Array.isArray(xArr) || !Array.isArray(yArr)) return NaN;
    const n = Math.min(xArr.length, yArr.length);
    let first = -1;
    let last = -1;
    for (let i = 0; i < n; i++) {{
      const x = Number(xArr[i]);
      const y = Number(yArr[i]);
      if (isFinite(x) && isFinite(y)) {{
        if (first < 0) first = i;
        last = i;
      }}
    }}
    if (first < 0) return NaN;
    const xFirst = Number(xArr[first]);
    const xLast = Number(xArr[last]);
    if (xVal <= xFirst) return Number(yArr[first]);
    if (xVal >= xLast) return Number(yArr[last]);
    for (let i = first; i < last; i++) {{
      const x0 = Number(xArr[i]);
      const x1 = Number(xArr[i + 1]);
      const y0 = Number(yArr[i]);
      const y1 = Number(yArr[i + 1]);
      if (!isFinite(x0) || !isFinite(x1) || !isFinite(y0) || !isFinite(y1)) continue;
      const inside = (x0 <= xVal && xVal <= x1) || (x1 <= xVal && xVal <= x0);
      if (!inside) continue;
      if (x1 === x0) return y0;
      const f = (xVal - x0) / (x1 - x0);
      return y0 + f * (y1 - y0);
    }}
    return NaN;
  }}

  function distanceAtGraphX(xVal) {{
    if (!isFinite(xVal)) return NaN;
    for (const trace of fig.data) {{
      if (trace.visible === false || trace.visible === "legendonly") continue;
      const xArr = arrayFrom(trace.x);
      const sArr = customColumn(trace.customdata, 0);
      const n = Math.min(xArr.length, sArr.length);
      if (n < 2) continue;
      let first = -1;
      let last = -1;
      for (let i = 0; i < n; i++) {{
        if (isFinite(Number(xArr[i])) && isFinite(Number(sArr[i]))) {{
          if (first < 0) first = i;
          last = i;
        }}
      }}
      if (first < 0) continue;
      const xFirst = Number(xArr[first]);
      const xLast = Number(xArr[last]);
      if (xVal <= xFirst) return Number(sArr[first]);
      if (xVal >= xLast) return Number(sArr[last]);
      for (let i = first; i < last; i++) {{
        const x0 = Number(xArr[i]);
        const x1 = Number(xArr[i + 1]);
        const s0 = Number(sArr[i]);
        const s1 = Number(sArr[i + 1]);
        if (!isFinite(x0) || !isFinite(x1) || !isFinite(s0) || !isFinite(s1)) continue;
        const inside = (x0 <= xVal && xVal <= x1) || (x1 <= xVal && xVal <= x0);
        if (!inside) continue;
        if (x1 === x0) return s0;
        const f = (xVal - x0) / (x1 - x0);
        return s0 + f * (s1 - s0);
      }}
    }}
    return xVal;
  }}

  function colorForTrace(trace) {{
    if (trace && trace.line && trace.line.color) return trace.line.color;
    if (trace && trace.marker && trace.marker.color) return trace.marker.color;
    return "#FFFFFF";
  }}

  function formatValue(value) {{
    if (!isFinite(value)) return "";
    const absValue = Math.abs(value);
    if (absValue >= 100) return value.toFixed(1);
    if (absValue >= 10) return value.toFixed(2);
    return value.toFixed(3);
  }}

  function subplotRefs() {{
    const refs = [];
    fig.data.forEach(trace => {{
      const yref = trace.yaxis || "y";
      const xref = trace.xaxis || "x";
      if (!refs.some(item => item.yref === yref)) refs.push({{ xref, yref }});
    }});
    refs.sort((a, b) => {{
      const ay = gd._fullLayout[axisLayoutName(a.yref, "y")];
      const by = gd._fullLayout[axisLayoutName(b.yref, "y")];
      const ad = ay && Array.isArray(ay.domain) ? ay.domain[1] : 0;
      const bd = by && Array.isArray(by.domain) ? by.domain[1] : 0;
      return bd - ad;
    }});
    return refs;
  }}

  function renderCursor(clientX, clientY) {{
    const xVal = plotAreaXFromEvent({{ clientX, clientY }});
    if (xVal === null || !isFinite(xVal)) return;
    if (lastCursorXVal !== null && Math.abs(lastCursorXVal - xVal) < 1e-6) return;
    lastCursorXVal = xVal;
    const shapes = baseShapes.slice();
    subplotRefs().forEach(ref => {{
      shapes.push({{
        type: "line",
        xref: ref.xref,
        yref: ref.yref + " domain",
        x0: xVal,
        x1: xVal,
        y0: 0,
        y1: 1,
        line: {{ color: "rgba(255,255,255,0.85)", width: 1, dash: "dot" }},
      }});
    }});

    const annotations = baseAnnotations.slice();
    const fullX = gd._fullLayout && gd._fullLayout.xaxis;
    const xrange = fullX && Array.isArray(fullX.range) ? fullX.range : null;
    const xFrac = xrange && isFinite(Number(xrange[0])) && isFinite(Number(xrange[1])) && Number(xrange[1]) !== Number(xrange[0])
      ? (xVal - Number(xrange[0])) / (Number(xrange[1]) - Number(xrange[0]))
      : 0.5;
    const nearRightEdge = xFrac > 0.88;
    const entries = [];
    fig.data.forEach(trace => {{
      if (trace.visible === false || trace.visible === "legendonly") return;
      const y = valueAtX(trace.x, trace.y, xVal);
      if (!isFinite(y)) return;
      const color = colorForTrace(trace);
      const xref = trace.xaxis || "x";
      const yref = trace.yaxis || "y";
      entries.push({{
        x: xVal,
        y: y,
        xref: xref,
        yref: yref,
        text: formatValue(y),
        color: color,
      }});
    }});
    const offsets = annotationOffsets(entries, nearRightEdge);
    entries.forEach((entry, idx) => {{
      annotations.push({{
        x: entry.x,
        y: entry.y,
        xref: entry.xref,
        yref: entry.yref,
        text: entry.text,
        showarrow: false,
        xanchor: nearRightEdge ? "right" : "left",
        yanchor: "middle",
        xshift: offsets.xShifts[idx],
        yshift: offsets.yShifts[idx],
        bgcolor: "rgba(20,20,23,0.96)",
        bordercolor: entry.color,
        borderwidth: 1,
        borderpad: 2,
        font: {{ color: entry.color, size: 11 }},
      }});
    }});
    Plotly.relayout(gd, {{ shapes: shapes, annotations: annotations }});
  }}

  function updateCursor(event) {{
    pendingCursorPoint = {{
      clientX: event.clientX,
      clientY: event.clientY,
    }};
    if (cursorRaf) return;
    cursorRaf = requestAnimationFrame(() => {{
      cursorRaf = 0;
      const point = pendingCursorPoint;
      pendingCursorPoint = null;
      if (!point) return;
      renderCursor(point.clientX, point.clientY);
    }});
  }}

  function selectCursor(event) {{
    if (event.button !== 0) return;
    const xVal = plotAreaXFromEvent(event);
    if (xVal === null || !isFinite(xVal)) return;
    const distanceM = distanceAtGraphX(xVal);
    if (!isFinite(distanceM)) return;
    selectedDistanceM = distanceM;
    renderCursor(event.clientX, event.clientY);
    renderSelectionMarkers();
  }}

  function clearCursor() {{
    if (selectedDistanceM !== null) return;
    pendingCursorPoint = null;
    lastCursorXVal = null;
    if (cursorRaf) {{
      cancelAnimationFrame(cursorRaf);
      cursorRaf = 0;
    }}
    Plotly.relayout(gd, {{ shapes: baseShapes, annotations: baseAnnotations }});
  }}

  populateCornerSelect();
  prepareFigure(fig);
  Plotly.newPlot(gd, fig.data, fig.layout, config).then(() => {{
    gd.addEventListener("mousemove", updateCursor);
    gd.addEventListener("mousedown", selectCursor);
    gd.addEventListener("mouseleave", clearCursor);
    document.addEventListener("fullscreenchange", resizePlotForMode);
    window.addEventListener("resize", resizePlotForMode);
    cornerSelect.addEventListener("change", event => changeCorner(event.target.value));
    fullscreenButton.addEventListener("click", toggleFullscreen);
    renderTrackFigure();
    renderGgFigure(cornerSelect.value);
  }});
}})();
</script>
</body>
</html>"""
    components.html(component_html, height=height_px + 8, scrolling=False)


def _overlay_figures(figs_by_run: dict[str, list[go.Figure]]) -> list[go.Figure]:
    """Overlay equally-indexed figures from multiple runs into one figure list."""
    if not figs_by_run:
        return []

    run_names = list(figs_by_run.keys())
    run_colors = _run_color_map(run_names)
    fig_count = len(next(iter(figs_by_run.values())))
    merged: list[go.Figure] = []

    for fig_idx in range(fig_count):
        base = go.Figure(next(iter(figs_by_run.values()))[fig_idx])
        base.data = ()
        variant_map: dict[str, int] = {}
        tickvals_by_axis: dict[str, set[float]] = {}

        for run_idx, run_name in enumerate(run_names):
            fig = figs_by_run[run_name][fig_idx]
            for trace_idx, trace in enumerate(fig.data):
                variant_key = getattr(trace, "name", None) or f"trace_{trace_idx}"
                if variant_key not in variant_map:
                    variant_map[variant_key] = len(variant_map)
                base.add_trace(
                    _style_trace_for_run(
                        trace,
                        run_name,
                        run_colors[run_name],
                        run_idx,
                        variant_map[variant_key],
                    )
                )

            for axis_name in fig.layout:
                if not str(axis_name).startswith("xaxis"):
                    continue
                tickvals = getattr(fig.layout[axis_name], "tickvals", None)
                if tickvals is None:
                    continue
                tickvals_by_axis.setdefault(str(axis_name), set()).update(float(v) for v in tickvals)

        for axis_name, tickvals in tickvals_by_axis.items():
            if tickvals:
                base.layout[axis_name].tickvals = sorted(tickvals)
        if any(getattr(t, "type", "") == "histogram" for t in base.data):
            base.update_layout(barmode="overlay")
            for trace in base.data:
                if getattr(trace, "type", "") == "histogram":
                    trace.opacity = min(getattr(trace, "opacity", 1.0) or 1.0, 0.65)
        merged.append(base)

    return merged


def _concat_run_tables(run_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Concatenate per-run tables, adding a `Run` column when needed."""
    tables: list[pl.DataFrame] = []
    for run_name, table in run_tables.items():
        if "Run" not in table.columns:
            table = table.with_columns(pl.lit(run_name).alias("Run"))
        tables.append(table)
    return pl.concat(tables, how="vertical_relaxed") if tables else pl.DataFrame()


def _show_summary_table(rows: list[dict[str, object]]) -> None:
    """Render a compact per-run summary table when multiple CSVs are loaded."""
    if rows:
        st.dataframe(pl.DataFrame(rows), use_container_width=True, hide_index=True)


def _run_cache_tokens(dfs: dict[str, pl.DataFrame]) -> tuple[tuple[str, FileSignature, str], ...]:
    """Stable cache tokens for current runs and selected laps."""
    file_signatures = st.session_state.get("_run_file_signatures", {})
    return tuple(
        (
            run_name,
            file_signatures.get(run_name, (0, 0)),
            _lap_signature(df),
        )
        for run_name, df in dfs.items()
    )


def _driver_run_tokens(
    dfs: dict[str, pl.DataFrame],
    file_signatures: dict[str, FileSignature],
) -> tuple[tuple[str, FileSignature, str], ...]:
    """Stable cache tokens for the current driver-tab runs and selected laps."""
    return tuple(
        (
            run_name,
            file_signatures[run_name],
            _lap_signature(df),
        )
        for run_name, df in dfs.items()
    )


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _pt_energy_per_lap_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return pt.energy_per_lap_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _pt_power_per_wheel_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return pt.power_per_wheel_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _pt_battery_status_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return pt.battery_status_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _pt_thermal_evolution_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return pt.thermal_evolution_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_ideal_braking_curve_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.ideal_braking_curve_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_decel_envelope_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.decel_envelope_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _skidpad_fig_cached(
    metric: str,
    df: pl.DataFrame,
    run_token: tuple[str, FileSignature, str],
) -> tuple[go.Figure, dict]:
    """Cached wrapper for single-run skidpad event figures."""
    _ = run_token
    funcs = {
        "event_time": skidpad.event_time_summary_fig,
        "lateral_g": skidpad.lateral_g_fig,
        "driven_radius": skidpad.driven_radius_fig,
        "balance": skidpad.balance_fig,
        "driver_smoothness": skidpad.driver_smoothness_fig,
        "gps_figure8": skidpad.gps_figure8_fig,
        "tv_intervention": skidpad.tv_intervention_fig,
        "lateral_load_dist": skidpad.lateral_load_dist_fig,
    }
    if metric not in funcs:
        raise KeyError(f"Unknown skidpad metric: {metric}")
    return funcs[metric](df)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_braking_stability_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.braking_stability_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_braking_stability_per_lap_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.braking_stability_per_lap_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_ideal_traction_curve_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.ideal_traction_curve_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_accel_envelope_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.accel_envelope_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_summary_cached(
    df: pl.DataFrame,
    run_token: tuple[str, FileSignature, str],
) -> dict:
    """Cached wrapper for per-run driver summary metrics."""
    _ = run_token
    return drv.driver_summary(df)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_throttle_histogram_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> go.Figure:
    """Cached wrapper for the driver throttle histogram."""
    _ = run_tokens
    return drv.throttle_histogram_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_full_throttle_time_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> go.Figure:
    """Cached wrapper for full-throttle per-lap figure."""
    _ = run_tokens
    return drv.full_throttle_time_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_throttle_speed_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> go.Figure:
    """Cached wrapper for throttle-speed per-lap figure."""
    _ = run_tokens
    return drv.throttle_speed_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_braking_effort_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> go.Figure:
    """Cached wrapper for braking-effort figure."""
    _ = run_tokens
    return drv.braking_effort_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_brake_application_point_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    turns_signature: tuple,
) -> go.Figure:
    """Cached wrapper for brake-application-point figure."""
    _ = run_tokens
    turns = [
        corn.TurnDef(
            turn_id=int(t[0]),
            s_entry_m=float(t[1]),
            s_apex_m=float(t[2]),
            s_exit_m=float(t[3]),
            apex_lat=float(t[4]),
            apex_lng=float(t[5]),
            lat=np.array([], dtype=float),
            lng=np.array([], dtype=float),
        )
        for t in turns_signature
    ]
    return drv.brake_application_point_fig(dfs, turns=turns)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_braking_aggressiveness_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> go.Figure:
    """Cached wrapper for braking-aggressiveness per-lap figure."""
    _ = run_tokens
    return drv.braking_aggressiveness_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_brake_release_smoothness_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> go.Figure:
    """Cached wrapper for brake-release smoothness per-lap figure."""
    _ = run_tokens
    return drv.brake_release_smoothness_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_steering_smoothness_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> go.Figure:
    """Cached wrapper for steering-smoothness figure."""
    _ = run_tokens
    return drv.steering_smoothness_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_steering_integral_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> go.Figure:
    """Cached wrapper for steering-integral figure."""
    _ = run_tokens
    return drv.steering_integral_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_steering_stability_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    turns_signature: tuple,
) -> go.Figure:
    """Cached wrapper for steering-stability box-plot figure."""
    _ = run_tokens
    turns = [
        corn.TurnDef(
            turn_id=int(t[0]),
            s_entry_m=float(t[1]),
            s_apex_m=float(t[2]),
            s_exit_m=float(t[3]),
            apex_lat=float(t[4]),
            apex_lng=float(t[5]),
            lat=np.array([], dtype=float),
            lng=np.array([], dtype=float),
        )
        for t in turns_signature
    ]
    return drv.steering_stability_fig(dfs, turns=turns)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_corner_curvature_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> go.Figure:
    """Cached wrapper for corner-curvature per-lap figure."""
    _ = run_tokens
    return drv.corner_curvature_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_circuit_map_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    selected_map: tuple[tuple[str, int], ...],
) -> go.Figure:
    """Cached wrapper for driver circuit map figure."""
    _ = run_tokens
    return drv.circuit_map_fig(dfs, list(selected_map))


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_circuit_map_stats_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    selected_map: tuple[tuple[str, int], ...],
) -> pl.DataFrame:
    """Cached wrapper for driver circuit map phase tables."""
    _ = run_tokens
    return drv.circuit_map_stats(dfs, list(selected_map))


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_lap_time_progression_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> go.Figure:
    """Cached wrapper for the lap-time progression figure."""
    _ = run_tokens
    return drv.lap_time_progression_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_lap_consistency_stats_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> pl.DataFrame:
    """Cached wrapper for lap-time consistency stats."""
    _ = run_tokens
    return drv.lap_consistency_stats(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_lap_time_distribution_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> go.Figure:
    """Cached wrapper for the lap-time distribution box plot."""
    _ = run_tokens
    return drv.lap_time_distribution_fig(dfs)


def _is_skidpad_lap(df: pl.DataFrame, lap_id: int) -> bool:
    """True if the given lap was logged in skidpad event mode."""
    if "lapcount_mode" not in df.columns or "laps" not in df.columns:
        return False
    sub = df.filter(pl.col("laps") == int(lap_id))
    if sub.is_empty():
        return False
    values = sub["lapcount_mode"].drop_nulls()
    if len(values) == 0:
        return False
    return str(values[0]).strip().lower() == "skidpad"


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_cornering_turns_cached(
    dfs,
    driver_run_tokens,
    R_thr_m,
    min_dur_s,
    merge_gap_m,
    ref_run,
    ref_lap,
) -> list:
    """Cached turn detection for the selected cornering reference lap."""
    _ = driver_run_tokens
    d = corn.compute_radius_curvature(dfs[ref_run])
    if _is_skidpad_lap(dfs[ref_run], int(ref_lap)):
        return corn.detect_skidpad_turn_on_lap(d, int(ref_lap))
    return corn.detect_turns_on_lap(
        d,
        ref_run,
        int(ref_lap),
        R_thr_m=float(R_thr_m),
        min_dur_s=float(min_dur_s),
        merge_gap_m=float(merge_gap_m),
    )


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_cornering_metrics_cached(
    dfs,
    driver_run_tokens,
    turns_signature,
) -> pl.DataFrame:
    """Cached cornering metrics projected onto all selected driver laps."""
    _ = driver_run_tokens
    turns = [
        corn.TurnDef(
            turn_id=int(t[0]),
            s_entry_m=float(t[1]),
            s_apex_m=float(t[2]),
            s_exit_m=float(t[3]),
            apex_lat=float(t[4]),
            apex_lng=float(t[5]),
            lat=np.array([], dtype=float),
            lng=np.array([], dtype=float),
        )
        for t in turns_signature
    ]
    return corn.compute_turn_metrics(dfs, turns)


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_fastest_lap_cached(
    dfs,
    driver_run_tokens,
    run_name: str,
) -> int | None:
    """Cached fastest valid lap for per-CSV sector summaries."""
    _ = driver_run_tokens
    return lsec.fastest_valid_lap(dfs[run_name])


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_lap_sectors_cached(
    dfs,
    driver_run_tokens,
    run_name: str,
    turns_signature: tuple,
    lap_end_m_token: float,
) -> list[lsec.Sector]:
    """Cached sectorization for one run's fastest valid lap."""
    _ = (dfs, driver_run_tokens, run_name, lap_end_m_token)
    turns = [
        corn.TurnDef(
            turn_id=int(t[0]),
            s_entry_m=float(t[1]),
            s_apex_m=float(t[2]),
            s_exit_m=float(t[3]),
            apex_lat=float(t[4]),
            apex_lng=float(t[5]),
            lat=np.array([], dtype=float),
            lng=np.array([], dtype=float),
        )
        for t in turns_signature
    ]
    return lsec.build_sectors(turns, float(lap_end_m_token))


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_csv_sector_summary_cached(
    dfs,
    driver_run_tokens,
    run_name: str,
    sectors_token: tuple,
) -> dict[str, dict[str, float] | None]:
    """Cached best/avg/potential metrics for one CSV."""
    _ = driver_run_tokens
    sectors = [
        lsec.Sector(
            index=int(token[0]),
            kind=str(token[1]),
            s_start_m=float(token[2]),
            s_end_m=float(token[3]),
            turn_id=None if int(token[4]) < 0 else int(token[4]),
        )
        for token in sectors_token
    ]
    return lsec.csv_metrics_summary(dfs[run_name], sectors)


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_whole_lap_metrics_cached(
    dfs,
    driver_run_tokens,
    run_name: str,
    lap_id: int,
) -> dict[str, float] | None:
    """Cached whole-lap phase metrics for the selected reference/comparison lap."""
    _ = driver_run_tokens
    return lsec.whole_lap_metrics(dfs[run_name], int(lap_id))


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_potential_lap_cached(
    dfs,
    driver_run_tokens,
    sectors_token: tuple,
) -> tuple[pl.DataFrame, dict[str, object]] | None:
    """Cached stitched potential lap built from the loaded Driver CSVs."""
    _ = driver_run_tokens
    sectors = [
        lsec.Sector(
            index=int(token[0]),
            kind=str(token[1]),
            s_start_m=float(token[2]),
            s_end_m=float(token[3]),
            turn_id=None if int(token[4]) < 0 else int(token[4]),
        )
        for token in sectors_token
    ]
    return drv.potential_lap_from_sectors(dfs, sectors)


# ── Function-level checks (per-controller "is it doing its job?") ─────────────


def _render_tc_function_check(dfs: dict[str, pl.DataFrame]) -> None:
    st.subheader("Function check — is TC keeping SR ≈ +0.20 in acceleration?")
    all_figs: dict[str, list[go.Figure]] = {}
    rows = []
    for detail_name, detail_df in dfs.items():
        try:
            figs, kpis = tc.tc_function_kpis(detail_df)
        except Exception as exc:
            st.warning(f"{detail_name}: TC function check unavailable: {exc}")
            continue
        all_figs[detail_name] = figs
        rows.append({
            "Run": Path(detail_name).stem,
            "Target met": "YES" if kpis["objective_ok"] else "NO",
            "In target [%]": round(kpis["pct_in_target"], 1),
            "Median SR": round(kpis["median_sr"], 3),
            "Target gap [%]": round(kpis["target_gap_pct"], 1),
            "Failure mode": kpis["failure_mode"],
            "Too low SR [%]": round(kpis["pct_all_underslip"], 1),
            "Too high SR [%]": round(kpis["pct_any_overslip"], 1),
            "TC response [%]": round(kpis["pct_cut_when_overslip"], 1),
            "Worst wheel": kpis["worst_wheel"],
        })
        for note in kpis.get("notes", []):
            st.info(f"{Path(detail_name).stem}: {note}")
    if rows:
        _show_summary_table(rows)
    for fig in _overlay_figures(all_figs):
        _plotly_chart(fig, use_container_width=True, theme=None)


def _render_tc_control_impact(dfs: dict[str, pl.DataFrame]) -> None:
    st.divider()
    st.subheader("TC behaviour — overslip recovery versus acceleration loss")
    try:
        run_results = {
            run_name: tc.tc_control_impact_figs_kpis(df)
            for run_name, df in dfs.items()
        }
    except Exception as exc:
        st.warning(f"TC attribution unavailable: {exc}")
        return

    if len(dfs) == 1:
        figs, kpis = next(iter(run_results.values()))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overslip", f"{_fmt(kpis['overslip_pct'], '.1f')}%")
        c2.metric("Mean SR excess", _fmt(kpis["mean_sr_excess"], ".3f"))
        c3.metric("Recovery time", f"{_fmt(kpis['recovery_time_ms'], '.0f')} ms")
        c4.metric("Overslip events cut", f"{_fmt(kpis['pct_events_with_cut'], '.1f')}%")
        c5, c6, c7 = st.columns(3)
        c5.metric("Cut without overslip", f"{_fmt(kpis['cut_without_overslip_pct'], '.1f')}%")
        c6.metric("ax penalty by cut", f"{_fmt(kpis['ax_penalty_cut_ms2'], '+.2f')} m/s²")
        c7.metric("Samples checked", str(kpis["eval_samples"]))
        for fig in figs:
            _plotly_chart(fig, use_container_width=True, theme=None)
        with st.expander("Per-lap attribution"):
            st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        return

    rows = []
    figs_by_run: dict[str, list[go.Figure]] = {}
    tables: dict[str, pl.DataFrame] = {}
    for run_name, (figs, kpis) in run_results.items():
        figs_by_run[run_name] = figs
        tables[run_name] = kpis["table"]
        rows.append({
            "Run": Path(run_name).stem,
            "Overslip [%]": round(kpis["overslip_pct"], 1),
            "Mean SR excess": round(kpis["mean_sr_excess"], 3),
            "Recovery [ms]": round(kpis["recovery_time_ms"], 0),
            "Events cut [%]": round(kpis["pct_events_with_cut"], 1),
            "Cut w/o overslip [%]": round(kpis["cut_without_overslip_pct"], 1),
            "ax penalty [m/s²]": round(kpis["ax_penalty_cut_ms2"], 2),
            "Samples": int(kpis["eval_samples"]),
        })
    _show_summary_table(rows)
    for fig in _overlay_figures(figs_by_run):
        _plotly_chart(fig, use_container_width=True, theme=None)
    with st.expander("Per-lap attribution"):
        st.dataframe(_concat_run_tables(tables), use_container_width=True, hide_index=True)


def _render_tv_function_check(dfs: dict[str, pl.DataFrame]) -> None:
    st.divider()
    st.subheader("Function check — is TV adding yaw moment so the car turns?")
    all_figs: dict[str, list[go.Figure]] = {}
    for run_name, df in dfs.items():
        try:
            figs, _kpis = tv.tv_function_kpis(df)
        except Exception as exc:
            st.warning(f"{run_name}: TV function check unavailable: {exc}")
            continue
        all_figs[run_name] = figs
    for fig in _overlay_figures(all_figs):
        _plotly_chart(fig, use_container_width=True, theme=None)


def _render_tv_control_attribution(dfs: dict[str, pl.DataFrame]) -> None:
    st.divider()
    st.subheader("TV behaviour — rotation and understeer/oversteer balance")
    try:
        run_results = {
            run_name: tv.tv_control_attribution_figs_kpis(df)
            for run_name, df in dfs.items()
        }
    except Exception as exc:
        st.warning(f"TV attribution unavailable: {exc}")
        return

    if len(dfs) == 1:
        figs, kpis = next(iter(run_results.values()))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Yaw gain", _fmt(kpis["yaw_gain_median"], ".3f"))
        c2.metric("Understeer", f"{_fmt(kpis['understeer_pct'], '.1f')}%")
        c3.metric("Oversteer", f"{_fmt(kpis['oversteer_pct'], '.1f')}%")
        c4.metric("Balanced", f"{_fmt(kpis['balanced_pct'], '.1f')}%")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Entry rotation", f"{_fmt(kpis['entry_rotation_radss'], '.2f')} rad/s²")
        c6.metric("Mz ↔ balance", _fmt(kpis["mz_balance_corr"], "+.3f"))
        c7.metric("L/R torque ↔ balance", _fmt(kpis["lr_balance_corr"], "+.3f"))
        c8.metric("Corner samples", str(kpis["corner_samples"]))
        for fig in figs:
            _plotly_chart(fig, use_container_width=True, theme=None)
        with st.expander("Per-lap attribution"):
            st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        return

    rows = []
    figs_by_run: dict[str, list[go.Figure]] = {}
    tables: dict[str, pl.DataFrame] = {}
    for run_name, (figs, kpis) in run_results.items():
        figs_by_run[run_name] = figs
        tables[run_name] = kpis["table"]
        rows.append({
            "Run": Path(run_name).stem,
            "Yaw gain": round(kpis["yaw_gain_median"], 3),
            "Understeer [%]": round(kpis["understeer_pct"], 1),
            "Oversteer [%]": round(kpis["oversteer_pct"], 1),
            "Balanced [%]": round(kpis["balanced_pct"], 1),
            "Entry rotation [rad/s²]": round(kpis["entry_rotation_radss"], 2),
            "Mz-balance corr": round(kpis["mz_balance_corr"], 3),
            "LR-balance corr": round(kpis["lr_balance_corr"], 3),
        })
    _show_summary_table(rows)
    for fig in _overlay_figures(figs_by_run):
        _plotly_chart(fig, use_container_width=True, theme=None)
    with st.expander("Per-lap attribution"):
        st.dataframe(_concat_run_tables(tables), use_container_width=True, hide_index=True)


def _render_tv_corner_balance(dfs: dict[str, pl.DataFrame]) -> None:
    st.divider()
    st.subheader("TV corner balance — understeer / oversteer per curve")
    try:
        R_thr_m = float(st.session_state.get("drv_corner_R_thr", 60.0))
        min_dur_s = float(st.session_state.get("drv_corner_min_dur", 0.5))
        merge_gap_m = float(st.session_state.get("drv_corner_merge_gap", 8.0))
        ref_option = st.session_state.get("drv_corner_ref")
        ref_run: str
        ref_lap: int
        if (
            isinstance(ref_option, tuple)
            and len(ref_option) >= 2
            and str(ref_option[0]) in dfs
        ):
            ref_run = str(ref_option[0])
            ref_lap = int(ref_option[1])
        else:
            ref_run, ref_lap = corn.select_reference_lap(dfs)

        ref_d = corn.compute_radius_curvature(dfs[ref_run])
        reference_turns = corn.detect_turns_on_lap(
            ref_d,
            ref_run,
            int(ref_lap),
            R_thr_m=R_thr_m,
            min_dur_s=min_dur_s,
            merge_gap_m=merge_gap_m,
        )
        reference_label = (
            f"{Path(ref_run).stem} lap {int(ref_lap)}, "
            f"R<{R_thr_m:.0f} m, min {min_dur_s:.1f} s, merge {merge_gap_m:.0f} m"
        )
        st.caption(f"Using the same turn definition as Lap Analysis: {reference_label}")
        run_results = {
            run_name: tv.tv_corner_under_oversteer_figs_kpis(
                df,
                reference_turns,
                reference_label=reference_label,
            )
            for run_name, df in dfs.items()
        }
    except Exception as exc:
        st.warning(f"TV corner balance unavailable: {exc}")
        return

    if len(dfs) == 1:
        figs, kpis = next(iter(run_results.values()))
        for warning in kpis.get("warnings", []):
            st.warning(warning)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Corners", str(kpis["corners"]))
        c2.metric("Median balance", f"{_fmt(kpis['median_balance_pct'], '+.1f')}%")
        c3.metric("Understeer corners", f"{_fmt(kpis['understeer_corners_pct'], '.1f')}%")
        c4.metric("Oversteer corners", f"{_fmt(kpis['oversteer_corners_pct'], '.1f')}%")
        for fig in figs:
            _plotly_chart(fig, use_container_width=True, theme=None)
        with st.expander("Per-lap understeer / oversteer"):
            st.dataframe(kpis["lap_table"], use_container_width=True, hide_index=True)
        with st.expander("Per-corner understeer / oversteer"):
            st.dataframe(kpis["turn_table"], use_container_width=True, hide_index=True)
        return

    rows = []
    lap_tables: dict[str, pl.DataFrame] = {}
    turn_tables: dict[str, pl.DataFrame] = {}
    for run_name, (figs, kpis) in run_results.items():
        for warning in kpis.get("warnings", []):
            st.warning(f"{run_name}: {warning}")
        lap_tables[run_name] = kpis["lap_table"]
        turn_tables[run_name] = kpis["turn_table"]
        rows.append({
            "Run": Path(run_name).stem,
            "Corners": int(kpis["corners"]),
            "Median balance [%]": round(kpis["median_balance_pct"], 1),
            "Understeer corners [%]": round(kpis["understeer_corners_pct"], 1),
            "Oversteer corners [%]": round(kpis["oversteer_corners_pct"], 1),
            "Neutral corners [%]": round(kpis["neutral_corners_pct"], 1),
        })
    _show_summary_table(rows)
    for run_name, (figs, _kpis) in run_results.items():
        st.caption(Path(run_name).stem)
        for fig in figs:
            _plotly_chart(fig, use_container_width=True, theme=None)
    with st.expander("Per-lap understeer / oversteer"):
        st.dataframe(_concat_run_tables(lap_tables), use_container_width=True, hide_index=True)
    with st.expander("Per-corner understeer / oversteer"):
        st.dataframe(_concat_run_tables(turn_tables), use_container_width=True, hide_index=True)


def _render_rb_function_check(dfs: dict[str, pl.DataFrame]) -> None:
    st.divider()
    st.subheader("Function check — is RB delivering SR ≈ −0.20 and recovering energy?")
    all_figs: dict[str, list[go.Figure]] = {}
    rows = []
    for run_name, df in dfs.items():
        try:
            figs, kpis = rb.rb_function_kpis(df)
        except Exception as exc:
            st.warning(f"{run_name}: RB function check unavailable: {exc}")
            continue
        all_figs[run_name] = figs
        rows.append({
            "Run": Path(run_name).stem,
            "In target [%]": round(kpis["pct_in_target"], 1),
            "Lock-up risk [%]": round(kpis["pct_lockup_risk"], 1),
            "Recovered total [Wh]": round(kpis["energy_recovered_wh_total"], 1),
            "Recovered / lap [Wh]": round(kpis["energy_recovered_wh_median_lap"], 1),
            "Regen coverage [%]": round(kpis["regen_coverage_pct"], 1),
        })
    if rows:
        _show_summary_table(rows)
    for fig in _overlay_figures(all_figs):
        _plotly_chart(fig, use_container_width=True, theme=None)


def _render_pc_function_check(dfs: dict[str, pl.DataFrame]) -> None:
    st.divider()
    st.subheader("Function check — Power Control: is P_bat under 80 kW and used to the cap?")
    all_figs: dict[str, list[go.Figure]] = {}
    rows = []
    for run_name, df in dfs.items():
        try:
            figs, kpis = pt.pc_function_kpis(df)
        except Exception as exc:
            st.warning(f"{run_name}: PC function check unavailable: {exc}")
            continue
        all_figs[run_name] = [
            fig for fig in figs
            if "Battery power vs time" not in (fig.layout.title.text or "")
        ]
        rows.append({
            "Run": Path(run_name).stem,
            "Over cap [%]": round(kpis["pct_over_cap"], 2),
            "Overshoot events": int(kpis["n_overshoot_events"]),
            "Peak Pbat [kW]": round(kpis["peak_kw"], 1),
            "Near cap @ full [%]": round(kpis["pct_near_cap_at_full"], 1),
        })
    if rows:
        _show_summary_table(rows)
    for fig in _overlay_figures(all_figs):
        _plotly_chart(fig, use_container_width=True, theme=None)


# ── Tab renderers ─────────────────────────────────────────────────────────────

def _tab_powertrain(dfs: dict[str, pl.DataFrame]) -> None:
    run_tokens = _run_cache_tokens(dfs)

    # ── Energy ───────────────────────────────────────────────────────────────
    st.subheader("Energy per Lap")
    energy_x_mode = _select_per_lap_axis("pt_energy_axis", default="laps")
    try:
        if len(dfs) == 1:
            fig, kpis = _pt_energy_per_lap_fig_cached(dfs, run_tokens, energy_x_mode)
            for w in kpis.get("warnings", []):
                st.warning(w)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean net / lap",     f"{kpis['e_mean']:.4f} kWh")
            c2.metric("Total net",          f"{kpis['e_total']:.3f} kWh")
            c3.metric("Coefficient of Variation", f"{kpis['cv']:.1f}%")
            c4.metric(
                f"R² (Enet vs {'laptime' if energy_x_mode == 'laptime' else 'lap'})",
                f"{kpis['r2']:.3f}",
            )
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Mean battery power", f"{kpis['p_mean']:.1f} kW")
            c6.metric("Consumed / lap",     f"{kpis['e_cons_mean']:.4f} kWh")
            c7.metric("Recovered / lap",    f"{kpis['e_rec_mean']:.4f} kWh")
            c8.metric("Fastest lap",        f"L{kpis['fastest_lap']} — {kpis['fastest_lt']:.2f} s")
            c9, c10, c11, c12 = st.columns(4)
            c9.metric("Total consumed",     f"{kpis['e_cons_total']:.3f} kWh")
            c10.metric("Total recovered",   f"{kpis['e_rec_total']:.3f} kWh")
            c11.metric("Min net lap",       f"L{kpis['min_e_lap']} — {kpis['min_e']:.4f} kWh")
            c12.metric("Max net lap",       f"L{kpis['max_e_lap']} — {kpis['max_e']:.4f} kWh")
            _plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {
                run_name: _pt_energy_per_lap_fig_cached(
                    {run_name: df},
                    _run_cache_tokens({run_name: df}),
                    energy_x_mode,
                )
                for run_name, df in dfs.items()
            }
            for _run_name, (_fig, kpis) in run_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(w)
            _show_summary_table([
                {
                    "Run": run_name,
                    "Mean net / lap [kWh]": round(kpis["e_mean"], 4),
                    "Total net [kWh]": round(kpis["e_total"], 3),
                    "Consumed / lap [kWh]": round(kpis["e_cons_mean"], 4),
                    "Recovered / lap [kWh]": round(kpis["e_rec_mean"], 4),
                    "Mean battery power [kW]": round(kpis["p_mean"], 1),
                    "Fastest lap": int(kpis["fastest_lap"]),
                    "Fastest lt [s]": round(kpis["fastest_lt"], 2),
                }
                for run_name, (_fig, kpis) in run_results.items()
            ])
            _plotly_chart(
                _overlay_figures({run_name: [fig] for run_name, (fig, _kpis) in run_results.items()})[0],
                use_container_width=True,
                theme=None,
            )
            with st.expander("Per-lap data"):
                st.dataframe(
                    _concat_run_tables({run_name: kpis["table"] for run_name, (_fig, kpis) in run_results.items()}),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.error(f"Energy KPIs unavailable: {exc}")

    st.divider()

    # ── Power per wheel ───────────────────────────────────────────────────────
    st.subheader("Power Distribution per Wheel")
    power_x_mode = _select_per_lap_axis("pt_power_axis", default="laps")
    try:
        if len(dfs) == 1:
            fig, kpis = _pt_power_per_wheel_fig_cached(dfs, run_tokens, power_x_mode)
            for w in kpis.get("warnings", []):
                st.warning(w)
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean total power", f"{kpis['mean_total_kw']:.1f} kW")
            c2.metric("Front / Rear",
                      f"{kpis['fr_pct']:.1f}% / {100 - kpis['fr_pct']:.1f}%")
            c3.metric("Left / Right",
                      f"{kpis['lr_pct']:.1f}% / {100 - kpis['lr_pct']:.1f}%")
            c4, c5, c6, c7 = st.columns(4)
            for w, col in zip(("FL", "FR", "RL", "RR"), [c4, c5, c6, c7]):
                col.metric(w,
                           f"{kpis['wheel_mean_kw'][w]:.2f} kW"
                           f"  ({kpis['wheel_pct'][w]:.1f}%)")
            _plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {
                run_name: _pt_power_per_wheel_fig_cached(
                    {run_name: df},
                    _run_cache_tokens({run_name: df}),
                    power_x_mode,
                )
                for run_name, df in dfs.items()
            }
            for _run_name, (_fig, kpis) in run_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(w)
            _show_summary_table([
                {
                    "Run": run_name,
                    "Mean total power [kW]": round(kpis["mean_total_kw"], 1),
                    "Front share [%]": round(kpis["fr_pct"], 1),
                    "Left share [%]": round(kpis["lr_pct"], 1),
                    "FL [kW]": round(kpis["wheel_mean_kw"]["FL"], 2),
                    "FR [kW]": round(kpis["wheel_mean_kw"]["FR"], 2),
                    "RL [kW]": round(kpis["wheel_mean_kw"]["RL"], 2),
                    "RR [kW]": round(kpis["wheel_mean_kw"]["RR"], 2),
                }
                for run_name, (_fig, kpis) in run_results.items()
            ])
            _plotly_chart(
                _overlay_figures({run_name: [fig] for run_name, (fig, _kpis) in run_results.items()})[0],
                use_container_width=True,
                theme=None,
            )
            with st.expander("Per-lap data"):
                st.dataframe(
                    _concat_run_tables({run_name: kpis["table"] for run_name, (_fig, kpis) in run_results.items()}),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.error(f"Power KPIs unavailable: {exc}")

    st.divider()

    # ── Battery status ────────────────────────────────────────────────────────
    st.subheader("Battery Status")
    battery_x_mode = _select_per_lap_axis("pt_battery_axis", default="laps")
    try:
        if len(dfs) == 1:
            fig, kpis = _pt_battery_status_fig_cached(dfs, run_tokens, battery_x_mode)
            for w in kpis.get("warnings", []):
                st.warning(w)
            c1, c2, c3 = st.columns(3)
            c1.metric("SoC start",         f"{_fmt(kpis['soc_start'], '.1f')}%")
            c2.metric("SoC end",           f"{_fmt(kpis['soc_end'], '.1f')}%",
                      delta=(f"-{_fmt(kpis['soc_total_drop'], '.1f')}%" if np.isfinite(kpis["soc_total_drop"]) else None))
            c3.metric("Voltage sag",        f"{kpis['voltage_sag']:.1f} V")
            c4, c5, c6 = st.columns(3)
            c4.metric("Mean SoC drop / lap", f"{kpis['soc_drop_per_lap']:.2f}%")
            c5.metric("Mean voltage",        f"{kpis['mean_voltage']:.1f} V")
            c6.metric("Min voltage",         f"{kpis['min_voltage']:.1f} V")
            c7 = st.columns(1)[0]
            c7.metric("Mean current",        f"{kpis['mean_current']:.1f} A")
            _plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {
                run_name: _pt_battery_status_fig_cached(
                    {run_name: df},
                    _run_cache_tokens({run_name: df}),
                    battery_x_mode,
                )
                for run_name, df in dfs.items()
            }
            for _run_name, (_fig, kpis) in run_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(w)
            _show_summary_table([
                {
                    "Run": run_name,
                    "Mean dSoC / lap [%]": round(kpis["soc_drop_per_lap"], 2),
                    "Voltage sag [V]": round(kpis["voltage_sag"], 1),
                    "Mean voltage [V]": round(kpis["mean_voltage"], 1),
                    "Min voltage [V]": round(kpis["min_voltage"], 1),
                    "Mean current [A]": round(kpis["mean_current"], 1),
                }
                for run_name, (_fig, kpis) in run_results.items()
            ])
            _plotly_chart(
                _overlay_figures({run_name: [fig] for run_name, (fig, _kpis) in run_results.items()})[0],
                use_container_width=True,
                theme=None,
            )
            with st.expander("Per-lap data"):
                st.dataframe(
                    _concat_run_tables({run_name: kpis["table"] for run_name, (_fig, kpis) in run_results.items()}),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.error(f"Battery KPIs unavailable: {exc}")

    st.divider()

    # ── Thermal evolution ─────────────────────────────────────────────────────
    st.subheader("Thermal Evolution")
    thermal_x_mode = _select_per_lap_axis("pt_thermal_axis", default="laps")
    try:
        if len(dfs) == 1:
            fig, kpis = _pt_thermal_evolution_fig_cached(dfs, run_tokens, thermal_x_mode)
            for w in kpis.get("warnings", []):
                st.warning(w)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Peak motor (P95)",    f"{kpis['peak_motor']:.1f} °C")
            c2.metric("Peak inverter (P95)", f"{kpis['peak_inverter']:.1f} °C")
            c3.metric("Peak battery Tmax",   f"{kpis['peak_batt_tmax']:.1f} °C")
            c4.metric("Motor thermal slope", f"{kpis['motor_thermal_slope']:+.2f} °C/lap")
            c5, c6, c7, c8 = st.columns(4)
            for w, col in zip(("FL", "FR", "RL", "RR"), [c5, c6, c7, c8]):
                col.metric(f"Motor {w} peak",
                           f"{kpis['motor_peak_by_wheel'][w]:.1f} °C")
            _plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {
                run_name: _pt_thermal_evolution_fig_cached(
                    {run_name: df},
                    _run_cache_tokens({run_name: df}),
                    thermal_x_mode,
                )
                for run_name, df in dfs.items()
            }
            for _run_name, (_fig, kpis) in run_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(w)
            _show_summary_table([
                {
                    "Run": run_name,
                    "Peak motor [°C]": round(kpis["peak_motor"], 1),
                    "Peak inverter [°C]": round(kpis["peak_inverter"], 1),
                    "Peak batt Tmax [°C]": round(kpis["peak_batt_tmax"], 1),
                    "Motor slope [°C/lap]": round(kpis["motor_thermal_slope"], 2),
                }
                for run_name, (_fig, kpis) in run_results.items()
            ])
            _plotly_chart(
                _overlay_figures({run_name: [fig] for run_name, (fig, _kpis) in run_results.items()})[0],
                use_container_width=True,
                theme=None,
            )
            with st.expander("Per-lap data"):
                st.dataframe(
                    _concat_run_tables({run_name: kpis["table"] for run_name, (_fig, kpis) in run_results.items()}),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.error(f"Thermal KPIs unavailable: {exc}")
    _render_pc_function_check(dfs)


def _tab_dynamics(dfs: dict[str, pl.DataFrame]) -> None:
    dyn_section = st.segmented_control(
        "Dynamics section",
        options=["Braking", "Cornering", "Acceleration", "Setup", "Grip Factors"],
        default="Braking",
        required=True,
        key="dyn_subsection",
        label_visibility="collapsed",
        width="stretch",
    )

    if dyn_section == "Braking":
        _render_dynamics_braking(dfs)
        return
    if dyn_section == "Acceleration":
        _render_dynamics_acceleration(dfs)
        return
    if dyn_section == "Setup":
        _render_dynamics_setup_signatures(dfs)
        return
    if dyn_section == "Grip Factors":
        _render_dynamics_grip_factors(dfs)
        return

    _render_dynamics_cornering(dfs)


def _tab_events(dfs: dict[str, pl.DataFrame]) -> None:
    section = st.segmented_control(
        "Events section",
        options=["Skidpad", "Acceleration"],
        default="Skidpad",
        required=True,
        key="events_subsection",
        label_visibility="collapsed",
        width="stretch",
    )
    if section == "Acceleration":
        _render_events_acceleration(dfs)
        return
    _render_events_skidpad(dfs)


def _render_events_acceleration(dfs: dict[str, pl.DataFrame]) -> None:
    _ = dfs
    st.info("Acceleration analysis coming soon.")


def _render_events_skidpad(dfs: dict[str, pl.DataFrame]) -> None:
    skidpad_dfs: dict[str, pl.DataFrame] = {}
    for run_name, df in dfs.items():
        if not skidpad.is_skidpad_run(df):
            continue
        filtered = _filter_skidpad_mode_df(df)
        if not filtered.is_empty():
            skidpad_dfs[run_name] = filtered

    if not skidpad_dfs:
        st.warning("No skidpad data in selected runs (lapcount_mode != 'skidpad').")
        return

    run_tokens = {token[0]: token for token in _run_cache_tokens(skidpad_dfs)}
    st.caption(
        "Skidpad KPIs use only samples tagged `lapcount_mode == 'skidpad'`. "
        "Circle side uses the current IMU convention: positive `Filtering_VN_ay` is treated as left."
    )

    _render_skidpad_top_kpis(skidpad_dfs, run_tokens)

    metric_specs = [
        ("event_time", "Event Time"),
        ("lateral_g", "Sustained Lateral G"),
        ("driven_radius", "Driven Radius"),
        ("balance", "Balance and Understeer"),
        ("driver_smoothness", "Driver Smoothness"),
        ("gps_figure8", "GPS Figure-8"),
    ]
    for metric, title in metric_specs:
        st.divider()
        _render_skidpad_metric(metric, title, skidpad_dfs, run_tokens)

    tv_dfs = {name: df for name, df in skidpad_dfs.items() if skidpad.has_tv_signals(df)}
    if tv_dfs:
        st.divider()
        _render_skidpad_metric("tv_intervention", "TV Intervention", tv_dfs, run_tokens)

    load_dfs = {name: df for name, df in skidpad_dfs.items() if skidpad.has_load_signals(df)}
    if load_dfs:
        st.divider()
        _render_skidpad_metric("lateral_load_dist", "Lateral Load Distribution", load_dfs, run_tokens)


def _filter_skidpad_mode_df(df: pl.DataFrame) -> pl.DataFrame:
    """Return only samples tagged as skidpad, preserving the input schema."""
    if "lapcount_mode" not in df.columns or len(df) == 0:
        return df.filter(pl.Series("__skidpad_mask", np.zeros(len(df), dtype=bool)))
    mask = np.array([str(value).lower() == "skidpad" for value in df["lapcount_mode"].to_list()])
    return df.filter(pl.Series("__skidpad_mask", mask))


def _render_skidpad_top_kpis(
    skidpad_dfs: dict[str, pl.DataFrame],
    run_tokens: dict[str, tuple[str, FileSignature, str]],
) -> None:
    rows: list[dict[str, object]] = []
    for run_name, df in skidpad_dfs.items():
        event_kpis: dict = {}
        lat_kpis: dict = {}
        radius_kpis: dict = {}
        try:
            _fig, event_kpis = _skidpad_fig_cached("event_time", df, run_tokens[run_name])
        except Exception as exc:
            st.error(f"{Path(run_name).stem}: skidpad event time unavailable: {exc}")
        try:
            _fig, lat_kpis = _skidpad_fig_cached("lateral_g", df, run_tokens[run_name])
        except Exception as exc:
            st.error(f"{Path(run_name).stem}: skidpad lateral G unavailable: {exc}")
        try:
            _fig, radius_kpis = _skidpad_fig_cached("driven_radius", df, run_tokens[run_name])
        except Exception as exc:
            st.error(f"{Path(run_name).stem}: skidpad radius unavailable: {exc}")

        rows.append({
            "Run": Path(run_name).stem,
            "Event time [s]": event_kpis.get("event_time_s", np.nan),
            "Timed R [s]": event_kpis.get("timed_R_s", np.nan),
            "Timed L [s]": event_kpis.get("timed_L_s", np.nan),
            "L/R asymmetry [s]": event_kpis.get("lr_asymmetry_s", np.nan),
            "ay max [g]": lat_kpis.get("ay_max_g", np.nan),
            "R error mean [m]": radius_kpis.get("radius_error_mean_m", np.nan),
        })

    if not rows:
        return
    if len(rows) == 1:
        vals = rows[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Event time", f"{_fmt(vals.get('Event time [s]', np.nan), '.3f')} s")
        c2.metric("L/R asymmetry", f"{_fmt(vals.get('L/R asymmetry [s]', np.nan), '.3f')} s")
        c3.metric("ay max", f"{_fmt(vals.get('ay max [g]', np.nan), '.2f')} g")
        c4.metric("R error mean", f"{_fmt(vals.get('R error mean [m]', np.nan), '+.2f')} m")
    else:
        _show_summary_table([_round_numeric_row(row) for row in rows])


def _render_skidpad_metric(
    metric: str,
    title: str,
    skidpad_dfs: dict[str, pl.DataFrame],
    run_tokens: dict[str, tuple[str, FileSignature, str]],
) -> None:
    st.subheader(title)
    table_rows: list[dict[str, object]] = []
    for run_name, df in skidpad_dfs.items():
        try:
            fig, kpis = _skidpad_fig_cached(metric, df, run_tokens[run_name])
            for warning in kpis.get("warnings", []):
                st.warning(f"{Path(run_name).stem}: {warning}")
            if len(skidpad_dfs) > 1:
                st.markdown(f"**{Path(run_name).stem}**")
            _plotly_chart(fig, use_container_width=True, theme=None)
            for row in kpis.get("rows", []):
                table_rows.append(_round_numeric_row({"Run": Path(run_name).stem, **row}))
        except Exception as exc:
            st.error(f"{Path(run_name).stem}: {title.lower()} unavailable: {exc}")
    if table_rows:
        st.dataframe(pl.DataFrame(table_rows), use_container_width=True, hide_index=True)


def _round_numeric_row(row: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, (float, np.floating)):
            out[key] = round(float(value), 3) if np.isfinite(value) else np.nan
        else:
            out[key] = value
    return out


def _render_ideal_braking_curve(dfs: dict[str, pl.DataFrame]) -> None:
    """Render ideal front/rear brake distribution vs measured regen."""
    st.subheader("Ideal Braking Curve  ·  Model vs Measured Regen")
    st.caption(
        "Brake-force plane (front axle vs rear axle). Curves are the ideal "
        "load-proportional distribution at 0, 15 and 25 m/s; points are "
        "measured regenerative force from motor torque. If a `BSE` trace "
        "appears, treat it as pedal-pressure demand, not as proven hydraulic "
        "brake torque at the disc."
    )
    try:
        fig, kpis = _dyn_ideal_braking_curve_fig_cached(dfs, _run_cache_tokens(dfs))
        for w in kpis.get("warnings", []):
            st.warning(w)

        run_kpis = kpis.get("runs", {})
        if len(run_kpis) == 1:
            _run_name, vals = next(iter(run_kpis.items()))
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Front bias", f"{_fmt(vals.get('front_bias_mean', np.nan) * 100.0, '.1f')} %")
            c2.metric("Bias std", f"{_fmt(vals.get('front_bias_std', np.nan) * 100.0, '.1f')} pp")
            c3.metric("RMS to ideal", f"{_fmt(vals.get('rms_dist_to_ideal_N', np.nan), '.0f')} N")
            c4.metric("Rear overbiased", f"{_fmt(vals.get('pct_time_rear_overbiased', np.nan), '.1f')} %")
            c5.metric("Peak brake", f"{_fmt(vals.get('peak_combined_brake_g', np.nan), '.2f')} g")
        elif len(run_kpis) > 1:
            rows = []
            for run_name, vals in run_kpis.items():
                rows.append({
                    "Run": run_name,
                    "Front bias [%]": round(vals.get("front_bias_mean", np.nan) * 100.0, 2),
                    "Bias std [pp]": round(vals.get("front_bias_std", np.nan) * 100.0, 2),
                    "RMS to ideal [N]": round(vals.get("rms_dist_to_ideal_N", np.nan), 1),
                    "Rear overbiased [%]": round(vals.get("pct_time_rear_overbiased", np.nan), 2),
                    "Peak brake [g]": round(vals.get("peak_combined_brake_g", np.nan), 3),
                    "Samples": int(vals.get("samples", 0)),
                })
            st.dataframe(pl.DataFrame(rows), use_container_width=True, hide_index=True)

        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Ideal braking curve unavailable: {exc}")


def _render_dynamics_braking(dfs: dict[str, pl.DataFrame]) -> None:
    """Longitudinal chassis response in braking."""
    _render_ideal_braking_curve(dfs)
    st.divider()

    st.subheader("Longitudinal Decel Envelope")
    st.caption(
        "Faint points are real braking samples. The thick line is the p95 "
        "deceleration envelope by 5 m/s speed bin. The dashed line is the "
        "CAT17x design target, brake.MaxDeceleration = 1.79 g."
    )
    try:
        fig, kpis = _dyn_decel_envelope_fig_cached(dfs, _run_cache_tokens(dfs))
        for w in kpis.get("warnings", []):
            st.warning(w)
        rows = [
            {
                "Run": run_name,
                "Peak p95 decel [g]": round(vals.get("peak_decel_p95_g", np.nan), 3),
                "Peak speed [m/s]": round(vals.get("speed_at_peak_mps", np.nan), 1),
                "Gap [g]": round(vals.get("gap_to_design_g", np.nan), 3),
                "Design use [%]": round(vals.get("pct_design_decel", np.nan), 1),
                "Samples": int(vals.get("samples", 0)),
            }
            for run_name, vals in kpis.get("runs", {}).items()
        ]
        if rows:
            _show_summary_table(rows)
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Decel envelope unavailable: {exc}")

    st.divider()
    st.subheader("Braking Stability  ·  Steering vs Yaw-rate smoothness ratio")
    st.caption(
        "Stability KPI (Rouelle, RCE Sept 2019): integral of steering "
        "smoothness divided by integral of yaw-rate smoothness on each "
        "straight-line braking event (|ay| < 0.2 g during brake phase). "
        "Steering [deg] and yaw rate [deg/s] use the same units as Figs 2-3 "
        "of the paper. Higher = more stable chassis."
    )
    brk_x_mode = _select_per_lap_axis("dyn_brake_stability_axis", default="laps")
    try:
        run_tokens = _run_cache_tokens(dfs)
        fig_evt, kpis_evt = _dyn_braking_stability_fig_cached(dfs, run_tokens)
        for w in kpis_evt.get("warnings", []):
            st.warning(w)
        run_kpis = kpis_evt.get("runs", {})
        if len(run_kpis) == 1:
            _name, v = next(iter(run_kpis.items()))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Events", str(int(v.get("events", 0))))
            c2.metric("Stability KPI (med)", f"{_fmt(v.get('stability_kpi_median'), '.3f')}")
            c3.metric("Lockup front [s]", f"{_fmt(v.get('lockup_front_total_s'), '.2f')}")
            c4.metric("Lockup rear [s]", f"{_fmt(v.get('lockup_rear_total_s'), '.2f')}")
            c5, c6, c7 = st.columns(3)
            c5.metric("Stability KPI P25", f"{_fmt(v.get('stability_kpi_p25'), '.3f')}")
            c6.metric("Stability KPI P75", f"{_fmt(v.get('stability_kpi_p75'), '.3f')}")
            c7.metric(
                "∫Steer / ∫Yaw (med)",
                f"{_fmt(v.get('steering_stability_median'), '.1f')} / "
                f"{_fmt(v.get('yaw_rate_stability_median'), '.1f')}",
            )
        elif len(run_kpis) > 1:
            rows = [
                {
                    "Run": run_name,
                    "Events": int(v.get("events", 0)),
                    "Stability KPI (med)": round(v.get("stability_kpi_median", np.nan), 3),
                    "P25": round(v.get("stability_kpi_p25", np.nan), 3),
                    "P75": round(v.get("stability_kpi_p75", np.nan), 3),
                    "Lockup front [s]": round(v.get("lockup_front_total_s", np.nan), 2),
                    "Lockup rear [s]": round(v.get("lockup_rear_total_s", np.nan), 2),
                }
                for run_name, v in run_kpis.items()
            ]
            if rows:
                _show_summary_table(rows)
        _plotly_chart(fig_evt, use_container_width=True, theme=None)

        fig_lap, kpis_lap = _dyn_braking_stability_per_lap_fig_cached(
            dfs, run_tokens, brk_x_mode
        )
        for w in kpis_lap.get("warnings", []):
            st.warning(w)
        _plotly_chart(fig_lap, use_container_width=True, theme=None)

        events_by_run = kpis_evt.get("events_by_run", {}) or {}
        non_empty = {name: evt for name, evt in events_by_run.items() if not evt.is_empty()}
        if non_empty:
            with st.expander("Braking stability events (per run)"):
                for name, evt_df in non_empty.items():
                    st.markdown(f"**{Path(name).stem}**")
                    st.dataframe(evt_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Braking stability unavailable: {exc}")


def _render_dynamics_acceleration(dfs: dict[str, pl.DataFrame]) -> None:
    """Longitudinal chassis response in acceleration."""
    st.subheader("Ideal Traction Curve  ·  Model vs Measured Drive")
    st.caption(
        "Drive-force plane (front axle vs rear axle). Curves are the ideal AWD "
        "load-proportional distribution under longitudinal load transfer."
    )
    try:
        fig, kpis = _dyn_ideal_traction_curve_fig_cached(dfs, _run_cache_tokens(dfs))
        for w in kpis.get("warnings", []):
            st.warning(w)
        rows = [
            {
                "Run": run_name,
                "Rear bias [%]": round(vals.get("rear_bias_mean", np.nan) * 100.0, 1),
                "Ideal rear [%]": round(vals.get("rear_bias_ideal_mean", np.nan) * 100.0, 1),
                "RMS to ideal [N]": round(vals.get("rms_dist_to_ideal_N", np.nan), 1),
                "Peak accel [g]": round(vals.get("peak_combined_accel_g", np.nan), 3),
                "Power-limited [%]": round(vals.get("pct_time_power_limited", np.nan), 1),
                "Samples": int(vals.get("samples", 0)),
            }
            for run_name, vals in kpis.get("runs", {}).items()
        ]
        if rows:
            _show_summary_table(rows)
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Ideal traction curve unavailable: {exc}")

    st.divider()

    st.subheader("Longitudinal Accel Envelope")
    st.caption(
        "P95 ax in acceleration events by 5 m/s speed bin, with grip-limited "
        f"μ={dyn.MU_TIRE:.2f} g and 80 kW power-limited references."
    )
    try:
        fig, kpis = _dyn_accel_envelope_fig_cached(dfs, _run_cache_tokens(dfs))
        for w in kpis.get("warnings", []):
            st.warning(w)
        rows = [
            {
                "Run": run_name,
                "Peak ax [g]": round(vals.get("peak_ax_g", np.nan), 3),
                "Samples": int(vals.get("samples", 0)),
            }
            for run_name, vals in kpis.get("runs", {}).items()
        ]
        if rows:
            _show_summary_table(rows)
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Accel envelope unavailable: {exc}")


def _render_setup_calibration_banner(df: pl.DataFrame) -> None:
    damp_cols = [c for c in ("DampFL", "DampFR", "DampRL", "DampRR") if c in df.columns]
    status = str(df["Pot_Calibration_Status"][0]) if "Pot_Calibration_Status" in df.columns else "failed"
    convention = str(df["Pot_Calibration_Convention"][0]) if "Pot_Calibration_Convention" in df.columns else "raw"
    r_roll = float(df["Pot_Calibration_r_roll"][0]) if "Pot_Calibration_r_roll" in df.columns else np.nan
    r_pitch = float(df["Pot_Calibration_r_pitch"][0]) if "Pot_Calibration_r_pitch" in df.columns else np.nan
    message = str(df["Pot_Calibration_Message"][0]) if "Pot_Calibration_Message" in df.columns else "Calibration failed - T1 metrics disabled."
    details = (
        f"{message} Convention: {convention}. "
        f"r_roll={_fmt(r_roll, '.2f')}, r_pitch={_fmt(r_pitch, '.2f')}. "
        f"Damp columns: {', '.join(damp_cols) if damp_cols else 'missing'}."
    )
    if status == "validated":
        st.success(details)
    elif status == "partial":
        st.warning(details)
    else:
        st.error(details + " Falling back to raw damper-count diagnostics where possible.")


def _render_dynamics_setup_signatures(dfs: dict[str, pl.DataFrame]) -> None:
    """Setup diagnostics for roll, pitch, dampers and static load sanity."""
    phase_label = st.segmented_control(
        "Damper phase",
        options=["ALL", "BRAKE", "CORNER", "ACCEL", "STRAIGHT"],
        default="ALL",
        required=True,
        key="dyn_damper_phase",
        label_visibility="collapsed",
    )
    phase = str(phase_label).lower()

    st.subheader("Lateral Load Transfer Distribution  ·  Mid-Corner Avg")
    st.caption(
        "Setup KPI from corner samples only. For each lap, mid-corner is the "
        "higher-|ay| part of radius-filtered corners, then LLTD_front is averaged: "
        "LLTD_front = |Fz_FR - Fz_FL| / (|Fz_FR - Fz_FL| + |Fz_RR - Fz_RL|). "
        "Use this by lap to track setup changes over a run; use per-corner only "
        "when comparing the same named turns in a dedicated corner analysis."
    )
    lltd_x_mode = _select_per_lap_axis("dyn_setup_lltd_axis", default="laps")
    try:
        fig, kpis = dyn.lltd_mid_corner_per_lap_fig(dfs, x_mode=lltd_x_mode)
        for w in kpis.get("warnings", []):
            st.warning(w)
        rows = [
            {
                "Run": Path(run_name).stem,
                "Laps": int(vals.get("laps", 0)),
                "Mean LLTD [%]": round(vals.get("lltd_mean_pct", np.nan), 3),
                "Min [%]": round(vals.get("lltd_min_pct", np.nan), 3),
                "Max [%]": round(vals.get("lltd_max_pct", np.nan), 3),
                "Span [pp]": round(vals.get("lltd_span_pct_points", np.nan), 4),
                "Mean samples": round(vals.get("samples_mean", np.nan), 0),
            }
            for run_name, vals in kpis.get("runs", {}).items()
        ]
        if rows:
            _show_summary_table(rows)
        _plotly_chart(fig, use_container_width=True, theme=None)
        table = kpis.get("table")
        if table is not None and not table.is_empty():
            with st.expander("Per-lap LLTD data"):
                st.dataframe(
                    table.with_columns([
                        pl.col("LapTime [s]").round(3),
                        pl.col("LLTD mid-corner avg [%]").round(4),
                        pl.col("LLTD mid-corner median [%]").round(4),
                        pl.col("LLTD mid-corner span [pp]").round(5),
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.error(f"LLTD mid-corner unavailable: {exc}")

    st.divider()

    for run_name, df in dfs.items():
        if len(dfs) > 1:
            st.markdown(f"### {Path(run_name).stem}")
        _render_setup_calibration_banner(df)

        st.subheader("Roll Gradient  ·  Measured vs Theoretical")
        st.caption(
            "Radius-filtered corner samples. The theoretical CAT17x reference uses "
            "m·h_roll/(Krollf+Krollr), approximately 0.52 deg/g."
        )
        try:
            fig, kpis = dyn.roll_gradient_fig(df)
            for w in kpis.get("warnings", []):
                st.warning(w)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Theory", f"{_fmt(kpis.get('theoretical_deg_per_g'), '.2f')} deg/g")
            c2.metric("Front grad", f"{_fmt(kpis.get('front_gradient_deg_per_g'), '+.2f')} deg/g")
            c3.metric("Front dev", f"{_fmt(kpis.get('front_deviation_pct'), '+.1f')} %")
            c4.metric("Rear grad", f"{_fmt(kpis.get('rear_gradient_deg_per_g'), '+.2f')} deg/g")
            c5.metric("Rear dev", f"{_fmt(kpis.get('rear_deviation_pct'), '+.1f')} %")
            _plotly_chart(fig, use_container_width=True, theme=None)
        except Exception as exc:
            st.error(f"Roll gradient unavailable: {exc}")

        st.divider()

        st.subheader("Pitch Gradient  ·  Braking vs Acceleration")
        st.caption("Pitch comparison only; CAT17x pitch stiffness is not documented for a theoretical line.")
        try:
            fig, kpis = dyn.pitch_gradient_fig(df)
            for w in kpis.get("warnings", []):
                st.warning(w)
            c1, c2, c3 = st.columns(3)
            c1.metric("Brake gradient", f"{_fmt(kpis.get('brake_gradient_deg_per_g'), '+.2f')} deg/g")
            c2.metric("Accel gradient", f"{_fmt(kpis.get('accel_gradient_deg_per_g'), '+.2f')} deg/g")
            c3.metric("Calibrated", "yes" if kpis.get("calibrated") else "no")
            _plotly_chart(fig, use_container_width=True, theme=None)
        except Exception as exc:
            st.error(f"Pitch gradient unavailable: {exc}")

        st.divider()

        st.subheader("Damper Velocity Histograms  ·  by Phase")
        try:
            figs, kpis = dyn.damper_histogram_figs(df, phase=phase)
            for w in kpis.get("warnings", []):
                st.warning(w)
            front_bump = kpis.get("bump_share_by_axle", {}).get("front", float("nan"))
            rear_bump = kpis.get("bump_share_by_axle", {}).get("rear", float("nan"))
            c1, c2, c3 = st.columns(3)
            c1.metric("Front bump share", f"{_fmt(front_bump * 100.0, '.1f')} %")
            c2.metric("Rear bump share", f"{_fmt(rear_bump * 100.0, '.1f')} %")
            c3.metric("Calibrated", "yes" if kpis.get("calibrated") else "no")
            for fig in figs:
                _plotly_chart(fig, use_container_width=True, theme=None)
        except Exception as exc:
            st.error(f"Damper histograms unavailable: {exc}")

        try:
            figs, _kpis = dyn.spring_velocity_histogram_figs(df, phase=phase)
            for fig in figs:
                _plotly_chart(fig, use_container_width=True, theme=None)
        except Exception:
            pass

        st.divider()

        st.subheader("Static Fz Reference")
        st.caption("Per-corner weight from straight low-input samples vs CAT17x design (706 N/corner). Color: green ≤±5%, yellow ≤±10%, red >±10%.")
        try:
            fig, kpis = dyn.static_fz_reference_fig(df)
            for w in kpis.get("warnings", []):
                st.warning(w)
            corners_data = kpis.get('corners', {})
            c1, c2, c3, c4 = st.columns(4)
            for col, corner in zip([c1, c2, c3, c4], ['FL', 'FR', 'RL', 'RR']):
                cd = corners_data.get(corner, {})
                col.metric(
                    corner,
                    f"{_fmt(cd.get('measured_n', float('nan')), '.0f')} N",
                    f"{_fmt(cd.get('deviation_pct', float('nan')), '+.1f')} % vs design",
                )
            c1, c2, c3, c4 = st.columns(4)
            fs = kpis.get('front_share_pct', float('nan'))
            ls = kpis.get('left_share_pct', float('nan'))
            cw = kpis.get('cross_weight_pct', float('nan'))
            c1.metric("Front share", f"{_fmt(fs, '.1f')} %", f"{_fmt(fs - 50.0, '+.1f')} % vs 50%")
            c2.metric("Left share", f"{_fmt(ls, '.1f')} %", f"{_fmt(ls - 50.0, '+.1f')} % vs 50%")
            c3.metric("Cross weight (FL+RR)", f"{_fmt(cw, '.1f')} %", f"{_fmt(cw - 50.0, '+.1f')} % vs 50%")
            c4.metric("Samples", str(kpis.get("samples", 0)))
            _plotly_chart(fig, use_container_width=True, theme=None)
        except Exception as exc:
            st.error(f"Static Fz reference unavailable: {exc}")

        try:
            fig, _kpis = dyn.aero_load_heave_fig(df)
            st.caption("Add k_heave_F and k_heave_R to cat17x_parameters.md to overlay theoretical aero deflection.")
            _plotly_chart(fig, use_container_width=True, theme=None)
        except Exception:
            pass

        if len(dfs) > 1:
            st.divider()


def _render_manual_gate_editor(
    ui_rev: str,
) -> tuple[tuple[tuple[float, float], tuple[float, float]] | None, bool]:
    """Render compact finish-line actions and return the current manual gate."""
    preview_line = st.session_state.get("_dyn_manual_gate_line")
    open_fullscreen = False

    cols = st.columns([1, 1.15, 1.2, 1.1])
    if cols[0].button(
        "Reset",
        key=f"dyn_gate_reset_{ui_rev}",
        use_container_width=True,
    ):
        for key in (
            "_dyn_manual_gate_line",
            "_dyn_manual_gate_result",
            "_dyn_track_last_event_id",
        ):
            st.session_state.pop(key, None)
        st.rerun()
    apply_disabled = preview_line is None
    if cols[1].button(
        "Apply To All CSVs",
        key=f"dyn_gate_apply_{ui_rev}",
        use_container_width=True,
        disabled=apply_disabled,
    ):
        assert preview_line is not None
        updated: list[str] = []
        with st.spinner("Recomputing laps from manual finish line..."):
            for path in _telemetry_csv_paths(DATA_DIR):
                n = lapcount.detect_and_write_laps(path, gate_line_lonlat=preview_line)
                updated.append(f"{path.name}: {n} laps")
        _clear_data_caches()
        st.session_state["_dyn_manual_gate_result"] = updated
        st.rerun()

    if cols[2].button(
        "Restore Auto",
        key=f"dyn_gate_restore_auto_{ui_rev}",
        use_container_width=True,
    ):
        st.session_state.pop("_dyn_manual_gate_line", None)
        updated: list[str] = []
        with st.spinner("Restoring automatic lap detection..."):
            for path in _telemetry_csv_paths(DATA_DIR):
                n = lapcount.detect_and_write_laps(path)
                updated.append(f"{path.name}: {n} laps")
        _clear_data_caches()
        st.session_state["_dyn_manual_gate_result"] = updated
        st.rerun()
    if cols[3].button(
        "Full Screen",
        key=f"dyn_gate_fullscreen_{ui_rev}",
        use_container_width=True,
    ):
        st.session_state["_dyn_track_open_fullscreen"] = True
        open_fullscreen = True
    result_lines = st.session_state.get("_dyn_manual_gate_result")
    if result_lines:
        st.caption(" | ".join(result_lines))
    return preview_line, open_fullscreen


@st.dialog("Track — Full Screen", width="large")
def _render_track_fullscreen_dialog(
    pool: dict[str, np.ndarray],
    visible_mask: np.ndarray,
    cross_range: tuple[float, float] | None,
    gg_idx: np.ndarray | None,
    ui_rev: str,
    lap_gates: dict[str, dict],
) -> None:
    """Render the track map in a large dialog using the same lasso/manual-line flow."""
    manual_gate_line = st.session_state.get("_dyn_manual_gate_line")
    st.caption(
        "Use lasso to filter the GG zone, or draw a line in the map toolbar to define the finish line."
    )
    if manual_gate_line is not None:
        st.caption(
            "Manual line: "
            f"({manual_gate_line[0][0]:.6f}, {manual_gate_line[0][1]:.6f}) → "
            f"({manual_gate_line[1][0]:.6f}, {manual_gate_line[1][1]:.6f})"
        )
    track_fig = dyn.track_map_fig(
        pool,
        visible_mask,
        cross_range,
        gg_idx,
        ui_rev + "|fullscreen",
        lap_gates=lap_gates,
        manual_gate_line=manual_gate_line,
    )
    track_event = tmc.render_track_map_component(
        tmc.serialize_figure(track_fig),
        height_px=760,
        key="dyn_track_component_fullscreen_" + ui_rev,
    )
    _consume_track_component_event(
        track_event,
        pool_len=len(pool["ax"]),
        event_state_key="_dyn_track_last_event_id_fullscreen",
    )


def _render_dynamics_cornering(dfs: dict[str, pl.DataFrame]) -> None:
    """Cornering chassis balance without driver trace/map duplication."""
    st.subheader("Understeer Angle")
    st.caption(
        "Mean per-lap understeer angle using the same radius-corner definition "
        f"as Lap Analysis: R = V²/|ay| < 60 m. Steering is divided by "
        f"the {dyn.STEERING_RATIO:.2f} steering ratio before comparison."
    )
    understeer_x_mode = _select_per_lap_axis("dyn_understeer_axis", default="laps")
    try:
        if len(dfs) == 1:
            _run_name, df = next(iter(dfs.items()))
            kpis = dyn.understeer_angle_kpis(df)
            for w in kpis.get("warnings", []):
                st.warning(w)
            if not kpis.get("warnings"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Valid laps", str(kpis["valid_laps"]))
                c2.metric("Mean understeer", f"{_fmt(kpis['mean_understeer'], '.2f')} deg")
                c3.metric("Min / Max", f"{_fmt(kpis['min_understeer'], '.2f')} / {_fmt(kpis['max_understeer'], '.2f')} deg")
                c4.metric("Fastest valid lap", f"L{kpis['fastest_lap']} - {_fmt(kpis['fastest_lt'], '.2f')} s")
            fig = dyn.understeer_angle_fig(df, x_mode=understeer_x_mode)
            _plotly_chart(fig, use_container_width=True, theme=None)
            if not kpis.get("warnings"):
                with st.expander("Per-lap data"):
                    st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {}
            for run_name, df in dfs.items():
                run_results[run_name] = (
                    [dyn.understeer_angle_fig(df, x_mode=understeer_x_mode)],
                    dyn.understeer_angle_kpis(df),
                )
                for w in run_results[run_name][1].get("warnings", []):
                    st.warning(f"{run_name}: {w}")
            _show_summary_table([
                {
                    "Run": run_name,
                    "Valid laps": kpis["valid_laps"],
                    "Mean understeer [deg]": round(kpis["mean_understeer"], 2),
                    "Min [deg]": round(kpis["min_understeer"], 2),
                    "Max [deg]": round(kpis["max_understeer"], 2),
                    "Fastest lap": int(kpis["fastest_lap"]),
                    "Fastest lt [s]": round(kpis["fastest_lt"], 2),
                }
                for run_name, (_figs, kpis) in run_results.items()
                if not kpis.get("warnings")
            ])
            _plotly_chart(
                _overlay_figures({run_name: figs for run_name, (figs, _kpis) in run_results.items()})[0],
                use_container_width=True,
                theme=None,
            )
            with st.expander("Per-lap data"):
                st.dataframe(
                    _concat_run_tables({
                        run_name: kpis["table"]
                        for run_name, (_figs, kpis) in run_results.items()
                        if not kpis.get("warnings")
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.warning(f"Understeer angle unavailable: {exc}")

    st.divider()

    st.subheader("Steering vs Lateral Acceleration  ·  US/OS Curve")
    st.caption(
        "Steady-state samples are radius-filtered corners plus low ay and steer jerk. "
        f"Steering is divided by the {dyn.STEERING_RATIO:.2f} steering ratio before comparison. "
        "Slope at low |ay| is the linear understeer gradient."
    )
    if len(dfs) == 1:
        _run_name, df_single = next(iter(dfs.items()))
        try:
            fig, kpis = dyn.steering_vs_ay_fig(df_single)
            for w in kpis.get("warnings", []):
                st.warning(w)
            c1, c2, c3 = st.columns(3)
            c1.metric("US gradient", f"{_fmt(kpis.get('understeer_gradient_deg_per_g'), '+.2f')} deg/g")
            c2.metric("Median vx", f"{_fmt(kpis.get('vx_median_mps'), '.1f')} m/s")
            c3.metric("Samples", str(kpis.get("samples", 0)))
            _plotly_chart(fig, use_container_width=True, theme=None)
        except Exception as exc:
            st.warning(f"Steering vs ay unavailable: {exc}")
    else:
        steer_results: dict[str, tuple[go.Figure, dict]] = {}
        for run_name, df_single in dfs.items():
            try:
                fig, kpis = dyn.steering_vs_ay_fig(df_single)
                steer_results[run_name] = (fig, kpis)
            except Exception as exc:
                st.warning(f"Steering vs ay unavailable ({Path(run_name).stem}): {exc}")
        if steer_results:
            for rn, (_f, kpis) in steer_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(f"{Path(rn).stem}: {w}")
            _show_summary_table([
                {
                    "Run": Path(rn).stem,
                    "US gradient [deg/g]": _fmt(kpis.get("understeer_gradient_deg_per_g"), "+.2f"),
                    "Median vx [m/s]": _fmt(kpis.get("vx_median_mps"), ".1f"),
                    "Samples": kpis.get("samples", 0),
                }
                for rn, (_f, kpis) in steer_results.items()
            ])
            merged = _overlay_figures({rn: [f] for rn, (f, _k) in steer_results.items()})
            _plotly_chart(merged[0], use_container_width=True, theme=None)

    st.divider()

    st.subheader("Lateral Load Transfer Distribution  ·  Front Share")
    st.caption(
        "Blue points are individual corner samples. The orange line is the median deviation "
        "by |ay| band, and the blue band shows the P10-P90 spread. "
        "Front LTD = |ΔFz_front| / (|ΔFz_front| + |ΔFz_rear|), using Est_FZ in "
        "radius-filtered corners. The target comes from the roll "
        f"stiffness split: Krollf / (Krollf + Krollr) = {dyn.KROLLF_NMRAD:.1f} / "
        f"({dyn.KROLLF_NMRAD:.1f} + {dyn.KROLLR_NMRAD:.1f}) = "
        f"{dyn.KROLLF_NMRAD / (dyn.KROLLF_NMRAD + dyn.KROLLR_NMRAD) * 100.0:.1f}%."
    )
    if len(dfs) == 1:
        _run_name, df_single = next(iter(dfs.items()))
        try:
            fig, kpis = dyn.lateral_load_transfer_fig(df_single)
            for w in kpis.get("warnings", []):
                st.warning(w)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Median front LTD", f"{_fmt(kpis.get('ltd_front_mean') * 100.0, '.1f')} %")
            c2.metric("Theoretical", f"{_fmt(kpis.get('ltd_theoretical') * 100.0, '.1f')} %")
            c3.metric("Deviation", f"{_fmt(kpis.get('deviation_pct'), '+.1f')} %")
            c4.metric("Samples", str(kpis.get("samples", 0)))
            st.caption(
                "Engineer read: use the orange line for the trend and the blue points/band for scatter. "
                "If the orange line stays below zero, the rear axle is taking more transfer than target."
            )
            if np.isfinite(kpis.get("geom_ltd_front_mean", np.nan)):
                st.caption(f"Roll-spring geometry cross-check: {_fmt(kpis.get('geom_ltd_front_mean') * 100.0, '.1f')}% front LTD.")
            _plotly_chart(fig, use_container_width=True, theme=None)
        except Exception as exc:
            st.warning(f"LTD unavailable: {exc}")
    else:
        ltd_results: dict[str, tuple[go.Figure, dict]] = {}
        for run_name, df_single in dfs.items():
            try:
                fig, kpis = dyn.lateral_load_transfer_fig(df_single)
                ltd_results[run_name] = (fig, kpis)
            except Exception as exc:
                st.warning(f"LTD unavailable ({Path(run_name).stem}): {exc}")
        if ltd_results:
            for rn, (_f, kpis) in ltd_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(f"{Path(rn).stem}: {w}")
            _show_summary_table([
                {
                    "Run": Path(rn).stem,
                    "Median front LTD [%]": _fmt(kpis.get("ltd_front_mean", float("nan")) * 100.0, ".1f"),
                    "Theoretical [%]": _fmt(kpis.get("ltd_theoretical", float("nan")) * 100.0, ".1f"),
                    "Deviation [%]": _fmt(kpis.get("deviation_pct"), "+.1f"),
                    "Samples": kpis.get("samples", 0),
                }
                for rn, (_f, kpis) in ltd_results.items()
            ])
            st.caption(
                "Engineer read: trend lines show the median deviation per run; "
                "if trend stays below zero, the rear axle is taking more transfer than target."
            )
            merged = _overlay_figures({rn: [f] for rn, (f, _k) in ltd_results.items()})
            fig_ltd = merged[0]
            fig_ltd.update_xaxes(autorange=True)
            fig_ltd.update_yaxes(autorange=True)
            _plotly_chart(fig_ltd, use_container_width=True, theme=None)

    st.divider()

    st.subheader("Understeer Angle vs LLTD")
    st.caption(
        "Equivalent to the Vehicle Balance vs LLTD slide, but keeping the existing "
        "Understeer Angle name because Vehicle Balance was the same KPI. "
        "Each point is one valid lap. X is median front LLTD in radius-filtered corners: "
        "LLTD_front = |Fz_FR - Fz_FL| / (|Fz_FR - Fz_FL| + |Fz_RR - Fz_RL|). "
        "Y is the mean understeer angle from the same corner samples. With one fixed "
        "setup, LLTD can be almost constant; the scatter becomes useful when comparing "
        "setups or runs whose load-transfer distribution actually changes."
    )
    try:
        fig, kpis = dyn.understeer_vs_lltd_fig(dfs)
        for w in kpis.get("warnings", []):
            st.warning(w)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Valid points", str(kpis.get("points", 0)))
        c2.metric("Mean LLTD", f"{_fmt(kpis.get('lltd_mean_pct'), '.3f')} %")
        c3.metric("LLTD span", f"{_fmt(kpis.get('lltd_span_pct'), '.3f')} pp")
        c4.metric("Slope", f"{_fmt(kpis.get('slope_deg_per_lltd_pct'), '+.2f')} deg/%")
        c5.metric("R²", _fmt(kpis.get("r2"), ".2f"))
        _plotly_chart(fig, use_container_width=True, theme=None)
        table = kpis.get("table")
        if table is not None and not table.is_empty():
            with st.expander("Per-lap data"):
                st.dataframe(
                    table.with_columns([
                        pl.col("LapTime [s]").round(3),
                        pl.col("Mean understeer [deg]").round(3),
                        pl.col("Front LLTD [%]").round(2),
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.warning(f"Understeer vs LLTD unavailable: {exc}")

    st.divider()

    with st.expander("Legacy · Tire Workload · Friction Circle", expanded=False):
        st.caption(
            "Radius-filtered corners using the Driver/Lap Analysis logic "
            "(R = V²/|ay| < 60 m). Utilization uses Est_FX/FY/FZ and μ tire."
        )
        for run_name, df_single in dfs.items():
            if len(dfs) > 1:
                st.markdown(f"**{Path(run_name).stem}**")
            try:
                figs, kpis = dyn.friction_circle_figs(df_single)
                for w in kpis.get("warnings", []):
                    st.warning(w)
                wheel_kpis = kpis.get("wheels", {})
                cols = st.columns(4)
                for col, wheel in zip(cols, ("FL", "FR", "RL", "RR")):
                    wk = wheel_kpis.get(wheel, {})
                    col.metric(f"{wheel} peak util", _fmt(wk.get("peak_util", float("nan")), ".2f"))
                cols = st.columns(4)
                for col, wheel in zip(cols, ("FL", "FR", "RL", "RR")):
                    wk = wheel_kpis.get(wheel, {})
                    col.metric(f"{wheel} >0.95", f"{_fmt(wk.get('sat_time_pct', float('nan')), '.1f')} %")
                for fig in [
                    fig for fig in figs
                    if "Tire utilization vs distance" not in (fig.layout.title.text or "")
                ]:
                    _plotly_chart(fig, use_container_width=True, theme=None)
            except Exception as exc:
                st.warning(f"Friction circle unavailable: {exc}")

    with st.expander("Legacy · Body Slip Angle beta", expanded=False):
        st.caption("beta = atan2(Est_vyCOG, Est_vxCOG). Apex samples use radius-curvature corners.")
        for run_name, df_single in dfs.items():
            if len(dfs) > 1:
                st.markdown(f"**{Path(run_name).stem}**")
            try:
                figs, kpis = dyn.body_slip_angle_fig(df_single)
                for w in kpis.get("warnings", []):
                    st.warning(w)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("|beta| peak", f"{_fmt(kpis.get('peak_abs_beta_deg'), '.2f')} deg")
                c2.metric("|beta| at apex", f"{_fmt(kpis.get('apex_abs_beta_deg'), '.2f')} deg")
                c3.metric("max |dbeta/dt|", f"{_fmt(kpis.get('max_beta_rate_degps'), '.1f')} deg/s")
                c4.metric("Corner samples", str(kpis.get("corner_samples", 0)))
                for fig in [
                    fig for fig in figs
                    if "Body slip angle β vs distance" not in (fig.layout.title.text or "")
                ]:
                    _plotly_chart(fig, use_container_width=True, theme=None)
            except Exception as exc:
                st.warning(f"Body slip angle unavailable: {exc}")

    with st.expander("Legacy · Tyre Balance · SA Front vs Rear", expanded=False):
        st.caption(
            "Balance index = (SA_rear - SA_front)/(SA_rear + SA_front). "
            "Positive = rear tyres use more slip angle; negative = front tyres use more."
        )
        try:
            balance_kpis = dyn.sa_balance_kpis(dfs)
            valid_kpis = {
                run_name: vals
                for run_name, vals in balance_kpis.items()
                if not vals.get("warnings")
            }
            for run_name, vals in balance_kpis.items():
                for w in vals.get("warnings", []):
                    st.warning(f"{run_name}: {w}")
            if len(valid_kpis) == 1:
                _run_name, vals = next(iter(valid_kpis.items()))
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Mean balance", _fmt(vals.get("balance_index_mean"), "+.3f"))
                c2.metric("Peak-|ay| balance", _fmt(vals.get("balance_index_at_peak"), "+.3f"))
                c3.metric("OS fraction", f"{_fmt(vals.get('os_fraction', np.nan) * 100.0, '.1f')} %")
                c4.metric("P95 SA front", f"{_fmt(vals.get('peak_sa_front_deg'), '.2f')} deg")
                c5.metric("P95 SA rear", f"{_fmt(vals.get('peak_sa_rear_deg'), '.2f')} deg")
            elif len(valid_kpis) > 1:
                _show_summary_table([
                    {
                        "Run": run_name,
                        "Mean balance": round(vals.get("balance_index_mean", np.nan), 4),
                        "Peak-|ay| balance": round(vals.get("balance_index_at_peak", np.nan), 4),
                        "OS fraction [%]": round(vals.get("os_fraction", np.nan) * 100.0, 2),
                        "P95 SA front [deg]": round(vals.get("peak_sa_front_deg", np.nan), 2),
                        "P95 SA rear [deg]": round(vals.get("peak_sa_rear_deg", np.nan), 2),
                        "Corner samples": int(vals.get("n_corner_samples", 0)),
                    }
                    for run_name, vals in valid_kpis.items()
                ])
            for fig in dyn.sa_balance_figs(dfs):
                _plotly_chart(fig, use_container_width=True, theme=None)
        except Exception as exc:
            st.warning(f"SA balance unavailable: {exc}")


def _render_dynamics_grip_factors(dfs: dict[str, pl.DataFrame]) -> None:
    """Per-lap grip factor analysis adapted to FS (no aero grip)."""
    st.subheader("Grip Factors")
    st.caption(
        "Mean grip per category over each lap, in G units "
        "(Buurman methodology, FS-adapted thresholds). "
        "Channels are independent: a sample can count for both cornering and "
        "braking, or cornering and traction. "
        "Aero grip omitted: FS speeds are too low to isolate downforce cleanly."
    )

    # ── Auto-estimate thresholds from data ─────────────────────────────
    data_key = tuple(sorted(dfs.keys()))
    if st.session_state.get("_gf_data_key") != data_key:
        st.session_state["_gf_data_key"] = data_key
        combined_df = pl.concat(list(dfs.values()), how="diagonal_relaxed")
        est = gf.estimate_thresholds(combined_df)
        st.session_state["gf_overall"] = est.overall_combined_g
        st.session_state["gf_corner_ay"] = est.cornering_ay_g
        st.session_state["gf_braking"] = est.braking_ax_g
        st.session_state["gf_traction_ax"] = est.traction_ax_g
        st.session_state["gf_traction_ay"] = est.traction_ay_g
        st.session_state["gf_min_samples"] = est.min_samples

    # ── Boundary conditions ──────────────────────────────────────────────
    with st.expander("Boundary conditions [G]"):
        st.caption(
            "Auto-estimated from the loaded data. "
            "Traction uses positive ax plus lateral load to isolate corner exit, "
            "which is more meaningful in FS than straight-line accel."
        )
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown(
                '<b style="color:#FFD700">Overall</b> '
                "— grip-limited combined acceleration",
                unsafe_allow_html=True,
            )
            overall = st.number_input(
                "Combined G >=",
                min_value=0.20, max_value=2.00, step=0.05,
                key="gf_overall", format="%.2f",
            )
        with bc2:
            st.markdown(
                '<b style="color:#D94F4F">Braking</b> '
                "— hard deceleration / trail-braking",
                unsafe_allow_html=True,
            )
            braking = st.number_input(
                "Deceleration G >=",
                min_value=0.20, max_value=2.00, step=0.05,
                key="gf_braking", format="%.2f",
            )
        bc3, bc4 = st.columns(2)
        with bc3:
            st.markdown(
                '<b style="color:#00BFBF">Cornering</b> '
                "— sustained lateral load",
                unsafe_allow_html=True,
            )
            cornering = st.number_input(
                "Lateral G >=",
                min_value=0.10, max_value=2.00, step=0.05,
                key="gf_corner_ay", format="%.2f",
            )
        with bc4:
            st.markdown(
                '<b style="color:#73D973">Traction</b> '
                "— corner exit under power",
                unsafe_allow_html=True,
            )
            traction_ax = st.number_input(
                "Acceleration G >=",
                min_value=0.10, max_value=2.00, step=0.05,
                key="gf_traction_ax", format="%.2f",
            )
            traction_ay = st.number_input(
                "Lateral G >=",
                min_value=0.10, max_value=2.00, step=0.05,
                key="gf_traction_ay", format="%.2f",
            )
        min_samples = st.number_input(
            "Min. samples per category per lap",
            min_value=5, max_value=500, step=5,
            key="gf_min_samples",
        )

    thresholds = gf.GripThresholds(
        overall_combined_g=overall,
        cornering_ay_g=cornering,
        braking_ax_g=braking,
        traction_ax_g=traction_ax,
        traction_ay_g=traction_ay,
        min_samples=min_samples,
    )

    try:
        if len(dfs) == 1:
            run_name, df = next(iter(dfs.items()))
            kpis = gf.grip_factor_kpis(df, thresholds)
            for w in kpis.get("warnings", []):
                st.warning(w)

            # ── Track maps (top — tune thresholds here) ──────────────
            if not kpis.get("warnings") and not kpis["table"].is_empty():
                lap_options = [int(l) for l in kpis["table"]["Lap"].to_list()]
                fastest = kpis.get("fastest_lap")
                default_idx = (
                    lap_options.index(int(fastest))
                    if fastest is not None and int(fastest) in lap_options
                    else 0
                )
                selected_lap = st.selectbox(
                    "Track map — lap",
                    options=lap_options,
                    index=default_idx,
                    format_func=lambda l: f"L{int(l)}",
                    key="dyn_gf_map_lap",
                )
                map_fig = gf.grip_factor_track_maps_fig(
                    df, int(selected_lap), thresholds,
                )
                _plotly_chart(map_fig, use_container_width=True, theme=None)

            # ── KPI metrics ──────────────────────────────────────────
            if not kpis.get("warnings"):
                m = kpis["means"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Overall",   f"{_fmt(m['Overall'],   '.2f')} G")
                c2.metric("Cornering", f"{_fmt(m['Cornering'], '.2f')} G")
                c3.metric("Braking",   f"{_fmt(m['Braking'],   '.2f')} G")
                c4.metric("Traction",  f"{_fmt(m['Traction'],  '.2f')} G")
                c5, c6 = st.columns(2)
                c5.metric("Valid laps", str(kpis["valid_laps"]))
                if kpis["fastest_lap"] is not None:
                    c6.metric(
                        "Fastest valid lap",
                        f"L{kpis['fastest_lap']} — {_fmt(kpis['fastest_lt'], '.2f')} s",
                    )

            # ── Evolution chart ──────────────────────────────────────
            gf_x_mode = _select_per_lap_axis("dyn_gf_axis", default="laps")
            evo_fig = gf.grip_factor_evolution_fig(
                kpis["table"], x_mode=gf_x_mode,
            )
            _plotly_chart(evo_fig, use_container_width=True, theme=None)

            if not kpis.get("warnings"):
                with st.expander("Per-lap grip factors"):
                    st.dataframe(
                        kpis["table"],
                        use_container_width=True,
                        hide_index=True,
                    )

        # ── Multi-CSV ────────────────────────────────────────────────
        else:
            run_results: dict[str, dict] = {}
            for run_name, df in dfs.items():
                run_results[run_name] = gf.grip_factor_kpis(df, thresholds)
                for w in run_results[run_name].get("warnings", []):
                    st.warning(f"{run_name}: {w}")
            valid_results = {
                rn: k for rn, k in run_results.items()
                if not k.get("warnings")
            }
            if not valid_results:
                return

            # ── Track maps (top) ─────────────────────────────────────
            map_runs = [
                rn for rn, k in valid_results.items()
                if not k["table"].is_empty()
            ]
            if map_runs:
                mc1, mc2 = st.columns([2, 1])
                with mc1:
                    map_run = st.selectbox(
                        "Track map — run",
                        options=map_runs,
                        key="dyn_gf_map_run_multi",
                    )
                run_table = valid_results[map_run]["table"]
                map_lap_opts = [int(l) for l in run_table["Lap"].to_list()]
                fastest = valid_results[map_run].get("fastest_lap")
                def_idx = (
                    map_lap_opts.index(int(fastest))
                    if fastest is not None and int(fastest) in map_lap_opts
                    else 0
                )
                with mc2:
                    map_lap = st.selectbox(
                        "Track map — lap",
                        options=map_lap_opts,
                        index=def_idx,
                        format_func=lambda l: f"L{int(l)}",
                        key="dyn_gf_map_lap_multi",
                    )
                map_fig = gf.grip_factor_track_maps_fig(
                    dfs[map_run], int(map_lap), thresholds,
                )
                _plotly_chart(map_fig, use_container_width=True, theme=None)

            # ── Summary table ────────────────────────────────────────
            _show_summary_table([
                {
                    "Run": rn,
                    "Valid laps": k["valid_laps"],
                    "Overall [G]":   round(k["means"]["Overall"],   3),
                    "Cornering [G]": round(k["means"]["Cornering"], 3),
                    "Braking [G]":   round(k["means"]["Braking"],   3),
                    "Traction [G]":  round(k["means"]["Traction"],  3),
                    "Fastest lap": k["fastest_lap"],
                    "Fastest lt [s]": (
                        round(k["fastest_lt"], 2)
                        if np.isfinite(k["fastest_lt"]) else None
                    ),
                }
                for rn, k in valid_results.items()
            ])

            # ── Evolution chart (category colours) ───────────────────
            gf_x_mode = _select_per_lap_axis("dyn_gf_axis", default="laps")
            evo_fig = gf.grip_factor_evolution_multi_fig(
                {rn: k["table"] for rn, k in valid_results.items()},
                x_mode=gf_x_mode,
            )
            _plotly_chart(evo_fig, use_container_width=True, theme=None)

            # ── Radar ────────────────────────────────────────────────
            radar_fig = gf.grip_factor_radar_fig({
                rn: k["table"] for rn, k in valid_results.items()
            })
            _plotly_chart(radar_fig, use_container_width=True, theme=None)

            with st.expander("Per-lap grip factors"):
                st.dataframe(
                    _concat_run_tables({
                        rn: k["table"] for rn, k in valid_results.items()
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.warning(f"Grip factors: {exc}")


def _tab_tc(dfs: dict[str, pl.DataFrame]) -> None:
    _render_tc_function_check(dfs)


def _tab_tv(dfs: dict[str, pl.DataFrame]) -> None:
    st.subheader("Torque Vectoring KPIs")
    tv_x_mode = _select_per_lap_axis("tv_axis", default="laps")
    try:
        if len(dfs) == 1:
            run_name, df = next(iter(dfs.items()))
            figs, kpis = tv.tv_figs_kpis(df, x_mode=tv_x_mode)
            for w in kpis.get("warnings", []):
                st.warning(w)
            if not kpis.get("warnings"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Valid laps", str(kpis["valid_laps"]))
                c2.metric("Yaw RMSE", _fmt(kpis["mean_yaw_rmse"], ".4f"))
                c3.metric("Yaw bias", _fmt(kpis["mean_yaw_bias"], "+.4f"))
                c4.metric("Corner coverage", f"{_fmt(kpis['mean_corner_coverage_pct'], '.1f')}%")
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Mz RMSE", f"{_fmt(kpis['mean_mz_rmse'], '.2f')} Nm")
                c6.metric("Mz bias", f"{_fmt(kpis['mean_mz_bias'], '+.2f')} Nm")
                c7.metric("FB / FF ratio", _fmt(kpis["mean_ratio"], ".3f"))
                c8.metric("FB share", _fmt(kpis["mean_fb_share"], ".3f"))
            filtered_figs = [
                fig for fig in figs
                if (fig.layout.title.text or "") not in {
                    f"Yaw Rate Tracking Error vs {'Lap Time' if tv_x_mode == 'laptime' else 'Lap'}",
                    f"Mz Tracking Error vs {'Lap Time' if tv_x_mode == 'laptime' else 'Lap'}",
                    f"Feedback to Feedforward Ratio vs {'Lap Time' if tv_x_mode == 'laptime' else 'Lap'}",
                }
            ]
            for fig in filtered_figs:
                _plotly_chart(fig, use_container_width=True, theme=None)
            if not kpis.get("warnings"):
                with st.expander("Per-lap data"):
                    st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {run_name: tv.tv_figs_kpis(df, x_mode=tv_x_mode) for run_name, df in dfs.items()}
            for run_name, (_figs, kpis) in run_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(f"{run_name}: {w}")
            _show_summary_table([
                {
                    "Run": run_name,
                    "Valid laps": kpis["valid_laps"],
                    "Yaw RMSE": round(kpis["mean_yaw_rmse"], 4),
                    "Yaw bias": round(kpis["mean_yaw_bias"], 4),
                    "Mz RMSE [Nm]": round(kpis["mean_mz_rmse"], 2),
                    "Mz bias [Nm]": round(kpis["mean_mz_bias"], 2),
                    "FB / FF ratio": round(kpis["mean_ratio"], 3),
                    "FB share": round(kpis["mean_fb_share"], 3),
                }
                for run_name, (_figs, kpis) in run_results.items()
                if not kpis.get("warnings")
            ])
            overlay_figs = _overlay_figures({run_name: figs for run_name, (figs, _kpis) in run_results.items()})
            filtered_figs = [
                fig for fig in overlay_figs
                if not any(
                    blocked in (fig.layout.title.text or "")
                    for blocked in (
                        "Yaw Rate Tracking Error vs ",
                        "Mz Tracking Error vs ",
                        "Feedback to Feedforward Ratio vs ",
                    )
                )
            ]
            for fig in filtered_figs:
                _plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(
                    _concat_run_tables({
                        run_name: kpis["table"]
                        for run_name, (_figs, kpis) in run_results.items()
                        if not kpis.get("warnings")
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.error(f"TV KPIs unavailable: {exc}")
    _render_tv_corner_balance(dfs)


def _tab_driver(
    dfs: dict[str, pl.DataFrame],
    file_signatures: dict[str, FileSignature],
    video_server: va.VideoServerInfo,
    raw_dfs: dict[str, pl.DataFrame],
    csv_files: list[str],
) -> None:
    driver_run_tokens = _driver_run_tokens(dfs, file_signatures)

    drv_section = st.segmented_control(
        "Driver section",
        options=["Lap Analysis", "Throttle", "Brake", "Steering", "Video Analysis"],
        default="Lap Analysis",
        required=True,
        key="driver_subsection",
        label_visibility="collapsed",
        width="stretch",
    )

    if drv_section == "Lap Analysis":
        _render_driver_lap_analysis_subtab(dfs, driver_run_tokens)
        return
    if drv_section == "Video Analysis":
        _render_driver_video_subtab(raw_dfs, file_signatures, video_server, csv_files)
        return

    summaries = _collect_driver_summaries(dfs, driver_run_tokens)
    if drv_section == "Throttle":
        _render_driver_throttle_subtab(dfs, summaries, driver_run_tokens)
    elif drv_section == "Brake":
        _render_driver_brake_subtab(dfs, summaries, driver_run_tokens)
    elif drv_section == "Steering":
        _render_driver_steering_subtab(dfs, summaries, driver_run_tokens)

    if summaries:
        with st.expander("Per-lap data"):
            for run_name, s in summaries.items():
                if s.get("valid_laps", 0) == 0:
                    continue
                if len(dfs) > 1:
                    st.markdown(f"**{run_name}**")
                st.dataframe(s["table"], use_container_width=True, hide_index=True)


def _collect_driver_summaries(
    dfs: dict[str, pl.DataFrame],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> dict[str, dict]:
    """Compute (and cache) per-run driver summaries; surface warnings inline."""
    summaries: dict[str, dict] = {}
    for run_name, df in dfs.items():
        try:
            summaries[run_name] = _driver_summary_cached(
                df,
                next(token for token in driver_run_tokens if token[0] == run_name),
            )
        except Exception as exc:
            st.error(f"Driver KPIs unavailable for `{run_name}`: {exc}")

    for run_name, s in summaries.items():
        for w in s.get("warnings", []):
            st.warning(f"{run_name}: {w}")
    return summaries


def _render_driver_circuit_map_section(
    dfs: dict[str, pl.DataFrame],
    file_signatures: dict[str, FileSignature],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> None:
    """Render the Circuit Map overview that lives above the Driver subtabs."""
    st.subheader("Circuit Map")
    st.caption(
        "■ Green = accelerating (Throttle ≥ 5 %, Brake < 5 %)  "
        "■ Red = braking (Brake ≥ 5 % and dominant)  "
        "■ Grey = coasting (both < 5 %)  "
        "■ White = plausibility (both ≥ 5 % and Throttle > Brake)"
    )

    try:
        # Build per-run entries: run_name → [(run, lap_id, laptime), ...] fastest first
        run_entries: dict[str, list[tuple[str, int, float]]] = {}
        for run_name, df in dfs.items():
            try:
                laps_arr = available_laps(df)
            except Exception:
                continue
            cols = cols_to_numpy(df, ["laps"] + (["laptime"] if "laptime" in df.columns else []))
            laps_col = cols["laps"]
            lt_col = cols.get("laptime", np.full(len(df), np.nan))
            entries: list[tuple[str, int, float]] = []
            for lap_id in laps_arr.tolist():
                lm = laps_col == float(lap_id)
                lt = (
                    float(np.nanmax(lt_col[lm]))
                    if lm.any() and np.any(np.isfinite(lt_col[lm]))
                    else np.nan
                )
                entries.append((run_name, int(lap_id), lt))
            run_entries[run_name] = sorted(
                entries, key=lambda e: e[2] if np.isfinite(e[2]) else 1e9
            )

        if not run_entries:
            st.warning("No valid laps found.")
        else:
            map_ui_rev = "|".join(
                f"{rn}:{_lap_signature(df)}" for rn, df in sorted(dfs.items())
            )

            selected_map: list[tuple[str, int]] = []
            with st.form(f"drv_map_form_{map_ui_rev}", clear_on_submit=False):
                # Keep lap picking local to this form so every click does not rerun
                # the whole Driver tab and all its KPI figures.
                sel_cols = st.columns(len(run_entries))
                for col, (run_name, entries) in zip(sel_cols, run_entries.items()):
                    color_map = dyn.build_color_map(entries)
                    lap_labels = [
                        f"L{l}  ({t:.2f} s)" if np.isfinite(t) else f"L{l}"
                        for _, l, t in entries
                    ]
                    label_to_key = {
                        lbl: (r, l)
                        for lbl, (r, l, _) in zip(lap_labels, entries)
                    }
                    with col:
                        st.markdown(f"**{run_name}**")
                        color_spans = " &nbsp;".join(
                            f'<span style="color:{color_map.get((r, l), "#ccc")}">■</span>'
                            f' <span style="color:#ccc">{lbl}</span>'
                            for lbl, (r, l, _) in zip(lap_labels, entries)
                        )
                        st.markdown(color_spans, unsafe_allow_html=True)
                        sel_labels = st.multiselect(
                            run_name,
                            options=lap_labels,
                            default=lap_labels,
                            key=f"drv_map_sel_{run_name}_{map_ui_rev}",
                            label_visibility="collapsed",
                        )
                        selected_map.extend(label_to_key[lbl] for lbl in sel_labels)
                st.form_submit_button(
                    "Apply map lap selection",
                    use_container_width=True,
                )

            if not selected_map:
                st.warning("Select at least one lap.")
            else:
                selected_map_key = tuple(sorted(selected_map))
                map_fig = _driver_circuit_map_fig_cached(
                    dfs,
                    driver_run_tokens,
                    selected_map_key,
                )
                _plotly_chart(map_fig, use_container_width=True, theme=None)

                # Tables stay independent from the map selector so they always
                # show every lap currently loaded through the sidebar filters.
                table_laps_by_run = {
                    run_name: tuple((run_name, lap_id) for _, lap_id, _ in entries)
                    for run_name, entries in run_entries.items()
                }

                # Tables: one column per run, aligned below each circuit panel
                if len(run_entries) > 1:
                    tbl_cols = st.columns(len(run_entries))
                    for col, run_name in zip(tbl_cols, run_entries.keys()):
                        table_laps = table_laps_by_run.get(run_name, ())
                        if not table_laps:
                            continue
                        stats = _driver_circuit_map_stats_cached(
                            {run_name: dfs[run_name]},
                            tuple(
                                token for token in driver_run_tokens
                                if token[0] == run_name
                            ),
                            table_laps,
                        )
                        if not stats.is_empty():
                            with col:
                                st.dataframe(
                                    style_per_lap_table(stats),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                else:
                    all_table_laps = tuple(
                        pair
                        for run_name in run_entries
                        for pair in table_laps_by_run[run_name]
                    )
                    stats_df = _driver_circuit_map_stats_cached(
                        dfs,
                        driver_run_tokens,
                        all_table_laps,
                    )
                    if not stats_df.is_empty():
                        st.dataframe(
                            style_per_lap_table(stats_df),
                            use_container_width=True,
                            hide_index=True,
                        )

    except Exception as exc:
        st.error(f"Circuit map unavailable: {exc}")


def _render_driver_throttle_subtab(
    dfs: dict[str, pl.DataFrame],
    summaries: dict[str, dict],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> None:
    st.subheader("Driver Throttle Summary")
    valid_rows = [
        {
            "Run": run_name,
            "Mean throttle [%]": round(s["mean_throttle_pct"], 1),
            "Full throttle / lap [s]": round(s["mean_full_t"], 2),
            "Full throttle [%]": round(s["mean_full_pct"], 1),
            "Off throttle [%]": round(s["mean_off_pct"], 1),
            "Median |dTP/dt| [%/s]": round(s["mean_speed"], 2),
            "Peak lap |dTP/dt| [%/s]": round(s["max_speed"], 1),
        }
        for run_name, s in summaries.items()
        if s.get("valid_laps", 0) > 0
    ]
    if valid_rows:
        st.dataframe(pl.DataFrame(valid_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Throttle Position Histogram")
    try:
        fig = _driver_throttle_histogram_fig_cached(dfs, driver_run_tokens)
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Throttle histogram unavailable: {exc}")

    st.divider()
    st.subheader("Full Throttle Time per Lap")
    full_x_mode = _select_per_lap_axis("driver_full_axis", default="laps")
    try:
        fig = _driver_full_throttle_time_fig_cached(dfs, driver_run_tokens, full_x_mode)
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Full throttle time unavailable: {exc}")

    st.divider()
    st.subheader("Throttle Speed per Lap")
    speed_x_mode = _select_per_lap_axis("driver_speed_axis", default="laps")
    try:
        fig = _driver_throttle_speed_fig_cached(dfs, driver_run_tokens, speed_x_mode)
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Throttle speed unavailable: {exc}")


def _driver_brake_zone_turns(
    dfs: dict[str, pl.DataFrame],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[tuple, str]:
    """Return (turns_signature, caption_label) for braking-zone box-plot figures.

    Picks the run with the fastest lap as the reference for turn detection and
    builds the same signature consumed by the cached figure wrappers
    (`_driver_brake_application_point_fig_cached`,
    `_driver_steering_stability_fig_cached`).
    """
    brake_candidates: list[tuple[float, str, int]] = []
    for run_name, df in dfs.items():
        fastest_lap = _driver_fastest_lap_cached(dfs, driver_run_tokens, run_name)
        if fastest_lap is None:
            continue
        lap_time_s = _lap_laptimes(df).get(int(fastest_lap), np.nan)
        brake_candidates.append((
            float(lap_time_s) if np.isfinite(lap_time_s) else np.inf,
            run_name,
            int(fastest_lap),
        ))
    if not brake_candidates:
        return (), ""
    _best_time_s, brake_ref_run, brake_ref_lap = min(
        brake_candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    balanced = _LAP_ANALYSIS_PRESETS["Balanced"]
    brake_R_thr_m = _curvature_thr_1pkm_to_R_thr_m(balanced["curvature_thr_1pkm"])
    brake_turns = _driver_cornering_turns_cached(
        dfs,
        driver_run_tokens,
        brake_R_thr_m,
        balanced["min_dur_s"],
        balanced["corner_merge_gap_m"],
        brake_ref_run,
        int(brake_ref_lap),
    )
    brake_turns_signature = _cornering_turns_signature(brake_turns)
    label = ""
    if brake_turns_signature:
        label = f"Turn labels from {Path(brake_ref_run).stem} L{int(brake_ref_lap)}."
    return brake_turns_signature, label


def _render_driver_brake_subtab(
    dfs: dict[str, pl.DataFrame],
    summaries: dict[str, dict],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> None:
    st.subheader("Driver Brake Summary")
    brake_rows = [
        {
            "Run": run_name,
            "Mean aggressiveness [%/s]": round(s["mean_brake_aggr"], 1),
            "Peak lap aggressiveness [%/s]": round(s["peak_brake_aggr"], 1),
            "Mean release smoothness [%/s]": round(s["mean_brake_release"], 1),
            "Peak lap release smoothness [%/s]": round(s["peak_brake_release"], 1),
        }
        for run_name, s in summaries.items()
        if s.get("valid_laps", 0) > 0
    ]
    if brake_rows:
        st.dataframe(pl.DataFrame(brake_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Braking Effort")
    try:
        fig = _driver_braking_effort_fig_cached(dfs, driver_run_tokens)
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Braking effort unavailable: {exc}")

    st.divider()
    st.subheader("Brake Application Point")
    st.caption(
        "Box/whisker distribution of where the driver first applies brake in "
        "repeated significant braking zones. The x-axis uses detected turn IDs "
        "when corner geometry is available. Minor brake taps are filtered out."
    )
    try:
        brake_turns_signature, turn_ref_label = _driver_brake_zone_turns(
            dfs, driver_run_tokens,
        )
        if turn_ref_label:
            st.caption(turn_ref_label)
        fig = _driver_brake_application_point_fig_cached(
            dfs,
            driver_run_tokens,
            brake_turns_signature,
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Brake application point unavailable: {exc}")

    st.divider()
    st.subheader("Braking Aggressiveness per Lap")
    brake_aggr_x_mode = _select_per_lap_axis("driver_brake_aggr_axis", default="laps")
    try:
        fig = _driver_braking_aggressiveness_fig_cached(
            dfs, driver_run_tokens, brake_aggr_x_mode,
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Braking aggressiveness unavailable: {exc}")

    st.divider()
    st.subheader("Brake Release Smoothness per Lap")
    brake_release_x_mode = _select_per_lap_axis("driver_brake_release_axis", default="laps")
    try:
        fig = _driver_brake_release_smoothness_fig_cached(
            dfs, driver_run_tokens, brake_release_x_mode,
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Brake release smoothness unavailable: {exc}")


def _render_driver_steering_subtab(
    dfs: dict[str, pl.DataFrame],
    summaries: dict[str, dict],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> None:
    st.subheader("Driver Steering Summary")
    steering_rows = [
        {
            "Run": run_name,
            "Mean smoothness [deg]": round(s["mean_steering_smoothness"], 4),
            "Peak lap smoothness [deg]": round(s["max_steering_smoothness"], 4),
            "Mean integral [deg*m]": round(s["mean_steering_integral"], 1),
            "Peak lap integral [deg*m]": round(s["max_steering_integral"], 1),
            "Mean curvature [1/m]": round(s["mean_curvature"], 5),
            "Peak lap curvature [1/m]": round(s["max_curvature"], 5),
        }
        for run_name, s in summaries.items()
        if s.get("valid_laps", 0) > 0
    ]
    if steering_rows:
        st.dataframe(pl.DataFrame(steering_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Steering Integral")
    st.caption(
        "Distance integral of absolute steering angle per lap. Higher value = "
        "more total steering demand, often a sign of understeer or a tighter line."
    )
    steering_integral_x_mode = _select_per_lap_axis(
        "driver_steering_integral_axis", default="laps",
    )
    try:
        fig = _driver_steering_integral_fig_cached(
            dfs, driver_run_tokens, steering_integral_x_mode,
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Steering integral unavailable: {exc}")

    st.divider()
    st.subheader("Steering Smoothness")
    st.caption(
        "Mean absolute difference between raw steering and a 1.0 s smoothed "
        "steering trace. Higher value = more high-frequency corrections."
    )
    steering_x_mode = _select_per_lap_axis("driver_steering_axis", default="laps")
    try:
        fig = _driver_steering_smoothness_fig_cached(
            dfs, driver_run_tokens, steering_x_mode,
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Steering smoothness unavailable: {exc}")

    st.divider()
    st.subheader("Steering Stability under Braking")
    st.caption(
        "Per significant braking event, integral of |dSteering/dt| over the "
        "samples in straight-line braking (|ay| < 0.2 g). Higher value = more "
        "steering corrections under braking (less stable). Same brake-event "
        "detection as Brake Application Point."
    )
    try:
        steer_turns_signature, steer_turn_label = _driver_brake_zone_turns(
            dfs, driver_run_tokens,
        )
        if steer_turn_label:
            st.caption(steer_turn_label)
        fig = _driver_steering_stability_fig_cached(
            dfs, driver_run_tokens, steer_turns_signature,
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Steering stability unavailable: {exc}")

    st.divider()
    st.subheader("Corner Curvature per Lap")
    curvature_x_mode = _select_per_lap_axis("driver_curvature_axis", default="laps")
    try:
        fig = _driver_corner_curvature_fig_cached(
            dfs, driver_run_tokens, curvature_x_mode,
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Corner curvature unavailable: {exc}")


def _cornering_reference_options(
    dfs: dict[str, pl.DataFrame],
) -> list[tuple[str, int, float]]:
    """Return selectable (run, lap, lap_time_s) tuples for cornering reference."""
    options: list[tuple[str, int, float]] = []
    for run_name, df in dfs.items():
        d = corn.compute_radius_curvature(df)
        for lap in np.unique(d["laps"][np.isfinite(d["laps"])]):
            mask = d["laps"] == lap
            lt = d["laptime"][mask]
            lt = lt[np.isfinite(lt)]
            lap_time_s = float(np.max(lt)) if len(lt) else np.nan
            options.append((run_name, int(lap), lap_time_s))
    return sorted(
        options,
        key=lambda item: (
            np.inf if not np.isfinite(item[2]) else item[2],
            item[0],
            item[1],
        ),
    )


def _cornering_turns_signature(turns: list[corn.TurnDef]) -> tuple:
    """Hashable turn definition for Streamlit cache invalidation."""
    return tuple(
        (
            int(t.turn_id),
            round(float(t.s_entry_m), 3),
            round(float(t.s_apex_m), 3),
            round(float(t.s_exit_m), 3),
            round(float(t.apex_lat), 8),
            round(float(t.apex_lng), 8),
        )
        for t in turns
    )


def _format_cornering_ref_option(option: tuple[str, int, float]) -> str:
    run_name, lap_id, lap_time_s = option
    base = f"{Path(run_name).stem} - Lap {int(lap_id)}"
    return f"{base} ({lap_time_s:.2f} s)" if np.isfinite(lap_time_s) else base


def _driver_lap_compare_options(
    dfs: dict[str, pl.DataFrame],
) -> list[tuple[str, int, float]]:
    options: list[tuple[str, int, float]] = []
    for run_name, df in dfs.items():
        lap_times = _lap_laptimes(df)
        for lap_id in available_laps(df).tolist():
            options.append((run_name, int(lap_id), lap_times.get(int(lap_id), np.nan)))
    return sorted(
        options,
        key=lambda item: (
            np.inf if not np.isfinite(item[2]) else item[2],
            item[0],
            item[1],
        ),
    )


def _format_driver_lap_compare_option(option: tuple[str, int, float]) -> str:
    run_name, lap_id, lap_time_s = option
    if run_name == POTENTIAL_LAP_RUN:
        base = "Potential lap"
        return f"{base} ({lap_time_s:.2f} s)" if np.isfinite(lap_time_s) else base
    base = f"{Path(run_name).stem} - L{int(lap_id)}"
    return f"{base} ({lap_time_s:.2f} s)" if np.isfinite(lap_time_s) else base


def _driver_default_compare_indices(
    lap_options: list[tuple[str, int, float]],
) -> tuple[int, int]:
    """Return default (reference, compared) indices for lap A/B selectors."""
    ref_default = 0
    cmp_default = 1 if len(lap_options) > 1 else 0
    for idx, option in enumerate(lap_options):
        if option[0] != lap_options[ref_default][0]:
            cmp_default = idx
            break
    return ref_default, cmp_default


def _format_cornering_turn_option(
    turn_id: int,
    labels: dict[int, str],
) -> str:
    return labels.get(int(turn_id), f"Turn {int(turn_id)}")


def _render_driver_cornering_section(
    dfs: dict[str, pl.DataFrame],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> None:
    st.subheader("Cornering Analysis")
    try:
        ref_options = _cornering_reference_options(dfs)
        default_ref = corn.select_reference_lap(dfs)
    except Exception as exc:
        st.warning(f"Cornering analysis unavailable: {exc}")
        return

    if not ref_options:
        st.warning("No valid laps available for cornering analysis.")
        return

    default_index = 0
    for i, (run_name, lap_id, _lap_time_s) in enumerate(ref_options):
        if (run_name, lap_id) == default_ref:
            default_index = i
            break

    with st.expander("Detection settings", expanded=False):
        R_thr_m = st.slider(
            "R threshold [m]",
            20.0,
            120.0,
            60.0,
            5.0,
            key="drv_corner_R_thr",
        )
        min_dur_s = st.slider(
            "Min corner duration [s]",
            0.2,
            2.0,
            0.5,
            0.1,
            key="drv_corner_min_dur",
        )
        merge_gap_m = st.slider(
            "Merge gap [m]",
            0.0,
            30.0,
            8.0,
            1.0,
            key="drv_corner_merge_gap",
        )
        ref_run, ref_lap, _ref_lap_time_s = st.selectbox(
            "Reference lap",
            options=ref_options,
            index=default_index,
            format_func=_format_cornering_ref_option,
            key="drv_corner_ref",
        )

    try:
        turns = _driver_cornering_turns_cached(
            dfs,
            driver_run_tokens,
            R_thr_m,
            min_dur_s,
            merge_gap_m,
            ref_run,
            ref_lap,
        )
        if not turns:
            st.warning("No corners detected with the current settings.")
            return

        turn_ids = [int(t.turn_id) for t in turns]
        turn_labels = {
            int(t.turn_id): (
                f"Turn {int(t.turn_id)} "
                f"({t.s_entry_m:.0f}-{t.s_exit_m:.0f} m, apex {t.s_apex_m:.0f} m)"
            )
            for t in turns
        }
        detected_turn_token = "-".join(str(int(turn.turn_id)) for turn in turns)
        turn_selector_key = f"drv_corner_active_turns_{ref_run}_{int(ref_lap)}_{detected_turn_token}"
        with st.expander("Corner selection", expanded=True):
            selected_turn_ids = st.multiselect(
                "Included in calculations",
                options=turn_ids,
                default=turn_ids,
                format_func=lambda turn_id: _format_cornering_turn_option(
                    turn_id, turn_labels
                ),
                key=turn_selector_key,
            )
            st.caption(
                f"{len(selected_turn_ids)} / {len(turn_ids)} detected turns included"
            )
        selected_turn_set = {int(turn_id) for turn_id in selected_turn_ids}
        active_turns = [
            turn for turn in turns if int(turn.turn_id) in selected_turn_set
        ]
        if not active_turns:
            st.warning("Select at least one detected turn for cornering analysis.")
            return

        active_turn_token = "-".join(str(int(turn.turn_id)) for turn in active_turns)

        metrics = corn.compute_turn_metrics(dfs, active_turns)
        if metrics.is_empty():
            st.warning("No cornering metrics available for the detected turns.")
            return
        st.caption(
            "Curves in A/B calculations: "
            + ", ".join(f"T{int(turn.turn_id)}" for turn in active_turns)
        )

        quick_findings = corn.corner_quick_findings_table(metrics)
        default_focus = int(active_turns[0].turn_id)
        if not quick_findings.is_empty():
            default_focus = int(quick_findings.get_column("Turn")[0])

        _plotly_chart(
            corn.corner_speed_delta_overview_fig(metrics),
            use_container_width=True,
            theme=None,
            key=f"drv_corner_speed_delta_overview_{active_turn_token}",
        )
        if not quick_findings.is_empty():
            st.dataframe(quick_findings, use_container_width=True, hide_index=True)

        focus_turn_id = st.selectbox(
            "Turn to analyse",
            options=[int(turn.turn_id) for turn in active_turns],
            index=[int(turn.turn_id) for turn in active_turns].index(default_focus),
            format_func=lambda turn_id: _format_cornering_turn_option(
                turn_id, turn_labels
            ),
            key=f"drv_corner_focus_turn_{active_turn_token}",
        )

        focus_cols = st.columns([1.05, 1.15])
        with focus_cols[0]:
            _plotly_chart(
                corn.track_map_focus_turn_fig(
                    dfs,
                    active_turns,
                    ref_run,
                    ref_lap,
                    int(focus_turn_id),
                ),
                use_container_width=True,
                theme=None,
                key=f"drv_corner_focus_map_{active_turn_token}_{focus_turn_id}",
            )
        with focus_cols[1]:
            _plotly_chart(
                corn.turn_speed_story_fig(metrics, int(focus_turn_id)),
                use_container_width=True,
                theme=None,
                key=f"drv_corner_speed_story_{active_turn_token}_{focus_turn_id}",
            )

        focus_table = corn.turn_focus_table(metrics, int(focus_turn_id))
        if not focus_table.is_empty():
            st.dataframe(focus_table, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Cornering analysis unavailable: {exc}")
        return

    st.caption(
        "Curvas detectadas a partir del radio de curvatura R = V²/|ay| en la "
        "vuelta de referencia, con separación de complejos largos por dirección "
        "lateral. Las curvas deseleccionadas en Corner selection no entran en "
        "los cálculos."
    )


_LAP_ANALYSIS_PRESETS: dict[str, dict[str, float]] = {
    "Balanced": {
        "curvature_thr_1pkm": 16.7,
        "min_dur_s": 0.5,
        "corner_merge_gap_m": 4.0,
    },
    "Tighter corners": {
        "curvature_thr_1pkm": 22.0,
        "min_dur_s": 0.4,
        "corner_merge_gap_m": 2.0,
    },
    "Open corners": {
        "curvature_thr_1pkm": 12.5,
        "min_dur_s": 0.7,
        "corner_merge_gap_m": 6.0,
    },
}

_LAP_ANALYSIS_ZONE_DEFAULTS: dict[str, float] = {
    "brake_threshold_pct": 5.0,
    "throttle_threshold_pct": 40.0,
    "min_zone_m": 10.0,
    "zone_merge_gap_m": 6.0,
}


def _curvature_thr_1pkm_to_R_thr_m(curvature_thr_1pkm: float) -> float:
    """Convert user-facing curvature threshold [1/km] to radius [m]."""
    return 1000.0 / max(float(curvature_thr_1pkm), 1.0e-6)


def _lap_analysis_settings_ui() -> dict[str, float]:
    """Compact detection controls for Lap Analysis."""
    preset = st.selectbox(
        "Detection",
        options=[*_LAP_ANALYSIS_PRESETS.keys(), "Manual"],
        index=0,
        key="drv_lap_detection_preset",
    )
    values = {**_LAP_ANALYSIS_ZONE_DEFAULTS, **_LAP_ANALYSIS_PRESETS["Balanced"]}
    if preset in _LAP_ANALYSIS_PRESETS:
        values.update(_LAP_ANALYSIS_PRESETS[preset])
        st.caption(
            f"{preset}: r≥{values['curvature_thr_1pkm']:.1f} 1/km, "
            f"corner ≥{values['min_dur_s']:.1f} s, "
            f"merge ≤{values['corner_merge_gap_m']:.0f} m"
        )
        return values

    with st.expander("Manual corner detection", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            values["curvature_thr_1pkm"] = st.slider(
                "Curvature threshold r [1/km]",
                8.0,
                35.0,
                values["curvature_thr_1pkm"],
                0.5,
                key="drv_lap_curvature_thr",
            )
        with c2:
            values["min_dur_s"] = st.slider(
                "Min corner duration [s]",
                0.2,
                2.0,
                values["min_dur_s"],
                0.1,
                key="drv_lap_corner_min_dur",
            )
        with c3:
            values["corner_merge_gap_m"] = st.slider(
                "Corner merge gap [m]",
                0.0,
                30.0,
                values["corner_merge_gap_m"],
                1.0,
                key="drv_lap_corner_merge_gap",
            )
    return values


def _render_driver_lap_analysis_subtab(
    dfs: dict[str, pl.DataFrame],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> None:
    """A/B lap analysis anchored to geometric corner phases.

    Corners come from curvature (r = 1/R) on the fastest selected lap. Apex is
    a point marker; only Entry, Exit and Straight are distance phases.
    """
    st.subheader("Lap Analysis")
    st.caption(
        "A/B comparison anchored to detected corners. Apex is a point; "
        "distance phases are Entry, Exit and Straight."
    )

    lap_options = _driver_lap_compare_options(dfs)
    if len(lap_options) < 2:
        st.warning("Select at least two laps to compare.")
        return

    with st.expander("Corner detection settings", expanded=False):
        detection_values = _lap_analysis_settings_ui()

    R_thr_m = _curvature_thr_1pkm_to_R_thr_m(detection_values["curvature_thr_1pkm"])
    potential_turns: list = []
    potential_candidates: list[tuple[float, str, int]] = []
    for run_name, df in dfs.items():
        fastest_lap = _driver_fastest_lap_cached(dfs, driver_run_tokens, run_name)
        if fastest_lap is None:
            continue
        lap_time_s = _lap_laptimes(df).get(int(fastest_lap), np.nan)
        potential_candidates.append((
            float(lap_time_s) if np.isfinite(lap_time_s) else np.inf,
            run_name,
            int(fastest_lap),
        ))
    if potential_candidates:
        _best_time_s, potential_geometry_run, potential_geometry_lap = min(
            potential_candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
    else:
        potential_geometry_run = str(lap_options[0][0])
        potential_geometry_lap = int(lap_options[0][1])
    potential_result: tuple[pl.DataFrame, dict[str, object]] | None = None
    try:
        potential_turns = _driver_cornering_turns_cached(
            dfs,
            driver_run_tokens,
            R_thr_m,
            detection_values["min_dur_s"],
            detection_values["corner_merge_gap_m"],
            potential_geometry_run,
            potential_geometry_lap,
        )
        potential_lap_end_m = lsec.lap_end_distance(
            dfs[potential_geometry_run],
            potential_geometry_lap,
        )
        potential_sectors = _driver_lap_sectors_cached(
            dfs,
            driver_run_tokens,
            potential_geometry_run,
            _cornering_turns_signature(potential_turns),
            round(float(potential_lap_end_m), 1)
            if np.isfinite(potential_lap_end_m) else np.nan,
        )
        potential_sectors_token = tuple(
            (
                int(sector.index),
                str(sector.kind),
                round(float(sector.s_start_m), 2),
                round(float(sector.s_end_m), 2),
                int(sector.turn_id) if sector.turn_id is not None else -1,
            )
            for sector in potential_sectors
        )
        potential_result = _driver_potential_lap_cached(
            dfs,
            driver_run_tokens,
            potential_sectors_token,
        )
    except Exception as exc:
        st.warning(f"Potential lap unavailable: {exc}")

    ref_options = list(lap_options)
    if potential_result is not None:
        _potential_df, potential_meta = potential_result
        ref_options = [
            (
                POTENTIAL_LAP_RUN,
                POTENTIAL_LAP_ID,
                float(potential_meta.get("lap_time_s", np.nan)),
            ),
            *lap_options,
        ]

    ref_default, _unused_cmp_default = _driver_default_compare_indices(ref_options)
    cmp_default = 0
    if ref_options[ref_default][0] != POTENTIAL_LAP_RUN:
        for idx, option in enumerate(lap_options):
            if option[0] != ref_options[ref_default][0] or option[1] != ref_options[ref_default][1]:
                cmp_default = idx
                break

    lc1, lc2 = st.columns(2)
    with lc1:
        ref_run, ref_lap, _ref_lt = st.selectbox(
            "Reference lap",
            options=ref_options,
            index=ref_default,
            format_func=_format_driver_lap_compare_option,
            key="drv_lap_cmp_ref",
        )
    with lc2:
        cmp_run, cmp_lap, _cmp_lt = st.selectbox(
            "Compared lap",
            options=lap_options,
            index=cmp_default,
            format_func=_format_driver_lap_compare_option,
            key="drv_lap_cmp_cmp",
        )

    analysis_dfs = dict(dfs)
    if potential_result is not None:
        potential_df, _potential_meta = potential_result
        analysis_dfs[POTENTIAL_LAP_RUN] = potential_df

    try:
        summary = drv.lap_comparison_summary(
            analysis_dfs, ref_run, int(ref_lap), cmp_run, int(cmp_lap)
        )
    except Exception as exc:
        st.error(f"Lap comparison unavailable: {exc}")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Reference", summary["ref_label"], f"{summary['ref_lap_time_s']:.3f} s")
    k2.metric("Compared", summary["cmp_label"], f"{summary['cmp_lap_time_s']:.3f} s")
    k3.metric("Net Δt", f"{summary['total_delta_s']:+.3f} s")
    k4.metric(
        "Lost / Gained",
        f"{summary['gross_lost_s']:.3f} / {summary['gross_gained_s']:.3f} s",
    )

    active_turns: list = []
    selected_turn_set: set[int] = set()
    turn_ids: list[int] = []
    included_turns_key = ""
    click_event_key = ""
    try:
        if ref_run == POTENTIAL_LAP_RUN:
            turns = potential_turns
            corner_run = potential_geometry_run
            corner_lap = potential_geometry_lap
        else:
            ref_lt = float(summary["ref_lap_time_s"])
            cmp_lt = float(summary["cmp_lap_time_s"])
            use_ref_for_corners = np.isfinite(ref_lt) and (
                not np.isfinite(cmp_lt) or ref_lt <= cmp_lt
            )
            corner_run = ref_run if use_ref_for_corners else cmp_run
            corner_lap = int(ref_lap if use_ref_for_corners else cmp_lap)
            turns = _driver_cornering_turns_cached(
                dfs,
                driver_run_tokens,
                R_thr_m,
                detection_values["min_dur_s"],
                detection_values["corner_merge_gap_m"],
                corner_run,
                corner_lap,
            )
        if turns:
            if ref_run == POTENTIAL_LAP_RUN:
                st.caption(
                    "Potential reference built from fastest sectors across loaded CSVs. "
                    f"Corner geometry from {Path(corner_run).stem} L{corner_lap}: "
                    f"{len(turns)} turns."
                )
            else:
                st.caption(
                    f"Corners detected on the fastest selected lap "
                    f"({corner_run} L{corner_lap}): {len(turns)} turns."
                )
            turn_ids = [int(t.turn_id) for t in turns]
            turn_labels = {
                int(t.turn_id): (
                    f"T{int(t.turn_id)} "
                    f"({t.s_entry_m:.0f}-{t.s_exit_m:.0f} m, apex {t.s_apex_m:.0f} m)"
                )
                for t in turns
            }
            detected_turn_token = "-".join(
                f"{int(turn.turn_id)}:"
                f"{round(float(turn.s_entry_m))}:"
                f"{round(float(turn.s_apex_m))}:"
                f"{round(float(turn.s_exit_m))}"
                for turn in turns
            )
            included_turns_key = (
                f"drv_lap_included_turns_{corner_run}_{int(corner_lap)}_"
                f"{detected_turn_token}"
            )
            click_event_key = (
                f"drv_lap_turn_click_event_{corner_run}_{int(corner_lap)}_"
                f"{detected_turn_token}"
            )
            if included_turns_key not in st.session_state:
                st.session_state[included_turns_key] = turn_ids
            selected_turn_set = {
                int(turn_id)
                for turn_id in st.session_state.get(included_turns_key, turn_ids)
                if int(turn_id) in set(turn_ids)
            }
            with st.expander("Curves included in Lap Analysis", expanded=True):
                st.caption(
                    "Click a highlighted curve on the map to include/exclude it."
                )
                if st.button(
                    "Include all curves",
                    key=f"drv_lap_include_all_{corner_run}_{int(corner_lap)}_{detected_turn_token}",
                ):
                    st.session_state[included_turns_key] = turn_ids
                    selected_turn_set = set(turn_ids)
                    st.rerun()
                included_names = [
                    _format_cornering_turn_option(turn_id, turn_labels)
                    for turn_id in turn_ids
                    if turn_id in selected_turn_set
                ]
                st.caption(
                    f"{len(selected_turn_set)} / {len(turn_ids)} detected curves included"
                )
                if included_names:
                    st.write(", ".join(included_names))
                else:
                    st.write("No curves selected.")
            active_turns = [
                turn for turn in turns if int(turn.turn_id) in selected_turn_set
            ]
            if not active_turns:
                st.warning("Select at least one curve to run Lap Analysis.")
    except Exception as exc:
        st.warning(f"Corner detection unavailable: {exc}")
        turns = []
        active_turns = []

    brake_thr = float(detection_values["brake_threshold_pct"])

    phase_table = pl.DataFrame()
    phase_bounds: list = []
    map_phase_bounds = drv.compute_lap_analysis_corner_phases(turns) if turns else []
    lap_gates = _lap_gates_from_run_tokens(driver_run_tokens)

    if active_turns:
        try:
            phase_table, phase_bounds = drv.corner_phase_delta_table(
                analysis_dfs, ref_run, int(ref_lap), cmp_run, int(cmp_lap),
                turns=active_turns,
                apex_half_window_m=5.0,
                brake_threshold_pct=brake_thr,
            )
        except Exception as exc:
            st.warning(f"Corner phase breakdown unavailable: {exc}")

    def _whole_lap_metrics_for_option(
        run_name: str,
        lap_id: int,
    ) -> dict[str, float] | None:
        if run_name == POTENTIAL_LAP_RUN:
            return lsec.whole_lap_metrics(analysis_dfs[run_name], int(lap_id))
        return _driver_whole_lap_metrics_cached(
            dfs,
            driver_run_tokens,
            run_name,
            int(lap_id),
        )

    def _sector_summary_for_run(run_name: str) -> dict[str, dict[str, float] | None] | None:
        fastest_lap = _driver_fastest_lap_cached(dfs, driver_run_tokens, run_name)
        if fastest_lap is None:
            return None
        run_turns = _driver_cornering_turns_cached(
            dfs,
            driver_run_tokens,
            R_thr_m,
            detection_values["min_dur_s"],
            detection_values["corner_merge_gap_m"],
            run_name,
            int(fastest_lap),
        )
        if turns:
            run_turns = [
                turn for turn in run_turns
                if int(turn.turn_id) in selected_turn_set
            ]
        turns_signature = _cornering_turns_signature(run_turns)
        lap_end_m = lsec.lap_end_distance(dfs[run_name], int(fastest_lap))
        sectors = _driver_lap_sectors_cached(
            dfs,
            driver_run_tokens,
            run_name,
            turns_signature,
            round(float(lap_end_m), 1) if np.isfinite(lap_end_m) else np.nan,
        )
        sectors_token = tuple(
            (
                int(sector.index),
                str(sector.kind),
                round(float(sector.s_start_m), 2),
                round(float(sector.s_end_m), 2),
                int(sector.turn_id) if sector.turn_id is not None else -1,
            )
            for sector in sectors
        )
        return _driver_csv_sector_summary_cached(
            dfs,
            driver_run_tokens,
            run_name,
            sectors_token,
        )

    top_left, top_right = st.columns([1.15, 0.85])
    with top_left:
        if map_phase_bounds:
            try:
                gate_ui_rev = f"{ref_run}_{ref_lap}_{cmp_run}_{cmp_lap}"
                manual_gate_line, open_fullscreen = _render_manual_gate_editor(f"driver_{gate_ui_rev}")
                if manual_gate_line is not None:
                    st.caption(
                        "Manual line: "
                        f"({manual_gate_line[0][0]:.6f}, {manual_gate_line[0][1]:.6f}) -> "
                        f"({manual_gate_line[1][0]:.6f}, {manual_gate_line[1][1]:.6f})"
                    )
                phase_fig = drv.lap_phase_track_fig(
                    analysis_dfs,
                    ref_run,
                    int(ref_lap),
                    cmp_run,
                    int(cmp_lap),
                    phases=map_phase_bounds,
                    active_turn_ids=selected_turn_set if turns else None,
                )
                phase_fig = _add_lap_detection_gates_to_fig(phase_fig, lap_gates)
                phase_fig = _add_manual_gate_line_to_fig(phase_fig, manual_gate_line)
                phase_fig_json = tmc.serialize_figure(phase_fig)
                phase_event = tmc.render_track_map_component(
                    phase_fig_json,
                    height_px=430,
                    key=f"drv_lap_phase_map_click_{ref_run}_{ref_lap}_{cmp_run}_{cmp_lap}",
                )
                _consume_track_component_event(
                    phase_event,
                    pool_len=0,
                    event_state_key=f"drv_lap_phase_manual_{ref_run}_{ref_lap}_{cmp_run}_{cmp_lap}",
                )
                if (
                    open_fullscreen
                    or bool(st.session_state.get("_dyn_track_open_fullscreen", False))
                    or bool(phase_event.get("fullscreen_event", False))
                ):
                    st.session_state["_dyn_track_open_fullscreen"] = False
                    _render_lap_phase_fullscreen_dialog(
                        phase_fig,
                        turn_ids=turn_ids,
                        included_turns_key=included_turns_key,
                        event_state_key=click_event_key,
                    )
                if turns and _consume_lap_turn_click_event(
                    phase_event,
                    all_turn_ids=turn_ids,
                    included_state_key=included_turns_key,
                    event_state_key=click_event_key,
                ):
                    st.rerun()
            except Exception as exc:
                st.warning(f"Phase map unavailable: {exc}")
        else:
            st.info("Corner phase map available once corners are detected.")

        top_runs = list(dfs.keys())[:2]
        p1_run = top_runs[0] if top_runs else None
        p2_run = top_runs[1] if len(top_runs) > 1 else None
        if p1_run is not None:
            p_caption = f"P1 = {Path(p1_run).stem}"
            if p2_run is not None:
                p_caption += f" | P2 = {Path(p2_run).stem}"
            st.caption(p_caption)
        if turns and not active_turns:
            st.info("Metrics table available once at least one curve is included.")
        elif p1_run is None:
            st.info("Metrics table unavailable without loaded CSVs.")
        else:
            try:
                p1_summary = _sector_summary_for_run(p1_run)
                p2_summary = _sector_summary_for_run(p2_run) if p2_run is not None else None
                metrics_table = lsec.build_metrics_table(
                    p1_run,
                    p2_run,
                    _whole_lap_metrics_for_option(ref_run, int(ref_lap)),
                    _whole_lap_metrics_for_option(cmp_run, int(cmp_lap)),
                    p1_summary,
                    p2_summary,
                    int(ref_lap),
                    int(cmp_lap),
                )
                st.dataframe(
                    style_metrics_table(
                        metrics_table,
                        lower_better={
                            "LapTime [s]": True,
                            "Throttle [%]": False,
                            "Braking [%]": True,
                            "Coasting [%]": True,
                            "Plausibility [%]": True,
                        },
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception as exc:
                st.warning(f"Metrics table unavailable: {exc}")

    with top_right:
        if phase_table.is_empty():
            st.info("Where is the time? is available once at least one curve is included.")
        else:
            _plotly_chart(
                drv.corner_phase_delta_fig(phase_table),
                use_container_width=True,
                theme=None,
                key=f"drv_corner_phase_bars_{ref_run}_{ref_lap}_{cmp_run}_{cmp_lap}",
            )

    # 2) Per-corner analysis: signal traces (left) + GG + Δt chart (right).
    if phase_bounds and not phase_table.is_empty():
        phase_by_id = {int(p.turn_id): p for p in phase_bounds}
        turn_by_id = {int(t.turn_id): t for t in active_turns}
        total_by_id = {
            int(row["Turn"]): float(row["Total [s]"])
            for row in phase_table.iter_rows(named=True)
        }
        corner_options = sorted(
            phase_by_id.keys(),
            key=lambda tid: -abs(total_by_id.get(tid, 0.0)),
        )
        sel_corner = st.selectbox(
            "Inspect corner",
            options=corner_options,
            format_func=lambda tid: (
                f"T{int(tid)}   Δt {total_by_id.get(int(tid), 0.0):+.3f} s"
            ),
            key=f"drv_corner_detail_sel_{ref_run}_{ref_lap}_{cmp_run}_{cmp_lap}",
        )
        detail_ref_run, detail_ref_lap = ref_run, int(ref_lap)

        detail_col, right_col = st.columns([1.0, 1.0])
        track_overview_fig: go.Figure | None = None
        track_overview_error: str | None = None
        try:
            track_overview_fig = drv.lap_comparison_track_fig(
                analysis_dfs,
                detail_ref_run,
                int(detail_ref_lap),
                cmp_run,
                int(cmp_lap),
                turns=turns,
                active_turn_ids=selected_turn_set if turns else None,
            )
            track_overview_fig = _add_lap_detection_gates_to_fig(track_overview_fig, lap_gates)
        except Exception as exc:
            track_overview_error = str(exc)

        # ── Left: signal selector controls + corner detail trace ──────────────
        with detail_col:
            signal_options = list(drv.LAP_SIGNAL_OPTIONS)
            hidden_signal = "__hidden__"
            signal_select_options = [hidden_signal, *signal_options]
            default_signal_keys = [
                "delta_s",
                "vx_mps",
                "throttle_pct",
                "brake_pct",
                "steering_deg",
                "ay_mps2",
            ]

            def _format_lap_signal_option(signal_key: str) -> str:
                if signal_key == hidden_signal:
                    return "Hide"
                return drv.LAP_SIGNAL_OPTIONS[signal_key]["label"]

            control_cols = st.columns([0.9, 1.2, 1.2, 1.2])
            with control_cols[0]:
                corner_detail_x_axis = st.selectbox(
                    "X-axis",
                    options=["distance", "time"],
                    format_func=lambda mode: "Distance" if mode == "distance" else "Time",
                    key=f"drv_corner_detail_x_axis_{ref_run}_{ref_lap}_{cmp_run}_{cmp_lap}",
                )
            selected_signal_keys: list[str] = []
            for slot_idx, default_key in enumerate(default_signal_keys[:3]):
                with control_cols[slot_idx + 1]:
                    selected_signal = st.selectbox(
                        f"Plot {slot_idx + 1}",
                        options=signal_select_options,
                        index=signal_select_options.index(default_key),
                        format_func=_format_lap_signal_option,
                        key=(
                            f"drv_corner_detail_signal_{slot_idx}_"
                            f"{ref_run}_{ref_lap}_{cmp_run}_{cmp_lap}"
                        ),
                    )
                if selected_signal != hidden_signal:
                    selected_signal_keys.append(str(selected_signal))
            lower_control_cols = st.columns(3)
            for slot_idx, default_key in enumerate(default_signal_keys[3:], start=3):
                with lower_control_cols[slot_idx - 3]:
                    selected_signal = st.selectbox(
                        f"Plot {slot_idx + 1}",
                        options=signal_select_options,
                        index=signal_select_options.index(default_key),
                        format_func=_format_lap_signal_option,
                        key=(
                            f"drv_corner_detail_signal_{slot_idx}_"
                            f"{ref_run}_{ref_lap}_{cmp_run}_{cmp_lap}"
                        ),
                    )
                if selected_signal != hidden_signal:
                    selected_signal_keys.append(str(selected_signal))

            try:
                detail_fig = drv.corner_detail_fig(
                    analysis_dfs, detail_ref_run, int(detail_ref_lap), cmp_run, int(cmp_lap),
                    turn_by_id[int(sel_corner)],
                    phases=phase_by_id[int(sel_corner)],
                    signal_keys=selected_signal_keys,
                    x_axis_mode=corner_detail_x_axis,
                )
                fullscreen_figures = {}
                fullscreen_gg_figures: dict[str, go.Figure] = {}
                fullscreen_selected = (
                    f"T{int(sel_corner)}   Δt {total_by_id.get(int(sel_corner), 0.0):+.3f} s"
                )
                for corner_id in corner_options:
                    corner_label = (
                        f"T{int(corner_id)}   Δt {total_by_id.get(int(corner_id), 0.0):+.3f} s"
                    )
                    if int(corner_id) == int(sel_corner):
                        fullscreen_figures[corner_label] = detail_fig
                        continue
                    fullscreen_figures[corner_label] = drv.corner_detail_fig(
                        analysis_dfs,
                        ref_run,
                        int(ref_lap),
                        cmp_run,
                        int(cmp_lap),
                        turn_by_id[int(corner_id)],
                        phases=phase_by_id[int(corner_id)],
                        signal_keys=selected_signal_keys,
                        x_axis_mode=corner_detail_x_axis,
                    )
                for corner_id in corner_options:
                    corner_label = (
                        f"T{int(corner_id)}   Δt {total_by_id.get(int(corner_id), 0.0):+.3f} s"
                    )
                    try:
                        fullscreen_gg_figures[corner_label] = drv.corner_gg_fig(
                            analysis_dfs,
                            ref_run,
                            int(ref_lap),
                            cmp_run,
                            int(cmp_lap),
                            phase_by_id[int(corner_id)],
                        )
                    except Exception:
                        continue
                _render_lap_detail_chart(
                    detail_fig,
                    key=(
                        f"drv_corner_detail_{detail_ref_run}_{detail_ref_lap}_{cmp_run}_{cmp_lap}_"
                        f"{int(sel_corner)}_{corner_detail_x_axis}_"
                        f"{'_'.join(selected_signal_keys) or 'empty'}"
                    ),
                    fullscreen_figures=fullscreen_figures,
                    fullscreen_selected=fullscreen_selected,
                    fullscreen_track_figure=track_overview_fig,
                    fullscreen_gg_figures=fullscreen_gg_figures,
                )
            except Exception as exc:
                st.warning(f"Corner detail unavailable: {exc}")

        # ── Right: track map + GG diagram ────────────────────────────────────
        with right_col:
            if track_overview_fig is not None:
                _plotly_chart(
                    track_overview_fig,
                    use_container_width=True,
                    theme=None,
                    key=f"drv_lap_track_{detail_ref_run}_{detail_ref_lap}_{cmp_run}_{cmp_lap}",
                )
            elif track_overview_error is not None:
                st.warning(f"Track map unavailable: {track_overview_error}")
            try:
                _plotly_chart(
                    drv.corner_gg_fig(
                        analysis_dfs,
                        detail_ref_run,
                        int(detail_ref_lap),
                        cmp_run,
                        int(cmp_lap),
                        phase_by_id[int(sel_corner)],
                    ),
                    use_container_width=True,
                    theme=None,
                    key=(
                        f"drv_corner_gg_{detail_ref_run}_{detail_ref_lap}_"
                        f"{cmp_run}_{cmp_lap}_{sel_corner}"
                    ),
                )
            except Exception as exc:
                st.warning(f"GG diagram unavailable: {exc}")

    # 5) Consistency: keep this opt-in because Streamlit executes collapsed
    # expanders and these figures scan every selected lap.
    show_consistency = st.checkbox(
        "Show consistency",
        value=False,
        key="drv_lap_show_consistency",
    )
    if show_consistency:
        st.markdown("**Lap Time Progression**")
        try:
            fig = _driver_lap_time_progression_fig_cached(dfs, driver_run_tokens)
            _plotly_chart(fig, use_container_width=True, theme=None)
        except Exception as exc:
            st.error(f"Lap time progression unavailable: {exc}")

        st.markdown("**Variability & Consistency**")
        try:
            stats = _driver_lap_consistency_stats_cached(dfs, driver_run_tokens)
            if stats.is_empty():
                st.warning("No valid laps to compute consistency stats.")
            else:
                st.dataframe(
                    style_per_lap_table(stats),
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception as exc:
            st.error(f"Consistency stats unavailable: {exc}")

        st.markdown("**Lap Time Distribution**")
        try:
            fig = _driver_lap_time_distribution_fig_cached(dfs, driver_run_tokens)
            _plotly_chart(fig, use_container_width=True, theme=None)
        except Exception as exc:
            st.error(f"Lap time distribution unavailable: {exc}")


def _render_driver_video_subtab(
    raw_dfs: dict[str, pl.DataFrame],
    file_signatures: dict[str, FileSignature],
    video_server: va.VideoServerInfo,
    csv_files: list[str],
) -> None:
    """Render the synced onboard-video workspace for one selected run."""
    st.caption(
        "Analyse one pilot at a time. The video runs continuously across the "
        "whole file, while the map and telemetry automatically switch to the "
        "current lap. Click a telemetry trace to seek the video."
    )

    run_names = list(raw_dfs)
    if not run_names:
        st.warning("No runs to display.")
        return

    selector_key = "driver_video_selected_run"
    if st.session_state.get(selector_key) not in run_names:
        st.session_state[selector_key] = run_names[0]

    selected_run = st.selectbox(
        "Pilot",
        options=run_names,
        key=selector_key,
        format_func=lambda run_name: Path(run_name).stem,
    )
    raw_df = raw_dfs[selected_run]

    compare_payload: dict | None = None
    compare_lap_id: int | None = None
    compare_sig_token = "none"
    compare_enabled = st.checkbox(
        "Compare with another lap",
        value=False,
        key="driver_video_compare_enabled",
    )
    if compare_enabled:
        default_compare_idx = (
            csv_files.index(selected_run)
            if selected_run in csv_files
            else 0
        )
        vc1, vc2 = st.columns([2, 1])
        with vc1:
            compare_file = st.selectbox(
                "Comparison CSV",
                options=csv_files,
                index=default_compare_idx,
                key="driver_video_compare_file",
                format_func=lambda fname: Path(fname).stem,
            )

        try:
            compare_sig = _file_signature(DATA_DIR / compare_file)
            compare_raw_df = (
                raw_dfs[compare_file]
                if compare_file in raw_dfs
                else load_run(str(DATA_DIR / compare_file), compare_sig)
            )
            compare_token = (compare_file, compare_sig)
            compare_lap_options = list(
                _available_laps_cached(compare_raw_df, compare_token)
            )
            if len(compare_lap_options) > 1:
                compare_lap_options = compare_lap_options[:-1]
            compare_lap_times = _lap_laptimes_cached(compare_raw_df, compare_token)
        except Exception as exc:
            st.warning(f"`{compare_file}`: comparison unavailable — {exc}")
        else:
            if compare_lap_options:
                with vc2:
                    compare_lap_id = int(st.selectbox(
                        "Comparison lap",
                        options=compare_lap_options,
                        key=f"driver_video_compare_lap_{compare_file}",
                        format_func=lambda lap, times=compare_lap_times: _format_lap_with_laptime(lap, times),
                    ))
                try:
                    compare_payload = _video_payload_cached(compare_raw_df, compare_token)
                except Exception as exc:
                    st.warning(f"`{compare_file}`: comparison payload failed — {exc}")
                    compare_payload = None
                    compare_lap_id = None
                else:
                    compare_sig_token = f"{compare_sig[0]}_{compare_sig[1]}_L{compare_lap_id}"
            else:
                st.warning(f"`{compare_file}`: no valid comparison laps.")

    try:
        selected_sig = file_signatures.get(selected_run, _file_signature(DATA_DIR / selected_run))
        payload = _video_payload_cached(raw_df, (selected_run, selected_sig))
    except KeyError as exc:
        st.error(f"`{selected_run}`: {exc}")
        return
    except Exception as exc:
        st.error(f"`{selected_run}`: video payload failed — {exc}")
        return

    video_url = va.video_url_for_csv(selected_run, video_server.available_videos)
    video_diag = va.video_diagnostics_for_csv(
        selected_run,
        video_server.diagnostics,
    )
    if video_url is None:
        st.info(
            f"No onboard video found for `{selected_run}`. "
            f"Drop a file at `videos/{Path(selected_run).stem}.mp4` to enable sync."
        )
    elif video_diag is not None:
        for warning in video_diag.warnings:
            st.warning(f"`{Path(selected_run).stem}.mp4`: {warning}")

    sig = file_signatures.get(selected_run)
    sig_token = f"{sig[0]}_{sig[1]}" if sig else "0_0"
    component_id = f"va_{Path(selected_run).stem}_{sig_token}_{compare_sig_token}"
    html = va.build_video_component_html(
        component_id=component_id,
        video_url=video_url,
        video_server_port=video_server.port,
        payload=payload,
        compare_payload=compare_payload,
        compare_lap_id=compare_lap_id,
        height_px=920,
    )
    components.html(html, height=940, scrolling=False)


def _tab_rb(dfs: dict[str, pl.DataFrame]) -> None:
    st.subheader("RB braking and regeneration behaviour")
    rb_x_mode = _select_per_lap_axis("rb_axis", default="laps")
    try:
        if len(dfs) == 1:
            run_name, df = next(iter(dfs.items()))
            figs, kpis = rb.rb_figs_kpis(df, x_mode=rb_x_mode)
            for note in kpis.get("notes", []):
                st.info(note)
            for w in kpis.get("warnings", []):
                st.warning(w)
            if not kpis.get("warnings"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Brake events", str(kpis["event_count"]))
                c2.metric("Mean decel", f"{_fmt(kpis['mean_decel_ms2'], '.2f')} m/s²")
                c3.metric("Recovered", f"{_fmt(kpis['total_recovered_wh'], '.1f')} Wh")
                c4.metric("Recovery efficiency", f"{_fmt(kpis['recovery_efficiency_pct'], '.1f')}%")
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Brake → decel gain", f"{_fmt(kpis['brake_decel_gain'], '.4f')} m/s²/%")
                c6.metric("Torque → decel gain", f"{_fmt(kpis['torque_decel_gain'], '.4f')} m/s²/Nm")
                c7.metric("Current delay", f"{_fmt(kpis['delay_current_ms'], '.0f')} ms")
                c8.metric("Current near target", f"{_fmt(kpis['current_near_target_pct'], '.1f')}%")
                c9, c10, c11, c12 = st.columns(4)
                c9.metric("Wh / s braking", f"{_fmt(kpis['regen_density_wh_s'], '.3f')}")
                c10.metric("Wh / m braking", f"{_fmt(kpis['regen_density_wh_m'], '.4f')}")
                c11.metric("Front regen share", f"{_fmt(kpis['front_regen_share_pct'], '.1f')}%")
                c12.metric("Yaw disturbance P95", f"{_fmt(kpis['yaw_disturbance_p95_radps'], '.3f')} rad/s")
                c13, c14, c15, c16 = st.columns(4)
                c13.metric("Yaw event P95", f"{_fmt(kpis['yaw_event_max_p95_radps'], '.3f')} rad/s")
                c14.metric("β peak P95", f"{_fmt(kpis['beta_peak_p95_deg'], '.2f')} deg")
                c15.metric("Lockup events", str(kpis["lockup_events_total"]))
                c16.metric("Lockup time", f"{_fmt(kpis['lockup_total_time_s'], '.2f')} s")
                c17, c18, c19 = st.columns(3)
                c17.metric("SR osc P95", f"{_fmt(kpis['sr_steady_oscillation_p95'], '.4f')}")
                c18.metric("Bias vs Fz MAE", f"{_fmt(kpis['bias_vs_fz_mae_mean_pct'], '.2f')}%")
                c19.metric("Pitch peak P95", f"{_fmt(kpis['pitch_peak_p95_radps'], '.3f')} rad/s")
            for fig in figs:
                _plotly_chart(fig, use_container_width=True, theme=None)
            if not kpis.get("warnings"):
                with st.expander("Braking events"):
                    st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
                with st.expander("Wheel locking per lap"):
                    st.dataframe(kpis["lockup_per_lap_table"], use_container_width=True, hide_index=True)
        else:
            run_results = {run_name: rb.rb_figs_kpis(df, x_mode=rb_x_mode) for run_name, df in dfs.items()}
            for run_name, (_figs, kpis) in run_results.items():
                for note in kpis.get("notes", []):
                    st.info(f"{run_name}: {note}")
                for w in kpis.get("warnings", []):
                    st.warning(f"{run_name}: {w}")
            _show_summary_table([
                {
                    "Run": run_name,
                    "Brake events": kpis["event_count"],
                    "Mean decel [m/s²]": round(kpis["mean_decel_ms2"], 3),
                    "Peak decel [m/s²]": round(kpis["peak_decel_ms2"], 3),
                    "Recovered [Wh]": round(kpis["total_recovered_wh"], 2),
                    "Recovery [%]": round(kpis["recovery_efficiency_pct"], 2),
                    "Brake-decel gain": round(kpis["brake_decel_gain"], 5),
                    "Torque-decel gain": round(kpis["torque_decel_gain"], 5),
                    "Current delay [ms]": round(kpis["delay_current_ms"], 1),
                    "Front share [%]": round(kpis["front_regen_share_pct"], 1),
                    "Yaw P95 [rad/s]": round(kpis["yaw_disturbance_p95_radps"], 3),
                    "Yaw event P95 [rad/s]": round(kpis["yaw_event_max_p95_radps"], 3),
                    "Beta peak P95 [deg]": round(kpis["beta_peak_p95_deg"], 3),
                    "Lockup events": kpis["lockup_events_total"],
                    "Lockup time [s]": round(kpis["lockup_total_time_s"], 3),
                    "SR osc P95": round(kpis["sr_steady_oscillation_p95"], 4),
                    "Bias vs Fz MAE [%]": round(kpis["bias_vs_fz_mae_mean_pct"], 2),
                    "Pitch peak P95 [rad/s]": round(kpis["pitch_peak_p95_radps"], 3),
                }
                for run_name, (_figs, kpis) in run_results.items()
                if not kpis.get("warnings")
            ])
            for fig in _overlay_figures({run_name: figs for run_name, (figs, _kpis) in run_results.items()}):
                _plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Braking events"):
                st.dataframe(
                    _concat_run_tables({
                        run_name: kpis["table"]
                        for run_name, (_figs, kpis) in run_results.items()
                        if not kpis.get("warnings")
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            with st.expander("Wheel locking per lap"):
                st.dataframe(
                    _concat_run_tables({
                        run_name: kpis["lockup_per_lap_table"]
                        for run_name, (_figs, kpis) in run_results.items()
                        if not kpis.get("warnings")
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.error(f"RB KPIs unavailable: {exc}")
    _render_rb_function_check(dfs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="CAT17x — Telemetry",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.markdown("### CAT17x")
    st.sidebar.caption("Formula Student Telemetry")
    st.sidebar.divider()

    csv_files = [p.name for p in _telemetry_csv_paths(DATA_DIR)]
    if not csv_files:
        st.error(f"No CSV files found in `{DATA_DIR}`.")
        return

    _autodetect_laps(DATA_DIR)
    video_server = _video_server_cached(
        str(REPO_ROOT),
        _video_dir_signature(REPO_ROOT),
    )
    if video_server.error:
        st.sidebar.error(f"Video assets unavailable: {video_server.error}")

    selected_files = [
        f for f in csv_files
        if st.sidebar.checkbox(f, value=True, key=f"csv_{f}")
    ]
    if not selected_files:
        st.sidebar.warning("Select at least one run.")
        return

    file_signatures = {
        fname: _file_signature(DATA_DIR / fname)
        for fname in selected_files
    }

    raw_dfs: dict[str, pl.DataFrame] = {}
    for fname in selected_files:
        try:
            raw_dfs[fname] = load_run(
                str(DATA_DIR / fname),
                file_signatures[fname],
            )
        except Exception as exc:
            st.sidebar.warning(f"Skipping `{fname}`: {exc}")
    if not raw_dfs:
        st.error("No runs could be loaded.")
        return

    _render_event_mode_selector(selected_files, raw_dfs)

    st.sidebar.divider()
    st.sidebar.markdown("### Laps")
    st.sidebar.caption("These lap selections are applied to every tab.")

    dfs: dict[str, pl.DataFrame] = {}
    run_source_files: dict[str, str] = {}
    run_file_signatures: dict[str, FileSignature] = {}
    for fname in selected_files:
        raw_df = raw_dfs.get(fname)
        if raw_df is None:
            continue
        run_token = (fname, file_signatures[fname])
        try:
            lap_options = list(_available_laps_cached(raw_df, run_token))
        except Exception as exc:
            st.sidebar.warning(f"`{fname}`: cannot list laps — {exc}")
            continue

        selected_laps = st.sidebar.multiselect(
            fname,
            options=lap_options,
            default=lap_options,
            key=f"laps_{fname}",
            format_func=_format_lap_label,
        )
        if not selected_laps:
            st.sidebar.warning(f"`{fname}`: select at least one lap.")
            continue

        try:
            dfs[fname] = _select_laps_df_cached(
                raw_df,
                run_token,
                tuple(int(lap) for lap in selected_laps),
            )
            run_source_files[fname] = fname
            run_file_signatures[fname] = file_signatures[fname]
        except Exception as exc:
            st.sidebar.warning(f"Skipping `{fname}`: {exc}")

    if not dfs:
        st.error("No runs remain after lap selection.")
        return

    st.session_state["_run_source_files"] = run_source_files
    st.session_state["_run_file_signatures"] = run_file_signatures

    section_renderers = {
        "Driver": _tab_driver,
        "Dynamics": _tab_dynamics,
        "Powertrain": _tab_powertrain,
        "TC": _tab_tc,
        "TV": _tab_tv,
        "RB": _tab_rb,
        "Events": _tab_events,
    }

    if "dashboard_section" not in st.session_state:
        st.session_state["dashboard_section"] = "Driver"

    # `st.tabs` renders every tab on each rerun, which hurts multi-run loading.
    active_section = st.segmented_control(
        "Section",
        options=list(section_renderers),
        default="Driver",
        required=True,
        key="dashboard_section",
        label_visibility="collapsed",
        width="stretch",
    )
    if active_section is not None:
        if active_section == "Driver":
            section_renderers[active_section](
                dfs,
                run_file_signatures,
                video_server,
                {run_name: raw_dfs[run_name] for run_name in dfs if run_name in raw_dfs},
                csv_files,
            )
        else:
            section_renderers[active_section](dfs)


if __name__ == "__main__":
    main()
