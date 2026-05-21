"""driver.py
----------
Driver Performance KPIs — throttle, brake, and steering behaviour analysis.

Metrics:
  1. Throttle position histogram                (% of samples per 5 % bin)
  2. Full Throttle Time per lap                 (seconds where TP > 95 %)
  3. Throttle Speed per lap                     (median |dTP/dt| with TP < 100 %
                                                 and brake released)
  4. Braking Effort                             (Brake [%] vs Filtering_VN_ax [m/s²])
  5. Brake Application Point                    (box plot by significant braking zone)
  6. Braking Aggressiveness per lap             (mean dBrake/dt for dBrake/dt > 5 %/s)
  7. Brake Release Smoothness per lap           (mean |dBrake/dt| for dBrake/dt < -5 %/s)
  8. Steering Smoothness                        (mean |Steering - smooth_1s(Steering)|
                                                 per lap)
  9. Steering Integral                          (∫|Steering| ds per lap)
 10. Corner Curvature                           (mean |ay| / vx² per lap)
 11. Steering Stability                         (∫|dSteering/dt| over straight-line braking,
                                                 box plot per significant braking zone)

All figure builders accept a `dfs: dict[str, pl.DataFrame]` so a single run
or a multi-run comparison (driver A vs driver B) share the same code path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter

from utils import (
    COMPLETE_LAPS_MARKER,
    add_lap_scatter,
    add_trend_line,
    available_laps,
    cols_to_numpy,
    ensure_complete_laps_df,
    exclude_lap0_and_last_lap,
    fill_short_false_gaps,
    keep_min_duration_segments,
    lap_dist_from_gps,
    make_dark_figure,
    per_lap_axis,
    robust_dt,
    smooth_signal,
    unique_laps,
)

from src.cornering import CornerPhases


@dataclass(frozen=True)
class LapAnalysisCornerPhase:
    """Geometric corner bounds for Lap Analysis.

    Apex is intentionally a point, not a distance window. Entry and exit are
    the curvature-bounded segments before and after that point.
    """

    turn_id: int
    s_entry_m: float
    s_apex_m: float
    s_exit_m: float

CSV_PATH = "data/run4_2025-08-24.csv"

FULL_THROTTLE_THRESHOLD = 95.0   # [%]  TP above this counts as full throttle
OFF_THROTTLE_THRESHOLD  = 5.0    # [%]  TP below this counts as throttle off
BRAKE_THRESHOLD         = 1.0    # [%]  Brake above this means driver is braking
THROTTLE_BIN_WIDTH      = 5.0    # [%]  histogram bin width
BRAKE_DERIV_THRESHOLD   = 5.0    # [%/s] threshold from the requested definition
BRAKE_EFFORT_MIN        = 5.0    # [%]   avoid low-pedal noise in the effort cloud
BRAKE_EFFORT_AX_MAX     = -0.5   # [m/s²] require actual longitudinal deceleration
BRAKE_EVENT_MIN_TIME_S  = 0.10   # [s]   ignore isolated brake spikes
BRAKE_EVENT_MAX_GAP_S   = 0.08   # [s]   bridge short drops inside one braking zone
BRAKE_SMOOTH_WINDOW_S   = 0.31   # [s]   light smoothing before differentiation
BRAKE_APPLICATION_MIN_PEAK_PCT = 15.0     # [%]    minimum peak demand for a real braking zone
BRAKE_APPLICATION_MIN_DECEL_MPS2 = -0.75  # [m/s²] minimum achieved deceleration
BRAKE_APPLICATION_MIN_DV_MPS = 2.0        # [m/s]  minimum speed drop across the event
BRAKE_APPLICATION_ZONE_GAP_FRAC = 0.035   # [-]    max start-position gap inside one zone
STEERING_SMOOTH_WINDOW_S = 0.21  # [s]   light smoothing before differentiating steering
STEERING_SMOOTHNESS_WINDOW_S = 1.00  # [s]   baseline trend for steering-correction KPI
CURVATURE_MIN_SPEED_MPS  = 3.0   # [m/s] avoid ay / vx² blow-ups near standstill
STEERING_STAB_AY_LIMIT_MPS2     = 0.2 * 9.80665  # [m/s²] |ay| < 0.2 g for "straight-line" braking
STEERING_STAB_MIN_STRAIGHT_FRAC = 0.10           # [-]    require enough straight-brake samples to avoid one-sample noise
_LONG_ACCEL_CANDIDATES  = ("Filtering_VN_ax", "VN_ax")
_LAT_ACCEL_CANDIDATES   = ("Filtering_VN_ay",)
_SPEED_CANDIDATES       = ("VN_vx",)

# Per-driver/run colours — extend if needed
DRIVER_COLORS = ("#4DB3F2", "#F27070", "#F2C94C", "#73D973")
POTENTIAL_LAP_RUN = "__potential_lap__"

_REQUIRED_COLS = ("TimeStamp", "laps", "laptime", "Throttle", "Brake")

# ── Circuit map constants ─────────────────────────────────────────────────────

_MAP_COLS = ("laps", "laptime", "VN_latitude", "VN_longitude", "Throttle", "Brake")

THROTTLE_MAP_THRESHOLD = 5.0   # [%]  Coasting: Throttle < 5 %
BRAKE_MAP_THRESHOLD    = 5.0   # [%]  Coasting: Brake    < 5 %

# Colours chosen for dark background (#141417)
MAP_PHASE_COLORS: dict[str, str] = {
    "ACCELERATING": "#4CAF50",  # green
    "BRAKING":      "#EF5350",  # red
    "COASTING":     "#78909C",  # blue-grey (readable on dark bg)
    "PLAUSIBILITY": "#FFFFFF",  # white — anomaly, needs to stand out
}

LAP_ZONE_COLORS: dict[str, str] = {
    "Braking": "#EF5350",
    "Corner": "#F2C94C",
    "Corner Entry": "#56CCF2",
    "Apex": "#F2C94C",
    "Corner Exit": "#9B51E0",
    "Acceleration": "#4CAF50",
    "Straight": "#78909C",
    "Ignored": "#3A3F46",
}

LAP_TURN_COLORS = (
    "#4DB3F2",
    "#F28C40",
    "#73D973",
    "#D973D9",
    "#F2C94C",
    "#56CCF2",
    "#EB5757",
    "#9B51E0",
    "#27AE60",
    "#F2994A",
)

LAP_SIGNAL_OPTIONS: dict[str, dict[str, str]] = {
    "delta_s": {"label": "Delta time [s]", "ylabel": "Delta [s]"},
    "loss_rate_ms_10m": {"label": "Local loss [ms/10m]", "ylabel": "ms / 10 m"},
    "vx_mps": {"label": "Speed [m/s]", "ylabel": "Speed [m/s]"},
    "throttle_pct": {"label": "Throttle [%]", "ylabel": "Throttle [%]"},
    "brake_pct": {"label": "Brake / regen demand [%]", "ylabel": "Brake [%]"},
    "steering_deg": {"label": "Steering [deg]", "ylabel": "Steering [deg]"},
    "ax_mps2": {"label": "Longitudinal acceleration [m/s²]", "ylabel": "ax [m/s²]"},
    "ay_mps2": {"label": "Lateral acceleration [m/s²]", "ylabel": "ay [m/s²]"},
    "radius_m": {"label": "Corner radius [m]", "ylabel": "Radius [m]"},
    "curvature_1pm": {"label": "Curvature [1/m]", "ylabel": "Curvature [1/m]"},
    "dvx_mps": {"label": "Speed delta [m/s]", "ylabel": "Δ speed [m/s]"},
    "dthrottle_pct": {"label": "Throttle delta [%]", "ylabel": "Δ throttle [%]"},
    "dbrake_pct": {"label": "Brake delta [%]", "ylabel": "Δ brake [%]"},
    "dsteering_deg": {"label": "Steering delta [deg]", "ylabel": "Δ steering [deg]"},
    "dax_mps2": {"label": "Longitudinal acceleration delta [m/s²]", "ylabel": "Δ ax [m/s²]"},
    "day_mps2": {"label": "Lateral acceleration delta [m/s²]", "ylabel": "Δ ay [m/s²]"},
    "dradius_m": {"label": "Corner radius delta [m]", "ylabel": "Δ radius [m]"},
    "dcurvature_1pm": {"label": "Curvature delta [1/m]", "ylabel": "Δ curvature [1/m]"},
}

_PHASE_ORDER  = ("ACCELERATING", "BRAKING", "COASTING", "PLAUSIBILITY")
_PHASE_LABELS = {
    "ACCELERATING": "Accelerating",
    "BRAKING":      "Braking",
    "COASTING":     "Coasting",
    "PLAUSIBILITY": "Plausibility (both pedals)",
}

_MAP_MAX_COLS = 4


def _driver_color(run_name: str, idx: int) -> str:
    """Stable run colour."""
    _ = run_name
    return DRIVER_COLORS[idx % len(DRIVER_COLORS)]


# ── Circuit map helpers ───────────────────────────────────────────────────────

def _classify_phases(thr: np.ndarray, brk: np.ndarray) -> np.ndarray:
    """Classify each sample into a driving phase (mutually exclusive).

    ACCELERATING : Throttle ≥ 5 %  AND  Brake < 5 %
    BRAKING      : Brake    ≥ 5 %  AND  Throttle < 5 %
                   OR both ≥ 5 %  AND  Brake ≥ Throttle  (brake-dominant overlap)
    COASTING     : both < 5 %
    PLAUSIBILITY : both ≥ 5 %  AND  Throttle > Brake  (throttle dominant — true anomaly)
    """
    phase = np.full(len(thr), "COASTING", dtype=object)
    phase[(thr >= THROTTLE_MAP_THRESHOLD) & (brk <  BRAKE_MAP_THRESHOLD)] = "ACCELERATING"
    phase[(brk >= BRAKE_MAP_THRESHOLD)    & (thr <  THROTTLE_MAP_THRESHOLD)] = "BRAKING"
    # Both pedals ≥ threshold: dominant pedal decides
    both = (thr >= THROTTLE_MAP_THRESHOLD) & (brk >= BRAKE_MAP_THRESHOLD)
    phase[both & (brk >= thr)] = "BRAKING"       # brake dominant → braking with residual throttle
    phase[both & (thr >  brk)] = "PLAUSIBILITY"  # throttle dominant → real plausibility fault
    return phase


def circuit_map_stats(
    dfs: dict[str, pl.DataFrame],
    selected: list[tuple[str, int]],
) -> pl.DataFrame:
    """Per-lap phase statistics [%] for the selected (run_name, lap_id) pairs."""
    selected_set = set(selected)
    multi_run    = len(dfs) > 1
    rows: list[dict] = []

    for run_name, df in dfs.items():
        if any(c not in df.columns for c in _MAP_COLS):
            continue
        cols = cols_to_numpy(df, ["laps", "laptime", "Throttle", "Brake"])
        laps_col = cols["laps"]
        lt_col = cols["laptime"]
        thr_col = cols["Throttle"]
        brk_col = cols["Brake"]

        for lap_id in np.unique(laps_col[np.isfinite(laps_col)]).astype(int).tolist():
            if (run_name, lap_id) not in selected_set:
                continue
            lm = laps_col == float(lap_id)
            if not lm.any():
                continue
            valid = lm & np.isfinite(thr_col) & np.isfinite(brk_col)
            if not valid.any():
                continue
            phase = _classify_phases(thr_col[valid], brk_col[valid])
            n     = int(valid.sum())
            lt_val = float(np.nanmax(lt_col[lm]))

            row: dict = {}
            if multi_run:
                row["Run"] = run_name
            row["Lap"]              = lap_id
            row["LapTime [s]"]      = round(lt_val, 2)
            row["Throttle [%]"]     = round(100.0 * float((phase == "ACCELERATING").sum()) / n, 1)
            row["Braking [%]"]      = round(100.0 * float((phase == "BRAKING").sum()) / n, 1)
            row["Coasting [%]"]     = round(100.0 * float((phase == "COASTING").sum()) / n, 1)
            row["Plausability [%]"] = round(100.0 * float((phase == "PLAUSIBILITY").sum()) / n, 1)
            rows.append(row)

    if not rows:
        return pl.DataFrame()

    pct_cols = ["Throttle [%]", "Braking [%]", "Coasting [%]", "Plausability [%]"]

    def _with_avg(tbl: pl.DataFrame, run_name: str | None = None) -> pl.DataFrame:
        """Sort by lap time and prepend one AVG row for this group."""
        tbl = tbl.sort("LapTime [s]").with_columns(pl.col("Lap").cast(pl.Utf8))
        avg: dict = {}
        if run_name is not None:
            avg["Run"] = run_name
        avg["Lap"]          = "AVG"
        avg["LapTime [s]"]  = round(float(tbl["LapTime [s]"].mean()), 2)
        for c in pct_cols:
            avg[c] = round(float(tbl[c].mean()), 1)
        avg_tbl = pl.DataFrame([avg]).with_columns(
            pl.col("Lap").cast(pl.Utf8),
            pl.col("LapTime [s]").cast(pl.Float64),
            *[pl.col(c).cast(pl.Float64) for c in pct_cols],
        )
        return pl.concat([avg_tbl, tbl])

    table = pl.DataFrame(rows)

    if multi_run:
        # Preserve the original run order
        run_order: list[str] = []
        seen: set[str] = set()
        for r in rows:
            rn = r["Run"]
            if rn not in seen:
                run_order.append(rn)
                seen.add(rn)
        pieces = [_with_avg(table.filter(pl.col("Run") == rn), rn) for rn in run_order]
        return pl.concat(pieces)
    else:
        return _with_avg(table)


def circuit_map_fig(
    dfs: dict[str, pl.DataFrame],
    selected: list[tuple[str, int]],
) -> go.Figure:
    """Side-by-side GPS track maps coloured by driving phase — ONE panel per run.

    All selected laps for a run are merged into its panel so different runs
    (drivers) can be compared side by side without overlapping.
    Phases — green: accelerating | red: braking | grey: coasting | white: plausibility.
    """
    if not selected:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#141417", plot_bgcolor="#141417",
                          font=dict(color="#EBEBEB"))
        return fig

    # Group selected laps by run, preserving dfs insertion order
    run_to_laps: dict[str, list[int]] = {}
    for run_name, lap_id in selected:
        run_to_laps.setdefault(run_name, []).append(lap_id)
    active_runs = [r for r in dfs if r in run_to_laps]

    n_panels = len(active_runs)
    if n_panels == 0:
        return go.Figure()

    # Panel titles: run name + lap list
    titles: list[str] = []
    for run_name in active_runs:
        lap_ids = sorted(run_to_laps[run_name])
        lap_str = ", ".join(f"L{l}" for l in lap_ids)
        titles.append(f"{run_name}  [{lap_str}]")

    fig = make_subplots(
        rows=1,
        cols=n_panels,
        subplot_titles=titles,
        horizontal_spacing=0.06,
    )

    fig.update_layout(
        paper_bgcolor="#141417",
        plot_bgcolor="#141417",
        font=dict(color="#EBEBEB", size=11),
        legend=dict(
            bgcolor="rgba(20,20,23,0.85)",
            bordercolor="rgba(128,128,128,0.3)",
            borderwidth=1,
            font=dict(color="#EBEBEB"),
            orientation="h",
            x=0.0,
            y=-0.10,
        ),
        margin=dict(l=10, r=10, t=55, b=80),
        height=400,
    )

    shown_phases: set[str] = set()

    for panel_idx, run_name in enumerate(active_runs):
        col = panel_idx + 1
        df  = dfs[run_name]
        if any(c not in df.columns for c in _MAP_COLS):
            continue

        cols = cols_to_numpy(df, ["laps", "Throttle", "Brake", "VN_latitude", "VN_longitude"])
        laps = cols["laps"]
        thr = cols["Throttle"]
        brk = cols["Brake"]
        lat = cols["VN_latitude"]
        lng = cols["VN_longitude"]

        # Union of all selected laps for this run
        lap_mask = np.zeros(len(laps), dtype=bool)
        for lap_id in run_to_laps[run_name]:
            lap_mask |= laps == float(lap_id)

        valid = lap_mask & np.isfinite(lat) & np.isfinite(lng) & np.isfinite(thr) & np.isfinite(brk)
        if not valid.any():
            continue

        phase_arr = _classify_phases(thr[valid], brk[valid])

        for ph in _PHASE_ORDER:
            mask = phase_arr == ph
            if not mask.any():
                continue
            show_leg = ph not in shown_phases
            if show_leg:
                shown_phases.add(ph)
            fig.add_trace(
                go.Scatter(
                    x=lng[valid][mask],
                    y=lat[valid][mask],
                    mode="markers",
                    marker=dict(color=MAP_PHASE_COLORS[ph], size=3, opacity=0.9),
                    name=_PHASE_LABELS[ph],
                    legendgroup=ph,
                    showlegend=show_leg,
                    hoverinfo="skip",
                ),
                row=1, col=col,
            )

    # Hide axes and grid for all panels
    for i in range(1, n_panels + 1):
        sfx = "" if i == 1 else str(i)
        axis_style = dict(showgrid=False, zeroline=False,
                          showticklabels=False, showline=False)
        fig.update_layout(**{
            f"xaxis{sfx}": axis_style,
            f"yaxis{sfx}": axis_style,
        })

    for ann in fig.layout.annotations:
        ann.font.color = "#EBEBEB"

    return fig


# ── Data preparation ──────────────────────────────────────────────────────────

def _prep(
    df: pl.DataFrame,
    extra_cols: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    df = ensure_complete_laps_df(df)
    required_cols = list(dict.fromkeys([*_REQUIRED_COLS, *extra_cols]))
    d = cols_to_numpy(df, required_cols)
    if COMPLETE_LAPS_MARKER in df.columns:
        d[COMPLETE_LAPS_MARKER] = cols_to_numpy(df, [COMPLETE_LAPS_MARKER])[COMPLETE_LAPS_MARKER]
    d["time"] = d["TimeStamp"] - d["TimeStamp"][0]
    return d


def _longitudinal_accel_col(df: pl.DataFrame) -> str:
    """Best available longitudinal-acceleration signal for braking effort."""
    for col in _LONG_ACCEL_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(
        "Missing longitudinal acceleration column. Expected one of "
        f"{list(_LONG_ACCEL_CANDIDATES)}."
    )


def _lateral_accel_col(df: pl.DataFrame) -> str:
    """Best available lateral-acceleration signal for curvature."""
    for col in _LAT_ACCEL_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(
        "Missing lateral acceleration column. Expected one of "
        f"{list(_LAT_ACCEL_CANDIDATES)}."
    )


def _speed_col(df: pl.DataFrame) -> str:
    """Best available longitudinal speed signal for curvature."""
    for col in _SPEED_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(
        "Missing speed column for curvature. Expected one of "
        f"{list(_SPEED_CANDIDATES)}."
    )


def _prep_steering_metrics(df: pl.DataFrame) -> tuple[dict[str, np.ndarray], str, str]:
    """Prepare steering metric arrays and attach lap distance when available."""
    lat_accel_col = _lateral_accel_col(df)
    speed_col = _speed_col(df)
    prepared_df = ensure_complete_laps_df(df)
    d = _prep(prepared_df, extra_cols=("Steering", lat_accel_col, speed_col))

    try:
        dist_m = lap_dist_from_gps(prepared_df)
    except (KeyError, ValueError):
        dist_m = None
    if dist_m is not None and len(dist_m) == len(d["time"]):
        d["dist_m"] = dist_m

    return d, lat_accel_col, speed_col


def _savgol_window_samples(n_samples: int, dt_s: float, window_s: float) -> int:
    """Odd Savitzky-Golay window length compatible with *n_samples*."""
    if n_samples < 5 or not np.isfinite(dt_s) or dt_s <= 0.0:
        return 0
    window = max(5, int(round(window_s / dt_s)))
    if window % 2 == 0:
        window += 1
    if window > n_samples:
        window = n_samples if n_samples % 2 == 1 else n_samples - 1
    return window if window >= 5 else 0


def _smooth_brake_signal(brake_pct: np.ndarray, dt_s: float) -> np.ndarray:
    """Lightly smooth the brake channel before differentiation."""
    window = _savgol_window_samples(len(brake_pct), dt_s, BRAKE_SMOOTH_WINDOW_S)
    if window == 0:
        return brake_pct.copy()
    smoothed = savgol_filter(brake_pct, window_length=window, polyorder=2, mode="interp")
    return np.clip(smoothed, 0.0, 100.0)


def _smooth_steering_signal(
    steering_rad: np.ndarray,
    dt_s: float,
    window_s: float = STEERING_SMOOTH_WINDOW_S,
) -> np.ndarray:
    """Smooth steering over *window_s* seconds."""
    window = _savgol_window_samples(len(steering_rad), dt_s, window_s)
    if window == 0:
        return steering_rad.copy()
    return savgol_filter(steering_rad, window_length=window, polyorder=2, mode="interp")


def _steering_integral_degm(steering_rad: np.ndarray, dist_m: np.ndarray) -> float:
    """Distance integral of absolute steering angle [deg*m]."""
    valid = np.isfinite(steering_rad) & np.isfinite(dist_m)
    if int(valid.sum()) < 2:
        return np.nan

    steering_abs_deg = np.abs(np.rad2deg(steering_rad[valid]))
    s_m = dist_m[valid]
    ds_m = np.diff(s_m)
    positive = np.isfinite(ds_m) & (ds_m > 0.0)
    if not positive.any():
        return np.nan

    area = 0.5 * (steering_abs_deg[:-1] + steering_abs_deg[1:]) * ds_m
    return float(np.sum(area[positive]))


def _brake_rate_arrays(
    time_s: np.ndarray,
    brake_pct: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return smoothed brake, dBrake/dt, and a cleaned brake-event mask."""
    dt_s = robust_dt(time_s)
    brake_smooth = _smooth_brake_signal(brake_pct, dt_s)
    brake_rate = np.gradient(brake_smooth, time_s)
    brake_event = keep_min_duration_segments(
        brake_smooth >= BRAKE_THRESHOLD,
        BRAKE_EVENT_MIN_TIME_S,
        dt_s,
    )
    return brake_smooth, brake_rate, brake_event


def _true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive [start, end] index pairs for true segments."""
    if not np.any(mask):
        return []
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _braking_effort_events(
    d: dict[str, np.ndarray],
    accel_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-event max Brake and min longitudinal acceleration."""
    valid = np.all(
        np.stack(
            [
                np.isfinite(d["time"]),
                np.isfinite(d["laps"]),
                np.isfinite(d["Brake"]),
                np.isfinite(d[accel_col]),
            ],
            axis=1,
        ),
        axis=1,
    )
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)

    time_s = d["time"]
    laps = d["laps"]
    brake_pct = d["Brake"]
    accel_arr = d[accel_col]
    dt_s = robust_dt(time_s)
    brake_smooth = _smooth_brake_signal(brake_pct, dt_s)

    event_mask = keep_min_duration_segments(
        brake_smooth >= BRAKE_EFFORT_MIN,
        BRAKE_EVENT_MIN_TIME_S,
        dt_s,
    )
    event_mask = fill_short_false_gaps(event_mask, BRAKE_EVENT_MAX_GAP_S, dt_s)

    event_laps: list[int] = []
    max_brake_pct: list[float] = []
    min_accel_mps2: list[float] = []
    duration_s: list[float] = []

    for lap in unique_laps(laps):
        lap_mask = (laps == lap) & event_mask
        for start_idx, end_idx in _true_segments(lap_mask):
            seg = slice(start_idx, end_idx + 1)
            event_min_accel = float(np.nanmin(accel_arr[seg]))
            if event_min_accel > BRAKE_EFFORT_AX_MAX:
                continue
            event_laps.append(int(lap))
            max_brake_pct.append(float(np.nanmax(brake_pct[seg])))
            min_accel_mps2.append(event_min_accel)
            duration_s.append(float(time_s[end_idx] - time_s[start_idx]))

    return (
        np.asarray(event_laps, dtype=int),
        np.asarray(max_brake_pct, dtype=float),
        np.asarray(min_accel_mps2, dtype=float),
        np.asarray(duration_s, dtype=float),
    )


def _prep_brake_application(
    df: pl.DataFrame,
    accel_col: str,
    speed_col: str,
    extra_cols: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """Prepare aligned arrays for distance-based brake-application analysis."""
    df = ensure_complete_laps_df(df)
    required_cols = list(
        dict.fromkeys([*_REQUIRED_COLS, accel_col, speed_col, *extra_cols])
    )
    d = cols_to_numpy(df, required_cols)
    d["dist_m"] = lap_dist_from_gps(df)
    if COMPLETE_LAPS_MARKER in df.columns:
        d[COMPLETE_LAPS_MARKER] = cols_to_numpy(df, [COMPLETE_LAPS_MARKER])[COMPLETE_LAPS_MARKER]
    d["time"] = d["TimeStamp"] - d["TimeStamp"][0]
    return d


def _iter_significant_brake_event_segments(
    d: dict[str, np.ndarray],
    accel_col: str,
    speed_col: str,
) -> list[tuple[int, int, dict[str, float]]]:
    """Yield (start_idx, end_idx, base_meta) for each qualifying brake event.

    `base_meta` carries the per-event geometry shared by every consumer of
    significant brake events: lap, start_dist_m, progress, lap_len_m,
    duration_s, peak_brake_pct, min_accel_mps2, speed_drop_mps. The caller
    decides what additional value to compute over `[start_idx, end_idx]`.
    """
    time_s = d["time"]
    laps = d["laps"]
    brake_pct = d["Brake"]
    accel_mps2 = d[accel_col]
    speed_mps = d[speed_col]
    dist_m = d["dist_m"]
    dt_s = robust_dt(time_s)
    brake_smooth = _smooth_brake_signal(brake_pct, dt_s)

    event_mask = keep_min_duration_segments(
        brake_smooth >= BRAKE_EFFORT_MIN,
        BRAKE_EVENT_MIN_TIME_S,
        dt_s,
    )
    event_mask = fill_short_false_gaps(event_mask, BRAKE_EVENT_MAX_GAP_S, dt_s)

    out: list[tuple[int, int, dict[str, float]]] = []
    for lap in unique_laps(laps):
        lap_mask = laps == lap
        if not lap_mask.any():
            continue

        lap_dist = dist_m[lap_mask]
        lap_len_m = float(np.nanmax(lap_dist))
        if not np.isfinite(lap_len_m) or lap_len_m <= 10.0:
            continue

        for start_idx, end_idx in _true_segments(lap_mask & event_mask):
            seg = slice(start_idx, end_idx + 1)
            peak_brake_pct = float(np.nanmax(brake_pct[seg]))
            min_accel_mps2 = float(np.nanmin(accel_mps2[seg]))
            entry_speed_mps = float(speed_mps[start_idx])
            min_speed_mps = float(np.nanmin(speed_mps[seg]))
            speed_drop_mps = entry_speed_mps - min_speed_mps

            if peak_brake_pct < BRAKE_APPLICATION_MIN_PEAK_PCT:
                continue
            if min_accel_mps2 > BRAKE_APPLICATION_MIN_DECEL_MPS2:
                continue
            if speed_drop_mps < BRAKE_APPLICATION_MIN_DV_MPS:
                continue

            start_dist_m = float(dist_m[start_idx])
            progress = start_dist_m / lap_len_m
            base = {
                "lap": float(lap),
                "start_dist_m": start_dist_m,
                "progress": float(np.clip(progress, 0.0, 1.0)),
                "lap_len_m": lap_len_m,
                "duration_s": float(time_s[end_idx] - time_s[start_idx]),
                "peak_brake_pct": peak_brake_pct,
                "min_accel_mps2": min_accel_mps2,
                "speed_drop_mps": float(speed_drop_mps),
            }
            out.append((start_idx, end_idx, base))
    return out


def _significant_brake_application_events(
    df: pl.DataFrame,
    accel_col: str,
    speed_col: str,
) -> list[dict[str, float]]:
    """Return significant brake-on events with their application distance.

    A significant zone needs a sustained brake signal, meaningful peak demand,
    real deceleration, and a measurable speed drop. This avoids creating boxes
    for every minor lift/tap before a geometric corner.
    """
    d = _prep_brake_application(df, accel_col, speed_col)
    valid = np.all(
        np.stack(
            [
                np.isfinite(d["time"]),
                np.isfinite(d["laps"]),
                np.isfinite(d["Brake"]),
                np.isfinite(d[accel_col]),
                np.isfinite(d[speed_col]),
                np.isfinite(d["dist_m"]),
            ],
            axis=1,
        ),
        axis=1,
    )
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)
    return [base for _, _, base in _iter_significant_brake_event_segments(d, accel_col, speed_col)]


def _steering_stability_events(
    df: pl.DataFrame,
    accel_col: str,
    speed_col: str,
    lat_accel_col: str,
) -> list[dict[str, float]]:
    """Return steering stability per significant braking event [deg].

    Uses the same brake-event detection as `_significant_brake_application_events`
    (peak brake / decel / Δv filters). Zone clustering happens later on this
    complete event list, so the x-axis zones match Brake Application Point.
    For each event, integrates |dSteering/dt| over the samples where the car is
    in straight-line braking — i.e. |lateral acceleration| < STEERING_STAB_AY_LIMIT_MPS2
    (≈ 0.2 g, matching the slide). Higher values mean the driver did more
    corrective steering while braking.
    """
    d = _prep_brake_application(
        df, accel_col, speed_col, extra_cols=(lat_accel_col, "Steering"),
    )
    # Same `valid` mask as _significant_brake_application_events — the brake-event
    # detection must be identical. NaNs in steering/ay are handled below.
    valid = np.all(
        np.stack(
            [
                np.isfinite(d["time"]),
                np.isfinite(d["laps"]),
                np.isfinite(d["Brake"]),
                np.isfinite(d[accel_col]),
                np.isfinite(d[speed_col]),
                np.isfinite(d["dist_m"]),
            ],
            axis=1,
        ),
        axis=1,
    )
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)

    time_s = d["time"]
    ay_mps2 = d[lat_accel_col]
    steering_rad = d["Steering"]

    dt_s = robust_dt(time_s)
    steering_finite = np.isfinite(steering_rad)
    ay_finite = np.isfinite(ay_mps2)
    # Linear interpolation across NaN gaps keeps the Savitzky-Golay smoother
    # continuous. Filling with zero here was wrong — the artificial step would
    # leak a giant fake gradient spike into the ~½-window of finite samples on
    # either side of the gap, inflating the integrated rate. Events that touch
    # any NaN within the smoothing pad are rejected below, so this scaffolding
    # never contributes to a reported value.
    steering_for_smoothing = steering_rad.copy()
    if not steering_finite.all() and steering_finite.any():
        finite_idx = np.flatnonzero(steering_finite)
        nan_idx = np.flatnonzero(~steering_finite)
        steering_for_smoothing[nan_idx] = np.interp(
            nan_idx, finite_idx, steering_rad[finite_idx],
        )
    steering_smooth = _smooth_steering_signal(steering_for_smoothing, dt_s)
    steer_rate_abs_degps = np.abs(np.rad2deg(np.gradient(steering_smooth, time_s)))
    straight_mask = ay_finite & (np.abs(ay_mps2) < STEERING_STAB_AY_LIMIT_MPS2)

    n_samples = len(time_s)
    # Contamination radius of one interpolated sample on the rate signal:
    # Savitzky-Golay half-width  +  one extra sample for the central-difference
    # gradient stencil. Use the actual SG window the smoother chose to avoid an
    # off-by-one when the configured window doesn't divide evenly by dt.
    sg_window = _savgol_window_samples(n_samples, dt_s, STEERING_SMOOTH_WINDOW_S)
    sg_pad = (sg_window // 2 + 1) if sg_window else 1

    rows: list[dict[str, float]] = []
    for start_idx, end_idx, base in _iter_significant_brake_event_segments(d, accel_col, speed_col):
        seg = slice(start_idx, end_idx + 1)
        row = dict(base)
        row["steering_stability_deg"] = np.nan
        row["straight_frac"] = np.nan

        win_lo = max(0, start_idx - sg_pad)
        win_hi = min(n_samples, end_idx + 1 + sg_pad)
        # Reject any event whose smoothing window touches a NaN — that's where
        # the contamination would creep in. Keep the row so zone detection stays
        # identical to Brake Application Point.
        if not steering_finite[win_lo:win_hi].all():
            rows.append(row)
            continue
        if not ay_finite[seg].all():
            rows.append(row)
            continue
        seg_straight = straight_mask[seg]
        straight_frac = float(seg_straight.mean())
        row["straight_frac"] = straight_frac
        if not seg_straight.any():
            rows.append(row)
            continue
        if straight_frac < STEERING_STAB_MIN_STRAIGHT_FRAC:
            rows.append(row)
            continue
        seg_t = time_s[seg]
        seg_rate = np.where(seg_straight, steer_rate_abs_degps[seg], 0.0)
        steering_stability_deg = float(np.trapezoid(seg_rate, seg_t))
        if np.isfinite(steering_stability_deg):
            row["steering_stability_deg"] = steering_stability_deg
        rows.append(row)
    return rows


def _assign_brake_application_zones(events: list[dict]) -> list[dict]:
    """Cluster significant brake events into repeated track zones."""
    if not events:
        return []

    order = np.argsort([float(row["progress"]) for row in events], kind="mergesort")
    clusters: list[list[int]] = [[int(order[0])]]
    prev_progress = float(events[int(order[0])]["progress"])

    for idx_raw in order[1:]:
        idx = int(idx_raw)
        progress = float(events[idx]["progress"])
        if progress - prev_progress <= BRAKE_APPLICATION_ZONE_GAP_FRAC:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
        prev_progress = progress

    selected_laps = {
        (str(row["run"]), int(row["lap"]))
        for row in events
    }
    min_zone_events = 1 if len(selected_laps) <= 1 else 2

    cluster_meta: list[tuple[float, list[int]]] = []
    for cluster in clusters:
        if len(cluster) < min_zone_events:
            continue
        progress_vals = np.asarray([float(events[idx]["progress"]) for idx in cluster])
        angles = 2.0 * np.pi * progress_vals
        center = float(
            (
                np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
                / (2.0 * np.pi)
            ) % 1.0
        )
        cluster_meta.append((center, cluster))

    if not cluster_meta:
        return []

    cluster_meta.sort(key=lambda item: item[0])
    out: list[dict] = []
    for zone_idx, (center, cluster) in enumerate(cluster_meta, start=1):
        zone_label = f"Zone {zone_idx}"
        for event_idx in cluster:
            row = events[event_idx].copy()
            row["zone"] = zone_label
            row["zone_center_progress"] = center
            out.append(row)
    return _first_event_per_zone_lap(out)


def _braking_turn_label(
    start_dist_m: float,
    turns: list[object] | None,
) -> tuple[str | None, int | None]:
    """Return the turn label for a brake application point."""
    if not turns or not np.isfinite(start_dist_m):
        return None, None

    candidates = [
        turn for turn in turns
        if np.isfinite(float(turn.s_entry_m))
        and np.isfinite(float(turn.s_apex_m))
        and np.isfinite(float(turn.s_exit_m))
    ]
    if not candidates:
        return None, None
    candidates.sort(key=lambda turn: float(turn.s_entry_m))

    for turn in candidates:
        turn_id = int(turn.turn_id)
        if (
            float(turn.s_entry_m) <= start_dist_m <= float(turn.s_exit_m)
            and start_dist_m <= float(turn.s_apex_m)
        ):
            return f"T{turn_id} Braking", turn_id

    ahead = [
        turn for turn in candidates
        if float(turn.s_entry_m) >= start_dist_m
    ]
    if ahead:
        turn = min(ahead, key=lambda item: float(item.s_entry_m))
    else:
        turn = candidates[0]
    turn_id = int(turn.turn_id)
    return f"T{turn_id} Braking", turn_id


def _label_brake_application_zones(
    events: list[dict],
    turns: list[object] | None,
) -> list[dict]:
    """Rename clustered braking zones with their associated detected turn."""
    if not turns or not events:
        return events

    zone_order = {
        zone: idx
        for idx, zone in enumerate(
            sorted(
                {str(row["zone"]) for row in events},
                key=lambda label: int(label.split()[-1]),
            )
        )
    }
    zone_labels: dict[str, tuple[str, int | None]] = {}
    used_labels: set[str] = set()
    for zone in zone_order:
        rows = [row for row in events if str(row["zone"]) == zone]
        first_application_m = float(np.nanmin([
            float(row["start_dist_m"]) for row in rows
        ]))
        label, turn_id = _braking_turn_label(first_application_m, turns)
        if label is None:
            label = zone
        if label in used_labels:
            label = f"{label} ({zone})"
        used_labels.add(label)
        zone_labels[zone] = (label, turn_id)

    out: list[dict] = []
    for row in events:
        old_zone = str(row["zone"])
        label, turn_id = zone_labels[old_zone]
        mapped = row.copy()
        mapped["zone_raw"] = old_zone
        mapped["zone"] = label
        mapped["turn_id"] = turn_id
        mapped["zone_order"] = zone_order[old_zone]
        out.append(mapped)
    return out


def _first_event_per_zone_lap(events: list[dict]) -> list[dict]:
    """Keep one brake-application point per run/lap/zone."""
    first: dict[tuple[str, int, str], dict] = {}
    for row in events:
        key = (str(row["run"]), int(row["lap"]), str(row["zone"]))
        current = first.get(key)
        if current is None or float(row["start_dist_m"]) < float(current["start_dist_m"]):
            first[key] = row
    return list(first.values())


def _steering_metrics_per_lap(
    d: dict[str, np.ndarray],
    lat_accel_col: str,
    speed_col: str,
) -> dict:
    """Return steering smoothness, integral, and curvature metrics per lap."""
    valid = np.all(
        np.stack(
            [
                np.isfinite(d["time"]),
                np.isfinite(d["laps"]),
                np.isfinite(d["laptime"]),
                np.isfinite(d["Steering"]),
                np.isfinite(d[lat_accel_col]),
                np.isfinite(d[speed_col]),
            ],
            axis=1,
        ),
        axis=1,
    )
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)

    time_s = d["time"]
    laps = d["laps"]
    laptime = d["laptime"]
    steering_rad = d["Steering"]
    ay_mps2 = d[lat_accel_col]
    vx_mps = np.abs(d[speed_col])

    curvature_inv_m = np.full(len(vx_mps), np.nan)
    curvature_mask = np.isfinite(ay_mps2) & np.isfinite(vx_mps) & (vx_mps >= CURVATURE_MIN_SPEED_MPS)
    curvature_inv_m[curvature_mask] = np.abs(ay_mps2[curvature_mask]) / np.square(vx_mps[curvature_mask])

    lap_list = unique_laps(laps)
    n = len(lap_list)
    lt_val = np.full(n, np.nan)
    mean_steering_smoothness = np.full(n, np.nan)
    steering_integral_degm = np.full(n, np.nan)
    mean_curvature = np.full(n, np.nan)

    for i, lap in enumerate(lap_list):
        lm = laps == lap
        if not lm.any():
            continue
        lt_val[i] = float(np.nanmax(laptime[lm]))
        lap_steering = steering_rad[lm]
        lap_dt_s = robust_dt(time_s[lm])
        steering_baseline_rad = _smooth_steering_signal(
            lap_steering, lap_dt_s, STEERING_SMOOTHNESS_WINDOW_S,
        )
        steering_smoothness_deg = np.abs(np.rad2deg(lap_steering - steering_baseline_rad))
        if np.isfinite(steering_smoothness_deg).any():
            mean_steering_smoothness[i] = float(np.nanmean(steering_smoothness_deg))
        if "dist_m" in d:
            steering_integral_degm[i] = _steering_integral_degm(
                lap_steering, d["dist_m"][lm],
            )
        cm = lm & np.isfinite(curvature_inv_m)
        if cm.any():
            mean_curvature[i] = float(np.nanmean(curvature_inv_m[cm]))

    table = pl.DataFrame({
        "Lap": lap_list.astype(int),
        "Steering smoothness [deg]": np.round(mean_steering_smoothness, 4),
        "Steering integral [deg*m]": np.round(steering_integral_degm, 1),
        "Curvature [1/m]": np.round(mean_curvature, 5),
    })

    return {
        "lap_list": lap_list,
        "lt_val": lt_val,
        "mean_steering_smoothness": mean_steering_smoothness,
        "steering_integral_degm": steering_integral_degm,
        "mean_curvature": mean_curvature,
        "table": table,
    }


def _per_lap(d: dict[str, np.ndarray]) -> dict:
    valid = np.all(
        np.stack([np.isfinite(d[k]) for k in _REQUIRED_COLS], axis=1), axis=1,
    )
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)

    dt      = robust_dt(d["time"])
    laps    = d["laps"]
    laptime = d["laptime"]
    thr     = d["Throttle"]
    brk     = d["Brake"]
    _, brake_rate, brake_event = _brake_rate_arrays(d["time"], brk)

    # Throttle speed: dTP/dt; ignore saturation and braking transitions.
    # Per-lap aggregate is the median |dTP/dt| (robust against the long tail
    # from fast transitions, which would dominate the mean on 100 Hz data).
    thr_speed  = np.gradient(thr, d["time"])
    speed_mask = (thr < 100.0) & (brk < BRAKE_THRESHOLD) & np.isfinite(thr_speed)

    full_thr = thr > FULL_THROTTLE_THRESHOLD
    off_thr  = thr < OFF_THROTTLE_THRESHOLD

    lap_list = unique_laps(laps)
    n        = len(lap_list)
    lt_val      = np.full(n, np.nan)
    full_t      = np.full(n, np.nan)   # [s]
    full_pct    = np.full(n, np.nan)   # [%]
    off_pct     = np.full(n, np.nan)   # [%]
    mean_thr    = np.full(n, np.nan)   # [%]
    mean_speed  = np.full(n, np.nan)   # [%/s]
    mean_brake_aggr    = np.full(n, np.nan)   # [%/s]
    mean_brake_release = np.full(n, np.nan)   # [%/s]
    n_speed_pts = np.zeros(n, dtype=int)

    for i, lap in enumerate(lap_list):
        lm = laps == lap
        if not lm.any():
            continue
        lt_val[i]   = laptime[lm].max()
        full_t[i]   = float(full_thr[lm].sum() * dt)
        full_pct[i] = 100.0 * float(np.mean(full_thr[lm]))
        off_pct[i]  = 100.0 * float(np.mean(off_thr[lm]))
        mean_thr[i] = float(np.nanmean(thr[lm]))
        sm = lm & speed_mask
        if sm.any():
            mean_speed[i]  = float(np.nanmedian(np.abs(thr_speed[sm])))
            n_speed_pts[i] = int(sm.sum())
        am = lm & brake_event & np.isfinite(brake_rate) & (brake_rate > BRAKE_DERIV_THRESHOLD)
        if am.any():
            mean_brake_aggr[i] = float(np.nanmean(brake_rate[am]))
        rm = lm & brake_event & np.isfinite(brake_rate) & (brake_rate < -BRAKE_DERIV_THRESHOLD)
        if rm.any():
            mean_brake_release[i] = float(np.nanmean(np.abs(brake_rate[rm])))

    table = pl.DataFrame({
        "Lap":                    lap_list.astype(int),
        "LapTime [s]":            np.round(lt_val, 3),
        "Mean throttle [%]":      np.round(mean_thr, 1),
        "Full throttle time [s]": np.round(full_t, 2),
        "Full throttle [%]":      np.round(full_pct, 1),
        "Off throttle [%]":       np.round(off_pct, 1),
        "Median |dTP/dt| [%/s]":  np.round(mean_speed, 2),
        "Brake aggressiveness [%/s]":     np.round(mean_brake_aggr, 2),
        "Brake release smoothness [%/s]": np.round(mean_brake_release, 2),
    })

    return {
        "lap_list":   lap_list,
        "lt_val":     lt_val,
        "full_t":     full_t,
        "full_pct":   full_pct,
        "off_pct":    off_pct,
        "mean_thr":   mean_thr,
        "mean_speed": mean_speed,
        "mean_brake_aggr":    mean_brake_aggr,
        "mean_brake_release": mean_brake_release,
        "table":      table,
    }


# ── Public KPI helpers ────────────────────────────────────────────────────────

def driver_summary(df: pl.DataFrame) -> dict:
    """Single-run driver summary stats and per-lap table."""
    d   = _prep(df)
    res = _per_lap(d)
    steer_res: dict | None = None
    if "Steering" in df.columns:
        try:
            steer_d, lat_accel_col, speed_col = _prep_steering_metrics(df)
            steer_res = _steering_metrics_per_lap(steer_d, lat_accel_col, speed_col)
            res["table"] = res["table"].join(steer_res["table"], on="Lap", how="left")
        except KeyError:
            steer_res = None
    ok  = np.isfinite(res["lt_val"]) & np.isfinite(res["mean_thr"])

    if not ok.any():
        return {
            "valid_laps": 0,
            "warnings": ["No valid laps for driver analysis."],
        }

    fast_idx = int(np.nanargmin(np.where(ok, res["lt_val"], np.inf)))
    full_t_ok      = res["full_t"][ok]
    mean_thr_ok    = res["mean_thr"][ok]
    full_pct_ok    = res["full_pct"][ok]
    off_pct_ok     = res["off_pct"][ok]
    speed_ok_mask  = ok & np.isfinite(res["mean_speed"])
    speed_vals     = res["mean_speed"][speed_ok_mask]
    brake_aggr_ok_mask = ok & np.isfinite(res["mean_brake_aggr"])
    brake_aggr_vals    = res["mean_brake_aggr"][brake_aggr_ok_mask]
    brake_release_ok_mask = ok & np.isfinite(res["mean_brake_release"])
    brake_release_vals    = res["mean_brake_release"][brake_release_ok_mask]
    steering_vals = (
        steer_res["mean_steering_smoothness"][np.isfinite(steer_res["mean_steering_smoothness"])]
        if steer_res is not None
        else np.asarray([], dtype=float)
    )
    steering_integral_vals = (
        steer_res["steering_integral_degm"][np.isfinite(steer_res["steering_integral_degm"])]
        if steer_res is not None
        else np.asarray([], dtype=float)
    )
    curvature_vals = (
        steer_res["mean_curvature"][np.isfinite(steer_res["mean_curvature"])]
        if steer_res is not None
        else np.asarray([], dtype=float)
    )

    return {
        "valid_laps":         int(ok.sum()),
        "fastest_lap":        int(res["lap_list"][fast_idx]),
        "fastest_lt":         float(res["lt_val"][fast_idx]),
        "mean_throttle_pct":  float(np.nanmean(mean_thr_ok)),
        "mean_full_t":        float(np.nanmean(full_t_ok)),
        "mean_full_pct":      float(np.nanmean(full_pct_ok)),
        "mean_off_pct":       float(np.nanmean(off_pct_ok)),
        "mean_speed":         float(np.nanmean(speed_vals)) if speed_vals.size else np.nan,
        "max_speed":          float(np.nanmax(speed_vals))  if speed_vals.size else np.nan,
        "mean_brake_aggr": (
            float(np.nanmean(brake_aggr_vals)) if brake_aggr_vals.size else np.nan
        ),
        "peak_brake_aggr": (
            float(np.nanmax(brake_aggr_vals)) if brake_aggr_vals.size else np.nan
        ),
        "mean_brake_release": (
            float(np.nanmean(brake_release_vals)) if brake_release_vals.size else np.nan
        ),
        "peak_brake_release": (
            float(np.nanmax(brake_release_vals)) if brake_release_vals.size else np.nan
        ),
        "mean_steering_smoothness": (
            float(np.nanmean(steering_vals)) if steering_vals.size else np.nan
        ),
        "max_steering_smoothness": (
            float(np.nanmax(steering_vals)) if steering_vals.size else np.nan
        ),
        "mean_steering_integral": (
            float(np.nanmean(steering_integral_vals)) if steering_integral_vals.size else np.nan
        ),
        "max_steering_integral": (
            float(np.nanmax(steering_integral_vals)) if steering_integral_vals.size else np.nan
        ),
        "mean_curvature": (
            float(np.nanmean(curvature_vals)) if curvature_vals.size else np.nan
        ),
        "max_curvature": (
            float(np.nanmax(curvature_vals)) if curvature_vals.size else np.nan
        ),
        "table":              res["table"],
        "warnings":           [],
    }


def throttle_summary(df: pl.DataFrame) -> dict:
    """Backward-compatible wrapper for existing dashboard/tests."""
    return driver_summary(df)


# ── Figures ───────────────────────────────────────────────────────────────────

def throttle_histogram_fig(dfs: dict[str, pl.DataFrame]) -> go.Figure:
    """Throttle position histogram (overlay when multiple runs are loaded)."""
    fig = make_dark_figure(
        title="Throttle Position Histogram",
        xlabel="Throttle position [%]",
        ylabel="Percent of samples [%]",
    )
    bins    = np.arange(0.0, 100.0 + THROTTLE_BIN_WIDTH, THROTTLE_BIN_WIDTH)
    centers = (bins[:-1] + bins[1:]) * 0.5

    for i, (run_name, df) in enumerate(dfs.items()):
        d    = _prep(df)
        thr  = d["Throttle"][np.isfinite(d["Throttle"])]
        if thr.size == 0:
            continue
        counts, _ = np.histogram(thr, bins=bins)
        pct       = counts / counts.sum() * 100.0
        color     = _driver_color(run_name, i)
        fig.add_trace(go.Bar(
            x=centers, y=pct, name=run_name,
            marker=dict(color=color, line=dict(width=0)),
            opacity=0.65 if len(dfs) > 1 else 1.0,
            width=THROTTLE_BIN_WIDTH * 0.9,
        ))

    fig.update_layout(
        barmode="overlay" if len(dfs) > 1 else "group",
        bargap=0.05,
    )
    fig.update_xaxes(range=[0, 100], dtick=10)
    return fig


def full_throttle_time_fig(
    dfs: dict[str, pl.DataFrame],
    x_mode: str = "laps",
) -> go.Figure:
    """Full throttle time per lap (TP > 95 %), overlay across runs."""
    fig = make_dark_figure(
        title="Full Throttle Time (TP > 95 %)",
        xlabel="Lap" if x_mode == "laps" else "Lap time [s]",
        ylabel="Full throttle time [s]",
    )
    all_laps: list[np.ndarray] = []
    for i, (run_name, df) in enumerate(dfs.items()):
        res = _per_lap(_prep(df))
        ok  = np.isfinite(res["full_t"]) & np.isfinite(res["lt_val"])
        if not ok.any():
            continue
        x_arr, order, _ = per_lap_axis(
            res["lap_list"][ok], res["lt_val"][ok], x_mode,
        )
        y_arr   = res["full_t"][ok][order]
        lap_ord = res["lap_list"][ok][order]
        color   = _driver_color(run_name, i)
        add_lap_scatter(fig, x_arr, y_arr, lap_ord, name=run_name, color=color)
        add_trend_line(fig, x_arr, y_arr, color=color)
        all_laps.append(res["lap_list"][ok])

    if x_mode == "laps" and all_laps:
        ticks = np.unique(np.concatenate(all_laps)).astype(int)
        fig.update_xaxes(tickvals=ticks)
    return fig


def throttle_speed_fig(
    dfs: dict[str, pl.DataFrame],
    x_mode: str = "laps",
) -> go.Figure:
    """Median |dTP/dt| per lap (TP < 100 %, brake released), overlay across runs."""
    fig = make_dark_figure(
        title="Throttle Speed — median |dTP/dt| (TP < 100 %, brake off)",
        xlabel="Lap" if x_mode == "laps" else "Lap time [s]",
        ylabel="|dTP/dt| [%/s]",
    )
    all_laps: list[np.ndarray] = []
    for i, (run_name, df) in enumerate(dfs.items()):
        res = _per_lap(_prep(df))
        ok  = np.isfinite(res["mean_speed"]) & np.isfinite(res["lt_val"])
        if not ok.any():
            continue
        x_arr, order, _ = per_lap_axis(
            res["lap_list"][ok], res["lt_val"][ok], x_mode,
        )
        y_arr   = res["mean_speed"][ok][order]
        lap_ord = res["lap_list"][ok][order]
        color   = _driver_color(run_name, i)
        add_lap_scatter(fig, x_arr, y_arr, lap_ord, name=run_name, color=color)
        add_trend_line(fig, x_arr, y_arr, color=color)
        all_laps.append(res["lap_list"][ok])

    if x_mode == "laps" and all_laps:
        ticks = np.unique(np.concatenate(all_laps)).astype(int)
        fig.update_xaxes(tickvals=ticks)
    return fig


def braking_effort_fig(dfs: dict[str, pl.DataFrame]) -> go.Figure:
    """Per-event brake demand versus achieved longitudinal acceleration."""
    accel_label = "Filtering_VN_ax [m/s²]"
    fig = make_dark_figure(
        title="Braking Effort — one point per braking event",
        xlabel="Brake [%]",
        ylabel=accel_label,
    )
    fig.update_layout(legend=dict(itemsizing="constant"))

    any_trace = False
    brake_max = 0.0
    accel_min = 0.0

    for i, (run_name, df) in enumerate(dfs.items()):
        accel_col = _longitudinal_accel_col(df)
        if accel_col != "Filtering_VN_ax":
            accel_label = f"{accel_col} [m/s²]"
            fig.update_yaxes(title_text=accel_label)
        d = _prep(df, extra_cols=(accel_col,))
        lap_ids, x_arr, y_arr, duration_s = _braking_effort_events(d, accel_col)
        valid = (
            np.isfinite(x_arr)
            & np.isfinite(y_arr)
            & np.isfinite(duration_s)
        )
        if not valid.any():
            continue

        any_trace = True
        lap_ids = lap_ids[valid]
        x_arr = x_arr[valid]
        y_arr = y_arr[valid]
        duration_s = duration_s[valid]
        color = _driver_color(run_name, i)
        brake_max = max(brake_max, float(np.nanmax(x_arr)))
        accel_min = min(accel_min, float(np.nanmin(y_arr)))
        customdata = np.column_stack([lap_ids, duration_s])

        fig.add_trace(go.Scattergl(
            x=x_arr,
            y=y_arr,
            mode="markers",
            name=run_name,
            marker=dict(
                color=color,
                size=5 if len(dfs) == 1 else 4,
                opacity=0.42 if len(dfs) == 1 else 0.28,
                line=dict(width=0),
            ),
            customdata=customdata,
            hovertemplate=(
                "Lap: %{customdata[0]:.0f}<br>"
                "Event duration: %{customdata[1]:.2f} s<br>"
                "Brake: %{x:.1f}%<br>"
                f"{accel_col}: " + "%{y:.2f} m/s²"
                f"<extra>{run_name}</extra>"
            ),
        ))

        brake_ref = float(np.nanmax(x_arr))
        accel_ref = float(np.nanmin(y_arr))

        fig.add_trace(go.Scatter(
            x=[0.0, brake_ref],
            y=[accel_ref, accel_ref],
            mode="lines",
            line=dict(color=color, dash="dot", width=1.2),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[brake_ref, brake_ref],
            y=[0.0, accel_ref],
            mode="lines",
            line=dict(color=color, dash="dot", width=1.2),
            showlegend=False,
            hoverinfo="skip",
        ))

    if not any_trace:
        raise ValueError("No valid braking samples passed the Brake/ax filter.")

    fig.update_xaxes(range=[0.0, max(80.0, min(100.0, brake_max + 5.0))], dtick=10)
    fig.update_yaxes(range=[accel_min - 0.5, 0.5], zeroline=False)
    return fig


def brake_application_point_fig(
    dfs: dict[str, pl.DataFrame],
    turns: list[object] | None = None,
) -> go.Figure:
    """Box plot of brake application point for repeated significant braking zones."""
    fig = make_dark_figure(
        title="Brake Application Point — significant braking zones",
        xlabel="Turn / braking zone",
        ylabel="Brake application point [m]",
    )

    events: list[dict] = []
    run_order = list(dfs.keys())
    for run_name, df in dfs.items():
        accel_col = _longitudinal_accel_col(df)
        speed_col = _speed_col(df)
        for row in _significant_brake_application_events(df, accel_col, speed_col):
            row["run"] = run_name
            row["accel_col"] = accel_col
            events.append(row)

    zoned_events = _label_brake_application_zones(
        _assign_brake_application_zones(events),
        turns,
    )
    if not zoned_events:
        raise ValueError(
            "No repeated significant braking zones found. "
            f"Filters: peak Brake >= {BRAKE_APPLICATION_MIN_PEAK_PCT:.0f} %, "
            f"min ax <= {BRAKE_APPLICATION_MIN_DECEL_MPS2:.2f} m/s², "
            f"speed drop >= {BRAKE_APPLICATION_MIN_DV_MPS:.1f} m/s."
        )

    if turns:
        zone_sort = {
            str(row["zone"]): int(row.get("zone_order", 0))
            for row in zoned_events
        }
        zone_order = sorted(
            {str(row["zone"]) for row in zoned_events},
            key=lambda label: zone_sort.get(label, 10_000),
        )
    else:
        zone_order = sorted(
            {str(row["zone"]) for row in zoned_events},
            key=lambda label: int(label.split()[-1]),
        )
    any_trace = False
    for run_idx, run_name in enumerate(run_order):
        run_rows = [row for row in zoned_events if row["run"] == run_name]
        if not run_rows:
            continue

        x_vals = [str(row["zone"]) for row in run_rows]
        y_vals = [float(row["start_dist_m"]) for row in run_rows]
        customdata = np.asarray(
            [
                [
                    float(row["lap"]),
                    float(row["duration_s"]),
                    float(row["peak_brake_pct"]),
                    float(row["min_accel_mps2"]),
                    float(row["speed_drop_mps"]),
                ]
                for row in run_rows
            ],
            dtype=float,
        )
        color = _driver_color(run_name, run_idx)
        any_trace = True
        fig.add_trace(go.Box(
            x=x_vals,
            y=y_vals,
            name=run_name,
            marker=dict(color=color, size=5, opacity=0.65),
            line=dict(color=color, width=1.6),
            boxmean=True,
            boxpoints="all",
            jitter=0.35,
            pointpos=0.0,
            customdata=customdata,
            hovertemplate=(
                "Turn/zone: %{x}<br>"
                "Lap: %{customdata[0]:.0f}<br>"
                "Application: %{y:.1f} m<br>"
                "Duration: %{customdata[1]:.2f} s<br>"
                "Peak Brake: %{customdata[2]:.1f} %<br>"
                "Min ax: %{customdata[3]:.2f} m/s²<br>"
                "Speed drop: %{customdata[4]:.1f} m/s"
                f"<extra>{run_name}</extra>"
            ),
        ))

    if not any_trace:
        raise ValueError("No valid brake-application points after zone grouping.")

    fig.update_layout(
        boxmode="group",
        legend=dict(itemsizing="constant"),
    )
    fig.update_xaxes(categoryorder="array", categoryarray=zone_order)
    return fig


def braking_aggressiveness_fig(
    dfs: dict[str, pl.DataFrame],
    x_mode: str = "laps",
) -> go.Figure:
    """Average braking aggressiveness per lap."""
    fig = make_dark_figure(
        title="Braking Aggressiveness — mean dBrake/dt (dBrake/dt > 5 %/s)",
        xlabel="Lap" if x_mode == "laps" else "Lap time [s]",
        ylabel="Avg. braking aggressiveness [%/s]",
    )
    all_laps: list[np.ndarray] = []
    for i, (run_name, df) in enumerate(dfs.items()):
        res = _per_lap(_prep(df))
        ok = np.isfinite(res["mean_brake_aggr"]) & np.isfinite(res["lt_val"])
        if not ok.any():
            continue
        x_arr, order, _ = per_lap_axis(
            res["lap_list"][ok], res["lt_val"][ok], x_mode,
        )
        y_arr = res["mean_brake_aggr"][ok][order]
        lap_ord = res["lap_list"][ok][order]
        color = _driver_color(run_name, i)
        add_lap_scatter(fig, x_arr, y_arr, lap_ord, name=run_name, color=color)
        add_trend_line(fig, x_arr, y_arr, color=color)
        all_laps.append(res["lap_list"][ok])

    if x_mode == "laps" and all_laps:
        ticks = np.unique(np.concatenate(all_laps)).astype(int)
        fig.update_xaxes(tickvals=ticks)
    return fig


def brake_release_smoothness_fig(
    dfs: dict[str, pl.DataFrame],
    x_mode: str = "laps",
) -> go.Figure:
    """Average brake-release smoothness per lap."""
    fig = make_dark_figure(
        title="Brake Release Smoothness — mean |dBrake/dt| (dBrake/dt < -5 %/s)",
        xlabel="Lap" if x_mode == "laps" else "Lap time [s]",
        ylabel="Avg. brake release smoothness [%/s]",
    )
    all_laps: list[np.ndarray] = []
    for i, (run_name, df) in enumerate(dfs.items()):
        res = _per_lap(_prep(df))
        ok = np.isfinite(res["mean_brake_release"]) & np.isfinite(res["lt_val"])
        if not ok.any():
            continue
        x_arr, order, _ = per_lap_axis(
            res["lap_list"][ok], res["lt_val"][ok], x_mode,
        )
        y_arr = res["mean_brake_release"][ok][order]
        lap_ord = res["lap_list"][ok][order]
        color = _driver_color(run_name, i)
        add_lap_scatter(fig, x_arr, y_arr, lap_ord, name=run_name, color=color)
        add_trend_line(fig, x_arr, y_arr, color=color)
        all_laps.append(res["lap_list"][ok])

    if x_mode == "laps" and all_laps:
        ticks = np.unique(np.concatenate(all_laps)).astype(int)
        fig.update_xaxes(tickvals=ticks)
    return fig


def steering_smoothness_fig(
    dfs: dict[str, pl.DataFrame],
    x_mode: str = "laps",
) -> go.Figure:
    """Average steering correction amplitude per lap."""
    fig = make_dark_figure(
        title="Steering Smoothness per Lap — mean |Steering - smooth_1s(Steering)|",
        xlabel="Lap" if x_mode == "laps" else "Lap time [s]",
        ylabel="Average steering smoothness [deg]",
    )
    all_laps: list[np.ndarray] = []

    for i, (run_name, df) in enumerate(dfs.items()):
        d, lat_accel_col, speed_col = _prep_steering_metrics(df)
        res = _steering_metrics_per_lap(d, lat_accel_col, speed_col)
        ok = np.isfinite(res["mean_steering_smoothness"]) & np.isfinite(res["lt_val"])
        if not ok.any():
            continue
        x_arr, order, _ = per_lap_axis(
            res["lap_list"][ok], res["lt_val"][ok], x_mode,
        )
        y_arr = res["mean_steering_smoothness"][ok][order]
        lap_ord = res["lap_list"][ok][order]
        color = _driver_color(run_name, i)
        add_lap_scatter(fig, x_arr, y_arr, lap_ord, name=run_name, color=color)
        add_trend_line(fig, x_arr, y_arr, color=color)
        all_laps.append(res["lap_list"][ok])

    if x_mode == "laps" and all_laps:
        ticks = np.unique(np.concatenate(all_laps)).astype(int)
        fig.update_xaxes(tickvals=ticks)
    return fig


def steering_integral_fig(
    dfs: dict[str, pl.DataFrame],
    x_mode: str = "laps",
) -> go.Figure:
    """Distance integral of absolute steering per lap."""
    fig = make_dark_figure(
        title="Steering Integral per Lap — integral of |Steering| over distance",
        xlabel="Lap" if x_mode == "laps" else "Lap time [s]",
        ylabel="Steering integral [deg*m]",
    )
    all_laps: list[np.ndarray] = []

    for i, (run_name, df) in enumerate(dfs.items()):
        d, lat_accel_col, speed_col = _prep_steering_metrics(df)
        res = _steering_metrics_per_lap(d, lat_accel_col, speed_col)
        ok = np.isfinite(res["steering_integral_degm"]) & np.isfinite(res["lt_val"])
        if not ok.any():
            continue
        x_arr, order, _ = per_lap_axis(
            res["lap_list"][ok], res["lt_val"][ok], x_mode,
        )
        y_arr = res["steering_integral_degm"][ok][order]
        lap_ord = res["lap_list"][ok][order]
        color = _driver_color(run_name, i)
        add_lap_scatter(fig, x_arr, y_arr, lap_ord, name=run_name, color=color)
        add_trend_line(fig, x_arr, y_arr, color=color)
        all_laps.append(res["lap_list"][ok])

    if x_mode == "laps" and all_laps:
        ticks = np.unique(np.concatenate(all_laps)).astype(int)
        fig.update_xaxes(tickvals=ticks)
    return fig


def steering_stability_fig(
    dfs: dict[str, pl.DataFrame],
    turns: list[object] | None = None,
) -> go.Figure:
    """Box plot of steering stability per significant braking zone.

    For each significant brake application (same detection as Brake Application
    Point), integrate |dSteering/dt| over the samples that are also in
    straight-line braking (|ay| < 0.2 g). Higher value means the driver was
    making more steering corrections during the braking event.
    """
    ay_threshold_g = STEERING_STAB_AY_LIMIT_MPS2 / 9.80665
    fig = make_dark_figure(
        title="Steering Stability — significant braking zones (straight-line)",
        xlabel="Turn / braking zone",
        ylabel="Steering stability [deg]",
    )

    events: list[dict] = []
    run_order = list(dfs.keys())
    for run_name, df in dfs.items():
        accel_col = _longitudinal_accel_col(df)
        speed_col = _speed_col(df)
        lat_accel_col = _lateral_accel_col(df)
        for row in _steering_stability_events(df, accel_col, speed_col, lat_accel_col):
            row["run"] = run_name
            events.append(row)

    zoned_events = _label_brake_application_zones(
        _assign_brake_application_zones(events),
        turns,
    )
    if not zoned_events:
        raise ValueError(
            "No repeated significant braking zones found for steering stability. "
            f"Filters: peak Brake >= {BRAKE_APPLICATION_MIN_PEAK_PCT:.0f} %, "
            f"min ax <= {BRAKE_APPLICATION_MIN_DECEL_MPS2:.2f} m/s², "
            f"speed drop >= {BRAKE_APPLICATION_MIN_DV_MPS:.1f} m/s, "
            f"|ay| < {ay_threshold_g:.2f} g (straight-line)."
        )

    if turns:
        zone_sort = {
            str(row["zone"]): int(row.get("zone_order", 0))
            for row in zoned_events
        }
        zone_order = sorted(
            {str(row["zone"]) for row in zoned_events},
            key=lambda label: zone_sort.get(label, 10_000),
        )
    else:
        zone_order = sorted(
            {str(row["zone"]) for row in zoned_events},
            key=lambda label: int(label.split()[-1]),
        )

    any_trace = False
    for run_idx, run_name in enumerate(run_order):
        run_rows = [row for row in zoned_events if row["run"] == run_name]
        if not run_rows:
            continue

        x_vals = [str(row["zone"]) for row in run_rows]
        y_vals = [float(row["steering_stability_deg"]) for row in run_rows]
        customdata = np.asarray(
            [
                [
                    float(row["lap"]),
                    float(row["duration_s"]),
                    float(row["peak_brake_pct"]),
                    float(row["min_accel_mps2"]),
                    float(row["speed_drop_mps"]),
                    float(row["straight_frac"]) * 100.0,
                ]
                for row in run_rows
            ],
            dtype=float,
        )
        color = _driver_color(run_name, run_idx)
        any_trace = True
        fig.add_trace(go.Box(
            x=x_vals,
            y=y_vals,
            name=run_name,
            marker=dict(color=color, size=5, opacity=0.65),
            line=dict(color=color, width=1.6),
            boxmean=True,
            boxpoints="all",
            jitter=0.35,
            pointpos=0.0,
            customdata=customdata,
            hovertemplate=(
                "Turn/zone: %{x}<br>"
                "Lap: %{customdata[0]:.0f}<br>"
                "Steering stability: %{y:.1f} deg<br>"
                "Event duration: %{customdata[1]:.2f} s<br>"
                "Straight-line samples: %{customdata[5]:.0f} %<br>"
                "Peak Brake: %{customdata[2]:.1f} %<br>"
                "Min ax: %{customdata[3]:.2f} m/s²<br>"
                "Speed drop: %{customdata[4]:.1f} m/s"
                f"<extra>{run_name}</extra>"
            ),
        ))

    if not any_trace:
        raise ValueError("No valid steering-stability points after zone grouping.")

    fig.update_layout(
        boxmode="group",
        legend=dict(itemsizing="constant"),
    )
    fig.update_xaxes(categoryorder="array", categoryarray=zone_order)
    return fig


# ── Lap Analysis: progression and consistency ────────────────────────────────

def _lap_times_per_run(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (lap_ids, lap_times_s) sorted by lap id, only for valid laps."""
    d = _prep(df)
    valid = np.isfinite(d["laps"]) & np.isfinite(d["laptime"])
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)

    lap_list = unique_laps(d["laps"]).astype(int)
    lt = np.full(len(lap_list), np.nan)
    for i, lap in enumerate(lap_list):
        lm = d["laps"] == lap
        if lm.any():
            lt[i] = float(np.nanmax(d["laptime"][lm]))
    ok = np.isfinite(lt)
    return lap_list[ok], lt[ok]


def lap_time_progression_fig(dfs: dict[str, pl.DataFrame]) -> go.Figure:
    """Lap time per lap (markers + line) and raising average per driver/run.

    Raising average: cumulative mean x_n = (x_1 + ... + x_n) / n — the same
    forward-looking trend shown in the lecture slide.
    """
    fig = make_dark_figure(
        title="Lap Time per Lap with Raising Average",
        xlabel="Lap",
        ylabel="Lap time [s]",
    )
    all_laps: list[np.ndarray] = []

    for i, (run_name, df) in enumerate(dfs.items()):
        lap_ids, lt = _lap_times_per_run(df)
        if lap_ids.size == 0:
            continue
        order = np.argsort(lap_ids)
        lap_ids = lap_ids[order]
        lt = lt[order]
        cum_avg = np.cumsum(lt) / np.arange(1, len(lt) + 1)
        color = _driver_color(run_name, i)

        fig.add_trace(go.Scatter(
            x=lap_ids, y=lt,
            mode="lines+markers",
            name=f"{run_name} — Lap time",
            legendgroup=run_name,
            line=dict(color=color, width=1.6),
            marker=dict(color=color, size=8, line=dict(width=0)),
            hovertemplate="Lap %{x}<br>Lap time: %{y:.3f} s<extra>" + run_name + "</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=lap_ids, y=cum_avg,
            mode="lines+markers",
            name=f"{run_name} — Raising avg",
            legendgroup=run_name,
            line=dict(color=color, width=1.4, dash="dash"),
            marker=dict(color=color, size=6, symbol="diamond"),
            opacity=0.85,
            hovertemplate="Lap %{x}<br>Raising avg: %{y:.3f} s<extra>" + run_name + "</extra>",
        ))
        all_laps.append(lap_ids)

    if all_laps:
        ticks = np.unique(np.concatenate(all_laps)).astype(int)
        fig.update_xaxes(tickvals=ticks)
    return fig


def lap_consistency_stats(dfs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Per-run lap-time variability/consistency statistics.

    Columns:
      - Laps             : number of valid laps used
      - Best [s]         : fastest lap time
      - Mean [s]         : arithmetic mean
      - Median [s]       : median (robust)
      - Std [s]          : standard deviation
      - CV [%]           : 100 * std / mean (key consistency index — lower is better)
      - MAD [s]          : median absolute deviation, robust to a single bad lap
      - Range [s]        : max - min
      - Gap to best [s]  : mean - best (avg time lost per lap vs. driver's best)
    """
    rows: list[dict] = []
    multi_run = len(dfs) > 1

    for run_name, df in dfs.items():
        try:
            _, lt = _lap_times_per_run(df)
        except Exception:
            continue
        if lt.size == 0:
            continue

        best   = float(np.nanmin(lt))
        mean   = float(np.nanmean(lt))
        median = float(np.nanmedian(lt))
        std    = float(np.nanstd(lt, ddof=1)) if lt.size >= 2 else 0.0
        cv_pct = 100.0 * std / mean if mean > 0 else np.nan
        mad    = float(np.nanmedian(np.abs(lt - median)))
        rng    = float(np.nanmax(lt) - np.nanmin(lt))
        gap    = mean - best

        row: dict = {}
        if multi_run:
            row["Run"] = run_name
        row["Laps"]            = int(lt.size)
        row["Best [s]"]        = round(best, 3)
        row["Mean [s]"]        = round(mean, 3)
        row["Median [s]"]      = round(median, 3)
        row["Std [s]"]         = round(std, 3)
        row["CV [%]"]          = round(cv_pct, 2)
        row["MAD [s]"]         = round(mad, 3)
        row["Range [s]"]       = round(rng, 3)
        row["Gap to best [s]"] = round(gap, 3)
        rows.append(row)

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def lap_time_distribution_fig(dfs: dict[str, pl.DataFrame]) -> go.Figure:
    """Box-plot of lap times per driver — visual companion to the consistency table."""
    fig = make_dark_figure(
        title="Lap Time Distribution per Driver",
        xlabel="",
        ylabel="Lap time [s]",
    )
    for i, (run_name, df) in enumerate(dfs.items()):
        try:
            _, lt = _lap_times_per_run(df)
        except Exception:
            continue
        if lt.size == 0:
            continue
        color = _driver_color(run_name, i)
        fig.add_trace(go.Box(
            y=lt,
            name=run_name,
            marker=dict(color=color),
            line=dict(color=color),
            fillcolor=color,
            opacity=0.55,
            boxmean="sd",
            boxpoints="all",
            jitter=0.4,
            pointpos=0.0,
        ))
    fig.update_layout(showlegend=False)
    return fig


def _lap_comparison_data(
    df: pl.DataFrame,
    lap_id: int,
) -> dict[str, np.ndarray | float]:
    """Return distance-aligned telemetry arrays for one lap."""
    speed_col = "VN_vx"
    ay_col = "Filtering_VN_ay"
    ax_col = "Filtering_VN_ax" if "Filtering_VN_ax" in df.columns else (
        "VN_ax" if "VN_ax" in df.columns else None
    )
    required = (
        "TimeStamp",
        "laps",
        "laptime",
        "Throttle",
        "Brake",
        "Steering",
        speed_col,
        ay_col,
    )
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for lap comparison: {missing}")

    s_all = lap_dist_from_gps(df)
    laps = cols_to_numpy(df, ["laps"])["laps"]
    mask = laps == float(lap_id)
    if int(mask.sum()) < 5:
        raise ValueError(f"Lap {lap_id} has too few samples for comparison.")

    fetch_cols = [
        "TimeStamp",
        "laptime",
        speed_col,
        "Throttle",
        "Brake",
        "Steering",
        ay_col,
        "VN_latitude",
        "VN_longitude",
    ]
    if ax_col is not None:
        fetch_cols.append(ax_col)
    cols = cols_to_numpy(df, fetch_cols)
    time_abs = cols["TimeStamp"][mask]
    time_s = time_abs - float(time_abs[0])
    s_m = s_all[mask]
    laptime = cols["laptime"][mask]

    data = {
        "s_m": s_m,
        "time_s": time_s,
        "vx_mps": cols[speed_col][mask],
        "throttle_pct": cols["Throttle"][mask],
        "brake_pct": cols["Brake"][mask],
        "steering_rad": cols["Steering"][mask],
        "ay_mps2": cols[ay_col][mask],
        "ax_mps2": cols[ax_col][mask] if ax_col is not None else np.full(int(mask.sum()), np.nan),
        "lap_time_s": float(np.nanmax(laptime[np.isfinite(laptime)]))
        if np.any(np.isfinite(laptime))
        else float(np.nanmax(time_s)),
    }
    radius_m = np.divide(
        np.square(data["vx_mps"]),
        np.maximum(np.abs(data["ay_mps2"]), 0.05),
        out=np.full_like(data["vx_mps"], np.nan, dtype=float),
        where=np.isfinite(data["vx_mps"]) & np.isfinite(data["ay_mps2"]),
    )
    radius_m = np.clip(radius_m, 0.0, 1.0e4)
    inv_radius = np.divide(
        1.0,
        radius_m,
        out=np.full_like(radius_m, np.nan, dtype=float),
        where=np.isfinite(radius_m) & (radius_m > 0.0),
    )
    dt_s = robust_dt(time_s) if len(time_s) >= 2 else 0.01
    radius_window = max(1, int(round(0.30 / max(dt_s, 1.0e-3))))
    curvature_1pm = smooth_signal(inv_radius, radius_window)
    data["radius_m"] = np.divide(
        1.0,
        curvature_1pm,
        out=np.full_like(curvature_1pm, 1.0e4, dtype=float),
        where=np.isfinite(curvature_1pm) & (curvature_1pm > 0.0),
    )
    data["radius_m"] = np.clip(data["radius_m"], 0.0, 1.0e4)
    data["curvature_1pm"] = curvature_1pm
    if "VN_latitude" in df.columns and "VN_longitude" in df.columns:
        data["latitude"] = cols["VN_latitude"][mask]
        data["longitude"] = cols["VN_longitude"][mask]

    valid = np.isfinite(data["s_m"]) & np.isfinite(data["time_s"])
    for key, value in list(data.items()):
        if isinstance(value, np.ndarray):
            data[key] = value[valid]

    order = np.argsort(data["s_m"], kind="mergesort")
    for key, value in list(data.items()):
        if isinstance(value, np.ndarray):
            data[key] = value[order]

    s_unique, unique_idx = np.unique(data["s_m"], return_index=True)
    for key, value in list(data.items()):
        if isinstance(value, np.ndarray):
            data[key] = value[unique_idx]
    data["s_m"] = s_unique
    return data


def _interp_on_grid(
    data: dict[str, np.ndarray | float],
    signal: str,
    grid: np.ndarray,
) -> np.ndarray:
    """Interpolate one lap signal on the shared distance grid."""
    if signal not in data or not isinstance(data[signal], np.ndarray):
        return np.full(len(grid), np.nan)
    s_m = data["s_m"]
    values = data[signal]
    valid = np.isfinite(s_m) & np.isfinite(values)
    if int(valid.sum()) < 2:
        return np.full(len(grid), np.nan)
    return np.interp(grid, s_m[valid], values[valid])


def _comparison_lap_label(run_name: str, lap_id: int) -> str:
    if run_name == POTENTIAL_LAP_RUN:
        return "Potential lap"
    return f"{run_name} L{int(lap_id)}"


def _lap_duration_between_distances(
    data: dict[str, np.ndarray | float],
    start_m: float,
    end_m: float,
) -> float:
    """Return elapsed lap time [s] between two distance coordinates."""
    s_m = np.asarray(data.get("s_m", []), dtype=float)
    time_s = np.asarray(data.get("time_s", []), dtype=float)
    if s_m.size < 2 or time_s.size != s_m.size:
        return np.nan
    if not (np.isfinite(start_m) and np.isfinite(end_m)) or end_m <= start_m:
        return np.nan

    valid = np.isfinite(s_m) & np.isfinite(time_s)
    if int(valid.sum()) < 2:
        return np.nan
    s_valid = s_m[valid]
    t_valid = time_s[valid]
    if start_m < float(s_valid[0]) or end_m > float(s_valid[-1]):
        return np.nan
    t0 = float(np.interp(float(start_m), s_valid, t_valid))
    t1 = float(np.interp(float(end_m), s_valid, t_valid))
    duration_s = t1 - t0
    return duration_s if np.isfinite(duration_s) and duration_s > 0.0 else np.nan


def best_lap_for_distance_window(
    dfs: dict[str, pl.DataFrame],
    start_m: float,
    end_m: float,
) -> dict[str, float | int | str] | None:
    """Find the real lap with the shortest elapsed time in a distance window.

    Lap 0 and each run's last lap are excluded unless the dashboard already
    applied an explicit lap selection. The returned lap is a real `(run, lap)`
    source, not a stitched synthetic lap.
    """
    best: dict[str, float | int | str] | None = None
    best_duration_s = np.inf

    for run_name, df in dfs.items():
        try:
            lap_ids = available_laps(df).astype(int)
        except Exception:
            continue
        if lap_ids.size == 0:
            continue
        explicit_lap_selection = COMPLETE_LAPS_MARKER in df.columns
        max_lap = int(np.max(lap_ids))
        for lap_id in lap_ids:
            lap_int = int(lap_id)
            if lap_int <= 0 or (not explicit_lap_selection and lap_int >= max_lap):
                continue
            try:
                data = _lap_comparison_data(df, lap_int)
            except Exception:
                continue
            duration_s = _lap_duration_between_distances(data, start_m, end_m)
            if np.isfinite(duration_s) and duration_s < best_duration_s:
                best_duration_s = float(duration_s)
                best = {
                    "run": run_name,
                    "lap": lap_int,
                    "duration_s": best_duration_s,
                    "start_m": float(start_m),
                    "end_m": float(end_m),
                }
    return best


def potential_lap_from_sectors(
    dfs: dict[str, pl.DataFrame],
    sectors: list[object],
    *,
    target_spacing_m: float = 1.0,
) -> tuple[pl.DataFrame, dict[str, object]] | None:
    """Build a synthetic potential lap from the fastest real sector sources.

    Each sector is copied from the real `(run, lap)` with the shortest elapsed
    time over that sector's distance window. The resulting DataFrame is a
    telemetry reference lap with stitched signals, a synthetic monotonic
    timestamp, `laps == 1`, and a cached `dist_m` column so downstream Lap
    Analysis code can consume it like any other lap.
    """
    valid_sectors = [
        sector for sector in sectors
        if (
            np.isfinite(float(sector.s_start_m))
            and np.isfinite(float(sector.s_end_m))
            and float(sector.s_end_m) > float(sector.s_start_m)
        )
    ]
    if not valid_sectors:
        return None

    segment_rows: list[dict[str, np.ndarray]] = []
    segment_meta: list[dict[str, float | int | str | None]] = []
    elapsed_offset_s = 0.0

    for sector_idx, sector in enumerate(valid_sectors):
        start_m = float(sector.s_start_m)
        end_m = float(sector.s_end_m)
        source = best_lap_for_distance_window(dfs, start_m, end_m)
        if source is None:
            return None

        run_name = str(source["run"])
        lap_id = int(source["lap"])
        data = _lap_comparison_data(dfs[run_name], lap_id)
        duration_s = float(source["duration_s"])
        n_points = max(3, int(np.ceil((end_m - start_m) / max(target_spacing_m, 0.2))) + 1)
        include_endpoint = sector_idx == len(valid_sectors) - 1
        s_grid = np.linspace(start_m, end_m, n_points, endpoint=include_endpoint)
        if s_grid.size < 2:
            continue

        src_time = _interp_on_grid(data, "time_s", s_grid)
        src_t0 = float(np.interp(start_m, data["s_m"], data["time_s"]))
        time_s = elapsed_offset_s + (src_time - src_t0)

        rows = {
            "TimeStamp": time_s,
            "laps": np.full(len(s_grid), 1.0),
            "laptime": np.full(len(s_grid), np.nan),
            "dist_m": s_grid,
            "VN_vx": _interp_on_grid(data, "vx_mps", s_grid),
            "Throttle": _interp_on_grid(data, "throttle_pct", s_grid),
            "Brake": _interp_on_grid(data, "brake_pct", s_grid),
            "Steering": _interp_on_grid(data, "steering_rad", s_grid),
            "Filtering_VN_ay": _interp_on_grid(data, "ay_mps2", s_grid),
            "Filtering_VN_ax": _interp_on_grid(data, "ax_mps2", s_grid),
            "VN_latitude": _interp_on_grid(data, "latitude", s_grid),
            "VN_longitude": _interp_on_grid(data, "longitude", s_grid),
        }
        segment_rows.append(rows)
        segment_meta.append({
            "sector_index": int(getattr(sector, "index", sector_idx)),
            "kind": str(getattr(sector, "kind", "")),
            "turn_id": (
                None
                if getattr(sector, "turn_id", None) is None
                else int(getattr(sector, "turn_id"))
            ),
            "s_start_m": start_m,
            "s_end_m": end_m,
            "run": run_name,
            "lap": lap_id,
            "duration_s": duration_s,
        })
        elapsed_offset_s += duration_s

    if not segment_rows or not np.isfinite(elapsed_offset_s) or elapsed_offset_s <= 0.0:
        return None

    columns = list(segment_rows[0].keys())
    stitched = {
        col: np.concatenate([rows[col] for rows in segment_rows])
        for col in columns
    }
    stitched["laptime"] = np.full(len(stitched["TimeStamp"]), elapsed_offset_s)
    out = pl.DataFrame(stitched)
    metadata: dict[str, object] = {
        "lap_time_s": float(elapsed_offset_s),
        "segments": segment_meta,
    }
    return out, metadata


def lap_delta_fig(
    dfs: dict[str, pl.DataFrame],
    ref_run: str,
    ref_lap: int,
    cmp_run: str,
    cmp_lap: int,
) -> go.Figure:
    """Delta time along distance: compared lap minus reference lap."""
    fig = make_dark_figure(
        "Delta Time vs Distance",
        "Distance from start line [m]",
        "Delta time [s]",
    )
    ref = _lap_comparison_data(dfs[ref_run], ref_lap)
    cmp = _lap_comparison_data(dfs[cmp_run], cmp_lap)
    max_s = min(float(np.nanmax(ref["s_m"])), float(np.nanmax(cmp["s_m"])))
    if not np.isfinite(max_s) or max_s <= 1.0:
        return fig

    grid = np.linspace(0.0, max_s, 900)
    ref_t = np.interp(grid, ref["s_m"], ref["time_s"])
    cmp_t = np.interp(grid, cmp["s_m"], cmp["time_s"])
    delta = cmp_t - ref_t
    ref_label = _comparison_lap_label(ref_run, int(ref_lap))
    cmp_label = _comparison_lap_label(cmp_run, int(cmp_lap))

    fig.add_trace(go.Scatter(
        x=grid,
        y=delta,
        mode="lines",
        name=f"{cmp_label} - {ref_label}",
        line=dict(color="#F2C94C", width=2.2),
        hovertemplate="Distance: %{x:.1f} m<br>Delta: %{y:+.3f} s<extra></extra>",
    ))
    fig.add_hline(y=0.0, line=dict(color="rgba(235,235,235,0.55)", dash="dash"))
    fig.update_layout(
        annotations=[
            dict(
                x=0.01,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"Positive = {cmp_label} is slower than reference",
                showarrow=False,
                font=dict(size=11, color="#EBEBEB"),
                bgcolor="rgba(20,20,23,0.75)",
            )
        ]
    )
    return fig


def _phase_segments(phases: np.ndarray) -> list[tuple[str, int, int]]:
    if len(phases) == 0:
        return []
    segments: list[tuple[str, int, int]] = []
    start = 0
    current = str(phases[0])
    for i in range(1, len(phases)):
        phase = str(phases[i])
        if phase != current:
            segments.append((current, start, i - 1))
            start = i
            current = phase
    segments.append((current, start, len(phases) - 1))
    return segments


def _classify_comparison_phase(
    ref_brake: np.ndarray,
    cmp_brake: np.ndarray,
    ref_throttle: np.ndarray,
    cmp_throttle: np.ndarray,
    ref_ay: np.ndarray,
    cmp_ay: np.ndarray,
) -> np.ndarray:
    max_brake = np.maximum(ref_brake, cmp_brake)
    max_throttle = np.maximum(ref_throttle, cmp_throttle)
    max_abs_ay = np.maximum(np.abs(ref_ay), np.abs(cmp_ay))
    phase = np.full(len(max_brake), "Straight", dtype=object)
    phase[max_throttle >= 60.0] = "Accel"
    phase[max_abs_ay >= 2.0] = "Corner"
    phase[max_brake >= 5.0] = "Brake"
    return phase


def lap_comparison_arrays(
    dfs: dict[str, pl.DataFrame],
    ref_run: str,
    ref_lap: int,
    cmp_run: str,
    cmp_lap: int,
    *,
    n_points: int = 1200,
) -> dict[str, np.ndarray | str]:
    """Return continuous A/B comparison arrays on a common distance grid."""
    ref = _lap_comparison_data(dfs[ref_run], ref_lap)
    cmp = _lap_comparison_data(dfs[cmp_run], cmp_lap)
    max_s = min(float(np.nanmax(ref["s_m"])), float(np.nanmax(cmp["s_m"])))
    if not np.isfinite(max_s) or max_s <= 1.0:
        raise ValueError("Cannot build comparison grid: invalid lap distance.")

    grid = np.linspace(0.0, max_s, int(n_points))
    ref_t = np.interp(grid, ref["s_m"], ref["time_s"])
    cmp_t = np.interp(grid, cmp["s_m"], cmp["time_s"])
    delta_s = cmp_t - ref_t

    ds = float(np.nanmedian(np.diff(grid))) if len(grid) > 1 else 1.0
    raw_loss_rate = np.gradient(delta_s, grid) * 10_000.0  # [ms / 10 m]
    smooth_window = max(5, int(round(20.0 / max(ds, 0.1))))
    loss_rate_ms_10m = smooth_signal(raw_loss_rate, smooth_window)

    ref_vx = _interp_on_grid(ref, "vx_mps", grid)
    cmp_vx = _interp_on_grid(cmp, "vx_mps", grid)
    ref_thr = _interp_on_grid(ref, "throttle_pct", grid)
    cmp_thr = _interp_on_grid(cmp, "throttle_pct", grid)
    ref_brake = _interp_on_grid(ref, "brake_pct", grid)
    cmp_brake = _interp_on_grid(cmp, "brake_pct", grid)
    ref_steer = _interp_on_grid(ref, "steering_rad", grid)
    cmp_steer = _interp_on_grid(cmp, "steering_rad", grid)
    ref_ay = _interp_on_grid(ref, "ay_mps2", grid)
    cmp_ay = _interp_on_grid(cmp, "ay_mps2", grid)
    ref_ax = _interp_on_grid(ref, "ax_mps2", grid)
    cmp_ax = _interp_on_grid(cmp, "ax_mps2", grid)
    ref_radius = _interp_on_grid(ref, "radius_m", grid)
    cmp_radius = _interp_on_grid(cmp, "radius_m", grid)
    ref_curvature = _interp_on_grid(ref, "curvature_1pm", grid)
    cmp_curvature = _interp_on_grid(cmp, "curvature_1pm", grid)
    ref_lat = _interp_on_grid(ref, "latitude", grid)
    ref_lng = _interp_on_grid(ref, "longitude", grid)
    cmp_lat = _interp_on_grid(cmp, "latitude", grid)
    cmp_lng = _interp_on_grid(cmp, "longitude", grid)

    return {
        "s_m": grid,
        "ref_time_s": ref_t,
        "cmp_time_s": cmp_t,
        "delta_s": delta_s,
        "loss_rate_ms_10m": loss_rate_ms_10m,
        "dvx_mps": cmp_vx - ref_vx,
        "dthrottle_pct": cmp_thr - ref_thr,
        "dbrake_pct": cmp_brake - ref_brake,
        "dsteering_rad": cmp_steer - ref_steer,
        "dsteering_deg": np.rad2deg(cmp_steer - ref_steer),
        "dax_mps2": cmp_ax - ref_ax,
        "day_mps2": cmp_ay - ref_ay,
        "dradius_m": cmp_radius - ref_radius,
        "dcurvature_1pm": cmp_curvature - ref_curvature,
        "ref_vx_mps": ref_vx,
        "cmp_vx_mps": cmp_vx,
        "ref_throttle_pct": ref_thr,
        "cmp_throttle_pct": cmp_thr,
        "ref_brake_pct": ref_brake,
        "cmp_brake_pct": cmp_brake,
        "ref_steering_rad": ref_steer,
        "cmp_steering_rad": cmp_steer,
        "ref_ay_mps2": ref_ay,
        "cmp_ay_mps2": cmp_ay,
        "ref_ax_mps2": ref_ax,
        "cmp_ax_mps2": cmp_ax,
        "ref_radius_m": ref_radius,
        "cmp_radius_m": cmp_radius,
        "ref_curvature_1pm": ref_curvature,
        "cmp_curvature_1pm": cmp_curvature,
        "ref_latitude": ref_lat,
        "ref_longitude": ref_lng,
        "cmp_latitude": cmp_lat,
        "cmp_longitude": cmp_lng,
        "phase": _classify_comparison_phase(
            ref_brake, cmp_brake, ref_thr, cmp_thr, ref_ay, cmp_ay
        ),
        "ref_label": _comparison_lap_label(ref_run, int(ref_lap)),
        "cmp_label": _comparison_lap_label(cmp_run, int(cmp_lap)),
        "ref_lap_time_s": float(ref["lap_time_s"]),
        "cmp_lap_time_s": float(cmp["lap_time_s"]),
    }


def _apply_phase_background(fig: go.Figure, comp: dict[str, np.ndarray | str], rows: int) -> None:
    colors = {
        "Brake": "rgba(235,87,87,0.12)",
        "Corner": "rgba(242,201,76,0.10)",
        "Accel": "rgba(39,174,96,0.10)",
        "Straight": "rgba(120,144,156,0.05)",
    }
    s_m = comp["s_m"]
    phases = comp["phase"]
    for phase, start, end in _phase_segments(phases):
        if end <= start:
            continue
        fill = colors.get(str(phase), "rgba(128,128,128,0.05)")
        for row in range(1, rows + 1):
            fig.add_vrect(
                x0=float(s_m[start]),
                x1=float(s_m[end]),
                fillcolor=fill,
                line_width=0,
                layer="below",
                row=row,
                col=1,
            )


def _turn_mask_on_grid(
    s_m: np.ndarray,
    turns: list[object] | None,
    active_turn_ids: set[int] | None = None,
) -> np.ndarray:
    mask = np.zeros(len(s_m), dtype=bool)
    if not turns:
        return mask
    for turn in turns:
        turn_id = int(turn.turn_id)
        if active_turn_ids is not None and turn_id not in active_turn_ids:
            continue
        mask |= (s_m >= float(turn.s_entry_m)) & (s_m <= float(turn.s_exit_m))
    return mask


def _excluded_turn_mask_on_grid(
    s_m: np.ndarray,
    turns: list[object] | None,
    active_turn_ids: set[int] | None,
) -> np.ndarray:
    """Samples inside detected turns that the user removed from Lap Analysis."""
    mask = np.zeros(len(s_m), dtype=bool)
    if not turns or active_turn_ids is None:
        return mask
    for turn in turns:
        turn_id = int(turn.turn_id)
        if turn_id in active_turn_ids:
            continue
        mask |= (s_m >= float(turn.s_entry_m)) & (s_m <= float(turn.s_exit_m))
    return mask


_AX_BRAKE_ON_MPS2 = 3.5     # ≈ 0.36 g — clear deceleration event
_AX_BRAKE_OFF_MPS2 = 1.5    # ≈ 0.15 g — hysteresis release
_AX_ACCEL_ON_MPS2 = 2.0     # ≈ 0.20 g — clear traction event
_AX_ACCEL_OFF_MPS2 = 0.8    # ≈ 0.08 g — hysteresis release
_APEX_PEAK_REL = 0.85       # apex band = curvature ≥ this fraction of peak
_APEX_HALF_BAND_M = 8.0     # fallback half-width of apex band when curvature is unreliable
_APEX_MIN_HALF_BAND_M = 3.0
_APEX_MAX_HALF_BAND_M = 18.0


def _hysteresis_mask(
    high: np.ndarray,
    low: np.ndarray,
) -> np.ndarray:
    """Activate when *high* is True; deactivate only when *low* becomes False.

    Implements two-threshold hysteresis: the input must rise above the strong
    threshold to turn the mask on and fall below the weak threshold to turn
    it off, killing fragmentation from micro on/off events.
    """
    n = len(high)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    active = False
    for i in range(n):
        if high[i]:
            active = True
        elif not low[i]:
            active = False
        out[i] = active
    return out


def _apex_band_for_turn(
    s_m: np.ndarray,
    curvature_ref: np.ndarray,
    entry_m: float,
    apex_m: float,
    exit_m: float,
) -> tuple[float, float]:
    """Return the (s_lo, s_hi) band that defines the Apex zone for a turn.

    Uses the band where the smoothed curvature is at least ``_APEX_PEAK_REL``
    of the local peak. Falls back to a fixed half-band around the apex when
    curvature data are missing or noisy.
    """
    in_corner = (s_m >= entry_m) & (s_m <= exit_m) & np.isfinite(curvature_ref)
    if not np.any(in_corner):
        half = _APEX_HALF_BAND_M
        return apex_m - half, apex_m + half
    seg_curv = curvature_ref[in_corner]
    seg_s = s_m[in_corner]
    peak = float(np.nanmax(np.abs(seg_curv)))
    if not np.isfinite(peak) or peak <= 0.0:
        half = _APEX_HALF_BAND_M
        return apex_m - half, apex_m + half
    apex_idx = int(np.argmin(np.abs(seg_s - apex_m)))
    threshold = peak * _APEX_PEAK_REL
    abs_curv = np.abs(seg_curv)
    # Walk left and right from the apex sample while curvature stays above threshold.
    lo_idx = apex_idx
    while lo_idx > 0 and abs_curv[lo_idx - 1] >= threshold:
        lo_idx -= 1
    hi_idx = apex_idx
    n = len(seg_s)
    while hi_idx < n - 1 and abs_curv[hi_idx + 1] >= threshold:
        hi_idx += 1
    s_lo = float(seg_s[lo_idx])
    s_hi = float(seg_s[hi_idx])
    half_lo = max(_APEX_MIN_HALF_BAND_M, min(apex_m - s_lo, _APEX_MAX_HALF_BAND_M))
    half_hi = max(_APEX_MIN_HALF_BAND_M, min(s_hi - apex_m, _APEX_MAX_HALF_BAND_M))
    return apex_m - half_lo, apex_m + half_hi


def _turn_id_at_grid(
    s_m: np.ndarray,
    turns: list[object] | None,
    active_turn_ids: set[int] | None,
) -> np.ndarray:
    """Per-sample integer label: turn_id inside a corner, -1 outside."""
    out = np.full(len(s_m), -1, dtype=np.int32)
    if not turns:
        return out
    for turn in turns:
        turn_id = int(turn.turn_id)
        if active_turn_ids is not None and turn_id not in active_turn_ids:
            continue
        mask = (s_m >= float(turn.s_entry_m)) & (s_m <= float(turn.s_exit_m))
        out[mask] = turn_id
    return out


def _clean_phase_segmented(
    s_m: np.ndarray,
    phase: np.ndarray,
    turn_id_at: np.ndarray,
    *,
    min_zone_m: float,
    merge_gap_m: float,
) -> np.ndarray:
    """Run distance-based cleanup independently per turn / between-turn run.

    Splitting by ``turn_id_at`` prevents the cleanup from fusing zones that
    belong to neighbouring turns (e.g. Corner Exit of T1 with Corner Entry
    of T2 in a chicane), while still letting micro fragments inside a single
    turn or between two turns merge with their neighbours.
    """
    if len(phase) == 0:
        return phase
    cleaned = phase.copy()
    boundaries = np.where(np.diff(turn_id_at) != 0)[0] + 1
    starts = [0, *boundaries.tolist()]
    ends = [*boundaries.tolist(), len(phase)]
    for start, end in zip(starts, ends):
        if end - start < 2:
            continue
        if turn_id_at[start] >= 0:
            # Inside a corner the sub-phases (Corner Entry / Apex / Corner
            # Exit / Braking / Acceleration) are already deterministic from
            # the geometry and pedal events; merging them here would erase
            # narrow apex bands.
            continue
        seg_s = s_m[start:end]
        seg_phase = cleaned[start:end]
        cleaned[start:end] = _clean_phase_by_distance(
            seg_s, seg_phase, min_zone_m=min_zone_m, merge_gap_m=merge_gap_m,
        )
    return cleaned


def _lap_analysis_phase_array(
    comp: dict[str, np.ndarray | str],
    *,
    turns: list[object] | None = None,
    active_turn_ids: set[int] | None = None,
    brake_threshold_pct: float = 5.0,
    throttle_threshold_pct: float = 40.0,
    ay_threshold_mps2: float | None = None,
) -> np.ndarray:
    """Classify comparison-grid samples for Lap Analysis geometry."""
    s_m = np.asarray(comp["s_m"], dtype=float)
    if turns:
        phase = np.full(len(s_m), "Straight", dtype=object)
        for ph in compute_lap_analysis_corner_phases(list(turns)):
            if active_turn_ids is not None and int(ph.turn_id) not in active_turn_ids:
                continue
            phase[(s_m >= ph.s_entry_m) & (s_m < ph.s_apex_m)] = "Entry"
            phase[(s_m > ph.s_apex_m) & (s_m <= ph.s_exit_m)] = "Exit"
            apex_idx = int(np.argmin(np.abs(s_m - ph.s_apex_m))) if len(s_m) else -1
            if apex_idx >= 0:
                phase[apex_idx] = "Apex"
        ignored_turn_mask = _excluded_turn_mask_on_grid(s_m, turns, active_turn_ids)
        phase[ignored_turn_mask] = "Ignored"
        return phase

    max_brake = np.maximum(comp["ref_brake_pct"], comp["cmp_brake_pct"])
    max_throttle = np.maximum(comp["ref_throttle_pct"], comp["cmp_throttle_pct"])
    max_abs_ay = np.maximum(np.abs(comp["ref_ay_mps2"]), np.abs(comp["cmp_ay_mps2"]))
    ref_ax = comp.get("ref_ax_mps2")
    cmp_ax = comp.get("cmp_ax_mps2")
    if isinstance(ref_ax, np.ndarray) and isinstance(cmp_ax, np.ndarray):
        # min picks the most negative (deepest braking); max picks strongest accel.
        min_ax = np.minimum(ref_ax, cmp_ax)
        max_ax = np.maximum(ref_ax, cmp_ax)
    else:
        min_ax = np.full(len(s_m), np.nan)
        max_ax = np.full(len(s_m), np.nan)

    brake_off_pct = max(0.0, brake_threshold_pct * 0.4)
    throttle_off_pct = max(0.0, throttle_threshold_pct * 0.6)

    brake_high = (
        (np.isfinite(max_brake) & (max_brake >= brake_threshold_pct))
        | (np.isfinite(min_ax) & (min_ax <= -_AX_BRAKE_ON_MPS2))
    )
    brake_low = (
        (np.isfinite(max_brake) & (max_brake >= brake_off_pct))
        | (np.isfinite(min_ax) & (min_ax <= -_AX_BRAKE_OFF_MPS2))
    )
    accel_high = (
        (np.isfinite(max_throttle) & (max_throttle >= throttle_threshold_pct))
        | (np.isfinite(max_ax) & (max_ax >= _AX_ACCEL_ON_MPS2))
    )
    accel_low = (
        (np.isfinite(max_throttle) & (max_throttle >= throttle_off_pct))
        | (np.isfinite(max_ax) & (max_ax >= _AX_ACCEL_OFF_MPS2))
    )

    brake_mask = _hysteresis_mask(brake_high, brake_low)
    accel_mask = _hysteresis_mask(accel_high, accel_low) & ~brake_mask

    phase = np.full(len(s_m), "Straight", dtype=object)
    phase[accel_mask] = "Acceleration"
    phase[brake_mask] = "Braking"

    if turns:
        ref_curvature = comp.get("ref_curvature_1pm")
        if not isinstance(ref_curvature, np.ndarray):
            ref_curvature = np.full(len(s_m), np.nan)
        for turn in turns:
            turn_id = int(turn.turn_id)
            if active_turn_ids is not None and turn_id not in active_turn_ids:
                continue
            entry_m = float(turn.s_entry_m)
            apex_m = float(turn.s_apex_m)
            exit_m = float(turn.s_exit_m)
            if not np.isfinite(entry_m + apex_m + exit_m) or exit_m <= entry_m:
                continue
            apex_m = float(np.clip(apex_m, entry_m, exit_m))
            apex_lo, apex_hi = _apex_band_for_turn(
                s_m, ref_curvature, entry_m, apex_m, exit_m,
            )
            apex_lo = float(np.clip(apex_lo, entry_m, exit_m))
            apex_hi = float(np.clip(apex_hi, entry_m, exit_m))
            if apex_hi <= apex_lo:
                apex_lo = max(entry_m, apex_m - _APEX_MIN_HALF_BAND_M)
                apex_hi = min(exit_m, apex_m + _APEX_MIN_HALF_BAND_M)
            in_turn = (s_m >= entry_m) & (s_m <= exit_m)
            in_apex = in_turn & (s_m >= apex_lo) & (s_m <= apex_hi)
            pre_apex = in_turn & (s_m < apex_lo)
            post_apex = in_turn & (s_m > apex_hi)
            # Apex always wins inside the apex band.
            phase[in_apex] = "Apex"
            # Pre-apex: rolling samples without an active braking event get
            # tagged as Corner Entry. Trail-braking samples keep their
            # Braking label from the pedal+ax detection above.
            phase[pre_apex & ~brake_mask & ~accel_mask] = "Corner Entry"
            # Post-apex: rolling samples without an active acceleration event
            # become Corner Exit; on-throttle samples keep Acceleration.
            phase[post_apex & ~accel_mask & ~brake_mask] = "Corner Exit"
    elif ay_threshold_mps2 is not None:
        corner_mask = np.isfinite(max_abs_ay) & (max_abs_ay >= ay_threshold_mps2)
        phase[corner_mask] = "Corner"
    ignored_turn_mask = _excluded_turn_mask_on_grid(s_m, turns, active_turn_ids)
    phase[ignored_turn_mask] = "Ignored"
    return phase


def _turn_for_distance(
    s_m: float,
    turns: list[object] | None,
    active_turn_ids: set[int] | None,
) -> object | None:
    if not turns:
        return None
    for turn in turns:
        turn_id = int(turn.turn_id)
        if active_turn_ids is not None and turn_id not in active_turn_ids:
            continue
        if float(turn.s_entry_m) <= s_m <= float(turn.s_exit_m):
            return turn
    return None


def _nearest_turn_id_for_zone(
    zone_type: str,
    center_m: float,
    turns: list[object] | None,
    active_turn_ids: set[int] | None,
) -> int | None:
    if not turns:
        return None
    candidates = [
        turn for turn in turns
        if active_turn_ids is None or int(turn.turn_id) in active_turn_ids
    ]
    if not candidates:
        return None
    if zone_type == "Braking":
        ahead = [turn for turn in candidates if float(turn.s_entry_m) >= center_m]
        if ahead:
            return int(min(ahead, key=lambda t: float(t.s_entry_m)).turn_id)
    if zone_type == "Acceleration":
        behind = [turn for turn in candidates if float(turn.s_exit_m) <= center_m]
        if behind:
            return int(max(behind, key=lambda t: float(t.s_exit_m)).turn_id)
    return int(min(candidates, key=lambda t: abs(float(t.s_apex_m) - center_m)).turn_id)


def _clean_phase_by_distance(
    s_m: np.ndarray,
    phase: np.ndarray,
    *,
    min_zone_m: float,
    merge_gap_m: float,
) -> np.ndarray:
    clean = phase.astype(object).copy()
    if len(clean) < 2:
        return clean

    for _ in range(2):
        for _phase, start, end in _phase_segments(clean):
            length_m = float(s_m[end] - s_m[start])
            if not np.isfinite(length_m) or length_m >= min_zone_m:
                continue
            prev_phase = clean[start - 1] if start > 0 else None
            next_phase = clean[end + 1] if end + 1 < len(clean) else None
            if prev_phase is not None and next_phase is not None:
                replacement = prev_phase if prev_phase == next_phase else next_phase
            else:
                replacement = prev_phase if prev_phase is not None else next_phase
            if replacement is not None:
                clean[start:end + 1] = replacement

    if merge_gap_m > 0.0:
        for _phase, start, end in _phase_segments(clean.copy()):
            if start == 0 or end + 1 >= len(clean):
                continue
            if clean[start - 1] != clean[end + 1]:
                continue
            gap_m = float(s_m[end] - s_m[start])
            if np.isfinite(gap_m) and gap_m <= merge_gap_m:
                clean[start:end + 1] = clean[start - 1]
    return clean


def _comparison_phase_zones_from_comp(
    comp: dict[str, np.ndarray | str],
    *,
    turns: list[object] | None = None,
    active_turn_ids: set[int] | None = None,
    brake_threshold_pct: float = 5.0,
    throttle_threshold_pct: float = 40.0,
    ay_threshold_mps2: float | None = None,
    min_zone_m: float = 10.0,
    merge_gap_m: float = 6.0,
) -> list[dict[str, float | str | int | None]]:
    s_m = np.asarray(comp["s_m"], dtype=float)
    phase = _lap_analysis_phase_array(
        comp,
        turns=turns,
        active_turn_ids=active_turn_ids,
        brake_threshold_pct=brake_threshold_pct,
        throttle_threshold_pct=throttle_threshold_pct,
        ay_threshold_mps2=ay_threshold_mps2,
    )
    if turns:
        turn_id_at = _turn_id_at_grid(s_m, turns, active_turn_ids)
        phase = _clean_phase_segmented(
            s_m, phase, turn_id_at,
            min_zone_m=min_zone_m,
            merge_gap_m=merge_gap_m,
        )
    else:
        phase = _clean_phase_by_distance(
            s_m, phase, min_zone_m=min_zone_m, merge_gap_m=merge_gap_m,
        )
    ignored_turn_mask = _excluded_turn_mask_on_grid(s_m, turns, active_turn_ids)

    counters = {
        "Braking": 1,
        "Corner": 1,
        "Entry": 1,
        "Corner Entry": 1,
        "Apex": 1,
        "Exit": 1,
        "Corner Exit": 1,
        "Acceleration": 1,
        "Straight": 1,
    }
    prefixes = {
        "Braking": "B",
        "Corner": "C",
        "Entry": "E",
        "Corner Entry": "E",
        "Apex": "AP",
        "Exit": "X",
        "Corner Exit": "X",
        "Acceleration": "A",
        "Straight": "S",
    }
    zones: list[dict[str, float | str | int | None]] = []
    for phase_name, start, end in _phase_segments(phase):
        phase_str = str(phase_name)
        if end <= start:
            continue
        if phase_str == "Ignored":
            continue
        s0 = float(s_m[start])
        s1 = float(s_m[end])
        if not np.isfinite(s0) or not np.isfinite(s1) or s1 <= s0:
            continue
        # Drop only very short Straight zones (< 2 m). Braking, Acceleration
        # and the corner sub-phases stay even when narrow so the Δt budget
        # adds up to the lap-time delta and no time is hidden in a discarded
        # micro-zone between two turns.
        zone_min_m = 2.0 if phase_str == "Straight" else 0.0
        if s1 - s0 < max(zone_min_m, 0.5):
            continue
        center_m = 0.5 * (s0 + s1)
        zone_mask = (s_m >= s0) & (s_m <= s1)
        if zone_mask.any() and np.any(ignored_turn_mask & zone_mask):
            continue
        turn = _turn_for_distance(center_m, turns, active_turn_ids)
        turn_id = int(turn.turn_id) if turn is not None else None
        if turn_id is None and phase_str in {"Braking", "Acceleration"}:
            turn_id = _nearest_turn_id_for_zone(
                phase_str, center_m, turns, active_turn_ids,
            )

        if phase_str in {"Entry", "Corner Entry"} and turn_id is not None:
            zone_name = f"T{turn_id} Entry"
        elif phase_str == "Apex" and turn_id is not None:
            zone_name = f"T{turn_id} Apex"
        elif phase_str in {"Exit", "Corner Exit"} and turn_id is not None:
            zone_name = f"T{turn_id} Exit"
        elif phase_str == "Braking" and turn_id is not None:
            zone_name = f"T{turn_id} Braking"
        elif phase_str == "Acceleration" and turn_id is not None:
            zone_name = f"T{turn_id} Accel"
        else:
            idx = counters.get(phase_str, 1)
            counters[phase_str] = idx + 1
            zone_name = f"{prefixes.get(phase_str, 'Z')}{idx}"
        zones.append({
            "zone": zone_name,
            "type": phase_str,
            "turn_id": turn_id,
            "s0_m": s0,
            "s1_m": s1,
        })
    return zones


def lap_comparison_summary(
    dfs: dict[str, pl.DataFrame],
    ref_run: str,
    ref_lap: int,
    cmp_run: str,
    cmp_lap: int,
) -> dict[str, float | str]:
    """Return headline A/B comparison numbers for the selected laps."""
    comp = lap_comparison_arrays(dfs, ref_run, ref_lap, cmp_run, cmp_lap)
    delta = comp["delta_s"]
    loss_rate = comp["loss_rate_ms_10m"]
    delta_steps = np.diff(delta)
    lost_s = float(np.nansum(np.where(delta_steps > 0.0, delta_steps, 0.0)))
    gained_s = float(-np.nansum(np.where(delta_steps < 0.0, delta_steps, 0.0)))
    total_delta_s = float(comp["cmp_lap_time_s"] - comp["ref_lap_time_s"])

    return {
        "ref_label": str(comp["ref_label"]),
        "cmp_label": str(comp["cmp_label"]),
        "ref_lap_time_s": float(comp["ref_lap_time_s"]),
        "cmp_lap_time_s": float(comp["cmp_lap_time_s"]),
        "total_delta_s": total_delta_s,
        "distance_delta_s": float(delta[-1] - delta[0]),
        "gross_lost_s": lost_s,
        "gross_gained_s": gained_s,
        "peak_loss_rate_ms_10m": float(np.nanmax(loss_rate)),
        "peak_gain_rate_ms_10m": float(np.nanmin(loss_rate)),
    }


def lap_comparison_track_fig(
    dfs: dict[str, pl.DataFrame],
    ref_run: str,
    ref_lap: int,
    cmp_run: str,
    cmp_lap: int,
    *,
    turns: list | None = None,
    active_turn_ids: set[int] | None = None,
) -> go.Figure:
    """GPS track coloured by local time gain/loss for the compared lap.

    If *turns* (list of TurnDef) is provided, corner apexes are labelled T1, T2, …
    """
    comp = lap_comparison_arrays(dfs, ref_run, ref_lap, cmp_run, cmp_lap)
    fig = make_dark_figure(
        "Where It Happens on Track",
        "Longitude",
        "Latitude",
    )

    ref_ok = np.isfinite(comp["ref_longitude"]) & np.isfinite(comp["ref_latitude"])
    cmp_ok = np.isfinite(comp["cmp_longitude"]) & np.isfinite(comp["cmp_latitude"])
    if not cmp_ok.any():
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="GPS latitude/longitude not available for this comparison",
            showarrow=False,
            font=dict(color="#EBEBEB"),
        )
        return fig

    if ref_ok.any():
        fig.add_trace(
            go.Scattergl(
                x=comp["ref_longitude"][ref_ok],
                y=comp["ref_latitude"][ref_ok],
                mode="lines",
                name=str(comp["ref_label"]),
                line=dict(color="rgba(235,235,235,0.45)", width=2.0),
                hoverinfo="skip",
            )
        )

    loss = comp["loss_rate_ms_10m"]
    finite_loss = loss[np.isfinite(loss)]
    cmax = float(np.nanpercentile(np.abs(finite_loss), 95)) if finite_loss.size else 1.0
    cmax = max(cmax, 10.0)
    fig.add_trace(
        go.Scattergl(
            x=comp["cmp_longitude"][cmp_ok],
            y=comp["cmp_latitude"][cmp_ok],
            mode="markers",
            name=str(comp["cmp_label"]),
            marker=dict(
                size=6,
                color=loss[cmp_ok],
                cmin=-cmax,
                cmax=cmax,
                colorscale=[
                    [0.0, "#27AE60"],
                    [0.5, "#F2C94C"],
                    [1.0, "#EB5757"],
                ],
                colorbar=dict(title="ms/10m"),
                line=dict(width=0),
            ),
            customdata=np.column_stack([comp["s_m"][cmp_ok], loss[cmp_ok]]),
            hovertemplate=(
                "Distance %{customdata[0]:.1f} m<br>"
                "Local Δt rate %{customdata[1]:+.1f} ms/10m"
                "<extra></extra>"
            ),
        )
    )
    if turns:
        for turn in turns:
            turn_id = int(turn.turn_id)
            is_active = active_turn_ids is None or turn_id in active_turn_ids
            fig.add_trace(
                go.Scattergl(
                    x=[float(turn.apex_lng)],
                    y=[float(turn.apex_lat)],
                    mode="markers+text" if is_active else "markers",
                    text=[f"T{turn_id}"] if is_active else None,
                    textposition="top center",
                    textfont=dict(
                        color="#F2C94C" if is_active else "rgba(235,235,235,0.45)",
                        size=10,
                    ),
                    marker=dict(
                        size=8 if is_active else 6,
                        color="#F2C94C" if is_active else "rgba(235,235,235,0.35)",
                        symbol="circle" if is_active else "circle-open",
                    ),
                    customdata=[["turn", turn_id]],
                    name=f"T{turn_id}",
                    showlegend=False,
                    hovertemplate=(
                        f"T{turn_id}<br>Apex {float(turn.s_apex_m):.0f} m"
                        + ("" if is_active else "<br>Excluded")
                        + "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(height=430, showlegend=True)
    fig.update_yaxes(scaleanchor="x", scaleratio=1.0)
    return fig


def lap_delta_cumulative_fig(
    dfs: dict[str, pl.DataFrame],
    ref_run: str,
    ref_lap: int,
    cmp_run: str,
    cmp_lap: int,
    *,
    turns: list | None = None,
    active_turn_ids: set[int] | None = None,
) -> go.Figure:
    """Cumulative Δt = t_cmp − t_ref vs lap distance, with corner bands.

    A rising line means the compared lap is losing time; falling means
    it is gaining. Shaded vertical bands mark detected corners so the
    user can see whether time is being lost in corners or on straights.
    """
    comp = lap_comparison_arrays(dfs, ref_run, ref_lap, cmp_run, cmp_lap)
    s = np.asarray(comp["s_m"], dtype=float)
    delta = np.asarray(comp["delta_s"], dtype=float)

    fig = make_dark_figure(
        "Cumulative Δt vs distance",
        "Distance [m]",
        "Δt = t_cmp − t_ref [s]",
    )

    pos = np.where(delta > 0.0, delta, 0.0)
    neg = np.where(delta < 0.0, delta, 0.0)
    fig.add_trace(
        go.Scatter(
            x=s, y=pos, mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(235,87,87,0.30)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=s, y=neg, mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(39,174,96,0.30)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=s, y=delta, mode="lines",
            line=dict(color="#F2C94C", width=2.0),
            name=f"{comp['cmp_label']} − {comp['ref_label']}",
            hovertemplate="Distance %{x:.1f} m<br>Δt %{y:+.3f} s<extra></extra>",
        )
    )

    fig.add_hline(
        y=0.0,
        line=dict(color="rgba(235,235,235,0.4)", width=1, dash="dash"),
    )

    if turns:
        for turn in turns:
            turn_id = int(turn.turn_id)
            is_active = active_turn_ids is None or turn_id in active_turn_ids
            if not is_active:
                continue
            fig.add_vrect(
                x0=float(turn.s_entry_m),
                x1=float(turn.s_exit_m),
                fillcolor="rgba(255,255,255,0.06)",
                line_width=0,
                annotation_text=f"T{turn_id}",
                annotation_position="top",
                annotation_font_color="#F2C94C",
                annotation_font_size=10,
            )

    fig.update_layout(height=320, showlegend=False)
    return fig


def _first_sustained_crossing_idx(
    signal: np.ndarray,
    threshold: float,
    *,
    sustain_n: int,
    start_idx: int,
    end_idx: int,
) -> int | None:
    """First index in [start_idx, end_idx) where signal stays > threshold for sustain_n samples."""
    if start_idx >= end_idx:
        return None
    sus = max(1, int(sustain_n))
    for i in range(start_idx, max(start_idx + 1, end_idx - sus + 1)):
        window = signal[i : i + sus]
        if window.size < sus:
            return None
        if np.all(np.isfinite(window)) and np.all(window > threshold):
            return i
    return None


def _pedal_distances_for_lap(
    data: dict[str, np.ndarray | float],
    turn: object,
    *,
    brake_thr_pct: float,
    throttle_thr_pct: float,
    sustain_brake_s: float = 0.05,
    sustain_throttle_s: float = 0.10,
    pre_pad_m: float = 50.0,
    post_pad_m: float = 50.0,
) -> tuple[float, float]:
    """Return (brake_to_apex_m, apex_to_throttle_m) for one lap and one corner.

    NaN when the event isn't detected within the search window.
    """
    s = np.asarray(data["s_m"], dtype=float)
    t = np.asarray(data["time_s"], dtype=float)
    brake = np.asarray(data["brake_pct"], dtype=float)
    thr = np.asarray(data["throttle_pct"], dtype=float)
    if s.size < 3:
        return float("nan"), float("nan")

    s_apex = float(turn.s_apex_m)
    s_entry = float(turn.s_entry_m)
    s_exit = float(turn.s_exit_m)

    dt = robust_dt(t) if t.size >= 2 else 0.01
    sustain_brake_n = max(1, int(round(sustain_brake_s / max(dt, 1.0e-3))))
    sustain_throttle_n = max(1, int(round(sustain_throttle_s / max(dt, 1.0e-3))))

    brake_lo = float(s_entry - pre_pad_m)
    brake_hi = float(s_apex)
    i0 = int(np.searchsorted(s, brake_lo, side="left"))
    i1 = int(np.searchsorted(s, brake_hi, side="right"))
    idx_b = _first_sustained_crossing_idx(
        brake, brake_thr_pct, sustain_n=sustain_brake_n, start_idx=i0, end_idx=i1
    )
    brake_to_apex_m = float(s_apex - s[idx_b]) if idx_b is not None else float("nan")

    thr_lo = float(s_apex)
    thr_hi = float(s_exit + post_pad_m)
    j0 = int(np.searchsorted(s, thr_lo, side="left"))
    j1 = int(np.searchsorted(s, thr_hi, side="right"))
    idx_t = _first_sustained_crossing_idx(
        thr, throttle_thr_pct, sustain_n=sustain_throttle_n, start_idx=j0, end_idx=j1
    )
    apex_to_throttle_m = float(s[idx_t] - s_apex) if idx_t is not None else float("nan")

    return brake_to_apex_m, apex_to_throttle_m


def lap_pedal_distances_fig(
    dfs: dict[str, pl.DataFrame],
    ref_run: str,
    ref_lap: int,
    cmp_run: str,
    cmp_lap: int,
    *,
    turns: list | None = None,
    active_turn_ids: set[int] | None = None,
    brake_thr_pct: float = 5.0,
    throttle_thr_pct: float = 40.0,
) -> go.Figure:
    """Per-(lap × corner) Apex → Throttle distance, one scatter plot.

    Background dots: every lap of every run, colored by run. Each run also
    highlights its fastest valid lap in purple.
    """
    _ = ref_run, ref_lap, cmp_run, cmp_lap
    fig = make_dark_figure(
        "Apex → Throttle distance per corner",
        "Corner",
        "Distance [m]",
    )

    active_turns = []
    if turns:
        for turn in turns:
            turn_id = int(turn.turn_id)
            if active_turn_ids is None or turn_id in active_turn_ids:
                active_turns.append(turn)
    active_turns.sort(key=lambda t: float(t.s_entry_m))

    if not active_turns:
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="No corners selected.",
            showarrow=False, font=dict(color="#EBEBEB"),
        )
        fig.update_layout(height=300, showlegend=False)
        return fig

    n_corners = len(active_turns)
    x_pos = np.arange(n_corners, dtype=float)
    x_ticktext = [f"T{int(t.turn_id)}" for t in active_turns]
    rng = np.random.default_rng(42)
    highlight_offsets = np.linspace(-0.10, 0.10, max(len(dfs), 1))
    fastest_color = "#9B51E0"
    all_y: list[float] = []

    for run_idx, (run_name, df) in enumerate(dfs.items()):
        try:
            laps = available_laps(df)
        except Exception:
            continue
        if laps.size == 0:
            continue

        bg_color = _driver_color(run_name, run_idx)
        highlight_x: list[float] = []
        highlight_y: list[float] = []
        highlight_h: list[str] = []
        fastest_lap_id: int | None = None

        try:
            lap_cols = cols_to_numpy(df, ["laps", "laptime"])
            lap_times: list[tuple[int, float]] = []
            for lap_id in laps.tolist():
                lap_mask = lap_cols["laps"] == float(lap_id)
                if lap_mask.any() and np.any(np.isfinite(lap_cols["laptime"][lap_mask])):
                    lap_times.append(
                        (int(lap_id), float(np.nanmax(lap_cols["laptime"][lap_mask])))
                    )
            if lap_times:
                fastest_lap_id = min(lap_times, key=lambda item: item[1])[0]
        except Exception:
            fastest_lap_id = None

        bg_x: list[float] = []
        bg_y: list[float] = []
        bg_h: list[str] = []

        for lap_id in laps:
            try:
                data = _lap_comparison_data(df, int(lap_id))
            except Exception:
                continue
            is_fastest = fastest_lap_id is not None and int(lap_id) == fastest_lap_id
            for ci, turn in enumerate(active_turns):
                _, rt_m = _pedal_distances_for_lap(
                    data, turn,
                    brake_thr_pct=brake_thr_pct,
                    throttle_thr_pct=throttle_thr_pct,
                )
                if not np.isfinite(rt_m):
                    continue
                all_y.append(float(rt_m))
                jitter = float(rng.uniform(-0.18, 0.18))
                label = f"{run_name} L{int(lap_id)} • T{int(turn.turn_id)}"
                if is_fastest:
                    highlight_x.append(float(x_pos[ci] + highlight_offsets[run_idx]))
                    highlight_y.append(float(rt_m))
                    highlight_h.append(label + " (fastest)")
                else:
                    bg_x.append(float(x_pos[ci] + jitter))
                    bg_y.append(float(rt_m))
                    bg_h.append(label)

        if bg_x:
            fig.add_trace(
                go.Scatter(
                    x=bg_x, y=bg_y, mode="markers",
                    name=run_name, legendgroup=run_name,
                    marker=dict(
                        color=bg_color, size=6, opacity=0.55,
                        line=dict(width=0.5, color="#1A1F2A"),
                    ),
                    hovertext=bg_h,
                    hovertemplate="%{hovertext}<br>%{y:.1f} m<extra></extra>",
                )
            )
        if highlight_x:
            fastest_label = (
                f"{run_name} fastest L{int(fastest_lap_id)}"
                if fastest_lap_id is not None else f"{run_name} fastest"
            )
            fig.add_trace(
                go.Scatter(
                    x=highlight_x, y=highlight_y, mode="markers",
                    name=fastest_label, legendgroup=f"{run_name}_fastest",
                    marker=dict(
                        color=fastest_color, size=11, opacity=1.0,
                        line=dict(width=1.5, color=bg_color),
                        symbol="circle",
                    ),
                    hovertext=highlight_h,
                    hovertemplate="%{hovertext}<br>%{y:.1f} m<extra></extra>",
                )
            )

    fig.update_layout(
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0.0),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    fig.update_xaxes(
        tickmode="array", tickvals=x_pos, ticktext=x_ticktext,
        range=[-0.5, n_corners - 0.5],
    )
    if all_y:
        p95 = float(np.nanpercentile(all_y, 95))
        upper = max(p95 * 1.10, 5.0)
        fig.update_yaxes(range=[0.0, upper])
    else:
        fig.update_yaxes(rangemode="tozero")
    return fig


_PHASE_COLORS: dict[str, str] = {
    "Straight": "#78909C",
    "Entry": "#56CCF2",
    "Exit": "#F28C40",
}


def compute_lap_analysis_corner_phases(
    turns: list[object],
) -> list[LapAnalysisCornerPhase]:
    """Return geometric Entry/Apex/Exit bounds for selected turns."""
    phases: list[LapAnalysisCornerPhase] = []
    for turn in sorted(turns, key=lambda t: float(t.s_entry_m)):
        s_entry = float(turn.s_entry_m)
        s_apex = float(turn.s_apex_m)
        s_exit = float(turn.s_exit_m)
        if not np.isfinite(s_entry + s_apex + s_exit):
            continue
        if s_exit <= s_entry:
            continue
        s_apex = float(np.clip(s_apex, s_entry, s_exit))
        phases.append(
            LapAnalysisCornerPhase(
                turn_id=int(turn.turn_id),
                s_entry_m=s_entry,
                s_apex_m=s_apex,
                s_exit_m=s_exit,
            )
        )
    return phases


def _full_session_gps(df: pl.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    """Return finite (lat, lng) arrays for the entire run, in sample order."""
    if df is None or df.is_empty():
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    if "VN_latitude" not in df.columns or "VN_longitude" not in df.columns:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    lat = df["VN_latitude"].to_numpy().astype(float)
    lng = df["VN_longitude"].to_numpy().astype(float)
    ok = np.isfinite(lat) & np.isfinite(lng)
    return lat[ok], lng[ok]


def lap_phase_track_fig(
    dfs: dict[str, pl.DataFrame],
    ref_run: str,
    ref_lap: int,
    cmp_run: str,
    cmp_lap: int,
    *,
    phases: list,
    active_turn_ids: set[int] | None = None,
) -> go.Figure:
    """GPS track coloured by geometric Lap Analysis phase.

    Apex is drawn as a point marker. The coloured line segments are Straight,
    Entry and Exit only.
    """
    comp = lap_comparison_arrays(dfs, ref_run, ref_lap, cmp_run, cmp_lap)
    s_m = np.asarray(comp["s_m"], dtype=float)
    lat = np.asarray(comp["ref_latitude"], dtype=float)
    lng = np.asarray(comp["ref_longitude"], dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lng)

    fig = make_dark_figure("Lap Analysis Phase Map", "Longitude", "Latitude")

    # Full-session GPS as a grey context layer so the user always sees the
    # whole track (figure-8 in skidpad, full lap in circuit) rather than only
    # the reference lap. Falls back silently if the run lacks GPS columns.
    full_lat, full_lng = _full_session_gps(dfs.get(ref_run))
    if full_lat.size > 0:
        fig.add_trace(go.Scattergl(
            x=full_lng, y=full_lat,
            mode="lines",
            line=dict(color="rgba(160,160,160,0.18)", width=2),
            showlegend=False, hoverinfo="skip",
        ))

    if not ok.any():
        if full_lat.size == 0:
            fig.add_annotation(
                x=0.5, y=0.5, xref="paper", yref="paper",
                text="GPS not available", showarrow=False,
                font=dict(color="#EBEBEB"),
            )
        return fig

    # Brighter outline for the reference lap on top of the session context.
    fig.add_trace(go.Scattergl(
        x=lng[ok], y=lat[ok],
        mode="lines",
        line=dict(color="rgba(160,160,160,0.45)", width=4),
        showlegend=False, hoverinfo="skip",
    ))

    # Straight is the only global phase trace. Entry/Exit are drawn per turn so
    # the visible blue/orange segments themselves carry the clicked turn_id.
    phase_labels = np.full(len(s_m), "Straight", dtype=object)
    for ph in phases:
        s_entry = float(ph.s_entry_m)
        s_exit = float(ph.s_exit_m)
        phase_labels[(s_m >= s_entry) & (s_m <= s_exit)] = "Corner"

    straight_mask = (phase_labels == "Straight") & ok
    if straight_mask.any():
        indices = np.where(straight_mask)[0]
        breaks = np.where(np.diff(indices) > 2)[0] + 1
        seg_starts = np.concatenate([[0], breaks])
        seg_ends = np.concatenate([breaks, [len(indices)]])
        xs: list[float] = []
        ys: list[float] = []
        for i0, i1 in zip(seg_starts, seg_ends):
            seg = indices[i0:i1]
            xs.extend(lng[seg].tolist())
            xs.append(float("nan"))
            ys.extend(lat[seg].tolist())
            ys.append(float("nan"))
        fig.add_trace(go.Scattergl(
            x=xs, y=ys,
            mode="lines",
            name="Straight",
            line=dict(color=_PHASE_COLORS["Straight"], width=3),
            hovertemplate="Straight<extra></extra>",
        ))

    entry_legend_shown = False
    exit_legend_shown = False
    for ph in phases:
        turn_id = int(ph.turn_id)
        is_active = active_turn_ids is None or turn_id in active_turn_ids

        for phase_name, s0, s1 in (
            ("Entry", float(ph.s_entry_m), float(ph.s_apex_m)),
            ("Exit", float(ph.s_apex_m), float(ph.s_exit_m)),
        ):
            seg_mask = ok & (s_m >= s0) & (s_m <= s1)
            if not seg_mask.any():
                continue
            phase_color = _PHASE_COLORS[phase_name]
            showlegend = False
            if is_active and phase_name == "Entry" and not entry_legend_shown:
                showlegend = True
                entry_legend_shown = True
            elif is_active and phase_name == "Exit" and not exit_legend_shown:
                showlegend = True
                exit_legend_shown = True
            fig.add_trace(go.Scattergl(
                x=lng[seg_mask],
                y=lat[seg_mask],
                mode="lines+markers",
                line=dict(color=phase_color if is_active else "rgba(235,235,235,0.42)", width=5),
                marker=dict(
                    size=7,
                    color=phase_color if is_active else "rgba(235,235,235,0.42)",
                    line=dict(width=0),
                ),
                customdata=[["turn", turn_id] for _ in range(int(seg_mask.sum()))],
                name=phase_name if is_active else f"T{turn_id} excluded",
                showlegend=showlegend,
                hovertemplate=(
                    f"T{turn_id} · {phase_name}<br>"
                    + ("Click to exclude" if is_active else "Click to include")
                    + "<extra></extra>"
                ),
            ))

        apex_lng = float(np.interp(ph.s_apex_m, s_m, lng))
        apex_lat = float(np.interp(ph.s_apex_m, s_m, lat))
        fig.add_trace(go.Scattergl(
            x=[apex_lng], y=[apex_lat],
            mode="markers+text" if is_active else "markers",
            text=[f"T{turn_id}"] if is_active else None,
            textposition="top center",
            textfont=dict(color="#FFFFFF" if is_active else "rgba(235,235,235,0.45)", size=9),
            marker=dict(
                size=7 if is_active else 6,
                color="#F2C94C" if is_active else "rgba(235,235,235,0.45)",
                symbol="circle" if is_active else "circle-open",
            ),
            customdata=[["turn", turn_id]],
            name="Apex",
            showlegend=False,
            hovertemplate=f"T{turn_id}  apex {ph.s_apex_m:.0f} m<extra></extra>",
        ))

    fig.update_layout(
        height=430,
        showlegend=True,
        clickmode="event",
        dragmode="pan",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=12),
        ),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1.0)

    # Force the visible bbox to cover the full session GPS so the second skidpad
    # circle (or far parts of a circuit) are not clipped by scaleanchor's
    # auto-shrinking behaviour.
    if full_lat.size > 0:
        lat_lo, lat_hi = float(full_lat.min()), float(full_lat.max())
        lng_lo, lng_hi = float(full_lng.min()), float(full_lng.max())
        lat_pad = max(0.05 * (lat_hi - lat_lo), 1e-5)
        lng_pad = max(0.05 * (lng_hi - lng_lo), 1e-5)
        fig.update_xaxes(range=[lng_lo - lng_pad, lng_hi + lng_pad])
        fig.update_yaxes(range=[lat_lo - lat_pad, lat_hi + lat_pad])
    return fig


def _add_event_window(fig: go.Figure, start_m: float, end_m: float, rows: int) -> None:
    """Shade the selected event across all subplot rows."""
    for row in range(1, rows + 1):
        fig.add_vrect(
            x0=start_m,
            x1=end_m,
            fillcolor="rgba(242,201,76,0.14)",
            line_width=0,
            layer="below",
            row=row,
            col=1,
        )
        fig.add_vline(
            x=start_m,
            line=dict(color="rgba(242,201,76,0.65)", dash="dot", width=1),
            row=row,
            col=1,
        )
        fig.add_vline(
            x=end_m,
            line=dict(color="rgba(242,201,76,0.65)", dash="dot", width=1),
            row=row,
            col=1,
        )


def lap_event_detail_fig(
    dfs: dict[str, pl.DataFrame],
    ref_run: str,
    ref_lap: int,
    cmp_run: str,
    cmp_lap: int,
    start_m: float,
    end_m: float,
    *,
    padding_m: float = 35.0,
    signal_keys: list[str] | None = None,
    x_axis_mode: str = "distance",
) -> go.Figure:
    """Zoomed event plot using the same signal-slot logic as Video Analysis."""
    ref_color = "#FFFFFF"
    comp = lap_comparison_arrays(dfs, ref_run, ref_lap, cmp_run, cmp_lap)
    s_m = comp["s_m"]
    lo = max(float(np.nanmin(s_m)), float(start_m) - padding_m)
    hi = min(float(np.nanmax(s_m)), float(end_m) + padding_m)
    mask = (s_m >= lo) & (s_m <= hi)
    x_mode = "time" if x_axis_mode == "time" else "distance"

    selected_keys = [
        key for key in (signal_keys or ["delta_s", "vx_mps", "throttle_pct", "steering_deg"])
        if key in LAP_SIGNAL_OPTIONS
    ]
    if not selected_keys:
        selected_keys = ["delta_s"]
    rows = len(selected_keys)

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=max(0.018, min(0.05, 0.20 / max(rows, 1))),
    )
    base = make_dark_figure("Selected Event Detail")
    fig.update_layout(
        title=base.layout.title,
        paper_bgcolor=base.layout.paper_bgcolor,
        plot_bgcolor=base.layout.plot_bgcolor,
        font=base.layout.font,
        height=max(380, 155 * rows + 150),
        showlegend=False,
        hovermode="x",
        hoversubplots="axis",
        hoverdistance=-1,
        spikedistance=-1,
    )

    ref_time_s = comp["ref_time_s"]
    cmp_time_s = comp["cmp_time_s"]
    ref_x = s_m if x_mode == "distance" else ref_time_s
    cmp_x = s_m if x_mode == "distance" else cmp_time_s
    x_for_single = ref_x
    event_start_x = float(start_m) if x_mode == "distance" else float(np.interp(start_m, s_m, ref_time_s))
    event_end_x = float(end_m) if x_mode == "distance" else float(np.interp(end_m, s_m, ref_time_s))
    _add_event_window(fig, event_start_x, event_end_x, rows=rows)

    for row in range(1, rows + 1):
        xaxis_kwargs = {"matches": "x"} if row > 1 else {}
        fig.update_xaxes(
            color="#E5E5E5",
            gridcolor="rgba(128,128,128,0.2)",
            linecolor="#E5E5E5" if row == rows else "rgba(0,0,0,0)",
            tickcolor="#E5E5E5",
            showline=row == rows,
            showticklabels=row == rows,
            showspikes=True,
            spikemode="across+marker",
            spikesnap="cursor",
            spikecolor="rgba(255,255,255,0.85)",
            spikedash="dot",
            spikethickness=1,
            row=row,
            col=1,
            **xaxis_kwargs,
        )
        fig.update_yaxes(
            color="#E5E5E5",
            gridcolor="rgba(128,128,128,0.2)",
            linecolor="#E5E5E5",
            tickcolor="#E5E5E5",
            zeroline=False,
            row=row,
            col=1,
        )

    pair_map = {
        "vx_mps": ("ref_vx_mps", "cmp_vx_mps", "#4DB3F2"),
        "throttle_pct": ("ref_throttle_pct", "cmp_throttle_pct", "#73D973"),
        "brake_pct": ("ref_brake_pct", "cmp_brake_pct", "#F27070"),
        "steering_deg": ("ref_steering_rad", "cmp_steering_rad", "#F2C94C"),
        "ax_mps2": ("ref_ax_mps2", "cmp_ax_mps2", "#F2994A"),
        "ay_mps2": ("ref_ay_mps2", "cmp_ay_mps2", "#D973D9"),
        "radius_m": ("ref_radius_m", "cmp_radius_m", "#56CCF2"),
        "curvature_1pm": ("ref_curvature_1pm", "cmp_curvature_1pm", "#9B51E0"),
    }
    single_map = {
        "delta_s": ("delta_s", "#F2C94C"),
        "loss_rate_ms_10m": ("loss_rate_ms_10m", "#EB5757"),
        "dvx_mps": ("dvx_mps", "#56CCF2"),
        "dthrottle_pct": ("dthrottle_pct", "#73D973"),
        "dbrake_pct": ("dbrake_pct", "#F27070"),
        "dsteering_deg": ("dsteering_deg", "#F2C94C"),
        "dax_mps2": ("dax_mps2", "#F2994A"),
        "day_mps2": ("day_mps2", "#D973D9"),
        "dradius_m": ("dradius_m", "#56CCF2"),
        "dcurvature_1pm": ("dcurvature_1pm", "#9B51E0"),
    }

    ref_label = str(comp["ref_label"])
    cmp_label = str(comp["cmp_label"])
    for row, key in enumerate(selected_keys, start=1):
        ylabel = LAP_SIGNAL_OPTIONS[key]["ylabel"]
        if key in pair_map:
            ref_key, cmp_key, color = pair_map[key]
            ref_y = comp[ref_key]
            cmp_y = comp[cmp_key]
            if key == "steering_deg":
                ref_y = np.rad2deg(ref_y)
                cmp_y = np.rad2deg(cmp_y)
            fig.add_trace(
                go.Scatter(
                    x=ref_x[mask],
                    y=ref_y[mask],
                    customdata=s_m[mask],
                    mode="lines",
                    name=f"{ref_label}",
                    legendgroup=f"{key}_ref",
                    showlegend=False,
                    line=dict(color=ref_color, width=1.45),
                    hovertemplate=f"<span style='color:{ref_color}'>%{{y:.4f}}</span><extra></extra>",
                    hoverlabel=dict(
                        bgcolor="rgba(20,20,23,0.96)",
                        bordercolor=ref_color,
                        font=dict(color=ref_color),
                    ),
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=cmp_x[mask],
                    y=cmp_y[mask],
                    customdata=s_m[mask],
                    mode="lines",
                    name=f"{cmp_label}",
                    legendgroup=f"{key}_cmp",
                    showlegend=False,
                    line=dict(color=color, width=1.65),
                    hovertemplate=f"<span style='color:{color}'>%{{y:.4f}}</span><extra></extra>",
                    hoverlabel=dict(
                        bgcolor="rgba(20,20,23,0.96)",
                        bordercolor=color,
                        font=dict(color=color),
                    ),
                ),
                row=row,
                col=1,
            )
        else:
            comp_key, color = single_map[key]
            fig.add_trace(
                go.Scatter(
                    x=x_for_single[mask],
                    y=comp[comp_key][mask],
                    customdata=s_m[mask],
                    mode="lines",
                    name=LAP_SIGNAL_OPTIONS[key]["label"],
                    showlegend=False,
                    line=dict(color=color, width=1.8),
                    hovertemplate=f"<span style='color:{color}'>%{{y:.4f}}</span><extra></extra>",
                    hoverlabel=dict(
                        bgcolor="rgba(20,20,23,0.96)",
                        bordercolor=color,
                        font=dict(color=color),
                    ),
                ),
                row=row,
                col=1,
            )

        fig.update_yaxes(title_text=ylabel, row=row, col=1)
        if key in {"throttle_pct", "brake_pct"}:
            fig.update_yaxes(range=[0, 105], row=row, col=1)
    fig.update_xaxes(
        title_text="Time [s]" if x_mode == "time" else "Distance from start line [m]",
        row=rows,
        col=1,
    )
    return fig


def corner_detail_fig(
    dfs: dict[str, pl.DataFrame],
    ref_run: str,
    ref_lap: int,
    cmp_run: str,
    cmp_lap: int,
    turn: object,
    *,
    padding_m: float = 30.0,
    phases: CornerPhases | LapAnalysisCornerPhase | None = None,
    signal_keys: list[str] | None = None,
    x_axis_mode: str = "distance",
) -> go.Figure:
    """Signal overlay zoomed to one corner with geometric phase markers.

    For Lap Analysis, apex is a point and the highlighted window is the
    curvature-bounded corner from entry to exit.
    """
    if isinstance(phases, LapAnalysisCornerPhase):
        start_m = float(phases.s_entry_m)
        end_m = float(phases.s_exit_m)
        s_apex = float(phases.s_apex_m)
    elif phases is not None:
        start_m = float(phases.s_brake_on_m)
        end_m = float(phases.s_exit_end_m)
        s_apex = float(phases.s_apex_m)
    else:
        start_m = float(turn.s_entry_m)
        end_m = float(turn.s_exit_m)
        s_apex = float(turn.s_apex_m)

    fig = lap_event_detail_fig(
        dfs, ref_run, ref_lap, cmp_run, cmp_lap,
        start_m, end_m, padding_m=padding_m,
        signal_keys=signal_keys,
        x_axis_mode=x_axis_mode,
    )
    fig.update_layout(title=dict(text=f"Turn {int(turn.turn_id)} - Entry · Apex · Exit"))

    selected_keys = [
        key for key in (signal_keys or ["delta_s", "vx_mps", "throttle_pct", "steering_deg"])
        if key in LAP_SIGNAL_OPTIONS
    ] or ["delta_s"]
    rows = len(selected_keys)
    if x_axis_mode == "time":
        comp = lap_comparison_arrays(dfs, ref_run, ref_lap, cmp_run, cmp_lap)
        marker_x = lambda s: float(np.interp(float(s), comp["s_m"], comp["ref_time_s"]))
    else:
        marker_x = lambda s: float(s)
    if isinstance(phases, LapAnalysisCornerPhase):
        markers = (
            (phases.s_entry_m, "#56CCF2", "entry"),
            (s_apex, "#F2C94C", "apex"),
            (phases.s_exit_m, "#F28C40", "exit"),
        )
    elif phases is not None:
        markers = (
            (phases.s_brake_on_m, "#EB5757", "brake-on"),
            (phases.s_brake_off_m, "#F2994A", "brake-off"),
            (s_apex, "#F2C94C", "apex"),
            (phases.s_exit_end_m, "#9B51E0", "exit-end"),
        )
    else:
        markers = ((s_apex, "#F2C94C", "apex"),)
    for x, color, label in markers:
        is_apex = label == "apex"
        line = dict(
            color=color,
            dash="dash" if is_apex else "dot",
            width=2.2 if is_apex else 1.1,
        )
        for row in range(1, rows + 1):
            kwargs = {}
            if is_apex and row == 1:
                kwargs = {
                    "annotation_text": "Apex",
                    "annotation_position": "top",
                    "annotation_font_color": color,
                    "annotation_font_size": 11,
                }
            fig.add_vline(x=marker_x(x), line=line, row=row, col=1, **kwargs)
    return fig


def corner_phase_delta_table(
    dfs: dict[str, pl.DataFrame],
    ref_run: str,
    ref_lap: int,
    cmp_run: str,
    cmp_lap: int,
    *,
    turns: list,
    apex_half_window_m: float = 5.0,
    brake_threshold_pct: float = 5.0,
) -> tuple[pl.DataFrame, list[LapAnalysisCornerPhase]]:
    """Per-corner Δt attribution split into geometric Entry / Exit.

    Apex is a marker point, not a phase with duration. Δt for each segment is
    the change in cumulative `delta_s = t_cmp - t_ref` between geometric
    distance bounds detected on the reference lap.
    """
    _ = apex_half_window_m, brake_threshold_pct
    schema = {
        "Turn": pl.Int64,
        "Δt entry [s]": pl.Float64,
        "Δt exit [s]": pl.Float64,
        "Total [s]": pl.Float64,
        "Worst": pl.Utf8,
        "s_entry_m": pl.Float64,
        "s_apex_m": pl.Float64,
        "s_exit_m": pl.Float64,
    }
    if not turns:
        return pl.DataFrame(schema=schema), []
    comp = lap_comparison_arrays(dfs, ref_run, ref_lap, cmp_run, cmp_lap)
    s = np.asarray(comp["s_m"], dtype=float)
    delta_s = np.asarray(comp["delta_s"], dtype=float)
    if s.size == 0:
        return pl.DataFrame(schema=schema), []
    phases = compute_lap_analysis_corner_phases(turns)
    if not phases:
        return pl.DataFrame(schema=schema), []

    def dt_between(a: float, b: float) -> float:
        if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
            return 0.0
        return float(np.interp(b, s, delta_s) - np.interp(a, s, delta_s))

    rows: list[dict[str, object]] = []
    for ph in phases:
        dt_entry = dt_between(ph.s_entry_m, ph.s_apex_m)
        dt_exit = dt_between(ph.s_apex_m, ph.s_exit_m)
        contribs = {"Entry": dt_entry, "Exit": dt_exit}
        worst = max(contribs.items(), key=lambda kv: abs(kv[1]))[0]
        rows.append({
            "Turn": int(ph.turn_id),
            "Δt entry [s]": round(dt_entry, 4),
            "Δt exit [s]": round(dt_exit, 4),
            "Total [s]": round(dt_entry + dt_exit, 4),
            "Worst": worst,
            "s_entry_m": round(ph.s_entry_m, 1),
            "s_apex_m": round(ph.s_apex_m, 1),
            "s_exit_m": round(ph.s_exit_m, 1),
        })
    return pl.DataFrame(rows, schema=schema), phases


def corner_phase_delta_fig(table: pl.DataFrame) -> go.Figure:
    """Stacked horizontal bars per corner: Δt by geometric Entry / Exit.

    Positive segments push right (cmp slower than ref in that phase), negative
    push left (cmp faster). Corners are sorted by |Total Δt| descending so the
    biggest opportunities sit on top.
    """
    fig = make_dark_figure("Where is the time? · Δt by corner geometry")
    if table.is_empty():
        fig.update_layout(annotations=[dict(
            text="No corners detected.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#E5E5E5"),
        )])
        return fig

    sorted_tbl = (
        table
        .with_columns(pl.col("Total [s]").abs().alias("_abs"))
        .sort("_abs", descending=False)  # smallest at the bottom in plotly horizontal bars
        .drop("_abs")
    )
    turns = [f"T{int(t)}" for t in sorted_tbl["Turn"].to_list()]
    phase_cols = (
        ("Entry", "Δt entry [s]", "#56CCF2"),
        ("Exit", "Δt exit [s]", "#F28C40"),
    )
    for label, col, color in phase_cols:
        values = sorted_tbl[col].to_list()
        fig.add_trace(go.Bar(
            x=values,
            y=turns,
            name=label,
            orientation="h",
            marker=dict(color=color),
            hovertemplate=f"%{{y}} · {label}<br>Δt = %{{x:+.3f}} s<extra></extra>",
        ))
    totals = sorted_tbl["Total [s]"].to_list()
    fig.add_trace(go.Scatter(
        x=totals,
        y=turns,
        mode="markers",
        marker=dict(color="#FFFFFF", size=8, symbol="diamond", line=dict(color="#111", width=1)),
        name="Total",
        hovertemplate="%{y} · Total Δt = %{x:+.3f} s<extra></extra>",
    ))
    fig.update_layout(
        barmode="relative",
        height=max(280, 28 * len(turns) + 140),
        xaxis=dict(
            title="Δt cmp − ref [s]   (positive = cmp slower)",
            zeroline=True, zerolinecolor="rgba(229,229,229,0.4)", zerolinewidth=1.2,
            color="#E5E5E5", gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(color="#E5E5E5", gridcolor="rgba(128,128,128,0.15)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def corner_gg_fig(
    dfs: dict[str, pl.DataFrame],
    ref_run: str,
    ref_lap: int,
    cmp_run: str,
    cmp_lap: int,
    phases: LapAnalysisCornerPhase,
) -> go.Figure:
    """GG diagram for a single corner (s_entry → s_exit), ref vs cmp."""
    fig = make_dark_figure(
        f"T{phases.turn_id} · GG",
        "ay [m/s²]",
        "ax [m/s²]",
    )
    s_lo = phases.s_entry_m
    s_hi = phases.s_exit_m
    entries = [
        (ref_run, ref_lap, "#56CCF2"),
        (cmp_run, cmp_lap, "#F28C40"),
    ]
    for run_name, lap_id, color in entries:
        try:
            data = _lap_comparison_data(dfs[run_name], int(lap_id))
        except Exception:
            continue
        s_m = data["s_m"]
        mask = (s_m >= s_lo) & (s_m <= s_hi)
        if int(mask.sum()) < 2:
            continue
        ax = data["ax_mps2"][mask]
        ay = data["ay_mps2"][mask]
        gg_s_m = s_m[mask]
        valid = np.isfinite(ax) & np.isfinite(ay)
        label = _comparison_lap_label(run_name, int(lap_id))
        fig.add_trace(go.Scatter(
            x=ay[valid],
            y=ax[valid],
            customdata=gg_s_m[valid],
            mode="markers",
            name=label,
            marker=dict(color=color, size=4, opacity=0.8),
            hovertemplate="ay=%{x:.2f}<br>ax=%{y:.2f}<extra>" + label + "</extra>",
        ))
    fig.update_layout(
        xaxis=dict(
            zeroline=True, zerolinecolor="rgba(229,229,229,0.4)", zerolinewidth=1.2,
        ),
        yaxis=dict(
            zeroline=True, zerolinecolor="rgba(229,229,229,0.4)", zerolinewidth=1.2,
            scaleanchor="x", scaleratio=1,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def lap_signal_overlay_fig(
    dfs: dict[str, pl.DataFrame],
    ref_run: str,
    ref_lap: int,
    cmp_run: str,
    cmp_lap: int,
    *,
    turns: list | None = None,
    active_turn_ids: set[int] | None = None,
    signal_keys: list[str] | None = None,
    brake_threshold_pct: float = 5.0,
    throttle_threshold_pct: float = 40.0,
    ay_threshold_mps2: float | None = None,
    show_phase_background: bool = False,
) -> go.Figure:
    """Configurable signal overlay on distance for two selected laps."""
    comp = lap_comparison_arrays(dfs, ref_run, ref_lap, cmp_run, cmp_lap)
    s_m = comp["s_m"]
    ref_label = str(comp["ref_label"])
    cmp_label = str(comp["cmp_label"])
    ref_color = "#4DB3F2"
    cmp_color = "#F28C40"
    selected_keys = [
        key for key in (signal_keys or [
            "throttle_pct", "brake_pct", "steering_deg", "vx_mps",
        ])
        if key in LAP_SIGNAL_OPTIONS
    ]
    if not selected_keys:
        selected_keys = ["delta_s"]
    rows = len(selected_keys)

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=max(0.010, min(0.030, 0.12 / max(rows, 1))),
        subplot_titles=[LAP_SIGNAL_OPTIONS[key]["label"] for key in selected_keys],
    )
    base = make_dark_figure("Lap Signal Overlay")
    fig.update_layout(
        title=base.layout.title,
        paper_bgcolor=base.layout.paper_bgcolor,
        plot_bgcolor=base.layout.plot_bgcolor,
        font=base.layout.font,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(20,20,23,0.75)",
        ),
        height=max(300, 120 * rows),
        hovermode="x unified",
    )
    for row in range(1, rows + 1):
        fig.update_xaxes(
            color="#E5E5E5",
            gridcolor="rgba(128,128,128,0.2)",
            linecolor="#E5E5E5",
            tickcolor="#E5E5E5",
            row=row,
            col=1,
        )
        fig.update_yaxes(
            color="#E5E5E5",
            gridcolor="rgba(128,128,128,0.2)",
            linecolor="#E5E5E5",
            tickcolor="#E5E5E5",
            row=row,
            col=1,
        )

    if show_phase_background:
        phase = _lap_analysis_phase_array(
            comp,
            turns=turns,
            active_turn_ids=active_turn_ids,
            brake_threshold_pct=brake_threshold_pct,
            throttle_threshold_pct=throttle_threshold_pct,
            ay_threshold_mps2=ay_threshold_mps2,
        )
        phase_fill = {
            "Braking": "rgba(239,83,80,0.10)",
            "Corner": "rgba(242,201,76,0.09)",
            "Entry": "rgba(86,204,242,0.09)",
            "Exit": "rgba(242,140,64,0.09)",
            "Corner Entry": "rgba(86,204,242,0.09)",
            "Apex": "rgba(242,201,76,0.11)",
            "Corner Exit": "rgba(155,81,224,0.09)",
            "Acceleration": "rgba(76,175,80,0.08)",
            "Straight": "rgba(120,144,156,0.035)",
            "Ignored": "rgba(58,63,70,0.16)",
        }
        for phase_name, start, end in _phase_segments(phase):
            if end <= start:
                continue
            for row in range(1, rows + 1):
                fig.add_vrect(
                    x0=float(s_m[start]),
                    x1=float(s_m[end]),
                    fillcolor=phase_fill.get(str(phase_name), "rgba(128,128,128,0.04)"),
                    line_width=0,
                    layer="below",
                    row=row,
                    col=1,
                )

    pair_map = {
        "vx_mps": ("ref_vx_mps", "cmp_vx_mps"),
        "throttle_pct": ("ref_throttle_pct", "cmp_throttle_pct"),
        "brake_pct": ("ref_brake_pct", "cmp_brake_pct"),
        "steering_deg": ("ref_steering_rad", "cmp_steering_rad"),
        "ax_mps2": ("ref_ax_mps2", "cmp_ax_mps2"),
        "ay_mps2": ("ref_ay_mps2", "cmp_ay_mps2"),
        "radius_m": ("ref_radius_m", "cmp_radius_m"),
        "curvature_1pm": ("ref_curvature_1pm", "cmp_curvature_1pm"),
    }
    single_map = {
        "delta_s": ("delta_s", "#F2C94C"),
        "loss_rate_ms_10m": ("loss_rate_ms_10m", "#EB5757"),
        "dvx_mps": ("dvx_mps", "#56CCF2"),
        "dthrottle_pct": ("dthrottle_pct", "#73D973"),
        "dbrake_pct": ("dbrake_pct", "#F27070"),
        "dsteering_deg": ("dsteering_deg", "#F2C94C"),
        "dax_mps2": ("dax_mps2", "#F2994A"),
        "day_mps2": ("day_mps2", "#D973D9"),
        "dradius_m": ("dradius_m", "#56CCF2"),
        "dcurvature_1pm": ("dcurvature_1pm", "#9B51E0"),
    }

    legend_shown = False
    for row, key in enumerate(selected_keys, start=1):
        ylabel = LAP_SIGNAL_OPTIONS[key]["ylabel"]
        if key in pair_map:
            ref_key, cmp_key = pair_map[key]
            ref_y = comp[ref_key]
            cmp_y = comp[cmp_key]
            if key == "steering_deg":
                ref_y = np.rad2deg(ref_y)
                cmp_y = np.rad2deg(cmp_y)
            fig.add_trace(
                go.Scatter(
                    x=s_m, y=ref_y, mode="lines",
                    name=ref_label, legendgroup="ref", showlegend=not legend_shown,
                    line=dict(color=ref_color, width=1.7),
                ),
                row=row, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=s_m, y=cmp_y, mode="lines",
                    name=cmp_label, legendgroup="cmp", showlegend=not legend_shown,
                    line=dict(color=cmp_color, dash="dash", width=1.7),
                ),
                row=row, col=1,
            )
            legend_shown = True
        else:
            comp_key, color = single_map[key]
            fig.add_trace(
                go.Scatter(
                    x=s_m,
                    y=comp[comp_key],
                    mode="lines",
                    name=LAP_SIGNAL_OPTIONS[key]["label"],
                    line=dict(color=color, width=2.0),
                    hovertemplate="Distance %{x:.1f} m<br>%{y:+.3f}<extra></extra>",
                ),
                row=row, col=1,
            )
            fig.add_hline(
                y=0.0,
                line=dict(color="rgba(235,235,235,0.35)", dash="dash"),
                row=row,
                col=1,
            )
        fig.update_yaxes(title_text=ylabel, row=row, col=1)

    fig.update_xaxes(title_text="Distance from start line [m]", row=rows, col=1)

    return fig


def corner_curvature_fig(
    dfs: dict[str, pl.DataFrame],
    x_mode: str = "laps",
) -> go.Figure:
    """Average curvature per lap using |ay| / vx²."""
    fig = make_dark_figure(
        title="Corner Curvature per Lap",
        xlabel="Lap" if x_mode == "laps" else "Lap time [s]",
        ylabel="Average curvature [1/m]",
    )
    all_laps: list[np.ndarray] = []

    for i, (run_name, df) in enumerate(dfs.items()):
        d, lat_accel_col, speed_col = _prep_steering_metrics(df)
        res = _steering_metrics_per_lap(d, lat_accel_col, speed_col)
        ok = np.isfinite(res["mean_curvature"]) & np.isfinite(res["lt_val"])
        if not ok.any():
            continue
        x_arr, order, _ = per_lap_axis(
            res["lap_list"][ok], res["lt_val"][ok], x_mode,
        )
        y_arr = res["mean_curvature"][ok][order]
        lap_ord = res["lap_list"][ok][order]
        color = _driver_color(run_name, i)
        add_lap_scatter(fig, x_arr, y_arr, lap_ord, name=run_name, color=color)
        add_trend_line(fig, x_arr, y_arr, color=color)
        all_laps.append(res["lap_list"][ok])

    if x_mode == "laps" and all_laps:
        ticks = np.unique(np.concatenate(all_laps)).astype(int)
        fig.update_xaxes(tickvals=ticks)
    return fig


# ── Standalone CLI ────────────────────────────────────────────────────────────

def main() -> None:
    from utils import load_data
    df = load_data(CSV_PATH)
    summary = driver_summary(df)
    print("\n─── Driver summary ───")
    if summary["valid_laps"] == 0:
        for w in summary.get("warnings", []):
            print(w)
        return
    print(
        f"Valid laps: {summary['valid_laps']} | "
        f"Fastest L{summary['fastest_lap']} ({summary['fastest_lt']:.2f}s)\n"
        f"Mean TP: {summary['mean_throttle_pct']:.1f}% | "
        f"Full throttle: {summary['mean_full_t']:.2f}s "
        f"({summary['mean_full_pct']:.1f}% of lap) | "
        f"Off throttle: {summary['mean_off_pct']:.1f}% | "
        f"Median |dTP/dt|: {summary['mean_speed']:.2f}%/s "
        f"(per-lap max {summary['max_speed']:.1f})\n"
        f"Brake aggressiveness: {summary['mean_brake_aggr']:.1f}%/s | "
        f"Brake release smoothness: {summary['mean_brake_release']:.1f}%/s\n"
        f"Steering smoothness: {summary['mean_steering_smoothness']:.3f} deg | "
        f"Steering integral: {summary['mean_steering_integral']:.1f} deg*m | "
        f"Curvature: {summary['mean_curvature']:.5f} 1/m"
    )
    print(summary["table"])

    dfs = {CSV_PATH: df}
    throttle_histogram_fig(dfs).show()
    full_throttle_time_fig(dfs).show()
    throttle_speed_fig(dfs).show()
    braking_effort_fig(dfs).show()
    braking_aggressiveness_fig(dfs).show()
    brake_release_smoothness_fig(dfs).show()
    steering_smoothness_fig(dfs).show()
    steering_integral_fig(dfs).show()
    corner_curvature_fig(dfs).show()


if __name__ == "__main__":
    main()
