"""driver.py
----------
Driver Performance KPIs — throttle, brake, and steering behaviour analysis.

Metrics:
  1. Throttle position histogram                (% of samples per 5 % bin)
  2. Full Throttle Time per lap                 (seconds where TP > 95 %)
  3. Throttle Speed per lap                     (median |dTP/dt| with TP < 100 %
                                                 and brake released)
  4. Braking Effort                             (Brake [%] vs Filtering_VN_ax [m/s²])
  5. Braking Aggressiveness per lap             (mean dBrake/dt for dBrake/dt > 5 %/s)
  6. Brake Release Smoothness per lap           (mean |dBrake/dt| for dBrake/dt < -5 %/s)
  7. Steering Smoothness                        (mean |dSteering/dt| per lap)
  8. Corner Curvature                           (mean |ay| / vx² per lap)

All figure builders accept a `dfs: dict[str, pl.DataFrame]` so a single run
or a multi-run comparison (driver A vs driver B) share the same code path.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter

from utils import (
    COMPLETE_LAPS_MARKER,
    add_lap_scatter,
    add_trend_line,
    ensure_complete_laps_df,
    exclude_lap0_and_last_lap,
    fill_short_false_gaps,
    keep_min_duration_segments,
    make_dark_figure,
    per_lap_axis,
    robust_dt,
    unique_laps,
)

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
STEERING_SMOOTH_WINDOW_S = 0.21  # [s]   light smoothing before differentiating steering
CURVATURE_MIN_SPEED_MPS  = 3.0   # [m/s] avoid ay / vx² blow-ups near standstill
_LONG_ACCEL_CANDIDATES  = ("Filtering_VN_ax", "VN_ax")
_LAT_ACCEL_CANDIDATES   = ("Filtering_VN_ay", "VN_ay")
_SPEED_CANDIDATES       = ("VN_vx", "Est_vxCOG")

# Per-driver/run colours — extend if needed
DRIVER_COLORS = ("#4DB3F2", "#F27070", "#F2C94C", "#73D973")

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

_PHASE_ORDER  = ("ACCELERATING", "BRAKING", "COASTING", "PLAUSIBILITY")
_PHASE_LABELS = {
    "ACCELERATING": "Accelerating",
    "BRAKING":      "Braking",
    "COASTING":     "Coasting",
    "PLAUSIBILITY": "Plausibility (both pedals)",
}

_MAP_MAX_COLS = 4


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
        laps_col = df["laps"].to_numpy().astype(float)
        lt_col   = df["laptime"].to_numpy().astype(float)
        thr_col  = df["Throttle"].to_numpy().astype(float)
        brk_col  = df["Brake"].to_numpy().astype(float)

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

        laps = df["laps"].to_numpy().astype(float)
        thr  = df["Throttle"].to_numpy().astype(float)
        brk  = df["Brake"].to_numpy().astype(float)
        lat  = df["VN_latitude"].to_numpy().astype(float)
        lng  = df["VN_longitude"].to_numpy().astype(float)

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
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    d = {c: df[c].to_numpy().astype(float) for c in required_cols}
    if COMPLETE_LAPS_MARKER in df.columns:
        d[COMPLETE_LAPS_MARKER] = df[COMPLETE_LAPS_MARKER].to_numpy().astype(float)
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


def _smooth_steering_signal(steering_rad: np.ndarray, dt_s: float) -> np.ndarray:
    """Lightly smooth steering so quantisation noise does not dominate dSteering/dt."""
    window = _savgol_window_samples(len(steering_rad), dt_s, STEERING_SMOOTH_WINDOW_S)
    if window == 0:
        return steering_rad.copy()
    return savgol_filter(steering_rad, window_length=window, polyorder=2, mode="interp")


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


def _steering_metrics_per_lap(
    d: dict[str, np.ndarray],
    lat_accel_col: str,
    speed_col: str,
) -> dict:
    """Return steering smoothness and curvature metrics per lap."""
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

    dt_s = robust_dt(time_s)
    steering_smooth = _smooth_steering_signal(steering_rad, dt_s)
    steering_rate_abs = np.abs(np.gradient(steering_smooth, time_s))

    curvature_inv_m = np.full(len(vx_mps), np.nan)
    curvature_mask = np.isfinite(ay_mps2) & np.isfinite(vx_mps) & (vx_mps >= CURVATURE_MIN_SPEED_MPS)
    curvature_inv_m[curvature_mask] = np.abs(ay_mps2[curvature_mask]) / np.square(vx_mps[curvature_mask])

    lap_list = unique_laps(laps)
    n = len(lap_list)
    lt_val = np.full(n, np.nan)
    mean_steering_smoothness = np.full(n, np.nan)
    mean_curvature = np.full(n, np.nan)

    for i, lap in enumerate(lap_list):
        lm = laps == lap
        if not lm.any():
            continue
        lt_val[i] = float(np.nanmax(laptime[lm]))
        sm = lm & np.isfinite(steering_rate_abs)
        if sm.any():
            mean_steering_smoothness[i] = float(np.nanmean(steering_rate_abs[sm]))
        cm = lm & np.isfinite(curvature_inv_m)
        if cm.any():
            mean_curvature[i] = float(np.nanmean(curvature_inv_m[cm]))

    table = pl.DataFrame({
        "Lap": lap_list.astype(int),
        "Steering smoothness [rad/s]": np.round(mean_steering_smoothness, 4),
        "Curvature [1/m]": np.round(mean_curvature, 5),
    })

    return {
        "lap_list": lap_list,
        "lt_val": lt_val,
        "mean_steering_smoothness": mean_steering_smoothness,
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
            lat_accel_col = _lateral_accel_col(df)
            speed_col = _speed_col(df)
            steer_d = _prep(df, extra_cols=("Steering", lat_accel_col, speed_col))
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
        color     = DRIVER_COLORS[i % len(DRIVER_COLORS)]
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
        color   = DRIVER_COLORS[i % len(DRIVER_COLORS)]
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
        color   = DRIVER_COLORS[i % len(DRIVER_COLORS)]
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
        color = DRIVER_COLORS[i % len(DRIVER_COLORS)]
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
        color = DRIVER_COLORS[i % len(DRIVER_COLORS)]
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
        color = DRIVER_COLORS[i % len(DRIVER_COLORS)]
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
    """Average steering smoothness per lap."""
    fig = make_dark_figure(
        title="Steering Smoothness per Lap",
        xlabel="Lap" if x_mode == "laps" else "Lap time [s]",
        ylabel="Average steering smoothness [rad/s]",
    )
    all_laps: list[np.ndarray] = []

    for i, (run_name, df) in enumerate(dfs.items()):
        lat_accel_col = _lateral_accel_col(df)
        speed_col = _speed_col(df)
        d = _prep(df, extra_cols=("Steering", lat_accel_col, speed_col))
        res = _steering_metrics_per_lap(d, lat_accel_col, speed_col)
        ok = np.isfinite(res["mean_steering_smoothness"]) & np.isfinite(res["lt_val"])
        if not ok.any():
            continue
        x_arr, order, _ = per_lap_axis(
            res["lap_list"][ok], res["lt_val"][ok], x_mode,
        )
        y_arr = res["mean_steering_smoothness"][ok][order]
        lap_ord = res["lap_list"][ok][order]
        color = DRIVER_COLORS[i % len(DRIVER_COLORS)]
        add_lap_scatter(fig, x_arr, y_arr, lap_ord, name=run_name, color=color)
        add_trend_line(fig, x_arr, y_arr, color=color)
        all_laps.append(res["lap_list"][ok])

    if x_mode == "laps" and all_laps:
        ticks = np.unique(np.concatenate(all_laps)).astype(int)
        fig.update_xaxes(tickvals=ticks)
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
        lat_accel_col = _lateral_accel_col(df)
        speed_col = _speed_col(df)
        d = _prep(df, extra_cols=("Steering", lat_accel_col, speed_col))
        res = _steering_metrics_per_lap(d, lat_accel_col, speed_col)
        ok = np.isfinite(res["mean_curvature"]) & np.isfinite(res["lt_val"])
        if not ok.any():
            continue
        x_arr, order, _ = per_lap_axis(
            res["lap_list"][ok], res["lt_val"][ok], x_mode,
        )
        y_arr = res["mean_curvature"][ok][order]
        lap_ord = res["lap_list"][ok][order]
        color = DRIVER_COLORS[i % len(DRIVER_COLORS)]
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
        f"Steering smoothness: {summary['mean_steering_smoothness']:.3f} rad/s | "
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
    corner_curvature_fig(dfs).show()


if __name__ == "__main__":
    main()
