"""Shared utilities for CAT17x data analysis (Formula Student 4WD Electric)."""
from __future__ import annotations
from typing import Literal
import numpy as np
import polars as pl
import plotly.graph_objects as go

# ── Dark theme constants ──────────────────────────────────────────────────────
_BG   = '#141417'
_TEXT = '#EBEBEB'
_GRID = 'rgba(128,128,128,0.2)'
_AXIS = '#E5E5E5'

# Per-wheel colours: FL=blue, FR=orange, RL=green, RR=purple
WHEEL_COLORS  = {'FL': '#4DB3F2', 'FR': '#F28C40', 'RL': '#73D973', 'RR': '#D973D9'}
WHEEL_SYMBOLS = {'FL': 'circle',  'FR': 'square',  'RL': 'triangle-up', 'RR': 'diamond'}
PerLapAxisMode = Literal['laps', 'laptime']
COMPLETE_LAPS_MARKER = '__complete_laps_only'
LOGIC_START_TIME_BY_CSV = {
    'Abel_FSG.csv': 40.8651,
}


# ── Figure helpers ────────────────────────────────────────────────────────────

def make_dark_figure(title: str = '', xlabel: str = '', ylabel: str = '') -> go.Figure:
    """Return a Plotly Figure with dark motorsport styling."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=_TEXT)),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT, size=11),
        xaxis=dict(title=xlabel, color=_AXIS, gridcolor=_GRID,
                   linecolor=_AXIS, tickcolor=_AXIS, showgrid=True),
        yaxis=dict(title=ylabel, color=_AXIS, gridcolor=_GRID,
                   linecolor=_AXIS, tickcolor=_AXIS, showgrid=True),
        legend=dict(bgcolor='rgba(20,20,23,0.85)',
                    bordercolor='rgba(128,128,128,0.3)',
                    font=dict(color=_TEXT)),
    )
    return fig


def add_lap_scatter(fig: go.Figure, x: np.ndarray, y: np.ndarray,
                    lap_ids: np.ndarray, name: str = '',
                    color: str = '#4DB3F2', symbol: str = 'circle',
                    size: int = 10) -> None:
    """Add scatter trace with lap number labels."""
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers+text',
        name=name,
        marker=dict(color=color, symbol=symbol, size=size, line=dict(width=0)),
        text=[f'  {int(l)}' for l in lap_ids],
        textposition='middle right',
        textfont=dict(color=_TEXT, size=10),
    ))


def add_trend_line(fig: go.Figure, x: np.ndarray, y: np.ndarray,
                   color: str = '#F28C40', dash: str = 'dash') -> None:
    """Add a linear regression line to *fig*."""
    if len(x) < 2:
        return
    p     = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    fig.add_trace(go.Scatter(
        x=x_fit, y=np.polyval(p, x_fit),
        mode='lines', name='Trend',
        line=dict(color=color, dash=dash, width=1.6),
        showlegend=False,
    ))


def add_zero_line(fig: go.Figure, x: np.ndarray) -> None:
    """Add a horizontal dashed reference line at y=0."""
    fig.add_hline(y=0, line=dict(color='rgba(200,200,200,0.5)',
                                 dash='dash', width=1.2))


def per_lap_axis(
    lap_ids: np.ndarray,
    lap_times_s: np.ndarray,
    mode: PerLapAxisMode,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return sorted x-values, sort order, and axis label for per-lap plots."""
    laps_arr = np.asarray(lap_ids, dtype=float)
    laptime_arr = np.asarray(lap_times_s, dtype=float)

    if mode == 'laps':
        x = laps_arr
        xlabel = 'Lap'
    elif mode == 'laptime':
        x = laptime_arr
        xlabel = 'Lap time [s]'
    else:
        raise ValueError(f'Unsupported per-lap axis mode: {mode}')

    order = np.argsort(x, kind='mergesort')
    return x[order], order, xlabel


# ── Data helpers ──────────────────────────────────────────────────────────────

def keep_min_duration_segments(mask: np.ndarray,
                                min_duration: float,
                                dt: float) -> np.ndarray:
    """Remove boolean segments shorter than *min_duration* seconds.

    Args:
        mask:         Boolean event array.
        min_duration: Minimum segment duration [s].
        dt:           Sample interval [s].

    Returns:
        Filtered boolean array (same shape as *mask*).
    """
    clean = np.zeros(len(mask), dtype=bool)
    if not np.any(mask):
        return clean
    min_samples = max(1, int(np.ceil(min_duration / dt)))
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    d      = np.diff(padded.astype(np.int8))
    starts = np.where(d ==  1)[0]
    ends   = np.where(d == -1)[0] - 1
    for s, e in zip(starts, ends):
        if e - s + 1 >= min_samples:
            clean[s:e + 1] = True
    return clean


def fill_short_false_gaps(
    mask: np.ndarray,
    max_gap_duration: float,
    dt: float,
) -> np.ndarray:
    """Fill false gaps shorter than *max_gap_duration* between true segments."""
    clean = mask.astype(bool).copy()
    if not np.any(clean):
        return clean

    max_gap_samples = max(1, int(np.ceil(max_gap_duration / dt)))
    padded = np.concatenate([[False], clean, [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1

    for prev_end, next_start in zip(ends[:-1], starts[1:]):
        gap_start = prev_end + 1
        gap_end = next_start - 1
        if gap_end >= gap_start and (gap_end - gap_start + 1) <= max_gap_samples:
            clean[gap_start:gap_end + 1] = True
    return clean


def _laps_filter_applied_dict(data: dict[str, np.ndarray]) -> bool:
    """Return True when *data* already carries an explicit lap selection."""
    marker = data.get(COMPLETE_LAPS_MARKER)
    return marker is not None and np.any(np.isfinite(marker) & (marker > 0.0))


def _laps_filter_applied_df(df: pl.DataFrame) -> bool:
    """Return True when *df* is already restricted to the desired laps."""
    return COMPLETE_LAPS_MARKER in df.columns


def ensure_detected_laps_df(df: pl.DataFrame) -> pl.DataFrame:
    """Validate that *df* contains detected lap IDs."""
    if "laps" not in df.columns or "laptime" not in df.columns:
        raise KeyError("CSV must contain `laps` and `laptime` columns.")

    laps = df["laps"].to_numpy().astype(float)
    if not np.any(np.isfinite(laps)):
        raise ValueError("No valid laps — run lap detection first.")
    return df


def apply_logic_start_time(df: pl.DataFrame, path: str | None = None) -> pl.DataFrame:
    """Apply per-CSV logic-start overrides before dashboard analysis."""
    if path is None or "TimeStamp" not in df.columns:
        return df

    csv_name = path.rsplit("/", 1)[-1]
    start_time_s = LOGIC_START_TIME_BY_CSV.get(csv_name)
    if start_time_s is None:
        return df

    out = df.filter(
        pl.col("TimeStamp").is_finite() & (pl.col("TimeStamp") >= float(start_time_s))
    )
    if out.is_empty():
        raise ValueError(
            f"{csv_name}: no samples remain after applying logic start at {start_time_s:.4f} s."
        )
    return out


def apply_special_lap_logic(df: pl.DataFrame, path: str | None = None) -> pl.DataFrame:
    """Apply one-off lap relabelling rules for specific CSVs."""
    if path is None:
        return df

    csv_name = path.rsplit("/", 1)[-1]
    if csv_name != 'Abel_FSG.csv':
        return df
    if "TimeStamp" not in df.columns or "laps" not in df.columns:
        return df

    time_s = df["TimeStamp"].to_numpy().astype(float)
    old_laps = df["laps"].to_numpy().astype(float)
    valid = np.isfinite(time_s) & np.isfinite(old_laps)
    if not np.any(valid):
        return df

    time_valid = time_s[valid]
    laps_valid = old_laps[valid]
    starts = np.concatenate([[0], np.where(np.diff(laps_valid) != 0.0)[0] + 1])
    if len(starts) == 0:
        return df

    new_laps_valid = np.full(len(time_valid), np.nan)
    new_laptime_valid = np.full(len(time_valid), np.nan)
    for lap_idx, start in enumerate(starts):
        end = int(starts[lap_idx + 1]) if lap_idx + 1 < len(starts) else len(time_valid)
        lap_time_s = float(time_valid[end - 1] - time_valid[start])
        new_laps_valid[start:end] = float(lap_idx + 1)
        new_laptime_valid[start:end] = lap_time_s

    new_laps = np.full(len(df), np.nan)
    new_laptime = np.full(len(df), np.nan)
    new_laps[valid] = new_laps_valid
    new_laptime[valid] = new_laptime_valid

    return df.with_columns([
        pl.Series("laps", new_laps),
        pl.Series("laptime", new_laptime),
    ])


def exclude_lap0_and_last_lap(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Filter out formation lap (laps <= 0) and the last (incomplete) lap.

    If *data* already carries the dashboard lap-selection marker, it is returned
    unchanged so downstream modules respect the user-selected laps.

    *data* must contain key ``'laps'``.
    Raises ``ValueError`` if fewer than 2 valid laps remain.
    """
    if _laps_filter_applied_dict(data):
        all_laps = np.unique(data['laps'][np.isfinite(data['laps'])])
        if len(all_laps) < 1:
            raise ValueError(
                'Not enough valid selected laps. Run lapcount.py first.'
            )
        return data

    laps  = data['laps']
    valid = laps > 0
    filt  = {k: v[valid] for k, v in data.items()}

    all_laps = np.unique(filt['laps'][np.isfinite(filt['laps'])])
    if len(all_laps) < 2:
        raise ValueError(
            'Not enough valid laps after excluding lap 0. '
            'Run lapcount.py first.'
        )
    keep = filt['laps'] != all_laps.max()
    return {k: v[keep] for k, v in filt.items()}


def ensure_complete_laps_df(df: pl.DataFrame) -> pl.DataFrame:
    """Return a DataFrame restricted to laps > 0 and excluding the last lap."""
    if _laps_filter_applied_df(df):
        return df

    df = ensure_detected_laps_df(df)

    out = df.filter((pl.col("laps") > 0) & pl.col("laptime").is_not_nan())
    if out.is_empty():
        raise ValueError("No valid laps — run lap detection first.")

    max_lap = out["laps"].max()
    if max_lap is not None:
        out = out.filter(pl.col("laps") < max_lap)
    if out.is_empty():
        raise ValueError("Only one lap detected — need at least 2 valid laps.")

    return out.with_columns(
        pl.Series(COMPLETE_LAPS_MARKER, np.ones(len(out), dtype=float)),
    )


def robust_dt(time: np.ndarray) -> float:
    """Return median sample interval [s], ignoring gaps and NaNs."""
    diffs = np.diff(time)
    valid = diffs[(diffs > 0) & np.isfinite(diffs)]
    if len(valid) == 0:
        raise ValueError('Cannot compute dt: no positive time step found.')
    return float(np.median(valid))


def unique_laps(laps: np.ndarray) -> np.ndarray:
    """Sorted unique lap IDs, NaN excluded."""
    u = np.unique(laps)
    return u[np.isfinite(u)]


def available_laps(df: pl.DataFrame) -> np.ndarray:
    """Sorted detected lap IDs available for dashboard selection.

    Lap 0 is excluded from the UI because it is the formation lap.
    """
    df = ensure_detected_laps_df(df)
    laps = unique_laps(df["laps"].to_numpy().astype(float))
    return laps[laps > 0].astype(int)


def select_laps_df(df: pl.DataFrame, lap_ids: list[int] | np.ndarray) -> pl.DataFrame:
    """Return *df* restricted to the requested lap IDs.

    The returned DataFrame carries the shared lap-selection marker so every
    analysis module uses exactly these laps without applying its own fallback
    filtering. Lap 0 is always excluded because it is the formation lap.
    """
    df = ensure_detected_laps_df(df)
    lap_arr = np.unique(np.asarray(lap_ids, dtype=int))
    lap_arr = lap_arr[lap_arr > 0]
    if lap_arr.size == 0:
        raise ValueError("Select at least one lap above 0.")
    if "TimeStamp" not in df.columns:
        raise KeyError("CSV must contain `TimeStamp` to filter laps.")

    out = df.filter(
        pl.col("laps").is_finite() & pl.col("laps").is_in(lap_arr.astype(float).tolist())
    )
    if out.is_empty():
        raise ValueError("Selected laps have no telemetry samples.")

    per_lap = out.group_by("laps").agg([
        (
            pl.when(pl.col("laptime").is_finite())
            .then(pl.col("laptime"))
            .otherwise(None)
            .max()
            .alias("__lap_laptime")
        ),
        (
            (pl.col("TimeStamp").max() - pl.col("TimeStamp").min())
            .alias("__lap_time_from_samples")
        ),
    ])

    return (
        out.join(per_lap, on="laps", how="left")
        .with_columns([
            pl.coalesce([
                pl.col("__lap_laptime"),
                pl.when(
                    pl.col("__lap_time_from_samples").is_finite()
                    & (pl.col("__lap_time_from_samples") > 0.0)
                )
                .then(pl.col("__lap_time_from_samples"))
                .otherwise(None),
            ]).alias("laptime"),
            pl.lit(1.0).alias(COMPLETE_LAPS_MARKER),
        ])
        .drop(["__lap_laptime", "__lap_time_from_samples"])
    )


def lap_dist_from_gps(df: pl.DataFrame) -> np.ndarray:
    """Haversine cumulative distance [m] per lap, reset to 0 at each lap start.

    Requires columns: VN_latitude, VN_longitude, laps.
    Returns an array of the same length as *df* (zeros where GPS is unavailable).
    """
    gps_cols = ("VN_latitude", "VN_longitude", "laps")
    if any(c not in df.columns for c in gps_cols):
        return np.zeros(len(df))

    lat  = df["VN_latitude"].to_numpy().astype(float)
    lng  = df["VN_longitude"].to_numpy().astype(float)
    laps = df["laps"].to_numpy().astype(float)

    R    = 6_371_000.0
    dist = np.zeros(len(lat))
    for lap_id in np.unique(laps[np.isfinite(laps)]):
        idx = np.where(laps == lap_id)[0]
        if len(idx) < 2:
            continue
        lat_r = np.radians(lat[idx])
        lng_r = np.radians(lng[idx])
        dlat  = np.diff(lat_r)
        dlng  = np.diff(lng_r)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(lat_r[:-1]) * np.cos(lat_r[1:]) * np.sin(dlng / 2) ** 2)
        inc = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        dist[idx] = np.concatenate([[0.0], np.cumsum(inc)])
    return dist


def smooth_signal(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """NaN-aware moving average with edge-preserving normalisation."""
    arr = np.asarray(signal, dtype=float)
    if window_samples <= 1 or len(arr) == 0:
        return arr.copy()

    finite = np.isfinite(arr)
    kernel = np.ones(int(window_samples), dtype=float)
    num = np.convolve(np.where(finite, arr, 0.0), kernel, mode="same")
    den = np.convolve(finite.astype(float), kernel, mode="same")

    out = arr.copy()
    ok = den > 0.0
    out[ok] = num[ok] / den[ok]
    return out


def _phase_masks_from_signals(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """Sample-wise phase masks from telemetry, before lap-to-lap stabilisation."""
    ay_col = "Filtering_VN_ay" if "Filtering_VN_ay" in df.columns else "VN_ay"
    ax_col = "Filtering_VN_ax" if "Filtering_VN_ax" in df.columns else "VN_ax"
    n = len(df)
    if any(c not in df.columns for c in ("Brake", "Steering", ay_col)):
        return {
            "BRAKE": np.zeros(n, dtype=bool),
            "CORNER": np.zeros(n, dtype=bool),
            "STRAIGHT": np.ones(n, dtype=bool),
        }

    brake = df["Brake"].to_numpy().astype(float)
    steering = df["Steering"].to_numpy().astype(float)
    ay = df[ay_col].to_numpy().astype(float)
    ax = df[ax_col].to_numpy().astype(float) if ax_col in df.columns else np.zeros(n)
    throttle = (
        df["Throttle"].to_numpy().astype(float)
        if "Throttle" in df.columns
        else np.zeros(n)
    )

    if "TimeStamp" in df.columns:
        time_s = df["TimeStamp"].to_numpy().astype(float)
        dt = robust_dt(time_s)
    elif "dt_s" in df.columns:
        dt_vals = df["dt_s"].to_numpy().astype(float)
        dt_valid = dt_vals[np.isfinite(dt_vals) & (dt_vals > 0.0)]
        dt = float(np.median(dt_valid)) if len(dt_valid) > 0 else 0.01
    else:
        dt = 0.01

    smooth_samples = max(1, int(round(0.10 / dt)))
    brake_sm = smooth_signal(brake, smooth_samples)
    steer_abs_sm = np.abs(smooth_signal(steering, smooth_samples))
    ay_abs_sm = np.abs(smooth_signal(ay, smooth_samples))
    ax_sm = smooth_signal(ax, smooth_samples)
    throttle_sm = smooth_signal(throttle, smooth_samples)

    if "laps" in df.columns:
        laps = df["laps"].to_numpy().astype(float)
        if _laps_filter_applied_df(df):
            valid_laps_mask = np.isfinite(laps)
        else:
            all_laps = np.unique(laps[np.isfinite(laps)])
            valid_laps_mask = (laps > 0) & (laps != all_laps.max() if len(all_laps) > 0 else True)
    else:
        valid_laps_mask = np.ones(n, dtype=bool)

    valid_dyn = valid_laps_mask & np.isfinite(ay_abs_sm) & np.isfinite(steer_abs_sm)
    ay_valid = ay_abs_sm[valid_dyn]
    steer_valid = steer_abs_sm[valid_dyn]

    ay_thr = float(np.percentile(ay_valid, 65)) if len(ay_valid) > 0 else 2.0
    ay_thr = max(ay_thr, 1.5)
    steer_thr = float(np.percentile(steer_valid, 60)) if len(steer_valid) > 0 else 0.08
    steer_thr = max(steer_thr, 0.06)

    # Exit acceleration with only residual steering should read as straight, not corner.
    power_out_raw = (
        np.isfinite(ax_sm)
        & np.isfinite(throttle_sm)
        & np.isfinite(steer_abs_sm)
        & np.isfinite(ay_abs_sm)
        & (ax_sm > 1.2)
        & (throttle_sm > 40.0)
        & (steer_abs_sm < max(0.12, 1.50 * steer_thr))
        & (ay_abs_sm < 1.05 * ay_thr)
    )

    corner_raw = (
        np.isfinite(ay_abs_sm)
        & np.isfinite(steer_abs_sm)
        & (ay_abs_sm >= ay_thr)
        & (steer_abs_sm >= steer_thr)
        & ~power_out_raw
    )
    corner_m = keep_min_duration_segments(corner_raw, min_duration=0.18, dt=dt)
    corner_m = fill_short_false_gaps(corner_m, max_gap_duration=0.12, dt=dt)

    brake_raw = np.isfinite(brake_sm) & (brake_sm > 5.0)
    brake_m = keep_min_duration_segments(brake_raw, min_duration=0.12, dt=dt)
    brake_m = fill_short_false_gaps(brake_m, max_gap_duration=0.08, dt=dt)
    brake_m &= ~corner_m

    straight_m = ~(brake_m | corner_m)
    return {"BRAKE": brake_m, "CORNER": corner_m, "STRAIGHT": straight_m}


def _stabilise_phase_masks_by_progress(
    df: pl.DataFrame,
    provisional: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Make track phases stable across laps using distance-normalised voting."""
    required = ("laps", "VN_latitude", "VN_longitude")
    n = len(df)
    if any(c not in df.columns for c in required):
        return provisional

    laps = df["laps"].to_numpy().astype(float)
    valid_laps = unique_laps(laps)
    if len(valid_laps) < 2:
        return provisional

    dist_m = lap_dist_from_gps(df)
    phase_order = ("BRAKE", "CORNER", "STRAIGHT")
    n_bins = 240

    used_laps = 0
    vote_counts = np.zeros((len(phase_order), n_bins), dtype=int)
    mapped_mask = np.zeros(n, dtype=bool)
    progress_bins = np.full(n, -1, dtype=int)

    for lap_id in valid_laps:
        idx = np.where(laps == lap_id)[0]
        if len(idx) < 20:
            continue

        lap_dist = dist_m[idx]
        lap_len = float(np.nanmax(lap_dist))
        if not np.isfinite(lap_len) or lap_len < 10.0:
            continue

        progress = np.clip(lap_dist / lap_len, 0.0, 1.0)
        bin_idx = np.clip((progress * (n_bins - 1)).astype(int), 0, n_bins - 1)

        per_lap_counts = np.zeros((len(phase_order), n_bins), dtype=int)
        occupied = np.zeros(n_bins, dtype=bool)
        occupied[np.unique(bin_idx)] = True
        for phase_i, phase_name in enumerate(phase_order):
            np.add.at(per_lap_counts[phase_i], bin_idx, provisional[phase_name][idx].astype(int))

        dominant = np.full(n_bins, phase_order.index("STRAIGHT"), dtype=int)
        if occupied.any():
            dominant[occupied] = np.argmax(per_lap_counts[:, occupied], axis=0)

        for phase_i in range(len(phase_order)):
            vote_counts[phase_i] += dominant == phase_i

        progress_bins[idx] = bin_idx
        mapped_mask[idx] = True
        used_laps += 1

    if used_laps < 2:
        return provisional

    vote_frac = vote_counts.astype(float) / float(used_laps)

    def _circular_smooth(arr: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return arr.copy()
        pad = window // 2
        kernel = np.ones(window, dtype=float) / float(window)
        ext = np.concatenate([arr[-pad:], arr, arr[:pad]])
        return np.convolve(ext, kernel, mode="valid")

    vote_frac_sm = np.vstack([
        _circular_smooth(vote_frac[phase_i], window=5)
        for phase_i in range(len(phase_order))
    ])
    bin_dt = 1.0 / n_bins

    corner_bins = vote_frac_sm[phase_order.index("CORNER")] >= 0.25
    corner_bins = keep_min_duration_segments(corner_bins, min_duration=0.015, dt=bin_dt)
    corner_bins = fill_short_false_gaps(corner_bins, max_gap_duration=0.008, dt=bin_dt)

    brake_bins = vote_frac_sm[phase_order.index("BRAKE")] >= 0.45
    brake_bins = keep_min_duration_segments(brake_bins, min_duration=0.010, dt=bin_dt)
    brake_bins = fill_short_false_gaps(brake_bins, max_gap_duration=0.006, dt=bin_dt)
    brake_bins &= ~corner_bins

    stable = {
        "BRAKE": provisional["BRAKE"].copy(),
        "CORNER": provisional["CORNER"].copy(),
        "STRAIGHT": provisional["STRAIGHT"].copy(),
    }
    valid_rows = mapped_mask & (progress_bins >= 0)
    stable["BRAKE"][valid_rows] = brake_bins[progress_bins[valid_rows]]
    stable["CORNER"][valid_rows] = corner_bins[progress_bins[valid_rows]]
    stable["STRAIGHT"][valid_rows] = ~(
        stable["BRAKE"][valid_rows] | stable["CORNER"][valid_rows]
    )
    return stable


def phase_masks_for_map(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """Adaptive phase masks for track-map visualisation.

    Thresholds are derived from the run's own data distribution:
      - BRAKE:    brake pedal > 5 % after short smoothing / segment cleanup
      - CORNER:   smoothed |ay| and |steering| with adaptive thresholds
      - STRAIGHT: everything else

    Priority: CORNER > BRAKE > STRAIGHT (mutually exclusive).
    The masks are deliberately smoothed to avoid point-by-point colour flicker
    on the GPS map.

    Returns:
        Dict with keys "BRAKE", "CORNER", "STRAIGHT" (boolean arrays, same length as *df*).
    """
    provisional = _phase_masks_from_signals(df)
    return _stabilise_phase_masks_by_progress(df, provisional)


def load_data(path: str, complete_laps_only: bool = True) -> pl.DataFrame:
    """Load a telemetry CSV and add sample time `dt_s`.

    When *complete_laps_only* is True, lap 0 and the last lap are excluded.
    When False, the full CSV is returned and the dashboard can choose laps later.
    """
    df = apply_logic_start_time(pl.read_csv(path), path)
    df = apply_special_lap_logic(df, path)
    df = ensure_detected_laps_df(df)
    if complete_laps_only:
        df = ensure_complete_laps_df(df)

    if "TimeStamp" not in df.columns:
        raise KeyError("CSV must contain `TimeStamp` to compute `dt_s`.")

    time_s = df["TimeStamp"].to_numpy().astype(float)
    dt_s = np.diff(time_s, prepend=np.nan)
    valid_dt = dt_s[np.isfinite(dt_s) & (dt_s > 0.0)]
    fill_dt = float(np.median(valid_dt)) if len(valid_dt) > 0 else np.nan
    bad_mask = ~np.isfinite(dt_s) | (dt_s <= 0.0)
    if np.isfinite(fill_dt):
        dt_s[bad_mask] = fill_dt

    return df.with_columns(pl.Series("dt_s", dt_s))
