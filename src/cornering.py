"""Corner entry / apex / exit analysis for driver telemetry."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter

import src.dynamics as dyn
from utils import (
    ensure_complete_laps_df,
    keep_min_duration_segments,
    lap_dist_from_gps,
    make_dark_figure,
    robust_dt,
    smooth_signal,
    unique_laps,
    cols_to_numpy,
)

R_CAP_M = 1.0e4
AY_EPS_MPS2 = 0.05
SMOOTH_WINDOW_S = 0.30
BRAKE_ON_PCT = 5.0
BRAKE_LOOKBACK_M = 120.0
G_MPS2 = 9.81
MIN_CORNER_SPEED_MPS = 2.5
MIN_PEAK_AY_G = 0.18
MIN_CORNER_LENGTH_M = 12.0
R_RELEASE_FACTOR = 1.35
SIGN_SPLIT_AY_G = 0.08
SIGN_SPLIT_MIN_PARENT_LENGTH_M = 70.0
SIGN_SPLIT_MIN_PART_LENGTH_M = 25.0
SIGN_SPLIT_MIN_PART_PEAK_AY_G = 0.35

# Same-sign chicane split: a single curvature seed segment is partitioned when
# two distinct peaks of the same direction are separated by a valley whose
# minimum drops below VALLEY_REL_FACTOR of the smaller peak.
SAME_SIGN_SPLIT_MIN_PARENT_LENGTH_M = 80.0
SAME_SIGN_SPLIT_MIN_PART_LENGTH_M = 22.0
SAME_SIGN_SPLIT_MIN_PEAK_RATIO = 1.20  # peaks must be at least this ratio over the valley
SAME_SIGN_SPLIT_VALLEY_REL_TO_PEAK = 0.55  # valley must drop at least to this fraction of smaller peak
SAME_SIGN_SPLIT_MIN_PEAK_SEPARATION_M = 18.0

_LAT_ACCEL_CANDIDATES = ("Filtering_VN_ay",)
_SPEED_CANDIDATES = ("VN_vx",)
_TURN_PALETTE = (
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
    "#2D9CDB",
    "#F2C94C",
)
_COMPARISON_TRACE_PALETTE = (
    "#4DB3F2",
    "#F28C40",
    "#73D973",
    "#D973D9",
    "#F2C94C",
    "#56CCF2",
    "#EB5757",
    "#9B51E0",
)


@dataclass(frozen=True)
class TurnDef:
    turn_id: int
    s_entry_m: float
    s_apex_m: float
    s_exit_m: float
    apex_lat: float
    apex_lng: float
    lat: np.ndarray
    lng: np.ndarray


@dataclass(frozen=True)
class CornerPhases:
    """Distance bounds for the 4 phases of a corner: brake / entry / apex / exit.

    Phases chain end-to-end so every metre belongs to one phase of one corner:
        Braking_n     [s_brake_on, s_brake_off]
        Entry_n       [s_brake_off, s_apex_lo]
        Apex_n        [s_apex_lo, s_apex_hi]   (window centred on r_max)
        Exit_n        [s_apex_hi, next Braking_(n+1).start  or  s_lap_end]
    """

    turn_id: int
    s_apex_m: float
    s_brake_on_m: float
    s_brake_off_m: float
    s_apex_lo_m: float
    s_apex_hi_m: float
    s_exit_end_m: float
    has_braking: bool


def compute_corner_phases(
    turns: list[TurnDef],
    s_m: np.ndarray,
    brake_pct: np.ndarray,
    *,
    apex_half_window_m: float = 5.0,
    brake_threshold_pct: float = 5.0,
    lap_start_m: float | None = None,
    lap_end_m: float | None = None,
) -> list[CornerPhases]:
    """Compute chained 4-phase boundaries per corner using brake input + apex.

    Apex (`s_apex_m` of the TurnDef) is the maximum-curvature point. Braking is
    located by walking the brake signal from the previous apex up to the next
    one and taking the contiguous brake-on region whose end (brake-off) is
    closest to the apex; this absorbs trail-braking into the Braking phase
    boundary. The pre-T1 stretch is folded into Braking_1 so no metre is
    orphaned.
    """
    if not turns:
        return []
    s_arr = np.asarray(s_m, dtype=float)
    brake_arr = np.asarray(brake_pct, dtype=float)
    if s_arr.size == 0 or brake_arr.shape != s_arr.shape:
        return []
    s0 = float(s_arr[0]) if lap_start_m is None else float(lap_start_m)
    s_end = float(s_arr[-1]) if lap_end_m is None else float(lap_end_m)
    sorted_turns = sorted(turns, key=lambda t: float(t.s_apex_m))

    brake_on_list: list[float] = []
    brake_off_list: list[float] = []
    has_brake_list: list[bool] = []

    for idx, turn in enumerate(sorted_turns):
        s_apex = float(turn.s_apex_m)
        s_apex_lo = s_apex - apex_half_window_m
        if idx == 0:
            search_lo = s0
        else:
            search_lo = float(sorted_turns[idx - 1].s_apex_m) + apex_half_window_m
        search_hi = s_apex
        win_mask = (s_arr >= search_lo) & (s_arr <= search_hi)
        brake_mask = win_mask & (brake_arr >= brake_threshold_pct)
        if not np.any(brake_mask):
            s_brake_off = s_apex_lo
            s_brake_on = s_apex_lo
            has_brake = False
        else:
            indices = np.where(brake_mask)[0]
            i_off = int(indices[-1])
            s_brake_off = float(s_arr[i_off])
            j = i_off
            while (
                j > 0
                and brake_arr[j - 1] >= brake_threshold_pct
                and s_arr[j - 1] >= search_lo
            ):
                j -= 1
            s_brake_on = float(s_arr[j])
            has_brake = True
        s_brake_off = min(s_brake_off, s_apex_lo)
        s_brake_on = max(min(s_brake_on, s_brake_off), search_lo)
        brake_on_list.append(s_brake_on)
        brake_off_list.append(s_brake_off)
        has_brake_list.append(has_brake)

    phases: list[CornerPhases] = []
    n = len(sorted_turns)
    for idx, turn in enumerate(sorted_turns):
        s_apex = float(turn.s_apex_m)
        s_apex_lo = s_apex - apex_half_window_m
        s_apex_hi = s_apex + apex_half_window_m
        s_brake_off = brake_off_list[idx]
        s_brake_on = s0 if idx == 0 else brake_on_list[idx]
        if idx < n - 1:
            s_exit_end = brake_on_list[idx + 1]
        else:
            s_exit_end = s_end
        s_exit_end = max(s_exit_end, s_apex_hi)
        phases.append(
            CornerPhases(
                turn_id=int(turn.turn_id),
                s_apex_m=s_apex,
                s_brake_on_m=s_brake_on,
                s_brake_off_m=s_brake_off,
                s_apex_lo_m=s_apex_lo,
                s_apex_hi_m=s_apex_hi,
                s_exit_end_m=s_exit_end,
                has_braking=bool(has_brake_list[idx]),
            )
        )
    return phases


def _lateral_accel_col(df: pl.DataFrame) -> str:
    for col in _LAT_ACCEL_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(
        "Missing lateral acceleration column. Expected one of "
        f"{list(_LAT_ACCEL_CANDIDATES)}."
    )


def _speed_col(df: pl.DataFrame) -> str:
    for col in _SPEED_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(
        "Missing speed column for cornering radius. Expected one of "
        f"{list(_SPEED_CANDIDATES)}."
    )


def _empty_metrics_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "run": pl.Utf8,
            "lap": pl.Int64,
            "lap_time_s": pl.Float64,
            "turn_id": pl.Int64,
            "v_entry_mps": pl.Float64,
            "v_apex_mps": pl.Float64,
            "v_exit_mps": pl.Float64,
            "R_min_m": pl.Float64,
            "ay_max_g": pl.Float64,
            "brake_to_apex_dist_m": pl.Float64,
            "apex_pos_pct": pl.Float64,
            "corner_length_m": pl.Float64,
        }
    )


def _finite_lap_time(laptime: np.ndarray, mask: np.ndarray) -> float:
    vals = laptime[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.max(vals)) if vals.size else np.nan


def _segment_bounds(mask: np.ndarray) -> list[tuple[int, int]]:
    if len(mask) == 0 or not np.any(mask):
        return []
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _fill_short_dist_gaps(
    mask: np.ndarray,
    s_lap_m: np.ndarray,
    signed_lat_accel_mps2: np.ndarray | None = None,
    merge_gap_m: float = 8.0,
) -> np.ndarray:
    clean = mask.astype(bool).copy()
    if merge_gap_m <= 0.0 or not np.any(clean):
        return clean

    segments = _segment_bounds(clean)
    for (_, prev_end), (next_start, _) in zip(segments[:-1], segments[1:]):
        gap_start = prev_end + 1
        gap_end = next_start - 1
        if gap_end < gap_start:
            continue
        if signed_lat_accel_mps2 is not None:
            prev_sign = _dominant_turn_sign(
                signed_lat_accel_mps2, max(0, prev_end - 3), prev_end
            )
            next_sign = _dominant_turn_sign(
                signed_lat_accel_mps2, next_start, min(len(clean) - 1, next_start + 3)
            )
            if prev_sign != 0 and next_sign != 0 and prev_sign != next_sign:
                continue
        gap_m = float(s_lap_m[next_start] - s_lap_m[prev_end])
        if np.isfinite(gap_m) and gap_m >= 0.0 and gap_m < merge_gap_m:
            clean[gap_start:gap_end + 1] = True
    return clean


def _dominant_turn_sign(values: np.ndarray, start: int, end: int) -> int:
    """Return the sustained lateral direction in a segment: -1, 0, or +1."""
    if end < start:
        return 0
    seg = values[start:end + 1]
    finite = np.isfinite(seg) & (np.abs(seg) >= SIGN_SPLIT_AY_G * G_MPS2)
    if not np.any(finite):
        return 0
    total = float(np.nansum(seg[finite]))
    if total > 0.0:
        return 1
    if total < 0.0:
        return -1
    return 0


def _filled_direction_sign(values: np.ndarray, start: int, end: int) -> np.ndarray:
    seg = values[start:end + 1]
    raw = np.where(
        np.isfinite(seg) & (np.abs(seg) >= SIGN_SPLIT_AY_G * G_MPS2),
        np.sign(seg),
        0.0,
    ).astype(int)
    if not np.any(raw != 0):
        return raw

    filled = raw.copy()
    last = 0
    for i, value in enumerate(filled):
        if value != 0:
            last = int(value)
        elif last != 0:
            filled[i] = last

    next_value = 0
    for i in range(len(filled) - 1, -1, -1):
        if filled[i] != 0:
            next_value = int(filled[i])
        elif next_value != 0:
            filled[i] = next_value
    return filled


def _split_segment_by_direction(
    start: int,
    end: int,
    signed_lat_accel_mps2: np.ndarray,
    s_lap_m: np.ndarray,
    dt_s: float,
    min_dur_s: float,
) -> list[tuple[int, int]]:
    """Split a continuous curvature zone when ay direction changes."""
    parent_length_m = float(s_lap_m[end] - s_lap_m[start])
    if (
        not np.isfinite(parent_length_m)
        or parent_length_m < SIGN_SPLIT_MIN_PARENT_LENGTH_M
    ):
        return [(start, end)]

    signs = _filled_direction_sign(signed_lat_accel_mps2, start, end)
    if len(signs) == 0 or not np.any(signs != 0):
        return [(start, end)]

    min_samples = max(3, int(np.ceil(min(0.20, min_dur_s * 0.5) / dt_s)))
    changes = np.where(np.diff(signs) != 0)[0] + 1
    if len(changes) == 0:
        return [(start, end)]

    bounds = [0, *changes.tolist(), len(signs)]
    parts: list[tuple[int, int]] = []
    for local_start, local_end_excl in zip(bounds[:-1], bounds[1:]):
        if local_end_excl - local_start < min_samples:
            continue
        part_start = start + local_start
        part_end = start + local_end_excl - 1
        part_length_m = float(s_lap_m[part_end] - s_lap_m[part_start])
        part_peak_ay = float(
            np.nanmax(np.abs(signed_lat_accel_mps2[part_start:part_end + 1]))
        )
        if (
            np.isfinite(part_length_m)
            and part_length_m >= SIGN_SPLIT_MIN_PART_LENGTH_M
            and np.isfinite(part_peak_ay)
            and part_peak_ay >= SIGN_SPLIT_MIN_PART_PEAK_AY_G * G_MPS2
        ):
            parts.append((part_start, part_end))
    return parts if len(parts) >= 2 else [(start, end)]


def _expand_seeds_to_support(seed: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Keep support segments that contain at least one stronger seed sample."""
    out = np.zeros(len(seed), dtype=bool)
    for start, end in _segment_bounds(support):
        if np.any(seed[start:end + 1]):
            out[start:end + 1] = True
    return out


def _peak_preserving_smooth(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """Savitzky-Golay smoothing that preserves curvature peaks.

    Falls back to the moving-average implementation when the signal is too
    short or contains too many NaNs for the SG filter to converge.
    """
    arr = np.asarray(signal, dtype=float)
    if window_samples <= 2 or len(arr) < 5:
        return smooth_signal(arr, max(1, int(window_samples)))
    win = int(window_samples)
    if win % 2 == 0:
        win += 1
    win = min(win, len(arr) - 1)
    if win < 5 or win % 2 == 0:
        return smooth_signal(arr, max(1, int(window_samples)))
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return arr.copy()
    if not np.all(finite_mask):
        # SG cannot handle NaN; fall back for safety on this signal.
        return smooth_signal(arr, int(window_samples))
    try:
        return savgol_filter(arr, window_length=win, polyorder=2, mode="interp")
    except Exception:
        return smooth_signal(arr, int(window_samples))


def _parabolic_subsample_offset(prev_v: float, cur_v: float, next_v: float) -> float:
    """Sub-sample offset of the local maximum from a 3-point parabolic fit.

    Returns a value in [-0.5, 0.5]; clipped to that range to avoid outliers
    when the parabola opens upwards (denominator near zero).
    """
    if not (np.isfinite(prev_v) and np.isfinite(cur_v) and np.isfinite(next_v)):
        return 0.0
    denom = prev_v - 2.0 * cur_v + next_v
    if abs(denom) < 1.0e-12:
        return 0.0
    offset = 0.5 * (prev_v - next_v) / denom
    if not np.isfinite(offset):
        return 0.0
    return float(np.clip(offset, -0.5, 0.5))


def _refine_apex(
    curvature: np.ndarray,
    s_lap_m: np.ndarray,
    lat: np.ndarray,
    lng: np.ndarray,
    apex_idx: int,
) -> tuple[float, float, float]:
    """Return (s_apex_m, apex_lat, apex_lng) refined to sub-sample precision."""
    n = len(curvature)
    if apex_idx <= 0 or apex_idx >= n - 1:
        return float(s_lap_m[apex_idx]), float(lat[apex_idx]), float(lng[apex_idx])
    offset = _parabolic_subsample_offset(
        float(curvature[apex_idx - 1]),
        float(curvature[apex_idx]),
        float(curvature[apex_idx + 1]),
    )
    if offset >= 0.0:
        nbr = apex_idx + 1
    else:
        nbr = apex_idx - 1
        offset = -offset
    s_apex = float(s_lap_m[apex_idx]) + offset * float(s_lap_m[nbr] - s_lap_m[apex_idx])
    apex_lat = float(lat[apex_idx]) + offset * float(lat[nbr] - lat[apex_idx])
    apex_lng = float(lng[apex_idx]) + offset * float(lng[nbr] - lng[apex_idx])
    return s_apex, apex_lat, apex_lng


def _split_segment_same_sign(
    start: int,
    end: int,
    curvature_smooth: np.ndarray,
    s_lap_m: np.ndarray,
) -> list[tuple[int, int]]:
    """Split one same-sign curvature seed when it contains two distinct peaks.

    Detects esses-like sequences where the valley between two peaks drops
    deep enough to make them physically distinct corners even though ay never
    changes sign.
    """
    parent_length_m = float(s_lap_m[end] - s_lap_m[start])
    if (
        not np.isfinite(parent_length_m)
        or parent_length_m < SAME_SIGN_SPLIT_MIN_PARENT_LENGTH_M
    ):
        return [(start, end)]

    seg = np.asarray(curvature_smooth[start:end + 1], dtype=float)
    finite = np.isfinite(seg)
    if not np.any(finite):
        return [(start, end)]
    sig = np.where(finite, seg, 0.0)
    if len(sig) < 9:
        return [(start, end)]

    # Find interior local maxima.
    is_peak = (sig[1:-1] > sig[:-2]) & (sig[1:-1] > sig[2:])
    peak_local = np.where(is_peak)[0] + 1
    if len(peak_local) < 2:
        return [(start, end)]

    # Sort peaks by curvature value, keep the two strongest.
    peak_values = sig[peak_local]
    order = np.argsort(peak_values)[-2:]
    peak_local = np.sort(peak_local[order])
    p1, p2 = int(peak_local[0]), int(peak_local[1])
    sep_m = float(s_lap_m[start + p2] - s_lap_m[start + p1])
    if not np.isfinite(sep_m) or sep_m < SAME_SIGN_SPLIT_MIN_PEAK_SEPARATION_M:
        return [(start, end)]

    valley_local_offset = int(np.argmin(sig[p1:p2 + 1]))
    valley_local = p1 + valley_local_offset
    valley_value = float(sig[valley_local])
    smaller_peak = float(min(sig[p1], sig[p2]))
    if smaller_peak <= 0.0 or valley_value <= 0.0:
        return [(start, end)]
    if smaller_peak / max(valley_value, 1.0e-9) < SAME_SIGN_SPLIT_MIN_PEAK_RATIO:
        return [(start, end)]
    if valley_value > SAME_SIGN_SPLIT_VALLEY_REL_TO_PEAK * smaller_peak:
        return [(start, end)]

    cut_global = start + valley_local
    part1 = (start, cut_global)
    part2 = (cut_global + 1, end)
    parts: list[tuple[int, int]] = []
    for part_start, part_end in (part1, part2):
        if part_end <= part_start:
            continue
        part_length = float(s_lap_m[part_end] - s_lap_m[part_start])
        if not np.isfinite(part_length) or part_length < SAME_SIGN_SPLIT_MIN_PART_LENGTH_M:
            return [(start, end)]
        parts.append((part_start, part_end))
    if len(parts) != 2:
        return [(start, end)]

    split_parts: list[tuple[int, int]] = []
    for part_start, part_end in parts:
        split_parts.extend(
            _split_segment_same_sign(
                part_start,
                part_end,
                curvature_smooth,
                s_lap_m,
            )
        )
    return split_parts


def _lap_arrays(d: dict[str, np.ndarray], lap_id: int) -> dict[str, np.ndarray]:
    mask = d["laps"] == float(lap_id)
    return {key: value[mask] for key, value in d.items()}


def _metrics_entries(metrics: pl.DataFrame) -> list[tuple[str, int, float]]:
    if metrics.is_empty():
        return []
    rows = (
        metrics.select(["run", "lap", "lap_time_s"])
        .unique()
        .sort(["lap_time_s", "run", "lap"])
        .iter_rows()
    )
    return [(str(run), int(lap), float(lap_time)) for run, lap, lap_time in rows]


def _turn_ids(metrics: pl.DataFrame) -> list[int]:
    if metrics.is_empty():
        return []
    return (
        metrics.select("turn_id")
        .unique()
        .sort("turn_id")
        .get_column("turn_id")
        .to_list()
    )


def _series_by_turn(
    metrics: pl.DataFrame,
    run_name: str,
    lap_id: int,
    value_col: str,
    turn_ids: list[int],
) -> list[float]:
    sub = metrics.filter((pl.col("run") == run_name) & (pl.col("lap") == lap_id))
    by_turn = {
        int(tid): float(val)
        for tid, val in sub.select(["turn_id", value_col]).iter_rows()
    }
    return [by_turn.get(tid, np.nan) for tid in turn_ids]


def _mean_series_by_turn(
    metrics: pl.DataFrame,
    run_name: str,
    value_col: str,
    turn_ids: list[int],
) -> list[float]:
    sub = metrics.filter(pl.col("run") == run_name)
    if sub.is_empty():
        return [np.nan for _ in turn_ids]

    grouped = (
        sub.group_by("turn_id")
        .agg(pl.col(value_col).mean().alias("__mean_value"))
    )
    by_turn = {
        int(tid): float(val)
        for tid, val in grouped.select(["turn_id", "__mean_value"]).iter_rows()
    }
    return [by_turn.get(tid, np.nan) for tid in turn_ids]


def _fastest_lap_for_run(metrics: pl.DataFrame, run_name: str) -> int | None:
    sub = metrics.filter(pl.col("run") == run_name)
    if sub.is_empty():
        return None
    laps = (
        sub.select(["lap", "lap_time_s"])
        .unique()
        .sort(["lap_time_s", "lap"])
    )
    for lap_id, lap_time_s in laps.iter_rows():
        if np.isfinite(float(lap_time_s)):
            return int(lap_id)
    return int(laps.get_column("lap")[0]) if laps.height > 0 else None


def _comparison_trace_color(index: int) -> str:
    return _COMPARISON_TRACE_PALETTE[index % len(_COMPARISON_TRACE_PALETTE)]


def _run_split_positions(metrics: pl.DataFrame) -> tuple[dict[str, float], float]:
    """Return x positions that split each turn subplot by run."""
    run_names = [
        str(run)
        for run in metrics.select("run").unique().sort("run").get_column("run").to_list()
    ]
    if not run_names:
        return {}, 0.12
    if len(run_names) == 1:
        return {run_names[0]: 0.5}, 0.16

    centers = np.linspace(0.25, 0.75, len(run_names))
    jitter_width = min(0.12, 0.35 / max(len(run_names), 1))
    return {
        run_name: float(center)
        for run_name, center in zip(run_names, centers)
    }, float(jitter_width)


def _stable_unit_jitter(*parts: object) -> float:
    return (hash(parts) % 1000) / 1000.0 - 0.5


def _dark_subplot_layout(fig: go.Figure, title: str) -> go.Figure:
    base = make_dark_figure(title)
    fig.update_layout(
        title=base.layout.title,
        paper_bgcolor=base.layout.paper_bgcolor,
        plot_bgcolor=base.layout.plot_bgcolor,
        font=base.layout.font,
        legend=base.layout.legend,
    )
    fig.update_xaxes(
        color=base.layout.xaxis.color,
        gridcolor=base.layout.xaxis.gridcolor,
        linecolor=base.layout.xaxis.linecolor,
        tickcolor=base.layout.xaxis.tickcolor,
        showgrid=False,
    )
    fig.update_yaxes(
        color=base.layout.yaxis.color,
        gridcolor=base.layout.yaxis.gridcolor,
        linecolor=base.layout.yaxis.linecolor,
        tickcolor=base.layout.yaxis.tickcolor,
        showgrid=True,
    )
    return fig


def _lap_color_entries(dfs: dict[str, pl.DataFrame]) -> list[tuple[str, int, float]]:
    entries: list[tuple[str, int, float]] = []
    for run_name, df in dfs.items():
        d = compute_radius_curvature(df)
        for lap in unique_laps(d["laps"]):
            mask = d["laps"] == lap
            entries.append((run_name, int(lap), _finite_lap_time(d["laptime"], mask)))
    return entries


def compute_radius_curvature(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """Keys: time_s, laps, laptime, vx_mps, ay_mps2, brake_pct,
    lat, lng, s_lap_m, R_m, R_smooth_m, curvature_smooth_1pm, ay_abs_smooth_mps2,
    ay_smooth_mps2, signed_curvature_smooth_inv_m."""
    df = ensure_complete_laps_df(df)
    ay_col = _lateral_accel_col(df)
    speed_col = _speed_col(df)
    cols = [
        "TimeStamp",
        "laps",
        "laptime",
        "Brake",
        "VN_latitude",
        "VN_longitude",
        ay_col,
        speed_col,
    ]
    data = cols_to_numpy(df, cols)

    time_raw = data["TimeStamp"]
    time_s = time_raw - time_raw[0] if len(time_raw) else time_raw
    laps = data["laps"]
    laptime = data["laptime"]
    vx_mps = data[speed_col]
    ay_mps2 = data[ay_col]
    brake_pct = data["Brake"]
    lat = data["VN_latitude"]
    lng = data["VN_longitude"]
    s_lap_m = lap_dist_from_gps(df)

    if len(time_s) >= 2:
        dt_s = robust_dt(time_s)
    else:
        dt_s = 0.01
    win = max(1, int(round(SMOOTH_WINDOW_S / dt_s)))

    ay_smooth_mps2 = _peak_preserving_smooth(ay_mps2, win)
    ay_abs = np.abs(ay_mps2)
    R_m = np.divide(
        vx_mps ** 2,
        np.maximum(ay_abs, AY_EPS_MPS2),
        out=np.full_like(vx_mps, np.nan, dtype=float),
        where=np.isfinite(vx_mps) & np.isfinite(ay_abs),
    )
    R_m = np.clip(R_m, 0.0, R_CAP_M)
    inv_R = np.divide(
        1.0,
        R_m,
        out=np.full_like(R_m, np.nan, dtype=float),
        where=np.isfinite(R_m) & (R_m > 0.0),
    )
    inv_R_smooth = _peak_preserving_smooth(inv_R, win)
    R_smooth_m = np.divide(
        1.0,
        inv_R_smooth,
        out=np.full_like(inv_R_smooth, R_CAP_M, dtype=float),
        where=np.isfinite(inv_R_smooth) & (inv_R_smooth > 0.0),
    )
    R_smooth_m = np.clip(R_smooth_m, 0.0, R_CAP_M)
    ay_abs_smooth_mps2 = _peak_preserving_smooth(ay_abs, win)
    signed_curvature_smooth_inv_m = np.divide(
        ay_smooth_mps2,
        vx_mps ** 2,
        out=np.full_like(ay_smooth_mps2, np.nan, dtype=float),
        where=np.isfinite(ay_smooth_mps2)
        & np.isfinite(vx_mps)
        & (np.abs(vx_mps) >= MIN_CORNER_SPEED_MPS),
    )

    return {
        "time_s": time_s,
        "laps": laps,
        "laptime": laptime,
        "vx_mps": vx_mps,
        "ay_mps2": ay_mps2,
        "brake_pct": brake_pct,
        "lat": lat,
        "lng": lng,
        "s_lap_m": s_lap_m,
        "R_m": R_m,
        "R_smooth_m": R_smooth_m,
        "curvature_smooth_1pm": inv_R_smooth,
        "ay_abs_smooth_mps2": ay_abs_smooth_mps2,
        "ay_smooth_mps2": ay_smooth_mps2,
        "signed_curvature_smooth_inv_m": signed_curvature_smooth_inv_m,
    }


def select_reference_lap(dfs: dict[str, pl.DataFrame]) -> tuple[str, int]:
    best: tuple[float, str, int] | None = None
    fallback: tuple[str, int] | None = None
    for run_name, df in dfs.items():
        d = compute_radius_curvature(df)
        for lap in unique_laps(d["laps"]):
            lap_id = int(lap)
            if fallback is None:
                fallback = (run_name, lap_id)
            mask = d["laps"] == lap
            lap_time_s = _finite_lap_time(d["laptime"], mask)
            if np.isfinite(lap_time_s) and (
                best is None or lap_time_s < best[0]
            ):
                best = (lap_time_s, run_name, lap_id)
    if best is not None:
        return best[1], best[2]
    if fallback is not None:
        return fallback
    raise ValueError("No valid laps available for cornering reference selection.")


def detect_skidpad_turn_on_lap(
    d: dict[str, np.ndarray], lap_id: int,
) -> list[TurnDef]:
    """Skidpad: each timed lap is a single sustained-radius circle.

    Returns a single TurnDef spanning the full lap, with the apex placed at
    the point of minimum smoothed radius (max curvature). Used in place of
    `detect_turns_on_lap` when the data is tagged as a skidpad event, since
    the generic detector splits the constant-radius circle into spurious
    sub-corners.
    """
    lap = _lap_arrays(d, lap_id)
    n = len(lap["s_lap_m"])
    if n < 2:
        return []
    finite_geom = (
        np.isfinite(lap["s_lap_m"])
        & np.isfinite(lap["lat"])
        & np.isfinite(lap["lng"])
    )
    if not np.any(finite_geom):
        return []
    valid_idx = np.where(finite_geom)[0]
    start = int(valid_idx[0])
    end = int(valid_idx[-1])
    segment = slice(start, end + 1)

    curvature_seg = lap["curvature_smooth_1pm"][segment]
    finite_curvature = np.isfinite(curvature_seg)
    if np.any(finite_curvature):
        local = int(np.nanargmax(np.where(finite_curvature, curvature_seg, np.nan)))
        apex_idx = start + local
        s_apex, apex_lat, apex_lng = _refine_apex(
            lap["curvature_smooth_1pm"],
            lap["s_lap_m"],
            lap["lat"],
            lap["lng"],
            apex_idx,
        )
    else:
        mid = (start + end) // 2
        s_apex = float(lap["s_lap_m"][mid])
        apex_lat = float(lap["lat"][mid])
        apex_lng = float(lap["lng"][mid])

    return [
        TurnDef(
            turn_id=1,
            s_entry_m=float(lap["s_lap_m"][start]),
            s_apex_m=s_apex,
            s_exit_m=float(lap["s_lap_m"][end]),
            apex_lat=apex_lat,
            apex_lng=apex_lng,
            lat=lap["lat"][segment].copy(),
            lng=lap["lng"][segment].copy(),
        )
    ]


def detect_turns_on_lap(
    d: dict[str, np.ndarray], run_name: str, lap_id: int,
    *, R_thr_m: float = 60.0, min_dur_s: float = 0.5, merge_gap_m: float = 8.0,
) -> list[TurnDef]:
    _ = run_name
    lap = _lap_arrays(d, lap_id)
    if len(lap["time_s"]) < 2:
        return []
    dt_s = robust_dt(lap["time_s"])
    curvature_thr_1pm = 1.0 / max(float(R_thr_m), 1.0)
    finite = (
        np.isfinite(lap["R_smooth_m"])
        & np.isfinite(lap["curvature_smooth_1pm"])
        & np.isfinite(lap["ay_abs_smooth_mps2"])
        & np.isfinite(lap["ay_smooth_mps2"])
        & np.isfinite(lap["vx_mps"])
        & np.isfinite(lap["s_lap_m"])
        & np.isfinite(lap["lat"])
        & np.isfinite(lap["lng"])
    )
    speed_ok = np.abs(lap["vx_mps"]) >= MIN_CORNER_SPEED_MPS
    seed = (
        finite
        & speed_ok
        & (lap["curvature_smooth_1pm"] >= curvature_thr_1pm)
        & (lap["ay_abs_smooth_mps2"] >= MIN_PEAK_AY_G * G_MPS2)
    )
    support = (
        finite
        & speed_ok
        & (lap["curvature_smooth_1pm"] >= curvature_thr_1pm / R_RELEASE_FACTOR)
        & (lap["ay_abs_smooth_mps2"] >= SIGN_SPLIT_AY_G * G_MPS2)
    )
    is_corner = _expand_seeds_to_support(seed, support)
    is_corner = keep_min_duration_segments(is_corner, min_dur_s, dt_s)
    is_corner = _fill_short_dist_gaps(
        is_corner,
        lap["s_lap_m"],
        lap["ay_smooth_mps2"],
        merge_gap_m,
    )
    is_corner = keep_min_duration_segments(is_corner, min_dur_s, dt_s)

    turns: list[TurnDef] = []
    segments: list[tuple[int, int]] = []
    for start, end in _segment_bounds(is_corner):
        for sub_start, sub_end in _split_segment_by_direction(
            start,
            end,
            lap["ay_smooth_mps2"],
            lap["s_lap_m"],
            dt_s,
            min_dur_s,
        ):
            segments.extend(
                _split_segment_same_sign(
                    sub_start,
                    sub_end,
                    lap["curvature_smooth_1pm"],
                    lap["s_lap_m"],
                )
            )

    for start, end in segments:
        if (end - start + 1) * dt_s < min_dur_s:
            continue
        corner_length_m = float(lap["s_lap_m"][end] - lap["s_lap_m"][start])
        if not np.isfinite(corner_length_m) or corner_length_m < MIN_CORNER_LENGTH_M:
            continue
        peak_ay = float(np.nanmax(lap["ay_abs_smooth_mps2"][start:end + 1]))
        if not np.isfinite(peak_ay) or peak_ay < MIN_PEAK_AY_G * G_MPS2:
            continue

        segment = slice(start, end + 1)
        curvature_seg = lap["curvature_smooth_1pm"][segment]
        finite_curvature = np.isfinite(curvature_seg)
        if not np.any(finite_curvature):
            continue
        local = int(np.nanargmax(np.where(finite_curvature, curvature_seg, np.nan)))
        apex_idx = start + local
        s_apex, apex_lat, apex_lng = _refine_apex(
            lap["curvature_smooth_1pm"],
            lap["s_lap_m"],
            lap["lat"],
            lap["lng"],
            apex_idx,
        )
        turns.append(
            TurnDef(
                turn_id=len(turns) + 1,
                s_entry_m=float(lap["s_lap_m"][start]),
                s_apex_m=s_apex,
                s_exit_m=float(lap["s_lap_m"][end]),
                apex_lat=apex_lat,
                apex_lng=apex_lng,
                lat=lap["lat"][segment].copy(),
                lng=lap["lng"][segment].copy(),
            )
        )

    turns.sort(key=lambda t: t.s_entry_m)
    return [
        TurnDef(
            turn_id=i,
            s_entry_m=t.s_entry_m,
            s_apex_m=t.s_apex_m,
            s_exit_m=t.s_exit_m,
            apex_lat=t.apex_lat,
            apex_lng=t.apex_lng,
            lat=t.lat,
            lng=t.lng,
        )
        for i, t in enumerate(turns, start=1)
    ]


def compute_turn_metrics(
    dfs: dict[str, pl.DataFrame], turns: list[TurnDef],
) -> pl.DataFrame:
    if not turns:
        return _empty_metrics_df()

    rows: list[dict[str, float | int | str]] = []
    for run_name, df in dfs.items():
        d = compute_radius_curvature(df)
        for lap in unique_laps(d["laps"]):
            lap_id = int(lap)
            lap_mask = d["laps"] == lap
            lap_time_s = _finite_lap_time(d["laptime"], lap_mask)
            lap_data = {key: value[lap_mask] for key, value in d.items()}

            for turn in turns:
                if turn.s_exit_m <= turn.s_entry_m:
                    continue
                win = (
                    (lap_data["s_lap_m"] >= turn.s_entry_m)
                    & (lap_data["s_lap_m"] <= turn.s_exit_m)
                )
                idx = np.where(win)[0]
                if idx.size < 5:
                    continue
                R_win = lap_data["R_smooth_m"][idx]
                finite_R = np.isfinite(R_win)
                if not np.any(finite_R):
                    continue

                apex_local = int(np.nanargmin(np.where(finite_R, R_win, np.nan)))
                apex_idx = int(idx[apex_local])
                entry_idx = int(idx[0])
                exit_idx = int(idx[-1])
                s_apex = float(lap_data["s_lap_m"][apex_idx])

                brake_mask = (
                    (lap_data["s_lap_m"] >= s_apex - BRAKE_LOOKBACK_M)
                    & (lap_data["s_lap_m"] <= s_apex)
                    & (lap_data["brake_pct"] >= BRAKE_ON_PCT)
                    & np.isfinite(lap_data["brake_pct"])
                )
                brake_idx = np.where(brake_mask)[0]
                brake_to_apex = (
                    float(s_apex - lap_data["s_lap_m"][brake_idx[0]])
                    if brake_idx.size
                    else np.nan
                )
                apex_pos_pct = np.clip(
                    100.0 * (s_apex - turn.s_entry_m)
                    / (turn.s_exit_m - turn.s_entry_m),
                    0.0,
                    100.0,
                )

                rows.append(
                    {
                        "run": run_name,
                        "lap": lap_id,
                        "lap_time_s": lap_time_s,
                        "turn_id": int(turn.turn_id),
                        "v_entry_mps": float(lap_data["vx_mps"][entry_idx]),
                        "v_apex_mps": float(lap_data["vx_mps"][apex_idx]),
                        "v_exit_mps": float(lap_data["vx_mps"][exit_idx]),
                        "R_min_m": float(lap_data["R_smooth_m"][apex_idx]),
                        "ay_max_g": float(
                            lap_data["ay_abs_smooth_mps2"][apex_idx] / G_MPS2
                        ),
                        "brake_to_apex_dist_m": brake_to_apex,
                        "apex_pos_pct": float(apex_pos_pct),
                        "corner_length_m": float(turn.s_exit_m - turn.s_entry_m),
                    }
                )

    if not rows:
        return _empty_metrics_df()
    return pl.DataFrame(rows)


def radius_trace_fig(dfs, turns, ref_run, ref_lap, R_thr_m) -> go.Figure:
    fig = make_dark_figure(
        "Corner Radius Trace",
        "Distance from start line [m]",
        "Smoothed radius R [m]",
    )
    entries = _lap_color_entries(dfs)
    color_map = dyn.build_color_map(entries)

    for run_name, df in dfs.items():
        d = compute_radius_curvature(df)
        for lap in unique_laps(d["laps"]):
            lap_id = int(lap)
            mask = d["laps"] == lap
            fig.add_trace(
                go.Scatter(
                    x=d["s_lap_m"][mask],
                    y=d["R_smooth_m"][mask],
                    mode="lines",
                    name=f"{run_name} L{lap_id}",
                    line=dict(
                        color=color_map.get((run_name, lap_id), "#EBEBEB"),
                        width=1.5,
                    ),
                    opacity=0.85,
                )
            )

    for turn in turns:
        color = _TURN_PALETTE[(int(turn.turn_id) - 1) % len(_TURN_PALETTE)]
        fig.add_vrect(
            x0=turn.s_entry_m,
            x1=turn.s_exit_m,
            fillcolor=color,
            opacity=0.12,
            line_width=0,
        )
        fig.add_vline(
            x=turn.s_apex_m,
            line=dict(color=color, dash="dash", width=1.2),
        )
    fig.add_hline(y=R_thr_m, line=dict(color="#EBEBEB", dash="dot", width=1.0))
    fig.update_yaxes(range=[0.0, R_thr_m * 3.0])
    fig.update_layout(legend_title_text="Run / lap")
    return fig


def track_map_with_turns_fig(dfs, turns, ref_run, ref_lap) -> go.Figure:
    fig = make_dark_figure("Track Map with Detected Turns", "Longitude", "Latitude")
    if ref_run not in dfs:
        return fig

    d = compute_radius_curvature(dfs[ref_run])
    mask = d["laps"] == float(ref_lap)
    fig.add_trace(
        go.Scattergl(
            x=d["lng"][mask],
            y=d["lat"][mask],
            mode="lines",
            name=f"{ref_run} L{int(ref_lap)}",
            line=dict(color="rgba(190,190,190,0.55)", width=2),
        )
    )
    for turn in turns:
        color = _TURN_PALETTE[(int(turn.turn_id) - 1) % len(_TURN_PALETTE)]
        name = f"Turn {turn.turn_id}"
        fig.add_trace(
            go.Scattergl(
                x=turn.lng,
                y=turn.lat,
                mode="lines",
                name=name,
                line=dict(color=color, width=5),
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=[turn.apex_lng],
                y=[turn.apex_lat],
                mode="markers",
                name=f"{name} apex",
                marker=dict(color=color, size=12, symbol="star"),
                showlegend=False,
            )
        )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(legend_title_text="Turns")
    return fig


def track_map_focus_turn_fig(
    dfs: dict[str, pl.DataFrame],
    turns: list[TurnDef],
    ref_run: str,
    ref_lap: int,
    focus_turn_id: int,
) -> go.Figure:
    """Track map with one selected turn highlighted and the rest muted."""
    fig = make_dark_figure("Selected Turn on Track", "Longitude", "Latitude")
    if ref_run not in dfs:
        return fig

    d = compute_radius_curvature(dfs[ref_run])
    mask = d["laps"] == float(ref_lap)
    fig.add_trace(
        go.Scattergl(
            x=d["lng"][mask],
            y=d["lat"][mask],
            mode="lines",
            name=f"{Path(ref_run).stem} L{int(ref_lap)}",
            line=dict(color="rgba(190,190,190,0.38)", width=2),
            hoverinfo="skip",
        )
    )
    for turn in turns:
        is_focus = int(turn.turn_id) == int(focus_turn_id)
        color = _TURN_PALETTE[(int(turn.turn_id) - 1) % len(_TURN_PALETTE)]
        fig.add_trace(
            go.Scattergl(
                x=turn.lng,
                y=turn.lat,
                mode="lines",
                name=f"Turn {int(turn.turn_id)}",
                line=dict(
                    color=color if is_focus else "rgba(235,235,235,0.22)",
                    width=7 if is_focus else 2,
                ),
                showlegend=is_focus,
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=[turn.apex_lng],
                y=[turn.apex_lat],
                mode="markers+text" if is_focus else "markers",
                text=[f"T{int(turn.turn_id)}"] if is_focus else None,
                textposition="top center",
                name=f"T{int(turn.turn_id)} apex",
                marker=dict(
                    color=color if is_focus else "rgba(235,235,235,0.35)",
                    size=13 if is_focus else 6,
                    symbol="star" if is_focus else "circle",
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(height=420, showlegend=True)
    return fig


def _run_names(metrics: pl.DataFrame) -> list[str]:
    if metrics.is_empty():
        return []
    return [
        str(run)
        for run in metrics.select("run").unique().sort("run").get_column("run").to_list()
    ]


def _run_mean_row(metrics: pl.DataFrame, run_name: str, turn_id: int) -> dict[str, float | str]:
    sub = metrics.filter((pl.col("run") == run_name) & (pl.col("turn_id") == turn_id))
    if sub.is_empty():
        return {}
    return {
        "Run": Path(run_name).stem,
        "Lap set": "Mean",
        "Entry [m/s]": round(float(sub.get_column("v_entry_mps").mean()), 2),
        "Apex [m/s]": round(float(sub.get_column("v_apex_mps").mean()), 2),
        "Exit [m/s]": round(float(sub.get_column("v_exit_mps").mean()), 2),
        "Min R [m]": round(float(sub.get_column("R_min_m").mean()), 1),
        "Max ay [g]": round(float(sub.get_column("ay_max_g").mean()), 2),
        "Brake to apex [m]": round(float(sub.get_column("brake_to_apex_dist_m").mean()), 1),
        "Apex pos [%]": round(float(sub.get_column("apex_pos_pct").mean()), 1),
    }


def _run_fastest_row(metrics: pl.DataFrame, run_name: str, turn_id: int) -> dict[str, float | str]:
    lap_id = _fastest_lap_for_run(metrics, run_name)
    if lap_id is None:
        return {}
    sub = metrics.filter(
        (pl.col("run") == run_name)
        & (pl.col("lap") == lap_id)
        & (pl.col("turn_id") == turn_id)
    )
    if sub.is_empty():
        return {}
    row = sub.row(0, named=True)
    return {
        "Run": Path(run_name).stem,
        "Lap set": f"Fastest L{int(lap_id)}",
        "Entry [m/s]": round(float(row["v_entry_mps"]), 2),
        "Apex [m/s]": round(float(row["v_apex_mps"]), 2),
        "Exit [m/s]": round(float(row["v_exit_mps"]), 2),
        "Min R [m]": round(float(row["R_min_m"]), 1),
        "Max ay [g]": round(float(row["ay_max_g"]), 2),
        "Brake to apex [m]": round(float(row["brake_to_apex_dist_m"]), 1),
        "Apex pos [%]": round(float(row["apex_pos_pct"]), 1),
    }


def turn_focus_table(metrics: pl.DataFrame, turn_id: int) -> pl.DataFrame:
    """Mean and fastest-lap corner metrics for one selected turn."""
    rows: list[dict[str, float | str]] = []
    for run_name in _run_names(metrics):
        mean_row = _run_mean_row(metrics, run_name, int(turn_id))
        fast_row = _run_fastest_row(metrics, run_name, int(turn_id))
        if mean_row:
            rows.append(mean_row)
        if fast_row:
            rows.append(fast_row)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def turn_speed_story_fig(metrics: pl.DataFrame, turn_id: int) -> go.Figure:
    """Entry-apex-exit speed story for one turn."""
    fig = make_dark_figure(
        f"Turn {int(turn_id)} Speed Story",
        "Corner point",
        "Speed [m/s]",
    )
    points = ["Entry", "Apex", "Exit"]
    for idx, run_name in enumerate(_run_names(metrics)):
        sub = metrics.filter((pl.col("run") == run_name) & (pl.col("turn_id") == int(turn_id)))
        if sub.is_empty():
            continue
        color = _comparison_trace_color(idx)
        mean_values = [
            float(sub.get_column("v_entry_mps").mean()),
            float(sub.get_column("v_apex_mps").mean()),
            float(sub.get_column("v_exit_mps").mean()),
        ]
        fig.add_trace(
            go.Scatter(
                x=points,
                y=mean_values,
                mode="lines+markers",
                name=f"{Path(run_name).stem} mean",
                line=dict(color=color, width=2.2),
                marker=dict(color=color, size=9),
            )
        )
        lap_id = _fastest_lap_for_run(metrics, run_name)
        if lap_id is None:
            continue
        fast = metrics.filter(
            (pl.col("run") == run_name)
            & (pl.col("lap") == lap_id)
            & (pl.col("turn_id") == int(turn_id))
        )
        if fast.is_empty():
            continue
        row = fast.row(0, named=True)
        fast_values = [
            float(row["v_entry_mps"]),
            float(row["v_apex_mps"]),
            float(row["v_exit_mps"]),
        ]
        fig.add_trace(
            go.Scatter(
                x=points,
                y=fast_values,
                mode="markers",
                name=f"{Path(run_name).stem} fastest L{int(lap_id)}",
                marker=dict(color=color, size=11, symbol="diamond"),
            )
        )
    fig.update_layout(height=420)
    return fig


def corner_turn_diagnosis_table(metrics: pl.DataFrame) -> pl.DataFrame:
    """Rank turns by the biggest mean difference between the first two runs."""
    run_names = _run_names(metrics)
    turn_ids = _turn_ids(metrics)
    if len(run_names) < 2 or not turn_ids:
        return pl.DataFrame()

    ref_run, cmp_run = run_names[:2]
    rows: list[dict[str, float | str | int]] = []
    for turn_id in turn_ids:
        ref = metrics.filter((pl.col("run") == ref_run) & (pl.col("turn_id") == turn_id))
        cmp = metrics.filter((pl.col("run") == cmp_run) & (pl.col("turn_id") == turn_id))
        if ref.is_empty() or cmp.is_empty():
            continue
        dv_apex = float(cmp.get_column("v_apex_mps").mean() - ref.get_column("v_apex_mps").mean())
        dv_exit = float(cmp.get_column("v_exit_mps").mean() - ref.get_column("v_exit_mps").mean())
        d_brake = float(
            cmp.get_column("brake_to_apex_dist_m").mean()
            - ref.get_column("brake_to_apex_dist_m").mean()
        )
        d_apex = float(cmp.get_column("apex_pos_pct").mean() - ref.get_column("apex_pos_pct").mean())
        if dv_exit < -0.6:
            limiter = "Exit speed"
        elif dv_apex < -0.6:
            limiter = "Minimum speed"
        elif np.isfinite(d_brake) and d_brake > 6.0:
            limiter = "Long braking"
        elif np.isfinite(d_apex) and abs(d_apex) > 10.0:
            limiter = "Apex placement"
        else:
            limiter = "Small difference"
        rows.append({
            "Turn": int(turn_id),
            "Main limiter": limiter,
            f"Δ apex [m/s]": round(dv_apex, 2),
            f"Δ exit [m/s]": round(dv_exit, 2),
            f"Δ brake dist [m]": round(d_brake, 1) if np.isfinite(d_brake) else np.nan,
            f"Δ apex pos [%]": round(d_apex, 1),
            "__score": abs(min(0.0, dv_exit)) + 0.7 * abs(min(0.0, dv_apex)) + 0.03 * abs(d_brake),
        })
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows)
        .sort("__score", descending=True)
        .drop("__score")
    )


def corner_speed_delta_overview_fig(metrics: pl.DataFrame) -> go.Figure:
    """Heatmap of mean speed delta at entry, apex and exit for all turns."""
    run_names = _run_names(metrics)
    turn_ids = _turn_ids(metrics)
    fig = make_dark_figure(
        "Corner Speed Delta Overview",
        "Turn",
        "Corner point",
    )
    if len(run_names) < 2 or not turn_ids:
        return fig

    ref_run, cmp_run = run_names[:2]
    rows = [
        ("Entry", "v_entry_mps"),
        ("Apex", "v_apex_mps"),
        ("Exit", "v_exit_mps"),
    ]
    z: list[list[float]] = []
    text: list[list[str]] = []
    for _label, col in rows:
        values: list[float] = []
        labels: list[str] = []
        for turn_id in turn_ids:
            ref = metrics.filter((pl.col("run") == ref_run) & (pl.col("turn_id") == turn_id))
            cmp = metrics.filter((pl.col("run") == cmp_run) & (pl.col("turn_id") == turn_id))
            if ref.is_empty() or cmp.is_empty():
                values.append(np.nan)
                labels.append("")
                continue
            delta = float(cmp.get_column(col).mean() - ref.get_column(col).mean())
            values.append(delta)
            labels.append(f"{delta:+.2f}")
        z.append(values)
        text.append(labels)

    finite = np.asarray(z, dtype=float)
    finite_vals = finite[np.isfinite(finite)]
    zmax = max(0.8, float(np.nanpercentile(np.abs(finite_vals), 90))) if finite_vals.size else 1.0
    fig.add_trace(
        go.Heatmap(
            x=[f"T{turn_id}" for turn_id in turn_ids],
            y=[label for label, _col in rows],
            z=z,
            text=text,
            texttemplate="%{text}",
            zmin=-zmax,
            zmax=zmax,
            colorscale=[
                [0.0, "#EB5757"],
                [0.5, "#2A2B31"],
                [1.0, "#27AE60"],
            ],
            colorbar=dict(title="Δ m/s"),
            hovertemplate=(
                "%{y} %{x}<br>"
                f"{Path(cmp_run).stem} - {Path(ref_run).stem}: "
                "%{z:+.2f} m/s<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=70, r=25, t=65, b=45),
    )
    return fig


def corner_quick_findings_table(metrics: pl.DataFrame) -> pl.DataFrame:
    """Small list of turns with the clearest speed loss, sorted by severity."""
    run_names = _run_names(metrics)
    turn_ids = _turn_ids(metrics)
    if len(run_names) < 2 or not turn_ids:
        return pl.DataFrame()

    ref_run, cmp_run = run_names[:2]
    rows: list[dict[str, float | str | int]] = []
    for turn_id in turn_ids:
        ref = metrics.filter((pl.col("run") == ref_run) & (pl.col("turn_id") == turn_id))
        cmp = metrics.filter((pl.col("run") == cmp_run) & (pl.col("turn_id") == turn_id))
        if ref.is_empty() or cmp.is_empty():
            continue
        deltas = {
            "Entry": float(cmp.get_column("v_entry_mps").mean() - ref.get_column("v_entry_mps").mean()),
            "Apex": float(cmp.get_column("v_apex_mps").mean() - ref.get_column("v_apex_mps").mean()),
            "Exit": float(cmp.get_column("v_exit_mps").mean() - ref.get_column("v_exit_mps").mean()),
        }
        worst_point, worst_delta = min(deltas.items(), key=lambda item: item[1])
        d_brake = float(
            cmp.get_column("brake_to_apex_dist_m").mean()
            - ref.get_column("brake_to_apex_dist_m").mean()
        )
        if worst_delta >= -0.35:
            continue
        if worst_point == "Exit":
            read = "loses on exit"
        elif worst_point == "Apex":
            read = "too slow at apex"
        else:
            read = "arrives slower"
        rows.append({
            "Turn": int(turn_id),
            "Where": worst_point,
            "Loss": read,
            "Δ speed [m/s]": round(worst_delta, 2),
            "Δ brake dist [m]": round(d_brake, 1) if np.isfinite(d_brake) else np.nan,
            "__severity": abs(worst_delta),
        })

    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows)
        .sort("__severity", descending=True)
        .drop("__severity")
        .head(6)
    )


def corner_metric_bars_fig(
    metrics: pl.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
) -> go.Figure:
    """Grouped per-turn bars: run mean plus fastest lap for each run."""
    fig = make_dark_figure(title, "Turn", ylabel)
    turn_ids = _turn_ids(metrics)
    run_names = _run_names(metrics)
    if not turn_ids or not run_names:
        return fig

    x = [f"T{turn_id}" for turn_id in turn_ids]
    color_idx = 0
    for run_name in run_names:
        run_label = Path(run_name).stem
        mean_values = _mean_series_by_turn(metrics, run_name, value_col, turn_ids)
        mean_color = _comparison_trace_color(color_idx)
        color_idx += 1
        fig.add_trace(
            go.Bar(
                x=x,
                y=mean_values,
                name=f"{run_label} mean",
                marker_color=mean_color,
                opacity=0.72,
            )
        )

        fastest_lap = _fastest_lap_for_run(metrics, run_name)
        if fastest_lap is None:
            continue
        fast_values = _series_by_turn(metrics, run_name, fastest_lap, value_col, turn_ids)
        fast_color = _comparison_trace_color(color_idx)
        color_idx += 1
        fig.add_trace(
            go.Scatter(
                x=x,
                y=fast_values,
                mode="markers",
                name=f"{run_label} fastest L{fastest_lap}",
                marker=dict(color=fast_color, size=10, symbol="diamond"),
            )
        )

    fig.update_layout(
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(20,20,23,0.75)",
        ),
        height=430,
    )
    return fig


def corner_delta_table(metrics: pl.DataFrame) -> pl.DataFrame:
    """Mean per-turn difference between the first two runs in sorted order."""
    run_names = _run_names(metrics)
    turn_ids = _turn_ids(metrics)
    if len(run_names) < 2 or not turn_ids:
        return pl.DataFrame()

    run_a, run_b = run_names[:2]
    rows: list[dict[str, float | int]] = []
    for turn_id in turn_ids:
        row: dict[str, float | int] = {"Turn": int(turn_id)}
        for col, label in (
            ("v_apex_mps", "Δ apex speed [m/s]"),
            ("v_exit_mps", "Δ exit speed [m/s]"),
            ("R_min_m", "Δ min radius [m]"),
            ("brake_to_apex_dist_m", "Δ brake dist [m]"),
            ("apex_pos_pct", "Δ apex pos [%]"),
        ):
            vals = []
            for run_name in (run_a, run_b):
                sub = metrics.filter(
                    (pl.col("run") == run_name)
                    & (pl.col("turn_id") == turn_id)
                    & pl.col(col).is_finite()
                )
                vals.append(float(sub.get_column(col).mean()) if not sub.is_empty() else np.nan)
            row[label] = round(vals[1] - vals[0], 3) if all(np.isfinite(vals)) else np.nan
        rows.append(row)

    table = pl.DataFrame(rows)
    return table.with_columns(
        pl.lit(f"{Path(run_b).stem} - {Path(run_a).stem}").alias("Comparison")
    ).select(["Comparison", *[c for c in table.columns]])


def braking_distance_columns_fig(metrics: pl.DataFrame) -> go.Figure:
    turn_ids = _turn_ids(metrics)
    if not turn_ids:
        return make_dark_figure("Braking Distance to Apex", "Turn", "Distance [m]")
    fig = make_subplots(
        rows=1,
        cols=len(turn_ids),
        subplot_titles=[f"Turn {tid}" for tid in turn_ids],
        shared_yaxes=False,
    )
    _dark_subplot_layout(fig, "Braking Distance to Apex")
    entries = _metrics_entries(metrics)
    color_map = dyn.build_color_map(entries)
    run_x, jitter_width = _run_split_positions(metrics)
    for col_idx, turn_id in enumerate(turn_ids, start=1):
        if len(run_x) == 2:
            fig.add_vline(
                x=0.5,
                line=dict(color="rgba(235,235,235,0.22)", width=1),
                row=1,
                col=col_idx,
            )
        for entry_idx, (run_name, lap_id, _lap_time_s) in enumerate(entries):
            sub = metrics.filter(
                (pl.col("run") == run_name)
                & (pl.col("lap") == lap_id)
                & (pl.col("turn_id") == turn_id)
            )
            if sub.is_empty():
                continue
            y = float(sub.get_column("brake_to_apex_dist_m")[0])
            x_center = run_x.get(run_name, 0.5)
            jitter = _stable_unit_jitter(run_name, lap_id, turn_id) * jitter_width
            fig.add_trace(
                go.Scatter(
                    x=[x_center + jitter],
                    y=[y],
                    mode="markers",
                    name=f"{run_name} L{lap_id}",
                    marker=dict(
                        color=color_map.get((run_name, lap_id), "#EBEBEB"),
                        size=9,
                    ),
                    showlegend=col_idx == 1,
                ),
                row=1,
                col=col_idx,
            )
        fig.update_xaxes(showticklabels=False, range=[0.0, 1.0], row=1, col=col_idx)
        fig.update_yaxes(title_text="Distance [m]" if col_idx == 1 else None, row=1, col=col_idx)
    fig.update_layout(legend_title_text="Run / lap")
    return fig


def apex_position_columns_fig(metrics: pl.DataFrame) -> go.Figure:
    turn_ids = _turn_ids(metrics)
    if not turn_ids:
        return make_dark_figure("Apex Position", "Turn", "Apex position [%]")
    fig = make_subplots(
        rows=1,
        cols=len(turn_ids),
        subplot_titles=[f"Turn {tid}" for tid in turn_ids],
        shared_yaxes=True,
    )
    _dark_subplot_layout(fig, "Apex Position")
    entries = _metrics_entries(metrics)
    color_map = dyn.build_color_map(entries)
    run_x, jitter_width = _run_split_positions(metrics)
    for col_idx, turn_id in enumerate(turn_ids, start=1):
        if len(run_x) == 2:
            fig.add_vline(
                x=0.5,
                line=dict(color="rgba(235,235,235,0.22)", width=1),
                row=1,
                col=col_idx,
            )
        fig.add_hrect(
            y0=0,
            y1=40,
            fillcolor="#27AE60",
            opacity=0.12,
            line_width=0,
            row=1,
            col=col_idx,
        )
        fig.add_hrect(
            y0=40,
            y1=60,
            fillcolor="#F2C94C",
            opacity=0.16,
            line_width=0,
            row=1,
            col=col_idx,
        )
        fig.add_hrect(
            y0=60,
            y1=100,
            fillcolor="#EB5757",
            opacity=0.12,
            line_width=0,
            row=1,
            col=col_idx,
        )
        for run_name, lap_id, _lap_time_s in entries:
            sub = metrics.filter(
                (pl.col("run") == run_name)
                & (pl.col("lap") == lap_id)
                & (pl.col("turn_id") == turn_id)
            )
            if sub.is_empty():
                continue
            y = float(sub.get_column("apex_pos_pct")[0])
            x_center = run_x.get(run_name, 0.5)
            jitter = _stable_unit_jitter(turn_id, run_name, lap_id) * jitter_width
            fig.add_trace(
                go.Scatter(
                    x=[x_center + jitter],
                    y=[y],
                    mode="markers",
                    name=f"{run_name} L{lap_id}",
                    marker=dict(
                        color=color_map.get((run_name, lap_id), "#EBEBEB"),
                        size=9,
                    ),
                    showlegend=col_idx == 1,
                ),
                row=1,
                col=col_idx,
            )
        fig.update_xaxes(showticklabels=False, range=[0.0, 1.0], row=1, col=col_idx)
        fig.update_yaxes(
            range=[0, 100],
            title_text="Apex position [%]" if col_idx == 1 else None,
            row=1,
            col=col_idx,
        )
    fig.update_layout(legend_title_text="Run / lap")
    return fig
