"""rb.py
------
Regenerative Braking (RB) KPIs — braking slip control and brake balance quality.

KPIs are computed during valid braking phases:
  brake >= threshold AND ax <= threshold AND vx >= min_speed

When available, active RB phases are further restricted to `RB_Enable == 1.0`.
Slip-ratio tracking uses the braking target SR = -0.20.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go

from utils import (
    COMPLETE_LAPS_MARKER,
    WHEEL_COLORS,
    WHEEL_SYMBOLS,
    cols_to_numpy,
    ensure_complete_laps_df,
    exclude_lap0_and_last_lap,
    keep_min_duration_segments,
    make_dark_figure,
    per_lap_axis,
    robust_dt,
    smooth_signal,
    unique_laps,
)

CSV_PATH = "data/run4_2025-08-24.csv"
WHEELS = ("FL", "FR", "RL", "RR")
MASTER_TORQUE_COLS = {
    "FL": "Master_frontLeftTrq",
    "FR": "Master_frontRightTrq",
    "RL": "Master_rearLeftTrq",
    "RR": "Master_rearRightTrq",
}
ACTUAL_TORQUE_COLS = {w: f"{w}_actualTorque" for w in WHEELS}

SR_TARGET_BRAKE = -0.20
DELTA_SR = 0.05
BRAKE_THRESHOLD = 5.0
AX_BRAKE_THRESHOLD = -0.50
MIN_SPEED = 4.0
MIN_EVENT_DURATION = 0.15
MIN_SAMPLES_PER_LAP = 40
VEHICLE_MASS_KG = 288.0
LOCKUP_SR_THRESHOLD = -0.30
LOCKUP_MIN_DURATION_S = 0.05
STEADY_MIN_DURATION_S = 0.20
STEADY_SMOOTH_WINDOW_S = 0.30
STEADY_BRAKE_STD_THRESHOLD = 5.0
STEADY_JERK_THRESHOLD = 2.0
STRAIGHT_STEER_THRESHOLD_RAD = 0.05
STRAIGHT_AY_THRESHOLD_MS2 = 3.0


def _load(columns: list[str]) -> dict[str, np.ndarray]:
    df = pl.read_csv(CSV_PATH, columns=columns)
    return cols_to_numpy(df, columns)


def _from_df(df: pl.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    df = ensure_complete_laps_df(df)
    cols = list(columns)
    if COMPLETE_LAPS_MARKER in df.columns and COMPLETE_LAPS_MARKER not in cols:
        cols.append(COMPLETE_LAPS_MARKER)
    return cols_to_numpy(df, cols)


def _ax_signal(columns: list[str]) -> str:
    return "Filtering_VN_ax" if "Filtering_VN_ax" in columns else "VN_ax"


def _vx_signal(columns: list[str]) -> str:
    return "Est_vxCOG" if "Est_vxCOG" in columns else "VN_vx"


def _ay_signal(columns: list[str]) -> str | None:
    if "Filtering_VN_ay" in columns:
        return "Filtering_VN_ay"
    if "VN_ay" in columns:
        return "VN_ay"
    return None


def _yaw_signal(columns: list[str]) -> str | None:
    if "VN_gz" in columns:
        return "VN_gz"
    if "AS_yaw_rate" in columns:
        return "AS_yaw_rate"
    return None


def _pitch_rate_signal(columns: list[str]) -> str | None:
    return "VN_gy" if "VN_gy" in columns else None


def _vy_signal(columns: list[str]) -> str | None:
    if "Est_vyCOG" in columns:
        return "Est_vyCOG"
    if "VN_vy" in columns:
        return "VN_vy"
    return None


def _finite_corr(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return np.nan
    x_m = x[m]
    y_m = y[m]
    if np.nanstd(x_m) < 1e-9 or np.nanstd(y_m) < 1e-9:
        return np.nan
    return float(np.corrcoef(x_m, y_m)[0, 1])


def _origin_slope(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1e-9)
    if int(m.sum()) < 3:
        return np.nan
    denom = float(np.nansum(x[m] ** 2))
    if denom <= 1e-9:
        return np.nan
    return float(np.nansum(x[m] * y[m]) / denom)


def _safe_mean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def _safe_median(x: np.ndarray) -> float:
    return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan


def _safe_p95(x: np.ndarray) -> float:
    return float(np.nanpercentile(x[np.isfinite(x)], 95)) if np.isfinite(x).any() else np.nan


def _safe_max(x: np.ndarray) -> float:
    return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan


def _segment_bounds(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    cuts = np.where(np.diff(idx) > 1)[0] + 1
    return [(int(seg[0]), int(seg[-1])) for seg in np.split(idx, cuts) if seg.size]


def _first_delay_ms(
    signal: np.ndarray,
    start: int,
    end: int,
    threshold: float,
    dt_s: float,
) -> float:
    if end < start:
        return np.nan
    rel = np.flatnonzero(np.isfinite(signal[start:end + 1]) & (signal[start:end + 1] >= threshold))
    if rel.size == 0:
        return np.nan
    return float(rel[0] * dt_s * 1000.0)


def _rolling_std(signal: np.ndarray, window_samples: int) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    if window_samples <= 1 or arr.size == 0:
        return np.zeros_like(arr, dtype=float)

    finite = np.isfinite(arr)
    kernel = np.ones(int(window_samples), dtype=float)
    sums = np.convolve(np.where(finite, arr, 0.0), kernel, mode="same")
    sums_sq = np.convolve(np.where(finite, arr * arr, 0.0), kernel, mode="same")
    counts = np.convolve(finite.astype(float), kernel, mode="same")

    out = np.full(arr.shape, np.nan, dtype=float)
    ok = counts >= max(2.0, 0.8 * window_samples)
    if not np.any(ok):
        return out

    mean = sums[ok] / counts[ok]
    variance = np.maximum(0.0, sums_sq[ok] / counts[ok] - mean * mean)
    out[ok] = np.sqrt(variance)
    return out


def _event_duration_s(start: int, end: int, dt_s: float) -> float:
    return float((end - start + 1) * dt_s)


def _yaw_event_metrics(
    yaw_rate_radps: np.ndarray,
    start: int,
    end: int,
    dt_s: float,
    is_straight_event: bool,
) -> dict[str, float]:
    seg = slice(start, end + 1)
    mask = np.isfinite(yaw_rate_radps[seg])
    if not is_straight_event or not np.any(mask):
        return {
            "Yaw straight peak [rad/s]": np.nan,
            "Yaw straight integral [rad]": np.nan,
        }

    yaw_abs = np.abs(yaw_rate_radps[seg][mask])
    return {
        "Yaw straight peak [rad/s]": float(np.nanmax(yaw_abs)),
        "Yaw straight integral [rad]": float(np.nansum(yaw_abs) * dt_s),
    }


def _is_straight_brake_event(
    ay_ms2: np.ndarray,
    steering_rad: np.ndarray,
    start: int,
    end: int,
) -> bool:
    seg = slice(start, end + 1)

    ay_seg = ay_ms2[seg]
    steer_seg = steering_rad[seg]
    ay_ok = True
    steer_ok = True
    has_ref = False

    ay_finite = np.isfinite(ay_seg)
    if np.any(ay_finite):
        has_ref = True
        ay_ok = float(np.nanmax(np.abs(ay_seg[ay_finite]))) < STRAIGHT_AY_THRESHOLD_MS2

    steer_finite = np.isfinite(steer_seg)
    if np.any(steer_finite):
        has_ref = True
        steer_ok = float(np.nanmax(np.abs(steer_seg[steer_finite]))) < STRAIGHT_STEER_THRESHOLD_RAD

    return has_ref and ay_ok and steer_ok


def _beta_event_metrics(
    vx_ms: np.ndarray,
    vy_ms: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    seg = slice(start, end + 1)
    mask = np.isfinite(vx_ms[seg]) & np.isfinite(vy_ms[seg]) & (np.abs(vx_ms[seg]) > 1e-6)
    if not np.any(mask):
        return {"Beta peak [deg]": np.nan}

    beta_deg = np.rad2deg(np.arctan2(vy_ms[seg][mask], vx_ms[seg][mask]))
    return {"Beta peak [deg]": float(np.nanmax(np.abs(beta_deg)))}


def _lr_decel_asymmetry(
    wheel_decel_ms2: dict[str, np.ndarray],
    start: int,
    end: int,
) -> dict[str, float]:
    seg = slice(start, end + 1)
    front_diff = np.abs(wheel_decel_ms2["FL"][seg] - wheel_decel_ms2["FR"][seg])
    rear_diff = np.abs(wheel_decel_ms2["RL"][seg] - wheel_decel_ms2["RR"][seg])
    return {
        "Front L-R decel asym [m/s²]": _safe_mean(front_diff),
        "Rear L-R decel asym [m/s²]": _safe_mean(rear_diff),
    }


def _lockup_events(
    sr: dict[str, np.ndarray],
    laps: np.ndarray,
    brake_mask: np.ndarray,
    dt_s: float,
) -> dict[str, object]:
    event_rows: list[dict[str, float | int | str]] = []
    masks_by_wheel: dict[str, np.ndarray] = {}
    counts_by_wheel = {w: 0 for w in WHEELS}
    counts_by_lap: dict[int, dict[str, int]] = {}
    worst_min_sr = np.nan

    for wheel in WHEELS:
        raw = brake_mask & np.isfinite(sr[wheel]) & (sr[wheel] < LOCKUP_SR_THRESHOLD)
        lock_mask = keep_min_duration_segments(raw, LOCKUP_MIN_DURATION_S, dt_s)
        masks_by_wheel[wheel] = lock_mask
        for start, end in _segment_bounds(lock_mask):
            seg = slice(start, end + 1)
            lap = int(round(_safe_median(laps[seg])))
            min_sr = float(np.nanmin(sr[wheel][seg]))
            worst_min_sr = min_sr if not np.isfinite(worst_min_sr) else min(worst_min_sr, min_sr)
            counts_by_wheel[wheel] += 1
            counts_by_lap.setdefault(lap, {w: 0 for w in WHEELS})
            counts_by_lap[lap][wheel] += 1
            event_rows.append({
                "wheel": wheel,
                "lap": lap,
                "start": start,
                "end": end,
                "min_sr": min_sr,
                "duration_ms": _event_duration_s(start, end, dt_s) * 1000.0,
            })

    return {
        "events": event_rows,
        "masks_by_wheel": masks_by_wheel,
        "counts_by_wheel": counts_by_wheel,
        "counts_by_lap": counts_by_lap,
        "worst_min_sr": worst_min_sr,
        "count_total": int(sum(counts_by_wheel.values())),
    }


def _lockup_per_lap(
    laps: np.ndarray,
    laptime_s: np.ndarray,
    brake_mask: np.ndarray,
    lock_masks_by_wheel: dict[str, np.ndarray],
    dt_s: float,
) -> dict[str, object]:
    lap_list = unique_laps(laps)
    n_laps = len(lap_list)
    lap_time_vals_s = np.full(n_laps, np.nan, dtype=float)
    brake_time_s = np.full(n_laps, np.nan, dtype=float)
    total_lock_time_s = np.full(n_laps, np.nan, dtype=float)
    total_lock_pct_brake = np.full(n_laps, np.nan, dtype=float)
    lock_time_s_by_wheel = {
        wheel: np.full(n_laps, np.nan, dtype=float) for wheel in WHEELS
    }
    lock_events_by_wheel = {
        wheel: np.zeros(n_laps, dtype=int) for wheel in WHEELS
    }

    for idx, lap in enumerate(lap_list):
        lap_mask = laps == lap
        if not np.any(lap_mask):
            continue

        lap_time_vals_s[idx] = float(np.nanmax(laptime_s[lap_mask]))
        lap_brake_time_s = float(np.sum(brake_mask[lap_mask]) * dt_s)
        brake_time_s[idx] = lap_brake_time_s

        lap_total_lock_time_s = 0.0
        for wheel in WHEELS:
            wheel_lock_mask = lap_mask & lock_masks_by_wheel[wheel]
            wheel_lock_time_s = float(np.sum(wheel_lock_mask) * dt_s)
            lock_time_s_by_wheel[wheel][idx] = wheel_lock_time_s
            lock_events_by_wheel[wheel][idx] = len(_segment_bounds(wheel_lock_mask))
            lap_total_lock_time_s += wheel_lock_time_s

        total_lock_time_s[idx] = lap_total_lock_time_s
        total_lock_pct_brake[idx] = (
            100.0 * lap_total_lock_time_s / lap_brake_time_s
            if lap_brake_time_s > 1e-9
            else np.nan
        )

    table_dict: dict[str, np.ndarray] = {
        "Lap": lap_list.astype(int),
        "LapTime [s]": np.round(lap_time_vals_s, 3),
        "Brake time [s]": np.round(brake_time_s, 3),
        "Total lock time [s]": np.round(total_lock_time_s, 3),
        "Total lock / brake [%]": np.round(total_lock_pct_brake, 2),
    }
    for wheel in WHEELS:
        table_dict[f"{wheel} lock time [s]"] = np.round(lock_time_s_by_wheel[wheel], 3)
        table_dict[f"{wheel} lock events"] = lock_events_by_wheel[wheel].astype(int)

    return {
        "lap_list": lap_list,
        "lap_time_vals_s": lap_time_vals_s,
        "brake_time_s": brake_time_s,
        "total_lock_time_s": total_lock_time_s,
        "total_lock_pct_brake": total_lock_pct_brake,
        "lock_time_s_by_wheel": lock_time_s_by_wheel,
        "lock_events_by_wheel": lock_events_by_wheel,
        "table": pl.DataFrame(table_dict),
    }


def _sr_steady_std(
    sr: dict[str, np.ndarray],
    steady_mask: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    seg = slice(start, end + 1)
    seg_mask = steady_mask[seg]
    per_wheel: dict[str, float] = {}
    for wheel in WHEELS:
        vals = sr[wheel][seg][seg_mask]
        per_wheel[wheel] = float(np.nanstd(vals)) if np.isfinite(vals).sum() >= 3 else np.nan

    agg = np.array(list(per_wheel.values()), dtype=float)
    return {
        "steady_std_by_wheel": per_wheel,
        "Steady SR osc mean [-]": _safe_mean(agg),
        "Steady SR osc max [-]": _safe_max(agg),
    }


def _bias_vs_fz(
    front_share: np.ndarray,
    fz_front_share: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    seg = slice(start, end + 1)
    real = front_share[seg]
    ideal = fz_front_share[seg]
    mask = np.isfinite(real) & np.isfinite(ideal)
    if not np.any(mask):
        return {
            "Front share mean [%]": np.nan,
            "Fz front share mean [%]": np.nan,
            "Front share vs Fz MAE [%]": np.nan,
            "Front share vs Fz bias [%]": np.nan,
        }

    err_pct = (real[mask] - ideal[mask]) * 100.0
    return {
        "Front share mean [%]": float(np.nanmean(real[mask]) * 100.0),
        "Fz front share mean [%]": float(np.nanmean(ideal[mask]) * 100.0),
        "Front share vs Fz MAE [%]": float(np.nanmean(np.abs(err_pct))),
        "Front share vs Fz bias [%]": float(np.nanmean(err_pct)),
    }


def _pitch_dive_event(
    pitch_rate_radps: np.ndarray,
    front_damper: np.ndarray,
    rear_damper: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    seg = slice(start, end + 1)
    pitch = np.abs(pitch_rate_radps[seg])
    peak_pitch = _safe_max(pitch)

    front_seg = front_damper[seg]
    rear_seg = rear_damper[seg]
    if np.isfinite(front_seg).any() and np.isfinite(rear_seg).any():
        front_ref = front_seg[np.isfinite(front_seg)][0]
        rear_ref = rear_seg[np.isfinite(rear_seg)][0]
        dive_trace = (front_seg - front_ref) - (rear_seg - rear_ref)
        dive_peak = _safe_max(np.abs(dive_trace))
    else:
        dive_peak = np.nan

    return {
        "Pitch peak [rad/s]": peak_pitch,
        "Dive asym peak [damper]": dive_peak,
    }


def _wheel_decel_from_speed(
    wheel_speed_ms: np.ndarray,
    dt_s: float,
    smooth_samples: int,
) -> np.ndarray:
    if not np.isfinite(wheel_speed_ms).any():
        return np.full_like(wheel_speed_ms, np.nan)
    return -np.gradient(smooth_signal(wheel_speed_ms, smooth_samples), dt_s)


def _prepare_arrays_from_df(
    df: pl.DataFrame,
    brake_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    ax_col = _ax_signal(df.columns)
    vx_col = _vx_signal(df.columns)
    cols = [
        "TimeStamp", "laps", "laptime", "Brake", ax_col, vx_col,
        "RB_Enable",
        "RB_intensityTarget",
        "Est_SRFL", "Est_SRFR", "Est_SRRL", "Est_SRRR",
    ]
    d = _from_df(df, cols)
    d["time"] = d["TimeStamp"] - d["TimeStamp"][0]
    d["ax"] = d.pop(ax_col)
    d["vx"] = d.pop(vx_col)
    if brake_mask is not None:
        d["__brake_mask"] = brake_mask.astype(float)
    return d


def _prepare_arrays_from_csv() -> dict[str, np.ndarray]:
    header = pl.read_csv(CSV_PATH, n_rows=1).columns
    ax_col = _ax_signal(header)
    vx_col = _vx_signal(header)
    d = _load([
        "TimeStamp", "laps", "laptime", "Brake", ax_col, vx_col,
        "RB_Enable",
        "RB_intensityTarget",
        "Est_SRFL", "Est_SRFR", "Est_SRRL", "Est_SRRR",
    ])
    d["time"] = d["TimeStamp"] - d["TimeStamp"][0]
    d["ax"] = d.pop(ax_col)
    d["vx"] = d.pop(vx_col)
    return d


def _compute_rb(d: dict[str, np.ndarray]) -> dict:
    has_ext = "__brake_mask" in d
    data_keys = [k for k in d if not k.startswith("__")]
    valid = np.all(np.stack([np.isfinite(d[k]) for k in data_keys], axis=1), axis=1)
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)

    dt = robust_dt(d["time"])
    laps = d["laps"]
    laptime = d["laptime"]

    if has_ext:
        brake_mask = d["__brake_mask"].astype(bool) & (np.abs(d["vx"]) >= MIN_SPEED)
    else:
        raw_brake = (
            (d["Brake"] >= BRAKE_THRESHOLD)
            & (d["ax"] <= AX_BRAKE_THRESHOLD)
            & (np.abs(d["vx"]) >= MIN_SPEED)
        )
        brake_mask = keep_min_duration_segments(raw_brake, MIN_EVENT_DURATION, dt)

    rb_enable_mask = d["RB_Enable"] == 1.0
    use_rb_enable = np.any(rb_enable_mask)
    if use_rb_enable:
        rb_active_raw = brake_mask & rb_enable_mask
        rb_mask = keep_min_duration_segments(rb_active_raw, MIN_EVENT_DURATION, dt)
    else:
        # CAT17x has no hydraulic braking: if RB_Enable is not trustworthy,
        # treat valid braking phases as regenerative phases.
        rb_mask = brake_mask.copy()

    sr = {w: d[f"Est_SR{w}"] for w in WHEELS}
    sr_mat = np.stack([sr[w] for w in WHEELS], axis=1)
    sr_global = np.nanmean(sr_mat, axis=1)

    lower_thr = SR_TARGET_BRAKE - DELTA_SR
    upper_thr = SR_TARGET_BRAKE + DELTA_SR
    in_target_glob = (sr_global >= lower_thr) & (sr_global <= upper_thr)
    overslip_glob = sr_global < lower_thr
    underslip_glob = sr_global > upper_thr

    lap_list = unique_laps(laps)
    n = len(lap_list)
    lt_val = np.full(n, np.nan)
    brake_samps = np.zeros(n, dtype=int)
    rb_samps = np.zeros(n, dtype=int)
    rb_cover = np.full(n, np.nan)

    sr_mae = np.full(n, np.nan)
    sr_bias = np.full(n, np.nan)
    in_target_pct = np.full(n, np.nan)
    overslip_pct = np.full(n, np.nan)
    underslip_pct = np.full(n, np.nan)
    intensity_mean = np.full(n, np.nan)

    for i, lap in enumerate(lap_list):
        lm = laps == lap
        lbm = lm & brake_mask
        lrm = lm & rb_mask
        brake_samps[i] = int(lbm.sum())
        rb_samps[i] = int(lrm.sum())
        if lm.any():
            lt_val[i] = laptime[lm].max()
            rb_cover[i] = lrm.sum() / lbm.sum() if lbm.sum() > 0 else np.nan
        if rb_samps[i] < MIN_SAMPLES_PER_LAP:
            continue

        err = sr_global[lrm] - SR_TARGET_BRAKE
        sr_mae[i] = np.nanmean(np.abs(err))
        sr_bias[i] = np.nanmean(err)
        in_target_pct[i] = 100.0 * np.mean(in_target_glob[lrm])
        overslip_pct[i] = 100.0 * np.mean(overslip_glob[lrm])
        underslip_pct[i] = 100.0 * np.mean(underslip_glob[lrm])
        intensity_mean[i] = np.nanmean(d["RB_intensityTarget"][lrm])

    valid_ok = np.isfinite(lt_val) & (rb_samps >= MIN_SAMPLES_PER_LAP) & np.isfinite(sr_mae)
    table = pl.DataFrame({
        "Lap": lap_list[valid_ok].astype(int),
        "LapTime [s]": np.round(lt_val[valid_ok], 3),
        "Brake samples": brake_samps[valid_ok].astype(int),
        "RB samples": rb_samps[valid_ok].astype(int),
        "RB active / brake [%]": np.round(rb_cover[valid_ok] * 100.0, 2),
        "SR MAE": np.round(sr_mae[valid_ok], 4),
        "SR Bias": np.round(sr_bias[valid_ok], 4),
        "In target [%]": np.round(in_target_pct[valid_ok], 2),
        "Overslip [%]": np.round(overslip_pct[valid_ok], 2),
        "Underslip [%]": np.round(underslip_pct[valid_ok], 2),
        "RB intensity target": np.round(intensity_mean[valid_ok], 3),
    })

    warnings: list[str] = []
    notes: list[str] = []
    if not use_rb_enable and np.any(brake_mask):
        notes.append(
            "RB KPIs inferred from braking phases because `RB_Enable` is always 0.0."
        )
    if not valid_ok.any():
        if not np.any(brake_mask):
            warnings.append(
                "RB tab has no data: no valid braking events passed the Brake/ax/vx filter."
            )
        else:
            warnings.append("No valid active RB laps for RB KPIs.")

    return {
        "lap_list": lap_list,
        "time": d["time"],
        "sr": sr,
        "rb_mask": rb_mask,
        "lt_val": lt_val,
        "valid_ok": valid_ok,
        "rb_cover": rb_cover,
        "sr_mae": sr_mae,
        "sr_bias": sr_bias,
        "in_target_pct": in_target_pct,
        "overslip_pct": overslip_pct,
        "underslip_pct": underslip_pct,
        "intensity_mean": intensity_mean,
        "table": table,
        "use_rb_enable": use_rb_enable,
        "notes": notes,
        "warnings": warnings,
    }


def _prepare_braking_regen_arrays(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return the signals needed to relate regen braking to vehicle response."""
    df = ensure_complete_laps_df(df)
    ax_col = _ax_signal(df.columns)
    vx_col = _vx_signal(df.columns)
    ay_col = _ay_signal(df.columns)
    yaw_col = _yaw_signal(df.columns)
    pitch_col = _pitch_rate_signal(df.columns)
    vy_col = _vy_signal(df.columns)
    wheel_speed_cols = [f"Est_VX{wheel}" for wheel in WHEELS]
    fz_cols = [f"Est_FZ{wheel}" for wheel in WHEELS]
    damper_cols = [f"Damp{wheel}" for wheel in WHEELS]

    required = [
        "TimeStamp", "laps", "laptime", "Brake", ax_col, vx_col,
        "Vbat", "Current", "RB_intensityTarget",
        "Est_SRFL", "Est_SRFR", "Est_SRRL", "Est_SRRR",
    ]
    optional = [
        "dist_km", "steering_actualPosRad",
        *MASTER_TORQUE_COLS.values(),
        *ACTUAL_TORQUE_COLS.values(),
        *wheel_speed_cols,
        *fz_cols,
        *damper_cols,
    ]
    if ay_col is not None:
        optional.append(ay_col)
    if yaw_col is not None:
        optional.append(yaw_col)
    if pitch_col is not None:
        optional.append(pitch_col)
    if vy_col is not None:
        optional.append(vy_col)

    cols = required + [c for c in optional if c in df.columns and c not in required]
    if COMPLETE_LAPS_MARKER in df.columns and COMPLETE_LAPS_MARKER not in cols:
        cols.append(COMPLETE_LAPS_MARKER)

    d = cols_to_numpy(df, cols)
    n = len(d["TimeStamp"])
    for c in optional:
        if c not in d:
            d[c] = np.full(n, np.nan)

    d["time"] = d["TimeStamp"] - d["TimeStamp"][0]
    d["ax"] = d.pop(ax_col)
    d["vx"] = d.pop(vx_col)
    d["ay"] = d.pop(ay_col) if ay_col is not None else np.full(n, np.nan)
    d["yaw_rate"] = d.pop(yaw_col) if yaw_col is not None else np.full(n, np.nan)
    d["pitch_rate"] = d.pop(pitch_col) if pitch_col is not None else np.full(n, np.nan)
    d["vy"] = d.pop(vy_col) if vy_col is not None else np.full(n, np.nan)
    d["distance_m"] = d["dist_km"] * 1000.0 if "dist_km" in d else np.full(n, np.nan)

    base_cols = ["time", "laps", "laptime", "Brake", "ax", "vx", "Vbat", "Current"]
    valid = np.all(np.stack([np.isfinite(d[c]) for c in base_cols], axis=1), axis=1)
    d = {k: v[valid] for k, v in d.items()}
    return exclude_lap0_and_last_lap(d)


def _braking_regen_analysis(
    df: pl.DataFrame,
    x_mode: str = "laps",
) -> tuple[list[go.Figure], dict]:
    d = _prepare_braking_regen_arrays(df)
    time_s = d["time"]
    dt_s = robust_dt(time_s)
    smooth_samples = max(1, int(round(0.05 / dt_s)))
    speed_ms = np.abs(d["vx"])
    decel_ms2 = np.maximum(0.0, -d["ax"])
    p_bat_w = d["Vbat"] * d["Current"]
    regen_power_w = np.maximum(0.0, -p_bat_w)
    regen_current_a = np.maximum(0.0, -d["Current"])
    target_current_a = np.maximum(0.0, -d["RB_intensityTarget"])

    master_torque = np.column_stack([d[c] for c in MASTER_TORQUE_COLS.values()])
    actual_torque = np.column_stack([d[c] for c in ACTUAL_TORQUE_COLS.values()])
    master_regen_wheel_nm = np.maximum(0.0, -master_torque)
    actual_regen_wheel_nm = np.maximum(0.0, -actual_torque)
    master_regen_nm = np.nansum(master_regen_wheel_nm, axis=1)
    actual_regen_nm = np.nansum(actual_regen_wheel_nm, axis=1)

    front_regen_nm = master_regen_wheel_nm[:, 0] + master_regen_wheel_nm[:, 1]
    rear_regen_nm = master_regen_wheel_nm[:, 2] + master_regen_wheel_nm[:, 3]
    left_regen_nm = master_regen_wheel_nm[:, 0] + master_regen_wheel_nm[:, 2]
    right_regen_nm = master_regen_wheel_nm[:, 1] + master_regen_wheel_nm[:, 3]
    regen_total_nm = front_regen_nm + rear_regen_nm
    front_share = np.divide(
        front_regen_nm,
        regen_total_nm,
        out=np.full_like(front_regen_nm, np.nan),
        where=regen_total_nm > 1e-6,
    )
    front_fz_n = d["Est_FZFL"] + d["Est_FZFR"]
    total_fz_n = front_fz_n + d["Est_FZRL"] + d["Est_FZRR"]
    fz_front_share = np.divide(
        front_fz_n,
        total_fz_n,
        out=np.full_like(front_fz_n, np.nan),
        where=np.abs(total_fz_n) > 1e-6,
    )
    lr_imbalance = np.divide(
        left_regen_nm - right_regen_nm,
        regen_total_nm,
        out=np.full_like(left_regen_nm, np.nan),
        where=regen_total_nm > 1e-6,
    )
    wheel_decel_ms2 = {
        wheel: _wheel_decel_from_speed(d[f"Est_VX{wheel}"], dt_s, smooth_samples)
        for wheel in WHEELS
    }
    front_damper = 0.5 * (d["DampFL"] + d["DampFR"])
    rear_damper = 0.5 * (d["DampRL"] + d["DampRR"])

    raw_brake = (d["Brake"] >= BRAKE_THRESHOLD) & (speed_ms >= MIN_SPEED)
    brake_event_mask = keep_min_duration_segments(raw_brake, MIN_EVENT_DURATION, dt_s)
    braking_response_mask = (
        brake_event_mask
        & (decel_ms2 >= max(0.2, -AX_BRAKE_THRESHOLD * 0.5))
    )

    if not brake_event_mask.any():
        raise ValueError("No valid brake events passed the Brake/vx filter.")

    distance_m = d["distance_m"].copy()
    valid_distance = np.isfinite(distance_m)
    distance_span_m = (
        float(np.nanmax(distance_m[valid_distance]) - np.nanmin(distance_m[valid_distance]))
        if valid_distance.any()
        else 0.0
    )
    if not valid_distance.any() or distance_span_m < 1.0:
        distance_m = np.cumsum(speed_ms * dt_s)
    else:
        finite_dist = np.isfinite(distance_m)
        if finite_dist.any():
            first = np.flatnonzero(finite_dist)[0]
            distance_m[:first] = distance_m[first]
            distance_m = np.maximum.accumulate(np.where(finite_dist, distance_m, distance_m[first]))

    steady_smooth_samples = max(1, int(round(STEADY_SMOOTH_WINDOW_S / dt_s)))
    ax_smooth = smooth_signal(d["ax"], steady_smooth_samples)
    jerk_ax = np.gradient(ax_smooth, dt_s)
    brake_smoothed = smooth_signal(d["Brake"], steady_smooth_samples)
    brake_std = _rolling_std(
        brake_smoothed,
        max(2, int(np.ceil(STEADY_MIN_DURATION_S / dt_s))),
    )
    steady_brake_mask = keep_min_duration_segments(
        brake_event_mask
        & np.isfinite(brake_std)
        & (brake_std <= STEADY_BRAKE_STD_THRESHOLD)
        & np.isfinite(jerk_ax)
        & (np.abs(jerk_ax) <= STEADY_JERK_THRESHOLD),
        STEADY_MIN_DURATION_S,
        dt_s,
    )
    lockup_info = _lockup_events(
        {wheel: d[f"Est_SR{wheel}"] for wheel in WHEELS},
        d["laps"],
        brake_event_mask,
        dt_s,
    )
    lockup_per_lap = _lockup_per_lap(
        d["laps"],
        d["laptime"],
        brake_event_mask,
        lockup_info["masks_by_wheel"],
        dt_s,
    )
    sr = {wheel: d[f"Est_SR{wheel}"] for wheel in WHEELS}

    events: list[dict[str, float | int]] = []
    yaw_event_peaks: list[float] = []
    yaw_event_integrals: list[float] = []
    beta_event_peaks: list[float] = []
    front_asym_events: list[float] = []
    rear_asym_events: list[float] = []
    sr_steady_event_max: list[float] = []
    sr_steady_by_wheel: dict[str, list[float]] = {wheel: [] for wheel in WHEELS}
    front_bias_mae_events: list[float] = []
    front_bias_signed_events: list[float] = []
    pitch_peak_events: list[float] = []
    dive_peak_events: list[float] = []
    straight_event_mask = np.zeros_like(brake_event_mask, dtype=bool)
    event_id = 0
    for start, end in _segment_bounds(brake_event_mask):
        if end - start + 1 < max(2, int(round(MIN_EVENT_DURATION / dt_s))):
            continue
        seg = slice(start, end + 1)
        event_id += 1
        entry_speed = float(speed_ms[start])
        exit_speed = float(speed_ms[end])
        delta_v = max(0.0, entry_speed - exit_speed)
        duration_s = float((end - start + 1) * dt_s)
        distance_delta = float(distance_m[end] - distance_m[start])
        if not np.isfinite(distance_delta) or distance_delta <= 0.0:
            distance_delta = float(np.nansum(speed_ms[seg]) * dt_s)
        energy_wh = float(np.nansum(regen_power_w[seg]) * dt_s / 3600.0)
        kinetic_wh = max(0.0, 0.5 * VEHICLE_MASS_KG * (entry_speed**2 - exit_speed**2) / 3600.0)
        recovery_pct = 100.0 * energy_wh / kinetic_wh if kinetic_wh > 1e-9 else np.nan
        target_m = np.isfinite(target_current_a[seg]) & (target_current_a[seg] > 1.0)
        current_error = regen_current_a[seg] - target_current_a[seg]
        if target_m.any():
            current_mae = float(np.nanmean(np.abs(current_error[target_m])))
            current_near = float(np.mean(np.abs(current_error[target_m]) <= np.maximum(5.0, 0.10 * target_current_a[seg][target_m])) * 100.0)
        else:
            current_mae = np.nan
            current_near = np.nan
        is_straight_event = _is_straight_brake_event(
            d["ay"],
            d["steering_actualPosRad"],
            start,
            end,
        )
        if is_straight_event:
            straight_event_mask[start:end + 1] = True
        yaw_metrics = _yaw_event_metrics(
            d["yaw_rate"],
            start,
            end,
            dt_s,
            is_straight_event,
        )
        beta_metrics = _beta_event_metrics(d["vx"], d["vy"], start, end)
        lr_asym = _lr_decel_asymmetry(wheel_decel_ms2, start, end)
        steady_metrics = _sr_steady_std(sr, steady_brake_mask, start, end)
        bias_metrics = _bias_vs_fz(front_share, fz_front_share, start, end)
        pitch_metrics = _pitch_dive_event(d["pitch_rate"], front_damper, rear_damper, start, end)

        lockup_counts = {}
        min_sr_by_wheel = {}
        event_lockups = 0
        event_worst_sr = np.nan
        fz_share_event_pct: dict[str, float] = {}
        brake_share_event_pct: dict[str, float] = {}
        mean_fz_event_n: dict[str, float] = {}
        mean_trq_event_nm: dict[str, float] = {}
        for wheel_idx, wheel in enumerate(WHEELS):
            wheel_lock_mask = lockup_info["masks_by_wheel"][wheel][seg]
            lockup_counts[wheel] = len(_segment_bounds(wheel_lock_mask))
            event_lockups += lockup_counts[wheel]
            wheel_sr = sr[wheel][seg]
            min_sr_by_wheel[wheel] = float(np.nanmin(wheel_sr)) if np.isfinite(wheel_sr).any() else np.nan
            if lockup_counts[wheel] > 0 and np.isfinite(min_sr_by_wheel[wheel]):
                event_worst_sr = (
                    min_sr_by_wheel[wheel]
                    if not np.isfinite(event_worst_sr)
                    else min(event_worst_sr, min_sr_by_wheel[wheel])
                )
            wheel_fz_share = np.divide(
                d[f"Est_FZ{wheel}"][seg],
                total_fz_n[seg],
                out=np.full(seg.stop - seg.start, np.nan),
                where=np.abs(total_fz_n[seg]) > 1e-6,
            )
            wheel_brake_share = np.divide(
                master_regen_wheel_nm[seg, wheel_idx],
                master_regen_nm[seg],
                out=np.full(seg.stop - seg.start, np.nan),
                where=np.abs(master_regen_nm[seg]) > 1e-6,
            )
            mean_fz_event_n[wheel] = _safe_mean(d[f"Est_FZ{wheel}"][seg])
            mean_trq_event_nm[wheel] = _safe_mean(master_regen_wheel_nm[seg, wheel_idx])
            fz_share_event_pct[wheel] = 100.0 * _safe_mean(wheel_fz_share)
            brake_share_event_pct[wheel] = 100.0 * _safe_mean(wheel_brake_share)

        for series, store in (
            (yaw_metrics["Yaw straight peak [rad/s]"], yaw_event_peaks),
            (yaw_metrics["Yaw straight integral [rad]"], yaw_event_integrals),
            (beta_metrics["Beta peak [deg]"], beta_event_peaks),
            (lr_asym["Front L-R decel asym [m/s²]"], front_asym_events),
            (lr_asym["Rear L-R decel asym [m/s²]"], rear_asym_events),
            (steady_metrics["Steady SR osc max [-]"], sr_steady_event_max),
            (bias_metrics["Front share vs Fz MAE [%]"], front_bias_mae_events),
            (bias_metrics["Front share vs Fz bias [%]"], front_bias_signed_events),
            (pitch_metrics["Pitch peak [rad/s]"], pitch_peak_events),
            (pitch_metrics["Dive asym peak [damper]"], dive_peak_events),
        ):
            if np.isfinite(series):
                store.append(float(series))
        for wheel in WHEELS:
            val = steady_metrics["steady_std_by_wheel"][wheel]
            if np.isfinite(val):
                sr_steady_by_wheel[wheel].append(float(val))

        events.append({
            "Event": event_id,
            "Lap": int(round(_safe_median(d["laps"][seg]))),
            "Start distance [m]": round(float(distance_m[start]), 1),
            "Duration [s]": round(duration_s, 3),
            "Distance [m]": round(distance_delta, 1),
            "Entry speed [m/s]": round(entry_speed, 2),
            "Exit speed [m/s]": round(exit_speed, 2),
            "Delta v [m/s]": round(delta_v, 2),
            "Mean brake [%]": round(_safe_mean(d["Brake"][seg]), 1),
            "Mean decel [m/s²]": round(_safe_mean(decel_ms2[seg]), 3),
            "Peak decel [m/s²]": round(float(np.nanmax(decel_ms2[seg])), 3),
            "Mean Master regen [Nm]": round(_safe_mean(master_regen_nm[seg]), 2),
            "Mean actual regen [Nm]": round(_safe_mean(actual_regen_nm[seg]), 2),
            "Recovered [Wh]": round(energy_wh, 3),
            "Kinetic lost [Wh]": round(kinetic_wh, 3),
            "Recovery [%]": round(recovery_pct, 2),
            "Wh/s braking": round(energy_wh / duration_s if duration_s > 1e-9 else np.nan, 3),
            "Wh/m braking": round(energy_wh / distance_delta if distance_delta > 1e-9 else np.nan, 4),
            "Wh per m/s": round(energy_wh / delta_v if delta_v > 1e-9 else np.nan, 3),
            "Front regen share [%]": round(_safe_mean(front_share[seg]) * 100.0, 2),
            "Current target MAE [A]": round(current_mae, 2),
            "Current near target [%]": round(current_near, 2),
            "Delay Master [ms]": round(_first_delay_ms(master_regen_nm, start, end, 2.0, dt_s), 1),
            "Delay current [ms]": round(_first_delay_ms(regen_current_a, start, end, 2.0, dt_s), 1),
            "Delay decel [ms]": round(_first_delay_ms(decel_ms2, start, end, 0.5, dt_s), 1),
            "Straight brake event": bool(is_straight_event),
            "Yaw straight peak [rad/s]": round(yaw_metrics["Yaw straight peak [rad/s]"], 3),
            "Yaw straight integral [rad]": round(yaw_metrics["Yaw straight integral [rad]"], 3),
            "Beta peak [deg]": round(beta_metrics["Beta peak [deg]"], 3),
            "Front L-R decel asym [m/s²]": round(lr_asym["Front L-R decel asym [m/s²]"], 3),
            "Rear L-R decel asym [m/s²]": round(lr_asym["Rear L-R decel asym [m/s²]"], 3),
            "Mean Fz FL [N]": round(mean_fz_event_n["FL"], 2),
            "Mean Fz FR [N]": round(mean_fz_event_n["FR"], 2),
            "Mean Fz RL [N]": round(mean_fz_event_n["RL"], 2),
            "Mean Fz RR [N]": round(mean_fz_event_n["RR"], 2),
            "Mean |Trq| FL [Nm]": round(mean_trq_event_nm["FL"], 2),
            "Mean |Trq| FR [Nm]": round(mean_trq_event_nm["FR"], 2),
            "Mean |Trq| RL [Nm]": round(mean_trq_event_nm["RL"], 2),
            "Mean |Trq| RR [Nm]": round(mean_trq_event_nm["RR"], 2),
            "Fz share FL [%]": round(fz_share_event_pct["FL"], 2),
            "Fz share FR [%]": round(fz_share_event_pct["FR"], 2),
            "Fz share RL [%]": round(fz_share_event_pct["RL"], 2),
            "Fz share RR [%]": round(fz_share_event_pct["RR"], 2),
            "Brake share FL [%]": round(brake_share_event_pct["FL"], 2),
            "Brake share FR [%]": round(brake_share_event_pct["FR"], 2),
            "Brake share RL [%]": round(brake_share_event_pct["RL"], 2),
            "Brake share RR [%]": round(brake_share_event_pct["RR"], 2),
            "Min SR FL": round(min_sr_by_wheel["FL"], 3),
            "Min SR FR": round(min_sr_by_wheel["FR"], 3),
            "Min SR RL": round(min_sr_by_wheel["RL"], 3),
            "Min SR RR": round(min_sr_by_wheel["RR"], 3),
            "Lockups": int(event_lockups),
            "Lockup worst SR": round(event_worst_sr, 3),
            "Steady SR osc mean [-]": round(steady_metrics["Steady SR osc mean [-]"], 4),
            "Steady SR osc max [-]": round(steady_metrics["Steady SR osc max [-]"], 4),
            "Front share mean [%]": round(bias_metrics["Front share mean [%]"], 2),
            "Fz front share mean [%]": round(bias_metrics["Fz front share mean [%]"], 2),
            "Front share vs Fz MAE [%]": round(bias_metrics["Front share vs Fz MAE [%]"], 2),
            "Front share vs Fz bias [%]": round(bias_metrics["Front share vs Fz bias [%]"], 2),
            "Pitch peak [rad/s]": round(pitch_metrics["Pitch peak [rad/s]"], 3),
            "Dive asym peak [damper]": round(pitch_metrics["Dive asym peak [damper]"], 3),
        })

    table = pl.DataFrame(events) if events else pl.DataFrame()
    if table.is_empty():
        raise ValueError("No valid brake events were long enough for RB event metrics.")

    response = braking_response_mask & np.isfinite(master_regen_nm) & (master_regen_nm > 1.0)
    target_mask = brake_event_mask & np.isfinite(target_current_a) & (target_current_a > 1.0)
    high_demand = brake_event_mask & (master_regen_nm >= np.nanpercentile(master_regen_nm[brake_event_mask], 75))
    if target_mask.any():
        current_shortfall = high_demand & (regen_current_a < 0.75 * target_current_a)
    else:
        power_ref = np.nanmedian(regen_power_w[brake_event_mask])
        current_shortfall = high_demand & (regen_power_w < power_ref)

    total_recovered_wh = float(table["Recovered [Wh]"].sum())
    total_kinetic_wh = float(table["Kinetic lost [Wh]"].sum())
    total_duration_s = float(table["Duration [s]"].sum())
    total_distance_m = float(table["Distance [m]"].sum())
    total_delta_v_ms = float(table["Delta v [m/s]"].sum())
    yaw_event_peaks_arr = np.array(yaw_event_peaks, dtype=float)
    yaw_event_integrals_arr = np.array(yaw_event_integrals, dtype=float)
    beta_event_peaks_arr = np.array(beta_event_peaks, dtype=float)
    front_asym_arr = np.array(front_asym_events, dtype=float)
    rear_asym_arr = np.array(rear_asym_events, dtype=float)
    sr_steady_event_arr = np.array(sr_steady_event_max, dtype=float)
    front_bias_mae_arr = np.array(front_bias_mae_events, dtype=float)
    front_bias_signed_arr = np.array(front_bias_signed_events, dtype=float)
    pitch_peak_arr = np.array(pitch_peak_events, dtype=float)
    dive_peak_arr = np.array(dive_peak_events, dtype=float)

    kpis = {
        "event_count": int(table.height),
        "mean_decel_ms2": float(table["Mean decel [m/s²]"].mean()),
        "peak_decel_ms2": float(table["Peak decel [m/s²]"].max()),
        "brake_decel_gain": _origin_slope(d["Brake"][braking_response_mask], decel_ms2[braking_response_mask]),
        "brake_decel_corr": _finite_corr(d["Brake"][braking_response_mask], decel_ms2[braking_response_mask]),
        "torque_decel_gain": _origin_slope(master_regen_nm[response], decel_ms2[response]),
        "torque_decel_corr": _finite_corr(master_regen_nm[response], decel_ms2[response]),
        "total_recovered_wh": total_recovered_wh,
        "mean_event_recovered_wh": float(table["Recovered [Wh]"].mean()),
        "recovery_efficiency_pct": 100.0 * total_recovered_wh / total_kinetic_wh if total_kinetic_wh > 1e-9 else np.nan,
        "regen_density_wh_s": total_recovered_wh / total_duration_s if total_duration_s > 1e-9 else np.nan,
        "regen_density_wh_m": total_recovered_wh / total_distance_m if total_distance_m > 1e-9 else np.nan,
        "regen_density_wh_dv": total_recovered_wh / total_delta_v_ms if total_delta_v_ms > 1e-9 else np.nan,
        "front_regen_share_pct": _safe_mean(front_share[brake_event_mask]) * 100.0,
        "current_target_mae_a": _safe_mean(np.abs(regen_current_a[target_mask] - target_current_a[target_mask])) if target_mask.any() else np.nan,
        "current_near_target_pct": float(np.mean(np.abs(regen_current_a[target_mask] - target_current_a[target_mask]) <= np.maximum(5.0, 0.10 * target_current_a[target_mask])) * 100.0) if target_mask.any() else np.nan,
        "acceptance_shortfall_pct": float(np.mean(current_shortfall[brake_event_mask]) * 100.0) if brake_event_mask.any() else np.nan,
        "delay_master_ms": float(table["Delay Master [ms]"].median()),
        "delay_current_ms": float(table["Delay current [ms]"].median()),
        "delay_decel_ms": float(table["Delay decel [ms]"].median()),
        "yaw_disturbance_p95_radps": _safe_p95(np.abs(d["yaw_rate"][straight_event_mask])),
        "lr_yaw_corr": _finite_corr(lr_imbalance[straight_event_mask], d["yaw_rate"][straight_event_mask]),
        "yaw_event_max_p95_radps": _safe_p95(yaw_event_peaks_arr),
        "yaw_event_max_worst_radps": _safe_max(yaw_event_peaks_arr),
        "yaw_event_integral_p95_rad": _safe_p95(yaw_event_integrals_arr),
        "yaw_event_integral_worst_rad": _safe_max(yaw_event_integrals_arr),
        "beta_peak_p95_deg": _safe_p95(beta_event_peaks_arr),
        "beta_peak_worst_deg": _safe_max(beta_event_peaks_arr),
        "front_lr_decel_asym_p95_ms2": _safe_p95(front_asym_arr),
        "rear_lr_decel_asym_p95_ms2": _safe_p95(rear_asym_arr),
        "lockup_events_total": int(lockup_info["count_total"]),
        "lockup_events_by_wheel": lockup_info["counts_by_wheel"],
        "lockup_events_by_lap": lockup_info["counts_by_lap"],
        "lockup_worst_sr": lockup_info["worst_min_sr"],
        "lockup_total_time_s": float(np.nansum(lockup_per_lap["total_lock_time_s"])),
        "lockup_mean_time_per_lap_s": _safe_mean(lockup_per_lap["total_lock_time_s"]),
        "lockup_peak_lap_time_s": _safe_max(lockup_per_lap["total_lock_time_s"]),
        "lockup_total_pct_brake": _safe_mean(lockup_per_lap["total_lock_pct_brake"]),
        "lockup_per_lap_table": lockup_per_lap["table"],
        "sr_steady_oscillation_p95": _safe_p95(sr_steady_event_arr),
        "sr_steady_oscillation_worst": _safe_max(sr_steady_event_arr),
        "sr_steady_oscillation_p95_by_wheel": {
            wheel: _safe_p95(np.array(vals, dtype=float))
            for wheel, vals in sr_steady_by_wheel.items()
        },
        "bias_vs_fz_mae_pct": _safe_p95(front_bias_mae_arr) if front_bias_mae_arr.size else np.nan,
        "bias_vs_fz_mae_mean_pct": _safe_mean(front_bias_mae_arr),
        "bias_vs_fz_signed_pct": _safe_mean(front_bias_signed_arr),
        "pitch_peak_p95_radps": _safe_p95(pitch_peak_arr),
        "pitch_peak_worst_radps": _safe_max(pitch_peak_arr),
        "dive_asym_p95_damper": _safe_p95(dive_peak_arr),
        "table": table,
        "warnings": [],
        "notes": [],
    }
    if not np.isfinite(d["yaw_rate"]).any():
        kpis["notes"].append("Yaw-event metrics skipped: no `VN_gz`/`AS_yaw_rate` channel.")
    if not np.isfinite(d["vy"]).any():
        kpis["notes"].append("Beta-event metrics skipped: no `Est_vyCOG`/`VN_vy` channel.")
    if not np.isfinite(front_fz_n).any():
        kpis["notes"].append("Front-bias-vs-Fz metrics skipped: missing `Est_FZ*` channels.")
    if not np.isfinite(d["pitch_rate"]).any():
        kpis["notes"].append("Pitch-event metrics skipped: no `VN_gy` channel.")
    if not np.isfinite(front_damper).any() or not np.isfinite(rear_damper).any():
        kpis["notes"].append("Dive asymmetry skipped: missing damper channels.")

    fig_torque_decel = make_dark_figure(
        title="Slip ratio vs longitudinal deceleration during braking",
        xlabel="Slip ratio [-]",
        ylabel="-ax [m/s²]",
    )
    for wheel in WHEELS:
        slip = sr[wheel][response]
        mask = np.isfinite(slip) & np.isfinite(decel_ms2[response])
        mask &= (slip >= -0.7) & (slip <= 0.1)
        if not np.any(mask):
            continue
        fig_torque_decel.add_trace(go.Scattergl(
            x=slip[mask],
            y=decel_ms2[response][mask],
            mode="markers",
            name=wheel,
            marker=dict(
                color=WHEEL_COLORS[wheel],
                size=4,
                opacity=0.45,
                symbol=WHEEL_SYMBOLS[wheel],
            ),
        ))
    fig_torque_decel.add_vrect(
        x0=SR_TARGET_BRAKE - DELTA_SR,
        x1=SR_TARGET_BRAKE + DELTA_SR,
        fillcolor="rgba(115, 217, 115, 0.08)",
        line_width=0,
    )
    fig_torque_decel.add_vline(
        x=SR_TARGET_BRAKE,
        line=dict(color="rgba(255,255,255,0.6)", dash="dash", width=1.4),
    )
    fig_min_sr = make_dark_figure(
        title="Minimum slip ratio by wheel across braking events",
        xlabel="Wheel",
        ylabel="Minimum SR [-]",
    )
    for wheel in WHEELS:
        col = f"Min SR {wheel}"
        valid_wheel = table[col].is_finite()
        if not valid_wheel.any():
            continue
        wheel_table = table.filter(valid_wheel)
        wheel_vals = wheel_table[col].to_numpy()
        wheel_events = wheel_table["Event"].to_numpy()
        wheel_laps = wheel_table["Lap"].to_numpy()
        fig_min_sr.add_trace(go.Box(
            x=np.full(wheel_vals.size, wheel),
            y=wheel_vals,
            name=wheel,
            marker=dict(color=WHEEL_COLORS[wheel], size=5, opacity=0.45),
            line=dict(color=WHEEL_COLORS[wheel]),
            boxmean=True,
            boxpoints="all",
            jitter=0.28,
            pointpos=0.0,
            customdata=np.column_stack([wheel_events, wheel_laps]),
            hovertemplate=(
                "Wheel %{x}"
                "<br>Min SR %{y:.3f}"
                "<br>Event %{customdata[0]:.0f}"
                "<br>Lap %{customdata[1]:.0f}"
                "<extra></extra>"
            ),
        ))
    fig_min_sr.add_trace(go.Scatter(
        x=list(WHEELS),
        y=[LOCKUP_SR_THRESHOLD] * len(WHEELS),
        mode="lines",
        name="Lockup threshold",
        line=dict(color="rgba(255,255,255,0.65)", dash="dash", width=1.3),
        hoverinfo="skip",
    ))

    x_axis_mode = x_mode if x_mode in ("laps", "laptime") else "laps"
    lock_x, lock_order, lock_xlabel = per_lap_axis(
        lockup_per_lap["lap_list"],
        lockup_per_lap["lap_time_vals_s"],
        x_axis_mode,
    )
    fig_lock_time = make_dark_figure(
        title="Wheel locking time per lap",
        xlabel=lock_xlabel,
        ylabel="Lock time [s]",
    )
    for wheel in WHEELS:
        wheel_vals = lockup_per_lap["lock_time_s_by_wheel"][wheel][lock_order]
        fig_lock_time.add_trace(go.Scatter(
            x=lock_x,
            y=wheel_vals,
            mode="lines+markers",
            name=wheel,
            line=dict(color=WHEEL_COLORS[wheel], width=2.0),
            marker=dict(
                color=WHEEL_COLORS[wheel],
                size=7,
                symbol=WHEEL_SYMBOLS[wheel],
            ),
            customdata=np.column_stack([
                lockup_per_lap["lap_list"][lock_order],
                lockup_per_lap["lock_events_by_wheel"][wheel][lock_order],
            ]),
            hovertemplate=(
                f"{wheel}"
                f"<br>{lock_xlabel} %{{x:.3f}}" if x_axis_mode == "laptime"
                else f"{wheel}<br>{lock_xlabel} %{{x:.0f}}"
            )
            + "<br>Lock time %{y:.3f} s"
            + "<br>Lock events %{customdata[1]:.0f}"
            + "<br>Lap %{customdata[0]:.0f}"
            + "<extra></extra>",
        ))

    fig_wheel_fz_torque = make_dark_figure(
        title="Braking torque vs vertical load by wheel across braking events",
        xlabel="Mean wheel Fz [N]",
        ylabel="Mean |Master regen torque| [Nm]",
    )
    finite_fz_vals: list[np.ndarray] = []
    finite_trq_vals: list[np.ndarray] = []
    for wheel in WHEELS:
        x_col = f"Mean Fz {wheel} [N]"
        y_col = f"Mean |Trq| {wheel} [Nm]"
        valid = table[x_col].is_finite() & table[y_col].is_finite()
        if not valid.any():
            continue
        wheel_table = table.filter(valid)
        x_vals = wheel_table[x_col].to_numpy()
        y_vals = wheel_table[y_col].to_numpy()
        finite_fz_vals.append(x_vals)
        finite_trq_vals.append(y_vals)
        fig_wheel_fz_torque.add_trace(go.Scattergl(
            x=x_vals,
            y=y_vals,
            mode="markers",
            name=wheel,
            marker=dict(
                color=WHEEL_COLORS[wheel],
                size=7,
                opacity=0.15,
            ),
            showlegend=False,
            customdata=np.column_stack([
                wheel_table["Event"].to_numpy(),
                wheel_table["Lap"].to_numpy(),
            ]),
            hovertemplate=(
                f"{wheel}"
                "<br>Mean Fz %{x:.1f} N"
                "<br>Mean |Trq| %{y:.2f} Nm"
                "<br>Event %{customdata[0]:.0f}"
                "<br>Lap %{customdata[1]:.0f}"
                "<extra></extra>"
            ),
        ))
        slope_nm_per_n = _origin_slope(x_vals, y_vals)
        if np.isfinite(slope_nm_per_n):
            x_line_max = float(np.nanmax(x_vals))
            fig_wheel_fz_torque.add_trace(go.Scatter(
                x=[0.0, x_line_max],
                y=[0.0, slope_nm_per_n * x_line_max],
                mode="lines",
                name=wheel,
                line=dict(color=WHEEL_COLORS[wheel], width=2.5),
                hovertemplate=(
                    f"{wheel}"
                    "<br>Fz %{x:.1f} N"
                    "<br>Fit |Trq| %{y:.2f} Nm"
                    "<extra></extra>"
                ),
            ))
    if finite_fz_vals and finite_trq_vals:
        all_fz_vals = np.concatenate(finite_fz_vals)
        all_trq_vals = np.concatenate(finite_trq_vals)
        total_fz_samples = [table[f"Mean Fz {wheel} [N]"].to_numpy() for wheel in WHEELS]
        total_trq_samples = [table[f"Mean |Trq| {wheel} [Nm]"].to_numpy() for wheel in WHEELS]
        total_fz_vals = np.concatenate(total_fz_samples)
        total_trq_vals = np.concatenate(total_trq_samples)
        slope_nm_per_n = _origin_slope(total_fz_vals, total_trq_vals)
        fz_min = max(0.0, float(np.nanmin(all_fz_vals)))
        fz_max = float(np.nanmax(all_fz_vals))
        fz_pad = max(25.0, 0.08 * (fz_max - fz_min))
        x0 = max(0.0, fz_min - fz_pad)
        x1 = fz_max + fz_pad
        trq_min = max(0.0, float(np.nanmin(all_trq_vals)))
        trq_max = float(np.nanmax(all_trq_vals))
        trq_pad = max(1.0, 0.10 * (trq_max - trq_min))
        if np.isfinite(slope_nm_per_n):
            fig_wheel_fz_torque.add_trace(go.Scatter(
                x=[0.0, x1],
                y=[0.0, slope_nm_per_n * x1],
                mode="lines",
                name="Fz-proportional reference",
                line=dict(color="rgba(255,255,255,0.55)", dash="dash", width=1.4),
                hoverinfo="skip",
                showlegend=False,
            ))
        fig_wheel_fz_torque.update_xaxes(range=[x0, x1])
        fig_wheel_fz_torque.update_yaxes(range=[0.0, max(trq_max + trq_pad, slope_nm_per_n * x1 if np.isfinite(slope_nm_per_n) else 0.0)])

    return [fig_torque_decel, fig_min_sr, fig_lock_time, fig_wheel_fz_torque], kpis


def rb_figs_kpis(
    df: pl.DataFrame,
    brake_mask: np.ndarray | None = None,
    x_mode: str = "laps",
) -> tuple[list[go.Figure], dict]:
    del brake_mask
    return _braking_regen_analysis(df, x_mode=x_mode)


def main() -> None:
    res = _compute_rb(_prepare_arrays_from_csv())
    if res["table"].is_empty():
        print("\n─── RB ───")
        print("No valid active RB laps for RB KPIs.")
    else:
        print("\n─── RB ───")
        print(res["table"])


# ═══════════════════════════════════════════════════════════════════════════════
# Function check  —  is RB delivering SR ≈ −0.20 and recovering energy?
# ═══════════════════════════════════════════════════════════════════════════════

def rb_function_kpis(df: pl.DataFrame) -> tuple[list[go.Figure], dict]:
    """Function-level check for Regenerative Braking.

    Pregunta: ¿está el RB consiguiendo SR ≈ −0.20 en frenada y cuánta
    energía recupera por vuelta?
    """
    df = ensure_complete_laps_df(df)
    ax_col = _ax_signal(df.columns)
    vx_col = _vx_signal(df.columns)
    needed = ["TimeStamp", "laps", "laptime", "Brake",
              "Vbat", "Current",
              "RB_Enable",
              "Est_SRFL", "Est_SRFR", "Est_SRRL", "Est_SRRR",
              ax_col, vx_col]
    arr = cols_to_numpy(df, needed)

    finite = np.all(np.stack([np.isfinite(arr[c]) for c in needed], axis=1), axis=1)
    arr = {c: v[finite] for c, v in arr.items()}
    laps_all = arr["laps"]
    keep = laps_all > 0
    arr = {c: v[keep] for c, v in arr.items()}
    if arr["TimeStamp"].size == 0:
        raise ValueError("No valid samples for RB function check.")
    lap_list_all = unique_laps(arr["laps"])
    if len(lap_list_all) == 0:
        raise ValueError("No laps available.")
    last_lap = lap_list_all.max()
    keep = arr["laps"] != last_lap
    arr = {c: v[keep] for c, v in arr.items()}

    time_s = arr["TimeStamp"] - arr["TimeStamp"][0]
    dt = robust_dt(time_s)

    raw_brake = (
        (arr["Brake"] >= BRAKE_THRESHOLD)
        & (arr[ax_col] <= AX_BRAKE_THRESHOLD)
        & (np.abs(arr[vx_col]) >= MIN_SPEED)
    )
    brake_mask = keep_min_duration_segments(raw_brake, MIN_EVENT_DURATION, dt)
    rb_active = brake_mask & (arr["RB_Enable"] == 1.0)
    if not rb_active.any():
        rb_active = brake_mask.copy()

    sr = {w: arr[f"Est_SR{w}"] for w in WHEELS}
    sr_mat = np.stack([sr[w] for w in WHEELS], axis=1)
    sr_global = np.nanmean(sr_mat, axis=1)

    lower_thr = SR_TARGET_BRAKE - DELTA_SR
    upper_thr = SR_TARGET_BRAKE + DELTA_SR

    s_g = sr_global[rb_active]
    pct_in_target = (
        float(((s_g >= lower_thr) & (s_g <= upper_thr)).mean() * 100.0) if s_g.size else np.nan
    )
    pct_lockup_risk = (
        float((s_g < lower_thr).mean() * 100.0) if s_g.size else np.nan
    )

    pct_in_target_w = {}
    for w in WHEELS:
        s = sr[w][rb_active]
        pct_in_target_w[w] = (
            float(((s >= lower_thr) & (s <= upper_thr)).mean() * 100.0) if s.size else np.nan
        )

    p_bat = arr["Vbat"] * arr["Current"]  # W (positive: drain, negative: regen)
    regen_mask = arr["Current"] < 0.0
    laps_arr = arr["laps"]
    lap_list = unique_laps(laps_arr)
    energy_wh = []
    lap_ids = []
    brake_time_pct = []
    for lap in lap_list:
        lm = laps_arr == lap
        lap_regen = lm & regen_mask
        if lap_regen.any():
            e_j = float(np.nansum(-p_bat[lap_regen]) * dt)  # J
            energy_wh.append(e_j / 3600.0)
        else:
            energy_wh.append(0.0)
        lap_ids.append(int(lap))
        n_lap = int(lm.sum())
        brake_time_pct.append(
            float(brake_mask[lm].sum() / n_lap * 100.0) if n_lap else np.nan
        )

    energy_wh_arr = np.array(energy_wh, dtype=float)
    brake_time_pct_arr = np.array(brake_time_pct, dtype=float)
    total_energy_wh = float(np.nansum(energy_wh_arr))
    median_energy_wh = float(np.nanmedian(energy_wh_arr)) if energy_wh_arr.size else np.nan

    if brake_mask.any():
        regen_coverage = float((rb_active.sum() / brake_mask.sum()) * 100.0)
    else:
        regen_coverage = np.nan

    # ── Fig 1: SR histogram per wheel while RB active ─────────────────────────
    fig_hist = make_dark_figure(
        title=f"SR distribution while braking (target = {SR_TARGET_BRAKE:+.2f})",
        xlabel="Slip ratio [-]",
        ylabel="Density",
    )
    for w in WHEELS:
        s = sr[w][rb_active]
        if s.size == 0:
            continue
        s = s[(s >= -0.7) & (s <= 0.3)]
        fig_hist.add_trace(go.Histogram(
            x=s, name=w, histnorm="probability density",
            marker=dict(color=WHEEL_COLORS[w]),
            opacity=0.55, nbinsx=80,
        ))
    fig_hist.update_layout(barmode="overlay")
    fig_hist.add_vrect(x0=lower_thr, x1=upper_thr,
                       fillcolor="rgba(115, 217, 115, 0.10)", line_width=0)
    fig_hist.add_vline(x=SR_TARGET_BRAKE,
                       line=dict(color="rgba(255,255,255,0.6)", dash="dash", width=1.4))

    # ── Fig 2: Energy recovered vs braking effort, per lap ────────────────────
    fig_brake_vs_energy = make_dark_figure(
        title="Energy recovered vs braking effort per lap",
        xlabel="Time braking [% of lap]",
        ylabel="Energy recovered [Wh]",
    )
    fig_brake_vs_energy.add_trace(go.Scatter(
        x=brake_time_pct_arr,
        y=energy_wh_arr,
        mode="markers+text",
        text=[f"L{lid}" for lid in lap_ids],
        textposition="top center",
        marker=dict(size=10, color="#73D973",
                    line=dict(width=1, color="#1A1A1A")),
        name="Lap",
    ))

    kpis = {
        "pct_in_target": pct_in_target,
        "pct_lockup_risk": pct_lockup_risk,
        "energy_recovered_wh_total": total_energy_wh,
        "energy_recovered_wh_median_lap": median_energy_wh,
        "regen_coverage_pct": regen_coverage,
        "pct_in_target_by_wheel": pct_in_target_w,
        "energy_per_lap_wh": dict(zip(lap_ids, energy_wh_arr.tolist())),
        "brake_time_pct_per_lap": dict(zip(lap_ids, brake_time_pct_arr.tolist())),
    }
    return [fig_hist, fig_brake_vs_energy], kpis


if __name__ == "__main__":
    main()
