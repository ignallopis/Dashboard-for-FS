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
    cols_to_numpy,
    ensure_complete_laps_df,
    exclude_lap0_and_last_lap,
    keep_min_duration_segments,
    make_dark_figure,
    robust_dt,
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

    required = [
        "TimeStamp", "laps", "laptime", "Brake", ax_col, vx_col,
        "Vbat", "Current", "RB_intensityTarget",
        "Est_SRFL", "Est_SRFR", "Est_SRRL", "Est_SRRR",
    ]
    optional = [
        "dist_km", "AS_yaw_rate", "steering_actualPosRad",
        *MASTER_TORQUE_COLS.values(),
        *ACTUAL_TORQUE_COLS.values(),
    ]
    if ay_col is not None:
        optional.append(ay_col)

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
    d["distance_m"] = d["dist_km"] * 1000.0 if "dist_km" in d else np.full(n, np.nan)

    base_cols = ["time", "laps", "laptime", "Brake", "ax", "vx", "Vbat", "Current"]
    valid = np.all(np.stack([np.isfinite(d[c]) for c in base_cols], axis=1), axis=1)
    d = {k: v[valid] for k, v in d.items()}
    return exclude_lap0_and_last_lap(d)


def _braking_regen_analysis(df: pl.DataFrame) -> tuple[list[go.Figure], dict]:
    d = _prepare_braking_regen_arrays(df)
    time_s = d["time"]
    dt_s = robust_dt(time_s)
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
    lr_imbalance = np.divide(
        left_regen_nm - right_regen_nm,
        regen_total_nm,
        out=np.full_like(left_regen_nm, np.nan),
        where=regen_total_nm > 1e-6,
    )

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

    events: list[dict[str, float | int]] = []
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

    straight_brake = (
        brake_event_mask
        & np.isfinite(d["AS_yaw_rate"])
        & (np.abs(d["steering_actualPosRad"]) < 0.05)
        & ((~np.isfinite(d["ay"])) | (np.abs(d["ay"]) < 2.0))
    )

    total_recovered_wh = float(table["Recovered [Wh]"].sum())
    total_kinetic_wh = float(table["Kinetic lost [Wh]"].sum())
    total_duration_s = float(table["Duration [s]"].sum())
    total_distance_m = float(table["Distance [m]"].sum())
    total_delta_v_ms = float(table["Delta v [m/s]"].sum())

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
        "yaw_disturbance_p95_radps": _safe_p95(np.abs(d["AS_yaw_rate"][straight_brake])),
        "lr_yaw_corr": _finite_corr(lr_imbalance[straight_brake], d["AS_yaw_rate"][straight_brake]),
        "table": table,
        "warnings": [],
        "notes": [],
    }

    fig_torque_decel = make_dark_figure(
        title="Master regen torque vs deceleration",
        xlabel="Total negative Master torque [Nm]",
        ylabel="-ax [m/s²]",
    )
    fig_torque_decel.add_trace(go.Scattergl(
        x=master_regen_nm[response],
        y=decel_ms2[response],
        mode="markers",
        name="Samples",
        marker=dict(color="rgba(77,179,242,0.45)", size=4),
    ))

    fig_energy = make_dark_figure(
        title="Recovered energy vs braking energy lost",
        xlabel="Kinetic energy lost [Wh]",
        ylabel="Recovered energy [Wh]",
    )
    fig_energy.add_trace(go.Scatter(
        x=table["Kinetic lost [Wh]"].to_numpy(),
        y=table["Recovered [Wh]"].to_numpy(),
        text=[f"E{int(e)} L{int(l)}" for e, l in zip(table["Event"], table["Lap"])],
        mode="markers+text",
        textposition="top center",
        name="Brake event",
        marker=dict(size=9, color="#73D973", line=dict(width=1, color="#1A1A1A")),
    ))

    return [fig_torque_decel, fig_energy], kpis


def rb_figs_kpis(
    df: pl.DataFrame,
    brake_mask: np.ndarray | None = None,
    x_mode: str = "laps",
) -> tuple[list[go.Figure], dict]:
    del brake_mask, x_mode
    return _braking_regen_analysis(df)


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
