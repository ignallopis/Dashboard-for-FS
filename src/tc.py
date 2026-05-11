"""tc.py
------
Traction Control (TC) KPIs — slip ratio regulation quality.

All metrics are computed during valid traction phases:
  throttle >= threshold AND ax >= threshold AND vx >= min_speed
  (optionally restricted to straights: |steering| <= threshold)

KPIs:
  1. Global SR MAE / Bias vs target (SR_target = 0.20)
  2. Per-wheel SR MAE / Bias + imbalance
  3. Time-in-target band [%]
  4. Overslip and underslip percentages
  5. Traction efficiency: mean ax when SR is in target
  6. Worst-wheel overslip
"""
from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import (
    COMPLETE_LAPS_MARKER,
    WHEEL_COLORS,
    WHEEL_SYMBOLS,
    add_lap_scatter,
    add_zero_line,
    cols_to_numpy,
    ensure_complete_laps_df,
    exclude_lap0_and_last_lap,
    keep_min_duration_segments,
    lap_dist_from_gps,
    make_dark_figure,
    per_lap_axis,
    robust_dt,
    unique_laps,
)

CSV_PATH = "data/run4_2025-08-24.csv"
WHEELS = ("FL", "FR", "RL", "RR")

# ── TC parameters ─────────────────────────────────────────────────────────────
SR_TARGET = 0.20
DELTA_SR = 0.05
THROTTLE_THRESHOLD = 10.0
FULL_THROTTLE_THRESHOLD = 95.0
AX_THRESHOLD = 0.50
MIN_SPEED = 4.0
STEERING_STRAIGHT = 0.08
USE_STRAIGHT_FILTER = True
MIN_EVENT_DURATION = 0.15
MIN_SAMPLES_PER_LAP = 40
TC_ENABLE_COL = "TCenable"
TC_ENABLE_MODE_COL = "TCEnableMode"
TC_TORQUE_COLS = ("TC_FL_MaxTrq", "TC_FR_MaxTrq", "TC_RL_MaxTrq", "TC_RR_MaxTrq")
TC_ANGVEL_COLS = ("TC_FL_MaxAngVel", "TC_FR_MaxAngVel", "TC_RL_MaxAngVel", "TC_RR_MaxAngVel")
MOTOR_VEL_COLS = ("FL_actualVelocity", "FR_actualVelocity", "RL_actualVelocity", "RR_actualVelocity")
MOTOR_TORQUE_COLS = ("FL_actualTorque", "FR_actualTorque", "RL_actualTorque", "RR_actualTorque")
TC_OPTIONAL_COLS = (
    TC_ENABLE_COL, TC_ENABLE_MODE_COL, "TCenableTorque", "TCenableVelocity",
    *TC_TORQUE_COLS, *TC_ANGVEL_COLS, *MOTOR_VEL_COLS, *MOTOR_TORQUE_COLS,
)


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


def _segment_bounds(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    cuts = np.where(np.diff(idx) > 1)[0] + 1
    return [(int(seg[0]), int(seg[-1])) for seg in np.split(idx, cuts) if seg.size]


def _prepare_arrays_from_df(
    df: pl.DataFrame,
    accel_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    df_eff = ensure_complete_laps_df(df)
    ax_col = _ax_signal(df.columns)
    vx_col = _vx_signal(df.columns)
    cols = [
        "TimeStamp", "laps", "laptime",
        "Throttle", "Steering", ax_col, vx_col,
        "Est_SRFL", "Est_SRFR", "Est_SRRL", "Est_SRRR",
    ]
    cols.extend(c for c in TC_OPTIONAL_COLS if c in df.columns)
    d = cols_to_numpy(df_eff, cols)
    if COMPLETE_LAPS_MARKER in df_eff.columns and COMPLETE_LAPS_MARKER not in d:
        d[COMPLETE_LAPS_MARKER] = cols_to_numpy(df_eff, [COMPLETE_LAPS_MARKER])[COMPLETE_LAPS_MARKER]
    s_m = lap_dist_from_gps(df_eff)
    d["s_m"] = s_m
    d["time"] = d["TimeStamp"] - d["TimeStamp"][0]
    d["ax"] = d.pop(ax_col)
    d["vx"] = d.pop(vx_col)
    if accel_mask is not None:
        d["__accel_mask"] = accel_mask.astype(float)
    return d


def _prepare_arrays_from_csv() -> dict[str, np.ndarray]:
    header = pl.read_csv(CSV_PATH, n_rows=1).columns
    ax_col = _ax_signal(header)
    vx_col = _vx_signal(header)
    d = _load([
        "TimeStamp", "laps", "laptime",
        "Throttle", "Steering", ax_col, vx_col,
        "Est_SRFL", "Est_SRFR", "Est_SRRL", "Est_SRRR",
    ])
    d["time"] = d["TimeStamp"] - d["TimeStamp"][0]
    d["ax"] = d.pop(ax_col)
    d["vx"] = d.pop(vx_col)
    return d


def _compute_tc(d: dict[str, np.ndarray]) -> dict:
    has_ext = "__accel_mask" in d
    data_keys = [k for k in d if not k.startswith("__")]
    valid = np.all(np.stack([np.isfinite(d[k]) for k in data_keys], axis=1), axis=1)
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)

    dt = robust_dt(d["time"])
    laps = d["laps"]
    laptime = d["laptime"]
    ax = d["ax"]
    vx = d["vx"]
    throttle = d["Throttle"]
    steering = d["Steering"]

    sr = {w: d[f"Est_SR{w}"] for w in WHEELS}

    if has_ext:
        traction_mask = d["__accel_mask"].astype(bool) & (np.abs(vx) >= MIN_SPEED)
    else:
        raw_traction = (
            (throttle >= THROTTLE_THRESHOLD)
            & (ax >= AX_THRESHOLD)
            & (np.abs(vx) >= MIN_SPEED)
        )
        if USE_STRAIGHT_FILTER:
            raw_traction &= np.abs(steering) <= STEERING_STRAIGHT
        traction_mask = keep_min_duration_segments(raw_traction, MIN_EVENT_DURATION, dt)

    sr_mat = np.stack([sr[w] for w in WHEELS], axis=1)
    sr_global = np.nanmean(sr_mat, axis=1)
    sr_worst = np.nanmax(sr_mat, axis=1)
    e_mat = sr_mat - SR_TARGET

    overslip_thr = SR_TARGET + DELTA_SR
    underslip_thr = SR_TARGET - DELTA_SR

    in_target_mat = (sr_mat >= underslip_thr) & (sr_mat <= overslip_thr)
    over_mat = sr_mat > overslip_thr
    under_mat = sr_mat < underslip_thr
    in_target_glob = (sr_global >= underslip_thr) & (sr_global <= overslip_thr)
    worst_over = sr_worst > overslip_thr

    lap_list = unique_laps(laps)
    n = len(lap_list)

    lt_val = np.full(n, np.nan)
    trac_samps = np.zeros(n, dtype=int)
    trac_cover = np.full(n, np.nan)

    mae_global = np.full(n, np.nan)
    bias_global = np.full(n, np.nan)
    mae_worst = np.full(n, np.nan)
    bias_worst = np.full(n, np.nan)

    mae_w = {w: np.full(n, np.nan) for w in WHEELS}
    bias_w = {w: np.full(n, np.nan) for w in WHEELS}
    imb_mae = np.full(n, np.nan)
    imb_bias = np.full(n, np.nan)

    in_tgt_pct_glob = np.full(n, np.nan)
    in_tgt_pct_w = {w: np.full(n, np.nan) for w in WHEELS}
    over_pct_glob = np.full(n, np.nan)
    under_pct_glob = np.full(n, np.nan)
    over_pct_w = {w: np.full(n, np.nan) for w in WHEELS}
    worst_over_pct = np.full(n, np.nan)

    ax_in_tgt_glob = np.full(n, np.nan)

    for i, lap in enumerate(lap_list):
        lm = laps == lap
        ltm = lm & traction_mask
        trac_samps[i] = int(ltm.sum())
        if lm.any():
            lt_val[i] = laptime[lm].max()
            trac_cover[i] = ltm.sum() / lm.sum()
        if trac_samps[i] < MIN_SAMPLES_PER_LAP:
            continue

        e_g = sr_global[ltm] - SR_TARGET
        mae_global[i] = np.nanmean(np.abs(e_g))
        bias_global[i] = np.nanmean(e_g)

        e_wst = sr_worst[ltm] - SR_TARGET
        mae_worst[i] = np.nanmean(np.abs(e_wst))
        bias_worst[i] = np.nanmean(e_wst)

        wheel_maes: list[float] = []
        wheel_biases: list[float] = []
        for j, w in enumerate(WHEELS):
            e_w = e_mat[ltm, j]
            mae_w[w][i] = np.nanmean(np.abs(e_w))
            bias_w[w][i] = np.nanmean(e_w)
            wheel_maes.append(mae_w[w][i])
            wheel_biases.append(bias_w[w][i])
        imb_mae[i] = max(wheel_maes) - min(wheel_maes)
        imb_bias[i] = max(wheel_biases) - min(wheel_biases)

        for j, w in enumerate(WHEELS):
            in_tgt_pct_w[w][i] = 100.0 * np.mean(in_target_mat[ltm, j])
            over_pct_w[w][i] = 100.0 * np.mean(over_mat[ltm, j])
        in_tgt_pct_glob[i] = np.mean([in_tgt_pct_w[w][i] for w in WHEELS])
        over_pct_glob[i] = np.mean([over_pct_w[w][i] for w in WHEELS])
        under_pct_glob[i] = 100.0 * np.mean(np.all(under_mat[ltm], axis=1))
        worst_over_pct[i] = 100.0 * np.mean(worst_over[ltm])

        ax_in = ax[ltm & in_target_glob]
        if ax_in.size > 0:
            ax_in_tgt_glob[i] = np.nanmean(ax_in)

    base_ok = np.isfinite(lt_val) & (trac_samps >= MIN_SAMPLES_PER_LAP)
    glob_ok = base_ok & np.isfinite(mae_global)
    wheel_ok = base_ok & np.all(
        np.stack([np.isfinite(mae_w[w]) for w in WHEELS], axis=1),
        axis=1,
    )
    eff_ok = base_ok & np.isfinite(ax_in_tgt_glob)

    table = pl.DataFrame({
        "Lap": lap_list[base_ok].astype(int),
        "LapTime [s]": np.round(lt_val[base_ok], 3),
        "Traction samples": trac_samps[base_ok].astype(int),
        "Coverage [%]": np.round(trac_cover[base_ok] * 100.0, 2),
        "Global MAE": np.round(mae_global[base_ok], 4),
        "Global Bias": np.round(bias_global[base_ok], 4),
        "Worst MAE": np.round(mae_worst[base_ok], 4),
        "Worst Bias": np.round(bias_worst[base_ok], 4),
        "In target [%]": np.round(in_tgt_pct_glob[base_ok], 2),
        "Overslip [%]": np.round(over_pct_glob[base_ok], 2),
        "Underslip [%]": np.round(under_pct_glob[base_ok], 2),
        "Worst overslip [%]": np.round(worst_over_pct[base_ok], 2),
        "ax in target [m/s²]": np.round(ax_in_tgt_glob[base_ok], 3),
        "MAE FL": np.round(mae_w["FL"][base_ok], 4),
        "MAE FR": np.round(mae_w["FR"][base_ok], 4),
        "MAE RL": np.round(mae_w["RL"][base_ok], 4),
        "MAE RR": np.round(mae_w["RR"][base_ok], 4),
        "Bias FL": np.round(bias_w["FL"][base_ok], 4),
        "Bias FR": np.round(bias_w["FR"][base_ok], 4),
        "Bias RL": np.round(bias_w["RL"][base_ok], 4),
        "Bias RR": np.round(bias_w["RR"][base_ok], 4),
        "MAE imbalance": np.round(imb_mae[base_ok], 4),
        "Bias imbalance": np.round(imb_bias[base_ok], 4),
    })

    warnings: list[str] = []
    if not base_ok.any():
        warnings.append("No valid traction laps for TC KPIs.")

    return {
        "lap_list": lap_list,
        "time": d["time"],
        "sr": sr,
        "traction_mask": traction_mask,
        "lt_val": lt_val,
        "trac_samps": trac_samps,
        "trac_cover": trac_cover,
        "mae_global": mae_global,
        "bias_global": bias_global,
        "mae_worst": mae_worst,
        "bias_worst": bias_worst,
        "mae_w": mae_w,
        "bias_w": bias_w,
        "imb_mae": imb_mae,
        "imb_bias": imb_bias,
        "in_tgt_pct_glob": in_tgt_pct_glob,
        "in_tgt_pct_w": in_tgt_pct_w,
        "over_pct_glob": over_pct_glob,
        "under_pct_glob": under_pct_glob,
        "over_pct_w": over_pct_w,
        "worst_over_pct": worst_over_pct,
        "ax_in_tgt_glob": ax_in_tgt_glob,
        "base_ok": base_ok,
        "glob_ok": glob_ok,
        "wheel_ok": wheel_ok,
        "eff_ok": eff_ok,
        "table": table,
        "warnings": warnings,
    }


def _build_tc_figures(res: dict, x_mode: str = "laps") -> list[go.Figure]:
    lap_list = res["lap_list"]
    lt_val = res["lt_val"]
    glob_ok = res["glob_ok"]
    wheel_ok = res["wheel_ok"]
    base_ok = res["base_ok"]
    eff_ok = res["eff_ok"]

    figs: list[go.Figure] = []

    x_glob, order_glob, xlabel = per_lap_axis(lap_list[glob_ok], lt_val[glob_ok], x_mode) if glob_ok.any() else (np.array([]), np.array([], dtype=int), "Lap")
    fig = make_dark_figure(f"Global SR MAE vs {'Lap Time' if x_mode == 'laptime' else 'Lap'}", xlabel, "MAE SR global")
    if glob_ok.any():
        add_lap_scatter(fig, x_glob, res["mae_global"][glob_ok][order_glob], lap_list[glob_ok][order_glob])
        if x_mode == "laps":
            fig.update_xaxes(tickvals=np.sort(lap_list[glob_ok].astype(int)))
    figs.append(fig)

    fig = make_dark_figure(f"Global SR Bias vs {'Lap Time' if x_mode == 'laptime' else 'Lap'}", xlabel, "Bias SR global")
    if glob_ok.any():
        add_lap_scatter(
            fig,
            x_glob,
            res["bias_global"][glob_ok][order_glob],
            lap_list[glob_ok][order_glob],
            color="#F27070",
        )
        add_zero_line(fig, x_glob)
        if x_mode == "laps":
            fig.update_xaxes(tickvals=np.sort(lap_list[glob_ok].astype(int)))
    figs.append(fig)

    x_wheel, order_wheel, xlabel_wheel = per_lap_axis(lap_list[wheel_ok], lt_val[wheel_ok], x_mode) if wheel_ok.any() else (np.array([]), np.array([], dtype=int), "Lap")
    fig = make_dark_figure(f"Per-Wheel SR MAE vs {'Lap Time' if x_mode == 'laptime' else 'Lap'}", xlabel_wheel, "MAE SR per wheel")
    if wheel_ok.any():
        for w in WHEELS:
            add_lap_scatter(
                fig,
                x_wheel,
                res["mae_w"][w][wheel_ok][order_wheel],
                lap_list[wheel_ok][order_wheel],
                name=w,
                color=WHEEL_COLORS[w],
                symbol=WHEEL_SYMBOLS[w],
            )
        if x_mode == "laps":
            fig.update_xaxes(tickvals=np.sort(lap_list[wheel_ok].astype(int)))
    figs.append(fig)

    fig = make_dark_figure(f"Per-Wheel SR Bias vs {'Lap Time' if x_mode == 'laptime' else 'Lap'}", xlabel_wheel, "Bias SR per wheel")
    if wheel_ok.any():
        for w in WHEELS:
            add_lap_scatter(
                fig,
                x_wheel,
                res["bias_w"][w][wheel_ok][order_wheel],
                lap_list[wheel_ok][order_wheel],
                name=w,
                color=WHEEL_COLORS[w],
                symbol=WHEEL_SYMBOLS[w],
            )
        add_zero_line(fig, x_wheel)
        if x_mode == "laps":
            fig.update_xaxes(tickvals=np.sort(lap_list[wheel_ok].astype(int)))
    figs.append(fig)

    ok_pct = base_ok & np.isfinite(res["in_tgt_pct_glob"])
    x_pct, order_pct, xlabel_pct = per_lap_axis(lap_list[ok_pct], lt_val[ok_pct], x_mode) if ok_pct.any() else (np.array([]), np.array([], dtype=int), "Lap")
    fig = make_dark_figure(f"Time in SR Target vs {'Lap Time' if x_mode == 'laptime' else 'Lap'}", xlabel_pct, "% of traction time")
    if ok_pct.any():
        add_lap_scatter(
            fig, x_pct, res["in_tgt_pct_glob"][ok_pct][order_pct], lap_list[ok_pct][order_pct],
            name="In target", color="#73D973",
        )
        add_lap_scatter(
            fig, x_pct, res["over_pct_glob"][ok_pct][order_pct], lap_list[ok_pct][order_pct],
            name="Overslip", color="#F27070", symbol="square",
        )
        add_lap_scatter(
            fig, x_pct, res["under_pct_glob"][ok_pct][order_pct], lap_list[ok_pct][order_pct],
            name="Underslip", color="#4DB3F2", symbol="diamond",
        )
        if x_mode == "laps":
            fig.update_xaxes(tickvals=np.sort(lap_list[ok_pct].astype(int)))
    figs.append(fig)

    x_eff, order_eff, xlabel_eff = per_lap_axis(lap_list[eff_ok], lt_val[eff_ok], x_mode) if eff_ok.any() else (np.array([]), np.array([], dtype=int), "Lap")
    fig = make_dark_figure(
        f"Traction Efficiency vs {'Lap Time' if x_mode == 'laptime' else 'Lap'}", xlabel_eff, "Mean ax when SR in target [m/s²]",
    )
    if eff_ok.any():
        add_lap_scatter(
            fig, x_eff, res["ax_in_tgt_glob"][eff_ok][order_eff], lap_list[eff_ok][order_eff],
            color="#73D973",
        )
        if x_mode == "laps":
            fig.update_xaxes(tickvals=np.sort(lap_list[eff_ok].astype(int)))
    figs.append(fig)

    fig = make_dark_figure(
        "Per-Wheel SR Error vs Time (traction only)",
        "Time [s]",
        "SR error (SR - target)",
    )
    t_trac = res["time"][res["traction_mask"]]
    for w in WHEELS:
        fig.add_trace(go.Scatter(
            x=t_trac,
            y=res["sr"][w][res["traction_mask"]] - SR_TARGET,
            mode="lines",
            name=w,
            line=dict(color=WHEEL_COLORS[w], width=1.0),
        ))
    fig.add_hline(y=0.0, line=dict(color="rgba(200,200,200,0.5)", dash="dash"))
    fig.add_hline(y=DELTA_SR, line=dict(color="rgba(200,200,200,0.3)", dash="dot"))
    fig.add_hline(y=-DELTA_SR, line=dict(color="rgba(200,200,200,0.3)", dash="dot"))
    figs.append(fig)

    return figs


def tc_figs_kpis(
    df: pl.DataFrame,
    accel_mask: np.ndarray | None = None,
    x_mode: str = "laps",
) -> tuple[list[go.Figure], dict]:
    """Dashboard API for TC figures and KPIs on a single run.

    Args:
        df:          Telemetry DataFrame (already filtered by load_data).
        accel_mask:  Optional boolean mask (same length as *df*) marking
                     acceleration phase samples. When provided, replaces the
                     internal throttle/ax/steering filter. Falls back to the
                     built-in heuristic when None.
    """
    res = _compute_tc(_prepare_arrays_from_df(df, accel_mask))
    base_ok = res["base_ok"]
    glob_ok = res["glob_ok"]
    wheel_ok = res["wheel_ok"]
    eff_ok = res["eff_ok"]

    kpis = {
        "valid_laps": int(base_ok.sum()),
        "mean_global_mae": float(np.nanmean(res["mae_global"][glob_ok])) if glob_ok.any() else np.nan,
        "mean_global_bias": float(np.nanmean(res["bias_global"][glob_ok])) if glob_ok.any() else np.nan,
        "mean_worst_mae": float(np.nanmean(res["mae_worst"][glob_ok])) if glob_ok.any() else np.nan,
        "mean_in_target_pct": (
            float(np.nanmean(res["in_tgt_pct_glob"][base_ok])) if base_ok.any() else np.nan
        ),
        "mean_overslip_pct": (
            float(np.nanmean(res["over_pct_glob"][base_ok])) if base_ok.any() else np.nan
        ),
        "mean_underslip_pct": (
            float(np.nanmean(res["under_pct_glob"][base_ok])) if base_ok.any() else np.nan
        ),
        "mean_worst_over_pct": (
            float(np.nanmean(res["worst_over_pct"][base_ok])) if base_ok.any() else np.nan
        ),
        "mean_ax_in_target": (
            float(np.nanmean(res["ax_in_tgt_glob"][eff_ok])) if eff_ok.any() else np.nan
        ),
        "mean_traction_coverage_pct": (
            float(np.nanmean(res["trac_cover"][base_ok]) * 100.0) if base_ok.any() else np.nan
        ),
        "mae_by_wheel": {
            w: float(np.nanmean(res["mae_w"][w][wheel_ok])) if wheel_ok.any() else np.nan
            for w in WHEELS
        },
        "bias_by_wheel": {
            w: float(np.nanmean(res["bias_w"][w][wheel_ok])) if wheel_ok.any() else np.nan
            for w in WHEELS
        },
        "table": res["table"],
        "warnings": res["warnings"],
    }
    return _build_tc_figures(res, x_mode=x_mode), kpis


def _print_tc_summary(kpis: dict) -> None:
    table = kpis["table"]
    if table.is_empty():
        print("\n─── TC ───")
        print("No valid traction laps for TC KPIs.")
        return

    print("\n─── TC ───")
    print(table)


def main() -> None:
    res = _compute_tc(_prepare_arrays_from_csv())
    kpis = {
        "table": res["table"],
        "warnings": res["warnings"],
    }
    _print_tc_summary(kpis)
    for fig in _build_tc_figures(res):
        fig.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Function check  —  is TC delivering its job (SR ≈ +0.20 in acceleration)?
# ═══════════════════════════════════════════════════════════════════════════════

def _tc_armed_mask(d: dict[str, np.ndarray]) -> np.ndarray:
    return (
        (d["Throttle"] >= THROTTLE_THRESHOLD)
        & (np.abs(d["vx"]) >= MIN_SPEED)
        & (d["ax"] >= AX_THRESHOLD)
    )


def _bool_signal_mask(d: dict[str, np.ndarray], column: str) -> np.ndarray | None:
    if column not in d:
        return None
    return np.isfinite(d[column]) & (d[column] == 1.0)


def _binned_sr_error_by_distance(
    s_m: np.ndarray,
    sr_mat: np.ndarray,
    mask: np.ndarray,
    n_bins: int = 160,
) -> tuple[np.ndarray, np.ndarray]:
    s_eval = s_m[mask]
    sr_eval = sr_mat[mask]
    valid = np.isfinite(s_eval) & np.all(np.isfinite(sr_eval), axis=1)
    if valid.sum() < 2:
        return np.array([], dtype=float), np.empty((len(WHEELS), 0), dtype=float)

    s_eval = s_eval[valid]
    sr_eval = sr_eval[valid]
    s_min = float(np.nanmin(s_eval))
    s_max = float(np.nanmax(s_eval))
    if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max <= s_min:
        return np.array([], dtype=float), np.empty((len(WHEELS), 0), dtype=float)

    bins = np.linspace(s_min, s_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bin_idx = np.digitize(s_eval, bins) - 1
    z = np.full((len(WHEELS), n_bins), np.nan, dtype=float)
    for j in range(len(WHEELS)):
        for i in range(n_bins):
            vals = sr_eval[bin_idx == i, j]
            if vals.size:
                z[j, i] = float(np.nanmedian(vals - SR_TARGET))
    return centers, z


def _tc_visual_figures(
    d: dict[str, np.ndarray],
    sr: dict[str, np.ndarray],
    sr_mat: np.ndarray,
    eval_mask: np.ndarray,
    tc_enable_mask: np.ndarray,
    full_throttle_accel: np.ndarray,
    over_thr: float,
    under_thr: float,
) -> list[go.Figure]:
    s_m = d["s_m"]
    sr_max = np.nanmax(sr_mat, axis=1)
    sr_min = np.nanmin(sr_mat, axis=1)
    any_high = sr_max > over_thr
    all_low = np.all(sr_mat < under_thr, axis=1)

    torque_min = np.full(len(s_m), np.nan, dtype=float)
    present_torque_cols = [c for c in TC_TORQUE_COLS if c in d]
    if present_torque_cols:
        torque_mat = np.stack([d[c] for c in present_torque_cols], axis=1)
        torque_min = np.nanmin(torque_mat, axis=1)

    fig_track = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=(
            "Slip ratio by wheel vs distance",
            "Median SR error by distance (SR - 0.20)",
            "TC state and torque cut",
        ),
        specs=[[{"type": "scatter"}], [{"type": "heatmap"}], [{"type": "scatter"}]],
    )

    plot_mask = eval_mask & np.isfinite(s_m)
    for w in WHEELS:
        fig_track.add_trace(
            go.Scattergl(
                x=s_m[plot_mask],
                y=sr[w][plot_mask],
                mode="markers",
                name=w,
                marker=dict(color=WHEEL_COLORS[w], size=3, opacity=0.45, symbol=WHEEL_SYMBOLS[w]),
                hovertemplate="Distance %{x:.1f} m<br>SR %{y:.3f}<extra>" + w + "</extra>",
            ),
            row=1,
            col=1,
        )
    fig_track.add_hrect(
        y0=under_thr,
        y1=over_thr,
        fillcolor="rgba(115, 217, 115, 0.12)",
        line_width=0,
        row=1,
        col=1,
    )
    fig_track.add_hline(y=SR_TARGET, line=dict(color="rgba(255,255,255,0.65)", dash="dash", width=1.3), row=1, col=1)

    heat_x, heat_z = _binned_sr_error_by_distance(s_m, sr_mat, eval_mask)
    if heat_x.size:
        fig_track.add_trace(
            go.Heatmap(
                x=heat_x,
                y=list(WHEELS),
                z=heat_z,
                zmin=-0.20,
                zmax=0.20,
                colorscale=[
                    [0.0, "#4DB3F2"],
                    [0.42, "#253447"],
                    [0.50, "#73D973"],
                    [0.58, "#473525"],
                    [1.0, "#F27070"],
                ],
                colorbar=dict(title="SR error", len=0.28, y=0.43),
                hovertemplate="Distance %{x:.1f} m<br>%{y}<br>SR error %{z:+.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    status_mask = full_throttle_accel & np.isfinite(s_m)
    fig_track.add_trace(
        go.Scattergl(
            x=s_m[status_mask],
            y=np.where(tc_enable_mask[status_mask], 1.0, 0.0),
            mode="markers",
            name="TC enable",
            marker=dict(color="#EBEBEB", size=3, opacity=0.35),
            hovertemplate="Distance %{x:.1f} m<br>TC enable %{y:.0f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig_track.add_trace(
        go.Scattergl(
            x=s_m[status_mask & any_high],
            y=np.full((status_mask & any_high).sum(), 1.12),
            mode="markers",
            name="Any wheel SR > 0.25",
            marker=dict(color="#F27070", size=5, opacity=0.75, symbol="x"),
            hovertemplate="Distance %{x:.1f} m<br>High SR<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig_track.add_trace(
        go.Scattergl(
            x=s_m[status_mask & all_low],
            y=np.full((status_mask & all_low).sum(), -0.12),
            mode="markers",
            name="All wheels SR < 0.15",
            marker=dict(color="#4DB3F2", size=4, opacity=0.45, symbol="triangle-down"),
            hovertemplate="Distance %{x:.1f} m<br>Low SR<extra></extra>",
        ),
        row=3,
        col=1,
    )
    if np.any(np.isfinite(torque_min)):
        scaled_torque = np.clip(torque_min / max(torque_m := abs(np.nanmin(torque_min)), 1.0), -1.0, 0.0)
        fig_track.add_trace(
            go.Scattergl(
                x=s_m[status_mask],
                y=scaled_torque[status_mask],
                mode="markers",
                name="TC torque cut (scaled)",
                marker=dict(color="#F28C40", size=3, opacity=0.45),
                hovertemplate="Distance %{x:.1f} m<br>Scaled cut %{y:.2f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    fig_track.update_layout(
        title=dict(text="TC target tracking over the lap", font=dict(size=14, color="#EBEBEB")),
        paper_bgcolor="#141417",
        plot_bgcolor="#141417",
        font=dict(color="#EBEBEB", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        height=820,
    )
    fig_track.update_xaxes(title_text="Distance from start line [m]", row=3, col=1, gridcolor="rgba(128,128,128,0.2)")
    fig_track.update_yaxes(title_text="Slip ratio [-]", row=1, col=1, range=[-0.05, 0.45], gridcolor="rgba(128,128,128,0.2)")
    fig_track.update_yaxes(title_text="Wheel", row=2, col=1, gridcolor="rgba(128,128,128,0.2)")
    fig_track.update_yaxes(title_text="State", row=3, col=1, range=[-1.05, 1.25], gridcolor="rgba(128,128,128,0.2)")

    fig_dist = make_dark_figure(
        title=f"SR distribution while TC armed at full throttle (target = +{SR_TARGET:.2f})",
        xlabel="Slip ratio [-]",
        ylabel="Density",
    )
    for w in WHEELS:
        s = sr[w][eval_mask]
        if s.size == 0:
            continue
        s = s[(s >= -0.5) & (s <= 0.8)]
        fig_dist.add_trace(go.Histogram(
            x=s,
            name=w,
            histnorm="probability density",
            marker=dict(color=WHEEL_COLORS[w]),
            opacity=0.55,
            nbinsx=80,
        ))
    fig_dist.update_layout(barmode="overlay")
    fig_dist.add_vrect(x0=under_thr, x1=over_thr, fillcolor="rgba(115, 217, 115, 0.10)", line_width=0)
    fig_dist.add_vline(x=SR_TARGET, line=dict(color="rgba(255,255,255,0.6)", dash="dash", width=1.4))

    fig_scatter = make_dark_figure(
        title="Slip ratio vs longitudinal acceleration (TC armed at full throttle)",
        xlabel="Slip ratio [-]",
        ylabel="ax [m/s²]",
    )
    if eval_mask.any():
        ax_eval = d["ax"][eval_mask]
        for w in WHEELS:
            s = sr[w][eval_mask]
            mask = (s >= -0.2) & (s <= 0.6) & np.isfinite(ax_eval)
            if not mask.any():
                continue
            fig_scatter.add_trace(go.Scattergl(
                x=s[mask], y=ax_eval[mask],
                mode="markers", name=w,
                marker=dict(color=WHEEL_COLORS[w], size=4, opacity=0.45,
                            symbol=WHEEL_SYMBOLS[w]),
            ))
        fig_scatter.add_vline(x=SR_TARGET, line=dict(color="rgba(255,255,255,0.6)", dash="dash", width=1.4))
        fig_scatter.add_vrect(x0=under_thr, x1=over_thr, fillcolor="rgba(115, 217, 115, 0.08)", line_width=0)

    return [fig_track, fig_dist, fig_scatter]


def _tc_objective_figures(
    d: dict[str, np.ndarray],
    sr: dict[str, np.ndarray],
    sr_mat: np.ndarray,
    eval_mask: np.ndarray,
    over_thr: float,
    under_thr: float,
) -> list[go.Figure]:
    """Aggregated TC objective plots, independent of distance."""
    figs: list[go.Figure] = []

    fig_box = make_dark_figure(
        title="Maximum slip ratio by wheel across TC events",
        xlabel="Wheel",
        ylabel="Maximum SR [-]",
    )
    event_rows: list[dict[str, float | int]] = []
    for event_id, (start, end) in enumerate(_segment_bounds(eval_mask), start=1):
        seg = slice(start, end + 1)
        lap = int(round(np.nanmedian(d["laps"][seg])))
        for w in WHEELS:
            vals = sr[w][seg]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            event_rows.append({
                "wheel": w,
                "event": event_id,
                "lap": lap,
                "max_sr": float(np.nanmax(vals)),
            })

    for w in WHEELS:
        wheel_rows = [row for row in event_rows if row["wheel"] == w]
        if not wheel_rows:
            continue
        wheel_vals = np.array([float(row["max_sr"]) for row in wheel_rows], dtype=float)
        wheel_events = np.array([int(row["event"]) for row in wheel_rows], dtype=int)
        wheel_laps = np.array([int(row["lap"]) for row in wheel_rows], dtype=int)
        fig_box.add_trace(go.Box(
            x=np.full(wheel_vals.size, w),
            y=wheel_vals,
            name=w,
            marker=dict(color=WHEEL_COLORS[w], size=5, opacity=0.45),
            line=dict(color=WHEEL_COLORS[w]),
            boxmean=True,
            boxpoints="all",
            jitter=0.28,
            pointpos=0.0,
            customdata=np.column_stack([wheel_events, wheel_laps]),
            hovertemplate=(
                "Wheel %{x}"
                "<br>Max SR %{y:.3f}"
                "<br>Event %{customdata[0]:.0f}"
                "<br>Lap %{customdata[1]:.0f}"
                "<extra></extra>"
            ),
        ))
    fig_box.add_trace(go.Scatter(
        x=list(WHEELS),
        y=[over_thr] * len(WHEELS),
        mode="lines",
        name="Overslip threshold",
        line=dict(color="rgba(255,255,255,0.65)", dash="dash", width=1.3),
        hoverinfo="skip",
    ))
    fig_box.update_yaxes(range=[-0.02, 0.45])
    figs.append(fig_box)

    low_pct: list[float] = []
    in_pct: list[float] = []
    high_pct: list[float] = []
    for w in WHEELS:
        s = sr[w][eval_mask]
        s = s[np.isfinite(s)]
        if s.size == 0:
            low_pct.append(np.nan)
            in_pct.append(np.nan)
            high_pct.append(np.nan)
            continue
        low_pct.append(float((s < under_thr).mean() * 100.0))
        in_pct.append(float(((s >= under_thr) & (s <= over_thr)).mean() * 100.0))
        high_pct.append(float((s > over_thr).mean() * 100.0))

    order = sorted(
        range(len(WHEELS)),
        key=lambda i: (0.0 if np.isnan(in_pct[i]) else in_pct[i]),
    )
    wheels_ordered = [WHEELS[i] for i in order]
    low_ordered = [-(0.0 if np.isnan(low_pct[i]) else low_pct[i]) for i in order]
    high_ordered = [(0.0 if np.isnan(high_pct[i]) else high_pct[i]) for i in order]
    in_ordered = [(0.0 if np.isnan(in_pct[i]) else in_pct[i]) for i in order]
    target_bar_width = 8.0

    fig_split = make_dark_figure(
        title="TC slip error balance by wheel",
        xlabel="Evaluated samples [%]  (short ← 0 → over)",
        ylabel="Wheel",
    )
    fig_split.add_trace(go.Bar(
        x=low_ordered,
        y=wheels_ordered,
        orientation="h",
        name=f"Short: SR < {under_thr:.2f}",
        marker=dict(color="#4DB3F2"),
        text=[f"{abs(v):.1f}%" for v in low_ordered],
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate="%{y}<br>Short: %{customdata:.1f}%<extra></extra>",
        customdata=[abs(v) for v in low_ordered],
    ))
    fig_split.add_trace(go.Bar(
        x=high_ordered,
        y=wheels_ordered,
        orientation="h",
        name=f"Over: SR > {over_thr:.2f}",
        marker=dict(color="#F27070"),
        text=[f"{v:.1f}%" for v in high_ordered],
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate="%{y}<br>Over: %{x:.1f}%<extra></extra>",
    ))
    fig_split.add_trace(go.Bar(
        x=[target_bar_width for _ in wheels_ordered],
        y=wheels_ordered,
        base=[-target_bar_width / 2.0 for _ in wheels_ordered],
        orientation="h",
        name=f"Target: {under_thr:.2f}-{over_thr:.2f}",
        marker=dict(color="#73D973", line=dict(color="#141417", width=1)),
        text=[f"{v:.1f}% target" for v in in_ordered],
        textposition="outside",
        textfont=dict(color="#73D973", size=11),
        hovertemplate="%{y}<br>Target: %{customdata:.1f}%<extra></extra>",
        customdata=in_ordered,
    ))
    fig_split.add_vline(x=0.0, line=dict(color="rgba(235,235,235,0.65)", width=1.2))
    fig_split.update_layout(barmode="relative", height=430)
    fig_split.update_xaxes(range=[-100, 100], tickvals=[-100, -75, -50, -25, 0, 25, 50, 75, 100],
                           ticktext=["100", "75", "50", "25", "0", "25", "50", "75", "100"])
    figs.append(fig_split)

    return figs


def _tc_control_diagnostic_figures(
    d: dict[str, np.ndarray],
    sr: dict[str, np.ndarray],
    sr_mat: np.ndarray,
    eval_mask: np.ndarray,
    over_thr: float,
    under_thr: float,
) -> list[go.Figure]:
    """Controller-focused figures: reference tracking and overslip response."""
    figs: list[go.Figure] = []

    # ── Overslip events: does SR return to target after crossing 0.25? ───────
    time_s = d["time"]
    worst_sr = np.nanmax(sr_mat, axis=1)
    high = eval_mask & (worst_sr > over_thr) & np.isfinite(time_s) & np.isfinite(worst_sr)
    starts = np.flatnonzero(high & ~np.r_[False, high[:-1]])
    min_gap_s = 0.30
    filtered_starts: list[int] = []
    last_t = -np.inf
    for idx in starts:
        t = float(time_s[idx])
        if t - last_t >= min_gap_s:
            filtered_starts.append(int(idx))
            last_t = t

    grid = np.linspace(-0.25, 1.00, 126)
    traces: list[np.ndarray] = []
    cut_traces: list[np.ndarray] = []
    torque_cols_present = [c for c in TC_TORQUE_COLS if c in d]
    if torque_cols_present:
        torque_mat = np.stack([d[c] for c in torque_cols_present], axis=1)
        any_cut = np.any(torque_mat < -1.0e-6, axis=1).astype(float)
    else:
        any_cut = np.full(len(time_s), np.nan)

    for idx in filtered_starts[:200]:
        t0 = time_s[idx]
        win = (
            (time_s >= t0 + grid[0])
            & (time_s <= t0 + grid[-1])
            & np.isfinite(time_s)
            & np.isfinite(worst_sr)
        )
        if win.sum() < 5:
            continue
        t_rel = time_s[win] - t0
        traces.append(np.interp(grid, t_rel, worst_sr[win]))
        if np.any(np.isfinite(any_cut[win])):
            cut_traces.append(np.interp(grid, t_rel, any_cut[win]))

    fig_response = make_dark_figure(
        title="Overslip response: does TC bring SR back to target?",
        xlabel="Time from SR > 0.25 event [s]",
        ylabel="Worst-wheel slip ratio [-]",
    )
    if traces:
        mat = np.vstack(traces)
        p25 = np.nanpercentile(mat, 25.0, axis=0)
        p50 = np.nanpercentile(mat, 50.0, axis=0)
        p75 = np.nanpercentile(mat, 75.0, axis=0)
        fig_response.add_trace(go.Scatter(
            x=np.r_[grid, grid[::-1]],
            y=np.r_[p75, p25[::-1]],
            fill="toself",
            fillcolor="rgba(77,179,242,0.18)",
            line=dict(color="rgba(77,179,242,0.0)"),
            name="P25-P75 SR",
            hoverinfo="skip",
        ))
        fig_response.add_trace(go.Scatter(
            x=grid,
            y=p50,
            mode="lines",
            name=f"Median worst SR ({len(traces)} events)",
            line=dict(color="#EBEBEB", width=2.4),
            hovertemplate="t %{x:+.2f}s<br>median worst SR %{y:.3f}<extra></extra>",
        ))
        if cut_traces:
            cut_med = np.nanmedian(np.vstack(cut_traces), axis=0)
            fig_response.add_trace(go.Scatter(
                x=grid,
                y=under_thr + cut_med * (over_thr - under_thr),
                mode="lines",
                name="TC cut active (scaled)",
                line=dict(color="#F28C40", width=1.6, dash="dot"),
                hovertemplate="t %{x:+.2f}s<br>TC cut scaled %{text}<extra></extra>",
                text=[f"{v:.0%}" for v in cut_med],
            ))
    else:
        fig_response.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="No SR > 0.25 events in the evaluated TC window",
            showarrow=False,
            font=dict(color="#EBEBEB", size=13),
        )
    fig_response.add_hrect(y0=under_thr, y1=over_thr, fillcolor="rgba(115,217,115,0.12)", line_width=0)
    fig_response.add_hline(y=SR_TARGET, line=dict(color="rgba(255,255,255,0.65)", dash="dash", width=1.3))
    fig_response.add_vline(x=0.0, line=dict(color="rgba(242,112,112,0.65)", dash="dash", width=1.3))
    fig_response.update_yaxes(range=[-0.02, 0.45])
    figs.append(fig_response)

    # ── Demand context: is low SR caused by no torque demand? ────────────────
    if all(c in d for c in MOTOR_TORQUE_COLS):
        fig_torque = make_dark_figure(
            title="SR vs delivered wheel torque during TC evaluation",
            xlabel="Slip ratio [-]",
            ylabel="Actual wheel torque [Nm]",
        )
        for j, w in enumerate(WHEELS):
            torque = d[MOTOR_TORQUE_COLS[j]][eval_mask]
            slip = sr[w][eval_mask]
            mask = np.isfinite(torque) & np.isfinite(slip)
            if not mask.any():
                continue
            fig_torque.add_trace(go.Scattergl(
                x=slip[mask],
                y=torque[mask],
                mode="markers",
                name=w,
                marker=dict(color=WHEEL_COLORS[w], size=4, opacity=0.35, symbol=WHEEL_SYMBOLS[w]),
                hovertemplate=f"{w}<br>SR %{{x:.3f}}<br>Torque %{{y:.1f}} Nm<extra></extra>",
            ))
        fig_torque.add_vrect(x0=under_thr, x1=over_thr, fillcolor="rgba(115,217,115,0.10)", line_width=0)
        fig_torque.add_vline(x=SR_TARGET, line=dict(color="rgba(255,255,255,0.65)", dash="dash", width=1.3))
        fig_torque.update_xaxes(range=[-0.08, 0.45])
        figs.append(fig_torque)

    return figs


def tc_function_kpis(df: pl.DataFrame) -> tuple[list[go.Figure], dict]:
    """Function-level check for Traction Control.

    Pregunta: ¿está el TC consiguiendo SR ≈ +0.20 cuando el coche acelera?
    """
    d = _prepare_arrays_from_df(df)
    data_keys = [k for k in d if not k.startswith("__")]
    valid = np.all(np.stack([np.isfinite(d[k]) for k in data_keys], axis=1), axis=1)
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)

    full_throttle_accel = (
        (d["Throttle"] >= FULL_THROTTLE_THRESHOLD)
        & (d["ax"] >= AX_THRESHOLD)
        & (np.abs(d["vx"]) >= MIN_SPEED)
    )
    tc_mode_mask = _bool_signal_mask(d, TC_ENABLE_MODE_COL)
    if tc_mode_mask is None:
        tc_mode_mask = np.ones(len(d["time"]), dtype=bool)

    tc_enable_mask = _bool_signal_mask(d, TC_ENABLE_COL)
    used_tc_enable_signal = tc_enable_mask is not None
    if tc_enable_mask is None:
        tc_enable_mask = _tc_armed_mask(d)

    armed = tc_mode_mask & tc_enable_mask & (np.abs(d["vx"]) >= MIN_SPEED)
    eval_mask = armed & full_throttle_accel

    sr = {w: d[f"Est_SR{w}"] for w in WHEELS}
    sr_mat = np.stack([sr[w] for w in WHEELS], axis=1)

    over_thr = SR_TARGET + DELTA_SR
    under_thr = SR_TARGET - DELTA_SR
    in_target_mat = (sr_mat >= under_thr) & (sr_mat <= over_thr)

    pct_in_target_w: dict[str, float] = {}
    pct_overslip_w: dict[str, float] = {}
    pct_underslip_w: dict[str, float] = {}
    median_sr_w: dict[str, float] = {}
    p95_sr_w: dict[str, float] = {}
    overshoot_med_w: dict[str, float] = {}
    for j, w in enumerate(WHEELS):
        s = sr[w][eval_mask]
        if s.size == 0:
            pct_in_target_w[w] = np.nan
            pct_overslip_w[w] = np.nan
            pct_underslip_w[w] = np.nan
            median_sr_w[w] = np.nan
            p95_sr_w[w] = np.nan
            overshoot_med_w[w] = np.nan
            continue
        pct_in_target_w[w] = float(((s >= under_thr) & (s <= over_thr)).mean() * 100.0)
        pct_overslip_w[w] = float((s > over_thr).mean() * 100.0)
        pct_underslip_w[w] = float((s < under_thr).mean() * 100.0)
        median_sr_w[w] = float(np.nanmedian(s))
        p95_sr_w[w] = float(np.nanpercentile(s, 95.0))
        over_vals = s[s > over_thr] - SR_TARGET
        overshoot_med_w[w] = float(np.nanmedian(over_vals)) if over_vals.size else 0.0

    sr_eval = sr_mat[eval_mask]
    in_target_eval = in_target_mat[eval_mask]
    pct_in_target = float(np.all(in_target_eval, axis=1).mean() * 100.0) if sr_eval.size else np.nan
    pct_wheel_samples_in_target = float(in_target_eval.mean() * 100.0) if sr_eval.size else np.nan

    sr_mean_eval = np.nanmean(sr_eval, axis=1) if sr_eval.size else np.array([], dtype=float)
    sr_max_eval = np.nanmax(sr_eval, axis=1) if sr_eval.size else np.array([], dtype=float)
    sr_min_eval = np.nanmin(sr_eval, axis=1) if sr_eval.size else np.array([], dtype=float)
    abs_err_eval = np.abs(sr_eval - SR_TARGET) if sr_eval.size else np.array([], dtype=float)
    median_sr_error = (
        float(np.nanmedian(sr_mean_eval - SR_TARGET)) if sr_mean_eval.size else np.nan
    )
    median_sr = float(np.nanmedian(sr_eval)) if sr_eval.size else np.nan
    median_abs_error = float(np.nanmedian(abs_err_eval)) if abs_err_eval.size else np.nan
    target_gap_pct = (
        float((median_sr - SR_TARGET) / SR_TARGET * 100.0)
        if np.isfinite(median_sr) and SR_TARGET != 0.0
        else np.nan
    )
    p90_abs_error = float(np.nanpercentile(abs_err_eval, 90.0)) if abs_err_eval.size else np.nan
    pct_any_overslip = float((sr_max_eval > over_thr).mean() * 100.0) if sr_max_eval.size else np.nan
    pct_any_underslip = float((sr_min_eval < under_thr).mean() * 100.0) if sr_min_eval.size else np.nan
    pct_all_underslip = float(np.all(sr_eval < under_thr, axis=1).mean() * 100.0) if sr_eval.size else np.nan
    pct_tc_needed = pct_any_overslip
    p95_worst_sr = float(np.nanpercentile(sr_max_eval, 95.0)) if sr_max_eval.size else np.nan
    median_wheel_spread = float(np.nanmedian(sr_max_eval - sr_min_eval)) if sr_eval.size else np.nan
    overshoot_vals = sr_max_eval[sr_max_eval > over_thr] - SR_TARGET
    undershoot_vals = SR_TARGET - sr_min_eval[sr_min_eval < under_thr]
    overshoot_med = float(np.nanmedian(overshoot_vals)) if overshoot_vals.size else 0.0
    undershoot_med = float(np.nanmedian(undershoot_vals)) if undershoot_vals.size else 0.0

    full_throttle = full_throttle_accel & tc_mode_mask
    pct_armed_when_full = (
        float(tc_enable_mask[full_throttle].mean() * 100.0) if full_throttle.any() else np.nan
    )

    worst_wheel = (
        min(WHEELS, key=lambda w: pct_in_target_w[w])
        if all(np.isfinite(pct_in_target_w[w]) for w in WHEELS)
        else "—"
    )

    negative_torque_mask = np.zeros(len(d["time"]), dtype=bool)
    present_torque_cols = [c for c in TC_TORQUE_COLS if c in d]
    if present_torque_cols:
        torque_mat = np.stack([d[c] for c in present_torque_cols], axis=1)
        negative_torque_mask = np.any(torque_mat < -1.0e-6, axis=1)
    pct_negative_torque_when_eval = (
        float(negative_torque_mask[eval_mask].mean() * 100.0)
        if eval_mask.any() and present_torque_cols
        else np.nan
    )
    overslip_eval_mask = eval_mask & np.any(sr_mat > over_thr, axis=1)
    pct_cut_when_overslip = (
        float(negative_torque_mask[overslip_eval_mask].mean() * 100.0)
        if overslip_eval_mask.any() and present_torque_cols
        else np.nan
    )
    tc_score_pct = (
        0.65 * pct_wheel_samples_in_target
        + 0.20 * (100.0 - pct_all_underslip)
        + 0.15 * np.nan_to_num(pct_cut_when_overslip, nan=0.0)
        if np.isfinite(pct_wheel_samples_in_target) and np.isfinite(pct_all_underslip)
        else np.nan
    )
    objective_ok = bool(
        np.isfinite(pct_wheel_samples_in_target)
        and np.isfinite(median_abs_error)
        and pct_wheel_samples_in_target >= 50.0
        and median_abs_error <= DELTA_SR
    )
    if not np.isfinite(median_sr):
        failure_mode = "No data"
    elif median_sr < under_thr:
        failure_mode = "Under target"
    elif pct_any_overslip > 20.0:
        failure_mode = "Over target"
    elif pct_wheel_samples_in_target < 50.0:
        failure_mode = "Mixed / unstable"
    else:
        failure_mode = "On target"
    tc_action_seen = bool(
        np.isfinite(pct_cut_when_overslip)
        and pct_cut_when_overslip > 0.0
    )

    notes: list[str] = []
    if not used_tc_enable_signal:
        notes.append("TCenable not found; using throttle/ax/vx heuristic as TC armed mask.")
    if not eval_mask.any():
        notes.append("No samples with full-throttle acceleration and TC armed after excluding lap 0 and the last lap.")
    if present_torque_cols and eval_mask.any() and np.nanmax(np.abs(torque_mat[eval_mask])) <= 1.0e-6:
        notes.append("TC torque command columns are zero during the evaluated samples.")

    figs = _tc_visual_figures(
        d,
        sr,
        sr_mat,
        eval_mask,
        tc_enable_mask,
        full_throttle_accel,
        over_thr,
        under_thr,
    )[1:]
    figs.extend(_tc_objective_figures(d, sr, sr_mat, eval_mask, over_thr, under_thr))
    figs.extend(_tc_control_diagnostic_figures(d, sr, sr_mat, eval_mask, over_thr, under_thr))

    kpis = {
        "pct_in_target": pct_in_target,
        "pct_wheel_samples_in_target": pct_wheel_samples_in_target,
        "median_sr": median_sr,
        "median_sr_error": median_sr_error,
        "median_abs_error": median_abs_error,
        "target_gap_pct": target_gap_pct,
        "p90_abs_error": p90_abs_error,
        "objective_ok": objective_ok,
        "failure_mode": failure_mode,
        "tc_action_seen": tc_action_seen,
        "tc_score_pct": tc_score_pct,
        "pct_tc_needed": pct_tc_needed,
        "pct_any_overslip": pct_any_overslip,
        "pct_any_underslip": pct_any_underslip,
        "pct_all_underslip": pct_all_underslip,
        "p95_worst_sr": p95_worst_sr,
        "median_wheel_spread": median_wheel_spread,
        "overshoot_med": overshoot_med,
        "undershoot_med": undershoot_med,
        "pct_armed_when_full": pct_armed_when_full,
        "pct_negative_torque_when_eval": pct_negative_torque_when_eval,
        "pct_cut_when_overslip": pct_cut_when_overslip,
        "worst_wheel": worst_wheel,
        "pct_in_target_by_wheel": pct_in_target_w,
        "pct_overslip_by_wheel": pct_overslip_w,
        "pct_underslip_by_wheel": pct_underslip_w,
        "median_sr_by_wheel": median_sr_w,
        "p95_sr_by_wheel": p95_sr_w,
        "overshoot_med_by_wheel": overshoot_med_w,
        "eval_samples": int(eval_mask.sum()),
        "full_throttle_accel_samples": int(full_throttle.sum()),
        "notes": notes,
    }
    return figs, kpis


MASTER_TORQUE_ALIASES = {
    "FL": ("Master_frontLeftTrq", "master_fl_torque", "master_front_left_torque"),
    "FR": ("Master_frontRightTrq", "master_fr_torque", "master_front_right_torque"),
    "RL": ("Master_rearLeftTrq", "master_rl_torque", "master_rear_left_torque"),
    "RR": ("Master_rearRightTrq", "master_rr_torque", "master_rear_right_torque"),
}
TC_TORQUE_ALIASES = {
    "FL": ("TC_FL_MaxTrq", "tc_fl_torque"),
    "FR": ("TC_FR_MaxTrq", "tc_fr_torque"),
    "RL": ("TC_RL_MaxTrq", "tc_rl_torque"),
    "RR": ("TC_RR_MaxTrq", "tc_rr_torque"),
}
ACTUAL_TORQUE_ALIASES = {
    "FL": ("FL_actualTorque", "fl_actual_torque"),
    "FR": ("FR_actualTorque", "fr_actual_torque"),
    "RL": ("RL_actualTorque", "rl_actual_torque"),
    "RR": ("RR_actualTorque", "rr_actual_torque"),
}


def _first_existing_col(df: pl.DataFrame, aliases: tuple[str, ...]) -> str | None:
    return next((col for col in aliases if col in df.columns), None)


def _series_or_nan(df: pl.DataFrame, aliases: tuple[str, ...]) -> np.ndarray:
    col = _first_existing_col(df, aliases)
    if col is None:
        return np.full(len(df), np.nan, dtype=float)
    return df[col].to_numpy().astype(float)


def _median_first_after(
    time_s: np.ndarray,
    starts: np.ndarray,
    response_mask: np.ndarray,
    *,
    max_latency_s: float,
) -> float:
    latencies: list[float] = []
    for idx in starts:
        t0 = time_s[idx]
        win = np.flatnonzero(
            response_mask
            & (time_s >= t0)
            & (time_s <= t0 + max_latency_s)
        )
        if win.size:
            latencies.append(float(time_s[win[0]] - t0))
    return float(np.nanmedian(latencies) * 1000.0) if latencies else np.nan


def tc_control_impact_figs_kpis(df: pl.DataFrame) -> tuple[list[go.Figure], dict]:
    """Observable TC behaviour: slip recovery versus acceleration loss."""
    df = ensure_complete_laps_df(df)
    d = _prepare_arrays_from_df(df)
    data_keys = [k for k in d if not k.startswith("__")]
    valid = np.all(np.stack([np.isfinite(d[k]) for k in data_keys], axis=1), axis=1)
    d = {k: v[valid] for k, v in d.items()}

    # Keep the dataframe aligned with the same validity mask used by d.
    df_eff = ensure_complete_laps_df(df).filter(pl.Series(valid))
    if len(df_eff) == 0:
        raise ValueError("No valid samples for TC behaviour metrics.")

    time_s = d["time"]
    dist_m = d["s_m"]
    laps = d["laps"]
    sr_mat = np.stack([d[f"Est_SR{w}"] for w in WHEELS], axis=1)
    over_thr = SR_TARGET + DELTA_SR
    under_thr = SR_TARGET - DELTA_SR

    tc_mode_mask = _bool_signal_mask(d, TC_ENABLE_MODE_COL)
    if tc_mode_mask is None:
        tc_mode_mask = np.ones(len(time_s), dtype=bool)
    tc_enable_mask = _bool_signal_mask(d, TC_ENABLE_COL)
    if tc_enable_mask is None:
        tc_enable_mask = _tc_armed_mask(d)
    accel_mask = (
        tc_mode_mask
        & (tc_enable_mask | _tc_armed_mask(d))
        & (d["Throttle"] >= THROTTLE_THRESHOLD)
        & (d["ax"] >= AX_THRESHOLD)
        & (np.abs(d["vx"]) >= MIN_SPEED)
    )

    tc_torque = np.stack([
        _series_or_nan(df_eff, TC_TORQUE_ALIASES[w]) for w in WHEELS
    ], axis=1)
    master_torque = np.stack([
        _series_or_nan(df_eff, MASTER_TORQUE_ALIASES[w]) for w in WHEELS
    ], axis=1)
    actual_torque = np.stack([
        _series_or_nan(df_eff, ACTUAL_TORQUE_ALIASES[w]) for w in WHEELS
    ], axis=1)
    tc_cut = np.minimum(tc_torque, 0.0)
    cut_active = np.any(tc_cut < -1.0e-6, axis=1)
    cut_total_nm = -np.nansum(tc_cut, axis=1)
    worst_sr = np.nanmax(sr_mat, axis=1)
    slip_error = worst_sr - SR_TARGET
    overslip = accel_mask & (worst_sr > over_thr)
    overslip_starts = np.flatnonzero(overslip & ~np.r_[False, overslip[:-1]])
    recovery_times_s: list[float] = []
    events_with_cut = 0
    for idx in overslip_starts:
        win = (time_s >= time_s[idx]) & (time_s <= time_s[idx] + 1.0)
        if np.any(cut_active & win):
            events_with_cut += 1
        recovered = np.flatnonzero(win & (worst_sr <= over_thr))
        if recovered.size:
            recovery_times_s.append(float(time_s[recovered[0]] - time_s[idx]))

    eval_mask = accel_mask & np.any(np.isfinite(tc_torque), axis=1)
    master_total = np.nansum(master_torque, axis=1)
    actual_total = np.nansum(actual_torque, axis=1)
    high_throttle = eval_mask & (d["Throttle"] >= 50.0)
    cut_eval = high_throttle & cut_active
    no_cut_eval = high_throttle & ~cut_active
    ax_cut = d["ax"][cut_eval]
    ax_no_cut = d["ax"][no_cut_eval]
    ax_penalty = (
        float(np.nanmedian(ax_no_cut) - np.nanmedian(ax_cut))
        if ax_cut.size and ax_no_cut.size else np.nan
    )
    cut_without_overslip_pct = (
        float(np.nanmean((worst_sr[cut_eval] <= over_thr)) * 100.0)
        if cut_eval.any() else np.nan
    )
    overslip_excess = np.clip(worst_sr - over_thr, 0.0, None)
    cut_starts = np.flatnonzero(cut_active & ~np.r_[False, cut_active[:-1]])
    ax_delta_after_cut: list[float] = []
    for idx in cut_starts:
        pre = (time_s >= time_s[idx] - 0.25) & (time_s < time_s[idx])
        post = (time_s >= time_s[idx]) & (time_s <= time_s[idx] + 0.35)
        if pre.sum() >= 3 and post.sum() >= 3:
            ax_delta_after_cut.append(float(np.nanmedian(d["ax"][post]) - np.nanmedian(d["ax"][pre])))
    recovery_time_ms = float(np.nanmedian(recovery_times_s) * 1000.0) if recovery_times_s else np.nan
    pct_events_with_cut = float(events_with_cut / len(overslip_starts) * 100.0) if len(overslip_starts) else np.nan

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=("Worst-wheel SR", "Longitudinal acceleration", "TC cut applied"),
    )
    order = np.argsort(dist_m)
    fig.add_trace(go.Scattergl(
        x=dist_m[order], y=worst_sr[order], mode="markers", name="Worst SR",
        marker=dict(color=cut_total_nm[order], colorscale="Turbo", size=3, opacity=0.50, colorbar=dict(title="Cut [Nm]")),
    ), row=1, col=1)
    fig.add_hrect(y0=under_thr, y1=over_thr, fillcolor="rgba(115,217,115,0.12)", line_width=0, row=1, col=1)
    fig.add_hline(y=SR_TARGET, line=dict(color="rgba(255,255,255,0.65)", dash="dash", width=1.2), row=1, col=1)
    fig.add_trace(go.Scattergl(
        x=dist_m[order], y=d["ax"][order], mode="markers", name="ax",
        marker=dict(color=worst_sr[order], colorscale="RdYlGn_r", size=3, opacity=0.50, colorbar=dict(title="Worst SR")),
    ), row=2, col=1)
    fig.add_trace(go.Scattergl(
        x=dist_m[order], y=cut_total_nm[order], mode="markers", name="TC cut",
        marker=dict(color="#F28C40", size=3, opacity=0.55),
    ), row=3, col=1)
    fig.update_layout(
        title=dict(text="TC: overslip recovery versus acceleration penalty", font=dict(size=14, color="#EBEBEB")),
        paper_bgcolor="#141417",
        plot_bgcolor="#141417",
        font=dict(color="#EBEBEB", size=11),
        height=760,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0.0),
    )
    fig.update_xaxes(title_text="Distance [m]", row=3, col=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(title_text="Slip ratio [-]", row=1, col=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(title_text="ax [m/s²]", row=2, col=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(title_text="Cut [Nm]", row=3, col=1, gridcolor="rgba(128,128,128,0.2)")

    fig_rel = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.10,
        subplot_titles=("Slip error vs acceleration", "TC cut vs acceleration"),
    )
    rel_mask = eval_mask & np.isfinite(slip_error) & np.isfinite(d["ax"])
    fig_rel.add_trace(go.Scattergl(
        x=slip_error[rel_mask],
        y=d["ax"][rel_mask],
        mode="markers",
        name="Slip -> ax",
        marker=dict(color=cut_total_nm[rel_mask], colorscale="Turbo", size=4, opacity=0.45, colorbar=dict(title="Cut")),
    ), row=1, col=1)
    fig_rel.add_trace(go.Scattergl(
        x=cut_total_nm[rel_mask],
        y=d["ax"][rel_mask],
        mode="markers",
        name="Cut -> ax",
        marker=dict(color=worst_sr[rel_mask], colorscale="RdYlGn_r", size=4, opacity=0.45, colorbar=dict(title="SR")),
    ), row=1, col=2)
    fig_rel.add_vline(x=DELTA_SR, line=dict(color="rgba(235,235,235,0.5)", dash="dash", width=1.0), row=1, col=1)
    fig_rel.update_layout(
        title=dict(text="TC variable relationships", font=dict(size=14, color="#EBEBEB")),
        paper_bgcolor="#141417",
        plot_bgcolor="#141417",
        font=dict(color="#EBEBEB", size=11),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0.0),
    )
    fig_rel.update_xaxes(title_text="Worst SR - target [-]", row=1, col=1, gridcolor="rgba(128,128,128,0.2)")
    fig_rel.update_yaxes(title_text="ax [m/s²]", row=1, col=1, gridcolor="rgba(128,128,128,0.2)")
    fig_rel.update_xaxes(title_text="TC cut [Nm]", row=1, col=2, gridcolor="rgba(128,128,128,0.2)")
    fig_rel.update_yaxes(title_text="ax [m/s²]", row=1, col=2, gridcolor="rgba(128,128,128,0.2)")

    lap_rows: list[dict[str, object]] = []
    for lap in unique_laps(laps):
        lm = laps == lap
        lem = lm & eval_mask
        if int(lem.sum()) < MIN_SAMPLES_PER_LAP:
            continue
        lap_cut = lm & cut_eval
        lap_no_cut = lm & no_cut_eval
        lap_ax_penalty = (
            float(np.nanmedian(d["ax"][lap_no_cut]) - np.nanmedian(d["ax"][lap_cut]))
            if lap_cut.any() and lap_no_cut.any() else np.nan
        )
        lap_rows.append({
            "Lap": int(lap),
            "TC samples": int(lem.sum()),
            "Overslip [%]": round(float(np.nanmean(worst_sr[lem] > over_thr) * 100.0), 1),
            "Mean SR excess": round(float(np.nanmean(overslip_excess[lem])), 4),
            "Cut without overslip [%]": round(float(np.nanmean(worst_sr[lap_cut] <= over_thr) * 100.0), 1) if lap_cut.any() else np.nan,
            "ax penalty cut [m/s²]": round(lap_ax_penalty, 3) if np.isfinite(lap_ax_penalty) else np.nan,
            "Mean ax [m/s²]": round(float(np.nanmean(d["ax"][lem])), 3),
            "Actual/Master lag [Nm]": round(float(np.nanmean(np.abs(actual_total[lem] - master_total[lem]))), 2),
        })

    kpis = {
        "overslip_pct": float(np.nanmean(overslip[eval_mask]) * 100.0) if eval_mask.any() else np.nan,
        "mean_sr_excess": float(np.nanmean(overslip_excess[eval_mask])) if eval_mask.any() else np.nan,
        "recovery_time_ms": recovery_time_ms,
        "pct_events_with_cut": pct_events_with_cut,
        "cut_without_overslip_pct": cut_without_overslip_pct,
        "ax_penalty_cut_ms2": ax_penalty,
        "ax_delta_after_cut_ms2": float(np.nanmedian(ax_delta_after_cut)) if ax_delta_after_cut else np.nan,
        "actual_master_mae_nm": float(np.nanmean(np.abs(actual_total[eval_mask] - master_total[eval_mask]))) if eval_mask.any() else np.nan,
        "eval_samples": int(eval_mask.sum()),
        "table": pl.DataFrame(lap_rows) if lap_rows else pl.DataFrame(),
        "warnings": [],
    }
    return [fig, fig_rel], kpis


if __name__ == "__main__":
    main()
