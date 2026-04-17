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

from utils import (
    COMPLETE_LAPS_MARKER,
    WHEEL_COLORS,
    WHEEL_SYMBOLS,
    add_lap_scatter,
    add_zero_line,
    ensure_complete_laps_df,
    exclude_lap0_and_last_lap,
    keep_min_duration_segments,
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
AX_THRESHOLD = 0.50
MIN_SPEED = 4.0
STEERING_STRAIGHT = 0.08
USE_STRAIGHT_FILTER = True
MIN_EVENT_DURATION = 0.15
MIN_SAMPLES_PER_LAP = 40


def _load(columns: list[str]) -> dict[str, np.ndarray]:
    df = pl.read_csv(CSV_PATH, columns=columns)
    return {c: df[c].to_numpy().astype(float) for c in columns}


def _from_df(df: pl.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    df = ensure_complete_laps_df(df)
    cols = list(columns)
    if COMPLETE_LAPS_MARKER in df.columns and COMPLETE_LAPS_MARKER not in cols:
        cols.append(COMPLETE_LAPS_MARKER)
    return {c: df[c].to_numpy().astype(float) for c in cols}


def _ax_signal(columns: list[str]) -> str:
    return "Filtering_VN_ax" if "Filtering_VN_ax" in columns else "VN_ax"


def _vx_signal(columns: list[str]) -> str:
    return "Est_vxCOG" if "Est_vxCOG" in columns else "VN_vx"


def _prepare_arrays_from_df(
    df: pl.DataFrame,
    accel_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    ax_col = _ax_signal(df.columns)
    vx_col = _vx_signal(df.columns)
    cols = [
        "TimeStamp", "laps", "laptime",
        "Throttle", "Steering", ax_col, vx_col,
        "Est_SRFL", "Est_SRFR", "Est_SRRL", "Est_SRRR",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    d = _from_df(df, cols)
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


if __name__ == "__main__":
    main()
