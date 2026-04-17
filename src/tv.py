"""tv.py
------
Torque Vectoring (TV) KPIs — yaw tracking and moment distribution quality.

KPIs (all computed during cornering: |ay| >= threshold AND |steering| >= threshold):
  1. Yaw rate error RMSE and bias per lap
  2. Mz error RMSE and bias per lap
  3. Feedback / Feedforward Mz ratio per lap
"""
from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go

from utils import (
    COMPLETE_LAPS_MARKER,
    add_lap_scatter,
    add_trend_line,
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

# ── Cornering filter parameters ───────────────────────────────────────────────
AY_THRESHOLD = 2.0
STEERING_THRESHOLD = 0.05
MIN_SPEED = 4.0
MIN_CORNER_DURATION = 0.20
MIN_CORNER_SAMPLES = 50

FF_MIN_FOR_RATIO = 5.0
EPS_RATIO = 1e-6


def _load(columns: list[str]) -> dict[str, np.ndarray]:
    df = pl.read_csv(CSV_PATH, columns=columns)
    return {c: df[c].to_numpy().astype(float) for c in columns}


def _from_df(df: pl.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    df = ensure_complete_laps_df(df)
    cols = list(columns)
    if COMPLETE_LAPS_MARKER in df.columns and COMPLETE_LAPS_MARKER not in cols:
        cols.append(COMPLETE_LAPS_MARKER)
    return {c: df[c].to_numpy().astype(float) for c in cols}


def _vx_signal(columns: list[str]) -> str:
    return "Est_vxCOG" if "Est_vxCOG" in columns else "VN_vx"


def _ay_signal(columns: list[str]) -> str:
    return "Filtering_VN_ay" if "Filtering_VN_ay" in columns else "VN_ay"


def _corner_mask(ay: np.ndarray, steering: np.ndarray, vx: np.ndarray, dt: float) -> np.ndarray:
    raw = (
        (np.abs(ay) >= AY_THRESHOLD)
        & (np.abs(steering) >= STEERING_THRESHOLD)
        & (np.abs(vx) >= MIN_SPEED)
    )
    return keep_min_duration_segments(raw, MIN_CORNER_DURATION, dt)


def _per_lap_error(
    err: np.ndarray,
    laps: np.ndarray,
    laptime: np.ndarray,
    corner_mask: np.ndarray,
    lap_list: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute RMSE and bias of *err* in corners per lap."""
    n = len(lap_list)
    rmse = np.full(n, np.nan)
    bias = np.full(n, np.nan)
    lt = np.full(n, np.nan)
    nsamp = np.zeros(n, dtype=int)
    cover = np.full(n, np.nan)

    for i, lap in enumerate(lap_list):
        lm = laps == lap
        lcm = lm & corner_mask
        nsamp[i] = int(lcm.sum())
        if lm.any():
            lt[i] = laptime[lm].max()
            cover[i] = lcm.sum() / lm.sum()
        if nsamp[i] < MIN_CORNER_SAMPLES:
            continue
        e = err[lcm]
        rmse[i] = np.sqrt(np.nanmean(e ** 2))
        bias[i] = np.nanmean(e)

    return rmse, bias, lt, nsamp, cover


def _prepare_arrays_from_df(
    df: pl.DataFrame,
    corner_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    ay_col = _ay_signal(df.columns)
    vx_col = _vx_signal(df.columns)
    cols = [
        "TimeStamp", "laps", "laptime",
        "TV_errorYawRate", "TV_errorMz",
        "TV_feedForwardMz", "TV_feedBackMz",
        "Steering", ay_col, vx_col,
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    d = _from_df(df, cols)
    d["time"] = d["TimeStamp"] - d["TimeStamp"][0]
    d["ay"] = d.pop(ay_col)
    d["vx"] = d.pop(vx_col)
    if corner_mask is not None:
        d["__corner_mask"] = corner_mask.astype(float)
    return d


def _prepare_arrays_from_csv() -> dict[str, np.ndarray]:
    header = pl.read_csv(CSV_PATH, n_rows=1).columns
    ay_col = _ay_signal(header)
    vx_col = _vx_signal(header)
    d = _load([
        "TimeStamp", "laps", "laptime",
        "TV_errorYawRate", "TV_errorMz",
        "TV_feedForwardMz", "TV_feedBackMz",
        "Steering", ay_col, vx_col,
    ])
    d["time"] = d["TimeStamp"] - d["TimeStamp"][0]
    d["ay"] = d.pop(ay_col)
    d["vx"] = d.pop(vx_col)
    return d


def _compute_tv(d: dict[str, np.ndarray]) -> dict:
    has_ext = "__corner_mask" in d
    data_keys = [k for k in d if not k.startswith("__")]
    valid = np.all(np.stack([np.isfinite(d[k]) for k in data_keys], axis=1), axis=1)
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)

    dt = robust_dt(d["time"])
    if has_ext:
        cm = d["__corner_mask"].astype(bool) & (np.abs(d["vx"]) >= MIN_SPEED)
    else:
        cm = _corner_mask(d["ay"], d["Steering"], d["vx"], dt)
    lap_list = unique_laps(d["laps"])

    yaw_rmse, yaw_bias, lt, nsamp, cover = _per_lap_error(
        d["TV_errorYawRate"], d["laps"], d["laptime"], cm, lap_list,
    )
    mz_rmse, mz_bias, _lt2, _nsamp2, _cover2 = _per_lap_error(
        d["TV_errorMz"], d["laps"], d["laptime"], cm, lap_list,
    )

    n = len(lap_list)
    ff_mean = np.full(n, np.nan)
    fb_mean = np.full(n, np.nan)
    ratio = np.full(n, np.nan)
    fb_share = np.full(n, np.nan)

    for i, lap in enumerate(lap_list):
        lm = d["laps"] == lap
        lcm = lm & cm
        if int(lcm.sum()) < MIN_CORNER_SAMPLES:
            continue

        ff_abs = np.abs(d["TV_feedForwardMz"][lcm])
        fb_abs = np.abs(d["TV_feedBackMz"][lcm])
        ff_mean[i] = np.nanmean(ff_abs)
        fb_mean[i] = np.nanmean(fb_abs)
        total = ff_mean[i] + fb_mean[i] + EPS_RATIO
        fb_share[i] = fb_mean[i] / total
        if ff_mean[i] > FF_MIN_FOR_RATIO:
            ratio[i] = fb_mean[i] / (ff_mean[i] + EPS_RATIO)

    yaw_ok = np.isfinite(yaw_rmse) & np.isfinite(lt) & (nsamp >= MIN_CORNER_SAMPLES)
    mz_ok = np.isfinite(mz_rmse) & np.isfinite(lt) & (nsamp >= MIN_CORNER_SAMPLES)
    ratio_ok = (
        np.isfinite(ratio)
        & np.isfinite(lt)
        & (nsamp >= MIN_CORNER_SAMPLES)
        & (ff_mean > FF_MIN_FOR_RATIO)
    )
    any_ok = yaw_ok | mz_ok | ratio_ok

    table = pl.DataFrame({
        "Lap": lap_list[any_ok].astype(int),
        "LapTime [s]": np.round(lt[any_ok], 3),
        "Corner samples": nsamp[any_ok].astype(int),
        "Coverage [%]": np.round(cover[any_ok] * 100.0, 2),
        "Yaw RMSE": np.round(yaw_rmse[any_ok], 4),
        "Yaw Bias": np.round(yaw_bias[any_ok], 4),
        "Mz RMSE [Nm]": np.round(mz_rmse[any_ok], 2),
        "Mz Bias [Nm]": np.round(mz_bias[any_ok], 2),
        "FF mean [Nm]": np.round(ff_mean[any_ok], 2),
        "FB mean [Nm]": np.round(fb_mean[any_ok], 2),
        "FB/FF ratio": np.round(ratio[any_ok], 3),
        "FB share": np.round(fb_share[any_ok], 3),
    })

    warnings: list[str] = []
    if not any_ok.any():
        warnings.append("No valid cornering laps for TV KPIs.")

    return {
        "lap_list": lap_list,
        "ay": d["ay"],
        "yaw_err": d["TV_errorYawRate"],
        "mz_err": d["TV_errorMz"],
        "corner_mask": cm,
        "lt": lt,
        "nsamp": nsamp,
        "cover": cover,
        "yaw_rmse": yaw_rmse,
        "yaw_bias": yaw_bias,
        "mz_rmse": mz_rmse,
        "mz_bias": mz_bias,
        "ff_mean": ff_mean,
        "fb_mean": fb_mean,
        "ratio": ratio,
        "fb_share": fb_share,
        "yaw_ok": yaw_ok,
        "mz_ok": mz_ok,
        "ratio_ok": ratio_ok,
        "any_ok": any_ok,
        "table": table,
        "warnings": warnings,
    }


def _build_tv_figures(res: dict, x_mode: str = "laps") -> list[go.Figure]:
    lap_list = res["lap_list"]
    lt = res["lt"]
    figs: list[go.Figure] = []

    x_yaw, order_yaw, xlabel_yaw = per_lap_axis(lap_list[res["yaw_ok"]], lt[res["yaw_ok"]], x_mode) if res["yaw_ok"].any() else (np.array([]), np.array([], dtype=int), "Lap")
    fig = make_dark_figure(f"Yaw Rate Tracking Error vs {'Lap Time' if x_mode == 'laptime' else 'Lap'}", xlabel_yaw, "Yaw rate error RMSE")
    if res["yaw_ok"].any():
        add_lap_scatter(fig, x_yaw, res["yaw_rmse"][res["yaw_ok"]][order_yaw], lap_list[res["yaw_ok"]][order_yaw])
        if x_mode == "laps":
            fig.update_xaxes(tickvals=np.sort(lap_list[res["yaw_ok"]].astype(int)))
    figs.append(fig)

    fig = make_dark_figure(
        "Yaw Rate Error vs Lateral Acceleration",
        "Lateral acceleration ay [m/s²]",
        "Yaw rate error",
    )
    ay_s = res["ay"][res["corner_mask"]]
    err_s = res["yaw_err"][res["corner_mask"]]
    fig.add_trace(go.Scatter(
        x=ay_s, y=err_s, mode="markers",
        marker=dict(color="#4DB3F2", size=3, opacity=0.5),
        name="Samples",
    ))
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Left turns: {int((ay_s > 0).sum())}<br>Right turns: {int((ay_s < 0).sum())}",
        showarrow=False,
        align="left",
        font=dict(color="#EBEBEB", size=10),
        bgcolor="rgba(20,20,23,0.8)",
    )
    if ay_s.size > 0:
        add_zero_line(fig, ay_s)
    figs.append(fig)

    x_mz, order_mz, xlabel_mz = per_lap_axis(lap_list[res["mz_ok"]], lt[res["mz_ok"]], x_mode) if res["mz_ok"].any() else (np.array([]), np.array([], dtype=int), "Lap")
    fig = make_dark_figure(f"Mz Tracking Error vs {'Lap Time' if x_mode == 'laptime' else 'Lap'}", xlabel_mz, "Mz error RMSE [Nm]")
    if res["mz_ok"].any():
        add_lap_scatter(fig, x_mz, res["mz_rmse"][res["mz_ok"]][order_mz], lap_list[res["mz_ok"]][order_mz])
        if x_mode == "laps":
            fig.update_xaxes(tickvals=np.sort(lap_list[res["mz_ok"]].astype(int)))
    figs.append(fig)

    fig = make_dark_figure(
        "Mz Error vs Lateral Acceleration",
        "Lateral acceleration ay [m/s²]",
        "Mz error [Nm]",
    )
    err_s = res["mz_err"][res["corner_mask"]]
    fig.add_trace(go.Scatter(
        x=ay_s, y=err_s, mode="markers",
        marker=dict(color="#4DB3F2", size=3, opacity=0.5),
        name="Samples",
    ))
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Left turns: {int((ay_s > 0).sum())}<br>Right turns: {int((ay_s < 0).sum())}",
        showarrow=False,
        align="left",
        font=dict(color="#EBEBEB", size=10),
        bgcolor="rgba(20,20,23,0.8)",
    )
    if ay_s.size > 0:
        add_zero_line(fig, ay_s)
    figs.append(fig)

    x_ratio, order_ratio, xlabel_ratio = per_lap_axis(lap_list[res["ratio_ok"]], lt[res["ratio_ok"]], x_mode) if res["ratio_ok"].any() else (np.array([]), np.array([], dtype=int), "Lap")
    fig = make_dark_figure(f"Feedback to Feedforward Ratio vs {'Lap Time' if x_mode == 'laptime' else 'Lap'}", xlabel_ratio, "FB / FF ratio")
    if res["ratio_ok"].any():
        add_lap_scatter(
            fig,
            x_ratio,
            res["ratio"][res["ratio_ok"]][order_ratio],
            lap_list[res["ratio_ok"]][order_ratio],
        )
        add_trend_line(fig, x_ratio, res["ratio"][res["ratio_ok"]][order_ratio])
        if x_mode == "laps":
            fig.update_xaxes(tickvals=np.sort(lap_list[res["ratio_ok"]].astype(int)))
    figs.append(fig)

    return figs


def tv_figs_kpis(
    df: pl.DataFrame,
    corner_mask: np.ndarray | None = None,
    x_mode: str = "laps",
) -> tuple[list[go.Figure], dict]:
    """Dashboard API for TV figures and KPIs on a single run.

    Args:
        df:           Telemetry DataFrame (already filtered by load_data).
        corner_mask:  Optional boolean mask (same length as *df*) marking
                      cornering phase samples. When provided, replaces the
                      internal ay/steering filter. Falls back to the built-in
                      heuristic when None.
    """
    res = _compute_tv(_prepare_arrays_from_df(df, corner_mask))
    kpis = {
        "valid_laps": int(res["any_ok"].sum()),
        "mean_yaw_rmse": float(np.nanmean(res["yaw_rmse"][res["yaw_ok"]])) if res["yaw_ok"].any() else np.nan,
        "mean_yaw_bias": float(np.nanmean(res["yaw_bias"][res["yaw_ok"]])) if res["yaw_ok"].any() else np.nan,
        "mean_mz_rmse": float(np.nanmean(res["mz_rmse"][res["mz_ok"]])) if res["mz_ok"].any() else np.nan,
        "mean_mz_bias": float(np.nanmean(res["mz_bias"][res["mz_ok"]])) if res["mz_ok"].any() else np.nan,
        "mean_ratio": float(np.nanmean(res["ratio"][res["ratio_ok"]])) if res["ratio_ok"].any() else np.nan,
        "mean_fb_share": float(np.nanmean(res["fb_share"][res["ratio_ok"]])) if res["ratio_ok"].any() else np.nan,
        "mean_corner_coverage_pct": (
            float(np.nanmean(res["cover"][res["any_ok"]]) * 100.0) if res["any_ok"].any() else np.nan
        ),
        "table": res["table"],
        "warnings": res["warnings"],
    }
    return _build_tv_figures(res, x_mode=x_mode), kpis


def _print_tv_summary(kpis: dict) -> None:
    table = kpis["table"]
    if table.is_empty():
        print("\n─── TV ───")
        print("No valid cornering laps for TV KPIs.")
        return

    print("\n─── TV ───")
    print(table)


def main() -> None:
    res = _compute_tv(_prepare_arrays_from_csv())
    kpis = {"table": res["table"], "warnings": res["warnings"]}
    _print_tv_summary(kpis)
    for fig in _build_tv_figures(res):
        fig.show()


if __name__ == "__main__":
    main()
