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
from plotly.subplots import make_subplots

from src import cornering
from src.dynamics import IZ_KGM2

from utils import (
    COMPLETE_LAPS_MARKER,
    add_lap_scatter,
    add_trend_line,
    add_zero_line,
    cols_to_numpy,
    ensure_complete_laps_df,
    exclude_lap0_and_last_lap,
    keep_min_duration_segments,
    lap_dist_from_gps,
    make_dark_figure,
    per_lap_axis_or_empty,
    robust_dt,
    series_or_nan,
    smooth_signal,
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
    return cols_to_numpy(df, columns)


def _from_df(df: pl.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    df = ensure_complete_laps_df(df)
    cols = list(columns)
    if COMPLETE_LAPS_MARKER in df.columns and COMPLETE_LAPS_MARKER not in cols:
        cols.append(COMPLETE_LAPS_MARKER)
    return cols_to_numpy(df, cols)


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


def _add_turn_direction_annotation(fig: go.Figure, ay_samples: np.ndarray) -> None:
    """Annotate a corner-sample scatter with the left/right turn counts."""
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Left turns: {int((ay_samples > 0).sum())}<br>Right turns: {int((ay_samples < 0).sum())}",
        showarrow=False,
        align="left",
        font=dict(color="#EBEBEB", size=10),
        bgcolor="rgba(20,20,23,0.8)",
    )


def _build_tv_figures(res: dict, x_mode: str = "laps") -> list[go.Figure]:
    lap_list = res["lap_list"]
    lt = res["lt"]
    figs: list[go.Figure] = []

    vs = "Lap Time" if x_mode == "laptime" else "Lap"
    x_yaw, order_yaw, xlabel_yaw = per_lap_axis_or_empty(lap_list, lt, x_mode, res["yaw_ok"])
    fig = make_dark_figure(f"Yaw Rate Tracking Error vs {vs}", xlabel_yaw, "Yaw rate error RMSE")
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
    _add_turn_direction_annotation(fig, ay_s)
    if ay_s.size > 0:
        add_zero_line(fig, ay_s)
    figs.append(fig)

    x_mz, order_mz, xlabel_mz = per_lap_axis_or_empty(lap_list, lt, x_mode, res["mz_ok"])
    fig = make_dark_figure(f"Mz Tracking Error vs {vs}", xlabel_mz, "Mz error RMSE [Nm]")
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
    _add_turn_direction_annotation(fig, ay_s)
    if ay_s.size > 0:
        add_zero_line(fig, ay_s)
    figs.append(fig)

    x_ratio, order_ratio, xlabel_ratio = per_lap_axis_or_empty(lap_list, lt, x_mode, res["ratio_ok"])
    fig = make_dark_figure(f"Feedback to Feedforward Ratio vs {vs}", xlabel_ratio, "FB / FF ratio")
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


def yaw_moment_balance_fig(df: pl.DataFrame) -> tuple[go.Figure, dict]:
    """Closed-loop yaw moment diagnostic using Iz * d(VN_gz)/dt.

    Compares TV desired Mz, TV actual Mz, and the inertial yaw moment implied by
    measured yaw acceleration. Metrics are evaluated in cornering samples.
    """
    df = ensure_complete_laps_df(df)
    ay_col = _ay_signal(df.columns)
    vx_col = _vx_signal(df.columns)
    cols = [
        "TimeStamp", "laps", "laptime",
        "Steering", ay_col, vx_col,
        "VN_gz",
        "TV_desiredMz", "TV_actualMz",
        "TV_feedForwardMz", "TV_feedBackMz",
    ]
    has_limit = "TV_limitMz" in df.columns
    if has_limit:
        cols.append("TV_limitMz")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing yaw moment balance columns: {missing}")

    arr = cols_to_numpy(df, cols)
    dist_m = lap_dist_from_gps(df)
    valid = np.all(np.stack([np.isfinite(arr[c]) for c in cols], axis=1), axis=1) & np.isfinite(dist_m)
    arr = {c: v[valid] for c, v in arr.items()}
    dist_m = dist_m[valid]
    if arr["TimeStamp"].size < 3:
        raise ValueError("Not enough valid samples for yaw moment balance.")

    time_s = arr["TimeStamp"] - arr["TimeStamp"][0]
    dt = robust_dt(time_s)
    yaw_rate = smooth_signal(arr["VN_gz"], max(1, int(round(0.05 / dt))))
    mz_inertial = IZ_KGM2 * np.gradient(yaw_rate, dt)
    cm = _corner_mask(arr[ay_col], arr["Steering"], arr[vx_col], dt)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.13,
        subplot_titles=(
            "Desired / actual / inertial yaw moment vs distance",
            "Desired vs actual yaw moment",
        ),
    )
    signal_specs = (
        ("TV_desiredMz", "Desired Mz", "#F28C40", "solid"),
        ("TV_actualMz", "Actual Mz", "#73D973", "solid"),
        ("__mz_inertial", "Inertial Iz·dψ̇/dt", "#4DB3F2", "dot"),
    )
    series = {**arr, "__mz_inertial": mz_inertial}
    showlegend = {name: True for _key, name, _color, _dash in signal_specs}
    dash_cycle = ("solid", "dash", "dot", "dashdot", "longdash")
    for lap_i, lap in enumerate(unique_laps(arr["laps"])):
        lm = arr["laps"] == lap
        if not lm.any():
            continue
        order = np.argsort(dist_m[lm])
        for key, name, color, dash in signal_specs:
            fig.add_trace(go.Scattergl(
                x=dist_m[lm][order],
                y=series[key][lm][order],
                mode="lines",
                name=name,
                legendgroup=name,
                showlegend=showlegend[name],
                line=dict(color=color, width=1.2, dash=dash if lap_i == 0 else dash_cycle[lap_i % len(dash_cycle)]),
                hovertemplate=f"L{int(lap)}<br>distance=%{{x:.1f}} m<br>Mz=%{{y:.1f}} Nm<extra></extra>",
            ), row=1, col=1)
            showlegend[name] = False

    scatter_mask = cm & np.isfinite(arr["TV_desiredMz"]) & np.isfinite(arr["TV_actualMz"])
    if scatter_mask.any():
        fig.add_trace(go.Scattergl(
            x=arr["TV_desiredMz"][scatter_mask],
            y=arr["TV_actualMz"][scatter_mask],
            mode="markers",
            marker=dict(color=np.abs(arr[ay_col][scatter_mask]), colorscale="Turbo", size=4, opacity=0.50, colorbar=dict(title="|ay| [m/s²]")),
            name="Corner samples",
        ), row=2, col=1)
        lo = float(np.nanmin([np.nanmin(arr["TV_desiredMz"][scatter_mask]), np.nanmin(arr["TV_actualMz"][scatter_mask])]))
        hi = float(np.nanmax([np.nanmax(arr["TV_desiredMz"][scatter_mask]), np.nanmax(arr["TV_actualMz"][scatter_mask])]))
        fig.add_trace(go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(color="rgba(235,235,235,0.7)", dash="dash", width=1.2),
            name="Ideal",
        ), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"Yaw moment closed-loop diagnostic · Iz={IZ_KGM2:.3f} kg·m²", font=dict(size=14, color="#EBEBEB")),
        paper_bgcolor="#141417",
        plot_bgcolor="#141417",
        font=dict(color="#EBEBEB", size=11),
        legend=dict(bgcolor="rgba(20,20,23,0.85)", bordercolor="rgba(128,128,128,0.3)"),
        height=760,
        margin=dict(l=70, r=40, t=80, b=60),
    )
    fig.update_xaxes(title_text="Distance [m]", color="#E5E5E5", gridcolor="rgba(128,128,128,0.2)", row=1, col=1)
    fig.update_yaxes(title_text="Mz [Nm]", color="#E5E5E5", gridcolor="rgba(128,128,128,0.2)", row=1, col=1)
    fig.update_xaxes(title_text="Desired Mz [Nm]", color="#E5E5E5", gridcolor="rgba(128,128,128,0.2)", row=2, col=1)
    fig.update_yaxes(title_text="Actual Mz [Nm]", color="#E5E5E5", gridcolor="rgba(128,128,128,0.2)", row=2, col=1)

    if scatter_mask.any():
        tracking_err = arr["TV_actualMz"][scatter_mask] - arr["TV_desiredMz"][scatter_mask]
        inertial_err = mz_inertial[scatter_mask] - arr["TV_actualMz"][scatter_mask]
        ff_abs = np.abs(arr["TV_feedForwardMz"][scatter_mask])
        fb_abs = np.abs(arr["TV_feedBackMz"][scatter_mask])
        if has_limit:
            limit = np.abs(arr["TV_limitMz"][scatter_mask])
            saturated_pct = float(np.nanmean((limit > 1.0) & (np.abs(arr["TV_actualMz"][scatter_mask]) >= 0.98 * limit)) * 100.0)
        else:
            saturated_pct = np.nan
        warnings = []
    else:
        tracking_err = inertial_err = ff_abs = fb_abs = np.array([], dtype=float)
        saturated_pct = np.nan
        warnings = ["No valid cornering samples for yaw moment balance."]

    kpis = {
        "tracking_rmse_nm": float(np.sqrt(np.nanmean(tracking_err ** 2))) if tracking_err.size else np.nan,
        "inertial_rmse_nm": float(np.sqrt(np.nanmean(inertial_err ** 2))) if inertial_err.size else np.nan,
        "saturated_pct": saturated_pct,
        "ff_over_fb_ratio": float(np.nanmean(ff_abs) / (np.nanmean(fb_abs) + EPS_RATIO)) if ff_abs.size else np.nan,
        "fb_share": float(np.nanmean(fb_abs) / (np.nanmean(ff_abs) + np.nanmean(fb_abs) + EPS_RATIO)) if ff_abs.size else np.nan,
        "corner_samples": int(scatter_mask.sum()),
        "warnings": warnings,
    }
    return fig, kpis


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


# ═══════════════════════════════════════════════════════════════════════════════
# Function check  —  is TV delivering yaw moment so the car turns as commanded?
# ═══════════════════════════════════════════════════════════════════════════════

def tv_function_kpis(df: pl.DataFrame) -> tuple[list[go.Figure], dict]:
    """Function-level check for Torque Vectoring.

    Pregunta: ¿está el TV añadiendo Mz para que el coche gire como pide el piloto?
    """
    df = ensure_complete_laps_df(df)
    needed = ["TimeStamp", "laps", "laptime",
              "TV_desiredYawRate", "TV_errorYawRate",
              "TV_desiredMz", "TV_actualMz",
              "Steering"]
    ay_col = _ay_signal(df.columns)
    vx_col = _vx_signal(df.columns)
    needed += [ay_col, vx_col]
    arr = cols_to_numpy(df, needed)
    valid = np.all(np.stack([np.isfinite(arr[c]) for c in needed], axis=1), axis=1)
    arr = {c: v[valid] for c, v in arr.items()}

    time_s = arr["TimeStamp"] - arr["TimeStamp"][0]
    laps = arr["laps"]
    keep = laps > 0
    arr = {c: v[keep] for c, v in arr.items()}
    time_s = time_s[keep]
    if arr["TimeStamp"].size == 0:
        raise ValueError("No valid samples for TV function check.")
    last_lap = unique_laps(arr["laps"]).max()
    keep = arr["laps"] != last_lap
    arr = {c: v[keep] for c, v in arr.items()}
    time_s = time_s[keep]

    dt = robust_dt(time_s)
    cm = _corner_mask(arr[ay_col], arr["Steering"], arr[vx_col], dt)

    yaw_ref = arr["TV_desiredYawRate"]
    yaw_err = arr["TV_errorYawRate"]
    yaw_real = yaw_ref - yaw_err  # error = ref − real (per TorqueVectoring.cpp)
    mz_ref = arr["TV_desiredMz"]
    mz_act = arr["TV_actualMz"]

    if cm.any():
        yaw_rmse = float(np.sqrt(np.nanmean(yaw_err[cm] ** 2)))
        mz_rmse = float(np.sqrt(np.nanmean((mz_act[cm] - mz_ref[cm]) ** 2)))
        mz_delivered = float(np.nanmedian(np.abs(mz_act[cm])))

        steer_corner = arr["Steering"][cm]
        left = steer_corner > 0.05
        right = steer_corner < -0.05
        yaw_bias_left = float(np.nanmean(yaw_err[cm][left])) if left.any() else np.nan
        yaw_bias_right = float(np.nanmean(yaw_err[cm][right])) if right.any() else np.nan
        bias_lr = (
            yaw_bias_left - yaw_bias_right
            if np.isfinite(yaw_bias_left) and np.isfinite(yaw_bias_right) else np.nan
        )
    else:
        yaw_rmse = mz_rmse = mz_delivered = np.nan
        yaw_bias_left = yaw_bias_right = bias_lr = np.nan

    # ── Fig 1: yaw_real vs yaw_ref overlay along time ────────────────────────
    fig_yaw = make_dark_figure(
        title="Yaw rate: real vs reference",
        xlabel="Time [s]",
        ylabel="Yaw rate [rad/s]",
    )
    fig_yaw.add_trace(go.Scattergl(
        x=time_s, y=yaw_ref, mode="lines", name="Reference (TV)",
        line=dict(color="#F28C40", width=1.4),
    ))
    fig_yaw.add_trace(go.Scattergl(
        x=time_s, y=yaw_real, mode="lines", name="Real (IMU)",
        line=dict(color="#4DB3F2", width=1.0),
    ))

    # ── Fig 2: Mz_actual vs Mz_ref overlay along time ────────────────────────
    fig_mz = make_dark_figure(
        title="TV yaw moment: requested vs delivered",
        xlabel="Time [s]",
        ylabel="Mz [Nm]",
    )
    fig_mz.add_trace(go.Scattergl(
        x=time_s, y=mz_ref, mode="lines", name="Mz desired",
        line=dict(color="#F28C40", width=1.4),
    ))
    fig_mz.add_trace(go.Scattergl(
        x=time_s, y=mz_act, mode="lines", name="Mz actual",
        line=dict(color="#73D973", width=1.0),
    ))

    # ── Fig 3: Yaw error vs lateral acceleration in corners ──────────────────
    fig_err_ay = make_dark_figure(
        title="Yaw error vs |Ay| in corners",
        xlabel="|Ay| [m/s²]",
        ylabel="Yaw error (ref − real) [rad/s]",
    )
    if cm.any():
        fig_err_ay.add_trace(go.Scattergl(
            x=np.abs(arr[ay_col][cm]), y=yaw_err[cm],
            mode="markers",
            marker=dict(color="#9B59B6", size=4, opacity=0.45),
            name="Corners",
        ))
        fig_err_ay.add_hline(y=0.0,
                             line=dict(color="rgba(200,200,200,0.6)",
                                       dash="dash", width=1.2))

    kpis = {
        "yaw_rmse": yaw_rmse,
        "mz_delivered_med": mz_delivered,
        "mz_delivery_rmse": mz_rmse,
        "yaw_bias_left": yaw_bias_left,
        "yaw_bias_right": yaw_bias_right,
        "yaw_bias_lr_diff": bias_lr,
    }
    return [fig_yaw, fig_mz, fig_err_ay], kpis


# ═══════════════════════════════════════════════════════════════════════════════
# Yaw-rate triple — real (IMU) vs desired (TV) vs steering-implied (kinematic)
# ═══════════════════════════════════════════════════════════════════════════════

WHEELBASE_M = 1.53  # [m]


def yaw_rate_triple_fig(df: pl.DataFrame) -> tuple[go.Figure, dict]:
    """Compare measured yaw rate, TV desired yaw rate, and the kinematic
    yaw rate implied by steering wheel angle.

    Kinematic implied:  ψ̇_kin = δ · vx / L   (bicycle model, low-speed limit)

    Useful as a fast diagnosis:
      * desired vs implied → quality of TV's reference model
      * real vs desired   → tracking quality of TV
      * real vs implied   → on-limit understeer / oversteer
    """
    df = ensure_complete_laps_df(df)
    needed = ['TimeStamp', 'laps', 'laptime',
              'TV_desiredYawRate', 'TV_errorYawRate',
              'Steering']
    vx_col = _vx_signal(df.columns)
    yaw_imu_col = 'VN_gz' if 'VN_gz' in df.columns else None
    cols = list(needed) + [vx_col]
    if yaw_imu_col is not None:
        cols.append(yaw_imu_col)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for yaw triple: {missing}")

    arr = cols_to_numpy(df, cols)
    valid = np.all(np.stack([np.isfinite(arr[c]) for c in cols], axis=1), axis=1)
    arr = {c: v[valid] for c, v in arr.items()}
    if arr['TimeStamp'].size == 0:
        raise ValueError("No valid samples for yaw triple.")

    time_s = arr['TimeStamp'] - arr['TimeStamp'][0]

    desired = arr['TV_desiredYawRate']
    # IMU-measured yaw rate: prefer VN_gz; fall back to ref - err.
    if yaw_imu_col is not None:
        real = arr[yaw_imu_col]
    else:
        real = desired - arr['TV_errorYawRate']

    vx = arr[vx_col]
    steer = arr['Steering']  # steering-pot value [rad], used directly (no STEERING_RATIO)
    with np.errstate(divide='ignore', invalid='ignore'):
        implied = steer * vx / WHEELBASE_M
    implied = np.where(np.abs(vx) >= MIN_SPEED, implied, np.nan)

    fig = make_dark_figure(
        title='Yaw rate · real (IMU) vs desired (TV) vs steering-implied',
        xlabel='Time [s]',
        ylabel='Yaw rate [rad/s]',
    )
    fig.add_trace(go.Scattergl(
        x=time_s, y=desired, mode='lines',
        name='Desired (TV)',
        line=dict(color='#F28C40', width=1.5),
    ))
    fig.add_trace(go.Scattergl(
        x=time_s, y=real, mode='lines',
        name='Real (IMU)',
        line=dict(color='#4DB3F2', width=1.1),
    ))
    fig.add_trace(go.Scattergl(
        x=time_s, y=implied, mode='lines',
        name='Steering-implied (δ·vx/L)',
        line=dict(color='#73D973', width=1.0, dash='dot'),
    ))
    fig.add_hline(y=0.0, line=dict(color='rgba(200,200,200,0.4)', dash='dot', width=1))

    finite_all = np.isfinite(real) & np.isfinite(desired) & np.isfinite(implied)
    if finite_all.any():
        rmse_real_desired   = float(np.sqrt(np.nanmean((real - desired) ** 2)))
        rmse_real_implied   = float(np.sqrt(np.nanmean((real - implied) ** 2)))
        rmse_desired_implied = float(np.sqrt(np.nanmean((desired - implied) ** 2)))
    else:
        rmse_real_desired = rmse_real_implied = rmse_desired_implied = np.nan

    kpis = {
        'rmse_real_vs_desired':   rmse_real_desired,
        'rmse_real_vs_implied':   rmse_real_implied,
        'rmse_desired_vs_implied': rmse_desired_implied,
        'imu_source': yaw_imu_col or 'derived from TV_errorYawRate',
        'warnings': [],
    }
    return fig, kpis


TV_TORQUE_COL_ALIASES = {
    "FL": ("TV_FL_Trq", "tv_fl_torque"),
    "FR": ("TV_FR_Trq", "tv_fr_torque"),
    "RL": ("TV_RL_Trq", "tv_rl_torque"),
    "RR": ("TV_RR_Trq", "tv_rr_torque"),
}
TV_SIGNAL_ALIASES = {
    "desired_yaw": ("TV_desiredYawRate", "tv_desired_yaw_rate"),
    "error_yaw": ("TV_errorYawRate", "tv_error_yaw_rate"),
    "desired_mz": ("TV_desiredMz", "tv_desired_mz"),
    "actual_mz": ("TV_actualMz", "tv_actualmz", "tv_actual_mz"),
    "error_mz": ("TV_errorMz", "tv_error_mz"),
    "limit_mz": ("TV_limitMz", "tv_limit_mz"),
    "desired_fx": ("TV_desiredFx", "tv_desired_fx"),
    "actual_fx": ("TV_actualFx", "tv_actualfx", "tv_actual_fx"),
    "error_fx": ("TV_errorFx", "tv_error_fx"),
    "exit_flag": ("TV_exitflag", "TV_exitFlag", "tv_exit_flag"),
    "command": ("LLC_Command", "llc_command"),
}
GEAR_RATIO = 9.05
WHEEL_RADIUS_M = 0.2032
FRONT_TRACK_M = 1.225
REAR_TRACK_M = 1.175
YAW_BALANCE_MIN_EXPECTED_RADPS = 0.15
YAW_BALANCE_NEUTRAL_BAND_PCT = 5.0


def tv_control_attribution_figs_kpis(df: pl.DataFrame) -> tuple[list[go.Figure], dict]:
    """Observable TV behaviour: yaw gain, balance and rotation response."""
    df = ensure_complete_laps_df(df)
    if len(df) == 0:
        raise ValueError("No valid samples for TV behaviour metrics.")

    time_s = df["TimeStamp"].to_numpy().astype(float) - float(df["TimeStamp"][0])
    dist_m = lap_dist_from_gps(df)
    steering = df["Steering"].to_numpy().astype(float)  # steering-pot value [rad]; no STEERING_RATIO
    ay_col = _ay_signal(df.columns)
    vx_col = _vx_signal(df.columns)
    ay = df[ay_col].to_numpy().astype(float)
    vx = df[vx_col].to_numpy().astype(float)
    dt = robust_dt(time_s)
    corner_mask = _corner_mask(ay, steering, vx, dt)

    desired_yaw = series_or_nan(df, TV_SIGNAL_ALIASES["desired_yaw"])
    yaw_error = series_or_nan(df, TV_SIGNAL_ALIASES["error_yaw"])
    if "VN_gz" in df.columns:
        yaw_real = df["VN_gz"].to_numpy().astype(float)
    else:
        yaw_real = desired_yaw - yaw_error
    actual_mz = series_or_nan(df, TV_SIGNAL_ALIASES["actual_mz"])
    torque_mat = np.stack(
        [series_or_nan(df, TV_TORQUE_COL_ALIASES[w]) for w in ("FL", "FR", "RL", "RR")],
        axis=1,
    )
    lr_delta_nm = (torque_mat[:, 1] + torque_mat[:, 3]) - (torque_mat[:, 0] + torque_mat[:, 2])

    yaw_implied = steering * vx / WHEELBASE_M
    yaw_gain = np.divide(
        yaw_real,
        yaw_implied,
        out=np.full_like(yaw_real, np.nan, dtype=float),
        where=np.abs(yaw_implied) > 0.08,
    )
    yaw_gain = np.where(np.abs(yaw_gain) < 4.0, yaw_gain, np.nan)
    balance_error_pct = (yaw_gain - 1.0) * 100.0
    eval_mask = (
        corner_mask
        & np.isfinite(yaw_gain)
        & np.isfinite(actual_mz)
        & (np.abs(vx) >= MIN_SPEED)
    )
    understeer = eval_mask & (yaw_gain < 0.90)
    oversteer = eval_mask & (yaw_gain > 1.10)
    balanced = eval_mask & (yaw_gain >= 0.90) & (yaw_gain <= 1.10)

    yaw_accel = np.gradient(yaw_real, time_s, edge_order=1)
    steer_rate = np.gradient(steering, time_s, edge_order=1)
    entry_mask = (
        eval_mask
        & (np.abs(steer_rate) > 0.08)
        & (np.abs(yaw_implied) > 0.08)
    )
    rotation_response = np.abs(yaw_accel)

    def _corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
        valid = mask & np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) < 20:
            return np.nan
        return float(np.corrcoef(x[valid], y[valid])[0, 1])

    mz_balance_corr = _corr(actual_mz, balance_error_pct, eval_mask)
    lr_balance_corr = _corr(lr_delta_nm, balance_error_pct, eval_mask)

    laps = df["laps"].to_numpy().astype(float)
    lap_rows: list[dict[str, object]] = []
    for lap in unique_laps(laps):
        lm = laps == lap
        lcm = lm & eval_mask
        if int(lcm.sum()) < MIN_CORNER_SAMPLES:
            continue
        entry_lap = lm & entry_mask
        lap_rows.append({
            "Lap": int(lap),
            "Corner samples": int(lcm.sum()),
            "Yaw gain median": round(float(np.nanmedian(yaw_gain[lcm])), 3),
            "Understeer [%]": round(float(np.nanmean(understeer[lcm]) * 100.0), 1),
            "Oversteer [%]": round(float(np.nanmean(oversteer[lcm]) * 100.0), 1),
            "Balanced [%]": round(float(np.nanmean(balanced[lcm]) * 100.0), 1),
            "Entry rotation [rad/s²]": round(float(np.nanmedian(rotation_response[entry_lap])), 3) if entry_lap.any() else np.nan,
            "Mz vs balance corr": round(_corr(actual_mz, balance_error_pct, lcm), 3),
        })

    order = np.argsort(dist_m)
    fig_balance = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=(
            "Yaw gain: measured yaw / steering-implied yaw",
            "Balance error: negative = understeer, positive = oversteer",
            "TV yaw moment and left/right torque delta",
        ),
    )
    fig_balance.add_trace(go.Scattergl(
        x=dist_m[order], y=yaw_gain[order], mode="markers", name="Yaw gain",
        marker=dict(color=np.abs(ay[order]), colorscale="Turbo", size=3, opacity=0.55, colorbar=dict(title="|ay|")),
    ), row=1, col=1)
    fig_balance.add_hrect(y0=0.90, y1=1.10, fillcolor="rgba(115,217,115,0.12)", line_width=0, row=1, col=1)
    fig_balance.add_hline(y=1.0, line=dict(color="rgba(235,235,235,0.7)", dash="dash", width=1.2), row=1, col=1)
    fig_balance.add_trace(go.Scattergl(
        x=dist_m[order], y=balance_error_pct[order], mode="markers", name="Balance error",
        marker=dict(color="#F28C40", size=3, opacity=0.50),
    ), row=2, col=1)
    fig_balance.add_hline(y=0.0, line=dict(color="rgba(235,235,235,0.7)", dash="dash", width=1.2), row=2, col=1)
    fig_balance.add_trace(go.Scattergl(
        x=dist_m[order], y=actual_mz[order], mode="lines", name="TV actual Mz",
        line=dict(color="#73D973", width=1.1),
    ), row=3, col=1)
    fig_balance.add_trace(go.Scattergl(
        x=dist_m[order], y=lr_delta_nm[order], mode="lines", name="TV right-left torque",
        line=dict(color="#4DB3F2", width=1.0, dash="dot"),
    ), row=3, col=1)
    fig_balance.update_layout(
        title=dict(text="TV observable balance and rotation", font=dict(size=14, color="#EBEBEB")),
        paper_bgcolor="#141417",
        plot_bgcolor="#141417",
        font=dict(color="#EBEBEB", size=11),
        height=830,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0.0),
    )
    fig_balance.update_xaxes(title_text="Distance [m]", row=3, col=1, gridcolor="rgba(128,128,128,0.2)")
    fig_balance.update_yaxes(title_text="Yaw gain [-]", row=1, col=1, range=[0.0, 2.0], gridcolor="rgba(128,128,128,0.2)")
    fig_balance.update_yaxes(title_text="Balance error [%]", row=2, col=1, gridcolor="rgba(128,128,128,0.2)")
    fig_balance.update_yaxes(title_text="Nm", row=3, col=1, gridcolor="rgba(128,128,128,0.2)")

    fig_rel = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.10,
        subplot_titles=("TV Mz vs balance", "Lateral acceleration vs yaw gain"),
    )
    rel_mask = eval_mask & np.isfinite(balance_error_pct)
    fig_rel.add_trace(go.Scattergl(
        x=actual_mz[rel_mask],
        y=balance_error_pct[rel_mask],
        mode="markers",
        name="Mz -> balance",
        marker=dict(color=vx[rel_mask], colorscale="Turbo", size=4, opacity=0.45, colorbar=dict(title="vx")),
    ), row=1, col=1)
    fig_rel.add_trace(go.Scattergl(
        x=np.abs(ay[rel_mask]),
        y=yaw_gain[rel_mask],
        mode="markers",
        name="ay -> yaw gain",
        marker=dict(color=actual_mz[rel_mask], colorscale="RdBu", size=4, opacity=0.45, colorbar=dict(title="Mz")),
    ), row=1, col=2)
    fig_rel.add_hline(y=0.0, line=dict(color="rgba(235,235,235,0.5)", dash="dash", width=1.0), row=1, col=1)
    fig_rel.add_hline(y=1.0, line=dict(color="rgba(235,235,235,0.5)", dash="dash", width=1.0), row=1, col=2)
    fig_rel.update_layout(
        title=dict(text="TV variable relationships", font=dict(size=14, color="#EBEBEB")),
        paper_bgcolor="#141417",
        plot_bgcolor="#141417",
        font=dict(color="#EBEBEB", size=11),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0.0),
    )
    fig_rel.update_xaxes(title_text="TV actual Mz [Nm]", row=1, col=1, gridcolor="rgba(128,128,128,0.2)")
    fig_rel.update_yaxes(title_text="Balance error [%]", row=1, col=1, gridcolor="rgba(128,128,128,0.2)")
    fig_rel.update_xaxes(title_text="|ay| [m/s²]", row=1, col=2, gridcolor="rgba(128,128,128,0.2)")
    fig_rel.update_yaxes(title_text="Yaw gain [-]", row=1, col=2, range=[0.0, 2.0], gridcolor="rgba(128,128,128,0.2)")

    kpis = {
        "corner_samples": int(eval_mask.sum()),
        "yaw_gain_median": float(np.nanmedian(yaw_gain[eval_mask])) if eval_mask.any() else np.nan,
        "understeer_pct": float(np.nanmean(understeer[eval_mask]) * 100.0) if eval_mask.any() else np.nan,
        "oversteer_pct": float(np.nanmean(oversteer[eval_mask]) * 100.0) if eval_mask.any() else np.nan,
        "balanced_pct": float(np.nanmean(balanced[eval_mask]) * 100.0) if eval_mask.any() else np.nan,
        "entry_rotation_radss": float(np.nanmedian(rotation_response[entry_mask])) if entry_mask.any() else np.nan,
        "mz_balance_corr": mz_balance_corr,
        "lr_balance_corr": lr_balance_corr,
        "table": pl.DataFrame(lap_rows) if lap_rows else pl.DataFrame(),
        "warnings": [],
    }
    return [fig_balance, fig_rel], kpis


def _turn_mask_from_reference(
    df: pl.DataFrame,
    reference_turns: list[cornering.TurnDef],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Project Lap Analysis reference turns onto every valid lap."""
    d = cornering.compute_radius_curvature(df)
    laps = d["laps"]
    mask = np.zeros(len(laps), dtype=bool)
    turn_id = np.full(len(laps), np.nan, dtype=float)
    lap_ids = unique_laps(laps)
    if lap_ids.size == 0:
        return d, mask, turn_id
    valid_laps = lap_ids[(lap_ids > 0) & (lap_ids != np.nanmax(lap_ids))]
    for lap in valid_laps:
        lap_mask = laps == lap
        if not lap_mask.any():
            continue
        for turn in reference_turns:
            tm = (
                lap_mask
                & (d["s_lap_m"] >= float(turn.s_entry_m))
                & (d["s_lap_m"] <= float(turn.s_exit_m))
            )
            mask |= tm
            turn_id[tm] = float(turn.turn_id)
    return d, mask, turn_id


def tv_corner_under_oversteer_figs_kpis(
    df: pl.DataFrame,
    reference_turns: list[cornering.TurnDef] | None = None,
    *,
    reference_label: str = "",
) -> tuple[list[go.Figure], dict]:
    """Understeer/oversteer score per radius-detected corner.

    Balance [%] = 100 * (measured yaw rate / path yaw rate - 1).
    Negative values mean less yaw than the path demands; positive values mean
    more yaw than the path demands.
    """
    df = ensure_complete_laps_df(df)
    required = ["TimeStamp", "laps", "laptime"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing TV corner balance columns: {missing}")

    if reference_turns is None:
        d_ref = cornering.compute_radius_curvature(df)
        lap_ids = unique_laps(d_ref["laps"])
        valid_laps = lap_ids[(lap_ids > 0) & (lap_ids != np.nanmax(lap_ids))] if lap_ids.size else np.array([])
        if valid_laps.size == 0:
            reference_turns = []
        else:
            best_lap = min(
                valid_laps,
                key=lambda lap: float(np.nanmax(d_ref["laptime"][d_ref["laps"] == lap])),
            )
            reference_turns = cornering.detect_turns_on_lap(d_ref, "", int(best_lap))
            reference_label = reference_label or f"lap {int(best_lap)}"

    radius, corner_mask, turn_id = _turn_mask_from_reference(df, reference_turns)
    n = len(df)
    if len(radius["time_s"]) != n:
        raise ValueError("Radius-based corner arrays do not match TV telemetry length.")

    laps = df["laps"].to_numpy().astype(float)
    laptime = df["laptime"].to_numpy().astype(float)
    if "VN_gz" in df.columns:
        yaw_real = df["VN_gz"].to_numpy().astype(float)
    else:
        desired_yaw = series_or_nan(df, TV_SIGNAL_ALIASES["desired_yaw"])
        yaw_error = series_or_nan(df, TV_SIGNAL_ALIASES["error_yaw"])
        yaw_real = desired_yaw - yaw_error
    actual_mz = series_or_nan(df, TV_SIGNAL_ALIASES["actual_mz"])

    vx = radius["vx_mps"]
    path_yaw = vx * radius["signed_curvature_smooth_inv_m"]
    base_mask = (
        corner_mask
        & np.isfinite(yaw_real)
        & np.isfinite(path_yaw)
        & (np.abs(path_yaw) >= YAW_BALANCE_MIN_EXPECTED_RADPS)
        & np.isfinite(vx)
        & (np.abs(vx) >= MIN_SPEED)
    )
    if base_mask.any():
        sign_alignment = float(np.nanmedian(yaw_real[base_mask] * path_yaw[base_mask]))
        if np.isfinite(sign_alignment) and sign_alignment < 0.0:
            path_yaw = -path_yaw

    eval_mask = (
        corner_mask
        & np.isfinite(yaw_real)
        & np.isfinite(path_yaw)
        & (np.abs(path_yaw) >= YAW_BALANCE_MIN_EXPECTED_RADPS)
        & np.isfinite(vx)
        & (np.abs(vx) >= MIN_SPEED)
    )
    yaw_gain = np.divide(
        yaw_real,
        path_yaw,
        out=np.full_like(yaw_real, np.nan, dtype=float),
        where=eval_mask,
    )
    yaw_gain = np.where((yaw_gain >= -1.0) & (yaw_gain <= 3.0), yaw_gain, np.nan)
    balance_pct = (yaw_gain - 1.0) * 100.0
    eval_mask &= np.isfinite(balance_pct)
    understeer = eval_mask & (balance_pct < -YAW_BALANCE_NEUTRAL_BAND_PCT)
    oversteer = eval_mask & (balance_pct > YAW_BALANCE_NEUTRAL_BAND_PCT)

    lap_rows: list[dict[str, object]] = []
    turn_rows: list[dict[str, object]] = []
    lap_ids = unique_laps(laps)
    valid_laps = lap_ids[(lap_ids > 0) & (lap_ids != np.nanmax(lap_ids))] if lap_ids.size else np.array([])
    for lap in valid_laps:
        lm = laps == lap
        lcm = lm & eval_mask
        if int(lcm.sum()) >= 20:
            lap_rows.append({
                "Lap": int(lap),
                "LapTime [s]": round(float(np.nanmax(laptime[lm])), 3) if lm.any() else np.nan,
                "Samples": int(lcm.sum()),
                "Balance [%]": round(float(np.nanmedian(balance_pct[lcm])), 1),
                "Understeer samples [%]": round(float(np.nanmean(understeer[lcm]) * 100.0), 1),
                "Oversteer samples [%]": round(float(np.nanmean(oversteer[lcm]) * 100.0), 1),
            })
        for tid in sorted({int(t) for t in turn_id[lm & np.isfinite(turn_id)]}):
            tm = lm & (turn_id == float(tid)) & eval_mask
            if int(tm.sum()) < 10:
                continue
            score = float(np.nanmedian(balance_pct[tm]))
            if score < -YAW_BALANCE_NEUTRAL_BAND_PCT:
                status = "Understeer"
            elif score > YAW_BALANCE_NEUTRAL_BAND_PCT:
                status = "Oversteer"
            else:
                status = "Neutral"
            turn_rows.append({
                "Lap": int(lap),
                "Turn": int(tid),
                "Status": status,
                "Balance [%]": round(score, 1),
                "Understeer samples [%]": round(float(np.nanmean(understeer[tm]) * 100.0), 1),
                "Oversteer samples [%]": round(float(np.nanmean(oversteer[tm]) * 100.0), 1),
                "Expected yaw [rad/s]": round(float(np.nanmedian(path_yaw[tm])), 3),
                "Real yaw [rad/s]": round(float(np.nanmedian(yaw_real[tm])), 3),
                "TV Mz [Nm]": round(float(np.nanmedian(actual_mz[tm])), 1) if np.isfinite(actual_mz[tm]).any() else np.nan,
                "Distance [m]": round(float(np.nanmedian(radius["s_lap_m"][tm])), 1),
                "Samples": int(tm.sum()),
            })

    turn_summary_rows: list[dict[str, object]] = []
    for tid in sorted({int(row["Turn"]) for row in turn_rows}):
        rows = [row for row in turn_rows if int(row["Turn"]) == tid]
        scores = np.array([float(row["Balance [%]"]) for row in rows], dtype=float)
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            continue
        score = float(np.nanmedian(scores))
        if score < -YAW_BALANCE_NEUTRAL_BAND_PCT:
            status = "Understeer"
        elif score > YAW_BALANCE_NEUTRAL_BAND_PCT:
            status = "Oversteer"
        else:
            status = "Neutral"
        turn_summary_rows.append({
            "Turn": tid,
            "Status": status,
            "Balance [%]": round(score, 1),
            "P25 [%]": round(float(np.nanpercentile(scores, 25)), 1),
            "P75 [%]": round(float(np.nanpercentile(scores, 75)), 1),
            "Distance [m]": round(float(np.nanmedian([float(row["Distance [m]"]) for row in rows])), 1),
            "Laps": int(len(rows)),
            "Worst lap": int(rows[int(np.nanargmax(np.abs([float(row["Balance [%]"]) for row in rows])))]["Lap"]),
        })

    score_values = np.array([float(row["Balance [%]"]) for row in turn_summary_rows], dtype=float)
    under_corners = score_values < -YAW_BALANCE_NEUTRAL_BAND_PCT
    over_corners = score_values > YAW_BALANCE_NEUTRAL_BAND_PCT
    neutral_corners = np.isfinite(score_values) & ~under_corners & ~over_corners

    fig_turn = go.Figure()
    if turn_summary_rows:
        x = [int(row["Turn"]) for row in turn_summary_rows]
        y = np.array([float(row["Balance [%]"]) for row in turn_summary_rows], dtype=float)
        p25 = np.array([float(row["P25 [%]"]) for row in turn_summary_rows], dtype=float)
        p75 = np.array([float(row["P75 [%]"]) for row in turn_summary_rows], dtype=float)
        colors = [
            "#4DB3F2" if value < -YAW_BALANCE_NEUTRAL_BAND_PCT
            else "#F28C40" if value > YAW_BALANCE_NEUTRAL_BAND_PCT
            else "#D8D8D8"
            for value in y
        ]
        custom = [
            [row["Status"], row["Distance [m]"], row["P25 [%]"], row["P75 [%]"], row["Laps"], row["Worst lap"]]
            for row in turn_summary_rows
        ]
        fig_turn.add_trace(go.Bar(
            x=x,
            y=y,
            customdata=custom,
            marker=dict(color=colors, line=dict(color="rgba(235,235,235,0.35)", width=0.8)),
            error_y=dict(
                type="data",
                symmetric=False,
                array=np.maximum(0.0, p75 - y),
                arrayminus=np.maximum(0.0, y - p25),
                color="rgba(235,235,235,0.55)",
                thickness=1.1,
                width=3,
            ),
            hovertemplate=(
                "Turn %{x}<br>"
                "Status: %{customdata[0]}<br>"
                "Balance: %{y:+.1f}%<br>"
                "P25/P75: %{customdata[2]:+.1f}% / %{customdata[3]:+.1f}%<br>"
                "Distance: %{customdata[1]:.1f} m<br>"
                "Laps used: %{customdata[4]}<br>"
                "Worst lap: %{customdata[5]}<extra></extra>"
            ),
            name="Turn balance",
        ))
        fig_turn.add_hrect(
            y0=-YAW_BALANCE_NEUTRAL_BAND_PCT,
            y1=YAW_BALANCE_NEUTRAL_BAND_PCT,
            fillcolor="rgba(216,216,216,0.12)",
            line_width=0,
        )
        fig_turn.add_hline(y=0.0, line=dict(color="rgba(235,235,235,0.7)", width=1.2))
        fig_turn.add_hline(
            y=-YAW_BALANCE_NEUTRAL_BAND_PCT,
            line=dict(color="#4DB3F2", dash="dash", width=1.1),
            annotation_text="understeer",
            annotation_position="bottom right",
        )
        fig_turn.add_hline(
            y=YAW_BALANCE_NEUTRAL_BAND_PCT,
            line=dict(color="#F28C40", dash="dash", width=1.1),
            annotation_text="oversteer",
            annotation_position="top right",
        )
    fig_turn.update_layout(
        title=dict(
            text=(
                "TV understeer / oversteer by Lap Analysis turn"
                + (f" ({reference_label})" if reference_label else "")
            ),
            font=dict(size=14, color="#EBEBEB"),
        ),
        paper_bgcolor="#141417",
        plot_bgcolor="#141417",
        font=dict(color="#EBEBEB", size=11),
        height=520,
        showlegend=False,
        bargap=0.28,
    )
    fig_turn.update_xaxes(title_text="Turn", dtick=1, gridcolor="rgba(128,128,128,0.2)")
    fig_turn.update_yaxes(
        title_text="Balance [%]  |  negative = understeer, positive = oversteer",
        gridcolor="rgba(128,128,128,0.2)",
        zeroline=False,
    )

    kpis = {
        "corners": int(len(turn_summary_rows)),
        "median_balance_pct": float(np.nanmedian(score_values)) if score_values.size else np.nan,
        "understeer_corners_pct": float(np.nanmean(under_corners) * 100.0) if score_values.size else np.nan,
        "oversteer_corners_pct": float(np.nanmean(over_corners) * 100.0) if score_values.size else np.nan,
        "neutral_corners_pct": float(np.nanmean(neutral_corners) * 100.0) if score_values.size else np.nan,
        "sample_understeer_pct": float(np.nanmean(understeer[eval_mask]) * 100.0) if eval_mask.any() else np.nan,
        "sample_oversteer_pct": float(np.nanmean(oversteer[eval_mask]) * 100.0) if eval_mask.any() else np.nan,
        "lap_table": pl.DataFrame(lap_rows) if lap_rows else pl.DataFrame(),
        "turn_table": pl.DataFrame(turn_summary_rows) if turn_summary_rows else pl.DataFrame(),
        "turn_lap_table": pl.DataFrame(turn_rows) if turn_rows else pl.DataFrame(),
        "reference_label": reference_label,
        "warnings": [] if turn_rows else ["No radius-detected TV corners with enough yaw demand."],
    }
    return [fig_turn], kpis


if __name__ == "__main__":
    main()
