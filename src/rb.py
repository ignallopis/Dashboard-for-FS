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

SR_TARGET_BRAKE = -0.20
DELTA_SR = 0.05
BRAKE_THRESHOLD = 5.0
AX_BRAKE_THRESHOLD = -0.50
MIN_SPEED = 4.0
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
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
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


def _build_rb_figures(res: dict, x_mode: str = "laps") -> list[go.Figure]:
    lap_list = res["lap_list"]
    ok = res["valid_ok"]
    figs: list[go.Figure] = []
    lt_val = res["lt_val"]
    x_arr, order, xlabel = per_lap_axis(lap_list[ok], lt_val[ok], x_mode) if ok.any() else (np.array([]), np.array([], dtype=int), "Lap")

    fig = make_dark_figure(f"Braking SR MAE vs {'Lap Time' if x_mode == 'laptime' else 'Lap'}", xlabel, "SR MAE")
    if ok.any():
        add_lap_scatter(fig, x_arr, res["sr_mae"][ok][order], lap_list[ok][order], color="#FFD700")
        if x_mode == "laps":
            fig.update_xaxes(tickvals=np.sort(lap_list[ok].astype(int)))
    figs.append(fig)

    fig = make_dark_figure(f"Braking SR Target Tracking vs {'Lap Time' if x_mode == 'laptime' else 'Lap'}", xlabel, "% of active RB time")
    if ok.any():
        add_lap_scatter(fig, x_arr, res["in_target_pct"][ok][order], lap_list[ok][order], name="In target", color="#73D973")
        add_lap_scatter(fig, x_arr, res["overslip_pct"][ok][order], lap_list[ok][order], name="Overslip", color="#F27070", symbol="square")
        add_lap_scatter(fig, x_arr, res["underslip_pct"][ok][order], lap_list[ok][order], name="Underslip", color="#4DB3F2", symbol="diamond")
        if x_mode == "laps":
            fig.update_xaxes(tickvals=np.sort(lap_list[ok].astype(int)))
    figs.append(fig)

    fig = make_dark_figure(
        f"RB Intensity Target vs {'Lap Time' if x_mode == 'laptime' else 'Lap'}",
        xlabel,
        "RB intensity target",
    )
    if ok.any():
        add_lap_scatter(
            fig,
            x_arr,
            res["intensity_mean"][ok][order],
            lap_list[ok][order],
            color="#D973D9",
        )
        if x_mode == "laps":
            fig.update_xaxes(tickvals=np.sort(lap_list[ok].astype(int)))
    figs.append(fig)

    fig = make_dark_figure("Per-Wheel SR Error vs Time (active RB only)", "Time [s]", "SR error (SR - target)")
    t_rb = res["time"][res["rb_mask"]]
    for w in WHEELS:
        fig.add_trace(go.Scatter(
            x=t_rb,
            y=res["sr"][w][res["rb_mask"]] - SR_TARGET_BRAKE,
            mode="lines",
            name=w,
            line=dict(color=WHEEL_COLORS[w], width=1.0),
        ))
    fig.add_hline(y=0.0, line=dict(color="rgba(200,200,200,0.5)", dash="dash"))
    fig.add_hline(y=DELTA_SR, line=dict(color="rgba(200,200,200,0.3)", dash="dot"))
    fig.add_hline(y=-DELTA_SR, line=dict(color="rgba(200,200,200,0.3)", dash="dot"))
    figs.append(fig)

    return figs


def rb_figs_kpis(
    df: pl.DataFrame,
    brake_mask: np.ndarray | None = None,
    x_mode: str = "laps",
) -> tuple[list[go.Figure], dict]:
    """Dashboard API for RB figures and KPIs on a single run.

    Args:
        df:          Telemetry DataFrame (already filtered by load_data).
        brake_mask:  Optional boolean mask (same length as *df*) marking
                     braking phase samples. When provided, replaces the
                     internal Brake/ax filter. Falls back to the built-in
                     heuristic when None.
    """
    res = _compute_rb(_prepare_arrays_from_df(df, brake_mask))
    ok = res["valid_ok"]

    kpis = {
        "valid_laps": int(ok.sum()),
        "mean_sr_mae": float(np.nanmean(res["sr_mae"][ok])) if ok.any() else np.nan,
        "mean_sr_bias": float(np.nanmean(res["sr_bias"][ok])) if ok.any() else np.nan,
        "mean_in_target_pct": float(np.nanmean(res["in_target_pct"][ok])) if ok.any() else np.nan,
        "mean_overslip_pct": float(np.nanmean(res["overslip_pct"][ok])) if ok.any() else np.nan,
        "mean_underslip_pct": float(np.nanmean(res["underslip_pct"][ok])) if ok.any() else np.nan,
        "mean_rb_cover_pct": float(np.nanmean(res["rb_cover"][ok]) * 100.0) if ok.any() else np.nan,
        "mean_intensity_target": float(np.nanmean(res["intensity_mean"][ok])) if ok.any() else np.nan,
        "table": res["table"],
        "notes": res.get("notes", []),
        "warnings": res["warnings"],
    }
    return _build_rb_figures(res, x_mode=x_mode), kpis


def main() -> None:
    res = _compute_rb(_prepare_arrays_from_csv())
    if res["table"].is_empty():
        print("\n─── RB ───")
        print("No valid active RB laps for RB KPIs.")
    else:
        print("\n─── RB ───")
        print(res["table"])
    for fig in _build_rb_figures(res):
        fig.show()


if __name__ == "__main__":
    main()
