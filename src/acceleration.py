"""acceleration.py
-----------------
Formula Student Acceleration event (75 m sprint) — diagnostic KPIs.

The 75 m sprint is decomposed into three phases:
    launch (0–~0.5 s) → traction (0–30 m) → power (45–75 m)

Each KPI answers one diagnostic question with a single number:
  * Where is time lost?            t_0_15m / t_15_45m / t_45_75m
  * Was the launch clean?          launch_ax_g_05s / launch_sr_peak
  * Is TC working?                 pct_sr_in_band_on_throttle / wheelspin_events
  * Is the power budget used?      p_dc_peak / pct_time_p_dc_70_80kw / pct_over_80kw
  * Is the battery up to it?       v_dc_sag_pct / i_dc_peak / energy_dc
  * Is the drivetrain balanced?    fr_torque_split / lr_imbalance_{front,rear}

Scatter figures show operating points (not signal traces):
  * sr_front_rear_scatter_fig      — F/R axle traction balance
  * ax_vs_vx_envelope_fig          — traction-limited → power-limited envelope
  * motor_torque_vs_wheel_speed_fig— motor torque-speed operating cloud
  * sr_vs_fx_per_wheel_fig         — TC operating point vs μ-slip curve
  * power_dc_vs_vx_scatter_fig     — DC power vs 80 kW regulatory ceiling

All public functions accept polars.DataFrame(s) pre-filtered to
acceleration-mode samples and return (go.Figure, dict). No Streamlit calls.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import FONT_FAMILY, driver_color, make_dark_figure
from dynamics import (
    WHEEL_RADIUS_M,
    GEAR_RATIO,
    T_MOTOR_MAX_NM,
    N_MOTORS,
    MAX_POWER_W,
    MASS_KG,
    CD_AERO,
    CL_AERO,
    A_AERO_M2,
    RHO_AIR_KGM3,
)

RPM_PER_RADS = 60.0 / (2.0 * np.pi)  # rad/s → rpm
DRIVELINE_EFF = 0.85  # motor+inverter+driveline efficiency (DC → wheel)
CRR_TIRE = 0.015  # rolling resistance coeff (matches powertrain.CRR_TIRE)

# ── Event constants ────────────────────────────────────────────────────────────
ACCEL_DISTANCE_M = 75.0
POWER_LIMIT_KW = 80.0  # FS-E regulation
POWER_TARGET_LO = 70.0  # Aprovechamiento del límite reglamentario
SR_TARGET = 0.20
SR_BAND_LO = 0.15
SR_BAND_HI = 0.25
SR_OVERSLIP = 0.30
MIN_SPEED_MPS = 1.0
THROTTLE_ON_PCT = 80.0
THROTTLE_FULL_PCT = 90.0
LAUNCH_WINDOW_S = 0.5
TRACTION_DIST_M = 30.0
WHEELSPIN_MIN_DURATION_S = 0.05
G_MPS2 = 9.81

WHEELS = ("FL", "FR", "RL", "RR")
_SR_COLS: dict[str, str] = dict(zip(WHEELS, ("Est_SRFL", "Est_SRFR", "Est_SRRL", "Est_SRRR")))
_ACTUAL_TRQ_COLS: dict[str, str] = dict(
    zip(WHEELS, ("FL_actualTorque", "FR_actualTorque", "RL_actualTorque", "RR_actualTorque"))
)
_ACTUAL_VEL_COLS: dict[str, str] = dict(
    zip(
        WHEELS, ("FL_actualVelocity", "FR_actualVelocity", "RL_actualVelocity", "RR_actualVelocity")
    )
)
_THROTTLE_ALIASES = ("APPS", "pedals_throttle", "Throttle", "APPS1")

_BG = "#141417"
_TEXT = "#EBEBEB"
_GRID = "rgba(128,128,128,0.2)"
_AXIS = "#E5E5E5"
_REF = "#F2D44D"


# ── Signal selection helpers ──────────────────────────────────────────────────


def is_acceleration_run(df: pl.DataFrame) -> bool:
    """True if the majority of samples are tagged as acceleration by lapcount."""
    if "lapcount_mode" not in df.columns or len(df) == 0:
        return False
    modes = [str(v).strip().lower() for v in df["lapcount_mode"].to_list()]
    return sum(v == "acceleration" for v in modes) > 0.5 * len(modes)


def _vx_col(cols: list[str]) -> str:
    return "Est_vxCOG" if "Est_vxCOG" in cols else "VN_vx"


def _ax_col(cols: list[str]) -> str:
    return "Filtering_VN_ax" if "Filtering_VN_ax" in cols else "VN_ax"


def _throttle_col(cols: list[str]) -> str | None:
    for c in _THROTTLE_ALIASES:
        if c in cols:
            return c
    return None


def _unique_laps(df: pl.DataFrame) -> list[int]:
    if "laps" not in df.columns:
        return []
    return sorted(int(v) for v in df["laps"].drop_nulls().unique().to_list() if v > 0)


def _lap_df(df: pl.DataFrame, lap_id: int) -> pl.DataFrame:
    return df.filter(pl.col("laps") == lap_id)


def _lap_distance_m(df_lap: pl.DataFrame) -> np.ndarray:
    """Cumulative distance from lap start [m]. Falls back to ∫|vx|dt when
    `dist_km` is missing, constant, or zero (some CSVs ship the column empty)."""
    if "dist_km" in df_lap.columns:
        d_km = df_lap["dist_km"].to_numpy().astype(float)
        d_m = (d_km - d_km[0]) * 1000.0
        if np.nanmax(d_m) > 1.0:
            return np.maximum(d_m, 0.0)
    t = df_lap["TimeStamp"].to_numpy().astype(float)
    dt = np.concatenate([[0.0], np.diff(t)])
    vx = df_lap[_vx_col(df_lap.columns)].to_numpy().astype(float)
    return np.maximum(np.cumsum(np.abs(vx) * dt), 0.0)


def _lap_time_rel(df_lap: pl.DataFrame) -> np.ndarray:
    """Time array relative to lap start [s]."""
    t = df_lap["TimeStamp"].to_numpy().astype(float)
    return t - t[0]


def _short_name(run_name: str) -> str:
    return run_name.rsplit("/", 1)[-1].replace(".csv", "")


def _apply_dark_layout(fig: go.Figure) -> None:
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG, font=dict(color=_TEXT, family=FONT_FAMILY)
    )
    fig.update_xaxes(gridcolor=_GRID, color=_AXIS, zerolinecolor=_GRID, linecolor=_AXIS)
    fig.update_yaxes(gridcolor=_GRID, color=_AXIS, zerolinecolor=_GRID, linecolor=_AXIS)


# ── KPI helpers ───────────────────────────────────────────────────────────────


def _all_sr_stacked(df_lap: pl.DataFrame) -> np.ndarray | None:
    """Stack 4-wheel SR into (4, N) array, or None if any column is missing."""
    cols = df_lap.columns
    if not all(_SR_COLS[w] in cols for w in WHEELS):
        return None
    return np.stack([df_lap[_SR_COLS[w]].to_numpy().astype(float) for w in WHEELS], axis=0)


def _count_wheelspin_events(max_sr: np.ndarray, dt_s: float) -> int:
    """Count contiguous segments where max-wheel SR > SR_OVERSLIP for ≥ 50 ms."""
    over = max_sr > SR_OVERSLIP
    min_samples = max(1, int(round(WHEELSPIN_MIN_DURATION_S / max(dt_s, 1e-3))))
    count = 0
    i = 0
    n = len(over)
    while i < n:
        if not over[i]:
            i += 1
            continue
        j = i
        while j < n and over[j]:
            j += 1
        if j - i >= min_samples:
            count += 1
        i = j
    return count


def _time_at_distance(dist_m: np.ndarray, t_rel: np.ndarray, target_m: float) -> float:
    """Return t when dist >= target_m, or NaN if never reached."""
    if dist_m[-1] < target_m:
        return float("nan")
    idx = int(np.searchsorted(dist_m, target_m))
    idx = int(np.clip(idx, 0, len(t_rel) - 1))
    return float(t_rel[idx])


def _time_at_speed(vx_mps: np.ndarray, t_rel: np.ndarray, target_kmh: float) -> float:
    """Return t when vx >= target_kmh, or NaN if never reached."""
    target_mps = target_kmh / 3.6
    above = vx_mps >= target_mps
    if not np.any(above):
        return float("nan")
    return float(t_rel[int(np.argmax(above))])


# ── Summary KPIs ──────────────────────────────────────────────────────────────


def summary_kpis(df: pl.DataFrame) -> dict:
    """
    Return scalar diagnostic KPIs for one acceleration run.

    Categories:
      event       — event_time_s, peak_vx_kmh, peak_ax_g, mean_ax_g, pct_full_thr
      phase split — t_0_15m_s, t_15_45m_s, t_45_75m_s, t_to_{30,60,100}kmh_s
      launch      — launch_ax_g_05s, launch_sr_peak, throttle_rise_time_s
      traction    — mean_ax_g_traction, pct_sr_in_band_on_throttle, wheelspin_events
      power       — p_dc_peak_kw, p_dc_mean_kw, pct_time_p_dc_70_80kw,
                    pct_time_p_dc_over_80kw, v_dc_sag_pct, i_dc_peak_a, energy_dc_kj
      drivetrain  — fr_torque_split_pct, lr_imbalance_{front,rear}_pct
      tc          — sr_mae_global, pct_all_in_band, pct_any_overslip, worst_wheel
    """
    warnings: list[str] = []
    cols = df.columns
    laps = _unique_laps(df)

    if not laps:
        return {"warnings": ["No valid acceleration laps found."]}

    lap_id = laps[0]
    lap = _lap_df(df, lap_id)
    if len(lap) < 10:
        return {"warnings": [f"Lap {lap_id}: too few samples ({len(lap)})."]}

    out: dict = {"lap_id": lap_id, "warnings": warnings}

    vx = lap[_vx_col(cols)].to_numpy().astype(float)
    ax = lap[_ax_col(cols)].to_numpy().astype(float)
    t_rel = _lap_time_rel(lap)
    dist_m = _lap_distance_m(lap)
    dt = float(np.median(np.diff(t_rel))) if len(t_rel) > 1 else 0.01
    moving = vx > MIN_SPEED_MPS

    # ── Event-level KPIs ──────────────────────────────────────────────────
    if "laptime" in cols:
        lt = lap["laptime"].to_numpy().astype(float)
        lt_valid = lt[np.isfinite(lt)]
        out["event_time_s"] = float(np.nanmean(lt_valid)) if len(lt_valid) > 0 else float(t_rel[-1])
    else:
        out["event_time_s"] = float(t_rel[-1])

    out["peak_vx_kmh"] = float(np.max(vx)) * 3.6
    if moving.any():
        out["peak_ax_g"] = float(np.max(ax[moving])) / G_MPS2
        out["mean_ax_g"] = float(np.mean(ax[moving])) / G_MPS2
    else:
        out["peak_ax_g"] = float("nan")
        out["mean_ax_g"] = float("nan")

    thr_col = _throttle_col(cols)
    apps = lap[thr_col].to_numpy().astype(float) if thr_col else None
    if apps is not None:
        out["pct_full_thr"] = (
            float(np.mean(apps[moving] > THROTTLE_FULL_PCT) * 100) if moving.any() else float("nan")
        )
    else:
        out["pct_full_thr"] = float("nan")
        warnings.append("Throttle column not found.")

    # ── Phase split by distance and by speed ──────────────────────────────
    t15 = _time_at_distance(dist_m, t_rel, 15.0)
    t45 = _time_at_distance(dist_m, t_rel, 45.0)
    t75 = _time_at_distance(dist_m, t_rel, ACCEL_DISTANCE_M)
    out["t_0_15m_s"] = t15
    out["t_15_45m_s"] = (t45 - t15) if (np.isfinite(t45) and np.isfinite(t15)) else float("nan")
    out["t_45_75m_s"] = (t75 - t45) if (np.isfinite(t75) and np.isfinite(t45)) else float("nan")
    out["t_to_30kmh_s"] = _time_at_speed(vx, t_rel, 30.0)
    out["t_to_60kmh_s"] = _time_at_speed(vx, t_rel, 60.0)
    out["t_to_100kmh_s"] = _time_at_speed(vx, t_rel, 100.0)

    # ── Launch (0–0.5 s) ──────────────────────────────────────────────────
    launch_mask = t_rel <= LAUNCH_WINDOW_S
    if launch_mask.any():
        out["launch_ax_g_05s"] = float(np.mean(ax[launch_mask])) / G_MPS2
    else:
        out["launch_ax_g_05s"] = float("nan")

    stacked = _all_sr_stacked(lap)
    if stacked is not None and launch_mask.any():
        out["launch_sr_peak"] = float(np.max(stacked[:, launch_mask]))
    else:
        out["launch_sr_peak"] = float("nan")

    if apps is not None and apps[0] < 10.0:
        idx10 = int(np.argmax(apps >= 10.0))
        rest = apps[idx10:]
        if (rest >= 90.0).any():
            idx90 = idx10 + int(np.argmax(rest >= 90.0))
            out["throttle_rise_time_s"] = float(t_rel[idx90] - t_rel[idx10])
        else:
            out["throttle_rise_time_s"] = float("nan")
    else:
        # APPS already > 10 % at lap start — pre-launch ramp not in window
        out["throttle_rise_time_s"] = float("nan")

    # ── Traction phase (0–30 m) ───────────────────────────────────────────
    traction_mask = (dist_m <= TRACTION_DIST_M) & moving
    if traction_mask.any():
        out["mean_ax_g_traction"] = float(np.mean(ax[traction_mask])) / G_MPS2
    else:
        out["mean_ax_g_traction"] = float("nan")

    if stacked is not None and apps is not None:
        sr_avg = np.mean(stacked, axis=0)
        on_thr = (apps > THROTTLE_ON_PCT) & moving
        if on_thr.any():
            in_band = (sr_avg >= SR_BAND_LO) & (sr_avg <= SR_BAND_HI)
            out["pct_sr_in_band_on_throttle"] = float(in_band[on_thr].mean() * 100)
        else:
            out["pct_sr_in_band_on_throttle"] = float("nan")

        max_sr = np.max(stacked, axis=0)
        out["wheelspin_events"] = int(_count_wheelspin_events(max_sr, dt))
    else:
        out["pct_sr_in_band_on_throttle"] = float("nan")
        out["wheelspin_events"] = 0
        if stacked is None:
            warnings.append("SR columns not found — TC KPIs unavailable.")

    # ── Power phase (DC bus) ──────────────────────────────────────────────
    if "Vbat" in cols and "Current" in cols:
        vbat = lap["Vbat"].to_numpy().astype(float)
        i_dc = lap["Current"].to_numpy().astype(float)
        p_dc = vbat * i_dc / 1000.0  # kW
        out["p_dc_peak_kw"] = float(np.max(p_dc))
        out["p_dc_mean_kw"] = float(np.mean(p_dc[moving])) if moving.any() else float("nan")
        out["pct_time_p_dc_70_80kw"] = float(
            np.mean((p_dc >= POWER_TARGET_LO) & (p_dc <= POWER_LIMIT_KW)) * 100
        )
        out["pct_time_p_dc_over_80kw"] = float(np.mean(p_dc > POWER_LIMIT_KW) * 100)

        v_max = float(np.max(vbat))
        v_min = float(np.min(vbat))
        out["v_dc_sag_pct"] = float((v_max - v_min) / v_max * 100) if v_max > 0 else float("nan")
        out["i_dc_peak_a"] = float(np.max(i_dc))

        t_abs = lap["TimeStamp"].to_numpy().astype(float)
        out["energy_dc_kj"] = float(np.trapezoid(p_dc, t_abs))  # kW·s = kJ
    else:
        for k in (
            "p_dc_peak_kw",
            "p_dc_mean_kw",
            "pct_time_p_dc_70_80kw",
            "pct_time_p_dc_over_80kw",
            "v_dc_sag_pct",
            "i_dc_peak_a",
            "energy_dc_kj",
        ):
            out[k] = float("nan")
        warnings.append("Vbat / Current not found — power KPIs unavailable.")

    # ── Drivetrain balance ────────────────────────────────────────────────
    if all(_ACTUAL_TRQ_COLS[w] in cols for w in WHEELS):
        trqs = {w: lap[_ACTUAL_TRQ_COLS[w]].to_numpy().astype(float) for w in WHEELS}
        front_arr = trqs["FL"] + trqs["FR"]
        rear_arr = trqs["RL"] + trqs["RR"]
        front_mean = float(np.mean(front_arr))
        rear_mean = float(np.mean(rear_arr))
        total = front_mean + rear_mean
        out["fr_torque_split_pct"] = (
            float(front_mean / total * 100) if abs(total) > 1e-3 else float("nan")
        )
        out["lr_imbalance_front_pct"] = (
            float(np.mean(np.abs(trqs["FL"] - trqs["FR"])) / max(abs(front_mean), 1e-3) * 100)
            if abs(front_mean) > 1e-3
            else float("nan")
        )
        out["lr_imbalance_rear_pct"] = (
            float(np.mean(np.abs(trqs["RL"] - trqs["RR"])) / max(abs(rear_mean), 1e-3) * 100)
            if abs(rear_mean) > 1e-3
            else float("nan")
        )
    else:
        out["fr_torque_split_pct"] = float("nan")
        out["lr_imbalance_front_pct"] = float("nan")
        out["lr_imbalance_rear_pct"] = float("nan")
        warnings.append("Actual torque columns not found — drivetrain KPIs unavailable.")

    # ── Legacy TC KPIs (kept for compatibility with existing UI) ──────────
    if stacked is not None:
        moving_idx = moving
        sr_maes: dict[str, float] = {}
        for k, w in enumerate(WHEELS):
            sr_m = stacked[k][moving_idx]
            mae = float(np.mean(np.abs(sr_m - SR_TARGET))) if len(sr_m) else float("nan")
            sr_maes[w] = mae
            out[f"sr_mae_{w}"] = mae
        out["sr_mae_global"] = float(np.mean(list(sr_maes.values()))) if sr_maes else float("nan")
        sr_moving = stacked[:, moving_idx]
        if sr_moving.shape[1] > 0:
            out["pct_all_in_band"] = float(
                np.mean(np.all((sr_moving >= SR_BAND_LO) & (sr_moving <= SR_BAND_HI), axis=0)) * 100
            )
            out["pct_any_overslip"] = float(np.mean(np.any(sr_moving > SR_OVERSLIP, axis=0)) * 100)
        else:
            out["pct_all_in_band"] = float("nan")
            out["pct_any_overslip"] = float("nan")
        out["worst_wheel"] = max(sr_maes, key=lambda k: sr_maes[k])
    else:
        out["sr_mae_global"] = float("nan")
        out["pct_all_in_band"] = float("nan")
        out["pct_any_overslip"] = float("nan")
        out["worst_wheel"] = ""

    return out


# ── Figure 1 — DC Power vs Speed (80 kW regulatory limit) ─────────────────────


def power_dc_vs_vx_scatter_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """
    DC bus power vs longitudinal speed.

    Diagnostic: in the power-limited phase (≈30–75 m) the cloud should hug the
    80 kW horizontal ceiling. A cloud sitting well below = battery sag / current
    cap is leaving power on the table. A cloud above 80 kW = regulatory breach.
    """
    fig = make_dark_figure(
        title=f"DC Power vs Speed  ·  FS-E limit {POWER_LIMIT_KW:.0f} kW",
        xlabel="vx [km/h]",
        ylabel="P_DC = Vbat × Current  [kW]",
    )
    warnings: list[str] = []

    fig.add_hrect(
        y0=POWER_TARGET_LO,
        y1=POWER_LIMIT_KW,
        fillcolor="rgba(100,200,100,0.10)",
        line_width=0,
        annotation_text="target zone  70–80 kW",
        annotation_position="bottom left",
        annotation_font_color="rgba(180,255,180,0.65)",
    )
    fig.add_hline(
        y=POWER_LIMIT_KW,
        line_color="#F25050",
        line_width=2,
        annotation_text=f"{POWER_LIMIT_KW:.0f} kW limit",
        annotation_position="top left",
        annotation_font_color="#F25050",
    )

    y_max_seen = POWER_LIMIT_KW * 1.15
    for run_name, df in dfs.items():
        color = driver_color(run_name)
        laps = _unique_laps(df)
        if not laps:
            continue
        cols = df.columns
        if "Vbat" not in cols or "Current" not in cols:
            warnings.append(f"{_short_name(run_name)}: Vbat / Current missing.")
            continue
        short = _short_name(run_name)
        for i_lap, lap_id in enumerate(laps):
            lap = _lap_df(df, lap_id)
            vx = lap[_vx_col(cols)].to_numpy().astype(float)
            v = lap["Vbat"].to_numpy().astype(float)
            i = lap["Current"].to_numpy().astype(float)
            p = v * i / 1000.0
            mask = vx > MIN_SPEED_MPS
            if not mask.any():
                continue
            y_max_seen = max(y_max_seen, float(np.max(p[mask])))
            label = short if len(laps) == 1 else f"{short} L{lap_id}"
            fig.add_trace(
                go.Scatter(
                    x=vx[mask] * 3.6,
                    y=p[mask],
                    mode="markers",
                    marker=dict(color=color, size=4, opacity=0.55),
                    name=label,
                    legendgroup=label,
                    showlegend=(i_lap == 0),
                )
            )

    fig.update_yaxes(range=[0, max(POWER_LIMIT_KW * 1.15, y_max_seen + 5)])
    fig.update_layout(legend=dict(orientation="h", y=-0.15), height=460)
    return fig, {"warnings": warnings}


# ── Figure 2 — Traction Balance (Front vs Rear SR) ───────────────────────────


def sr_front_rear_scatter_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """
    Front axle SR vs rear axle SR (per-sample), restricted to on-throttle samples.

    Diagnostic: a cluster on the diagonal y=x inside the green [0.15, 0.25]²
    square = balanced traction. Off-diagonal cluster = one axle is wasted.
    Above diagonal = rear overslipping; below diagonal = front overslipping.
    """
    fig = make_dark_figure(
        title="Traction Balance  ·  Front axle SR vs Rear axle SR  (APPS > 80 %)",
        xlabel="Front SR  ·  (SR_FL + SR_FR) / 2",
        ylabel="Rear SR  ·  (SR_RL + SR_RR) / 2",
    )
    warnings: list[str] = []

    fig.add_shape(
        type="rect",
        x0=SR_BAND_LO,
        x1=SR_BAND_HI,
        y0=SR_BAND_LO,
        y1=SR_BAND_HI,
        line=dict(width=0),
        fillcolor="rgba(100,200,100,0.10)",
        layer="below",
    )
    fig.add_trace(
        go.Scatter(
            x=[-0.05, 0.55],
            y=[-0.05, 0.55],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.30)", dash="dash", width=1),
            name="balanced (F = R)",
            showlegend=True,
        )
    )
    fig.add_vline(x=SR_TARGET, line_color=_REF, line_dash="dot", line_width=1)
    fig.add_hline(y=SR_TARGET, line_color=_REF, line_dash="dot", line_width=1)

    for run_name, df in dfs.items():
        color = driver_color(run_name)
        laps = _unique_laps(df)
        if not laps:
            continue
        cols = df.columns
        if not all(_SR_COLS[w] in cols for w in WHEELS):
            warnings.append(f"{_short_name(run_name)}: SR columns missing.")
            continue
        thr_col = _throttle_col(cols)
        if thr_col is None:
            warnings.append(f"{_short_name(run_name)}: throttle column missing.")
            continue

        short = _short_name(run_name)
        for i_lap, lap_id in enumerate(laps):
            lap = _lap_df(df, lap_id)
            vx = lap[_vx_col(cols)].to_numpy().astype(float)
            apps = lap[thr_col].to_numpy().astype(float)
            sr_f = (
                lap[_SR_COLS["FL"]].to_numpy().astype(float)
                + lap[_SR_COLS["FR"]].to_numpy().astype(float)
            ) / 2.0
            sr_r = (
                lap[_SR_COLS["RL"]].to_numpy().astype(float)
                + lap[_SR_COLS["RR"]].to_numpy().astype(float)
            ) / 2.0
            mask = (apps > THROTTLE_ON_PCT) & (vx > MIN_SPEED_MPS)
            if not mask.any():
                continue
            label = short if len(laps) == 1 else f"{short} L{lap_id}"
            fig.add_trace(
                go.Scatter(
                    x=sr_f[mask],
                    y=sr_r[mask],
                    mode="markers",
                    marker=dict(color=color, size=5, opacity=0.55),
                    name=label,
                    legendgroup=label,
                    showlegend=(i_lap == 0),
                )
            )

    fig.update_xaxes(range=[-0.05, 0.55])
    fig.update_yaxes(range=[-0.05, 0.55])
    fig.update_layout(legend=dict(orientation="h", y=-0.15), height=460)
    return fig, {"warnings": warnings}


# ── Figure 3 — Acceleration Envelope (ax vs vx) ──────────────────────────────


def ax_vs_vx_envelope_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """
    Longitudinal G vs speed — the core acceleration envelope.

    Two regimes are visible:
      * low speed  → traction-limited plateau (grip / TC sets the ceiling)
      * high speed → power-limited roll-off, ax ≈ P / (m·v) as the 80 kW DC
                     ceiling bites

    The dashed curve is the achievable power-limited bound, net of driveline
    losses (η = 0.85 DC→wheel), aero drag and rolling resistance:
        F_down = ½ρ·|CL|·A·v²
        a = (η·P_DC/v − ½ρ·Cd·A·v² − Crr·(m·g + F_down)) / m
    The cloud should ride just under it once power-limited. Diagnostic for
    run-to-run comparison: at a given speed, more ax = faster. A run sagging
    below the curve early is leaving power on the table (battery sag / current
    cap); a run short of the plateau at low speed is traction-limited (wheelspin).
    """
    fig = make_dark_figure(
        title="Acceleration Envelope  ·  Longitudinal G vs Speed",
        xlabel="vx [km/h]",
        ylabel="ax [G]",
    )
    warnings: list[str] = []

    peak_ax_seen = 0.0
    vx_max_kmh = 60.0
    for run_name, df in dfs.items():
        color = driver_color(run_name)
        laps = _unique_laps(df)
        if not laps:
            continue
        cols = df.columns
        short = _short_name(run_name)
        for i_lap, lap_id in enumerate(laps):
            lap = _lap_df(df, lap_id)
            vx = lap[_vx_col(cols)].to_numpy().astype(float)
            ax_g = lap[_ax_col(cols)].to_numpy().astype(float) / G_MPS2
            mask = (vx > MIN_SPEED_MPS) & (ax_g > 0)
            if not mask.any():
                continue
            peak_ax_seen = max(peak_ax_seen, float(np.max(ax_g[mask])))
            vx_max_kmh = max(vx_max_kmh, float(np.max(vx[mask])) * 3.6)
            label = short if len(laps) == 1 else f"{short} L{lap_id}"
            fig.add_trace(
                go.Scatter(
                    x=vx[mask] * 3.6,
                    y=ax_g[mask],
                    mode="markers",
                    marker=dict(color=color, size=4, opacity=0.5),
                    name=label,
                    legendgroup=label,
                    showlegend=(i_lap == 0),
                )
            )

    # Achievable power-limited bound, net of driveline losses, drag and rolling:
    #   F_down = ½ρ·|CL|·A·v² ; a = (η·P/v − ½ρ·Cd·A·v² − Crr·(m·g + F_down)) / m
    v_kmh = np.linspace(15.0, vx_max_kmh * 1.02, 200)
    v_mps = v_kmh / 3.6
    f_tract = DRIVELINE_EFF * MAX_POWER_W / v_mps
    f_drag = 0.5 * RHO_AIR_KGM3 * CD_AERO * A_AERO_M2 * v_mps**2
    f_down = 0.5 * RHO_AIR_KGM3 * abs(CL_AERO) * A_AERO_M2 * v_mps**2
    f_roll = CRR_TIRE * (MASS_KG * G_MPS2 + f_down)
    a_power_g = (f_tract - f_drag - f_roll) / MASS_KG / G_MPS2
    fig.add_trace(
        go.Scatter(
            x=v_kmh,
            y=a_power_g,
            mode="lines",
            line=dict(color=_REF, dash="dash", width=1.5),
            name=f"power-limited  ·  {DRIVELINE_EFF * MAX_POWER_W / 1000:.0f} kW mech − drag − roll",
            legendgroup="ref",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    if peak_ax_seen > 0:
        fig.add_hline(
            y=peak_ax_seen,
            line_color="rgba(255,255,255,0.30)",
            line_dash="dot",
            line_width=1,
            annotation_text=f"peak ax  {peak_ax_seen:.2f} G",
            annotation_position="top right",
            annotation_font_color="rgba(235,235,235,0.7)",
        )
    fig.update_yaxes(range=[0, max(1.3, peak_ax_seen * 1.15)])
    fig.update_layout(legend=dict(orientation="h", y=-0.15), height=460)
    return fig, {"warnings": warnings}


# ── Figure 4 — Motor Torque-Speed Operating Cloud ────────────────────────────


def motor_torque_vs_wheel_speed_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """
    Motor torque vs wheel speed — the powertrain operating cloud (all 4 motors
    pooled per run, on-throttle tractive samples only).

    `*_actualVelocity` logs motor angular speed [rad/s]; it is converted to
    wheel RPM (÷ gear ratio). `*_actualTorque` is per-motor torque [Nm].

    Two capability bounds frame the cloud:
      * horizontal at 27.5 Nm   → constant-torque ceiling (max motor torque)
      * hyperbola 17 kW/motor   → constant-power ceiling, mechanical:
                                   τ = P_mech / ω_motor with
                                   P_mech = η · 80 kW ÷ 4  (η = 0.85)

    Diagnostic: the cloud should pin the 27.5 Nm ceiling at low rpm (full torque
    off the line) then bend down along the power hyperbola. A cloud short of the
    torque ceiling at low rpm = torque not delivered (TC cutting / fault); a
    cloud well under the power hyperbola at high rpm = power left unused.
    """
    fig = make_dark_figure(
        title="Powertrain Operating Cloud  ·  Motor Torque vs Wheel Speed  (APPS > 80 %)",
        xlabel="Wheel speed [rpm]",
        ylabel="Motor torque [Nm]",
    )
    warnings: list[str] = []

    rpm_max_seen = 100.0
    for run_name, df in dfs.items():
        color = driver_color(run_name)
        laps = _unique_laps(df)
        if not laps:
            continue
        cols = df.columns
        if not all(_ACTUAL_VEL_COLS[w] in cols and _ACTUAL_TRQ_COLS[w] in cols for w in WHEELS):
            warnings.append(f"{_short_name(run_name)}: motor velocity / torque columns missing.")
            continue
        thr_col = _throttle_col(cols)
        short = _short_name(run_name)
        for i_lap, lap_id in enumerate(laps):
            lap = _lap_df(df, lap_id)
            vx = lap[_vx_col(cols)].to_numpy().astype(float)
            if thr_col is not None:
                apps = lap[thr_col].to_numpy().astype(float)
                base = (apps > THROTTLE_ON_PCT) & (vx > MIN_SPEED_MPS)
            else:
                base = vx > MIN_SPEED_MPS
            # Pool all four motors
            rpm_all: list[np.ndarray] = []
            trq_all: list[np.ndarray] = []
            for w in WHEELS:
                omega_motor = lap[_ACTUAL_VEL_COLS[w]].to_numpy().astype(float)  # rad/s
                wheel_rpm = omega_motor / GEAR_RATIO * RPM_PER_RADS
                trq = lap[_ACTUAL_TRQ_COLS[w]].to_numpy().astype(float)
                m = base & (trq > 0.0)
                if m.any():
                    rpm_all.append(wheel_rpm[m])
                    trq_all.append(trq[m])
            if not rpm_all:
                continue
            x = np.concatenate(rpm_all)
            y = np.concatenate(trq_all)
            rpm_max_seen = max(rpm_max_seen, float(np.max(x)))
            label = short if len(laps) == 1 else f"{short} L{lap_id}"
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(color=color, size=3, opacity=0.45),
                    name=label,
                    legendgroup=label,
                    showlegend=(i_lap == 0),
                )
            )

    # Constant-torque ceiling
    fig.add_hline(
        y=T_MOTOR_MAX_NM,
        line_color="#F25050",
        line_dash="dash",
        line_width=1.5,
        annotation_text=f"{T_MOTOR_MAX_NM:.1f} Nm  max torque",
        annotation_position="top right",
        annotation_font_color="#F25050",
    )
    # Constant-power hyperbola (mechanical): τ_motor = P_mech / ω_motor,
    #   P_mech = η · 80 kW ÷ 4  (driveline losses already removed)
    p_motor_w = DRIVELINE_EFF * MAX_POWER_W / N_MOTORS
    rpm_grid = np.linspace(1.0, rpm_max_seen * 1.02, 300)
    omega_motor_grid = rpm_grid / RPM_PER_RADS * GEAR_RATIO  # wheel rpm → motor rad/s
    tau_grid = np.minimum(p_motor_w / np.maximum(omega_motor_grid, 1e-3), T_MOTOR_MAX_NM * 1.5)
    fig.add_trace(
        go.Scatter(
            x=rpm_grid,
            y=tau_grid,
            mode="lines",
            line=dict(color=_REF, dash="dot", width=1.5),
            name=f"{p_motor_w / 1000:.0f} kW/motor mech  (η·80 kW ÷ 4)",
            legendgroup="ref",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    fig.update_yaxes(range=[0, T_MOTOR_MAX_NM * 1.25])
    fig.update_xaxes(range=[0, rpm_max_seen * 1.05])
    fig.update_layout(legend=dict(orientation="h", y=-0.15), height=460)
    return fig, {"warnings": warnings}


# ── Figure 4 — TC Operating Point (SR vs Fx per Wheel) ───────────────────────


def sr_vs_fx_per_wheel_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """
    Slip ratio vs longitudinal tyre force per wheel (APPS > 80 % only).

    Fx = τ_motor × i_gear / R_wheel  ·  estimated from per-motor actual torque.
    Diagnostic: each wheel cloud should sit in the green band SR ∈ [0.15, 0.25]
    while Fx > 0. Clusters drifting past SR = 0.30 = TC not capping in time
    (wheelspin). Clusters short of SR = 0.15 with small Fx = torque is being
    left on the table.
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=list(WHEELS),
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
        shared_yaxes=True,
    )
    warnings: list[str] = []
    _pos = {"FL": (1, 1), "FR": (1, 2), "RL": (2, 1), "RR": (2, 2)}

    for w, (r, c) in _pos.items():
        fig.add_vrect(
            x0=SR_BAND_LO,
            x1=SR_BAND_HI,
            fillcolor="rgba(100,200,100,0.10)",
            line_width=0,
            row=r,
            col=c,
        )
        fig.add_vline(x=SR_TARGET, line_color=_REF, line_dash="dash", line_width=1, row=r, col=c)
        fig.add_vline(
            x=SR_OVERSLIP,
            line_color="rgba(255,80,80,0.5)",
            line_dash="dot",
            line_width=1,
            row=r,
            col=c,
        )

    for run_name, df in dfs.items():
        color = driver_color(run_name)
        laps = _unique_laps(df)
        if not laps:
            continue
        cols = df.columns
        if not all(_SR_COLS[w] in cols and _ACTUAL_TRQ_COLS[w] in cols for w in WHEELS):
            warnings.append(f"{_short_name(run_name)}: SR or torque columns missing.")
            continue
        thr_col = _throttle_col(cols)
        short = _short_name(run_name)

        for i_lap, lap_id in enumerate(laps):
            lap = _lap_df(df, lap_id)
            vx = lap[_vx_col(cols)].to_numpy().astype(float)
            if thr_col is not None:
                apps = lap[thr_col].to_numpy().astype(float)
                mask = (apps > THROTTLE_ON_PCT) & (vx > MIN_SPEED_MPS)
            else:
                mask = vx > MIN_SPEED_MPS
            if not mask.any():
                continue

            label = short if len(laps) == 1 else f"{short} L{lap_id}"
            for w, (r, c) in _pos.items():
                sr = lap[_SR_COLS[w]].to_numpy().astype(float)
                trq = lap[_ACTUAL_TRQ_COLS[w]].to_numpy().astype(float)
                fx = trq * GEAR_RATIO / WHEEL_RADIUS_M
                fig.add_trace(
                    go.Scatter(
                        x=sr[mask],
                        y=fx[mask],
                        mode="markers",
                        marker=dict(color=color, size=3, opacity=0.5),
                        name=label,
                        legendgroup=label,
                        showlegend=(w == "FL" and i_lap == 0),
                    ),
                    row=r,
                    col=c,
                )

    fig.update_xaxes(range=[-0.05, 0.55], title_text="Slip Ratio", row=2)
    fig.update_yaxes(title_text="Fx [N]", col=1)
    fig.update_layout(
        title="TC Operating Point  ·  SR vs Fx per Wheel  (APPS > 80 %)",
        template="plotly_dark",
        height=560,
        legend=dict(orientation="h", y=-0.12),
    )
    _apply_dark_layout(fig)
    return fig, {"warnings": warnings}
