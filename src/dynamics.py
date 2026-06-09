"""dynamics.py
------------
Vehicle dynamics KPIs (understeer, load transfer, roll/pitch, dampers, envelopes…).

Usage:
    python src/dynamics.py                    — standalone CLI (loads from CSV_PATH)
    understeer_angle_fig(df)                  — dashboard (takes polars DataFrame)
"""

from __future__ import annotations
from typing import Literal
import numpy as np
import polars as pl
import plotly.graph_objects as go

from utils import (
    COMPLETE_LAPS_MARKER,
    make_dark_figure,
    add_lap_scatter,
    add_trend_line,
    add_zero_line,
    cols_to_numpy,
    driver_color,
    ensure_complete_laps_df,
    lap_dist_from_gps,
    keep_min_duration_segments,
    exclude_lap0_and_last_lap,
    robust_dt,
    smooth_signal,
    unique_laps,
    phase_masks_for_map,
    per_lap_axis,
    WHEEL_COLORS,
)

CSV_PATH = "data/run4_2025-08-24.csv"

# ── Shared filter parameters ──────────────────────────────────────────────────
AY_THRESHOLD = 2.0  # [m/s²] min |ay| to classify as cornering
STEERING_THRESHOLD = 0.05  # [rad]  min |steering| for cornering
MIN_SPEED = 3.0  # [m/s]  min vehicle speed
MIN_CORNER_DURATION = 0.20  # [s]    min cornering event length
MIN_CORNER_SAMPLES = 50  # per lap

WHEELBASE_EQ = 1.53  # [m]  equivalent wheelbase for bicycle model
STEERING_RATIO = 3.15  # [-]  mechanical column steering-wheel:road-wheel
# ratio (Parameters.m, reference only). The `Steering`
# channel is the steering-potentiometer value [rad]
# (Variables_CSV.pdf); the team's understeer/yaw
# formulas use it DIRECTLY, so do NOT divide by this.
MIN_SA_MEAN_DEG = 1.0  # [deg] slip angles below this are ignored
MAX_SA_MEAN_DEG = 15.0  # [deg]
MAX_SA_EFF = 5.0  # [m/s²/deg] sanity cap on efficiency

# ── Vehicle geometry & damper calibration ─────────────────────────────────────
# These constants drive roll/pitch from damper sensors and physical KPIs.
# DampXX in the CSV are raw potentiometer counts with different zero/scale per
# wheel — provide calibration to get real mm at the wheel.
TRACK_FRONT_M = 1.225  # [m]  front track width — Vhcl.tf from Parameters.m
TRACK_REAR_M = 1.175  # [m]  rear track width  — Vhcl.tr
WHEELBASE_M = 1.53  # [m]  axle-to-axle wheelbase — Vhcl.Wheel_Base

# ── Vehicle inertial & aero (Parameters.m) ───────────────────────────────────
MASS_KG = 288.0  # Vhcl.m — 220 kg car + 68 kg driver
IZ_KGM2 = 129.024  # Vhcl.Iz — yaw inertia
COG_Z_M = 0.278  # Vhcl.CoG_z
LF_M = 0.765  # Vhcl.lf
LR_M = 0.765  # Vhcl.lr
KROLLF_NMRAD = 36929.4  # Vhcl.Krollf
KROLLR_NMRAD = 40833.7  # Vhcl.Krollr
HRCF_M = 0.012  # Vhcl.hrcf
HRCR_M = 0.042  # Vhcl.hrcr
CL_AERO = -5.913  # Vhcl.Coef_Lift (negative = downforce)
CD_AERO = 1.803  # Vhcl.Coef_Drag
A_AERO_M2 = 1.0  # Vhcl.A
COP_X_FROM_FRONT = -0.7547  # Vhcl.CoP_x
RHO_AIR_KGM3 = 1.225  # Standard air density
MU_TIRE = 1.70  # Estimated FS slick peak friction coefficient
G_MPS2 = 9.81
G_MS2 = G_MPS2
WHEEL_RADIUS_M = 0.2032  # Vhcl.Wheel_Radius
GEAR_RATIO = 9.05  # Vhcl.i
T_MOTOR_MAX_NM = 27.5  # Max tractive torque per motor
MAX_POWER_W = 80_000  # Battery power ceiling
N_MOTORS = 4

# ── Brake system (Parameters.m) ──────────────────────────────────────────────
BRAKE_PISTONS_F = 8
BRAKE_PISTONS_R = 4
BRAKE_PISTON_DIAM_M = 0.023
BRAKE_PAD_RE_M = 0.0927
BRAKE_PAD_RI_M = 0.0608
BRAKE_PAD_MU = 0.617
BRAKE_FRONT_BALANCE = 0.67

# Counts → mm at the damper rod (per wheel; placeholder = 1.0 means raw counts).
# Set to your sensor calibration to get angles in physical degrees.
DAMPER_COUNTS_PER_MM = {"FL": 1.0, "FR": 1.0, "RL": 1.0, "RR": 1.0}
# Damper rod travel → wheel travel motion ratio (wheel_mm = damper_mm * MR).
# Typical FS pushrod: ~0.9–1.2.
DAMPER_MOTION_RATIO = {"FL": 1.0, "FR": 1.0, "RL": 1.0, "RR": 1.0}
# True when the calibration above is real. When False, dashboard adds a banner
# warning that roll/pitch values are uncalibrated and only the SHAPE is meaningful.
DAMPER_CALIBRATED = False

# Damper velocity split between low-speed and high-speed regimes [mm/s].
DAMPER_LSHS_SPLIT_MMPS = 25.0

_US_COLS_COG = ["TimeStamp", "laps", "laptime", "Steering", "Filtering_VN_ay", "Est_vxCOG"]
_US_COLS_VX = ["TimeStamp", "laps", "laptime", "Steering", "Filtering_VN_ay", "VN_vx"]
_US_LLTD_FZ_COLS = ["Est_FZFL", "Est_FZFR", "Est_FZRL", "Est_FZRR"]
_LLTD_SETUP_COLS = [
    "TimeStamp",
    "laps",
    "laptime",
    "Filtering_VN_ay",
    "VN_vx",
    "Est_FZFL",
    "Est_FZFR",
    "Est_FZRL",
    "Est_FZRR",
]


# ── Data loading ──────────────────────────────────────────────────────────────


def _load(columns: list[str]) -> dict[str, np.ndarray]:
    df = pl.read_csv(CSV_PATH, columns=columns)
    return cols_to_numpy(df, columns)


def _from_df(df: pl.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    df = ensure_complete_laps_df(df)
    cols = list(columns)
    if COMPLETE_LAPS_MARKER in df.columns and COMPLETE_LAPS_MARKER not in cols:
        cols.append(COMPLETE_LAPS_MARKER)
    return cols_to_numpy(df, cols)


def _base_validity(*arrays: np.ndarray) -> np.ndarray:
    return np.all(np.stack([np.isfinite(a) for a in arrays], axis=1), axis=1)


def _display_laps(lap_ids: np.ndarray) -> np.ndarray:
    """Return lap IDs as displayed in dashboard figures and tables."""
    return np.asarray(lap_ids, dtype=int)


def _radius_corner_mask(
    vx_mps: np.ndarray,
    ay_mps2: np.ndarray,
    dt_s: float,
    *,
    radius_threshold_m: float = 60.0,
    min_speed_mps: float = MIN_SPEED,
    min_duration_s: float = MIN_CORNER_DURATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Corner mask using the Driver/Lap Analysis curvature logic.

    The driver cornering analysis detects curves from radius
    ``R = V^2 / |ay|`` with a default 60 m threshold. This local copy avoids
    importing `src.cornering`, which already imports this module.
    """
    ay_abs = np.abs(np.asarray(ay_mps2, dtype=float))
    vx = np.asarray(vx_mps, dtype=float)
    radius_m = np.divide(
        vx**2,
        np.maximum(ay_abs, 0.05),
        out=np.full_like(vx, np.nan, dtype=float),
        where=np.isfinite(vx) & np.isfinite(ay_abs),
    )
    inv_radius = np.divide(
        1.0,
        radius_m,
        out=np.zeros_like(radius_m, dtype=float),
        where=np.isfinite(radius_m) & (radius_m > 0.0),
    )
    win = max(1, int(round(0.30 / dt_s)))
    inv_radius_sm = smooth_signal(inv_radius, win)
    radius_sm_m = np.divide(
        1.0,
        inv_radius_sm,
        out=np.full_like(inv_radius_sm, np.nan, dtype=float),
        where=np.isfinite(inv_radius_sm) & (inv_radius_sm > 0.0),
    )
    raw = (
        np.isfinite(radius_sm_m)
        & (np.abs(vx) >= min_speed_mps)
        & (radius_sm_m < radius_threshold_m)
    )
    return keep_min_duration_segments(raw, min_duration_s, dt_s), radius_sm_m


# ── 2. Understeer angle evolution ─────────────────────────────────────────────


def _compute_understeer(
    d: dict[str, np.ndarray],
    x_mode: str = "laps",
) -> tuple[go.Figure, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Core computation for understeer angle.

    Returns (fig, lap_list, und_mean, lt_val, n_samps, ok_mask).
    """
    d["time"] = d["TimeStamp"] - d["TimeStamp"][0]

    valid = _base_validity(*d.values()) & (np.abs(d["vx"]) >= 4.0)
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)

    dt = robust_dt(d["time"])
    steering = d["Steering"]
    ay_filt = d["Filtering_VN_ay"]
    vx = d["vx"]
    laps = d["laps"]
    laptime = d["laptime"]

    if "__corner_mask" in d:
        corner_mask = d["__corner_mask"].astype(bool) & (np.abs(vx) >= 4.0)
    else:
        corner_mask, _radius_m = _radius_corner_mask(
            vx,
            ay_filt,
            dt,
            radius_threshold_m=60.0,
            min_speed_mps=4.0,
            min_duration_s=MIN_CORNER_DURATION,
        )

    # Team understeer formula: |Steering| - |L·ay/vx²|. `Steering` is the
    # steering-potentiometer value [rad], used directly (no STEERING_RATIO).
    ideal_steer = WHEELBASE_EQ * ay_filt / (vx**2)
    und_rad = np.abs(steering) - np.abs(ideal_steer)
    und_deg = np.rad2deg(und_rad)

    lap_list = unique_laps(laps)
    n = len(lap_list)
    und_mean = np.full(n, np.nan)
    lt_val = np.full(n, np.nan)
    n_samps = np.zeros(n, dtype=int)

    for i, lap in enumerate(lap_list):
        lm = laps == lap
        lcm = lm & corner_mask
        n_samps[i] = lcm.sum()
        if lm.any():
            lt_val[i] = laptime[lm].max()
        if n_samps[i] >= MIN_CORNER_SAMPLES:
            und_mean[i] = np.nanmean(und_deg[lcm])

    ok = (
        np.isfinite(und_mean)
        & np.isfinite(lt_val)
        & (n_samps >= MIN_CORNER_SAMPLES)
        & (np.abs(und_mean) < 20.0)
    )

    lap_disp = _display_laps(lap_list[ok]) if ok.any() else np.array([], dtype=int)
    x_arr, order, xlabel = (
        per_lap_axis(lap_disp, lt_val[ok], x_mode)
        if ok.any()
        else (np.array([]), np.array([], dtype=int), "Lap")
    )
    fig = make_dark_figure(
        title=f"Average Understeer Angle vs {'Lap Time' if x_mode == 'laptime' else 'Lap'}",
        xlabel=xlabel,
        ylabel="Mean understeer angle [deg]",
    )
    if ok.any():
        add_lap_scatter(fig, x_arr, und_mean[ok][order], lap_disp[order])
        add_trend_line(fig, x_arr, und_mean[ok][order])
        add_zero_line(fig, x_arr)
        if x_mode == "laps":
            fig.update_xaxes(tickvals=np.sort(lap_disp.astype(int)))

    return fig, lap_list, und_mean, lt_val, n_samps, ok


def understeer_angle() -> go.Figure:
    """CLI version: loads from CSV, prints KPIs, returns figure."""
    try:
        d = _load(_US_COLS_COG)
        d["vx"] = d.pop("Est_vxCOG")
    except Exception:
        d = _load(_US_COLS_VX)
        d["vx"] = d.pop("VN_vx")

    fig, lap_list, und_mean, lt_val, n_samps, ok = _compute_understeer(d)

    print("\n─── Understeer Angle per Lap ───")
    print(f"{'Lap':>4}  {'LapTime[s]':>10}  {'Und_mean[deg]':>14}  {'Samples':>8}")
    for lap, lt, um, ns in zip(lap_list[ok], lt_val[ok], und_mean[ok], n_samps[ok]):
        print(f"{int(lap):>4}  {lt:>10.3f}  {um:>14.3f}  {ns:>8d}")

    return fig


def understeer_angle_fig(
    df: pl.DataFrame,
    corner_mask: np.ndarray | None = None,
    x_mode: str = "laps",
) -> go.Figure:
    """Dashboard version: takes a polars DataFrame, returns figure (no print)."""
    try:
        d = _from_df(df, _US_COLS_COG)
        d["vx"] = d.pop("Est_vxCOG")
    except (KeyError, Exception):
        d = _from_df(df, _US_COLS_VX)
        d["vx"] = d.pop("VN_vx")

    if corner_mask is not None:
        d["__corner_mask"] = corner_mask.astype(float)
    fig, *_ = _compute_understeer(d, x_mode=x_mode)
    return fig


def understeer_angle_kpis(
    df: pl.DataFrame,
    corner_mask: np.ndarray | None = None,
) -> dict:
    """Dashboard KPIs for understeer angle."""
    try:
        d = _from_df(df, _US_COLS_COG)
        d["vx"] = d.pop("Est_vxCOG")
    except (KeyError, Exception):
        d = _from_df(df, _US_COLS_VX)
        d["vx"] = d.pop("VN_vx")

    if corner_mask is not None:
        d["__corner_mask"] = corner_mask.astype(float)
    _fig, lap_list, und_mean, lt_val, n_samps, ok = _compute_understeer(d)
    if not ok.any():
        return {"warnings": ["No valid laps for understeer KPIs."]}

    valid_laps = lap_list[ok]
    valid_laps_disp = _display_laps(valid_laps)
    valid_und = und_mean[ok]
    valid_lt = lt_val[ok]
    valid_samples = n_samps[ok]

    table = pl.DataFrame(
        {
            "Lap": valid_laps_disp.astype(int),
            "LapTime [s]": np.round(valid_lt, 3),
            "Mean understeer [deg]": np.round(valid_und, 3),
            "Corner samples": valid_samples.astype(int),
        }
    )

    return {
        "valid_laps": int(ok.sum()),
        "mean_understeer": float(np.nanmean(valid_und)),
        "min_understeer": float(np.nanmin(valid_und)),
        "max_understeer": float(np.nanmax(valid_und)),
        "fastest_lap": int(valid_laps_disp[int(np.nanargmin(valid_lt))]),
        "fastest_lt": float(np.nanmin(valid_lt)),
        "mean_corner_samples": float(np.nanmean(valid_samples)),
        "table": table,
        "warnings": [],
    }


def _lltd_mid_corner_table_for_run(
    df: pl.DataFrame, run_name: str
) -> tuple[pl.DataFrame, list[str]]:
    """Per-lap front LLTD using only mid-corner samples."""
    missing = [c for c in _LLTD_SETUP_COLS if c not in df.columns]
    if missing:
        return pl.DataFrame(), [f"Missing LLTD setup columns: {missing}"]
    try:
        arr = _from_df(df, _LLTD_SETUP_COLS)
    except Exception as exc:
        return pl.DataFrame(), [str(exc)]

    valid = _base_validity(*(arr[c] for c in _LLTD_SETUP_COLS)) & (np.abs(arr["VN_vx"]) >= 4.0)
    arr = {k: v[valid] for k, v in arr.items()}
    try:
        arr = exclude_lap0_and_last_lap(arr)
    except ValueError as exc:
        return pl.DataFrame(), [str(exc)]
    if arr["TimeStamp"].size == 0:
        return pl.DataFrame(), ["No valid samples for LLTD setup metric."]

    time_s = arr["TimeStamp"] - arr["TimeStamp"][0]
    dt = robust_dt(time_s)
    corner_mask, _radius_m = _radius_corner_mask(
        arr["VN_vx"],
        arr["Filtering_VN_ay"],
        dt,
        radius_threshold_m=60.0,
        min_speed_mps=4.0,
        min_duration_s=MIN_CORNER_DURATION,
    )

    dfz_front_n = arr["Est_FZFR"] - arr["Est_FZFL"]
    dfz_rear_n = arr["Est_FZRR"] - arr["Est_FZRL"]
    denom_n = np.abs(dfz_front_n) + np.abs(dfz_rear_n)
    lltd_front_pct = 100.0 * np.divide(
        np.abs(dfz_front_n),
        denom_n,
        out=np.full_like(denom_n, np.nan, dtype=float),
        where=np.isfinite(denom_n) & (denom_n > 1.0),
    )
    ay_abs = np.abs(arr["Filtering_VN_ay"])

    rows: list[dict[str, object]] = []
    for lap in unique_laps(arr["laps"]):
        lap_mask = arr["laps"] == lap
        lap_corner = lap_mask & corner_mask & np.isfinite(lltd_front_pct) & np.isfinite(ay_abs)
        if int(lap_corner.sum()) < MIN_CORNER_SAMPLES:
            continue
        ay_corner = ay_abs[lap_corner]
        ay_threshold = max(AY_THRESHOLD, float(np.nanpercentile(ay_corner, 60.0)))
        mid_corner = lap_corner & (ay_abs >= ay_threshold)
        if int(mid_corner.sum()) < max(20, MIN_CORNER_SAMPLES // 5):
            mid_corner = lap_corner
        lltd_vals = lltd_front_pct[mid_corner]
        if not np.any(np.isfinite(lltd_vals)):
            continue
        rows.append(
            {
                "Run": run_name,
                "Lap": int(_display_laps(np.array([lap]))[0]),
                "LapTime [s]": float(np.nanmax(arr["laptime"][lap_mask]))
                if lap_mask.any()
                else np.nan,
                "LLTD mid-corner avg [%]": float(np.nanmean(lltd_vals)),
                "LLTD mid-corner median [%]": float(np.nanmedian(lltd_vals)),
                "LLTD mid-corner span [pp]": float(np.nanmax(lltd_vals) - np.nanmin(lltd_vals)),
                "Mid-corner samples": int(mid_corner.sum()),
                "Corner samples": int(lap_corner.sum()),
            }
        )

    if not rows:
        return pl.DataFrame(), ["No valid mid-corner samples for LLTD setup metric."]
    return pl.DataFrame(rows), []


def lltd_mid_corner_per_lap_fig(
    dfs: dict[str, pl.DataFrame],
    x_mode: str = "laps",
) -> tuple[go.Figure, dict]:
    """Setup metric: average front LLTD in mid-corner samples, per lap."""
    fig = make_dark_figure(
        "LLTD Mid-Corner Avg",
        "Lap",
        "Front LLTD mid-corner avg [%]",
    )
    tables: list[pl.DataFrame] = []
    warnings: list[str] = []
    runs: dict[str, dict[str, float | int]] = {}

    for idx, (run_name, df) in enumerate(dfs.items()):
        table, run_warnings = _lltd_mid_corner_table_for_run(df, run_name)
        warnings.extend(f"{run_name}: {w}" for w in run_warnings)
        if table.is_empty():
            continue
        tables.append(table)
        laps = table["Lap"].to_numpy()
        laptimes = table["LapTime [s]"].to_numpy()
        y = table["LLTD mid-corner avg [%]"].to_numpy()
        spans = table["LLTD mid-corner span [pp]"].to_numpy()
        samples = table["Mid-corner samples"].to_numpy()
        x, order, xlabel = per_lap_axis(laps, laptimes, x_mode)
        color = driver_color(run_name)
        customdata = np.column_stack([laps[order], laptimes[order], spans[order], samples[order]])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y[order],
                mode="lines+markers",
                name=run_name,
                line=dict(color=color, width=2.2),
                marker=dict(
                    color=color,
                    size=9,
                    line=dict(color="#EBEBEB", width=0.6),
                ),
                customdata=customdata,
                hovertemplate=(
                    "Lap=%{customdata[0]:.0f}<br>"
                    "LapTime=%{customdata[1]:.2f} s<br>"
                    "LLTD avg=%{y:.3f}%<br>"
                    "Lap LLTD span=%{customdata[2]:.4f} pp<br>"
                    "samples=%{customdata[3]:.0f}<extra></extra>"
                ),
            )
        )

        # Short rolling reference line, matching the setup-change reading in the slide.
        if len(y) >= 3:
            y_ordered = y[order]
            roll = np.array(
                [
                    np.nanmedian(y_ordered[max(0, i - 1) : min(len(y_ordered), i + 2)])
                    for i in range(len(y_ordered))
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=roll,
                    mode="lines",
                    name=f"{run_name} trend",
                    legendgroup=run_name,
                    showlegend=False,
                    line=dict(color=color, width=1.7, dash="dash"),
                    hoverinfo="skip",
                )
            )

        runs[run_name] = {
            "laps": int(len(table)),
            "lltd_mean_pct": float(np.nanmean(y)),
            "lltd_min_pct": float(np.nanmin(y)),
            "lltd_max_pct": float(np.nanmax(y)),
            "lltd_span_pct_points": float(np.nanmax(y) - np.nanmin(y)),
            "samples_mean": float(np.nanmean(samples)),
        }
        fig.update_xaxes(title_text=xlabel)

    kroll_split_pct = KROLLF_NMRAD / (KROLLF_NMRAD + KROLLR_NMRAD) * 100.0
    fig.add_hline(
        y=kroll_split_pct,
        line=dict(color="#73D973", width=1.5, dash="dot"),
        annotation_text=f"Kroll split {kroll_split_pct:.1f}%",
        annotation_position="top right",
    )
    fig.update_layout(
        height=520,
        margin=dict(l=70, r=35, t=55, b=65),
        hovermode="closest",
        legend=dict(
            bgcolor="rgba(20,20,23,0.85)",
            bordercolor="rgba(128,128,128,0.3)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
    )
    if tables:
        table_all = pl.concat(tables, how="vertical")
        y_all = table_all["LLTD mid-corner avg [%]"].to_numpy()
        y_pad = max(0.35, (float(np.nanmax(y_all)) - float(np.nanmin(y_all))) * 0.5)
        fig.update_yaxes(range=[float(np.nanmin(y_all)) - y_pad, float(np.nanmax(y_all)) + y_pad])
    else:
        table_all = pl.DataFrame()

    return fig, {
        "runs": runs,
        "table": table_all,
        "kroll_split_pct": float(kroll_split_pct),
        "warnings": warnings,
    }


# ── 3. Per-lap colour map ────────────────────────────────────────────────────
# Sole survivor of the old interactive pilot / GG view (replaced by
# Dynamics › Overview, see gripfactor.gg_scatter_fig). Still used by cornering.py.

_PURPLE_FASTEST = "rgb(170, 60, 230)"  # fastest-lap highlight for build_color_map


def build_color_map(
    entries: list[tuple[str, int, float]],
) -> dict[tuple[str, int], str]:
    """Map (run_name, lap_id) → color. Purple = fastest, RdYlGn gradient for rest."""
    import plotly.colors as pc

    if not entries:
        return {}
    ordered = sorted(entries, key=lambda e: e[2])
    n = len(ordered)
    colors: dict[tuple[str, int], str] = {(ordered[0][0], ordered[0][1]): _PURPLE_FASTEST}
    if n == 1:
        return colors
    positions = [1.0 - (i / (n - 1)) for i in range(n)]
    scale = pc.sample_colorscale("RdYlGn", positions)
    for i in range(1, n):
        colors[(ordered[i][0], ordered[i][1])] = scale[i]
    return colors


# ── 4. Damper velocity histograms (LSB/HSB/LSR/HSR) ──────────────────────────

_DAMPER_COLS = ["TimeStamp", "laps", "DampFL", "DampFR", "DampRL", "DampRR"]

# Quadrant colours: bump = warm, rebound = cool; high-speed = darker
_DAMP_QUAD_COLORS = {
    "HSR": "#1F77B4",  # high-speed rebound  (very negative)
    "LSR": "#7CB6E0",  # low-speed rebound
    "LSB": "#F2A65A",  # low-speed bump
    "HSB": "#D94F4F",  # high-speed bump     (very positive)
}
_DAMPER_PHASE_COLS = ["Filtering_VN_ax", "Filtering_VN_ay", "VN_vx", "Brake", "Throttle"]


def _damper_mm(counts: np.ndarray, wheel: str) -> np.ndarray:
    """Convert raw damper counts to mm at the damper rod (uses calibration)."""
    cpm = DAMPER_COUNTS_PER_MM.get(wheel, 1.0)
    return counts / cpm if cpm not in (0.0, None) else counts.astype(float)


def _wheel_mm_from_damper_mm(damp_mm: np.ndarray, wheel: str) -> np.ndarray:
    """Damper rod travel [mm] → wheel travel [mm] using motion ratio."""
    mr = DAMPER_MOTION_RATIO.get(wheel, 1.0)
    return damp_mm * mr


def _setup_phase_mask(arr: dict[str, np.ndarray], phase: str, dt: float) -> np.ndarray:
    n = len(next(iter(arr.values()))) if arr else 0
    if phase == "all":
        return np.ones(n, dtype=bool)
    ax = arr["Filtering_VN_ax"]
    ay = arr["Filtering_VN_ay"]
    vx = arr["VN_vx"]
    brake = arr["Brake"]
    throttle = arr["Throttle"]
    brake_m = (ax < -1.0) & (brake > 5.0)
    accel_m = (ax > 1.0) & (throttle > 5.0)
    corner_m, _radius_m = _radius_corner_mask(vx, ay, dt, radius_threshold_m=60.0)
    if phase == "brake":
        return brake_m
    if phase == "accel":
        return accel_m
    if phase == "corner":
        return corner_m
    if phase == "straight":
        return ~(brake_m | accel_m | corner_m)
    raise ValueError(f"Unsupported damper phase: {phase}")


def _t1_calibration_ok(df: pl.DataFrame) -> bool:
    if "Pot_Calibration_Status" not in df.columns:
        return False
    try:
        return str(df["Pot_Calibration_Status"][0]) in {"validated", "partial"}
    except Exception:
        return False


def damper_histogram_figs(
    df: pl.DataFrame,
    phase: Literal["all", "brake", "corner", "accel", "straight"] = "all",
) -> tuple[list[go.Figure], dict]:
    """Damper velocity histograms per wheel split into LSB/HSB/LSR/HSR.

    Returns a single 2×2 figure (FL / FR / RL / RR) plus a KPIs dict with the
    fraction of time in each quadrant per wheel and balance metrics.
    """
    df = ensure_complete_laps_df(df)
    required = list(_DAMPER_COLS)
    if phase != "all":
        required += [c for c in _DAMPER_PHASE_COLS if c not in required]
    t1_ok = _t1_calibration_ok(df) and all(
        f"Pot_Speed_{w}" in df.columns for w in ("FL", "FR", "RL", "RR")
    )
    if t1_ok:
        required += [f"Pot_Speed_{w}" for w in ("FL", "FR", "RL", "RR")]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing damper columns: {missing}")

    arr = cols_to_numpy(df, required)
    time_s = arr["TimeStamp"] - arr["TimeStamp"][0]
    dt = robust_dt(time_s)
    sample_mask = _setup_phase_mask(arr, phase, dt)
    if not sample_mask.any():
        raise ValueError(f"No samples for damper phase `{phase}`.")

    split = float(DAMPER_LSHS_SPLIT_MMPS)
    quad_share: dict[str, dict[str, float]] = {}
    quad_p95: dict[str, dict[str, float]] = {}
    calibrated = bool(DAMPER_CALIBRATED or t1_ok)

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=False,
        shared_yaxes=False,
        subplot_titles=("FL", "FR", "RL", "RR"),
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )
    grid_pos = {"FL": (1, 1), "FR": (1, 2), "RL": (2, 1), "RR": (2, 2)}
    show_legend_once = {"HSR", "LSR", "LSB", "HSB"}

    # Damper signals can be integer-quantised → derivative gets stuck at 0.
    # Smooth over ~50 ms before differentiating to recover physical velocity.
    from utils import smooth_signal

    smooth_n = max(1, int(round(0.05 / dt)))

    for w in ("FL", "FR", "RL", "RR"):
        if t1_ok:
            vel = np.asarray(arr[f"Pot_Speed_{w}"], dtype=float)
        else:
            damp_mm = _damper_mm(arr[f"Damp{w}"], w)
            wheel_mm = _wheel_mm_from_damper_mm(damp_mm, w)
            wheel_mm = smooth_signal(wheel_mm, smooth_n)
            vel = np.gradient(wheel_mm, dt)  # [mm/s] when calibrated, counts/s otherwise
        vel = vel[sample_mask]
        finite = np.isfinite(vel)
        vel = vel[finite]
        if vel.size == 0:
            continue

        lo = float(np.nanpercentile(vel, 1))
        hi = float(np.nanpercentile(vel, 99))
        bound = max(abs(lo), abs(hi), split * 2.0)
        bins = np.linspace(-bound, bound, 81)

        masks = {
            "HSR": vel <= -split,
            "LSR": (vel > -split) & (vel < 0.0),
            "LSB": (vel >= 0.0) & (vel < split),
            "HSB": vel >= split,
        }
        total = float(vel.size)
        quad_share[w] = {q: float(m.sum()) / total for q, m in masks.items()}
        quad_p95[w] = {
            "bump_p95": float(np.nanpercentile(vel[vel > 0], 95)) if (vel > 0).any() else np.nan,
            "rebound_p95": float(np.nanpercentile(vel[vel < 0], 5)) if (vel < 0).any() else np.nan,
        }

        r, c = grid_pos[w]
        for q in ("HSR", "LSR", "LSB", "HSB"):
            sub = vel[masks[q]]
            if sub.size == 0:
                continue
            fig.add_trace(
                go.Histogram(
                    x=sub,
                    xbins=dict(start=bins[0], end=bins[-1], size=bins[1] - bins[0]),
                    marker_color=_DAMP_QUAD_COLORS[q],
                    opacity=0.85,
                    name=q,
                    legendgroup=q,
                    showlegend=q in show_legend_once,
                ),
                row=r,
                col=c,
            )
            show_legend_once.discard(q)

        fig.add_vline(
            x=-split, line=dict(color="rgba(255,255,255,0.35)", dash="dot", width=1), row=r, col=c
        )
        fig.add_vline(
            x=+split, line=dict(color="rgba(255,255,255,0.35)", dash="dot", width=1), row=r, col=c
        )
        fig.add_vline(
            x=0.0, line=dict(color="rgba(255,255,255,0.55)", dash="dash", width=1), row=r, col=c
        )

        share = quad_share[w]
        ann = (
            f"HSR {share['HSR'] * 100:4.1f}%   LSR {share['LSR'] * 100:4.1f}%<br>"
            f"LSB {share['LSB'] * 100:4.1f}%   HSB {share['HSB'] * 100:4.1f}%"
        )
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            text=ann,
            showarrow=False,
            align="left",
            font=dict(size=10, color="#EBEBEB"),
            bgcolor="rgba(20,20,23,0.78)",
            bordercolor="rgba(128,128,128,0.35)",
            borderwidth=1,
            borderpad=3,
            row=r,
            col=c,
        )

    fig.update_layout(
        title=dict(
            text=(
                f"Damper velocity histograms · {phase.upper()} — LSB/HSB/LSR/HSR (split at "
                f"±{split:.0f} mm/s)" + ("" if calibrated else "  ·  uncalibrated counts/s")
            ),
            font=dict(size=14, color="#EBEBEB"),
        ),
        paper_bgcolor="#141417",
        plot_bgcolor="#141417",
        font=dict(color="#EBEBEB", size=11),
        barmode="overlay",
        bargap=0.02,
        height=620,
        legend=dict(
            bgcolor="rgba(20,20,23,0.85)",
            bordercolor="rgba(128,128,128,0.3)",
            font=dict(color="#EBEBEB"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
        margin=dict(l=60, r=20, t=70, b=50),
    )
    unit = "mm/s" if calibrated else "counts/s"
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(
                title_text=f"Damper velocity [{unit}]",
                gridcolor="rgba(128,128,128,0.2)",
                row=r,
                col=c,
            )
            fig.update_yaxes(title_text="Samples", gridcolor="rgba(128,128,128,0.2)", row=r, col=c)

    # KPIs: per-wheel quadrant share + bump/rebound balance
    kpis = {
        "split_mmps": split,
        "phase": phase,
        "calibrated": calibrated,
        "quad_share": quad_share,
        "quad_p95": quad_p95,
        "bump_share_by_axle": {
            "front": float(
                np.mean(
                    [
                        quad_share[w]["LSB"] + quad_share[w]["HSB"]
                        for w in ("FL", "FR")
                        if w in quad_share
                    ]
                )
            )
            if quad_share
            else np.nan,
            "rear": float(
                np.mean(
                    [
                        quad_share[w]["LSB"] + quad_share[w]["HSB"]
                        for w in ("RL", "RR")
                        if w in quad_share
                    ]
                )
            )
            if quad_share
            else np.nan,
        },
        "warnings": []
        if calibrated
        else [
            "Damper signals are uncalibrated raw counts — set DAMPER_COUNTS_PER_MM and "
            "DAMPER_MOTION_RATIO in src/dynamics.py for absolute mm/s values."
        ],
    }
    return [fig], kpis


# ── 5. Roll gradient (deg/g) ─────────────────────────────────────────────────

_ROLL_COLS = [
    "TimeStamp",
    "laps",
    "laptime",
    "Filtering_VN_ay",
    "VN_vx",
    "DampFL",
    "DampFR",
    "DampRL",
    "DampRR",
]


def _roll_angle_deg(
    damp_left_mm: np.ndarray,
    damp_right_mm: np.ndarray,
    wheel_left: str,
    wheel_right: str,
    track_m: float,
) -> np.ndarray:
    """Estimate axle roll angle [deg] from damper positions.

    Positive roll = body rolls towards the right (left wheel compresses,
    right wheel extends) → matches positive ay in left-handed corner exits.
    """
    wheel_left_mm = _wheel_mm_from_damper_mm(damp_left_mm, wheel_left)
    wheel_right_mm = _wheel_mm_from_damper_mm(damp_right_mm, wheel_right)
    delta_mm = wheel_left_mm - wheel_right_mm
    return np.rad2deg(np.arctan2(delta_mm / 1000.0, track_m))


def roll_gradient_fig(df: pl.DataFrame) -> tuple[go.Figure, dict]:
    """Body roll angle vs lateral acceleration, regressed per axle."""
    df = ensure_complete_laps_df(df)
    t1_ok = _t1_calibration_ok(df) and all(
        c in df.columns for c in ("Roll_Front", "Roll_Rear", "Roll")
    )
    cols = list(_ROLL_COLS)
    if t1_ok:
        cols += ["Roll_Front", "Roll_Rear", "Roll"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing roll columns: {missing}")

    arr = cols_to_numpy(df, cols)
    valid = _base_validity(*arr.values())
    arr = {k: v[valid] for k, v in arr.items()}

    time_s = arr["TimeStamp"] - arr["TimeStamp"][0]
    dt = robust_dt(time_s)
    ay = arr["Filtering_VN_ay"]
    vx = arr["VN_vx"]
    in_corner, _radius_m = _radius_corner_mask(vx, ay, dt, radius_threshold_m=60.0)

    if t1_ok:
        roll_front = arr["Roll_Front"]
        roll_rear = arr["Roll_Rear"]
        y_suffix = ""
    else:
        # Reference each damper to its own median (≈ static ride height) so
        # the roll angle is a delta relative to straight running.
        damp = {}
        for w in ("FL", "FR", "RL", "RR"):
            mm = _damper_mm(arr[f"Damp{w}"], w)
            damp[w] = mm - float(np.nanmedian(mm))
        roll_front = _roll_angle_deg(damp["FL"], damp["FR"], "FL", "FR", TRACK_FRONT_M)
        roll_rear = _roll_angle_deg(damp["RL"], damp["RR"], "RL", "RR", TRACK_REAR_M)
        y_suffix = "  (uncalibrated)"
    ay_g = ay / 9.81
    h_roll_m = COG_Z_M - 0.5 * (HRCF_M + HRCR_M)
    theory_deg_per_g = MASS_KG * h_roll_m / (KROLLF_NMRAD + KROLLR_NMRAD) * (180.0 / np.pi) * G_MPS2

    fig = make_dark_figure(
        title="Roll Gradient  ·  Measured vs Theoretical",
        xlabel="Lateral acceleration ay [g]",
        ylabel="Roll angle [deg]" + y_suffix,
    )

    kpis: dict = {
        "calibrated": bool(DAMPER_CALIBRATED or t1_ok),
        "theoretical_deg_per_g": float(theory_deg_per_g),
    }
    x_min = np.nan
    x_max = np.nan
    for axle, (label, color, roll) in {
        "front": ("Front", "#4DB3F2", roll_front),
        "rear": ("Rear", "#F28C40", roll_rear),
    }.items():
        m = in_corner & np.isfinite(roll) & np.isfinite(ay_g)
        if not m.any():
            kpis[f"{axle}_gradient_deg_per_g"] = np.nan
            kpis[f"{axle}_r2"] = np.nan
            continue
        x = ay_g[m]
        y = roll[m]
        x_min = np.nanmin([x_min, np.nanmin(x)]) if np.isfinite(x_min) else float(np.nanmin(x))
        x_max = np.nanmax([x_max, np.nanmax(x)]) if np.isfinite(x_max) else float(np.nanmax(x))
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker=dict(color=color, size=3, opacity=0.4),
                name=f"{label} samples",
            )
        )
        if x.size >= 10:
            slope, intercept = np.polyfit(x, y, 1)
            xfit = np.linspace(np.nanmin(x), np.nanmax(x), 50)
            fig.add_trace(
                go.Scatter(
                    x=xfit,
                    y=slope * xfit + intercept,
                    mode="lines",
                    line=dict(color=color, width=2.4),
                    name=f"{label} fit · {slope:+.2f} deg/g",
                )
            )
            ss_res = float(np.sum((y - (slope * x + intercept)) ** 2))
            ss_tot = float(np.sum((y - np.nanmean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            kpis[f"{axle}_gradient_deg_per_g"] = float(slope)
            kpis[f"{axle}_r2"] = float(r2)
            kpis[f"{axle}_deviation_pct"] = float(
                (abs(slope) - theory_deg_per_g) / theory_deg_per_g * 100.0
            )
        else:
            kpis[f"{axle}_gradient_deg_per_g"] = np.nan
            kpis[f"{axle}_r2"] = np.nan
            kpis[f"{axle}_deviation_pct"] = np.nan

    if np.isfinite(x_min) and np.isfinite(x_max):
        xfit = np.linspace(x_min, x_max, 50)
        fig.add_trace(
            go.Scatter(
                x=xfit,
                y=theory_deg_per_g * xfit,
                mode="lines",
                line=dict(color="#73D973", width=2.2, dash="dash"),
                name=f"Theory · {theory_deg_per_g:.2f} deg/g",
            )
        )
    fig.add_hline(y=0.0, line=dict(color="rgba(200,200,200,0.4)", dash="dot", width=1))
    fig.add_vline(x=0.0, line=dict(color="rgba(200,200,200,0.4)", dash="dot", width=1))
    kpis["warnings"] = (
        []
        if (DAMPER_CALIBRATED or t1_ok)
        else [
            "Roll values are uncalibrated — set DAMPER_COUNTS_PER_MM/MOTION_RATIO in src/dynamics.py."
        ]
    )
    return fig, kpis


# ── 6. Brake distribution: ideal model vs measured regen ─────────────────────

_BRAKE_DIST_COLS = [
    "TimeStamp",
    "laps",
    "laptime",
    "Filtering_VN_ax",
    "VN_vx",
    "FL_actualTorque",
    "FR_actualTorque",
    "RL_actualTorque",
    "RR_actualTorque",
]
_BRAKE_DIST_OPTIONAL_COLS = ["BSEFront", "BSERear"]


def _ideal_brake_forces(
    ax_abs_mps2: np.ndarray,
    vx_mps: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ideal front/rear brake force split from vertical load distribution."""
    ax_abs = np.asarray(ax_abs_mps2, dtype=float)
    vx = np.asarray(vx_mps, dtype=float)
    aero_n = 0.5 * RHO_AIR_KGM3 * (vx**2) * abs(CL_AERO) * A_AERO_M2
    front_aero_frac = _aero_front_fraction()
    d_fz_n = MASS_KG * ax_abs * COG_Z_M / WHEELBASE_M
    fz_front_n = MASS_KG * G_MPS2 * LR_M / WHEELBASE_M + d_fz_n + aero_n * front_aero_frac
    fz_rear_n = MASS_KG * G_MPS2 * LF_M / WHEELBASE_M - d_fz_n + aero_n * (1.0 - front_aero_frac)
    fz_total_n = fz_front_n + fz_rear_n
    front_frac = np.divide(
        fz_front_n,
        fz_total_n,
        out=np.full_like(fz_front_n, np.nan, dtype=float),
        where=np.isfinite(fz_total_n) & (fz_total_n > 0.0),
    )
    total_fx_n = MASS_KG * ax_abs
    return total_fx_n * front_frac, total_fx_n * (1.0 - front_frac), front_frac


def _regen_force_from_motor_torque(torque_nm: np.ndarray) -> np.ndarray:
    """Positive braking force [N] from negative motor torque [N·m]."""
    regen_torque_nm = -np.minimum(0.0, torque_nm) * GEAR_RATIO
    return regen_torque_nm / WHEEL_RADIUS_M


def _brake_pressure_demand_force(pressure_bar: np.ndarray, piston_count: int) -> np.ndarray:
    """Hydraulic brake demand force [N] from pressure [bar], if sensors exist."""
    piston_area_m2 = np.pi * (BRAKE_PISTON_DIAM_M * 0.5) ** 2
    pad_radius_m = 0.5 * (BRAKE_PAD_RE_M + BRAKE_PAD_RI_M)
    pressure_pa = np.maximum(0.0, pressure_bar) * 1e5
    return (
        pressure_pa * piston_count * piston_area_m2 * BRAKE_PAD_MU * pad_radius_m / WHEEL_RADIUS_M
    )


def _rms_distance_to_ideal_curve(
    ff_n: np.ndarray,
    fr_n: np.ndarray,
    vx_mps: np.ndarray,
    ax_grid_mps2: np.ndarray,
) -> np.ndarray:
    """Per-sample Euclidean distance [N] to the ideal curve at sample speed."""
    out = np.full_like(ff_n, np.nan, dtype=float)
    chunk_size = 4000
    for start in range(0, len(ff_n), chunk_size):
        end = min(start + chunk_size, len(ff_n))
        for i in range(start, end):
            f_curve, r_curve, _front_frac = _ideal_brake_forces(ax_grid_mps2, vx_mps[i])
            dist = np.hypot(f_curve - ff_n[i], r_curve - fr_n[i])
            out[i] = float(np.nanmin(dist)) if np.any(np.isfinite(dist)) else np.nan
    return out


def _ideal_rear_at_front_force(
    ff_n: np.ndarray,
    vx_mps: np.ndarray,
    ax_grid_mps2: np.ndarray,
) -> np.ndarray:
    """Ideal rear force [N] interpolated at measured front force and speed."""
    out = np.full_like(ff_n, np.nan, dtype=float)
    for i, (front_force_n, vx) in enumerate(zip(ff_n, vx_mps)):
        f_curve, r_curve, _front_frac = _ideal_brake_forces(ax_grid_mps2, vx)
        order = np.argsort(f_curve)
        out[i] = np.interp(front_force_n, f_curve[order], r_curve[order])
    return out


def _empty_brake_distribution_fig(message: str) -> go.Figure:
    fig = make_dark_figure(
        title="Ideal braking curve vs measured regen",
        xlabel="Front braking force [kN]",
        ylabel="Rear braking force [kN]",
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        text=message,
        font=dict(color="#EBEBEB", size=12),
    )
    return fig


def ideal_braking_curve_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """Compare ideal front/rear braking distribution with measured regen force.

    Returns a force-plane figure plus KPIs per run.
    """
    if not dfs:
        return _empty_brake_distribution_fig("No runs selected"), {
            "runs": {},
            "warnings": ["No runs selected."],
        }

    ax_grid = np.linspace(0.0, MU_TIRE * G_MPS2, 200)
    run_payloads: list[dict[str, object]] = []
    kpi_runs: dict[str, dict[str, float]] = {}
    warnings: list[str] = []
    color_max = 1.0
    required = list(_BRAKE_DIST_COLS)
    ideal_curve_limits: list[tuple[float, float]] = []

    for run_name, df_in in dfs.items():
        df = ensure_complete_laps_df(df_in)
        missing = [c for c in required if c not in df.columns]
        if missing:
            warnings.append(f"{run_name}: missing brake distribution columns: {missing}")
            continue

        cols = required + [c for c in _BRAKE_DIST_OPTIONAL_COLS if c in df.columns]
        arr = cols_to_numpy(df, cols)
        dist_m = lap_dist_from_gps(df)
        valid = _base_validity(*(arr[c] for c in required)) & np.isfinite(dist_m)
        if not valid.any():
            warnings.append(f"{run_name}: no finite brake distribution samples.")
            continue

        arr = {k: v[valid] for k, v in arr.items()}
        dist_m = dist_m[valid]
        vx_mps = np.abs(arr["VN_vx"])
        ax_abs_mps2 = np.abs(arr["Filtering_VN_ax"])
        f_reg_fl = _regen_force_from_motor_torque(arr["FL_actualTorque"])
        f_reg_fr = _regen_force_from_motor_torque(arr["FR_actualTorque"])
        f_reg_rl = _regen_force_from_motor_torque(arr["RL_actualTorque"])
        f_reg_rr = _regen_force_from_motor_torque(arr["RR_actualTorque"])
        f_reg_front_n = f_reg_fl + f_reg_fr
        f_reg_rear_n = f_reg_rl + f_reg_rr
        total_regen_n = f_reg_front_n + f_reg_rear_n

        brake_mask = (
            (arr["Filtering_VN_ax"] < -1.0)
            & (total_regen_n > 50.0)
            & np.isfinite(vx_mps)
            & np.isfinite(dist_m)
        )
        if not brake_mask.any():
            warnings.append(f"{run_name}: no regen braking samples meet the filter.")
            continue

        ff_n = f_reg_front_n[brake_mask]
        fr_n = f_reg_rear_n[brake_mask]
        vx_b = vx_mps[brake_mask]
        ax_b = ax_abs_mps2[brake_mask]
        dist_b = dist_m[brake_mask]
        total_b = ff_n + fr_n
        front_bias = np.divide(
            ff_n,
            total_b,
            out=np.full_like(ff_n, np.nan, dtype=float),
            where=total_b > 0.0,
        )

        equivalent_ax = np.clip(total_b / MASS_KG, 0.0, MU_TIRE * G_MPS2)
        _ideal_ff, _ideal_fr, ideal_front_bias = _ideal_brake_forces(equivalent_ax, vx_b)
        bias_error = front_bias - ideal_front_bias
        dist_to_ideal_n = _rms_distance_to_ideal_curve(ff_n, fr_n, vx_b, ax_grid)
        ideal_rear_n = _ideal_rear_at_front_force(ff_n, vx_b, ax_grid)
        rear_overbiased = fr_n > ideal_rear_n
        peak_combined_brake_g = np.nanmax(total_b) / (MASS_KG * G_MPS2)

        hyd_payload = None
        if "BSEFront" in arr and "BSERear" in arr:
            bse_f = arr["BSEFront"][brake_mask]
            bse_r = arr["BSERear"][brake_mask]
            bse_active = np.nanstd(bse_f) > 0.05 or np.nanstd(bse_r) > 0.05
            if bse_active:
                hyd_payload = {
                    "front_kN": _brake_pressure_demand_force(bse_f, BRAKE_PISTONS_F) / 1000.0,
                    "rear_kN": _brake_pressure_demand_force(bse_r, BRAKE_PISTONS_R) / 1000.0,
                }

        color_max = max(color_max, float(np.nanpercentile(ax_b, 95)) if ax_b.size else 1.0)
        run_payloads.append(
            {
                "run_name": run_name,
                "front_kN": ff_n / 1000.0,
                "rear_kN": fr_n / 1000.0,
                "ax_abs": ax_b,
                "hyd": hyd_payload,
            }
        )
        kpi_runs[run_name] = {
            "front_bias_mean": float(np.nanmean(front_bias)),
            "front_bias_std": float(np.nanstd(front_bias)),
            "rms_dist_to_ideal_N": float(np.sqrt(np.nanmean(dist_to_ideal_n**2))),
            "pct_time_rear_overbiased": float(np.nanmean(rear_overbiased) * 100.0),
            "peak_combined_brake_g": float(peak_combined_brake_g),
            "samples": int(brake_mask.sum()),
        }

    if not run_payloads:
        return _empty_brake_distribution_fig("No valid regen braking samples"), {
            "runs": kpi_runs,
            "warnings": warnings or ["No valid regen braking samples."],
        }

    fig = make_dark_figure(
        xlabel="Front braking force [kN]",
        ylabel="Rear braking force [kN]",
    )

    curve_colors = {0.0: "#3155D4", 15.0: "#32A6F0", 25.0: "#54F0FF"}
    for speed_mps, color in curve_colors.items():
        f_curve, r_curve, _front_frac = _ideal_brake_forces(ax_grid, speed_mps)
        ideal_curve_limits.append(
            (float(np.nanmax(f_curve / 1000.0)), float(np.nanmax(r_curve / 1000.0)))
        )
        fig.add_trace(
            go.Scatter(
                x=f_curve / 1000.0,
                y=r_curve / 1000.0,
                mode="lines",
                line=dict(color=color, width=2.4),
                name=f"Ideal {speed_mps:.0f} m/s",
                hovertemplate="Ff=%{x:.2f} kN<br>Fr=%{y:.2f} kN<extra></extra>",
            )
        )

    force_limit_kN = MU_TIRE * MASS_KG * G_MPS2 / 1000.0
    fig.add_trace(
        go.Scatter(
            x=[0.0, force_limit_kN],
            y=[force_limit_kN, 0.0],
            mode="lines",
            line=dict(color="rgba(235,235,235,0.55)", width=1.5, dash="dot"),
            name=f"μ envelope v=0 ({MU_TIRE:.2f})",
            hoverinfo="skip",
        )
    )
    balance_slope = (1.0 - BRAKE_FRONT_BALANCE) / BRAKE_FRONT_BALANCE
    fig.add_trace(
        go.Scatter(
            x=[0.0, force_limit_kN],
            y=[0.0, balance_slope * force_limit_kN],
            mode="lines",
            line=dict(color="#F2D44D", width=1.8, dash="dash"),
            name=f"LLC balance {BRAKE_FRONT_BALANCE * 100:.0f}/{(1.0 - BRAKE_FRONT_BALANCE) * 100:.0f}",
            hoverinfo="skip",
        )
    )

    for payload in run_payloads:
        run_name = str(payload["run_name"])
        fig.add_trace(
            go.Scattergl(
                x=payload["front_kN"],
                y=payload["rear_kN"],
                mode="markers",
                marker=dict(
                    color=driver_color(run_name),
                    size=4,
                    opacity=0.45,
                ),
                name=f"{run_name} regen",
                customdata=payload["ax_abs"],
                hovertemplate=(
                    f"{run_name}<br>Ff=%{{x:.2f}} kN<br>Fr=%{{y:.2f}} kN"
                    "<br>|ax|=%{customdata:.2f} m/s²<extra></extra>"
                ),
            )
        )
        hyd = payload.get("hyd")
        if isinstance(hyd, dict):
            fig.add_trace(
                go.Scattergl(
                    x=hyd["front_kN"],
                    y=hyd["rear_kN"],
                    mode="markers",
                    marker=dict(
                        color="rgba(235,235,235,0.0)",
                        line=dict(color=driver_color(run_name), width=1.2),
                        size=6,
                        symbol="circle-open",
                        opacity=0.55,
                    ),
                    name=f"{run_name} BSE demand",
                    hovertemplate=(
                        f"{run_name} BSE demand<br>Ff=%{{x:.2f}} kN"
                        "<br>Fr=%{y:.2f} kN<extra></extra>"
                    ),
                )
            )

    visible_front_kN = [float(np.nanpercentile(p["front_kN"], 99.7)) for p in run_payloads]
    visible_rear_kN = [float(np.nanpercentile(p["rear_kN"], 99.7)) for p in run_payloads]
    for p in run_payloads:
        hyd = p.get("hyd")
        if isinstance(hyd, dict):
            visible_front_kN.append(float(np.nanpercentile(hyd["front_kN"], 99.7)))
            visible_rear_kN.append(float(np.nanpercentile(hyd["rear_kN"], 99.7)))
    for front_lim_kN, rear_lim_kN in ideal_curve_limits:
        visible_front_kN.append(front_lim_kN)
        visible_rear_kN.append(rear_lim_kN)
    axis_max = max(visible_front_kN + visible_rear_kN) * 1.12
    fig.update_layout(
        paper_bgcolor="#141417",
        plot_bgcolor="#141417",
        font=dict(color="#EBEBEB", size=11),
        height=860,
        margin=dict(l=80, r=120, t=35, b=75),
        legend=dict(
            bgcolor="rgba(20,20,23,0.85)",
            bordercolor="rgba(128,128,128,0.3)",
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1.0,
        ),
        hovermode="closest",
    )
    fig.update_xaxes(title_text="Front braking force [kN]", range=[0.0, 5.0])
    fig.update_yaxes(
        title_text="Rear braking force [kN]", range=[0.0, 2.0], scaleanchor="x", scaleratio=1
    )

    return fig, {
        "runs": kpi_runs,
        "warnings": warnings,
        "front_balance_reference": BRAKE_FRONT_BALANCE,
        "mu_tire": MU_TIRE,
    }


# ── 9. Acceleration/traction helpers ─────────────────────────────────────────


def _accel_envelope_curves(
    vx_mps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Model tractive-force envelopes [N] vs speed [m/s]."""
    vx = np.asarray(vx_mps, dtype=float)
    aero_n = 0.5 * RHO_AIR_KGM3 * (vx**2) * abs(CL_AERO) * A_AERO_M2
    fx_tire_n = MU_TIRE * (MASS_KG * G_MS2 + aero_n)
    fx_torque_n = np.full_like(
        vx, N_MOTORS * T_MOTOR_MAX_NM * GEAR_RATIO / WHEEL_RADIUS_M, dtype=float
    )
    fx_power_n = MAX_POWER_W / np.maximum(vx, 0.5)
    fx_max_n = np.minimum(np.minimum(fx_tire_n, fx_torque_n), fx_power_n)
    return fx_tire_n, fx_torque_n, fx_power_n, fx_max_n


def _empty_xy_fig(title: str, xlabel: str, ylabel: str, message: str) -> go.Figure:
    fig = make_dark_figure(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        text=message,
        font=dict(color="#EBEBEB", size=12),
    )
    return fig


def _binned_percentile(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bin_width: float,
    x_min: float,
    x_max: float,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.arange(x_min, x_max + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.full_like(centers, np.nan, dtype=float)
    counts = np.zeros_like(centers, dtype=int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (x >= lo) & (x < hi) & np.isfinite(y)
        counts[i] = int(m.sum())
        if counts[i] >= 5:
            values[i] = float(np.nanpercentile(y[m], percentile))
    return centers, values, counts


def decel_envelope_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """P95 longitudinal deceleration by speed bin during braking events."""
    required = ["Filtering_VN_ax", "VN_vx", "Brake"]
    if not dfs:
        return _empty_xy_fig(
            "Longitudinal Decel Envelope", "Vehicle speed [m/s]", "P95 |ax| [g]", "No runs selected"
        ), {"runs": {}, "warnings": ["No runs selected."]}

    reference_g = 1.79
    fig = make_dark_figure(
        "Longitudinal Decel Envelope", "Vehicle speed [m/s]", "Deceleration |ax| [g]"
    )
    warnings: list[str] = []
    runs: dict[str, dict[str, float]] = {}
    for idx, (run_name, df_in) in enumerate(dfs.items()):
        df = ensure_complete_laps_df(df_in)
        missing = [c for c in required if c not in df.columns]
        if missing:
            warnings.append(f"{run_name}: missing decel-envelope columns: {missing}")
            continue
        arr = cols_to_numpy(df, required)
        vx = np.abs(arr["VN_vx"])
        ax = arr["Filtering_VN_ax"]
        brake = arr["Brake"]
        m = (ax < -1.0) & (brake > 5.0) & np.isfinite(vx)
        if not m.any():
            warnings.append(f"{run_name}: no braking samples for decel envelope.")
            continue
        vx_b = vx[m]
        decel_g = np.abs(ax[m]) / G_MPS2
        centers, p95_g, counts = _binned_percentile(
            vx_b,
            decel_g,
            bin_width=5.0,
            x_min=0.0,
            x_max=40.0,
            percentile=95.0,
        )
        valid = np.isfinite(p95_g)
        color = driver_color(run_name)
        stride = max(1, int(np.ceil(vx_b.size / 6000)))
        fig.add_trace(
            go.Scattergl(
                x=vx_b[::stride],
                y=decel_g[::stride],
                mode="markers",
                marker=dict(color=color, size=3, opacity=0.10),
                name=f"{run_name} samples",
                legendgroup=run_name,
                showlegend=False,
                hovertemplate=f"{run_name}<br>V=%{{x:.1f}} m/s<br>|ax|=%{{y:.2f}} g<extra></extra>",
            )
        )
        marker_sizes = np.clip(
            8.0 + counts[valid] / max(float(np.nanmax(counts[valid])), 1.0) * 10.0, 8.0, 18.0
        )
        customdata = np.column_stack(
            [
                counts[valid],
                reference_g - p95_g[valid],
                np.divide(
                    p95_g[valid],
                    reference_g,
                    out=np.full_like(p95_g[valid], np.nan, dtype=float),
                    where=reference_g > 0.0,
                )
                * 100.0,
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=centers[valid],
                y=p95_g[valid],
                mode="lines+markers",
                line=dict(color=color, width=3.0),
                marker=dict(
                    color=color,
                    size=marker_sizes,
                    line=dict(color="#EBEBEB", width=0.8),
                ),
                customdata=customdata,
                name=f"{run_name} p95 envelope",
                legendgroup=run_name,
                hovertemplate=(
                    "Speed bin center=%{x:.1f} m/s<br>"
                    "P95 |ax|=%{y:.2f} g<br>"
                    "Gap to design=%{customdata[1]:.2f} g<br>"
                    "Design use=%{customdata[2]:.1f}%<br>"
                    "Samples=%{customdata[0]:.0f}<extra></extra>"
                ),
            )
        )
        peak_idx = int(np.nanargmax(p95_g)) if valid.any() else -1
        peak_g = float(np.nanmax(p95_g)) if valid.any() else np.nan
        speed_at_peak = (
            float(centers[peak_idx]) if peak_idx >= 0 and np.isfinite(p95_g[peak_idx]) else np.nan
        )
        runs[run_name] = {
            "peak_decel_p95_g": peak_g,
            "speed_at_peak_mps": speed_at_peak,
            "gap_to_design_g": float(reference_g - peak_g) if np.isfinite(peak_g) else np.nan,
            "pct_design_decel": float(peak_g / reference_g * 100.0)
            if np.isfinite(peak_g)
            else np.nan,
            "samples": int(m.sum()),
        }
    fig.add_hline(
        y=reference_g,
        line=dict(color="#73D973", width=2.4, dash="dash"),
        annotation_text="CAT17x design target 1.79 g",
        annotation_position="top right",
    )
    fig.add_hrect(
        y0=reference_g,
        y1=max(2.05, reference_g * 1.08),
        fillcolor="rgba(115,217,115,0.08)",
        line_width=0,
        annotation_text="design target zone",
        annotation_position="top left",
    )
    fig.update_layout(
        height=560,
        margin=dict(l=70, r=35, t=55, b=65),
        hovermode="closest",
        legend=dict(
            bgcolor="rgba(20,20,23,0.85)",
            bordercolor="rgba(128,128,128,0.3)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
    )
    fig.update_xaxes(range=[0.0, 40.0])
    fig.update_yaxes(range=[0.0, 2.05])
    return fig, {"runs": runs, "warnings": warnings, "reference_decel_g": reference_g}


def _ideal_traction_forces(
    ax_mps2: np.ndarray,
    vx_mps: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ideal front/rear drive-force split from acceleration load distribution."""
    ax = np.asarray(ax_mps2, dtype=float)
    vx = np.asarray(vx_mps, dtype=float)
    aero_n = 0.5 * RHO_AIR_KGM3 * (vx**2) * abs(CL_AERO) * A_AERO_M2
    front_aero_frac = _aero_front_fraction()
    d_fz_n = MASS_KG * ax * COG_Z_M / WHEELBASE_M
    fz_front_n = MASS_KG * G_MPS2 * LR_M / WHEELBASE_M - d_fz_n + aero_n * front_aero_frac
    fz_rear_n = MASS_KG * G_MPS2 * LF_M / WHEELBASE_M + d_fz_n + aero_n * (1.0 - front_aero_frac)
    fz_total_n = fz_front_n + fz_rear_n
    front_frac = np.divide(
        fz_front_n,
        fz_total_n,
        out=np.full_like(fz_front_n, np.nan, dtype=float),
        where=np.isfinite(fz_total_n) & (fz_total_n > 0.0),
    )
    total_fx_n = MASS_KG * ax
    return total_fx_n * front_frac, total_fx_n * (1.0 - front_frac), front_frac


def _drive_force_from_motor_torque(torque_nm: np.ndarray) -> np.ndarray:
    """Positive drive force [N] from positive motor torque [N.m]."""
    return np.maximum(0.0, torque_nm) * GEAR_RATIO / WHEEL_RADIUS_M


def _rms_distance_to_ideal_traction_curve(
    ff_n: np.ndarray,
    fr_n: np.ndarray,
    vx_mps: np.ndarray,
    ax_grid_mps2: np.ndarray,
) -> np.ndarray:
    out = np.full_like(ff_n, np.nan, dtype=float)
    for i, (front_force_n, rear_force_n, vx) in enumerate(zip(ff_n, fr_n, vx_mps)):
        f_curve, r_curve, _front_frac = _ideal_traction_forces(ax_grid_mps2, vx)
        dist = np.hypot(f_curve - front_force_n, r_curve - rear_force_n)
        out[i] = float(np.nanmin(dist)) if np.any(np.isfinite(dist)) else np.nan
    return out


def ideal_traction_curve_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """Compare ideal AWD front/rear drive distribution with measured drive force."""
    required = [
        "Filtering_VN_ax",
        "VN_vx",
        "Throttle",
        "Est_FXFL",
        "Est_FXFR",
        "Est_FXRL",
        "Est_FXRR",
    ]
    if not dfs:
        return _empty_xy_fig(
            "Ideal Traction Curve vs Measured Drive",
            "Front drive force [kN]",
            "Rear drive force [kN]",
            "No runs selected",
        ), {"runs": {}, "warnings": ["No runs selected."]}

    ax_grid = np.linspace(0.0, MU_TIRE * G_MPS2, 200)
    fig = make_dark_figure(
        "Ideal Traction Curve  ·  Model vs Measured Drive",
        "Front drive force [kN]",
        "Rear drive force [kN]",
    )
    warnings: list[str] = []
    runs: dict[str, dict[str, float]] = {}
    payloads: list[dict[str, object]] = []
    color_max = 1.0

    for run_name, df_in in dfs.items():
        df = ensure_complete_laps_df(df_in)
        missing = [c for c in required if c not in df.columns]
        if missing:
            warnings.append(f"{run_name}: missing traction columns: {missing}")
            continue
        arr = cols_to_numpy(df, required)
        valid = _base_validity(*(arr[c] for c in required))
        arr = {k: v[valid] for k, v in arr.items()}
        ff_n = arr["Est_FXFL"] + arr["Est_FXFR"]
        fr_n = arr["Est_FXRL"] + arr["Est_FXRR"]
        total_n = ff_n + fr_n
        vx = np.abs(arr["VN_vx"])
        ax = arr["Filtering_VN_ax"]
        m = (ax > 1.0) & (arr["Throttle"] > 5.0) & (ff_n > 0.0) & (fr_n > 0.0)
        if not m.any():
            warnings.append(f"{run_name}: no drive samples for ideal traction curve.")
            continue
        ff_net = ff_n[m]
        fr_net = fr_n[m]
        total_m = total_n[m]
        vx_m = vx[m]
        ax_m = ax[m]
        rear_bias = np.divide(
            fr_net, total_m, out=np.full_like(fr_net, np.nan), where=total_m > 0.0
        )
        eq_ax = np.clip(ax_m, 0.0, MU_TIRE * G_MPS2)
        _idf, _idr, ideal_front_bias = _ideal_traction_forces(eq_ax, vx_m)
        ideal_rear_bias = 1.0 - ideal_front_bias
        dist_to_ideal_n = _rms_distance_to_ideal_traction_curve(ff_net, fr_net, vx_m, ax_grid)
        fx_tire_n, _fx_torque_n, fx_power_n, _fx_max_n = _accel_envelope_curves(vx_m)
        grip_limited = fx_tire_n <= fx_power_n
        color_max = max(color_max, float(np.nanpercentile(ax_m, 95.0)))
        mean_vx = float(np.nanmean(vx_m))
        payloads.append(
            {
                "run_name": run_name,
                "front_kN": ff_net / 1000.0,
                "rear_kN": fr_net / 1000.0,
                "ax": ax_m,
                "mean_vx": mean_vx,
            }
        )
        runs[run_name] = {
            "rear_bias_mean": float(np.nanmean(rear_bias)),
            "rear_bias_ideal_mean": float(np.nanmean(ideal_rear_bias)),
            "rms_dist_to_ideal_N": float(np.sqrt(np.nanmean(dist_to_ideal_n**2))),
            "peak_combined_accel_g": float(np.nanmax(ax_m) / G_MPS2),
            "pct_time_grip_limited": float(np.nanmean(grip_limited) * 100.0),
            "pct_time_power_limited": float(np.nanmean(~grip_limited) * 100.0),
            "samples": int(m.sum()),
        }

    for idx, payload in enumerate(payloads):
        mean_vx = payload["mean_vx"]
        f_curve, r_curve, _ = _ideal_traction_forces(ax_grid, mean_vx)
        color = driver_color(str(payload["run_name"]))
        fig.add_trace(
            go.Scatter(
                x=f_curve / 1000.0,
                y=r_curve / 1000.0,
                mode="lines",
                line=dict(color=color, width=2.2),
                name=f"Ideal {payload['run_name']} ({mean_vx:.0f} m/s)",
            )
        )
    for payload in payloads:
        run_name = str(payload["run_name"])
        fig.add_trace(
            go.Scattergl(
                x=payload["front_kN"],
                y=payload["rear_kN"],
                mode="markers",
                marker=dict(color=driver_color(run_name), size=4, opacity=0.46),
                name=f"{run_name} drive",
                customdata=payload["ax"],
                hovertemplate=(
                    f"{run_name}<br>Ff=%{{x:.2f}} kN<br>Fr=%{{y:.2f}} kN"
                    "<br>ax=%{customdata:.2f} m/s²<extra></extra>"
                ),
            )
        )
    force_limit_kN = MU_TIRE * MASS_KG * G_MPS2 / 1000.0
    fig.add_trace(
        go.Scatter(
            x=[0.0, force_limit_kN],
            y=[force_limit_kN, 0.0],
            mode="lines",
            line=dict(color="rgba(235,235,235,0.55)", width=1.5, dash="dot"),
            name=f"μ envelope v=0 ({MU_TIRE:.2f})",
            hoverinfo="skip",
        )
    )
    fig.update_layout(height=760, margin=dict(l=80, r=110, t=55, b=75), hovermode="closest")
    fig.update_xaxes(range=[0.0, 2.0])
    fig.update_yaxes(range=[0.0, 2.0], scaleanchor="x", scaleratio=1)
    return fig, {"runs": runs, "warnings": warnings, "mu_tire": MU_TIRE}


def accel_envelope_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """Scatter vx vs ax in acceleration events, with grip and power-limit reference curves."""
    required = ["Filtering_VN_ax", "VN_vx", "Throttle"]
    if not dfs:
        return _empty_xy_fig(
            "Longitudinal Accel Envelope", "Vehicle speed [m/s]", "ax [g]", "No runs selected"
        ), {"runs": {}, "warnings": ["No runs selected."]}
    fig = make_dark_figure("Longitudinal Accel Envelope", "Vehicle speed [m/s]", "ax [g]")
    warnings: list[str] = []
    runs: dict[str, dict[str, float]] = {}
    all_ax_g: list[float] = []
    for idx, (run_name, df_in) in enumerate(dfs.items()):
        df = ensure_complete_laps_df(df_in)
        missing = [c for c in required if c not in df.columns]
        if missing:
            warnings.append(f"{run_name}: missing accel-envelope columns: {missing}")
            continue
        arr = cols_to_numpy(df, required)
        vx = np.abs(arr["VN_vx"])
        ax = arr["Filtering_VN_ax"]
        throttle = arr["Throttle"]
        m = (ax > 1.0) & (throttle > 5.0) & np.isfinite(vx)
        if not m.any():
            warnings.append(f"{run_name}: no acceleration samples for accel envelope.")
            continue
        vx_m = vx[m]
        ax_g_m = ax[m] / G_MPS2
        all_ax_g.append(float(np.nanmax(ax_g_m)))
        stride = max(1, int(np.ceil(vx_m.size / 8000)))
        color = driver_color(run_name)
        fig.add_trace(
            go.Scattergl(
                x=vx_m[::stride],
                y=ax_g_m[::stride],
                mode="markers",
                marker=dict(color=color, size=3, opacity=0.45),
                name=run_name,
                hovertemplate="vx=%{x:.1f} m/s<br>ax=%{y:.2f} g<extra></extra>",
            )
        )
        # P95 envelope line per run
        centers, p95_g, _ = _binned_percentile(
            vx_m, ax_g_m, bin_width=3.0, x_min=0.0, x_max=40.0, percentile=95.0
        )
        valid = np.isfinite(p95_g)
        if valid.any():
            fig.add_trace(
                go.Scatter(
                    x=centers[valid],
                    y=p95_g[valid],
                    mode="lines",
                    line=dict(color=color, width=2.2),
                    showlegend=False,
                    hovertemplate="vx=%{x:.1f} m/s<br>P95=%{y:.2f} g<extra></extra>",
                )
            )
        runs[run_name] = {
            "peak_ax_g": float(np.nanmax(ax_g_m)),
            "samples": int(m.sum()),
        }
    # Reference curves — power limit clipped to grip limit to keep y-axis sane
    v_grid = np.linspace(1.0, 40.0, 400)
    power_g = MAX_POWER_W / (MASS_KG * v_grid) / G_MPS2
    power_g_clipped = np.minimum(
        power_g, MU_TIRE * 1.05
    )  # clip for display; crossover is the key feature
    crossover_mps = MAX_POWER_W / (MASS_KG * MU_TIRE * G_MPS2)
    fig.add_trace(
        go.Scatter(
            x=v_grid,
            y=np.full_like(v_grid, MU_TIRE),
            mode="lines",
            line=dict(color="#73D973", width=1.8, dash="dash"),
            name=f"Grip limit μ={MU_TIRE:.2f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=v_grid,
            y=power_g_clipped,
            mode="lines",
            line=dict(color="#F2D44D", width=1.8, dash="dash"),
            name=f"80 kW power limit (crossover {crossover_mps:.1f} m/s)",
        )
    )
    y_max = max(MU_TIRE * 1.15, max(all_ax_g) * 1.15) if all_ax_g else MU_TIRE * 1.15
    fig.update_layout(height=520, margin=dict(l=70, r=30, t=50, b=65))
    fig.update_yaxes(range=[0.0, y_max])
    fig.update_xaxes(range=[0.0, 40.0])
    return fig, {
        "runs": runs,
        "warnings": warnings,
        "mu_tire": MU_TIRE,
        "max_power_w": MAX_POWER_W,
        "crossover_mps": crossover_mps,
    }


def lateral_load_transfer_fig(df: pl.DataFrame) -> tuple[go.Figure, dict]:
    """Front lateral-load-transfer share in radius-filtered corners."""
    required = [
        "TimeStamp",
        "Filtering_VN_ay",
        "VN_vx",
        "Est_FZFL",
        "Est_FZFR",
        "Est_FZRL",
        "Est_FZRR",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing LTD columns: {missing}")
    df = ensure_complete_laps_df(df)
    cols = list(required)
    geom_cols = ["Front_Roll_Spring_Length", "Rear_Roll_Spring_Length"]
    geom_ok = _t1_calibration_ok(df) and all(c in df.columns for c in geom_cols)
    if geom_ok:
        cols += geom_cols
    arr = cols_to_numpy(df, cols)
    valid = _base_validity(*(arr[c] for c in required))
    arr = {k: v[valid] for k, v in arr.items()}
    time_s = arr["TimeStamp"] - arr["TimeStamp"][0]
    dt = robust_dt(time_s)
    cm, _radius_m = _radius_corner_mask(
        arr["VN_vx"], arr["Filtering_VN_ay"], dt, radius_threshold_m=60.0
    )
    dfz_front = arr["Est_FZFR"] - arr["Est_FZFL"]
    dfz_rear = arr["Est_FZRR"] - arr["Est_FZRL"]
    denom = np.abs(dfz_front) + np.abs(dfz_rear)
    ltd_front = np.divide(
        np.abs(dfz_front),
        denom,
        out=np.full_like(denom, np.nan, dtype=float),
        where=np.isfinite(denom) & (denom > 1.0),
    )
    ay_g = np.abs(arr["Filtering_VN_ay"]) / G_MPS2
    m = cm & np.isfinite(ltd_front) & np.isfinite(ay_g)

    theory_share = KROLLF_NMRAD / (KROLLF_NMRAD + KROLLR_NMRAD)
    theory_slope = KROLLR_NMRAD / KROLLF_NMRAD

    fig = make_dark_figure(
        "Lateral Load Transfer Distribution  ·  Deviation from CAT17x target",
        "|ay| [g]",
        "Front LTD deviation [percentage points]",
    )

    measured_slope = np.nan
    measured_share = np.nan
    binned_median = np.array([], dtype=float)
    if m.any():
        x = ay_g[m]
        y_share_pct = ltd_front[m] * 100.0
        y_dev_pp = y_share_pct - theory_share * 100.0
        stride = max(1, int(np.ceil(x.size / 12000)))
        fig.add_trace(
            go.Scattergl(
                x=x[::stride],
                y=y_dev_pp[::stride],
                mode="markers",
                marker=dict(
                    color="#4DB3F2",
                    size=5,
                    opacity=0.62,
                    line=dict(color="rgba(235,235,235,0.25)", width=0.4),
                ),
                name="Corner samples",
                customdata=y_share_pct[::stride],
                hovertemplate=(
                    "|ay|=%{x:.2f} g<br>"
                    "Deviation=%{y:+.1f} pp<br>"
                    "Front LTD=%{customdata:.1f}%<extra></extra>"
                ),
            )
        )

        bin_edges = np.arange(0.0, max(2.6, float(np.nanpercentile(x, 99.0)) + 0.25), 0.25)
        centers: list[float] = []
        p10: list[float] = []
        p50: list[float] = []
        p90: list[float] = []
        ltd50: list[float] = []
        counts: list[int] = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            bm = (x >= lo) & (x < hi) & np.isfinite(y_share_pct)
            if int(bm.sum()) < 20:
                continue
            centers.append(0.5 * (lo + hi))
            p10.append(float(np.nanpercentile(y_dev_pp[bm], 10.0)))
            p50.append(float(np.nanpercentile(y_dev_pp[bm], 50.0)))
            p90.append(float(np.nanpercentile(y_dev_pp[bm], 90.0)))
            ltd50.append(float(np.nanpercentile(y_share_pct[bm], 50.0)))
            counts.append(int(bm.sum()))
        if centers:
            c = np.asarray(centers, dtype=float)
            lo = np.asarray(p10, dtype=float)
            med = np.asarray(p50, dtype=float)
            hi = np.asarray(p90, dtype=float)
            ltd_med = np.asarray(ltd50, dtype=float)
            count_arr = np.asarray(counts, dtype=int)
            binned_median = med
            customdata = np.column_stack([ltd_med, count_arr, lo, hi])
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([c, c[::-1]]),
                    y=np.concatenate([hi, lo[::-1]]),
                    mode="lines",
                    line=dict(color="rgba(77,179,242,0.0)"),
                    fill="toself",
                    fillcolor="rgba(77,179,242,0.14)",
                    name="P10-P90 band",
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=c,
                    y=med,
                    mode="lines+markers",
                    line=dict(color="#F2A03D", width=2.8),
                    marker=dict(
                        color="#F2A03D",
                        size=9,
                        line=dict(color="#EBEBEB", width=0.8),
                    ),
                    customdata=customdata,
                    name="Median deviation by |ay| bin",
                    hovertemplate=(
                        "|ay| bin center=%{x:.2f} g<br>"
                        "Median deviation=%{y:+.1f} pp<br>"
                        "Median front LTD=%{customdata[0]:.1f}%<br>"
                        "P10/P90=%{customdata[2]:+.1f}/%{customdata[3]:+.1f} pp<br>"
                        "samples=%{customdata[1]:.0f}<extra></extra>"
                    ),
                )
            )

        xf = dfz_front[m]
        yr = dfz_rear[m]
        slope_denom = float(np.sum(xf * xf))
        if slope_denom > 1.0:
            measured_slope = float(np.sum(xf * yr) / slope_denom)
        measured_share = float(np.nanmedian(ltd_front[m]))

    fig.add_hline(
        y=0.0,
        line=dict(color="#73D973", width=2.6, dash="dash"),
        annotation_text=f"Target 0 pp = {theory_share * 100.0:.1f}% front LTD",
        annotation_position="top right",
    )
    if np.isfinite(measured_share):
        fig.add_hline(
            y=(measured_share - theory_share) * 100.0,
            line=dict(color="#4DB3F2", width=2.0, dash="dot"),
            annotation_text=f"Overall median {(measured_share - theory_share) * 100.0:+.1f} pp",
            annotation_position="bottom right",
        )

    fig.update_layout(
        height=560,
        margin=dict(l=70, r=35, t=55, b=65),
        hovermode="closest",
        legend=dict(
            bgcolor="rgba(20,20,23,0.85)",
            bordercolor="rgba(128,128,128,0.3)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
    )
    if m.any():
        y_range_src = ltd_front[m] * 100.0 - theory_share * 100.0
        y_abs = float(np.nanpercentile(np.abs(y_range_src), 99.0))
        y_lim = min(25.0, max(6.0, y_abs * 1.25))
        x_lim = max(2.5, float(np.nanpercentile(ay_g[m], 99.0)) * 1.05)
    else:
        y_lim = 10.0
        x_lim = 2.5
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.98,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        align="left",
        text=(
            "points: individual corner samples<br>"
            "orange line: median by |ay| bin<br>"
            "blue band: P10-P90 spread"
        ),
        font=dict(size=11, color="#CFCFCF"),
        bgcolor="rgba(20,20,23,0.78)",
        bordercolor="rgba(128,128,128,0.30)",
        borderwidth=1,
        borderpad=4,
    )
    fig.update_xaxes(range=[0.0, x_lim])
    fig.update_yaxes(range=[-y_lim, y_lim], zeroline=True, zerolinecolor="#73D973")

    geom_mean = np.nan
    if geom_ok:
        geom_denom = np.abs(arr["Front_Roll_Spring_Length"]) + np.abs(
            arr["Rear_Roll_Spring_Length"]
        )
        geom_ltd = np.divide(
            np.abs(arr["Front_Roll_Spring_Length"]),
            geom_denom,
            out=np.full_like(geom_denom, np.nan),
            where=geom_denom > 1e-6,
        )
        gm = cm & np.isfinite(geom_ltd)
        geom_mean = float(np.nanmean(geom_ltd[gm])) if gm.any() else np.nan

    return fig, {
        "samples": int(m.sum()),
        "ltd_front_median": measured_share,
        "ltd_theoretical": float(theory_share),
        "deviation_pct": float((measured_share - theory_share) / theory_share * 100.0)
        if np.isfinite(measured_share)
        else np.nan,
        "median_abs_error_pct_points": float(np.nanmedian(np.abs(binned_median)))
        if binned_median.size
        else np.nan,
        "measured_slope": measured_slope,
        "theory_slope": float(theory_slope),
        "krollf_nmrad": float(KROLLF_NMRAD),
        "krollr_nmrad": float(KROLLR_NMRAD),
        "geom_ltd_front_mean": geom_mean,
        "warnings": [] if m.any() else ["No radius-filtered corner samples for LTD."],
    }


def pitch_gradient_fig(df: pl.DataFrame) -> tuple[go.Figure, dict]:
    """Pitch angle vs longitudinal acceleration, split braking/acceleration."""
    df = ensure_complete_laps_df(df)
    t1_ok = _t1_calibration_ok(df) and "Pitch" in df.columns
    required = [
        "TimeStamp",
        "Filtering_VN_ax",
        "Brake",
        "Throttle",
        "DampFL",
        "DampFR",
        "DampRL",
        "DampRR",
    ]
    if t1_ok:
        required.append("Pitch")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing pitch columns: {missing}")
    arr = cols_to_numpy(df, required)
    valid = _base_validity(*arr.values())
    arr = {k: v[valid] for k, v in arr.items()}
    if t1_ok:
        pitch_deg = arr["Pitch"]
        calibrated = True
    else:
        damp = {}
        for w in ("FL", "FR", "RL", "RR"):
            mm = _damper_mm(arr[f"Damp{w}"], w)
            damp[w] = mm - float(np.nanmedian(mm))
        front_avg = 0.5 * (
            _wheel_mm_from_damper_mm(damp["FL"], "FL") + _wheel_mm_from_damper_mm(damp["FR"], "FR")
        )
        rear_avg = 0.5 * (
            _wheel_mm_from_damper_mm(damp["RL"], "RL") + _wheel_mm_from_damper_mm(damp["RR"], "RR")
        )
        pitch_deg = np.rad2deg(np.arctan2((front_avg - rear_avg) / 1000.0, WHEELBASE_M))
        calibrated = bool(DAMPER_CALIBRATED)
    ax_g = arr["Filtering_VN_ax"] / G_MPS2
    masks = {
        "brake": (arr["Filtering_VN_ax"] < -1.0) & (arr["Brake"] > 5.0),
        "accel": (arr["Filtering_VN_ax"] > 1.0) & (arr["Throttle"] > 5.0),
    }
    fig = make_dark_figure(
        "Pitch Gradient  ·  Braking vs Acceleration",
        "Longitudinal acceleration ax [g]",
        "Pitch angle [deg]" + ("" if calibrated else " (uncalibrated)"),
    )
    colors = {"brake": "#F27070", "accel": "#73D973"}
    kpis: dict[str, float | int | list[str] | bool] = {"calibrated": calibrated}
    for phase, mask in masks.items():
        m = mask & np.isfinite(ax_g) & np.isfinite(pitch_deg)
        fig.add_trace(
            go.Scattergl(
                x=ax_g[m],
                y=pitch_deg[m],
                mode="markers",
                marker=dict(color=colors[phase], size=4, opacity=0.40),
                name=f"{phase} samples",
            )
        )
        kpis[f"{phase}_samples"] = int(m.sum())
        if int(m.sum()) >= 20:
            slope, intercept = np.polyfit(ax_g[m], pitch_deg[m], 1)
            xfit = np.linspace(float(np.nanmin(ax_g[m])), float(np.nanmax(ax_g[m])), 50)
            fig.add_trace(
                go.Scatter(
                    x=xfit,
                    y=slope * xfit + intercept,
                    mode="lines",
                    line=dict(color=colors[phase], width=2.4),
                    name=f"{phase} fit · {slope:+.2f} deg/g",
                )
            )
            kpis[f"{phase}_gradient_deg_per_g"] = float(slope)
        else:
            kpis[f"{phase}_gradient_deg_per_g"] = np.nan
    fig.add_hline(y=0.0, line=dict(color="rgba(200,200,200,0.4)", dash="dot", width=1))
    fig.add_vline(x=0.0, line=dict(color="rgba(200,200,200,0.4)", dash="dot", width=1))
    kpis["warnings"] = (
        []
        if calibrated
        else ["Pitch is derived from uncalibrated damper counts; compare shape only."]
    )
    return fig, kpis


def static_fz_reference_fig(df: pl.DataFrame) -> tuple[go.Figure, dict]:
    """Per-corner static Fz vs CAT17x design, revealing weight distribution and side-to-side asymmetry."""
    required = [
        "VN_vx",
        "Filtering_VN_ax",
        "Filtering_VN_ay",
        "Throttle",
        "Brake",
        "Est_FZFL",
        "Est_FZFR",
        "Est_FZRL",
        "Est_FZRR",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing static-Fz columns: {missing}")
    df = ensure_complete_laps_df(df)
    arr = cols_to_numpy(df, required)
    m = (
        (np.abs(arr["VN_vx"]) > 4.0)
        & (np.abs(arr["Filtering_VN_ax"]) < 1.0)  # exclude coast-down / aero drag decel
        & (np.abs(arr["Filtering_VN_ay"]) < 0.5)
        & (arr["Throttle"] < 5.0)
        & (arr["Brake"] < 5.0)
    )
    DESIGN_N = (288.0 / 4.0) * G_MPS2  # 706.32 N per corner (288 kg / 4)
    corners = ["FL", "FR", "RL", "RR"]
    keys = ["Est_FZFL", "Est_FZFR", "Est_FZRL", "Est_FZRR"]
    meas: dict[str, float] = {
        c: float(np.nanmean(arr[k][m])) if m.any() else np.nan for c, k in zip(corners, keys)
    }
    total = sum(meas.values()) if all(np.isfinite(v) for v in meas.values()) else np.nan
    dev_pct = {c: (meas[c] - DESIGN_N) / DESIGN_N * 100.0 for c in corners}
    share_pct = {
        c: meas[c] / total * 100.0 if np.isfinite(total) and total > 0 else np.nan for c in corners
    }
    front_share = (
        (meas["FL"] + meas["FR"]) / total * 100.0 if np.isfinite(total) and total > 0 else np.nan
    )
    left_share = (
        (meas["FL"] + meas["RL"]) / total * 100.0 if np.isfinite(total) and total > 0 else np.nan
    )
    cross_weight = (
        (meas["FL"] + meas["RR"]) / total * 100.0 if np.isfinite(total) and total > 0 else np.nan
    )

    def _bar_color(dev: float) -> str:
        if abs(dev) <= 5.0:
            return "#73D973"
        if abs(dev) <= 10.0:
            return "#F2C94C"
        return "#F25757"

    fig = make_dark_figure(
        "Static Fz Reference  ·  Corner Weights (Straight Samples)", "Corner", "Vertical load [N]"
    )
    fig.add_trace(
        go.Bar(
            x=corners,
            y=[meas[c] for c in corners],
            name="Measured",
            marker_color=[_bar_color(dev_pct[c]) for c in corners],
            text=[f"{dev_pct[c]:+.1f}%<br>{share_pct[c]:.1f}% of total" for c in corners],
            textposition="outside",
            textfont=dict(size=11, color="#CCCCCC"),
            showlegend=False,
        )
    )
    max_meas = max((meas[c] for c in corners if np.isfinite(meas[c])), default=DESIGN_N)
    fig.add_hline(
        y=DESIGN_N,
        line_dash="dash",
        line_color="#888888",
        line_width=1.5,
        annotation_text=f"Design {DESIGN_N:.0f} N (25%)",
        annotation_position="right",
        annotation_font=dict(color="#888888", size=11),
    )
    fig.update_layout(
        height=480,
        showlegend=False,
        margin=dict(l=70, r=150, t=55, b=55),
        yaxis=dict(range=[0, max_meas * 1.28]),
    )
    return fig, {
        "samples": int(m.sum()),
        "corners": {
            c: {"measured_n": meas[c], "deviation_pct": dev_pct[c], "share_pct": share_pct[c]}
            for c in corners
        },
        "front_share_pct": front_share,
        "left_share_pct": left_share,
        "cross_weight_pct": cross_weight,
        "design_corner_n": DESIGN_N,
        "warnings": [] if m.any() else ["No straight low-input samples for static Fz reference."],
    }


def aero_load_heave_fig(df: pl.DataFrame) -> tuple[go.Figure, dict]:
    """Measured heave vs speed in straight samples."""
    required = ["VN_vx", "Filtering_VN_ay", "Throttle", "Brake", "Heave_Front", "Heave_Rear"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing aero-load columns: {missing}")
    if not _t1_calibration_ok(df):
        raise ValueError("T1 potentiometer calibration is not validated.")
    df = ensure_complete_laps_df(df)
    arr = cols_to_numpy(df, required)
    m = (
        (np.abs(arr["VN_vx"]) > 4.0)
        & (np.abs(arr["Filtering_VN_ay"]) < 0.5)
        & (arr["Throttle"] < 5.0)
        & (arr["Brake"] < 5.0)
    )
    fig = make_dark_figure("Aero Load  ·  Heave vs Speed", "Vehicle speed [m/s]", "Heave [mm]")
    for name, color in [("Heave_Front", "#4DB3F2"), ("Heave_Rear", "#F28C40")]:
        fig.add_trace(
            go.Scattergl(
                x=np.abs(arr["VN_vx"][m]),
                y=arr[name][m],
                mode="markers",
                marker=dict(color=color, size=4, opacity=0.36),
                name=name.replace("_", " "),
            )
        )
    fig.update_layout(height=520, margin=dict(l=70, r=30, t=55, b=65))
    return fig, {"samples": int(m.sum()), "warnings": []}


def spring_velocity_histogram_figs(
    df: pl.DataFrame,
    phase: Literal["all", "brake", "corner", "accel", "straight"] = "all",
) -> tuple[list[go.Figure], dict]:
    """Heave/roll spring velocity histograms split by setup phase."""
    speed_cols = [
        "Length_front_heave_Speed",
        "Length_front_roll_Speed",
        "Length_rear_heave_Speed",
        "Length_rear_roll_Speed",
    ]
    required = ["TimeStamp"] + _DAMPER_PHASE_COLS + speed_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing spring velocity columns: {missing}")
    if not _t1_calibration_ok(df):
        raise ValueError("T1 potentiometer calibration is not validated.")
    df = ensure_complete_laps_df(df)
    arr = cols_to_numpy(df, required)
    time_s = arr["TimeStamp"] - arr["TimeStamp"][0]
    dt = robust_dt(time_s)
    sample_mask = _setup_phase_mask(arr, phase, dt)
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Front heave", "Front roll", "Rear heave", "Rear roll"),
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )
    positions = {
        "Length_front_heave_Speed": (1, 1, "#4DB3F2"),
        "Length_front_roll_Speed": (1, 2, "#D973D9"),
        "Length_rear_heave_Speed": (2, 1, "#F28C40"),
        "Length_rear_roll_Speed": (2, 2, "#73D973"),
    }
    split = DAMPER_LSHS_SPLIT_MMPS
    hs_share: dict[str, float] = {}
    for col, (row, col_idx, color) in positions.items():
        vel = arr[col][sample_mask]
        vel = vel[np.isfinite(vel)]
        if vel.size == 0:
            continue
        bound = max(
            abs(float(np.nanpercentile(vel, 1))), abs(float(np.nanpercentile(vel, 99))), split * 2.0
        )
        hs_share[col] = float(np.nanmean(np.abs(vel) >= split))
        fig.add_trace(
            go.Histogram(
                x=vel,
                xbins=dict(start=-bound, end=bound, size=(2.0 * bound) / 80.0),
                marker_color=color,
                opacity=0.82,
                name=col.replace("_Speed", "").replace("_", " "),
                showlegend=False,
            ),
            row=row,
            col=col_idx,
        )
        fig.add_vline(
            x=-split,
            line=dict(color="rgba(255,255,255,0.35)", dash="dot", width=1),
            row=row,
            col=col_idx,
        )
        fig.add_vline(
            x=split,
            line=dict(color="rgba(255,255,255,0.35)", dash="dot", width=1),
            row=row,
            col=col_idx,
        )
        fig.add_vline(
            x=0.0,
            line=dict(color="rgba(255,255,255,0.55)", dash="dash", width=1),
            row=row,
            col=col_idx,
        )
    fig.update_layout(
        title=dict(
            text=f"Spring velocity histograms · {phase.upper()}",
            font=dict(size=14, color="#EBEBEB"),
        ),
        paper_bgcolor="#141417",
        plot_bgcolor="#141417",
        font=dict(color="#EBEBEB", size=11),
        barmode="overlay",
        height=620,
        margin=dict(l=60, r=20, t=70, b=50),
    )
    for row in (1, 2):
        for col_idx in (1, 2):
            fig.update_xaxes(
                title_text="Spring velocity [mm/s]",
                gridcolor="rgba(128,128,128,0.2)",
                row=row,
                col=col_idx,
            )
            fig.update_yaxes(
                title_text="Samples", gridcolor="rgba(128,128,128,0.2)", row=row, col=col_idx
            )
    return [fig], {"phase": phase, "hs_share": hs_share, "warnings": []}


# ── Aero load helper ─────────────────────────────────────────────────────────


def _aero_front_fraction() -> float:
    """Front aero load fraction from CoP distance behind the front axle."""
    cop_from_front_m = abs(COP_X_FROM_FRONT)
    return float(np.clip((WHEELBASE_M - cop_from_front_m) / WHEELBASE_M, 0.0, 1.0))


# ── 11. Steering vs ay (steady-state US/OS curve) ────────────────────────────

_STEER_COLS = ["TimeStamp", "laps", "laptime", "Filtering_VN_ay", "Steering", "VN_vx"]


def steering_vs_ay_fig(df: pl.DataFrame) -> tuple[go.Figure, dict]:
    """Steering angle vs lateral acceleration in steady-state cornering.

    The slope at low |ay| is the linear-region understeer gradient
    (rad of road-wheel angle per m/s²).  Departure of the curve from the
    bicycle-model ideal δ = L·ay/vx² indicates US (above ideal) or OS (below).
    """
    df = ensure_complete_laps_df(df)
    missing = [c for c in _STEER_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing steering-vs-ay columns: {missing}")

    arr = cols_to_numpy(df, _STEER_COLS)
    valid = _base_validity(*arr.values())
    arr = {k: v[valid] for k, v in arr.items()}

    time_s = arr["TimeStamp"] - arr["TimeStamp"][0]
    dt = robust_dt(time_s)

    ay = arr["Filtering_VN_ay"]
    # `Steering` = steering-potentiometer value [rad], used directly per the
    # team's understeer formula — consistent with understeer_angle_fig and the
    # Ackermann ideal below. (Do NOT divide by STEERING_RATIO.)
    steer = arr["Steering"]
    vx = arr["VN_vx"]

    # Steady state: Lap-Analysis radius cornering plus low ay/steer jerk.
    smooth_n = max(1, int(round(0.10 / dt)))
    day_dt = np.gradient(smooth_signal(ay, smooth_n), dt)
    dsteer_dt = np.gradient(smooth_signal(steer, smooth_n), dt)
    radius_corner, _radius_m = _radius_corner_mask(vx, ay, dt, radius_threshold_m=60.0)
    raw_corner = radius_corner & (np.abs(day_dt) < 15.0) & (np.abs(dsteer_dt) < 1.5)
    cm = keep_min_duration_segments(raw_corner, MIN_CORNER_DURATION, dt)
    if not cm.any():
        fig = make_dark_figure(
            title="Steering vs ay  ·  steady-state cornering",
            xlabel="ay [m/s²]",
            ylabel="Steering [deg]",
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            text="No steady-state cornering samples found",
            font=dict(color="#EBEBEB", size=12),
        )
        return fig, {"warnings": ["No steady-state cornering samples for steering vs ay."]}

    ay_c = ay[cm]
    steer_c = np.rad2deg(steer[cm])
    vx_c = vx[cm]

    fig = make_dark_figure(
        title="Steering vs Lateral Acceleration  ·  US/OS handling curve",
        xlabel="Lateral acceleration ay [m/s²]",
        ylabel="Steering [deg]",
    )
    fig.add_trace(
        go.Scattergl(
            x=ay_c,
            y=steer_c,
            mode="markers",
            marker=dict(color="#4DB3F2", size=3, opacity=0.45),
            name="Samples",
        )
    )

    # Bicycle-model ideal: δ_ideal = L·ay/vx²  (rad)
    vx_ref = float(np.nanmedian(np.abs(vx_c)))
    ay_grid = np.linspace(np.nanmin(ay_c), np.nanmax(ay_c), 80)
    if vx_ref > 1.0:
        ideal_deg = np.rad2deg(WHEELBASE_M * ay_grid / (vx_ref**2))
        fig.add_trace(
            go.Scatter(
                x=ay_grid,
                y=ideal_deg,
                mode="lines",
                line=dict(color="#73D973", dash="dash", width=2.0),
                name=f"Ideal δ = L·ay/vx² (vx={vx_ref:.1f} m/s)",
            )
        )

    # Linear-region fit (|ay| < 4 m/s²) for the understeer gradient
    lin = np.abs(ay_c) < 4.0
    grad_deg_per_g = np.nan
    if lin.sum() >= 30:
        slope, intercept = np.polyfit(ay_c[lin], steer_c[lin], 1)
        xfit = np.linspace(np.nanmin(ay_c[lin]), np.nanmax(ay_c[lin]), 50)
        fig.add_trace(
            go.Scatter(
                x=xfit,
                y=slope * xfit + intercept,
                mode="lines",
                line=dict(color="#F28C40", width=2.2),
                name=f"Linear fit  ·  {slope:+.3f} deg/(m/s²)",
            )
        )
        grad_deg_per_g = float(slope) * 9.81

    fig.add_hline(y=0.0, line=dict(color="rgba(200,200,200,0.4)", dash="dot", width=1))
    fig.add_vline(x=0.0, line=dict(color="rgba(200,200,200,0.4)", dash="dot", width=1))

    kpis = {
        "samples": int(ay_c.size),
        "understeer_gradient_deg_per_g": grad_deg_per_g,
        "vx_median_mps": vx_ref,
        "warnings": [],
    }
    return fig, kpis


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    understeer_angle().show()


if __name__ == "__main__":
    main()
