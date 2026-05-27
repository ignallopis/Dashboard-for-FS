"""Skidpad event KPIs and figures.

The functions in this module are pure dashboard helpers: they accept a
``polars.DataFrame`` already loaded by ``utils.load_data`` and return a Plotly
figure plus a KPI dictionary. Rendering belongs in ``src/dashboard.py``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.dynamics import (
    G_MPS2,
    GEAR_RATIO,
    MASS_KG,
    STEERING_RATIO,
    WHEEL_RADIUS_M,
    WHEELBASE_EQ,
)
from src.lapcount import gps_to_local_xy
from utils import cols_to_numpy, make_dark_figure, robust_dt, smooth_signal

SKIDPAD_INNER_RADIUS_M = 7.625
SKIDPAD_OUTER_RADIUS_M = 10.625
SKIDPAD_IDEAL_RADIUS_M = 9.125
SKIDPAD_PATH_WIDTH_M = 3.0
SKIDPAD_CIRCLE_GAP_M = 18.25
SKIDPAD_TIMED_LAPS = (2, 4)

TRACK_F_M = 1.225
TRACK_R_M = 1.175
COG_HEIGHT_M = 0.278
KROLL_F_NM_RAD = 36929.4
KROLL_R_NM_RAD = 40833.7
LLTD_THEORY = KROLL_F_NM_RAD / (KROLL_F_NM_RAD + KROLL_R_NM_RAD)

AY_POSITIVE_SIDE = "L"
MIN_SKIDPAD_SPEED_MPS = 2.0
MIN_SKIDPAD_AY_MPS2 = 1.0
MIN_BALANCE_SPEED_MPS = 3.0
MIN_BALANCE_AY_MPS2 = 2.0
MIN_FIT_SAMPLES = 20

WHEELS = ("FL", "FR", "RL", "RR")
_TV_TORQUE_COLS = ("TV_FL_Trq", "TV_FR_Trq", "TV_RL_Trq", "TV_RR_Trq")
_TV_OPTIONAL_COLS = ("TV_desiredYawRate", "TV_errorYawRate", "TV_actualMz")
_ROLL_QUAT_COLS = ("VN_ox", "VN_oy", "VN_oz", "VN_ow")
_FZ_COLS = ("Est_FZFL", "Est_FZFR", "Est_FZRL", "Est_FZRR")
_SA_COLS = ("Est_SAFL", "Est_SAFR", "Est_SARL", "Est_SARR")
_THROTTLE_ALIASES = ("APPS", "pedals_throttle", "Throttle", "APPS1")
_BRAKE_ALIASES = ("BSE", "Brake", "BSEFront", "BSERear")
_YAW_ALIASES = ("VN_gz", "AS_yaw_rate")

_BG = "#141417"
_TEXT = "#EBEBEB"
_GRID = "rgba(128,128,128,0.2)"
_AXIS = "#E5E5E5"
_REFERENCE = "#F2D44D"
_RUN_COLORS = ("#4DB3F2", "#F28C40", "#73D973", "#D973D9", "#F27070", "#F2C94C")
_SIDE_COLORS = {"R": "#4DB3F2", "L": "#F28C40"}


def is_skidpad_run(df: pl.DataFrame) -> bool:
    """Return True if most samples are tagged as skidpad by lapcount."""
    if "lapcount_mode" not in df.columns or len(df) == 0:
        return False
    modes = [str(value).strip().lower() for value in df["lapcount_mode"].to_list()]
    return sum(value == "skidpad" for value in modes) > 0.5 * len(modes)


def has_tv_signals(df: pl.DataFrame) -> bool:
    """Return True when TV torque and yaw-error signals are available."""
    return all(col in df.columns for col in _TV_TORQUE_COLS) and "TV_errorYawRate" in df.columns


def has_load_signals(df: pl.DataFrame) -> bool:
    """Return True when all estimated vertical-load signals are available."""
    return all(col in df.columns for col in _FZ_COLS)


def has_roll_signals(df: pl.DataFrame) -> bool:
    """Return True when the VectorNav quaternion columns are available."""
    return all(col in df.columns for col in _ROLL_QUAT_COLS)


def classify_circles(df: pl.DataFrame) -> dict[int, dict[str, Any]]:
    """Classify each skidpad timed lap by side.

    The first lap seen for each side is treated as warm-up and dropped; only
    timed circles are returned. When more than one timed lap exists for a
    side, the quickest one is flagged as ``official_timed``.
    """
    required = ["TimeStamp", "laps", "Filtering_VN_ay"]
    if "laptime" in df.columns:
        required.append("laptime")
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)

    laps = arr["laps"]
    lap_ids = [int(lap) for lap in np.unique(laps[np.isfinite(laps)]) if int(lap) > 0]
    rows: dict[int, dict[str, Any]] = {}
    for lap_id in sorted(lap_ids):
        mask = laps == float(lap_id)
        if not mask.any():
            continue
        time_lap = arr["TimeStamp"][mask]
        ay_lap = arr["Filtering_VN_ay"][mask]
        mean_ay = _safe_mean(ay_lap)
        side = AY_POSITIVE_SIDE if mean_ay >= 0.0 else _opposite_side(AY_POSITIVE_SIDE)
        laptime_s = _lap_time_s(arr, mask)
        first_time_s = _safe_min(time_lap)
        rows[lap_id] = {
            "side": side,
            "role": "warmup",
            "laptime_s": laptime_s,
            "n_samples": int(mask.sum()),
            "mean_ay_mps2": mean_ay,
            "first_time_s": first_time_s,
            "official_timed": False,
        }

    for side in ("R", "L"):
        side_laps = sorted(
            [lap for lap, info in rows.items() if info["side"] == side],
            key=lambda lap: rows[lap]["first_time_s"],
        )
        for idx, lap_id in enumerate(side_laps):
            rows[lap_id]["role"] = "timed" if idx >= 1 else "warmup"
        timed_laps = [
            lap for lap in side_laps
            if rows[lap]["role"] == "timed" and np.isfinite(rows[lap]["laptime_s"])
        ]
        if timed_laps:
            best_lap = min(timed_laps, key=lambda lap: rows[lap]["laptime_s"])
            rows[best_lap]["official_timed"] = True

    timed_rows = {lap_id: info for lap_id, info in rows.items() if info["role"] == "timed"}
    return dict(sorted(timed_rows.items(), key=lambda item: item[1]["first_time_s"]))


def event_time_summary_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Return table and bar chart for skidpad circle times."""
    circles = classify_circles(df)
    rows = _circle_summary_rows(circles)
    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "table"}], [{"type": "xy"}]],
        row_heights=[0.42, 0.58],
        vertical_spacing=0.10,
    )
    _add_table_trace(fig, rows, row=1, col=1)

    labels = [_circle_label(lap_id, info) for lap_id, info in circles.items()]
    times = [float(info["laptime_s"]) for info in circles.values()]
    colors = [
        "#73D973" if info.get("official_timed", False) else "#F28C40"
        for info in circles.values()
    ]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=times,
            marker=dict(color=colors, line=dict(color="#22252B", width=1)),
            text=[_fmt_num(t, ".3f") for t in times],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:.3f} s<extra></extra>",
            name="Circle time",
        ),
        row=2,
        col=1,
    )

    official = _official_timed_times(circles)
    event_time_s = _mean_if_finite((official.get("R", np.nan), official.get("L", np.nan)))
    lr_asymmetry_s = _abs_diff(official.get("L", np.nan), official.get("R", np.nan))
    if np.isfinite(event_time_s):
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=[event_time_s] * len(labels),
                mode="lines",
                name=f"Event time {event_time_s:.3f} s",
                line=dict(color=_REFERENCE, dash="dash", width=1.2),
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

    _apply_dark_layout(fig, "Skidpad event time summary", height=650)
    fig.update_xaxes(title_text="Circle", row=2, col=1)
    fig.update_yaxes(title_text="Lap time [s]", row=2, col=1)

    return fig, {
        "event_time_s": event_time_s,
        "LR_asymmetry_s": lr_asymmetry_s,
        "timed_R_s": official.get("R", np.nan),
        "timed_L_s": official.get("L", np.nan),
        "circles": circles,
        "table": pl.DataFrame(rows),
        "warnings": _classification_warnings(circles),
    }


def lateral_g_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """|ay| distribution per timed circle (OptimumG-style histogram)."""
    circles = classify_circles(df)
    required = ["laps", "Filtering_VN_ay"]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)

    fig = make_dark_figure("Skidpad |ay| distribution", "|ay| [g]", "Time share [%]")

    rows: list[dict[str, object]] = []
    ay_max_global_g = np.nan
    any_data = False
    for idx, (lap_id, info) in enumerate(circles.items()):
        mask = arr["laps"] == float(lap_id)
        ay_g = np.abs(arr["Filtering_VN_ay"][mask]) / G_MPS2
        ay_g = ay_g[np.isfinite(ay_g)]
        if ay_g.size == 0:
            continue
        any_data = True
        color = _RUN_COLORS[idx % len(_RUN_COLORS)]
        fig.add_trace(
            go.Histogram(
                x=ay_g,
                name=_circle_label(lap_id, info),
                marker_color=color,
                opacity=0.55,
                xbins=dict(start=0.0, end=2.4, size=0.05),
                histnorm="percent",
                hovertemplate="%{x:.2f} g<br>%{y:.1f} %<extra></extra>",
            )
        )
        ay_max_global_g = np.nanmax([ay_max_global_g, np.nanmax(ay_g)])
        rows.append({
            "Lap": int(lap_id),
            "Side": info["side"],
            "ay_mean_g": _round(_safe_mean(ay_g), 3),
            "ay_max_g": _round(_safe_max(ay_g), 3),
            "ay_p95_g": _round(_safe_percentile(ay_g, 95), 3),
            "ay_std_g": _round(_safe_std(ay_g), 3),
            "Samples": int(ay_g.size),
        })

    fig.update_layout(barmode="overlay", height=440)
    fig.update_xaxes(range=[0.0, 2.4])
    warnings = _classification_warnings(circles)
    if not any_data:
        warnings.append("No |ay| samples available for the histogram.")
    return fig, {
        "ay_max_global_g": ay_max_global_g,
        "table": pl.DataFrame(rows),
        "per_circle": rows,
        "warnings": warnings,
    }


def driven_radius_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Driven-radius distribution per timed circle (OptimumG-style histogram)."""
    circles = classify_circles(df)
    required = ["TimeStamp", "laps", "Filtering_VN_ay", "VN_vx"]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)
    radius_m = _driven_radius_m(arr["TimeStamp"], arr["VN_vx"], arr["Filtering_VN_ay"])
    sustained = _sustained_radius_mask(arr["VN_vx"], arr["Filtering_VN_ay"], radius_m)

    fig = make_dark_figure("Driven radius distribution", "R [m]", "Time share [%]")
    fig.add_vrect(
        x0=SKIDPAD_INNER_RADIUS_M,
        x1=SKIDPAD_OUTER_RADIUS_M,
        fillcolor="rgba(115,217,115,0.12)",
        line_width=0,
        layer="below",
    )
    fig.add_vline(
        x=SKIDPAD_IDEAL_RADIUS_M,
        line=dict(color=_REFERENCE, dash="dash", width=1.4),
        annotation_text=f"Ideal {SKIDPAD_IDEAL_RADIUS_M:.3f} m",
        annotation_font_color=_TEXT,
    )

    rows: list[dict[str, object]] = []
    all_radius: list[np.ndarray] = []
    any_data = False
    for idx, (lap_id, info) in enumerate(circles.items()):
        mask = (arr["laps"] == float(lap_id)) & sustained
        values = radius_m[mask]
        values = values[np.isfinite(values)]
        values = values[(values >= 2.0) & (values <= 20.0)]
        if values.size == 0:
            continue
        any_data = True
        color = _RUN_COLORS[idx % len(_RUN_COLORS)]
        fig.add_trace(
            go.Histogram(
                x=values,
                name=_circle_label(lap_id, info),
                marker_color=color,
                opacity=0.55,
                xbins=dict(start=2.0, end=20.0, size=0.25),
                histnorm="percent",
                hovertemplate="%{x:.2f} m<br>%{y:.1f} %<extra></extra>",
            )
        )
        all_radius.append(values)
        pct_band = 100.0 * np.nanmean(
            (values >= SKIDPAD_INNER_RADIUS_M) & (values <= SKIDPAD_OUTER_RADIUS_M)
        )
        r_mean = _safe_mean(values)
        rows.append({
            "Lap": int(lap_id),
            "Side": info["side"],
            "R_mean_m": _round(r_mean, 3),
            "R_std_m": _round(_safe_std(values), 3),
            "pct_time_in_band_pct": _round(pct_band, 1),
            "radius_error_m": _round(r_mean - SKIDPAD_IDEAL_RADIUS_M, 3),
            "Samples": int(values.size),
        })

    radius_all = np.concatenate(all_radius) if all_radius else np.array([], dtype=float)
    r_mean_all = _safe_mean(radius_all)
    pct_band_all = (
        100.0 * np.nanmean(
            (radius_all >= SKIDPAD_INNER_RADIUS_M) & (radius_all <= SKIDPAD_OUTER_RADIUS_M)
        )
        if radius_all.size else np.nan
    )
    fig.update_layout(barmode="overlay", height=460)
    fig.update_xaxes(range=[2.0, 20.0])
    warnings = _classification_warnings(circles)
    if not any_data:
        warnings.append("No sustained-cornering samples for the radius histogram.")
    return fig, {
        "R_mean_m": r_mean_all,
        "R_std_m": _safe_std(radius_all),
        "pct_time_in_band_pct": pct_band_all,
        "radius_error_m": r_mean_all - SKIDPAD_IDEAL_RADIUS_M if np.isfinite(r_mean_all) else np.nan,
        "table": pl.DataFrame(rows),
        "per_circle": rows,
        "warnings": warnings,
    }


def balance_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Summarise front/rear slip-angle balance and understeer angle by circle."""
    circles = classify_circles(df)
    yaw_col = _first_existing_col(df, _YAW_ALIASES)
    if yaw_col is None:
        raise KeyError("Missing yaw-rate column: VN_gz/AS_yaw_rate")
    required = ["laps", "Filtering_VN_ay", "VN_vx", "Steering", yaw_col, *_SA_COLS]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)
    front_sa_deg, rear_sa_deg = _front_rear_sa_deg(arr)
    understeer_deg = _understeer_angle_deg(arr["Steering"], arr[yaw_col], arr["VN_vx"])
    sustained = _sustained_balance_mask(arr["VN_vx"], arr["Filtering_VN_ay"])

    rows: list[dict[str, object]] = []
    labels: list[str] = []
    front_means: list[float] = []
    rear_means: list[float] = []
    balance_idx: list[float] = []
    understeer_means: list[float] = []
    for lap_id, info in circles.items():
        mask = (arr["laps"] == float(lap_id)) & sustained
        good = (
            mask
            & np.isfinite(front_sa_deg)
            & np.isfinite(rear_sa_deg)
            & np.isfinite(understeer_deg)
        )
        if not good.any():
            continue
        sign = _turn_sign_from_side(info["side"])
        front_mean = _safe_mean(front_sa_deg[good])
        rear_mean = _safe_mean(rear_sa_deg[good])
        us_mean = _safe_mean(understeer_deg[good] * sign)
        bal = rear_mean - front_mean
        labels.append(_circle_label(lap_id, info))
        front_means.append(front_mean)
        rear_means.append(rear_mean)
        balance_idx.append(bal)
        understeer_means.append(us_mean)
        rows.append({
            "Lap": int(lap_id),
            "Side": info["side"],
            "SA_F_mean_deg": _round(front_mean, 3),
            "SA_R_mean_deg": _round(rear_mean, 3),
            "SA_F_p95_deg": _round(_safe_percentile(front_sa_deg[good], 95), 3),
            "SA_R_p95_deg": _round(_safe_percentile(rear_sa_deg[good], 95), 3),
            "balance_index_deg": _round(bal, 3),
            "understeer_angle_mean_deg": _round(us_mean, 3),
            "Samples": int(good.sum()),
        })

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Slip angle by axle", "Balance index and understeer angle"),
    )
    fig.add_trace(go.Bar(x=labels, y=front_means, name="Front SA mean", marker_color="#4DB3F2"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=rear_means, name="Rear SA mean", marker_color="#F28C40"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=balance_idx, name="SA_R - SA_F", marker_color="#73D973"), row=2, col=1)
    fig.add_trace(go.Bar(x=labels, y=understeer_means, name="Understeer angle", marker_color="#D973D9"), row=2, col=1)
    fig.add_hline(y=0.0, line=dict(color="rgba(235,235,235,0.45)", dash="dash", width=1), row=2, col=1)
    _apply_dark_layout(fig, "Skidpad balance", height=620)
    fig.update_yaxes(title_text="Slip angle [deg]", row=1, col=1)
    fig.update_yaxes(title_text="Angle [deg]", row=2, col=1)
    fig.update_layout(barmode="group")

    return fig, {
        "balance_index_deg": _safe_mean(np.asarray(balance_idx, dtype=float)),
        "understeer_angle_mean_deg": _safe_mean(np.asarray(understeer_means, dtype=float)),
        "table": pl.DataFrame(rows),
        "per_circle": rows,
        "warnings": _classification_warnings(circles),
    }


def understeer_gradient_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Fit dynamic steering error versus lateral acceleration."""
    circles = classify_circles(df)
    payload = _understeer_fit_payload(df, circles)
    x = payload["ay_g"]
    y = payload["delta_dyn_deg"]
    side_by_sample = payload["side"]
    fit = _linear_fit(x, y)

    fig = make_dark_figure(
        "Understeer gradient",
        "Lateral acceleration ay [g]",
        "delta_real - delta_ackermann [deg]",
    )
    fig.add_hline(y=0.0, line=dict(color="rgba(235,235,235,0.35)", dash="dash", width=1))
    fig.add_vline(x=0.0, line=dict(color="rgba(235,235,235,0.35)", dash="dash", width=1))

    per_side: dict[str, dict[str, float]] = {}
    for side in ("R", "L"):
        mask = side_by_sample == side
        if not mask.any():
            continue
        fig.add_trace(
            go.Scattergl(
                x=x[mask],
                y=y[mask],
                mode="markers",
                name=f"{side} samples",
                marker=dict(color=_SIDE_COLORS[side], size=4, opacity=0.45),
                hovertemplate="ay=%{x:+.2f} g<br>delta_dyn=%{y:+.2f} deg<extra></extra>",
            )
        )
        side_fit = _linear_fit(x[mask], y[mask])
        per_side[side] = side_fit
        if np.isfinite(side_fit["slope"]):
            _add_fit_line(fig, x[mask], side_fit, _SIDE_COLORS[side], f"{side} fit")

    if np.isfinite(fit["slope"]):
        _add_fit_line(fig, x, fit, _REFERENCE, "Global fit", dash="dash")

    fig.update_layout(height=520)
    warnings = _classification_warnings(circles)
    if not np.isfinite(fit["slope"]):
        warnings.append("Not enough sustained skidpad samples for understeer-gradient fit.")
    return fig, {
        "K_us_deg_per_g": fit["slope"],
        "R2_fit": fit["r2"],
        "delta_dyn_at_1g_deg": fit["slope"] + fit["intercept"] if np.isfinite(fit["slope"]) else np.nan,
        "fit": fit,
        "per_side": per_side,
        "samples": int(len(x)),
        "warnings": warnings,
    }


def roll_gradient_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Fit VectorNav roll angle versus lateral acceleration."""
    circles = classify_circles(df)
    required = ["laps", "Filtering_VN_ay", "VN_vx", *_ROLL_QUAT_COLS]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)
    roll_deg = _roll_from_quaternion_deg(arr["VN_ox"], arr["VN_oy"], arr["VN_oz"], arr["VN_ow"])
    ay_g = arr["Filtering_VN_ay"] / G_MPS2
    sustained = _sustained_balance_mask(arr["VN_vx"], arr["Filtering_VN_ay"]) & _lap_mask(arr["laps"], circles)
    good = sustained & np.isfinite(roll_deg) & np.isfinite(ay_g)
    x = ay_g[good]
    y = roll_deg[good]
    fit = _linear_fit(x, y)
    theory_deg_per_g = np.rad2deg(MASS_KG * G_MPS2 * COG_HEIGHT_M / (KROLL_F_NM_RAD + KROLL_R_NM_RAD))
    measured_abs = abs(fit["slope"]) if np.isfinite(fit["slope"]) else np.nan
    deviation_pct = (
        100.0 * (measured_abs - theory_deg_per_g) / theory_deg_per_g
        if np.isfinite(measured_abs) and theory_deg_per_g > 0.0 else np.nan
    )

    fig = make_dark_figure("Roll gradient from VectorNav quaternion", "Lateral acceleration ay [g]", "Roll angle [deg]")
    for side in ("R", "L"):
        side_laps = _circle_laps_for_side(circles, side)
        mask = good & np.isin(arr["laps"], np.asarray(side_laps, dtype=float))
        if not mask.any():
            continue
        fig.add_trace(
            go.Scattergl(
                x=ay_g[mask],
                y=roll_deg[mask],
                mode="markers",
                name=f"{side} samples",
                marker=dict(color=_SIDE_COLORS[side], size=4, opacity=0.45),
                hovertemplate="ay=%{x:+.2f} g<br>roll=%{y:+.2f} deg<extra></extra>",
            )
        )
    if np.isfinite(fit["slope"]):
        _add_fit_line(fig, x, fit, _REFERENCE, "Measured fit", dash="dash")
    fig.update_layout(height=520)
    warnings = _classification_warnings(circles)
    if not np.isfinite(fit["slope"]):
        warnings.append("Not enough sustained skidpad samples for roll-gradient fit.")
    elif np.isfinite(deviation_pct) and abs(deviation_pct) > 20.0:
        warnings.append(
            "Measured roll gradient deviates by more than 20% from the CAT17x roll-stiffness model; "
            "check quaternion convention, setup changes, or roll-stiffness assumptions."
        )
    return fig, {
        "K_roll_deg_per_g": measured_abs,
        "K_roll_signed_deg_per_g": fit["slope"],
        "K_roll_theory_deg_per_g": theory_deg_per_g,
        "K_roll_delta_pct": deviation_pct,
        "R2_fit": fit["r2"],
        "fit": fit,
        "samples": int(len(x)),
        "warnings": warnings,
    }


def driver_smoothness_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Compute steering, pedal and speed smoothness metrics by circle."""
    circles = classify_circles(df)
    throttle_col = _first_existing_col(df, _THROTTLE_ALIASES)
    brake_col = _first_existing_col(df, _BRAKE_ALIASES)
    required = ["TimeStamp", "laps", "Steering", "VN_vx"]
    optional = [col for col in (throttle_col, brake_col) if col is not None]
    _require_columns(df, [*required, *optional])
    arr = cols_to_numpy(df, [*required, *optional])

    rows: list[dict[str, object]] = []
    labels: list[str] = []
    steer_rms: list[float] = []
    throttle_var: list[float] = []
    brake_var: list[float] = []
    vx_var: list[float] = []
    for lap_id, info in circles.items():
        mask = arr["laps"] == float(lap_id)
        if int(mask.sum()) < 3:
            continue
        t = arr["TimeStamp"][mask]
        steering_deg = np.rad2deg(arr["Steering"][mask] / STEERING_RATIO)
        good_time = np.isfinite(t) & np.isfinite(steering_deg)
        if int(good_time.sum()) < 3:
            continue
        steer_rate = np.gradient(steering_deg[good_time], t[good_time], edge_order=1)
        steer_rate_rms = float(np.sqrt(np.nanmean(steer_rate ** 2)))
        thr_var = _safe_var(arr[throttle_col][mask]) if throttle_col is not None else np.nan
        brk_var = _safe_var(arr[brake_col][mask]) if brake_col is not None else np.nan
        speed_var = _safe_var(arr["VN_vx"][mask])
        label = _circle_label(lap_id, info)
        labels.append(label)
        steer_rms.append(steer_rate_rms)
        throttle_var.append(thr_var)
        brake_var.append(brk_var)
        vx_var.append(speed_var)
        rows.append({
            "Lap": int(lap_id),
            "Side": info["side"],
            "steer_rate_rms_deg_s": _round(steer_rate_rms, 2),
            "throttle_var": _round(thr_var, 3),
            "brake_var": _round(brk_var, 3),
            "vx_var_m2_s2": _round(speed_var, 3),
            "Throttle source": throttle_col or "missing",
            "Brake source": brake_col or "missing",
        })

    fig = make_subplots(
        rows=2,
        cols=2,
        vertical_spacing=0.16,
        horizontal_spacing=0.11,
        subplot_titles=("Steering-rate RMS", "Throttle variance", "Brake variance", "Speed variance"),
    )
    fig.add_trace(go.Bar(x=labels, y=steer_rms, marker_color="#4DB3F2", name="Steer RMS"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=throttle_var, marker_color="#73D973", name="Throttle var"), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=brake_var, marker_color="#F28C40", name="Brake var"), row=2, col=1)
    fig.add_trace(go.Bar(x=labels, y=vx_var, marker_color="#D973D9", name="vx var"), row=2, col=2)
    _apply_dark_layout(fig, "Driver smoothness in skidpad", height=720, showlegend=False)
    fig.update_yaxes(title_text="deg/s", row=1, col=1)
    fig.update_yaxes(title_text="signal^2", row=1, col=2)
    fig.update_yaxes(title_text="signal^2", row=2, col=1)
    fig.update_yaxes(title_text="(m/s)^2", row=2, col=2)

    return fig, {
        "table": pl.DataFrame(rows),
        "per_circle": rows,
        "throttle_source": throttle_col,
        "brake_source": brake_col,
        "warnings": _classification_warnings(circles),
    }


def tv_intervention_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """TV yaw moment versus lateral grip (OptimumG-style control cross-plot)."""
    circles = classify_circles(df)
    required = ["laps", "Filtering_VN_ay", "VN_vx", *_TV_TORQUE_COLS, "TV_errorYawRate"]
    optional = [col for col in ("TV_desiredYawRate", "TV_actualMz") if col in df.columns]
    _require_columns(df, [*required, *optional])
    arr = cols_to_numpy(df, [*required, *optional])
    mz_applied = _tv_yaw_moment_from_torque(arr)
    yaw_err = arr["TV_errorYawRate"]
    ay_g = arr["Filtering_VN_ay"] / G_MPS2
    sustained = _sustained_balance_mask(arr["VN_vx"], arr["Filtering_VN_ay"])

    fig = make_dark_figure(
        "TV intervention versus lateral grip",
        "|ay| [g]",
        "|Mz| [Nm]",
    )

    rows: list[dict[str, object]] = []
    side_mz: dict[str, list[float]] = {"R": [], "L": []}
    any_data = False
    for idx, (lap_id, info) in enumerate(circles.items()):
        mask = (arr["laps"] == float(lap_id)) & sustained
        mz_lap = mz_applied[mask]
        err_lap = yaw_err[mask]
        ay_lap = np.abs(ay_g[mask])
        good_mz = np.isfinite(mz_lap) & np.isfinite(ay_lap)
        good_err = np.isfinite(err_lap)
        if good_mz.any():
            any_data = True
            color = _RUN_COLORS[idx % len(_RUN_COLORS)]
            fig.add_trace(
                go.Scattergl(
                    x=ay_lap[good_mz],
                    y=np.abs(mz_lap[good_mz]),
                    mode="markers",
                    name=_circle_label(lap_id, info),
                    marker=dict(color=color, size=4, opacity=0.55),
                    hovertemplate="|ay|=%{x:.2f} g<br>|Mz|=%{y:.1f} Nm<extra></extra>",
                )
            )
            side_mz[info["side"]].append(_safe_mean(mz_lap[good_mz]))
        rows.append({
            "Lap": int(lap_id),
            "Side": info["side"],
            "Mz_mean_Nm": _round(_safe_mean(mz_lap[good_mz]), 2),
            "Mz_abs_mean_Nm": _round(_safe_mean(np.abs(mz_lap[good_mz])), 2),
            "yaw_err_rms_rad_s": _round(_rms(err_lap[good_err]), 4),
            "Samples": int(good_mz.sum()),
        })

    fig.update_layout(height=480)
    warnings = _classification_warnings(circles)
    if not any_data:
        warnings.append("No sustained-cornering TV samples to plot.")
    all_good = np.isfinite(mz_applied) & sustained
    yaw_good = np.isfinite(yaw_err) & sustained
    return fig, {
        "Mz_mean_Nm": _safe_mean(mz_applied[all_good]),
        "yaw_err_rms_rad_s": _rms(yaw_err[yaw_good]),
        "Mz_R_mean_Nm": _safe_mean(np.asarray(side_mz["R"], dtype=float)),
        "Mz_L_mean_Nm": _safe_mean(np.asarray(side_mz["L"], dtype=float)),
        "table": pl.DataFrame(rows),
        "per_circle": rows,
        "warnings": warnings,
    }


def lateral_load_dist_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Compute measured lateral load-transfer distribution in skidpad."""
    circles = classify_circles(df)
    required = ["laps", "Filtering_VN_ay", *_FZ_COLS]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)

    labels: list[str] = []
    lltd_values: list[float] = []
    rows: list[dict[str, object]] = []
    all_lltd: list[np.ndarray] = []
    for lap_id, info in circles.items():
        mask = arr["laps"] == float(lap_id)
        front_transfer, rear_transfer = _axle_load_transfer(info["side"], arr, mask)
        denom = front_transfer + rear_transfer
        good = (
            np.isfinite(front_transfer)
            & np.isfinite(rear_transfer)
            & np.isfinite(denom)
            & (denom > 50.0)
        )
        if not good.any():
            continue
        lltd = 100.0 * front_transfer[good] / denom[good]
        lltd = lltd[np.isfinite(lltd) & (lltd >= 0.0) & (lltd <= 100.0)]
        if lltd.size == 0:
            continue
        all_lltd.append(lltd)
        mean_lltd = _safe_mean(lltd)
        labels.append(_circle_label(lap_id, info))
        lltd_values.append(mean_lltd)
        rows.append({
            "Lap": int(lap_id),
            "Side": info["side"],
            "LLTD_meas_pct": _round(mean_lltd, 2),
            "LLTD_theory_pct": _round(100.0 * LLTD_THEORY, 2),
            "delta_pct": _round(mean_lltd - 100.0 * LLTD_THEORY, 2),
            "Samples": int(lltd.size),
        })

    fig = make_dark_figure("Lateral load-transfer distribution", "Circle", "Front LLTD [%]")
    fig.add_trace(
        go.Bar(
            x=labels,
            y=lltd_values,
            marker=dict(color="#4DB3F2", line=dict(color="#22252B", width=1)),
            text=[_fmt_num(v, ".1f") for v in lltd_values],
            textposition="outside",
            hovertemplate="%{x}<br>LLTD=%{y:.1f}%<extra></extra>",
            name="Measured",
        )
    )
    fig.add_hline(
        y=100.0 * LLTD_THEORY,
        line=dict(color=_REFERENCE, dash="dash", width=1.4),
        annotation_text=f"Theory {100.0 * LLTD_THEORY:.1f}%",
        annotation_font_color=_TEXT,
    )
    fig.update_yaxes(range=[0.0, 100.0])
    fig.update_layout(height=470)
    lltd_all = np.concatenate(all_lltd) if all_lltd else np.array([], dtype=float)
    lltd_mean = _safe_mean(lltd_all)
    return fig, {
        "LLTD_meas_pct": lltd_mean,
        "LLTD_theory_pct": 100.0 * LLTD_THEORY,
        "delta_pct": lltd_mean - 100.0 * LLTD_THEORY if np.isfinite(lltd_mean) else np.nan,
        "table": pl.DataFrame(rows),
        "per_circle": rows,
        "warnings": _classification_warnings(circles),
    }


def lr_asymmetry_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Aggregate right-minus-left skidpad asymmetry metrics."""
    circles = classify_circles(df)
    warnings = _classification_warnings(circles)
    rows: list[dict[str, object]] = []

    def add_metric(name: str, right: float, left: float, unit: str, decimals: int = 3) -> None:
        delta = right - left if np.isfinite(right) and np.isfinite(left) else np.nan
        rows.append({
            "Metric": name,
            "R": _round(right, decimals),
            "L": _round(left, decimals),
            "R-L": _round(delta, decimals),
            "Unit": unit,
        })

    timed = _official_timed_times(circles)
    add_metric("laptime_timed", timed.get("R", np.nan), timed.get("L", np.nan), "s", 3)

    side_laps = {side: _side_eval_laps(circles, side) for side in ("R", "L")}
    arr_core = _optional_arrays(df, ["laps", "Filtering_VN_ay", "VN_vx", "Steering"])
    if arr_core is not None:
        ay_side = {
            side: _side_mean_abs(arr_core["Filtering_VN_ay"] / G_MPS2, arr_core["laps"], laps)
            for side, laps in side_laps.items()
        }
        steer_side = {
            side: _side_mean_abs(np.rad2deg(arr_core["Steering"] / STEERING_RATIO), arr_core["laps"], laps)
            for side, laps in side_laps.items()
        }
        radius = _driven_radius_m(
            np.arange(len(arr_core["VN_vx"]), dtype=float),
            arr_core["VN_vx"],
            arr_core["Filtering_VN_ay"],
        )
        radius_side = {
            side: _side_mean_abs(radius, arr_core["laps"], laps, absolute=False)
            for side, laps in side_laps.items()
        }
        add_metric("ay_mean", ay_side["R"], ay_side["L"], "g", 3)
        add_metric("steering_mean", steer_side["R"], steer_side["L"], "deg", 3)
        add_metric("driven_radius", radius_side["R"], radius_side["L"], "m", 3)

    try:
        us_payload = _understeer_fit_payload(df, circles, side_laps=side_laps)
        us_side: dict[str, float] = {}
        for side in ("R", "L"):
            mask = us_payload["side"] == side
            fit = _linear_fit(us_payload["ay_g"][mask], us_payload["delta_dyn_deg"][mask])
            us_side[side] = fit["slope"]
        add_metric("K_us", us_side.get("R", np.nan), us_side.get("L", np.nan), "deg/g", 3)
    except Exception as exc:
        warnings.append(f"Understeer-gradient asymmetry unavailable: {exc}")

    if all(col in df.columns for col in _TV_TORQUE_COLS) and "Filtering_VN_ay" in df.columns:
        arr_tv = cols_to_numpy(df, ["laps", "Filtering_VN_ay", *_TV_TORQUE_COLS])
        mz = np.abs(_tv_yaw_moment_from_torque(arr_tv))
        mz_side = {
            side: _side_mean_abs(mz, arr_tv["laps"], laps, absolute=False)
            for side, laps in side_laps.items()
        }
        add_metric("TV_Mz_abs_mean", mz_side["R"], mz_side["L"], "Nm", 2)

    if has_load_signals(df):
        arr_fz = cols_to_numpy(df, ["laps", "Filtering_VN_ay", *_FZ_COLS])
        lltd_side = {}
        for side, laps in side_laps.items():
            side_mask = np.isin(arr_fz["laps"], np.asarray(laps, dtype=float))
            front, rear = _axle_load_transfer(side, arr_fz, side_mask)
            denom = front + rear
            good = np.isfinite(front) & np.isfinite(rear) & (denom > 50.0)
            lltd = 100.0 * front[good] / denom[good] if good.any() else np.array([], dtype=float)
            lltd_side[side] = _safe_mean(lltd[(lltd >= 0.0) & (lltd <= 100.0)])
        add_metric("LLTD", lltd_side.get("R", np.nan), lltd_side.get("L", np.nan), "pct", 2)

    plot_rows = [row for row in rows if np.isfinite(float(row["R-L"]))]
    fig = make_dark_figure("Right-left skidpad asymmetry", "R - L difference", "Metric")
    fig.add_vline(x=0.0, line=dict(color="rgba(235,235,235,0.45)", dash="dash", width=1))
    if plot_rows:
        deltas = [float(row["R-L"]) for row in plot_rows]
        labels = [f"{row['Metric']} [{row['Unit']}]" for row in plot_rows]
        fig.add_trace(
            go.Bar(
                x=deltas,
                y=labels,
                orientation="h",
                marker_color=["#73D973" if abs(v) < 1e-9 else "#F28C40" for v in deltas],
                text=[_fmt_num(v, "+.3f") for v in deltas],
                textposition="outside",
                hovertemplate="%{y}<br>R-L=%{x:+.3f}<extra></extra>",
                name="R-L",
            )
        )
    fig.update_layout(height=max(430, 70 * max(1, len(plot_rows))), showlegend=False)
    return fig, {
        "table": pl.DataFrame(rows),
        "metrics": rows,
        "warnings": warnings,
    }


def gps_figure8_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Plot local GPS trajectory with theoretical skidpad circles."""
    circles = classify_circles(df)
    required = ["laps", "VN_latitude", "VN_longitude", "Filtering_VN_ay"]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)
    lap_mask = _lap_mask(arr["laps"], circles)
    gps_valid = (
        lap_mask
        & np.isfinite(arr["VN_latitude"])
        & np.isfinite(arr["VN_longitude"])
        & ((np.abs(arr["VN_latitude"]) > 1e-9) | (np.abs(arr["VN_longitude"]) > 1e-9))
    )
    if int(gps_valid.sum()) < 3:
        raise ValueError("Not enough valid GPS samples for skidpad map.")

    x_valid, y_valid = gps_to_local_xy(arr["VN_latitude"][gps_valid], arr["VN_longitude"][gps_valid])
    x = np.full(len(df), np.nan, dtype=float)
    y = np.full(len(df), np.nan, dtype=float)
    x[gps_valid] = x_valid
    y[gps_valid] = y_valid

    fig = make_dark_figure("Skidpad GPS figure-8", "Local X [m]", "Local Y [m]")
    fig.add_trace(
        go.Scattergl(
            x=x[gps_valid],
            y=y[gps_valid],
            mode="markers",
            name="Trajectory",
            marker=dict(
                color=np.abs(arr["Filtering_VN_ay"][gps_valid]) / G_MPS2,
                colorscale="Turbo",
                size=4,
                opacity=0.82,
                colorbar=dict(title="|ay| [g]"),
            ),
            hovertemplate="x=%{x:.1f} m<br>y=%{y:.1f} m<br>|ay|=%{marker.color:.2f} g<extra></extra>",
        )
    )

    centers = _skidpad_centers_from_laps(circles, arr["laps"], x, y)
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    for side, center in centers.items():
        cx, cy = center
        for radius, dash, name in (
            (SKIDPAD_INNER_RADIUS_M, "dash", "inner"),
            (SKIDPAD_OUTER_RADIUS_M, "dash", "outer"),
            (SKIDPAD_IDEAL_RADIUS_M, "dot", "ideal"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=cx + radius * np.cos(theta),
                    y=cy + radius * np.sin(theta),
                    mode="lines",
                    name=f"{side} {name}",
                    line=dict(color=_REFERENCE, dash=dash, width=1.0),
                    opacity=0.78,
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers",
                name=f"{side} fitted center",
                marker=dict(color=_SIDE_COLORS[side], size=9, symbol="x"),
                hovertemplate=f"{side} center<br>x=%{{x:.1f}} m<br>y=%{{y:.1f}} m<extra></extra>",
            )
        )

    fig.update_yaxes(scaleanchor="x", scaleratio=1.0)
    fig.update_layout(height=650)
    warnings = _classification_warnings(circles)
    if len(centers) < 2:
        warnings.append("Only one skidpad circle center could be estimated from GPS.")
    return fig, {
        "gps_samples": int(gps_valid.sum()),
        "centers": {side: (float(center[0]), float(center[1])) for side, center in centers.items()},
        "warnings": warnings,
    }


def event_time_bars_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Backward-compatible alias for older dashboard cache keys."""
    return event_time_summary_fig(df)


def lateral_g_histogram_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Backward-compatible alias; now returns the lateral-G time figure."""
    return lateral_g_fig(df)


def driven_radius_histogram_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Backward-compatible alias; now returns the driven-radius time figure."""
    return driven_radius_fig(df)


def understeer_chart_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Backward-compatible alias for understeer-gradient diagnostics."""
    return understeer_gradient_fig(df)


def slip_angle_vs_ay_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Backward-compatible alias for the balance summary."""
    return balance_fig(df)


def yaw_rate_vs_ay_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Compatibility diagnostic for yaw rate versus lateral acceleration."""
    circles = classify_circles(df)
    yaw_col = _first_existing_col(df, _YAW_ALIASES)
    if yaw_col is None:
        raise KeyError("Missing yaw-rate column: VN_gz/AS_yaw_rate")
    required = ["laps", "Filtering_VN_ay", "VN_vx", yaw_col]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)
    sustained = _sustained_balance_mask(arr["VN_vx"], arr["Filtering_VN_ay"]) & _lap_mask(arr["laps"], circles)
    fig = make_dark_figure("Yaw rate versus lateral acceleration", "ay [g]", "Yaw rate [rad/s]")
    fig.add_hline(y=0.0, line=dict(color="rgba(235,235,235,0.35)", dash="dash", width=1))
    fig.add_vline(x=0.0, line=dict(color="rgba(235,235,235,0.35)", dash="dash", width=1))
    for side in ("R", "L"):
        mask = sustained & np.isin(arr["laps"], np.asarray(_circle_laps_for_side(circles, side), dtype=float))
        if not mask.any():
            continue
        fig.add_trace(
            go.Scattergl(
                x=arr["Filtering_VN_ay"][mask] / G_MPS2,
                y=arr[yaw_col][mask],
                mode="markers",
                name=f"{side} samples",
                marker=dict(color=_SIDE_COLORS[side], size=4, opacity=0.45),
                hovertemplate="ay=%{x:+.2f} g<br>yaw=%{y:+.3f} rad/s<extra></extra>",
            )
        )
    return fig, {"warnings": _classification_warnings(circles), "yaw_source": yaw_col}


def lltd_scatter_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Backward-compatible alias for the LLTD bar chart."""
    return lateral_load_dist_fig(df)


def _require_columns(df: pl.DataFrame, columns: list[str] | tuple[str, ...]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing skidpad columns: {missing}")


def _optional_arrays(df: pl.DataFrame, columns: list[str]) -> dict[str, np.ndarray] | None:
    if any(col not in df.columns for col in columns):
        return None
    return cols_to_numpy(df, columns)


def _lap_time_s(arr: dict[str, np.ndarray], mask: np.ndarray) -> float:
    if "laptime" in arr:
        laptime = arr["laptime"][mask]
        finite_laptime = laptime[np.isfinite(laptime) & (laptime > 0.0)]
        if finite_laptime.size:
            return float(np.nanmax(finite_laptime))
    time_lap = arr["TimeStamp"][mask]
    finite_time = time_lap[np.isfinite(time_lap)]
    if finite_time.size >= 2:
        return float(np.nanmax(finite_time) - np.nanmin(finite_time))
    return np.nan


def _circle_summary_rows(circles: dict[int, dict[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for lap_id, info in circles.items():
        rows.append({
            "Lap": int(lap_id),
            "Side": str(info["side"]),
            "Official": "yes" if info.get("official_timed", False) else "",
            "Laptime [s]": _round(float(info["laptime_s"]), 3),
            "Samples": int(info["n_samples"]),
        })
    return rows


def _add_table_trace(fig: go.Figure, rows: list[dict[str, object]], *, row: int, col: int) -> None:
    if rows:
        headers = list(rows[0].keys())
        values = [[entry.get(header, "") for entry in rows] for header in headers]
    else:
        headers = ["Lap", "Side", "Official", "Laptime [s]", "Samples"]
        values = [[""] for _ in headers]
    fig.add_trace(
        go.Table(
            header=dict(values=headers, fill_color="#22252B", font=dict(color=_TEXT, size=11), align="left"),
            cells=dict(values=values, fill_color="#171A1F", font=dict(color=_TEXT, size=10), align="left"),
        ),
        row=row,
        col=col,
    )


def _apply_dark_layout(fig: go.Figure, title: str, *, height: int, showlegend: bool = True) -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=_TEXT)),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT, size=11),
        legend=dict(
            bgcolor="rgba(20,20,23,0.85)",
            bordercolor="rgba(128,128,128,0.3)",
            font=dict(color=_TEXT),
        ),
        height=height,
        showlegend=showlegend,
    )
    fig.update_xaxes(color=_AXIS, gridcolor=_GRID, linecolor=_AXIS, tickcolor=_AXIS, showgrid=True)
    fig.update_yaxes(color=_AXIS, gridcolor=_GRID, linecolor=_AXIS, tickcolor=_AXIS, showgrid=True)


def _classification_warnings(circles: dict[int, dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    n_total = len(circles)
    n_r = sum(1 for info in circles.values() if info["side"] == "R")
    n_l = sum(1 for info in circles.values() if info["side"] == "L")
    if n_r < 1 or n_l < 1:
        warnings.append(
            f"Expected at least 1 timed circle per side, got {n_total} ({n_r} R, {n_l} L)."
        )
    if not any(info.get("official_timed", False) and info["side"] == "R" for info in circles.values()):
        warnings.append("No official timed right-hand circle could be identified.")
    if not any(info.get("official_timed", False) and info["side"] == "L" for info in circles.values()):
        warnings.append("No official timed left-hand circle could be identified.")
    return warnings


def _official_timed_times(circles: dict[int, dict[str, Any]]) -> dict[str, float]:
    out = {"R": np.nan, "L": np.nan}
    for side in ("R", "L"):
        candidates = [
            float(info["laptime_s"])
            for info in circles.values()
            if info["side"] == side and info["role"] == "timed" and np.isfinite(info["laptime_s"])
        ]
        if candidates:
            out[side] = float(np.nanmin(candidates))
    return out


def _side_eval_laps(circles: dict[int, dict[str, Any]], side: str) -> list[int]:
    official = [
        lap_id for lap_id, info in circles.items()
        if info["side"] == side and info.get("official_timed", False)
    ]
    if official:
        return official
    timed = [lap_id for lap_id, info in circles.items() if info["side"] == side and info["role"] == "timed"]
    if timed:
        return timed
    return [lap_id for lap_id, info in circles.items() if info["side"] == side]


def _circle_laps_for_side(circles: dict[int, dict[str, Any]], side: str) -> list[int]:
    return [lap_id for lap_id, info in circles.items() if info["side"] == side]


def _lap_mask(laps: np.ndarray, circles: dict[int, dict[str, Any]]) -> np.ndarray:
    if not circles:
        return np.zeros_like(laps, dtype=bool)
    return np.isin(laps, np.asarray(list(circles), dtype=float))


def _circle_label(lap_id: int, info: dict[str, Any]) -> str:
    suffix = " official" if info.get("official_timed", False) else ""
    return f"L{int(lap_id)} {info['side']}{suffix}"


def _opposite_side(side: str) -> str:
    return "R" if side == "L" else "L"


def _turn_sign_from_side(side: str) -> float:
    return 1.0 if side == AY_POSITIVE_SIDE else -1.0


def _driven_radius_m(time_s: np.ndarray, vx_mps: np.ndarray, ay_mps2: np.ndarray) -> np.ndarray:
    vx = np.asarray(vx_mps, dtype=float)
    ay_abs = np.abs(np.asarray(ay_mps2, dtype=float))
    radius = np.divide(
        vx ** 2,
        ay_abs,
        out=np.full_like(vx, np.nan, dtype=float),
        where=np.isfinite(vx) & np.isfinite(ay_abs) & (ay_abs > 0.2),
    )
    if len(radius) >= 5:
        try:
            dt_s = robust_dt(np.asarray(time_s, dtype=float))
            radius = smooth_signal(radius, max(1, int(round(0.15 / dt_s))))
        except Exception:
            pass
    return radius


def _sustained_radius_mask(vx_mps: np.ndarray, ay_mps2: np.ndarray, radius_m: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(vx_mps)
        & np.isfinite(ay_mps2)
        & np.isfinite(radius_m)
        & (np.abs(vx_mps) >= MIN_SKIDPAD_SPEED_MPS)
        & (np.abs(ay_mps2) >= MIN_SKIDPAD_AY_MPS2)
        & (radius_m > 2.0)
        & (radius_m < 60.0)
    )


def _sustained_balance_mask(vx_mps: np.ndarray, ay_mps2: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(vx_mps)
        & np.isfinite(ay_mps2)
        & (np.abs(vx_mps) >= MIN_BALANCE_SPEED_MPS)
        & (np.abs(ay_mps2) >= MIN_BALANCE_AY_MPS2)
    )


def _front_rear_sa_deg(arr: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    front_sa_deg = 0.5 * (
        np.rad2deg(np.abs(arr["Est_SAFL"])) + np.rad2deg(np.abs(arr["Est_SAFR"]))
    )
    rear_sa_deg = 0.5 * (
        np.rad2deg(np.abs(arr["Est_SARL"])) + np.rad2deg(np.abs(arr["Est_SARR"]))
    )
    return front_sa_deg, rear_sa_deg


def _understeer_angle_deg(steering: np.ndarray, yaw_rate: np.ndarray, vx_mps: np.ndarray) -> np.ndarray:
    steering_rad = np.asarray(steering, dtype=float) / STEERING_RATIO
    yaw = np.asarray(yaw_rate, dtype=float)
    vx = np.asarray(vx_mps, dtype=float)
    ackermann_rad = np.divide(
        WHEELBASE_EQ * yaw,
        vx,
        out=np.full_like(vx, np.nan, dtype=float),
        where=np.isfinite(yaw) & np.isfinite(vx) & (np.abs(vx) > 0.5),
    )
    return np.rad2deg(steering_rad - ackermann_rad)


def _understeer_fit_payload(
    df: pl.DataFrame,
    circles: dict[int, dict[str, Any]],
    *,
    side_laps: dict[str, list[int]] | None = None,
) -> dict[str, np.ndarray]:
    yaw_col = _first_existing_col(df, _YAW_ALIASES)
    if yaw_col is None:
        raise KeyError("Missing yaw-rate column: VN_gz/AS_yaw_rate")
    required = ["laps", "Filtering_VN_ay", "VN_vx", "Steering", yaw_col]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)
    delta_dyn_deg = _understeer_angle_deg(arr["Steering"], arr[yaw_col], arr["VN_vx"])
    ay_g = arr["Filtering_VN_ay"] / G_MPS2
    sustained = _sustained_balance_mask(arr["VN_vx"], arr["Filtering_VN_ay"])

    side_arr = np.full(len(arr["laps"]), "", dtype=object)
    source = side_laps if side_laps is not None else {
        side: _circle_laps_for_side(circles, side) for side in ("R", "L")
    }
    for side, laps in source.items():
        side_arr[np.isin(arr["laps"], np.asarray(laps, dtype=float))] = side

    good = sustained & (side_arr != "") & np.isfinite(ay_g) & np.isfinite(delta_dyn_deg)
    return {
        "ay_g": ay_g[good],
        "delta_dyn_deg": delta_dyn_deg[good],
        "side": side_arr[good],
    }


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    good = np.isfinite(x) & np.isfinite(y)
    x_fit = np.asarray(x[good], dtype=float)
    y_fit = np.asarray(y[good], dtype=float)
    if x_fit.size < MIN_FIT_SAMPLES or float(np.nanmax(x_fit) - np.nanmin(x_fit)) <= 1e-6:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": int(x_fit.size)}
    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    pred = slope * x_fit + intercept
    ss_res = float(np.nansum((y_fit - pred) ** 2))
    ss_tot = float(np.nansum((y_fit - np.nanmean(y_fit)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2), "n": int(x_fit.size)}


def _add_fit_line(
    fig: go.Figure,
    x: np.ndarray,
    fit: dict[str, float],
    color: str,
    name: str,
    *,
    dash: str = "solid",
) -> None:
    good = np.isfinite(x)
    if not good.any() or not np.isfinite(fit["slope"]):
        return
    x_fit = np.linspace(float(np.nanmin(x[good])), float(np.nanmax(x[good])), 80)
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=fit["slope"] * x_fit + fit["intercept"],
            mode="lines",
            name=name,
            line=dict(color=color, width=2.0, dash=dash),
            hovertemplate=f"slope={fit['slope']:+.3f}<extra></extra>",
        )
    )


def _roll_from_quaternion_deg(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, qw: np.ndarray) -> np.ndarray:
    roll_rad = np.arctan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx ** 2 + qy ** 2),
    )
    return np.rad2deg(roll_rad)


def _tv_yaw_moment_from_torque(arr: dict[str, np.ndarray]) -> np.ndarray:
    fl = arr["TV_FL_Trq"] * GEAR_RATIO / WHEEL_RADIUS_M
    fr = arr["TV_FR_Trq"] * GEAR_RATIO / WHEEL_RADIUS_M
    rl = arr["TV_RL_Trq"] * GEAR_RATIO / WHEEL_RADIUS_M
    rr = arr["TV_RR_Trq"] * GEAR_RATIO / WHEEL_RADIUS_M
    return (fr - fl) * (TRACK_F_M / 2.0) + (rr - rl) * (TRACK_R_M / 2.0)


def _axle_load_transfer(
    side: str,
    arr: dict[str, np.ndarray],
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if side == "R":
        front_transfer = arr["Est_FZFL"][mask] - arr["Est_FZFR"][mask]
        rear_transfer = arr["Est_FZRL"][mask] - arr["Est_FZRR"][mask]
    else:
        front_transfer = arr["Est_FZFR"][mask] - arr["Est_FZFL"][mask]
        rear_transfer = arr["Est_FZRR"][mask] - arr["Est_FZRL"][mask]
    return front_transfer, rear_transfer


def _skidpad_centers_from_laps(
    circles: dict[int, dict[str, Any]],
    laps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, tuple[float, float]]:
    centers: dict[str, tuple[float, float]] = {}
    for side in ("R", "L"):
        lap_ids = _circle_laps_for_side(circles, side)
        mask = np.isin(laps, np.asarray(lap_ids, dtype=float)) & np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 10:
            continue
        center = _fit_circle_center(x[mask], y[mask])
        if center is None:
            center = (float(np.nanmedian(x[mask])), float(np.nanmedian(y[mask])))
        centers[side] = center

    if len(centers) == 2:
        c_r = np.asarray(centers["R"], dtype=float)
        c_l = np.asarray(centers["L"], dtype=float)
        delta = c_l - c_r
        dist = float(np.hypot(delta[0], delta[1]))
        if np.isfinite(dist) and dist > 1.0:
            unit = delta / dist
            mid = 0.5 * (c_r + c_l)
            centers["R"] = tuple((mid - 0.5 * SKIDPAD_CIRCLE_GAP_M * unit).tolist())
            centers["L"] = tuple((mid + 0.5 * SKIDPAD_CIRCLE_GAP_M * unit).tolist())
    return centers


def _fit_circle_center(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size < 10:
        return None
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x ** 2 + y ** 2
    try:
        solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = float(solution[0]), float(solution[1])
    if not np.isfinite(cx) or not np.isfinite(cy):
        return None
    return cx, cy


def _side_mean_abs(
    values: np.ndarray,
    laps: np.ndarray,
    lap_ids: list[int],
    *,
    absolute: bool = True,
) -> float:
    if not lap_ids:
        return np.nan
    mask = np.isin(laps, np.asarray(lap_ids, dtype=float))
    vals = values[mask]
    vals = np.abs(vals) if absolute else vals
    return _safe_mean(vals)


def _first_existing_col(df: pl.DataFrame, aliases: tuple[str, ...]) -> str | None:
    return next((col for col in aliases if col in df.columns), None)


def _safe_mean(values: np.ndarray | list[float] | tuple[float, ...]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmean(arr)) if arr.size else np.nan


def _safe_std(values: np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanstd(arr)) if arr.size else np.nan


def _safe_var(values: np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanvar(arr)) if arr.size else np.nan


def _safe_max(values: np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmax(arr)) if arr.size else np.nan


def _safe_min(values: np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmin(arr)) if arr.size else np.nan


def _safe_percentile(values: np.ndarray | list[float], percentile: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanpercentile(arr, percentile)) if arr.size else np.nan


def _mean_if_finite(values: tuple[float, ...] | list[float]) -> float:
    return _safe_mean(np.asarray(values, dtype=float))


def _abs_diff(a: float, b: float) -> float:
    return abs(a - b) if np.isfinite(a) and np.isfinite(b) else np.nan


def _rms(values: np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.sqrt(np.nanmean(arr ** 2))) if arr.size else np.nan


def _round(value: float, decimals: int) -> float:
    return round(float(value), decimals) if np.isfinite(value) else np.nan


def _fmt_num(value: float, pattern: str) -> str:
    return format(float(value), pattern) if np.isfinite(value) else ""
