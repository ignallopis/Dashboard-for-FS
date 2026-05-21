"""Skidpad event analysis for Formula Student telemetry.

The dashboard-facing functions in this module are pure: they receive one
Polars DataFrame and return Plotly figures plus KPI dictionaries. Rendering
belongs in ``src/dashboard.py``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.dynamics import G_MPS2, MASS_KG, STEERING_RATIO, WHEELBASE_EQ
from src.lapcount import gps_to_local_xy
from utils import cols_to_numpy, make_dark_figure, robust_dt, smooth_signal, WHEEL_COLORS

SKIDPAD_INNER_RADIUS_M = 7.625
SKIDPAD_OUTER_RADIUS_M = 10.625
SKIDPAD_IDEAL_RADIUS_M = 9.125
SKIDPAD_PATH_WIDTH_M = 3.0
SKIDPAD_CIRCLE_GAP_M = 18.25
SKIDPAD_TIMED_LAPS = (2, 4)

AY_POSITIVE_SIDE = "L"
MIN_SKIDPAD_SPEED_MPS = 2.0
MIN_SKIDPAD_AY_MPS2 = 1.0
MIN_BALANCE_SPEED_MPS = 3.0
MIN_BALANCE_AY_MPS2 = 2.0

_BG = "#141417"
_TEXT = "#EBEBEB"
_GRID = "rgba(128,128,128,0.2)"
_AXIS = "#E5E5E5"
_REFERENCE = "#F2D44D"
_RUN_COLORS = ("#4DB3F2", "#F28C40", "#73D973", "#F27070", "#D973D9", "#F2C94C")

_TV_ACTUAL_MZ_ALIASES = ("TV_actualMz", "TV_actualmz", "tv_actual_mz", "tv_actualmz")
_TV_FF_MZ_ALIASES = ("TV_feedForwardMz", "TV_feedforwardMz", "tv_feedforward_mz")
_TV_FB_MZ_ALIASES = ("TV_feedBackMz", "TV_feedbackMz", "tv_feedback_mz")
_TV_DESIRED_YAW_ALIASES = ("TV_desiredYawRate", "tv_desired_yaw_rate")
_TV_ERROR_YAW_ALIASES = ("TV_errorYawRate", "tv_error_yaw_rate")

_THROTTLE_ALIASES = ("APPS", "pedals_throttle", "Throttle", "TP", "APPS1")
_BRAKE_ALIASES = ("BSE", "Brake", "pedals_brake", "BSEFront")


def is_skidpad_run(df: pl.DataFrame) -> bool:
    """Return True when most samples are tagged as skidpad by lapcount."""
    if "lapcount_mode" not in df.columns or len(df) == 0:
        return False
    modes = df["lapcount_mode"].to_list()
    n_skidpad = sum(str(value).lower() == "skidpad" for value in modes)
    return n_skidpad > 0.5 * len(modes)


def classify_circles(df: pl.DataFrame) -> dict[int, dict[str, Any]]:
    """Classify each skidpad lap as side R/L and warmup/timed."""
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
        if int(mask.sum()) == 0:
            continue
        time_lap = arr["TimeStamp"][mask]
        ay_lap = arr["Filtering_VN_ay"][mask]
        mean_ay = _safe_mean(ay_lap)
        side = AY_POSITIVE_SIDE if mean_ay >= 0.0 else _opposite_side(AY_POSITIVE_SIDE)

        laptime_s = np.nan
        if "laptime" in arr:
            lap_laptime = arr["laptime"][mask]
            finite_laptime = lap_laptime[np.isfinite(lap_laptime) & (lap_laptime > 0.0)]
            if finite_laptime.size:
                laptime_s = float(np.nanmax(finite_laptime))
        if not np.isfinite(laptime_s):
            finite_time = time_lap[np.isfinite(time_lap)]
            if finite_time.size >= 2:
                laptime_s = float(np.nanmax(finite_time) - np.nanmin(finite_time))

        rows[lap_id] = {
            "side": side,
            "role": "warmup",
            "laptime_s": laptime_s,
            "n_samples": int(mask.sum()),
            "mean_ay_mps2": mean_ay,
            "first_time_s": float(np.nanmin(time_lap)) if np.isfinite(time_lap).any() else np.nan,
            "official_timed": False,
        }

    for side in ("R", "L"):
        side_laps = sorted(
            (lap for lap, info in rows.items() if info["side"] == side),
            key=lambda lap: rows[lap]["first_time_s"],
        )
        for idx, lap_id in enumerate(side_laps):
            rows[lap_id]["role"] = "timed" if idx >= 1 else "warmup"

        timed = [
            lap for lap in side_laps
            if rows[lap]["role"] == "timed" and np.isfinite(rows[lap]["laptime_s"])
        ]
        if timed:
            best_lap = min(timed, key=lambda lap: rows[lap]["laptime_s"])
            rows[best_lap]["official_timed"] = True

    return dict(sorted(rows.items(), key=lambda item: item[1]["first_time_s"]))


def event_time_summary_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Return event-time summary table and circle laptime bars."""
    circles = classify_circles(df)
    rows = _circle_rows(circles)
    official = _official_timed_times(circles)
    event_time_s = _mean_if_finite([official.get("R", np.nan), official.get("L", np.nan)])
    lr_asymmetry_s = (
        abs(official["L"] - official["R"])
        if np.isfinite(official.get("L", np.nan)) and np.isfinite(official.get("R", np.nan))
        else np.nan
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "table"}], [{"type": "xy"}]],
        row_heights=[0.42, 0.58],
        vertical_spacing=0.10,
    )
    fig.add_trace(_circle_table(rows), row=1, col=1)

    x = [row["Circle"] for row in rows]
    y = [row["Lap time [s]"] for row in rows]
    colors = ["#73D973" if row["Official timed"] else "#6C7A89" for row in rows]
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            name="Lap time",
            marker=dict(color=colors),
            hovertemplate="%{x}<br>Lap time=%{y:.3f} s<extra></extra>",
        ),
        row=2,
        col=1,
    )
    _apply_dark_layout(fig, "Skidpad event time summary", height=560)
    fig.update_xaxes(title_text="Circle", row=2, col=1)
    fig.update_yaxes(title_text="Lap time [s]", row=2, col=1)

    kpis = {
        "event_time_s": event_time_s,
        "timed_R_s": official.get("R", np.nan),
        "timed_L_s": official.get("L", np.nan),
        "lr_asymmetry_s": lr_asymmetry_s,
        "rows": rows,
        "classification": circles,
        "warnings": _classification_warnings(circles),
    }
    return fig, kpis


def lateral_g_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Plot lateral acceleration by circle and return sustained-G KPIs."""
    circles = classify_circles(df)
    required = ["TimeStamp", "laps", "Filtering_VN_ay"]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)

    fig = make_dark_figure("Skidpad lateral acceleration", "Circle time [s]", "ay [g]")
    rows: list[dict[str, Any]] = []
    all_abs_g: list[np.ndarray] = []

    for idx, (lap_id, info) in enumerate(circles.items()):
        mask = arr["laps"] == float(lap_id)
        t = _relative_time(arr["TimeStamp"][mask])
        ay_g = arr["Filtering_VN_ay"][mask] / G_MPS2
        finite = np.isfinite(t) & np.isfinite(ay_g)
        if not finite.any():
            continue
        color = _RUN_COLORS[idx % len(_RUN_COLORS)]
        fig.add_trace(
            go.Scattergl(
                x=t[finite],
                y=ay_g[finite],
                mode="lines",
                name=_circle_name(lap_id, info),
                line=dict(color=color, width=1.4),
                hovertemplate="t=%{x:.2f} s<br>ay=%{y:.3f} g<extra></extra>",
            )
        )
        abs_g = np.abs(ay_g[finite])
        all_abs_g.append(abs_g)
        rows.append({
            "Lap": lap_id,
            "Side": info["side"],
            "Role": info["role"],
            "Official timed": bool(info.get("official_timed", False)),
            "ay mean [g]": float(np.nanmean(abs_g)),
            "ay max [g]": float(np.nanmax(abs_g)),
            "ay p95 [g]": _safe_percentile(abs_g, 95.0),
            "ay std [g]": float(np.nanstd(ay_g[finite])),
            "Samples": int(finite.sum()),
        })

    fig.add_hline(y=0.0, line=dict(color="rgba(200,200,200,0.45)", dash="dash", width=1))
    global_abs = np.concatenate(all_abs_g) if all_abs_g else np.array([], dtype=float)
    kpis = {
        "rows": rows,
        "ay_max_g": float(np.nanmax(global_abs)) if global_abs.size else np.nan,
        "ay_p95_g": _safe_percentile(global_abs, 95.0),
        "warnings": _classification_warnings(circles),
    }
    return fig, kpis


def driven_radius_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Plot driven radius from R = vx^2 / |ay| against FS skidpad bounds."""
    circles = classify_circles(df)
    required = ["TimeStamp", "laps", "Filtering_VN_ay", "VN_vx"]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)
    radius_m = _driven_radius_m(arr["VN_vx"], arr["Filtering_VN_ay"])
    valid = _sustained_radius_mask(arr["VN_vx"], arr["Filtering_VN_ay"], radius_m)

    fig = make_dark_figure("Driven radius vs skidpad target", "Circle time [s]", "Radius [m]")
    fig.add_hrect(
        y0=SKIDPAD_INNER_RADIUS_M,
        y1=SKIDPAD_OUTER_RADIUS_M,
        fillcolor="rgba(115,217,115,0.12)",
        line_width=0,
        layer="below",
    )
    fig.add_hline(
        y=SKIDPAD_IDEAL_RADIUS_M,
        line=dict(color=_REFERENCE, dash="dash", width=1.5),
        annotation_text="Ideal 9.125 m",
        annotation_font_color=_TEXT,
    )

    rows: list[dict[str, Any]] = []
    timed_errors: list[float] = []
    for idx, (lap_id, info) in enumerate(circles.items()):
        mask = (arr["laps"] == float(lap_id)) & valid
        t = _relative_time(arr["TimeStamp"][mask])
        r = radius_m[mask]
        finite = np.isfinite(t) & np.isfinite(r)
        if not finite.any():
            continue
        color = _RUN_COLORS[idx % len(_RUN_COLORS)]
        fig.add_trace(
            go.Scattergl(
                x=t[finite],
                y=np.where(r[finite] <= 30.0, r[finite], np.nan),
                mode="lines",
                name=_circle_name(lap_id, info),
                line=dict(color=color, width=1.4),
                hovertemplate="t=%{x:.2f} s<br>R=%{y:.2f} m<extra></extra>",
            )
        )
        r_lap = r[finite]
        r_mean = float(np.nanmean(r_lap))
        r_error = r_mean - SKIDPAD_IDEAL_RADIUS_M
        if info.get("official_timed", False):
            timed_errors.append(r_error)
        rows.append({
            "Lap": lap_id,
            "Side": info["side"],
            "Role": info["role"],
            "Official timed": bool(info.get("official_timed", False)),
            "R mean [m]": r_mean,
            "R std [m]": float(np.nanstd(r_lap)),
            "Time in band [%]": float(np.nanmean(
                (r_lap >= SKIDPAD_INNER_RADIUS_M) & (r_lap <= SKIDPAD_OUTER_RADIUS_M)
            ) * 100.0),
            "Radius error [m]": r_error,
            "Samples": int(finite.sum()),
        })

    kpis = {
        "rows": rows,
        "radius_error_mean_m": _mean_if_finite(timed_errors),
        "R_mean_m": _mean_if_finite(row["R mean [m]"] for row in rows),
        "pct_time_in_band_pct": _mean_if_finite(row["Time in band [%]"] for row in rows),
        "warnings": _classification_warnings(circles),
    }
    return fig, kpis


def balance_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Slip-angle and yaw-rate balance for sustained skidpad cornering."""
    circles = classify_circles(df)
    yaw_col = _first_existing_col(df, ("VN_gz", "AS_yaw_rate"))
    required = [
        "TimeStamp", "laps", "Filtering_VN_ay", "VN_vx", "Steering",
        "Est_SAFL", "Est_SAFR", "Est_SARL", "Est_SARR",
    ]
    if yaw_col is not None:
        required.append(yaw_col)
    _require_columns(df, required)
    if yaw_col is None:
        raise KeyError("Missing yaw-rate column: VN_gz/AS_yaw_rate")
    arr = cols_to_numpy(df, required)

    front_sa_deg = 0.5 * (
        np.rad2deg(np.abs(arr["Est_SAFL"])) + np.rad2deg(np.abs(arr["Est_SAFR"]))
    )
    rear_sa_deg = 0.5 * (
        np.rad2deg(np.abs(arr["Est_SARL"])) + np.rad2deg(np.abs(arr["Est_SARR"]))
    )
    steering_rw_rad = arr["Steering"] / STEERING_RATIO
    yaw_rate = arr[yaw_col]
    ideal_steer_from_yaw = np.divide(
        WHEELBASE_EQ * yaw_rate,
        arr["VN_vx"],
        out=np.full_like(yaw_rate, np.nan, dtype=float),
        where=np.isfinite(arr["VN_vx"]) & (np.abs(arr["VN_vx"]) > 0.5),
    )
    understeer_deg = np.rad2deg(steering_rw_rad - ideal_steer_from_yaw)
    sustained = _sustained_balance_mask(arr["VN_vx"], arr["Filtering_VN_ay"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Front/rear slip angle", "Understeer angle from yaw rate"),
    )
    rows: list[dict[str, Any]] = []
    balance_vals: list[float] = []

    for idx, (lap_id, info) in enumerate(circles.items()):
        mask = (arr["laps"] == float(lap_id)) & sustained
        t = _relative_time(arr["TimeStamp"][mask])
        finite = (
            np.isfinite(t)
            & np.isfinite(front_sa_deg[mask])
            & np.isfinite(rear_sa_deg[mask])
            & np.isfinite(understeer_deg[mask])
        )
        if not finite.any():
            continue
        color = _RUN_COLORS[idx % len(_RUN_COLORS)]
        name = _circle_name(lap_id, info)
        front_lap = front_sa_deg[mask][finite]
        rear_lap = rear_sa_deg[mask][finite]
        us_lap = understeer_deg[mask][finite]
        t_lap = t[finite]
        fig.add_trace(
            go.Scattergl(
                x=t_lap,
                y=front_lap,
                mode="lines",
                name=f"{name} front",
                line=dict(color=color, width=1.2),
                legendgroup=name,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=t_lap,
                y=rear_lap,
                mode="lines",
                name=f"{name} rear",
                line=dict(color=color, width=1.2, dash="dot"),
                legendgroup=name,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=t_lap,
                y=us_lap,
                mode="lines",
                name=f"{name} US",
                line=dict(color=color, width=1.2),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        balance_index = float(np.nanmean(rear_lap) - np.nanmean(front_lap))
        balance_vals.append(balance_index)
        rows.append({
            "Lap": lap_id,
            "Side": info["side"],
            "Role": info["role"],
            "Official timed": bool(info.get("official_timed", False)),
            "SA front mean [deg]": float(np.nanmean(front_lap)),
            "SA rear mean [deg]": float(np.nanmean(rear_lap)),
            "SA front p95 [deg]": _safe_percentile(front_lap, 95.0),
            "SA rear p95 [deg]": _safe_percentile(rear_lap, 95.0),
            "Balance index [deg]": balance_index,
            "Understeer mean [deg]": float(np.nanmean(us_lap)),
            "Understeer p95 [deg]": _safe_percentile(np.abs(us_lap), 95.0),
            "Samples": int(finite.sum()),
        })

    _apply_dark_layout(fig, "Skidpad balance: slip angle and yaw response", height=620)
    fig.update_xaxes(title_text="Circle time [s]", row=2, col=1)
    fig.update_yaxes(title_text="Slip angle [deg]", row=1, col=1)
    fig.update_yaxes(title_text="Understeer angle [deg]", row=2, col=1)
    fig.add_hline(y=0.0, row=2, col=1, line=dict(color="rgba(200,200,200,0.45)", dash="dash", width=1))

    kpis = {
        "rows": rows,
        "balance_index_mean_deg": _mean_if_finite(balance_vals),
        "understeer_mean_deg": _mean_if_finite(row["Understeer mean [deg]"] for row in rows),
        "yaw_source": yaw_col,
        "warnings": _classification_warnings(circles),
    }
    return fig, kpis


def driver_smoothness_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Driver smoothness KPIs per skidpad circle."""
    circles = classify_circles(df)
    throttle_col = _first_existing_col(df, _THROTTLE_ALIASES)
    brake_col = _first_existing_col(df, _BRAKE_ALIASES)
    required = ["TimeStamp", "laps", "Steering", "VN_vx"]
    if throttle_col is not None:
        required.append(throttle_col)
    if brake_col is not None and brake_col not in required:
        required.append(brake_col)
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)

    rows: list[dict[str, Any]] = []
    for lap_id, info in circles.items():
        mask = arr["laps"] == float(lap_id)
        t = arr["TimeStamp"][mask]
        if t.size < 3:
            continue
        try:
            dt = robust_dt(t)
        except ValueError:
            continue
        steering_deg = np.rad2deg(arr["Steering"][mask] / STEERING_RATIO)
        dsteer_dt = np.gradient(steering_deg, dt, edge_order=1)
        throttle = arr[throttle_col][mask] if throttle_col is not None else np.full(t.size, np.nan)
        brake = arr[brake_col][mask] if brake_col is not None else np.full(t.size, np.nan)
        vx = arr["VN_vx"][mask]

        finite_steer = np.isfinite(dsteer_dt)
        rows.append({
            "Lap": lap_id,
            "Side": info["side"],
            "Role": info["role"],
            "Official timed": bool(info.get("official_timed", False)),
            "Steering rate RMS [deg/s]": _rms(dsteer_dt[finite_steer]),
            "Throttle variance": _safe_var(throttle),
            "Brake variance": _safe_var(brake),
            "vx variance [(m/s)^2]": _safe_var(vx),
            "Samples": int(mask.sum()),
        })

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Steering smoothness", "Pedal and speed consistency"),
    )
    x = [_circle_label_from_row(row) for row in rows]
    fig.add_trace(
        go.Bar(
            x=x,
            y=[row["Steering rate RMS [deg/s]"] for row in rows],
            name="Steering rate RMS",
            marker=dict(color="#4DB3F2"),
        ),
        row=1,
        col=1,
    )
    for name, color in (
        ("Throttle variance", "#F28C40"),
        ("Brake variance", "#F27070"),
        ("vx variance [(m/s)^2]", "#73D973"),
    ):
        fig.add_trace(
            go.Bar(
                x=x,
                y=[row[name] for row in rows],
                name=name,
                marker=dict(color=color),
            ),
            row=2,
            col=1,
        )

    _apply_dark_layout(fig, "Driver smoothness in skidpad", height=580)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="RMS [deg/s]", row=1, col=1)
    fig.update_yaxes(title_text="Variance", row=2, col=1)
    fig.update_xaxes(title_text="Circle", row=2, col=1)

    warnings = _classification_warnings(circles)
    if throttle_col is None:
        warnings.append("No throttle signal found for smoothness variance.")
    if brake_col is None:
        warnings.append("No brake signal found for smoothness variance.")
    kpis = {
        "rows": rows,
        "throttle_source": throttle_col,
        "brake_source": brake_col,
        "steering_rate_rms_mean_deg_s": _mean_if_finite(
            row["Steering rate RMS [deg/s]"] for row in rows
        ),
        "warnings": warnings,
    }
    return fig, kpis


def tv_intervention_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """TV yaw moment and yaw-rate error against ideal skidpad yaw rate."""
    circles = classify_circles(df)
    actual_mz_col = _first_existing_col(df, _TV_ACTUAL_MZ_ALIASES)
    ff_mz_col = _first_existing_col(df, _TV_FF_MZ_ALIASES)
    fb_mz_col = _first_existing_col(df, _TV_FB_MZ_ALIASES)
    desired_yaw_col = _first_existing_col(df, _TV_DESIRED_YAW_ALIASES)
    error_yaw_col = _first_existing_col(df, _TV_ERROR_YAW_ALIASES)

    if actual_mz_col is None and ff_mz_col is None and fb_mz_col is None:
        raise KeyError("Missing TV yaw moment signals.")

    required = ["TimeStamp", "laps", "Filtering_VN_ay", "VN_vx"]
    for col in (actual_mz_col, ff_mz_col, fb_mz_col, desired_yaw_col, error_yaw_col):
        if col is not None and col not in required:
            required.append(col)
    if "VN_gz" in df.columns:
        required.append("VN_gz")
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)

    actual_mz = (
        arr[actual_mz_col]
        if actual_mz_col is not None
        else np.zeros(len(df), dtype=float)
    )
    if actual_mz_col is None:
        if ff_mz_col is not None:
            actual_mz = actual_mz + arr[ff_mz_col]
        if fb_mz_col is not None:
            actual_mz = actual_mz + arr[fb_mz_col]

    yaw_real = np.full(len(df), np.nan, dtype=float)
    yaw_source = None
    if "VN_gz" in arr:
        yaw_real = arr["VN_gz"]
        yaw_source = "VN_gz"
    elif desired_yaw_col is not None and error_yaw_col is not None:
        yaw_real = arr[desired_yaw_col] - arr[error_yaw_col]
        yaw_source = f"{desired_yaw_col} - {error_yaw_col}"

    ideal_yaw = np.sign(arr["Filtering_VN_ay"]) * np.abs(arr["VN_vx"]) / SKIDPAD_IDEAL_RADIUS_M
    sign_mask = np.isfinite(yaw_real) & np.isfinite(ideal_yaw) & (np.abs(ideal_yaw) > 0.05)
    if sign_mask.any() and float(np.nanmedian(yaw_real[sign_mask] * ideal_yaw[sign_mask])) < 0.0:
        ideal_yaw = -ideal_yaw
    yaw_error = yaw_real - ideal_yaw

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("TV yaw moment", "Yaw-rate error vs ideal skidpad yaw"),
    )
    rows: list[dict[str, Any]] = []
    for idx, (lap_id, info) in enumerate(circles.items()):
        mask = (arr["laps"] == float(lap_id)) & _sustained_balance_mask(arr["VN_vx"], arr["Filtering_VN_ay"])
        t = _relative_time(arr["TimeStamp"][mask])
        finite_mz = np.isfinite(t) & np.isfinite(actual_mz[mask])
        finite_yaw = finite_mz & np.isfinite(yaw_error[mask])
        if not finite_mz.any():
            continue
        color = _RUN_COLORS[idx % len(_RUN_COLORS)]
        name = _circle_name(lap_id, info)
        fig.add_trace(
            go.Scattergl(
                x=t[finite_mz],
                y=actual_mz[mask][finite_mz],
                mode="lines",
                name=name,
                line=dict(color=color, width=1.3),
            ),
            row=1,
            col=1,
        )
        if finite_yaw.any():
            fig.add_trace(
                go.Scattergl(
                    x=t[finite_yaw],
                    y=yaw_error[mask][finite_yaw],
                    mode="lines",
                    name=f"{name} yaw error",
                    line=dict(color=color, width=1.2),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        yaw_lap = yaw_error[mask][finite_yaw]
        mz_lap = actual_mz[mask][finite_mz]
        rows.append({
            "Lap": lap_id,
            "Side": info["side"],
            "Role": info["role"],
            "Official timed": bool(info.get("official_timed", False)),
            "Mz mean [Nm]": float(np.nanmean(mz_lap)) if mz_lap.size else np.nan,
            "Mz abs mean [Nm]": float(np.nanmean(np.abs(mz_lap))) if mz_lap.size else np.nan,
            "Yaw err RMS [rad/s]": _rms(yaw_lap),
            "Samples": int(max(finite_mz.sum(), finite_yaw.sum())),
        })

    _apply_dark_layout(fig, "TV intervention in skidpad", height=620)
    fig.update_xaxes(title_text="Circle time [s]", row=2, col=1)
    fig.update_yaxes(title_text="Mz [Nm]", row=1, col=1)
    fig.update_yaxes(title_text="Yaw error [rad/s]", row=2, col=1)
    fig.add_hline(y=0.0, row=1, col=1, line=dict(color="rgba(200,200,200,0.4)", dash="dash", width=1))
    fig.add_hline(y=0.0, row=2, col=1, line=dict(color="rgba(200,200,200,0.4)", dash="dash", width=1))

    warnings = _classification_warnings(circles)
    if yaw_source is None:
        warnings.append("No yaw-rate source found; yaw-error KPI is unavailable.")
    kpis = {
        "rows": rows,
        "Mz_mean_Nm": _mean_if_finite(row["Mz mean [Nm]"] for row in rows),
        "Mz_abs_mean_Nm": _mean_if_finite(row["Mz abs mean [Nm]"] for row in rows),
        "yaw_err_rms_rad_s": _mean_if_finite(row["Yaw err RMS [rad/s]"] for row in rows),
        "mz_source": actual_mz_col or "TV_feedForwardMz + TV_feedBackMz",
        "yaw_source": yaw_source,
        "warnings": warnings,
    }
    return fig, kpis


def lateral_load_dist_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Side-aware lateral load transfer distribution in skidpad circles."""
    circles = classify_circles(df)
    required = [
        "TimeStamp", "laps", "Est_FZFL", "Est_FZFR", "Est_FZRL", "Est_FZRR",
    ]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Estimated vertical load", "Front/rear load transfer ratio"),
    )
    rows: list[dict[str, Any]] = []
    lltd_all: list[float] = []

    for idx, (lap_id, info) in enumerate(circles.items()):
        mask = arr["laps"] == float(lap_id)
        t = _relative_time(arr["TimeStamp"][mask])
        if not np.isfinite(t).any():
            continue
        color = _RUN_COLORS[idx % len(_RUN_COLORS)]
        for wheel in ("FL", "FR", "RL", "RR"):
            fig.add_trace(
                go.Scattergl(
                    x=t,
                    y=arr[f"Est_FZ{wheel}"][mask],
                    mode="lines",
                    name=f"{_circle_name(lap_id, info)} {wheel}",
                    line=dict(color=WHEEL_COLORS[wheel], width=0.8),
                    opacity=0.55,
                    legendgroup=wheel,
                    showlegend=idx == 0,
                ),
                row=1,
                col=1,
            )

        lltd = _lltd_for_side(info["side"], arr, mask)
        finite = np.isfinite(t) & np.isfinite(lltd)
        if finite.any():
            fig.add_trace(
                go.Scattergl(
                    x=t[finite],
                    y=lltd[finite],
                    mode="lines",
                    name=_circle_name(lap_id, info),
                    line=dict(color=color, width=1.3),
                ),
                row=2,
                col=1,
            )
            lltd_lap = lltd[finite]
            lltd_all.extend(lltd_lap.astype(float).tolist())
            rows.append({
                "Lap": lap_id,
                "Side": info["side"],
                "Role": info["role"],
                "Official timed": bool(info.get("official_timed", False)),
                "LLTD F/R ratio mean [-]": float(np.nanmean(lltd_lap)),
                "LLTD F/R ratio std [-]": float(np.nanstd(lltd_lap)),
                "Samples": int(finite.sum()),
            })

    _apply_dark_layout(fig, "Lateral load distribution in skidpad", height=620)
    fig.update_xaxes(title_text="Circle time [s]", row=2, col=1)
    fig.update_yaxes(title_text="Fz [N]", row=1, col=1)
    fig.update_yaxes(title_text="Front transfer / rear transfer [-]", row=2, col=1)
    fig.add_hline(y=1.0, row=2, col=1, line=dict(color=_REFERENCE, dash="dash", width=1.2))

    kpis = {
        "rows": rows,
        "lltd_ratio_mean": _mean_if_finite(lltd_all),
        "mass_kg": MASS_KG,
        "warnings": _classification_warnings(circles),
    }
    return fig, kpis


def gps_figure8_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Plot local GPS trajectory with theoretical skidpad bounds."""
    circles = classify_circles(df)
    required = ["TimeStamp", "laps", "VN_latitude", "VN_longitude", "Filtering_VN_ay"]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)
    gps_valid = (
        np.isfinite(arr["VN_latitude"])
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
            hovertemplate="x=%{x:.1f} m<br>y=%{y:.1f} m<extra></extra>",
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
                    name=f"{side} {name} radius",
                    line=dict(color=_REFERENCE, dash=dash, width=1.0),
                    opacity=0.72,
                    hoverinfo="skip",
                )
            )

    fig.update_yaxes(scaleanchor="x", scaleratio=1.0)
    fig.update_layout(height=650)

    kpis = {
        "gps_samples": int(gps_valid.sum()),
        "centers": {side: (float(center[0]), float(center[1])) for side, center in centers.items()},
        "warnings": _classification_warnings(circles),
    }
    if len(centers) < 2:
        kpis["warnings"].append("Only one skidpad circle center could be estimated from GPS.")
    return fig, kpis


def has_tv_signals(df: pl.DataFrame) -> bool:
    """Return True when at least one TV yaw-moment signal is present."""
    return any(col in df.columns for col in (*_TV_ACTUAL_MZ_ALIASES, *_TV_FF_MZ_ALIASES, *_TV_FB_MZ_ALIASES))


def has_load_signals(df: pl.DataFrame) -> bool:
    """Return True when all estimated vertical load signals are present."""
    return all(col in df.columns for col in ("Est_FZFL", "Est_FZFR", "Est_FZRL", "Est_FZRR"))


def _require_columns(df: pl.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing skidpad columns: {missing}")


def _first_existing_col(df: pl.DataFrame, aliases: tuple[str, ...]) -> str | None:
    return next((col for col in aliases if col in df.columns), None)


def _opposite_side(side: str) -> str:
    return "R" if side == "L" else "L"


def _safe_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanmean(finite)) if finite.size else np.nan


def _safe_var(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanvar(finite)) if finite.size else np.nan


def _safe_percentile(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanpercentile(finite, q)) if finite.size else np.nan


def _mean_if_finite(values: Any) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmean(arr)) if arr.size else np.nan


def _rms(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.nanmean(finite ** 2))) if finite.size else np.nan


def _relative_time(time_s: np.ndarray) -> np.ndarray:
    time = np.asarray(time_s, dtype=float)
    finite = time[np.isfinite(time)]
    if finite.size == 0:
        return np.full_like(time, np.nan, dtype=float)
    return time - float(np.nanmin(finite))


def _circle_name(lap_id: int, info: dict[str, Any]) -> str:
    suffix = "official" if info.get("official_timed", False) else info["role"]
    return f"L{lap_id} {info['side']} {suffix}"


def _circle_rows(circles: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "Lap": lap_id,
            "Circle": _circle_name(lap_id, info),
            "Side": info["side"],
            "Role": info["role"],
            "Official timed": bool(info.get("official_timed", False)),
            "Lap time [s]": float(info["laptime_s"]),
            "Samples": int(info["n_samples"]),
            "Mean ay [m/s2]": float(info["mean_ay_mps2"]),
        }
        for lap_id, info in circles.items()
    ]


def _circle_label_from_row(row: dict[str, Any]) -> str:
    return f"L{row['Lap']} {row['Side']} {row['Role']}"


def _circle_table(rows: list[dict[str, Any]]) -> go.Table:
    headers = ["Lap", "Side", "Role", "Official", "Lap time [s]", "Samples", "Mean ay [m/s2]"]
    cells = [
        [row["Lap"] for row in rows],
        [row["Side"] for row in rows],
        [row["Role"] for row in rows],
        ["yes" if row["Official timed"] else "" for row in rows],
        [_fmt_float(row["Lap time [s]"], 3) for row in rows],
        [row["Samples"] for row in rows],
        [_fmt_float(row["Mean ay [m/s2]"], 3) for row in rows],
    ]
    return go.Table(
        header=dict(values=headers, fill_color="#22252B", font=dict(color=_TEXT), align="left"),
        cells=dict(values=cells, fill_color="#171A1F", font=dict(color=_TEXT), align="left"),
    )


def _fmt_float(value: float, digits: int) -> str:
    return f"{value:.{digits}f}" if np.isfinite(value) else "n/a"


def _official_timed_times(circles: dict[int, dict[str, Any]]) -> dict[str, float]:
    official = {"R": np.nan, "L": np.nan}
    for side in ("R", "L"):
        candidates = [
            info["laptime_s"]
            for info in circles.values()
            if info["side"] == side and info["role"] == "timed" and np.isfinite(info["laptime_s"])
        ]
        if candidates:
            official[side] = float(np.nanmin(candidates))
    return official


def _classification_warnings(circles: dict[int, dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    n_total = len(circles)
    n_r = sum(1 for info in circles.values() if info["side"] == "R")
    n_l = sum(1 for info in circles.values() if info["side"] == "L")
    if n_total != 4 or n_r != 2 or n_l != 2:
        warnings.append(
            f"Expected 4 skidpad circles (2 R + 2 L), got {n_total} ({n_r} R, {n_l} L)."
        )
    return warnings


def _driven_radius_m(vx_mps: np.ndarray, ay_mps2: np.ndarray) -> np.ndarray:
    ay_abs = np.abs(np.asarray(ay_mps2, dtype=float))
    vx = np.asarray(vx_mps, dtype=float)
    radius = np.divide(
        vx ** 2,
        ay_abs,
        out=np.full_like(vx, np.nan, dtype=float),
        where=np.isfinite(vx) & np.isfinite(ay_abs) & (ay_abs > 0.2),
    )
    if len(radius) >= 5:
        try:
            dt = 0.01
            radius = smooth_signal(radius, max(1, int(round(0.15 / dt))))
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


def _lltd_for_side(side: str, arr: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    if side == "R":
        f_outer = arr["Est_FZFL"][mask]
        f_inner = arr["Est_FZFR"][mask]
        r_outer = arr["Est_FZRL"][mask]
        r_inner = arr["Est_FZRR"][mask]
    else:
        f_outer = arr["Est_FZFR"][mask]
        f_inner = arr["Est_FZFL"][mask]
        r_outer = arr["Est_FZRR"][mask]
        r_inner = arr["Est_FZRL"][mask]
    front_transfer = f_outer - f_inner
    rear_transfer = r_outer - r_inner
    return np.divide(
        front_transfer,
        rear_transfer,
        out=np.full_like(front_transfer, np.nan, dtype=float),
        where=np.isfinite(front_transfer) & np.isfinite(rear_transfer) & (np.abs(rear_transfer) > 1.0),
    )


def _skidpad_centers_from_laps(
    circles: dict[int, dict[str, Any]],
    laps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, tuple[float, float]]:
    centers: dict[str, tuple[float, float]] = {}
    for side in ("R", "L"):
        lap_ids = [lap_id for lap_id, info in circles.items() if info["side"] == side]
        if not lap_ids:
            continue
        mask = np.isin(laps, np.asarray(lap_ids, dtype=float)) & np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 10:
            continue
        centers[side] = (float(np.nanmedian(x[mask])), float(np.nanmedian(y[mask])))

    if len(centers) == 2:
        sides = list(centers)
        c0 = np.asarray(centers[sides[0]], dtype=float)
        c1 = np.asarray(centers[sides[1]], dtype=float)
        delta = c1 - c0
        dist = float(np.hypot(delta[0], delta[1]))
        if np.isfinite(dist) and dist > 1.0:
            unit = delta / dist
            mid = 0.5 * (c0 + c1)
            centers[sides[0]] = tuple((mid - 0.5 * SKIDPAD_CIRCLE_GAP_M * unit).tolist())
            centers[sides[1]] = tuple((mid + 0.5 * SKIDPAD_CIRCLE_GAP_M * unit).tolist())
    return centers


def _apply_dark_layout(fig: go.Figure, title: str, *, height: int) -> None:
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
    )
    fig.update_xaxes(color=_AXIS, gridcolor=_GRID, linecolor=_AXIS, tickcolor=_AXIS, showgrid=True)
    fig.update_yaxes(color=_AXIS, gridcolor=_GRID, linecolor=_AXIS, tickcolor=_AXIS, showgrid=True)
