"""Skidpad event KPI plots for Formula Student telemetry.

OptimumG (Claude Rouelle) methodology: KPIs are aggregated characteristic
plots, never time-series. Each public ``*_fig`` returns a Plotly figure plus
a small support dict (mainly warnings). The only spatial figure is the GPS
figure-8 map. No `variable vs time` plots are produced here.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import plotly.graph_objects as go

from src.dynamics import G_MPS2, MASS_KG, STEERING_RATIO, WHEELBASE_EQ
from src.lapcount import gps_to_local_xy
from utils import cols_to_numpy, make_dark_figure, smooth_signal

SKIDPAD_INNER_RADIUS_M = 7.625
SKIDPAD_OUTER_RADIUS_M = 10.625
# CoG follows inner cone + half of the front track (Vhcl.tf = 1.225 m), not the 3 m path centerline.
SKIDPAD_IDEAL_RADIUS_M = 7.625 + 1.225 / 2
SKIDPAD_PATH_WIDTH_M = 3.0
SKIDPAD_CIRCLE_GAP_M = 18.25
SKIDPAD_TIMED_LAPS = (2, 4)

AY_POSITIVE_SIDE = "L"
MIN_SKIDPAD_SPEED_MPS = 2.0
MIN_SKIDPAD_AY_MPS2 = 1.0
MIN_BALANCE_SPEED_MPS = 3.0
MIN_BALANCE_AY_MPS2 = 2.0

_TEXT = "#EBEBEB"
_REFERENCE = "#F2D44D"
_FRONT_COLOR = "#4DB3F2"
_REAR_COLOR = "#F28C40"
_RUN_COLORS = ("#4DB3F2", "#F28C40", "#73D973", "#F27070", "#D973D9", "#F2C94C")

_TV_ACTUAL_MZ_ALIASES = ("TV_actualMz", "TV_actualmz", "tv_actual_mz", "tv_actualmz")
_TV_FF_MZ_ALIASES = ("TV_feedForwardMz", "TV_feedforwardMz", "tv_feedforward_mz")
_TV_FB_MZ_ALIASES = ("TV_feedBackMz", "TV_feedbackMz", "tv_feedback_mz")


def is_skidpad_run(df: pl.DataFrame) -> bool:
    """Return True when most samples are tagged as skidpad by lapcount."""
    if "lapcount_mode" not in df.columns or len(df) == 0:
        return False
    modes = df["lapcount_mode"].to_list()
    n_skidpad = sum(str(value).lower() == "skidpad" for value in modes)
    return n_skidpad > 0.5 * len(modes)


def has_tv_signals(df: pl.DataFrame) -> bool:
    """Return True when at least one TV yaw-moment signal is present."""
    return any(col in df.columns for col in (*_TV_ACTUAL_MZ_ALIASES, *_TV_FF_MZ_ALIASES, *_TV_FB_MZ_ALIASES))


def has_load_signals(df: pl.DataFrame) -> bool:
    """Return True when all estimated vertical load signals are present."""
    return all(col in df.columns for col in ("Est_FZFL", "Est_FZFR", "Est_FZRL", "Est_FZRR"))


def classify_circles(df: pl.DataFrame) -> dict[int, dict[str, Any]]:
    """Return only the official (timed) skidpad laps, labelled by side R/L.

    Warm-up laps are dropped entirely — the FS skidpad result is built from
    the timed lap of each side, so warm-up samples never feed any KPI.
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

    timed_only = {lap_id: info for lap_id, info in rows.items() if info["role"] == "timed"}
    return dict(sorted(timed_only.items(), key=lambda item: item[1]["first_time_s"]))


def event_time_bars_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Per-circle laptime bar chart (no time axis)."""
    circles = classify_circles(df)
    fig = make_dark_figure("Skidpad lap time per circle", "Circle", "Lap time [s]")
    if not circles:
        return fig, {"warnings": ["No skidpad circles detected."]}

    labels = [_circle_name(lap_id, info) for lap_id, info in circles.items()]
    times = [float(info["laptime_s"]) for info in circles.values()]
    colors = [
        "#73D973" if info.get("official_timed", False) else "#6C7A89"
        for info in circles.values()
    ]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=times,
            marker=dict(color=colors, line=dict(color="#22252B", width=1)),
            text=[f"{t:.3f} s" if np.isfinite(t) else "" for t in times],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:.3f} s<extra></extra>",
            name="Lap time",
        )
    )

    official = _official_timed_times(circles)
    event_time_s = _mean_if_finite([official.get("R", np.nan), official.get("L", np.nan)])
    if np.isfinite(event_time_s):
        fig.add_hline(
            y=event_time_s,
            line=dict(color=_REFERENCE, dash="dash", width=1.2),
            annotation_text=f"Event time {event_time_s:.3f} s",
            annotation_font_color=_TEXT,
        )

    fig.update_layout(height=420, showlegend=False)
    return fig, {"warnings": _classification_warnings(circles)}


def lateral_g_histogram_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """|ay| distribution per circle (overlaid histograms)."""
    circles = classify_circles(df)
    required = ["laps", "Filtering_VN_ay"]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)

    fig = make_dark_figure("Skidpad |ay| distribution", "|ay| [g]", "Time share [%]")
    any_data = False
    for idx, (lap_id, info) in enumerate(circles.items()):
        mask = arr["laps"] == float(lap_id)
        ay_g = np.abs(arr["Filtering_VN_ay"][mask]) / G_MPS2
        ay_g = ay_g[np.isfinite(ay_g)]
        if ay_g.size == 0:
            continue
        any_data = True
        fig.add_trace(
            go.Histogram(
                x=ay_g,
                name=_circle_name(lap_id, info),
                marker_color=_RUN_COLORS[idx % len(_RUN_COLORS)],
                opacity=0.55,
                xbins=dict(start=0.0, end=2.4, size=0.05),
                histnorm="percent",
                hovertemplate="%{x:.2f} g<br>%{y:.1f} %<extra></extra>",
            )
        )

    fig.update_layout(barmode="overlay", height=440)
    fig.update_xaxes(range=[0.0, 2.4])
    warnings = _classification_warnings(circles)
    if not any_data:
        warnings.append("No |ay| samples available for the histogram.")
    return fig, {"warnings": warnings}


def driven_radius_histogram_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Driven radius distribution per circle with FS bands and ideal line."""
    circles = classify_circles(df)
    required = ["laps", "Filtering_VN_ay", "VN_vx"]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)
    radius_m = _driven_radius_m(arr["VN_vx"], arr["Filtering_VN_ay"])
    valid = _sustained_radius_mask(arr["VN_vx"], arr["Filtering_VN_ay"], radius_m)

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

    any_data = False
    for idx, (lap_id, info) in enumerate(circles.items()):
        mask = (arr["laps"] == float(lap_id)) & valid
        r_lap = radius_m[mask]
        r_lap = r_lap[np.isfinite(r_lap)]
        r_lap = r_lap[(r_lap >= 2.0) & (r_lap <= 20.0)]
        if r_lap.size == 0:
            continue
        any_data = True
        fig.add_trace(
            go.Histogram(
                x=r_lap,
                name=_circle_name(lap_id, info),
                marker_color=_RUN_COLORS[idx % len(_RUN_COLORS)],
                opacity=0.55,
                xbins=dict(start=2.0, end=20.0, size=0.25),
                histnorm="percent",
                hovertemplate="%{x:.2f} m<br>%{y:.1f} %<extra></extra>",
            )
        )

    fig.update_layout(barmode="overlay", height=440)
    fig.update_xaxes(range=[2.0, 20.0])
    warnings = _classification_warnings(circles)
    if not any_data:
        warnings.append("No sustained-cornering samples for the radius histogram.")
    return fig, {"warnings": warnings}


def understeer_chart_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Steering angle (at wheel) vs lateral G with per-side linear fit."""
    circles = classify_circles(df)
    required = ["laps", "Filtering_VN_ay", "VN_vx", "Steering"]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)
    sustained = _sustained_balance_mask(arr["VN_vx"], arr["Filtering_VN_ay"])
    steering_deg = np.rad2deg(arr["Steering"] / STEERING_RATIO)
    ay_g = arr["Filtering_VN_ay"] / G_MPS2

    fig = make_dark_figure(
        "Understeer chart: steering at wheel vs ay",
        "Lateral G [g]",
        "Steering at wheel [deg]",
    )
    fig.add_hline(y=0.0, line=dict(color="rgba(200,200,200,0.4)", dash="dash", width=1))
    fig.add_vline(x=0.0, line=dict(color="rgba(200,200,200,0.4)", dash="dash", width=1))

    fit_lines: dict[str, tuple[float, float, int]] = {}
    for side, color in (("R", "#4DB3F2"), ("L", "#F28C40")):
        side_laps = [lap_id for lap_id, info in circles.items() if info["side"] == side]
        if not side_laps:
            continue
        mask = np.isin(arr["laps"], np.asarray(side_laps, dtype=float)) & sustained
        x = ay_g[mask]
        y = steering_deg[mask]
        good = np.isfinite(x) & np.isfinite(y)
        x = x[good]
        y = y[good]
        if x.size < 5:
            continue
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                name=f"{side} side",
                marker=dict(color=color, size=4, opacity=0.45),
                hovertemplate="ay=%{x:.2f} g<br>δ=%{y:.2f}°<extra></extra>",
            )
        )
        slope, intercept = np.polyfit(x, y, 1)
        x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 30)
        fig.add_trace(
            go.Scatter(
                x=x_fit,
                y=slope * x_fit + intercept,
                mode="lines",
                name=f"{side} fit",
                line=dict(color=color, width=2),
                hovertemplate=f"slope={slope:+.2f} °/g<extra></extra>",
            )
        )
        fit_lines[side] = (float(slope), float(intercept), int(x.size))

    fig.update_layout(height=480)
    warnings = _classification_warnings(circles)
    if not fit_lines:
        warnings.append("Not enough sustained samples to fit the understeer chart.")
    return fig, {"warnings": warnings, "fits": fit_lines}


def slip_angle_vs_ay_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Front and rear average slip angle vs lateral G."""
    circles = classify_circles(df)
    required = [
        "laps", "Filtering_VN_ay", "VN_vx",
        "Est_SAFL", "Est_SAFR", "Est_SARL", "Est_SARR",
    ]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)
    sustained = _sustained_balance_mask(arr["VN_vx"], arr["Filtering_VN_ay"]) & _lap_mask(
        arr["laps"], circles
    )

    front_sa_deg = 0.5 * (
        np.rad2deg(np.abs(arr["Est_SAFL"])) + np.rad2deg(np.abs(arr["Est_SAFR"]))
    )
    rear_sa_deg = 0.5 * (
        np.rad2deg(np.abs(arr["Est_SARL"])) + np.rad2deg(np.abs(arr["Est_SARR"]))
    )
    abs_ay_g = np.abs(arr["Filtering_VN_ay"]) / G_MPS2

    fig = make_dark_figure(
        "Slip angle vs lateral G",
        "|ay| [g]",
        "|SA| [deg]",
    )
    any_data = False
    for axle_name, axle_arr, color in (
        ("Front (avg)", front_sa_deg, _FRONT_COLOR),
        ("Rear (avg)", rear_sa_deg, _REAR_COLOR),
    ):
        mask = sustained & np.isfinite(abs_ay_g) & np.isfinite(axle_arr)
        x = abs_ay_g[mask]
        y = axle_arr[mask]
        if x.size == 0:
            continue
        any_data = True
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                name=axle_name,
                marker=dict(color=color, size=4, opacity=0.45),
                hovertemplate=f"{axle_name}<br>ay=%{{x:.2f}} g<br>SA=%{{y:.2f}}°<extra></extra>",
            )
        )
        if x.size >= 5:
            slope, intercept = np.polyfit(x, y, 1)
            x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 30)
            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=slope * x_fit + intercept,
                    mode="lines",
                    name=f"{axle_name} fit",
                    line=dict(color=color, width=2, dash="dash"),
                    showlegend=False,
                )
            )

    fig.update_layout(height=480)
    warnings = _classification_warnings(circles)
    if not any_data:
        warnings.append("No sustained-cornering samples for slip-angle scatter.")
    return fig, {"warnings": warnings}


def yaw_rate_vs_ay_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Yaw rate vs lateral G with the steady-state skidpad reference curve."""
    circles = classify_circles(df)
    yaw_col = _first_existing_col(df, ("VN_gz", "AS_yaw_rate"))
    if yaw_col is None:
        raise KeyError("Missing yaw-rate column: VN_gz/AS_yaw_rate")
    required = ["laps", "Filtering_VN_ay", "VN_vx", yaw_col]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)
    sustained = _sustained_balance_mask(arr["VN_vx"], arr["Filtering_VN_ay"])

    ay_g = arr["Filtering_VN_ay"] / G_MPS2
    yaw = arr[yaw_col]

    fig = make_dark_figure(
        "Yaw rate vs lateral G",
        "Lateral G [g]",
        "Yaw rate [rad/s]",
    )
    fig.add_hline(y=0.0, line=dict(color="rgba(200,200,200,0.4)", dash="dash", width=1))
    fig.add_vline(x=0.0, line=dict(color="rgba(200,200,200,0.4)", dash="dash", width=1))

    for side, color in (("R", "#4DB3F2"), ("L", "#F28C40")):
        side_laps = [lap_id for lap_id, info in circles.items() if info["side"] == side]
        if not side_laps:
            continue
        mask = np.isin(arr["laps"], np.asarray(side_laps, dtype=float)) & sustained
        x = ay_g[mask]
        y = yaw[mask]
        good = np.isfinite(x) & np.isfinite(y)
        x = x[good]
        y = y[good]
        if x.size == 0:
            continue
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                name=f"{side} side",
                marker=dict(color=color, size=4, opacity=0.45),
                hovertemplate="ay=%{x:.2f} g<br>r=%{y:+.2f} rad/s<extra></extra>",
            )
        )

    ay_ref = np.linspace(-2.4, 2.4, 200)
    yaw_ref = np.sign(ay_ref) * np.sqrt(np.abs(ay_ref) * G_MPS2 / SKIDPAD_IDEAL_RADIUS_M)
    fig.add_trace(
        go.Scatter(
            x=ay_ref,
            y=yaw_ref,
            mode="lines",
            name=f"Steady-state R={SKIDPAD_IDEAL_RADIUS_M:.3f} m",
            line=dict(color=_REFERENCE, dash="dash", width=1.6),
            hoverinfo="skip",
        )
    )

    fig.update_layout(height=480)
    return fig, {"warnings": _classification_warnings(circles), "yaw_source": yaw_col}


def lltd_scatter_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Front lateral load transfer vs rear lateral load transfer."""
    circles = classify_circles(df)
    required = ["laps", "Filtering_VN_ay", "Est_FZFL", "Est_FZFR", "Est_FZRL", "Est_FZRR"]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)

    fig = make_dark_figure(
        "Lateral load transfer balance",
        "Rear axle transfer [N]",
        "Front axle transfer [N]",
    )
    fig.add_hline(y=0.0, line=dict(color="rgba(200,200,200,0.4)", dash="dash", width=1))
    fig.add_vline(x=0.0, line=dict(color="rgba(200,200,200,0.4)", dash="dash", width=1))

    any_data = False
    max_abs = 0.0
    for idx, (lap_id, info) in enumerate(circles.items()):
        mask = arr["laps"] == float(lap_id)
        if int(mask.sum()) == 0:
            continue
        front_transfer, rear_transfer = _signed_axle_transfers(info["side"], arr, mask)
        good = np.isfinite(front_transfer) & np.isfinite(rear_transfer)
        if not good.any():
            continue
        any_data = True
        x = rear_transfer[good]
        y = front_transfer[good]
        max_abs = max(max_abs, float(np.nanmax(np.abs(np.concatenate([x, y])))))
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                name=_circle_name(lap_id, info),
                marker=dict(color=_RUN_COLORS[idx % len(_RUN_COLORS)], size=4, opacity=0.5),
                hovertemplate="Rear=%{x:.0f} N<br>Front=%{y:.0f} N<extra></extra>",
            )
        )

    if max_abs > 0.0:
        lim = max_abs * 1.05
        fig.add_trace(
            go.Scatter(
                x=[-lim, lim],
                y=[-lim, lim],
                mode="lines",
                name="Front = Rear (neutral)",
                line=dict(color=_REFERENCE, dash="dash", width=1.4),
                hoverinfo="skip",
            )
        )
        fig.update_xaxes(range=[-lim, lim])
        fig.update_yaxes(range=[-lim, lim], scaleanchor="x", scaleratio=1.0)

    fig.update_layout(height=520)
    warnings = _classification_warnings(circles)
    if not any_data:
        warnings.append("No load samples available for the LLTD scatter.")
    return fig, {"warnings": warnings}


def gps_figure8_fig(df: pl.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    """Plot local GPS trajectory coloured by |ay| with theoretical bounds."""
    circles = classify_circles(df)
    required = ["TimeStamp", "laps", "VN_latitude", "VN_longitude", "Filtering_VN_ay"]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)
    gps_valid = (
        np.isfinite(arr["VN_latitude"])
        & np.isfinite(arr["VN_longitude"])
        & ((np.abs(arr["VN_latitude"]) > 1e-9) | (np.abs(arr["VN_longitude"]) > 1e-9))
        & _lap_mask(arr["laps"], circles)
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

    warnings = _classification_warnings(circles)
    if len(centers) < 2:
        warnings.append("Only one skidpad circle center could be estimated from GPS.")
    return fig, {
        "gps_samples": int(gps_valid.sum()),
        "centers": {side: (float(center[0]), float(center[1])) for side, center in centers.items()},
        "warnings": warnings,
    }


def _require_columns(df: pl.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing skidpad columns: {missing}")


def _lap_mask(laps: np.ndarray, circles: dict[int, dict[str, Any]]) -> np.ndarray:
    """Boolean mask selecting samples that belong to the given (timed) laps."""
    if not circles:
        return np.zeros_like(laps, dtype=bool)
    lap_ids = np.asarray(list(circles), dtype=float)
    return np.isin(laps, lap_ids)


def _first_existing_col(df: pl.DataFrame, aliases: tuple[str, ...]) -> str | None:
    return next((col for col in aliases if col in df.columns), None)


def _opposite_side(side: str) -> str:
    return "R" if side == "L" else "L"


def _safe_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanmean(finite)) if finite.size else np.nan


def _mean_if_finite(values: Any) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmean(arr)) if arr.size else np.nan


def _circle_name(lap_id: int, info: dict[str, Any]) -> str:
    suffix = "official" if info.get("official_timed", False) else info["role"]
    return f"L{lap_id} {info['side']} {suffix}"


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
    if n_total != 2 or n_r != 1 or n_l != 1:
        warnings.append(
            f"Expected 2 timed skidpad circles (1 R + 1 L), got {n_total} ({n_r} R, {n_l} L)."
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


def _signed_axle_transfers(
    side: str,
    arr: dict[str, np.ndarray],
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Signed (outer - inner) load transfer per axle, side-aware."""
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
