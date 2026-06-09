"""Skidpad event KPIs and figures.

The functions in this module are pure dashboard helpers: they accept a
``polars.DataFrame`` already loaded by ``utils.load_data`` and return a Plotly
figure plus a KPI dictionary. Rendering belongs in ``src/dashboard.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.dynamics import (
    G_MPS2,
    GEAR_RATIO,
    WHEEL_RADIUS_M,
    WHEELBASE_EQ,
)
from src.lapcount import gps_to_local_xy
from utils import (
    cols_to_numpy,
    driver_color,
    first_existing_col,
    make_dark_figure,
    robust_dt,
    smooth_signal,
)

SKIDPAD_INNER_RADIUS_M = 7.625
SKIDPAD_OUTER_RADIUS_M = 10.625
SKIDPAD_IDEAL_RADIUS_M = 9.125
SKIDPAD_PATH_WIDTH_M = 3.0
SKIDPAD_CIRCLE_GAP_M = 18.25
SKIDPAD_TIMED_LAPS = (2, 4)

TRACK_F_M = 1.225
TRACK_R_M = 1.175
KROLL_F_NM_RAD = 36929.4
KROLL_R_NM_RAD = 40833.7
LLTD_THEORY = KROLL_F_NM_RAD / (KROLL_F_NM_RAD + KROLL_R_NM_RAD)

AY_POSITIVE_SIDE = "L"
MIN_SKIDPAD_SPEED_MPS = 2.0
MIN_SKIDPAD_AY_MPS2 = 1.0
MIN_BALANCE_SPEED_MPS = 3.0
MIN_BALANCE_AY_MPS2 = 2.0

WHEELS = ("FL", "FR", "RL", "RR")
_TV_TORQUE_COLS = ("TV_FL_Trq", "TV_FR_Trq", "TV_RL_Trq", "TV_RR_Trq")
_TV_OPTIONAL_COLS = ("TV_desiredYawRate", "TV_errorYawRate", "TV_actualMz")
_FZ_COLS = ("Est_FZFL", "Est_FZFR", "Est_FZRL", "Est_FZRR")
_SA_COLS = ("Est_SAFL", "Est_SAFR", "Est_SARL", "Est_SARR")
_YAW_ALIASES = ("VN_gz", "AS_yaw_rate")

_BG = "#141417"
_TEXT = "#EBEBEB"
_GRID = "rgba(128,128,128,0.2)"
_AXIS = "#E5E5E5"
_REFERENCE = "#F2D44D"
_SIDE_COLORS = {"R": "#4DB3F2", "L": "#F28C40"}


def _run_label(run_name: str) -> str:
    """Short, human-readable run name for legends and tables."""
    return Path(run_name).stem


def _run_color(run_name: str) -> str:
    """Stable per-run colour — shared driver-identity palette used everywhere."""
    return driver_color(run_name)


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
            lap
            for lap in side_laps
            if rows[lap]["role"] == "timed" and np.isfinite(rows[lap]["laptime_s"])
        ]
        if timed_laps:
            best_lap = min(timed_laps, key=lambda lap: rows[lap]["laptime_s"])
            rows[best_lap]["official_timed"] = True

    timed_rows = {lap_id: info for lap_id, info in rows.items() if info["role"] == "timed"}
    return dict(sorted(timed_rows.items(), key=lambda item: item[1]["first_time_s"]))


def event_time_summary_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict[str, Any]]:
    """Overlay official right/left timed circle times for every run."""
    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "table"}], [{"type": "xy"}]],
        row_heights=[0.42, 0.58],
        vertical_spacing=0.10,
    )
    table_rows: list[dict[str, object]] = []
    runs: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for idx, (run_name, df) in enumerate(dfs.items()):
        label = _run_label(run_name)
        color = _run_color(run_name)
        circles = classify_circles(df)
        for warning in _classification_warnings(circles):
            warnings.append(f"{label}: {warning}")
        official = _official_timed_times(circles)
        event_time_s = _mean_if_finite((official.get("R", np.nan), official.get("L", np.nan)))
        lr_asymmetry_s = _abs_diff(official.get("L", np.nan), official.get("R", np.nan))
        y = [official.get("R", np.nan), official.get("L", np.nan)]
        fig.add_trace(
            go.Bar(
                x=["R timed", "L timed"],
                y=y,
                name=label,
                marker=dict(color=color, line=dict(color="#22252B", width=1)),
                text=[_fmt_num(t, ".3f") for t in y],
                textposition="outside",
                hovertemplate=f"{label}<br>%{{x}}<br>%{{y:.3f}} s<extra></extra>",
            ),
            row=2,
            col=1,
        )
        for row in _circle_summary_rows(circles):
            table_rows.append({"Run": label, **row})
        runs[run_name] = {
            "event_time_s": event_time_s,
            "LR_asymmetry_s": lr_asymmetry_s,
            "timed_R_s": official.get("R", np.nan),
            "timed_L_s": official.get("L", np.nan),
        }

    _add_table_trace(fig, table_rows, row=1, col=1)
    _apply_dark_layout(fig, "Skidpad event time summary", height=650)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Circle", row=2, col=1)
    fig.update_yaxes(title_text="Lap time [s]", row=2, col=1)

    return fig, {
        "runs": runs,
        "table": pl.DataFrame(table_rows),
        "warnings": warnings,
    }


def lateral_g_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict[str, Any]]:
    """|ay| distribution overlaid with one colour per run."""
    fig = make_dark_figure("Skidpad |ay| distribution", "|ay| [g]", "Time share [%]")

    rows: list[dict[str, object]] = []
    runs: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for idx, (run_name, df) in enumerate(dfs.items()):
        label = _run_label(run_name)
        color = _run_color(run_name)
        circles = classify_circles(df)
        for warning in _classification_warnings(circles):
            warnings.append(f"{label}: {warning}")
        required = ["laps", "Filtering_VN_ay"]
        _require_columns(df, required)
        arr = cols_to_numpy(df, required)
        mask = _lap_mask(arr["laps"], circles)
        ay_g = np.abs(arr["Filtering_VN_ay"][mask]) / G_MPS2
        ay_g = ay_g[np.isfinite(ay_g)]
        if ay_g.size == 0:
            warnings.append(f"{label}: No |ay| samples available for the histogram.")
            runs[run_name] = {"ay_sustained_mean_g": np.nan, "ay_max_global_g": np.nan}
            continue
        fig.add_trace(
            go.Histogram(
                x=ay_g,
                name=label,
                marker_color=color,
                opacity=0.55,
                xbins=dict(start=0.0, end=2.4, size=0.05),
                histnorm="percent",
                hovertemplate=f"{label}<br>%{{x:.2f}} g<br>%{{y:.1f}} %<extra></extra>",
            )
        )
        runs[run_name] = {
            "ay_sustained_mean_g": _safe_mean(ay_g),
            "ay_max_global_g": _safe_max(ay_g),
            "ay_mean_g": _safe_mean(ay_g),
            "ay_p95_g": _safe_percentile(ay_g, 95),
            "ay_std_g": _safe_std(ay_g),
        }
        rows.append(
            {
                "Run": label,
                "ay_mean_g": _round(_safe_mean(ay_g), 3),
                "ay_max_g": _round(_safe_max(ay_g), 3),
                "ay_p95_g": _round(_safe_percentile(ay_g, 95), 3),
                "ay_std_g": _round(_safe_std(ay_g), 3),
                "Samples": int(ay_g.size),
            }
        )

    fig.update_layout(barmode="overlay", height=440)
    fig.update_xaxes(range=[0.0, 2.4])
    return fig, {
        "runs": runs,
        "table": pl.DataFrame(rows),
        "warnings": warnings,
    }


def driven_radius_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict[str, Any]]:
    """Driven-radius distribution overlaid with one colour per run."""
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
    runs: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for idx, (run_name, df) in enumerate(dfs.items()):
        label = _run_label(run_name)
        color = _run_color(run_name)
        circles = classify_circles(df)
        for warning in _classification_warnings(circles):
            warnings.append(f"{label}: {warning}")
        required = ["TimeStamp", "laps", "Filtering_VN_ay", "VN_vx"]
        _require_columns(df, required)
        arr = cols_to_numpy(df, required)
        radius_m = _driven_radius_m(arr["TimeStamp"], arr["VN_vx"], arr["Filtering_VN_ay"])
        sustained = _sustained_radius_mask(arr["VN_vx"], arr["Filtering_VN_ay"], radius_m)
        mask = _lap_mask(arr["laps"], circles) & sustained
        values = radius_m[mask]
        values = values[np.isfinite(values)]
        values = values[(values >= 2.0) & (values <= 20.0)]
        if values.size == 0:
            warnings.append(f"{label}: No sustained-cornering samples for the radius histogram.")
            runs[run_name] = {"radius_error_m": np.nan}
            continue
        fig.add_trace(
            go.Histogram(
                x=values,
                name=label,
                marker_color=color,
                opacity=0.55,
                xbins=dict(start=2.0, end=20.0, size=0.25),
                histnorm="percent",
                hovertemplate=f"{label}<br>%{{x:.2f}} m<br>%{{y:.1f}} %<extra></extra>",
            )
        )
        pct_band = 100.0 * np.nanmean(
            (values >= SKIDPAD_INNER_RADIUS_M) & (values <= SKIDPAD_OUTER_RADIUS_M)
        )
        r_mean = _safe_mean(values)
        runs[run_name] = {
            "R_mean_m": r_mean,
            "R_std_m": _safe_std(values),
            "pct_time_in_band_pct": pct_band,
            "radius_error_m": r_mean - SKIDPAD_IDEAL_RADIUS_M if np.isfinite(r_mean) else np.nan,
        }
        rows.append(
            {
                "Run": label,
                "R_mean_m": _round(r_mean, 3),
                "R_std_m": _round(_safe_std(values), 3),
                "pct_time_in_band_pct": _round(pct_band, 1),
                "radius_error_m": _round(r_mean - SKIDPAD_IDEAL_RADIUS_M, 3),
                "Samples": int(values.size),
            }
        )

    fig.update_layout(barmode="overlay", height=460)
    fig.update_xaxes(range=[2.0, 20.0])
    return fig, {
        "runs": runs,
        "table": pl.DataFrame(rows),
        "warnings": warnings,
    }


def balance_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict[str, Any]]:
    """Front/rear slip-angle balance and understeer angle, one colour per run."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=("Slip angle by axle", "Balance & understeer angle (+ = understeer)"),
    )
    rows: list[dict[str, object]] = []
    runs: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for idx, (run_name, df) in enumerate(dfs.items()):
        label = _run_label(run_name)
        color = _run_color(run_name)
        circles = classify_circles(df)
        for warning in _classification_warnings(circles):
            warnings.append(f"{label}: {warning}")
        required = ["laps", *_SA_COLS]
        _require_columns(df, required)
        arr = cols_to_numpy(df, required)
        front_sa_deg, rear_sa_deg = _front_rear_sa_deg(arr)
        # Skidpad: a whole timed circle is steady-state, so average slip angles
        # over the full timed lap (only dropping non-finite samples).
        good = (
            _lap_mask(arr["laps"], circles) & np.isfinite(front_sa_deg) & np.isfinite(rear_sa_deg)
        )
        try:
            us_mean = _steady_state_understeer(df, circles)["mean_deg"]
        except KeyError as exc:
            warnings.append(f"{label}: {exc}")
            us_mean = np.nan
        if not good.any():
            runs[run_name] = {"balance_index_deg": np.nan, "understeer_angle_mean_deg": us_mean}
            continue
        front_mean = _safe_mean(front_sa_deg[good])
        rear_mean = _safe_mean(rear_sa_deg[good])
        # Positive = understeer: the front axle works at a higher slip angle.
        bal = front_mean - rear_mean
        fig.add_trace(
            go.Bar(
                x=["Front SA", "Rear SA"], y=[front_mean, rear_mean], name=label, marker_color=color
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=["Balance (SA_F-SA_R)", "Understeer angle"],
                y=[bal, us_mean],
                name=label,
                marker_color=color,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        runs[run_name] = {
            "balance_index_deg": bal,
            "understeer_angle_mean_deg": us_mean,
        }
        rows.append(
            {
                "Run": label,
                "SA_F_mean_deg": _round(front_mean, 3),
                "SA_R_mean_deg": _round(rear_mean, 3),
                "SA_F_p95_deg": _round(_safe_percentile(front_sa_deg[good], 95), 3),
                "SA_R_p95_deg": _round(_safe_percentile(rear_sa_deg[good], 95), 3),
                "balance_index_deg": _round(bal, 3),
                "understeer_angle_mean_deg": _round(us_mean, 3),
                "Samples": int(good.sum()),
            }
        )

    fig.add_hline(
        y=0.0, line=dict(color="rgba(235,235,235,0.45)", dash="dash", width=1), row=2, col=1
    )
    _apply_dark_layout(fig, "Skidpad balance", height=620)
    fig.update_yaxes(title_text="Slip angle [deg]", row=1, col=1)
    fig.update_yaxes(title_text="Angle [deg]", row=2, col=1)
    fig.update_layout(barmode="group")

    return fig, {
        "runs": runs,
        "table": pl.DataFrame(rows),
        "warnings": warnings,
    }


def understeer_gradient_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict[str, Any]]:
    """Steady-state understeer angle per circle side, one colour per run.

    On skidpad each timed circle is steady-state cornering, so a deg/g gradient
    is not meaningful (ay is ~constant per circle). Instead we report the
    steady-state understeer angle ``delta_real - delta_ackermann`` averaged over
    each timed circle, per side (positive = understeer).
    """
    fig = make_dark_figure(
        "Understeer angle (steady-state, + = understeer)",
        "Circle",
        "delta_real - delta_ackermann [deg]",
    )
    fig.add_hline(y=0.0, line=dict(color="rgba(235,235,235,0.45)", dash="dash", width=1))

    rows: list[dict[str, object]] = []
    runs: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for idx, (run_name, df) in enumerate(dfs.items()):
        label = _run_label(run_name)
        color = _run_color(run_name)
        circles = classify_circles(df)
        for warning in _classification_warnings(circles):
            warnings.append(f"{label}: {warning}")
        try:
            us = _steady_state_understeer(df, circles)
        except KeyError as exc:
            warnings.append(f"{label}: {exc}")
            runs[run_name] = {
                "understeer_angle_deg": np.nan,
                "understeer_angle_R_deg": np.nan,
                "understeer_angle_L_deg": np.nan,
                "samples": 0,
            }
            continue
        y = [us["R_deg"], us["L_deg"]]
        fig.add_trace(
            go.Bar(
                x=["R circle", "L circle"],
                y=y,
                name=label,
                marker=dict(color=color, line=dict(color="#22252B", width=1)),
                text=[_fmt_num(v, "+.2f") for v in y],
                textposition="outside",
                hovertemplate=f"{label}<br>%{{x}}<br>%{{y:+.2f}} deg<extra></extra>",
            )
        )
        if not np.isfinite(us["mean_deg"]):
            warnings.append(f"{label}: Not enough timed skidpad samples for the understeer angle.")
        runs[run_name] = {
            "understeer_angle_deg": us["mean_deg"],
            "understeer_angle_R_deg": us["R_deg"],
            "understeer_angle_L_deg": us["L_deg"],
            "samples": int(us["n"]),
        }
        rows.append(
            {
                "Run": label,
                "understeer_R_deg": _round(us["R_deg"], 3),
                "understeer_L_deg": _round(us["L_deg"], 3),
                "understeer_mean_deg": _round(us["mean_deg"], 3),
                "Samples": int(us["n"]),
            }
        )

    fig.update_layout(height=460, barmode="group")
    return fig, {
        "runs": runs,
        "table": pl.DataFrame(rows),
        "warnings": warnings,
    }


def tv_intervention_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict[str, Any]]:
    """TV yaw moment versus lateral grip, one colour per run."""
    fig = make_dark_figure(
        "TV intervention versus lateral grip",
        "|ay| [g]",
        "|Mz| [Nm]",
    )

    rows: list[dict[str, object]] = []
    runs: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for idx, (run_name, df) in enumerate(dfs.items()):
        label = _run_label(run_name)
        color = _run_color(run_name)
        circles = classify_circles(df)
        for warning in _classification_warnings(circles):
            warnings.append(f"{label}: {warning}")
        required = ["laps", "Filtering_VN_ay", "VN_vx", *_TV_TORQUE_COLS, "TV_errorYawRate"]
        optional = [col for col in ("TV_desiredYawRate", "TV_actualMz") if col in df.columns]
        _require_columns(df, [*required, *optional])
        arr = cols_to_numpy(df, [*required, *optional])
        mz_applied = _tv_yaw_moment_from_torque(arr)
        yaw_err = arr["TV_errorYawRate"]
        ay_g = arr["Filtering_VN_ay"] / G_MPS2
        sustained = _sustained_balance_mask(arr["VN_vx"], arr["Filtering_VN_ay"]) & _lap_mask(
            arr["laps"], circles
        )

        good_mz = sustained & np.isfinite(mz_applied) & np.isfinite(ay_g)
        if good_mz.any():
            fig.add_trace(
                go.Scattergl(
                    x=np.abs(ay_g[good_mz]),
                    y=np.abs(mz_applied[good_mz]),
                    mode="markers",
                    name=label,
                    marker=dict(color=color, size=4, opacity=0.55),
                    hovertemplate=f"{label}<br>|ay|=%{{x:.2f}} g<br>|Mz|=%{{y:.1f}} Nm<extra></extra>",
                )
            )
        else:
            warnings.append(f"{label}: No sustained-cornering TV samples to plot.")
        yaw_good = sustained & np.isfinite(yaw_err)
        side_mz: dict[str, float] = {}
        for side in ("R", "L"):
            side_mask = good_mz & np.isin(
                arr["laps"], np.asarray(_circle_laps_for_side(circles, side), dtype=float)
            )
            side_mz[side] = _safe_mean(mz_applied[side_mask])
        runs[run_name] = {
            "Mz_mean_Nm": _safe_mean(mz_applied[good_mz]),
            "yaw_err_rms_rad_s": _rms(yaw_err[yaw_good]),
            "Mz_R_mean_Nm": side_mz["R"],
            "Mz_L_mean_Nm": side_mz["L"],
        }
        rows.append(
            {
                "Run": label,
                "Mz_mean_Nm": _round(_safe_mean(mz_applied[good_mz]), 2),
                "Mz_abs_mean_Nm": _round(_safe_mean(np.abs(mz_applied[good_mz])), 2),
                "yaw_err_rms_rad_s": _round(_rms(yaw_err[yaw_good]), 4),
                "Samples": int(good_mz.sum()),
            }
        )

    fig.update_layout(height=480)
    return fig, {
        "runs": runs,
        "table": pl.DataFrame(rows),
        "warnings": warnings,
    }


def lateral_load_dist_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict[str, Any]]:
    """Measured lateral load-transfer distribution, one bar per run."""
    fig = make_dark_figure("Lateral load-transfer distribution", "Run", "Front LLTD [%]")

    labels: list[str] = []
    colors: list[str] = []
    lltd_values: list[float] = []
    rows: list[dict[str, object]] = []
    runs: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for idx, (run_name, df) in enumerate(dfs.items()):
        label = _run_label(run_name)
        color = _run_color(run_name)
        circles = classify_circles(df)
        for warning in _classification_warnings(circles):
            warnings.append(f"{label}: {warning}")
        required = ["laps", "Filtering_VN_ay", *_FZ_COLS]
        _require_columns(df, required)
        arr = cols_to_numpy(df, required)

        run_lltd: list[np.ndarray] = []
        for side in ("R", "L"):
            side_mask = np.isin(
                arr["laps"], np.asarray(_circle_laps_for_side(circles, side), dtype=float)
            )
            front_transfer, rear_transfer = _axle_load_transfer(side, arr, side_mask)
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
            if lltd.size:
                run_lltd.append(lltd)

        lltd_all = np.concatenate(run_lltd) if run_lltd else np.array([], dtype=float)
        mean_lltd = _safe_mean(lltd_all)
        labels.append(label)
        colors.append(color)
        lltd_values.append(mean_lltd)
        runs[run_name] = {
            "LLTD_meas_pct": mean_lltd,
            "LLTD_theory_pct": 100.0 * LLTD_THEORY,
            "delta_pct": mean_lltd - 100.0 * LLTD_THEORY if np.isfinite(mean_lltd) else np.nan,
        }
        rows.append(
            {
                "Run": label,
                "LLTD_meas_pct": _round(mean_lltd, 2),
                "LLTD_theory_pct": _round(100.0 * LLTD_THEORY, 2),
                "delta_pct": _round(mean_lltd - 100.0 * LLTD_THEORY, 2),
                "Samples": int(lltd_all.size),
            }
        )

    fig.add_trace(
        go.Bar(
            x=labels,
            y=lltd_values,
            marker=dict(color=colors, line=dict(color="#22252B", width=1)),
            text=[_fmt_num(v, ".1f") for v in lltd_values],
            textposition="outside",
            hovertemplate="%{x}<br>LLTD=%{y:.1f}%<extra></extra>",
            name="Measured",
            showlegend=False,
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
    return fig, {
        "runs": runs,
        "table": pl.DataFrame(rows),
        "warnings": warnings,
    }


def _lr_asymmetry_rows(
    df: pl.DataFrame, circles: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, object]], list[str]]:
    """Right-minus-left asymmetry metric rows for a single run."""
    warnings: list[str] = []
    rows: list[dict[str, object]] = []

    def add_metric(name: str, right: float, left: float, unit: str, decimals: int = 3) -> None:
        delta = right - left if np.isfinite(right) and np.isfinite(left) else np.nan
        rows.append(
            {
                "Metric": name,
                "R": _round(right, decimals),
                "L": _round(left, decimals),
                "R-L": _round(delta, decimals),
                "Unit": unit,
            }
        )

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
            side: _side_mean_abs(
                np.rad2deg(arr_core["Steering"]), arr_core["laps"], laps
            )  # Steering-pot [rad]→deg; no STEERING_RATIO
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
        us = _steady_state_understeer(df, circles, side_laps=side_laps)
        add_metric("understeer_angle", us["R_deg"], us["L_deg"], "deg", 3)
    except Exception as exc:
        warnings.append(f"Understeer-angle asymmetry unavailable: {exc}")

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

    return rows, warnings


def lr_asymmetry_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict[str, Any]]:
    """Right-minus-left skidpad asymmetry, grouped horizontal bars per run."""
    fig = make_dark_figure("Right-left skidpad asymmetry", "R - L difference", "Metric")
    fig.add_vline(x=0.0, line=dict(color="rgba(235,235,235,0.45)", dash="dash", width=1))

    table_rows: list[dict[str, object]] = []
    warnings: list[str] = []
    max_metrics = 1
    for idx, (run_name, df) in enumerate(dfs.items()):
        label = _run_label(run_name)
        color = _run_color(run_name)
        circles = classify_circles(df)
        for warning in _classification_warnings(circles):
            warnings.append(f"{label}: {warning}")
        rows, run_warnings = _lr_asymmetry_rows(df, circles)
        for warning in run_warnings:
            warnings.append(f"{label}: {warning}")
        for row in rows:
            table_rows.append({"Run": label, **row})
        plot_rows = [row for row in rows if np.isfinite(float(row["R-L"]))]
        max_metrics = max(max_metrics, len(plot_rows))
        if plot_rows:
            deltas = [float(row["R-L"]) for row in plot_rows]
            labels = [f"{row['Metric']} [{row['Unit']}]" for row in plot_rows]
            fig.add_trace(
                go.Bar(
                    x=deltas,
                    y=labels,
                    orientation="h",
                    name=label,
                    marker_color=color,
                    text=[_fmt_num(v, "+.3f") for v in deltas],
                    textposition="outside",
                    hovertemplate=f"{label}<br>%{{y}}<br>R-L=%{{x:+.3f}}<extra></extra>",
                )
            )

    fig.update_layout(height=max(430, 70 * max_metrics), barmode="group")
    return fig, {
        "table": pl.DataFrame(table_rows),
        "warnings": warnings,
    }


def gps_figure8_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict[str, Any]]:
    """Local GPS trajectory with theoretical skidpad circles, one colour per run."""
    fig = make_dark_figure("Skidpad GPS figure-8", "Local X [m]", "Local Y [m]")
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    runs: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    any_traj = False
    for idx, (run_name, df) in enumerate(dfs.items()):
        label = _run_label(run_name)
        color = _run_color(run_name)
        circles = classify_circles(df)
        for warning in _classification_warnings(circles):
            warnings.append(f"{label}: {warning}")
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
            warnings.append(f"{label}: Not enough valid GPS samples for skidpad map.")
            runs[run_name] = {"gps_samples": int(gps_valid.sum()), "centers": {}}
            continue

        x_valid, y_valid = gps_to_local_xy(
            arr["VN_latitude"][gps_valid], arr["VN_longitude"][gps_valid]
        )
        x = np.full(len(df), np.nan, dtype=float)
        y = np.full(len(df), np.nan, dtype=float)
        x[gps_valid] = x_valid
        y[gps_valid] = y_valid
        any_traj = True
        fig.add_trace(
            go.Scattergl(
                x=x[gps_valid],
                y=y[gps_valid],
                mode="markers",
                name=label,
                marker=dict(color=color, size=4, opacity=0.7),
                hovertemplate=f"{label}<br>x=%{{x:.1f}} m<br>y=%{{y:.1f}} m<extra></extra>",
            )
        )

        centers = _skidpad_centers_from_laps(circles, arr["laps"], x, y)
        for side, center in centers.items():
            cx, cy = center
            for radius, dash in (
                (SKIDPAD_INNER_RADIUS_M, "dash"),
                (SKIDPAD_OUTER_RADIUS_M, "dash"),
                (SKIDPAD_IDEAL_RADIUS_M, "dot"),
            ):
                fig.add_trace(
                    go.Scatter(
                        x=cx + radius * np.cos(theta),
                        y=cy + radius * np.sin(theta),
                        mode="lines",
                        name=f"{label} {side}",
                        line=dict(color=color, dash=dash, width=1.0),
                        opacity=0.5,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
        if len(centers) < 2:
            warnings.append(f"{label}: Only one skidpad circle center could be estimated from GPS.")
        runs[run_name] = {
            "gps_samples": int(gps_valid.sum()),
            "centers": {side: (float(c[0]), float(c[1])) for side, c in centers.items()},
        }

    if not any_traj:
        raise ValueError("Not enough valid GPS samples for skidpad map.")
    fig.update_yaxes(scaleanchor="x", scaleratio=1.0)
    fig.update_layout(height=650)
    return fig, {
        "runs": runs,
        "warnings": warnings,
    }


def yaw_rate_vs_ay_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict[str, Any]]:
    """Yaw rate versus lateral acceleration, one colour per run."""
    fig = make_dark_figure("Yaw rate versus lateral acceleration", "ay [g]", "Yaw rate [rad/s]")
    fig.add_hline(y=0.0, line=dict(color="rgba(235,235,235,0.35)", dash="dash", width=1))
    fig.add_vline(x=0.0, line=dict(color="rgba(235,235,235,0.35)", dash="dash", width=1))
    warnings: list[str] = []
    for idx, (run_name, df) in enumerate(dfs.items()):
        label = _run_label(run_name)
        color = _run_color(run_name)
        circles = classify_circles(df)
        for warning in _classification_warnings(circles):
            warnings.append(f"{label}: {warning}")
        yaw_col = first_existing_col(df, _YAW_ALIASES)
        if yaw_col is None:
            raise KeyError("Missing yaw-rate column: VN_gz/AS_yaw_rate")
        required = ["laps", "Filtering_VN_ay", "VN_vx", yaw_col]
        _require_columns(df, required)
        arr = cols_to_numpy(df, required)
        mask = _sustained_balance_mask(arr["VN_vx"], arr["Filtering_VN_ay"]) & _lap_mask(
            arr["laps"], circles
        )
        if not mask.any():
            continue
        fig.add_trace(
            go.Scattergl(
                x=arr["Filtering_VN_ay"][mask] / G_MPS2,
                y=arr[yaw_col][mask],
                mode="markers",
                name=label,
                marker=dict(color=color, size=4, opacity=0.45),
                hovertemplate=f"{label}<br>ay=%{{x:+.2f}} g<br>yaw=%{{y:+.3f}} rad/s<extra></extra>",
            )
        )
    return fig, {"warnings": warnings}


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
        rows.append(
            {
                "Lap": int(lap_id),
                "Side": str(info["side"]),
                "Official": "yes" if info.get("official_timed", False) else "",
                "Laptime [s]": _round(float(info["laptime_s"]), 3),
                "Samples": int(info["n_samples"]),
            }
        )
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
            header=dict(
                values=headers, fill_color="#22252B", font=dict(color=_TEXT, size=11), align="left"
            ),
            cells=dict(
                values=values, fill_color="#171A1F", font=dict(color=_TEXT, size=10), align="left"
            ),
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
    if not any(
        info.get("official_timed", False) and info["side"] == "R" for info in circles.values()
    ):
        warnings.append("No official timed right-hand circle could be identified.")
    if not any(
        info.get("official_timed", False) and info["side"] == "L" for info in circles.values()
    ):
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
        lap_id
        for lap_id, info in circles.items()
        if info["side"] == side and info.get("official_timed", False)
    ]
    if official:
        return official
    timed = [
        lap_id
        for lap_id, info in circles.items()
        if info["side"] == side and info["role"] == "timed"
    ]
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


def _driven_radius_m(time_s: np.ndarray, vx_mps: np.ndarray, ay_mps2: np.ndarray) -> np.ndarray:
    vx = np.asarray(vx_mps, dtype=float)
    ay_abs = np.abs(np.asarray(ay_mps2, dtype=float))
    radius = np.divide(
        vx**2,
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


def _sustained_radius_mask(
    vx_mps: np.ndarray, ay_mps2: np.ndarray, radius_m: np.ndarray
) -> np.ndarray:
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
    front_sa_deg = 0.5 * (np.rad2deg(np.abs(arr["Est_SAFL"])) + np.rad2deg(np.abs(arr["Est_SAFR"])))
    rear_sa_deg = 0.5 * (np.rad2deg(np.abs(arr["Est_SARL"])) + np.rad2deg(np.abs(arr["Est_SARR"])))
    return front_sa_deg, rear_sa_deg


def _understeer_angle_deg(
    steering_rad: np.ndarray, ay_mps2: np.ndarray, vx_mps: np.ndarray
) -> np.ndarray:
    """Steady-state understeer angle [deg]: ``|Steering| - |delta_ackermann|``.

    ``Steering`` is the steering-potentiometer value in radians; the team's
    formula uses it directly (no STEERING_RATIO division). The Ackermann
    reference is acceleration-based, ``delta_ackermann = L * ay / vx^2``.
    Positive = understeer.
    """
    steering = np.abs(np.asarray(steering_rad, dtype=float))
    vx = np.asarray(vx_mps, dtype=float)
    ideal_rad = np.divide(
        WHEELBASE_EQ * np.asarray(ay_mps2, dtype=float),
        vx**2,
        out=np.full_like(vx, np.nan, dtype=float),
        where=np.isfinite(vx) & (np.abs(vx) > 0.5),
    )
    return np.rad2deg(steering - np.abs(ideal_rad))


def _steady_state_understeer(
    df: pl.DataFrame,
    circles: dict[int, dict[str, Any]],
    *,
    side_laps: dict[str, list[int]] | None = None,
) -> dict[str, float]:
    """Steady-state understeer angle over the timed skidpad circles.

    Uses ``|Steering| - |L * ay / vx^2|`` (``Steering`` = steering-potentiometer
    value [rad], used directly; ``ay = Filtering_VN_ay``, ``vx = Est_vxCOG``);
    positive = understeer. A whole timed circle is steady-state cornering, so the angle is
    averaged over the full timed lap, only dropping non-finite samples.
    """
    vx_col = first_existing_col(df, ("Est_vxCOG", "VN_vx"))
    if vx_col is None:
        raise KeyError("Missing speed column: Est_vxCOG/VN_vx")
    required = ["laps", "Steering", "Filtering_VN_ay", vx_col]
    _require_columns(df, required)
    arr = cols_to_numpy(df, required)
    understeer_deg = _understeer_angle_deg(arr["Steering"], arr["Filtering_VN_ay"], arr[vx_col])

    source = (
        side_laps
        if side_laps is not None
        else {side: _circle_laps_for_side(circles, side) for side in ("R", "L")}
    )
    out: dict[str, float] = {"mean_deg": np.nan, "R_deg": np.nan, "L_deg": np.nan, "n": 0}
    total = 0
    for side in ("R", "L"):
        laps = np.asarray(source.get(side, []), dtype=float)
        mask = np.isin(arr["laps"], laps) & np.isfinite(understeer_deg)
        if not mask.any():
            continue
        out[f"{side}_deg"] = _safe_mean(understeer_deg[mask])
        total += int(mask.sum())
    out["mean_deg"] = _safe_mean([v for v in (out["R_deg"], out["L_deg"]) if np.isfinite(v)])
    out["n"] = total
    return out


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
    b = x**2 + y**2
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
    return float(np.sqrt(np.nanmean(arr**2))) if arr.size else np.nan


def _round(value: float, decimals: int) -> float:
    return round(float(value), decimals) if np.isfinite(value) else np.nan


def _fmt_num(value: float, pattern: str) -> str:
    return format(float(value), pattern) if np.isfinite(value) else ""
