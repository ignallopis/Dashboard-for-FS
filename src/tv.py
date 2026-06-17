"""tv.py
------
Torque Vectoring (TV) KPIs — OptimumG/Rouelle-style, all reduced to KPIs over a physical
axis (never distance/time). Two groups:

  A · Does the controller work?      A1 yaw-rate tracking · A2 PI loop health ·
                                     A4 moment authority & utilisation
  B · How does it bias the car?      B1 intended rotation bias · B2 side-slip stability

All computed during cornering (|ay| >= threshold AND |steering| >= threshold).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go

from src import cornering

from utils import (
    cols_to_numpy,
    driver_color,
    ensure_complete_laps_df,
    keep_min_duration_segments,
    lap_dist_from_gps,
    make_dark_figure,
    robust_dt,
    unique_laps,
)


# ── Cornering filter parameters ───────────────────────────────────────────────
AY_THRESHOLD = 2.0
STEERING_THRESHOLD = 0.05
MIN_SPEED = 4.0
MIN_CORNER_DURATION = 0.20

MAX_ROBUST_SLOPE_POINTS = 257

G_MPS2 = 9.81  # [m/s²] gravity, for |Ay| in g
RING_DEADBAND_NM = 30.0  # [Nm] min Mz_fb excursion to count a PI sign change (ignore noise)


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


def _unique_required_cols(columns: list[str]) -> list[str]:
    """Return *columns* without duplicates while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for col in columns:
        if col in seen:
            continue
        seen.add(col)
        out.append(col)
    return out


def _prepare_tv_control_arrays(
    df: pl.DataFrame,
    signal_cols: list[str],
) -> dict[str, np.ndarray]:
    """Prepare complete-lap arrays shared by the TV control figures."""
    df = ensure_complete_laps_df(df)
    ay_col = _ay_signal(df.columns)
    vx_col = _vx_signal(df.columns)
    cols = _unique_required_cols(
        ["TimeStamp", "laps", "laptime", "Steering", ay_col, vx_col, *signal_cols]
    )
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing TV control columns: {missing}")

    arr = cols_to_numpy(df, cols)
    dist_m = lap_dist_from_gps(df)
    valid_keys = [col for col in cols if col in arr]
    valid = np.all(
        np.stack([np.isfinite(arr[col]) for col in valid_keys], axis=1), axis=1
    ) & np.isfinite(dist_m)
    arr = {col: values[valid] for col, values in arr.items()}
    dist_m = dist_m[valid]
    if arr["TimeStamp"].size < 2:
        raise ValueError("Not enough valid TV control samples.")

    time_s = arr["TimeStamp"] - arr["TimeStamp"][0]
    dt = robust_dt(time_s)
    ay = arr[ay_col]
    vx = arr[vx_col]
    corner_mask = _corner_mask(ay, arr["Steering"], vx, dt)
    arr.update(
        {
            "time_s": time_s,
            "dist_m": dist_m,
            "ay": ay,
            "vx": vx,
            "corner_mask": corner_mask,
        }
    )
    return arr


def _binned_percentile(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bin_width: float,
    x_min: float,
    x_max: float,
    pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile of *y* in fixed *x* bins (median+band idiom, ≥5 samples/bin)."""
    edges = np.arange(x_min, x_max + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.full_like(centers, np.nan, dtype=float)
    counts = np.zeros_like(centers, dtype=int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (x >= lo) & (x < hi) & np.isfinite(y)
        counts[i] = int(m.sum())
        if counts[i] >= 5:
            values[i] = float(np.nanpercentile(y[m], pct))
    return centers, values, counts


def _beta_deg(vy: np.ndarray, vx: np.ndarray) -> np.ndarray:
    """Side-slip angle [deg] from COG velocity estimates (estimator output)."""
    return np.rad2deg(np.arctan2(vy, vx))


def _annotate_empty(fig: go.Figure) -> None:
    """Centre 'no data' note on a figure that ended up with no run traces."""
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        text="No valid TV corner samples",
        font=dict(color="#EBEBEB", size=12),
    )


def _r2(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    """Coefficient of determination of a linear fit y≈slope·x+intercept."""
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return np.nan
    yhat = slope * x + intercept
    ss_res = float(np.nansum((y - yhat) ** 2))
    ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
    if ss_tot <= 0.0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def tv_yaw_tracking_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """A1: does the car follow TV's own yaw-rate target? (all runs overlaid)

    Real yaw (VN_gz) vs TV target (TV_desiredYawRate) over cornering samples, with the
    y=x perfect-tracking line. RMSE is the tracking error; the robust slope (ideal ≈1)
    flags systematic over/under-rotation relative to what the controller asks.
    """
    fig = make_dark_figure(
        "TV yaw-rate tracking  ·  real vs target",
        "TV target yaw rate [rad/s]",
        "Real yaw rate [rad/s]",
    )
    runs: dict[str, dict] = {}
    warnings: list[str] = []
    lo_all: list[float] = []
    hi_all: list[float] = []
    for run_name, df in dfs.items():
        try:
            arr = _prepare_tv_control_arrays(df, ["TV_desiredYawRate", "VN_gz"])
        except Exception as exc:
            warnings.append(f"{run_name}: {exc}")
            continue
        desired = arr["TV_desiredYawRate"]
        real = arr["VN_gz"]
        cm = arr["corner_mask"] & np.isfinite(real) & np.isfinite(desired)
        if not cm.any():
            warnings.append(f"{run_name}: no valid TV corner samples for yaw tracking.")
            continue
        x = desired[cm]
        y = real[cm]
        color = driver_color(run_name)
        stride = max(1, int(np.ceil(x.size / 6000)))
        fig.add_trace(
            go.Scattergl(
                x=x[::stride],
                y=y[::stride],
                mode="markers",
                marker=dict(color=color, size=3, opacity=0.15),
                name=f"{run_name} samples",
                legendgroup=run_name,
                showlegend=False,
                hovertemplate=f"{run_name}<br>target=%{{x:.3f}}<br>real=%{{y:.3f}} rad/s<extra></extra>",
            )
        )
        slope, intercept = _robust_slope(x, y)
        r2 = _r2(x, y, slope, intercept)
        lo = float(min(np.nanmin(x), np.nanmin(y)))
        hi = float(max(np.nanmax(x), np.nanmax(y)))
        lo_all.append(lo)
        hi_all.append(hi)
        if np.isfinite(slope):
            fig.add_trace(
                go.Scatter(
                    x=[lo, hi],
                    y=[slope * lo + intercept, slope * hi + intercept],
                    mode="lines",
                    line=dict(color=color, width=2.6),
                    legendgroup=run_name,
                    name=f"{run_name} (slope={slope:.2f})",
                )
            )
        runs[run_name] = {
            "tracking_rmse": float(np.sqrt(np.nanmean((y - x) ** 2))),
            "tracking_slope": float(slope),
            "tracking_r2": float(r2),
            "corner_samples": int(cm.sum()),
        }

    if lo_all:
        lo = min(lo_all)
        hi = max(hi_all)
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(color="#73D973", dash="dash", width=2.0),
                name="perfect tracking (y=x)",
            )
        )
    else:
        _annotate_empty(fig)
    return fig, {"runs": runs, "warnings": warnings}


def _ringing_rate(mz_fb: np.ndarray, cm: np.ndarray, dt: float, deadband: float) -> float:
    """Sign changes of the feedback moment per second of cornering, jitter-free.

    Counted **within each contiguous corner segment** (so the gaps between separate
    corners never create a spurious flip). Only significant samples (|Mz_fb| >= deadband)
    are kept; a count is added each time consecutive significant samples flip sign. The
    total is normalised by the corner time spanned (n_corner · dt).
    """
    n = int(cm.sum())
    if n < 2 or not np.isfinite(dt) or dt <= 0.0:
        return np.nan
    idx = np.where(cm)[0]
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    flips = 0
    for seg in np.split(idx, breaks):
        if seg.size < 2:
            continue
        seg_vals = mz_fb[seg]
        signs = np.sign(seg_vals[np.abs(seg_vals) >= deadband])
        if signs.size >= 2:
            flips += int(np.count_nonzero(np.diff(signs) != 0))
    return flips / (n * dt)


def tv_pi_loop_health_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """A2: is the PI feedback loop healthy — right effective gain, no ringing? (overlaid)

    Feedback moment TV_feedBackMz vs yaw error TV_errorYawRate in corners: the robust
    slope is the effective loop response (it mixes P and the I term, so it is NOT pure
    Kp), and ringing_rate counts how often Mz_fb flips sign per second (loop chatter).
    """
    fig = make_dark_figure(
        "TV PI loop health  ·  feedback vs error",
        "Yaw error [rad/s]",
        "Feedback moment Mz_fb [Nm]",
    )
    runs: dict[str, dict] = {}
    warnings: list[str] = []
    any_run = False
    for run_name, df in dfs.items():
        try:
            arr = _prepare_tv_control_arrays(df, ["TV_feedBackMz", "TV_errorYawRate"])
        except Exception as exc:
            warnings.append(f"{run_name}: {exc}")
            continue
        mz_fb = arr["TV_feedBackMz"]
        err = arr["TV_errorYawRate"]
        dt = robust_dt(arr["time_s"])
        cm = arr["corner_mask"] & np.isfinite(mz_fb) & np.isfinite(err)
        if not cm.any():
            warnings.append(f"{run_name}: no valid TV corner samples for PI loop health.")
            continue
        any_run = True
        x = err[cm]
        y = mz_fb[cm]
        color = driver_color(run_name)
        stride = max(1, int(np.ceil(x.size / 6000)))
        fig.add_trace(
            go.Scattergl(
                x=x[::stride],
                y=y[::stride],
                mode="markers",
                marker=dict(color=color, size=3, opacity=0.15),
                name=f"{run_name} samples",
                legendgroup=run_name,
                showlegend=False,
                hovertemplate=f"{run_name}<br>error=%{{x:.3f}} rad/s<br>Mz_fb=%{{y:.0f}} Nm<extra></extra>",
            )
        )
        slope, intercept = _robust_slope(x, y)
        r2 = _r2(x, y, slope, intercept)
        if np.isfinite(slope):
            lo = float(np.nanmin(x))
            hi = float(np.nanmax(x))
            fig.add_trace(
                go.Scatter(
                    x=[lo, hi],
                    y=[slope * lo + intercept, slope * hi + intercept],
                    mode="lines",
                    line=dict(color=color, width=2.6),
                    legendgroup=run_name,
                    name=f"{run_name} (gain={slope:.0f})",
                )
            )
        runs[run_name] = {
            "effective_gain": float(slope),
            "gain_r2": float(r2),
            "ringing_rate": _ringing_rate(mz_fb, cm, dt, RING_DEADBAND_NM),
            "corner_samples": int(cm.sum()),
        }

    if any_run:
        fig.add_hline(y=0.0, line=dict(color="#9AA0A6", dash="dash", width=1.2))
    else:
        _annotate_empty(fig)
    return fig, {"runs": runs, "warnings": warnings}


def tv_authority_utilisation_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """A4: how close to its moment limit does the TV run (does it run out of authority)?

    Utilisation = |TV_desiredMz| / |TV_limitMz| (1.0 = saturated) binned vs |Ay|. The p95
    curve (solid) shows the worst-case demand, the median (dotted) the typical demand;
    mz_tracking_rmse is how far the QP-delivered moment (TV_actualMz, a Bz·T reconstruction)
    falls from the request. All runs overlaid by colour.
    """
    fig = make_dark_figure(
        "TV moment authority  ·  utilisation vs lateral g",
        "|Ay| [g]",
        "Moment utilisation |Mz| / |Mz limit|",
    )
    runs: dict[str, dict] = {}
    warnings: list[str] = []
    any_run = False
    for run_name, df in dfs.items():
        try:
            arr = _prepare_tv_control_arrays(df, ["TV_desiredMz", "TV_limitMz", "TV_actualMz"])
        except Exception as exc:
            warnings.append(f"{run_name}: {exc}")
            continue
        desired = np.abs(arr["TV_desiredMz"])
        limit = np.abs(arr["TV_limitMz"])
        actual = arr["TV_actualMz"]
        ay_g = np.abs(arr["ay"]) / G_MPS2
        cm = arr["corner_mask"] & np.isfinite(desired) & (limit > 1.0) & np.isfinite(ay_g)
        if not cm.any():
            warnings.append(f"{run_name}: no valid TV corner samples for authority.")
            continue
        any_run = True
        util = desired[cm] / limit[cm]
        x = ay_g[cm]
        color = driver_color(run_name)
        stride = max(1, int(np.ceil(x.size / 6000)))
        fig.add_trace(
            go.Scattergl(
                x=x[::stride],
                y=util[::stride],
                mode="markers",
                marker=dict(color=color, size=3, opacity=0.08),
                name=f"{run_name} samples",
                legendgroup=run_name,
                showlegend=False,
                hovertemplate=f"{run_name}<br>|Ay|=%{{x:.2f}} g<br>util=%{{y:.2f}}<extra></extra>",
            )
        )
        centers, p50, _ = _binned_percentile(x, util, bin_width=0.1, x_min=0.0, x_max=1.7, pct=50.0)
        _, p95, _ = _binned_percentile(x, util, bin_width=0.1, x_min=0.0, x_max=1.7, pct=95.0)
        valid = np.isfinite(p50)
        fig.add_trace(
            go.Scatter(
                x=centers[valid],
                y=p95[valid],
                mode="lines+markers",
                line=dict(color=color, width=2.6),
                legendgroup=run_name,
                name=f"{run_name} p95",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=centers[valid],
                y=p50[valid],
                mode="lines",
                line=dict(color=color, width=1.6, dash="dot"),
                legendgroup=run_name,
                showlegend=False,
                name=f"{run_name} median",
            )
        )
        runs[run_name] = {
            "util_p95": float(np.nanpercentile(util, 95.0)),
            "mz_tracking_rmse": float(
                np.sqrt(np.nanmean((actual[cm] - arr["TV_desiredMz"][cm]) ** 2))
            ),
            "corner_samples": int(cm.sum()),
        }

    if any_run:
        fig.add_hline(
            y=1.0,
            line=dict(color="#E5564E", dash="dash", width=2.0),
            annotation_text="moment limit",
            annotation_position="top left",
        )
    else:
        _annotate_empty(fig)
    return fig, {"runs": runs, "warnings": warnings}


def _robust_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return np.nan, np.nan
    x_valid = x[valid]
    y_valid = y[valid]
    order = np.argsort(x_valid)
    x_valid = x_valid[order]
    y_valid = y_valid[order]
    if x_valid.size > MAX_ROBUST_SLOPE_POINTS:
        idx = np.linspace(0, x_valid.size - 1, MAX_ROBUST_SLOPE_POINTS).astype(int)
        x_valid = x_valid[idx]
        y_valid = y_valid[idx]

    dx = x_valid[None, :] - x_valid[:, None]
    dy = y_valid[None, :] - y_valid[:, None]
    upper = np.triu(np.ones_like(dx, dtype=bool), k=1) & (np.abs(dx) > 1e-9)
    slopes = dy[upper] / dx[upper]
    slopes = slopes[np.isfinite(slopes)]
    if slopes.size == 0:
        return np.nan, np.nan
    slope = float(np.nanmedian(slopes))
    intercept = float(np.nanmedian(y_valid - slope * x_valid))
    return slope, intercept


YAW_BALANCE_MIN_EXPECTED_RADPS = 0.15


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


def _reference_turns(
    df: pl.DataFrame,
    reference_turns: list[cornering.TurnDef] | None,
    reference_label: str,
) -> tuple[list[cornering.TurnDef], str]:
    """Resolve the geometry-lap reference turns (fastest valid lap) if not provided."""
    if reference_turns is not None:
        return reference_turns, reference_label
    d_ref = cornering.compute_radius_curvature(df)
    lap_ids = unique_laps(d_ref["laps"])
    valid_laps = (
        lap_ids[(lap_ids > 0) & (lap_ids != np.nanmax(lap_ids))] if lap_ids.size else np.array([])
    )
    if valid_laps.size == 0:
        return [], reference_label
    best_lap = min(
        valid_laps,
        key=lambda lap: float(np.nanmax(d_ref["laptime"][d_ref["laps"] == lap])),
    )
    turns = cornering.detect_turns_on_lap(d_ref, "", int(best_lap))
    return turns, reference_label or f"lap {int(best_lap)}"


def _balance_pct(yaw: np.ndarray, path_yaw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """US/OS balance [%] = 100·(yaw/path_yaw − 1), clipped to a sane gain range."""
    gain = np.divide(yaw, path_yaw, out=np.full_like(yaw, np.nan, dtype=float), where=mask)
    gain = np.where((gain >= -1.0) & (gain <= 3.0), gain, np.nan)
    return (gain - 1.0) * 100.0


def _intended_balance_arrays(
    df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-run intended-balance arrays: (intended%, |Ay| g, turn_id, eval mask) or None."""
    df = ensure_complete_laps_df(df)
    required = ["TimeStamp", "laps", "laptime", "VN_gz", "TV_desiredYawRate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing TV balance columns: {missing}")

    reference_turns, _ = _reference_turns(df, None, "")
    radius, corner_mask, turn_id = _turn_mask_from_reference(df, reference_turns)
    if len(radius["time_s"]) != len(df):
        raise ValueError("Radius-based corner arrays do not match TV telemetry length.")

    yaw_real = df["VN_gz"].to_numpy().astype(float)
    yaw_intended = df["TV_desiredYawRate"].to_numpy().astype(float)
    vx = radius["vx_mps"]
    path_yaw = vx * radius["signed_curvature_smooth_inv_m"]
    ay_g = np.abs(radius["ay_abs_smooth_mps2"]) / G_MPS2

    base = (
        corner_mask
        & np.isfinite(yaw_real)
        & np.isfinite(path_yaw)
        & (np.abs(path_yaw) >= YAW_BALANCE_MIN_EXPECTED_RADPS)
        & np.isfinite(vx)
        & (np.abs(vx) >= MIN_SPEED)
    )
    # Sign-align the path reference to the car's actual rotation sense (the measured yaw
    # reliably shares the corner's sign), then apply to the intended target.
    if base.any():
        alignment = float(np.nanmedian(yaw_real[base] * path_yaw[base]))
        if np.isfinite(alignment) and alignment < 0.0:
            path_yaw = -path_yaw

    eval_mask = base & np.isfinite(yaw_intended)
    intended = _balance_pct(yaw_intended, path_yaw, eval_mask)
    m = eval_mask & np.isfinite(intended) & np.isfinite(ay_g)
    if not m.any():
        return None
    return intended, ay_g, turn_id, m


def tv_intended_balance_figs_kpis(dfs: dict[str, pl.DataFrame]) -> tuple[list[go.Figure], dict]:
    """B1: the US/OS balance the TV is tuned to *intend* across the cornering envelope.

    Intended balance [%] = 100·(TV_desiredYawRate / path_yaw − 1), where path_yaw =
    vx·curvature is the neutral geometric reference. >0 means the TV asks for more rotation
    than the path demands (oversteer-side bias); <0 means it pulls the car toward understeer.
    Shown vs |Ay| (how the bias scales with corner load) and per numbered corner, all runs
    overlaid by colour.

    Honest scope: this is the TV's *intended* effect on balance. Whether the car realises
    it cannot be proven without a TV-off baseline; tracking of that intent is A1.
    """
    fig_env = make_dark_figure(
        "TV intended rotation bias  ·  vs lateral g",
        "|Ay| [g]",
        "Intended balance [%]   (>0 oversteer-side · <0 understeer-side)",
    )
    fig_corner = make_dark_figure(
        "TV intended bias per corner",
        "Corner",
        "Intended balance [%]",
    )
    runs: dict[str, dict] = {}
    warnings: list[str] = []
    per_run_turns: dict[str, dict[int, float]] = {}
    for run_name, df in dfs.items():
        try:
            res = _intended_balance_arrays(df)
        except Exception as exc:
            warnings.append(f"{run_name}: {exc}")
            continue
        if res is None:
            warnings.append(f"{run_name}: no valid TV corner samples for intended balance.")
            continue
        intended, ay_g, turn_id, m = res
        color = driver_color(run_name)
        centers, p_int, _ = _binned_percentile(
            ay_g[m], intended[m], bin_width=0.1, x_min=0.0, x_max=1.7, pct=50.0
        )
        valid = np.isfinite(p_int)
        fig_env.add_trace(
            go.Scatter(
                x=centers[valid],
                y=p_int[valid],
                mode="lines+markers",
                line=dict(color=color, width=2.8),
                legendgroup=run_name,
                name=run_name,
                hovertemplate=f"{run_name}<br>|Ay|=%{{x:.2f}} g<br>intended=%{{y:.1f}}%<extra></extra>",
            )
        )
        tdict: dict[int, float] = {}
        for tid in sorted({int(t) for t in turn_id[m & np.isfinite(turn_id)]}):
            tm = m & (turn_id == float(tid))
            if int(tm.sum()) >= 10:
                tdict[tid] = float(np.nanmedian(intended[tm]))
        per_run_turns[run_name] = tdict
        median_int = float(np.nanmedian(intended[m]))
        curve = p_int[valid]
        peak_intended = float(curve[int(np.nanargmax(np.abs(curve)))]) if curve.size else np.nan
        runs[run_name] = {
            "median_intended_balance": median_int,
            "peak_intended_balance": peak_intended,
            "balance_sign": "OS" if median_int > 0 else "US",
            "corner_samples": int(m.sum()),
        }

    if runs:
        fig_env.add_hline(
            y=0.0,
            line=dict(color="#9AA0A6", dash="dash", width=1.5),
            annotation_text="neutral (path)",
            annotation_position="bottom left",
        )
        all_tids = sorted({t for d in per_run_turns.values() for t in d})
        labels = [f"T{t}" for t in all_tids]
        for run_name, tdict in per_run_turns.items():
            fig_corner.add_trace(
                go.Bar(
                    x=labels,
                    y=[tdict.get(t) for t in all_tids],
                    marker_color=driver_color(run_name),
                    name=run_name,
                    hovertemplate=f"{run_name}<br>%{{x}}<br>intended=%{{y:.1f}}%<extra></extra>",
                )
            )
        fig_corner.update_layout(barmode="group")
        fig_corner.add_hline(y=0.0, line=dict(color="#9AA0A6", dash="dash", width=1.5))
    else:
        _annotate_empty(fig_env)
        _annotate_empty(fig_corner)
    return [fig_env, fig_corner], {"runs": runs, "warnings": warnings}


def tv_sideslip_stability_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """B2: how planted the car stays — side-slip angle β across the cornering envelope.

    β = atan2(Est_vyCOG, Est_vxCOG) [deg] (an estimator output, not measured). Its p95 vs
    |Ay| shows how much the rear steps out as lateral load builds; a steep rise toward high
    g means the car is getting loose (stability margin shrinking). All runs overlaid.
    """
    fig = make_dark_figure(
        "TV side-slip stability  ·  |β| vs lateral g",
        "|Ay| [g]",
        "|β| side-slip angle [deg]",
    )
    runs: dict[str, dict] = {}
    warnings: list[str] = []
    any_run = False
    for run_name, df in dfs.items():
        try:
            arr = _prepare_tv_control_arrays(df, ["Est_vyCOG", "Est_vxCOG"])
        except Exception as exc:
            warnings.append(f"{run_name}: {exc}")
            continue
        beta = np.abs(_beta_deg(arr["Est_vyCOG"], arr["Est_vxCOG"]))
        ay_g = np.abs(arr["ay"]) / G_MPS2
        cm = arr["corner_mask"] & np.isfinite(beta) & np.isfinite(ay_g)
        if not cm.any():
            warnings.append(f"{run_name}: no valid TV corner samples for side-slip.")
            continue
        any_run = True
        x = ay_g[cm]
        y = beta[cm]
        color = driver_color(run_name)
        stride = max(1, int(np.ceil(x.size / 6000)))
        fig.add_trace(
            go.Scattergl(
                x=x[::stride],
                y=y[::stride],
                mode="markers",
                marker=dict(color=color, size=3, opacity=0.08),
                name=f"{run_name} samples",
                legendgroup=run_name,
                showlegend=False,
                hovertemplate=f"{run_name}<br>|Ay|=%{{x:.2f}} g<br>|β|=%{{y:.2f}}°<extra></extra>",
            )
        )
        centers, p95, _ = _binned_percentile(x, y, bin_width=0.1, x_min=0.0, x_max=1.7, pct=95.0)
        valid = np.isfinite(p95)
        fig.add_trace(
            go.Scatter(
                x=centers[valid],
                y=p95[valid],
                mode="lines+markers",
                line=dict(color=color, width=2.8),
                legendgroup=run_name,
                name=f"{run_name} p95 |β|",
            )
        )
        peak_bins = np.where(valid)[0]
        runs[run_name] = {
            "beta_p95_deg": float(np.nanpercentile(y, 95.0)),
            "beta_peak_g": float(p95[peak_bins[-1]]) if peak_bins.size else np.nan,
            "corner_samples": int(cm.sum()),
        }

    if not any_run:
        _annotate_empty(fig)
    return fig, {"runs": runs, "warnings": warnings}
