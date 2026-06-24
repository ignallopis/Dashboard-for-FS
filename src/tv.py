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

from utils import (
    cols_to_numpy,
    driver_color,
    ensure_complete_laps_df,
    keep_min_duration_segments,
    lap_dist_from_gps,
    make_dark_figure,
    robust_dt,
)


# ── Cornering filter parameters ───────────────────────────────────────────────
AY_THRESHOLD = 2.0
STEERING_THRESHOLD = 0.05
MIN_SPEED = 4.0
MIN_CORNER_DURATION = 0.20

MAX_ROBUST_SLOPE_POINTS = 257

G_MPS2 = 9.81  # [m/s²] gravity, for |Ay| in g
RING_DEADBAND_NM = 30.0  # [Nm] min Mz_fb excursion to count a PI sign change (ignore noise)
MZ_DELIVERY_MIN_NM = 50.0  # [Nm] min |desired Mz| to score delivery ratio (avoid ÷~0 near 0)


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


def _tv_laps_df(df: pl.DataFrame) -> pl.DataFrame:
    """Valid-lap samples for the sample-based TV figures, single-lap tolerant.

    Multi-lap events (skidpad, endurance) use the standard complete-laps filter (drop lap
    0 and the last lap). Single-lap events (autocross, acceleration) keep their one timed
    lap so the section still renders — these figures scatter/bin over samples, not
    lap-distance-aligned traces, so they don't need ≥2 laps.
    """
    if "laps" in df.columns and "laptime" in df.columns:
        valid = df.filter((pl.col("laps") > 0) & pl.col("laptime").is_not_nan())
        if not valid.is_empty() and valid["laps"].n_unique() < 2:
            return valid
    return ensure_complete_laps_df(df)


def _prepare_tv_control_arrays(
    df: pl.DataFrame,
    signal_cols: list[str],
) -> dict[str, np.ndarray]:
    """Prepare valid-lap arrays shared by the TV control figures."""
    df = _tv_laps_df(df)
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
        height=520,
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
        height=520,
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


def tv_feedforward_share_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """#4: how much of the yaw-moment command is anticipation (FF) vs correction (FB)?

    FF share = |TV_feedForwardMz| / (|TV_feedForwardMz| + |TV_feedBackMz|), a magnitude
    share in [0,1] (0.5 = FF and FB do equal work), binned by |Steering| over cornering
    samples. A well-tuned TV does most of the work in feedforward (share high, flat); a
    share that collapses as steering grows means the FF map runs out of authority at the
    limit and the PI loop is left to correct. All runs overlaid by colour.
    """
    fig = make_dark_figure(
        "TV feedforward share  ·  anticipation vs correction",
        "|Steering| [rad]",
        "FF share  |Mz_ff| / (|Mz_ff| + |Mz_fb|)",
        height=520,
    )
    runs: dict[str, dict] = {}
    warnings: list[str] = []
    any_run = False
    for run_name, df in dfs.items():
        try:
            arr = _prepare_tv_control_arrays(df, ["TV_feedForwardMz", "TV_feedBackMz"])
        except Exception as exc:
            warnings.append(f"{run_name}: {exc}")
            continue
        ff = arr["TV_feedForwardMz"]
        fb = arr["TV_feedBackMz"]
        denom = np.abs(ff) + np.abs(fb)
        share = np.divide(
            np.abs(ff), denom, out=np.full_like(ff, np.nan, dtype=float), where=denom > 1.0
        )
        steer = np.abs(arr["Steering"])
        cm = arr["corner_mask"] & np.isfinite(share) & np.isfinite(steer)
        if not cm.any():
            warnings.append(f"{run_name}: no valid TV corner samples for feedforward share.")
            continue
        any_run = True
        x = steer[cm]
        y = share[cm]
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
                hovertemplate=f"{run_name}<br>|steer|=%{{x:.3f}} rad<br>FF share=%{{y:.2f}}<extra></extra>",
            )
        )
        centers, p50, _ = _binned_percentile(x, y, bin_width=0.05, x_min=0.0, x_max=0.5, pct=50.0)
        valid = np.isfinite(p50)
        fig.add_trace(
            go.Scatter(
                x=centers[valid],
                y=p50[valid],
                mode="lines+markers",
                line=dict(color=color, width=2.8),
                legendgroup=run_name,
                name=f"{run_name} median",
            )
        )
        runs[run_name] = {
            "median_ff_share": float(np.nanmedian(y)),
            "fb_led_pct": float(100.0 * np.nanmean(y < 0.5)),
            "corner_samples": int(cm.sum()),
        }

    if any_run:
        fig.add_hline(
            y=0.5,
            line=dict(color="#9AA0A6", dash="dash", width=1.5),
            annotation_text="FF = FB",
            annotation_position="bottom left",
        )
        fig.update_yaxes(range=[0.0, 1.0])
    else:
        _annotate_empty(fig)
    return fig, {"runs": runs, "warnings": warnings}


def tv_authority_utilisation_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """#2: how much yaw-moment authority does the TV use, and does it deliver it?

    Utilisation = |TV_desiredMz| / |TV_limitMz| (1.0 = saturated) binned vs |Ay|. The p95
    curve (solid) shows the worst-case demand, the median (dotted) the typical demand;
    delivery_ratio_p50 = median |TV_actualMz| / |TV_desiredMz| (a Bz·T reconstruction of the
    moment the wheels actually made) tells whether the allocator delivers the commanded
    moment (≈1 = faithful). All runs overlaid by colour.
    """
    fig = make_dark_figure(
        "TV moment authority  ·  utilisation vs lateral g",
        "|Ay| [g]",
        "Moment utilisation |Mz| / |Mz limit|",
        height=520,
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
                name=f"{run_name} p95 (worst-case)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=centers[valid],
                y=p50[valid],
                mode="lines",
                line=dict(color=color, width=1.6, dash="dot"),
                legendgroup=run_name,
                name=f"{run_name} median (typical)",
            )
        )
        des_cm = desired[cm]
        big = des_cm > MZ_DELIVERY_MIN_NM
        delivery_p50 = (
            float(np.nanmedian(np.abs(actual[cm][big]) / des_cm[big])) if big.any() else np.nan
        )
        runs[run_name] = {
            "util_p95": float(np.nanpercentile(util, 95.0)),
            "delivery_ratio_p50": delivery_p50,
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


def tv_fx_envelope_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """#1: how much of the longitudinal-force envelope does the TV demand, vs lateral g?

    Fx envelope-use = |TV_desiredFx| / |TV_limitFx| (1.0 = the wheels can give no more
    drive/brake force), over the whole moving lap (not corner-only — Fx peaks on straights
    and in braking zones), binned vs |Ay|. The |·| folds traction and braking together and
    sidesteps the hydraulic-brake force the motors can't deliver. Read against #2: as |Ay|
    grows the car trades longitudinal force (this curve falls) for yaw moment (#2 rises) —
    the QP's α weighting at work. All runs overlaid by colour.
    """
    fig = make_dark_figure(
        "TV longitudinal-force use  ·  envelope vs lateral g",
        "|Ay| [g]",
        "Fx envelope-use  |Fx| / |Fx limit|",
        height=520,
    )
    runs: dict[str, dict] = {}
    warnings: list[str] = []
    any_run = False
    for run_name, df in dfs.items():
        try:
            arr = _prepare_tv_control_arrays(df, ["TV_desiredFx", "TV_limitFx"])
        except Exception as exc:
            warnings.append(f"{run_name}: {exc}")
            continue
        limit = np.abs(arr["TV_limitFx"])
        use = np.divide(
            np.abs(arr["TV_desiredFx"]),
            limit,
            out=np.full_like(limit, np.nan, dtype=float),
            where=limit > 1.0,
        )
        ay_g = np.abs(arr["ay"]) / G_MPS2
        moving = (np.abs(arr["vx"]) >= MIN_SPEED) & np.isfinite(use) & np.isfinite(ay_g)
        if not moving.any():
            warnings.append(f"{run_name}: no valid moving samples for Fx envelope.")
            continue
        any_run = True
        x = ay_g[moving]
        y = use[moving]
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
                hovertemplate=f"{run_name}<br>|Ay|=%{{x:.2f}} g<br>Fx use=%{{y:.2f}}<extra></extra>",
            )
        )
        centers, p50, _ = _binned_percentile(x, y, bin_width=0.2, x_min=0.0, x_max=2.0, pct=50.0)
        _, p95, _ = _binned_percentile(x, y, bin_width=0.2, x_min=0.0, x_max=2.0, pct=95.0)
        valid = np.isfinite(p95)
        fig.add_trace(
            go.Scatter(
                x=centers[valid],
                y=p95[valid],
                mode="lines+markers",
                line=dict(color=color, width=2.6),
                legendgroup=run_name,
                name=f"{run_name} p95 (worst-case)",
            )
        )
        valid50 = np.isfinite(p50)
        fig.add_trace(
            go.Scatter(
                x=centers[valid50],
                y=p50[valid50],
                mode="lines",
                line=dict(color=color, width=1.6, dash="dot"),
                legendgroup=run_name,
                name=f"{run_name} median (typical)",
            )
        )
        runs[run_name] = {
            "fx_use_p95": float(np.nanpercentile(y, 95.0)),
            "time_at_limit_pct": float(100.0 * np.nanmean(y >= 0.95)),
            "moving_samples": int(moving.sum()),
        }

    if any_run:
        fig.add_hline(
            y=1.0,
            line=dict(color="#E5564E", dash="dash", width=2.0),
            annotation_text="force limit",
            annotation_position="top left",
        )
        fig.update_yaxes(range=[0.0, 1.05])
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
