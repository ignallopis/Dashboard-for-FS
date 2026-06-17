"""tc.py
------
Traction Control (TC) control-behaviour figures (redesign 2026-06-16).

Spine — engage → fight/aim → hold — event-based on the TC-armed segments
(``TCenable == 1``) so it works on acceleration logs (laps 0→1) and circuit logs
alike, never the complete-laps filter (which needs ≥2 laps and erases an accel run):
  1. ``tc_engagement_fig``        — did TC arm at the right moment, and for how long?
  2. ``tc_slip_distribution_fig`` — what it fights: per-wheel slip vs +0.20 / +0.33.
  3. ``tc_fidelity_fig`` (toggled) — does it hold its reference?
                                     old car = velocity cap, new car = torque cut.

Spec: docs/superpowers/specs/2026-06-16-tc-control-section-redesign-design.md
"""

from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go

from utils import (
    WHEEL_COLORS,
    driver_color,
    make_dark_figure,
)

CSV_PATH = "data/run8_2025-08-09.csv"
WHEELS = ("FL", "FR", "RL", "RR")

# ── TC parameters ─────────────────────────────────────────────────────────────
SR_TARGET = 0.20  # team grip-optimal slip ratio under traction
TC_ENABLE_COL = "TCenable"
TC_TORQUE_COLS = ("TC_FL_MaxTrq", "TC_FR_MaxTrq", "TC_RL_MaxTrq", "TC_RR_MaxTrq")


def _vx_signal(columns: list[str]) -> str:
    return "Est_vxCOG" if "Est_vxCOG" in columns else "VN_vx"


# ══════════════════════════════════════════════════════════════════════════════
# Event-based TC control section (redesign 2026-06-16)
#
# Spec: docs/superpowers/specs/2026-06-16-tc-control-section-redesign-design.md
# Spine: engage → fight/aim → hold. Works on acceleration logs (laps 0→1) AND
# circuit logs because the analysis window is the TC-armed segments (TCenable==1),
# NOT the complete-laps filter (which needs ≥2 laps and erases an accel run).
# ══════════════════════════════════════════════════════════════════════════════

TC_VX_GUARD = 2.0  # [m/s] below this, slip ratio is unreliable (standing start)
SR_CAP_IMPLIED = 0.33  # old-car velocity-cap implied slip setpoint (vs SR_TARGET 0.20)
SR_DISPLAY_CLIP = 1.0  # clip Est_SR for display/aggregation (it carries ±inf artifacts)


def tc_mode_for_df(df: pl.DataFrame) -> str:
    """Auto-detect the TC actuation logged in *df*.

    'velocity' = old car (wheel-speed cap, TC_*_MaxAngVel; no torque cut logged).
    'torque'   = new car (negative torque cut, TC_*_MaxTrq is live).
    """
    for c in TC_TORQUE_COLS:
        if c in df.columns:
            arr = df[c].to_numpy()
            if np.isfinite(arr).any() and float(np.nanmax(np.abs(arr))) > 1.0e-6:
                return "torque"
    return "velocity"


def _annotate_empty_tc(fig: go.Figure, text: str = "No TC-armed samples in this log") -> None:
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        text=text,
        font=dict(color="#EBEBEB", size=12),
    )


def _tc_event_arrays(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """Event-based arrays for the TC control figures (no complete-laps filter).

    The analysis window is the TC-armed segments. Slip ratio uses the logged
    estimate ``Est_SR{wheel}``, guarded to ``vx > TC_VX_GUARD`` (it explodes at the
    standing start) and clipped to ±``SR_DISPLAY_CLIP``.
    """
    cols = df.columns
    if TC_ENABLE_COL not in cols:
        raise KeyError("TCenable not logged — cannot analyse TC engagement.")
    n = len(df)
    if n < 2:
        raise ValueError("Not enough samples.")

    def col(name: str) -> np.ndarray:
        return df[name].to_numpy().astype(float) if name in cols else np.full(n, np.nan)

    vx_col = _vx_signal(cols)
    out: dict[str, np.ndarray] = {}
    ts = df["TimeStamp"].to_numpy().astype(float)
    out["time"] = ts - ts[0]
    out["throttle"] = col("Throttle")
    out["command"] = col("LLC_Command")
    out["vx"] = col(vx_col)
    armed_raw = df[TC_ENABLE_COL].to_numpy().astype(float)
    out["armed"] = np.isfinite(armed_raw) & (armed_raw == 1.0)

    vx_guard = out["vx"] > TC_VX_GUARD
    for w in WHEELS:
        sr = col(f"Est_SR{w}")
        sr = np.where(np.isfinite(sr) & vx_guard, sr, np.nan)
        out[f"sr_{w}"] = np.clip(sr, -SR_DISPLAY_CLIP, SR_DISPLAY_CLIP)
        out[f"wv_{w}"] = col(f"{w}_actualVelocity")
        out[f"cap_{w}"] = col(f"TC_{w}_MaxAngVel")
        out[f"cut_{w}"] = col(f"TC_{w}_MaxTrq")
    return out


def _worst_sr(d: dict[str, np.ndarray], sl: slice | np.ndarray) -> np.ndarray:
    stk = np.stack([d[f"sr_{w}"][sl] for w in WHEELS], axis=1)
    out = np.full(stk.shape[0], np.nan)
    ok = np.any(np.isfinite(stk), axis=1)
    if ok.any():
        out[ok] = np.nanmax(stk[ok], axis=1)
    return out


def tc_slip_distribution_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """Fig 2 — Slip distribution: where is traction lost while TC is armed?

    A box per wheel of the slip ratio (Est_SR) inside TC-armed segments — median, IQR and
    spread — against the +0.20 team optimum and the +0.33 old-car velocity-cap target.
    Fronts towering over rears = a front-limited launch (load goes rearward, unloading the
    front tyres); a box pushing past +0.33 means the wheels escape the cap.
    """
    fig = make_dark_figure(
        "TC slip distribution  ·  what it fights, where",
        "Wheel",
        "Slip ratio [-]",
    )
    runs: dict[str, dict] = {}
    warnings: list[str] = []
    any_run = False
    for run_name, df in dfs.items():
        try:
            d = _tc_event_arrays(df)
        except Exception as exc:
            warnings.append(f"{run_name}: {exc}")
            continue
        armed = d["armed"]
        if not armed.any():
            warnings.append(f"{run_name}: no TC-armed samples.")
            continue
        color = driver_color(run_name)
        p95: dict[str, float] = {}
        nper: dict[str, int] = {}
        plotted = False
        for w in WHEELS:
            s = d[f"sr_{w}"][armed]
            s = s[np.isfinite(s)]
            nper[w] = int(s.size)
            if s.size < 5:
                p95[w] = np.nan
                continue
            p95[w] = float(np.nanpercentile(s, 95))
            fig.add_trace(
                go.Box(
                    y=s,
                    x=[w] * s.size,
                    name=run_name,
                    legendgroup=run_name,
                    showlegend=not plotted,
                    marker_color=color,
                    line=dict(color=color),
                    boxpoints=False,
                    whiskerwidth=0.5,
                )
            )
            plotted = True
        if not plotted:
            warnings.append(f"{run_name}: too few armed slip samples.")
            continue
        any_run = True
        p95arr = [p95[w] for w in WHEELS]
        runs[run_name] = {
            "front_p95_sr": round(float(np.nanmax(p95arr[:2])), 3),
            "rear_p95_sr": round(float(np.nanmax(p95arr[2:])), 3),
            "worst_wheel": WHEELS[int(np.nanargmax(p95arr))],
            "armed_samples": int(sum(nper.values())),
        }

    if any_run:
        fig.add_hline(
            y=SR_TARGET,
            line=dict(color="rgba(255,255,255,0.6)", dash="dash", width=1.3),
            annotation_text="optimum +0.20",
            annotation_position="top left",
            annotation_font_color="#EBEBEB",
        )
        fig.add_hline(
            y=SR_CAP_IMPLIED,
            line=dict(color="rgba(242,140,64,0.75)", dash="dot", width=1.3),
            annotation_text="old-car cap +0.33",
            annotation_position="bottom left",
            annotation_font_color="#F28C40",
        )
        fig.update_layout(boxmode="group")
        fig.update_xaxes(categoryorder="array", categoryarray=list(WHEELS))
        fig.update_yaxes(range=[-0.15, 0.8])
    else:
        _annotate_empty_tc(fig)
    return fig, {"runs": runs, "warnings": warnings}


def tc_overslip_severity_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """Fig 3 — Overslip severity: how much armed time each wheel spends past +0.20.

    Per wheel, the share of TC-armed time the slip ratio exceeds the +0.20 optimum (bar)
    vs the +0.33 old-car escape line (hover) — an operating-point *persistence* gauge of
    where traction is lost, NOT a controller response time (the cut channel is unrecorded
    on these logs). Taller = the wheel rides in overslip more of the time it is armed.
    """
    fig = make_dark_figure(
        "TC overslip severity  ·  share of armed time past +0.20",
        "Wheel",
        "Armed time over target [%]",
    )
    runs: dict[str, dict] = {}
    warnings: list[str] = []
    any_run = False
    for run_name, df in dfs.items():
        try:
            d = _tc_event_arrays(df)
        except Exception as exc:
            warnings.append(f"{run_name}: {exc}")
            continue
        armed = d["armed"]
        if not armed.any():
            warnings.append(f"{run_name}: no TC-armed samples.")
            continue
        color = driver_color(run_name)
        over: dict[str, float] = {}
        severe: dict[str, float] = {}
        peak: dict[str, float] = {}
        nper: dict[str, int] = {}
        for w in WHEELS:
            s = d[f"sr_{w}"][armed]
            s = s[np.isfinite(s)]
            nper[w] = int(s.size)
            if s.size < 5:
                over[w] = severe[w] = peak[w] = np.nan
                continue
            over[w] = 100.0 * float(np.mean(s > SR_TARGET))
            severe[w] = 100.0 * float(np.mean(s > SR_CAP_IMPLIED))
            peak[w] = float(np.nanpercentile(s, 95))
        if all(not np.isfinite(over[w]) for w in WHEELS):
            warnings.append(f"{run_name}: too few armed slip samples.")
            continue
        any_run = True
        fig.add_trace(
            go.Bar(
                x=list(WHEELS),
                y=[over[w] for w in WHEELS],
                name=run_name,
                marker_color=color,
                customdata=[[severe[w], peak[w]] for w in WHEELS],
                hovertemplate=(
                    "%{x}<br>over +0.20=%{y:.1f}%<br>over +0.33=%{customdata[0]:.1f}%"
                    "<br>p95 peak SR=%{customdata[1]:.2f}"
                    f"<extra>{run_name}</extra>"
                ),
            )
        )
        over_vals = [over[w] for w in WHEELS]
        peak_vals = [peak[w] for w in WHEELS if np.isfinite(peak[w])]
        worst = WHEELS[int(np.nanargmax(over_vals))]
        runs[run_name] = {
            "worst_wheel": worst,
            "pct_time_over_target": round(float(np.nanmax(over_vals)), 1),
            "p95_peak_sr": round(float(np.nanmax(peak_vals)), 3) if peak_vals else float("nan"),
            "armed_samples": int(sum(nper.values())),
        }
    if any_run:
        fig.update_layout(barmode="group")
        fig.update_xaxes(categoryorder="array", categoryarray=list(WHEELS))
        fig.update_yaxes(rangemode="tozero")
    else:
        _annotate_empty_tc(fig)
    return fig, {"runs": runs, "warnings": warnings}


def _fidelity_velocity_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """OLD car: actual wheel speed vs the TC velocity cap (its own reference)."""
    fig = make_dark_figure(
        "TC control fidelity  ·  wheel speed vs velocity cap (old car)",
        "TC velocity cap [rad/s]",
        "Actual wheel speed [rad/s]",
    )
    runs: dict[str, dict] = {}
    warnings: list[str] = []
    lo_all: list[float] = []
    hi_all: list[float] = []
    shown: set[str] = set()
    any_pts = False
    for run_name, df in dfs.items():
        try:
            d = _tc_event_arrays(df)
        except Exception as exc:
            warnings.append(f"{run_name}: {exc}")
            continue
        armed = d["armed"]
        if not armed.any():
            warnings.append(f"{run_name}: no TC-armed samples.")
            continue
        over: dict[str, float] = {}
        p95r: dict[str, float] = {}
        for w in WHEELS:
            cap = d[f"cap_{w}"][armed]
            wv = d[f"wv_{w}"][armed]
            m = np.isfinite(cap) & np.isfinite(wv) & (cap > 1.0)
            if m.sum() < 5:
                continue
            x = cap[m]
            y = wv[m]
            any_pts = True
            stride = max(1, int(np.ceil(x.size / 4000)))
            fig.add_trace(
                go.Scattergl(
                    x=x[::stride],
                    y=y[::stride],
                    mode="markers",
                    marker=dict(color=WHEEL_COLORS[w], size=3, opacity=0.3),
                    name=w,
                    legendgroup=w,
                    showlegend=w not in shown,
                    hovertemplate=f"{run_name} {w}<br>cap=%{{x:.0f}}<br>actual=%{{y:.0f}} rad/s<extra></extra>",
                )
            )
            shown.add(w)
            ratio = y / x
            over[w] = 100.0 * float(np.mean(ratio > 1.0))
            p95r[w] = float(np.nanpercentile(ratio, 95))
            lo_all.append(float(min(x.min(), y.min())))
            hi_all.append(float(max(x.max(), y.max())))
        if over:
            runs[run_name] = {
                "front_over_cap_pct": round(max(over.get("FL", 0.0), over.get("FR", 0.0)), 1),
                "rear_over_cap_pct": round(max(over.get("RL", 0.0), over.get("RR", 0.0)), 1),
                "front_p95_ratio": round(max(p95r.get("FL", 0.0), p95r.get("FR", 0.0)), 2),
                "rear_p95_ratio": round(max(p95r.get("RL", 0.0), p95r.get("RR", 0.0)), 2),
            }
    if any_pts and lo_all:
        lo = min(lo_all)
        hi = max(hi_all)
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(color="#73D973", dash="dash", width=2.0),
                name="holds the cap (y=x)",
            )
        )
    else:
        _annotate_empty_tc(fig)
    return fig, {"runs": runs, "warnings": warnings}


def _fidelity_torque_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """NEW car: torque cut vs slip-over-target (PI response). Dark until new-car data."""
    fig = make_dark_figure(
        "TC control fidelity  ·  torque cut vs slip error (new car)",
        "Worst-wheel slip over target [-]",
        "TC torque cut applied [Nm]",
    )
    runs: dict[str, dict] = {}
    warnings: list[str] = []
    any_cut = False
    for run_name, df in dfs.items():
        try:
            d = _tc_event_arrays(df)
        except Exception as exc:
            warnings.append(f"{run_name}: {exc}")
            continue
        armed = d["armed"]
        if not armed.any():
            warnings.append(f"{run_name}: no TC-armed samples.")
            continue
        cut = np.stack([d[f"cut_{w}"][armed] for w in WHEELS], axis=1)
        cut_total = np.nansum(np.minimum(cut, 0.0), axis=1)
        sr_over = _worst_sr(d, armed) - SR_TARGET
        m = np.isfinite(cut_total) & np.isfinite(sr_over) & (cut_total < -1.0e-6)
        if m.sum() >= 5:
            any_cut = True
            color = driver_color(run_name)
            fig.add_trace(
                go.Scattergl(
                    x=sr_over[m],
                    y=cut_total[m],
                    mode="markers",
                    marker=dict(color=color, size=3, opacity=0.3),
                    name=run_name,
                    hovertemplate=f"{run_name}<br>slip over=%{{x:.3f}}<br>cut=%{{y:.1f}} Nm<extra></extra>",
                )
            )
            runs[run_name] = {
                "p95_cut_nm": round(float(np.nanpercentile(-cut_total[m], 95)), 1),
                "cut_samples": int(m.sum()),
            }
    if any_cut:
        fig.add_hline(y=0.0, line=dict(color="#9AA0A6", dash="dash", width=1.2))
    else:
        _annotate_empty_tc(
            fig, "No torque-cut data in this log (old-car velocity TC, or TC never cut)"
        )
    return fig, {"runs": runs, "warnings": warnings}


def tc_fidelity_fig(dfs: dict[str, pl.DataFrame], mode: str = "velocity") -> tuple[go.Figure, dict]:
    """Fig 4 — Control fidelity (toggled): how tightly does TC hold its reference?

    mode='velocity' (old car): actual wheel speed vs the TC velocity cap, y=x = perfect
    hold; points above = the wheel escaped the cap.
    mode='torque' (new car): torque cut vs slip-over-target (PI response); shows an
    explicit empty-state until new-car logs carry a live cut.
    """
    if mode == "torque":
        return _fidelity_torque_fig(dfs)
    return _fidelity_velocity_fig(dfs)


def main() -> None:
    """Standalone smoke test of the TC control figures on CSV_PATH."""
    df = pl.read_csv(CSV_PATH, infer_schema_length=2000)
    dfs = {CSV_PATH: df}
    mode = tc_mode_for_df(df)
    print(f"TC actuation detected: {mode}")
    _, kpis = tc_slip_distribution_fig(dfs)
    print("slip_distribution", kpis["runs"], kpis["warnings"])
    _, kpis = tc_fidelity_fig(dfs, mode=mode)
    print("fidelity", kpis["runs"], kpis["warnings"])


if __name__ == "__main__":
    main()
