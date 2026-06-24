"""tc.py
------
Traction Control (TC) figures — one per controller reference (redesign 2026-06-21).

Event-based on the TC-armed segments (``TCenable == 1``) so it works on acceleration
logs (laps 0→1) and circuit logs alike, never the complete-laps filter (which needs ≥2
laps and erases an accel run). One figure per stage of the TC pipeline:
  1. ``tc_optimal_slip_ratio_fig`` — setpoint: achieved slip vs the +0.20 optimum and the
                                     controller's own target (backed out from the cap).
  2. ``tc_reference_velocity_fig`` — does the wheel hold the TC velocity reference?
  3. ``tc_reference_torque_fig``   — actuation effort (torque cut); placeholder until a
                                     torque-mode (new-car) log carries a live cut.

Spec: docs/superpowers/specs/2026-06-21-controls-tc-three-references-redesign-design.md
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

CSV_PATH = "data/Acceleration_FSS.csv"
WHEELS = ("FL", "FR", "RL", "RR")

# ── TC parameters ─────────────────────────────────────────────────────────────
SR_TARGET = 0.20  # team grip-optimal slip ratio under traction
TC_ENABLE_COL = "TCenable"
TC_TORQUE_COLS = ("TC_FL_MaxTrq", "TC_FR_MaxTrq", "TC_RL_MaxTrq", "TC_RR_MaxTrq")


def _vx_signal(columns: list[str]) -> str:
    return "Est_vxCOG" if "Est_vxCOG" in columns else "VN_vx"


# ══════════════════════════════════════════════════════════════════════════════
# One figure per TC controller reference (redesign 2026-06-21).
#
# Spec: docs/superpowers/specs/2026-06-21-controls-tc-three-references-redesign-design.md
# The analysis window is the TC-armed segments (TCenable==1), NOT the complete-laps
# filter (which needs ≥2 laps and erases an accel run), so it works on FSS accel logs.
# ══════════════════════════════════════════════════════════════════════════════

TC_VX_GUARD = 2.0  # [m/s] below this, slip ratio is unreliable (standing start)
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


def tc_optimal_slip_ratio_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """Fig 1 — Optimal slip ratio: where the tyre operated, vs the +0.20 optimum.

    Per-wheel distribution of the slip ratio reached while TC is armed (Est_SR, guarded to
    vx>2 m/s and clipped — the standing-start estimate explodes), against the +0.20 grip
    optimum. An operating-point read — *where slip landed* — NOT a control-quality one.
    """
    fig = make_dark_figure(
        "TC optimal slip ratio  ·  where the tyre operates vs +0.20",
        "Wheel",
        "Slip ratio [-]",
        height=520,
    )
    runs: dict[str, dict] = {}
    warnings: list[str] = []
    samples: dict[str, list[np.ndarray]] = {w: [] for w in WHEELS}
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
        p95: dict[str, float] = {}
        over: dict[str, float] = {}
        nper: dict[str, int] = {}
        for w in WHEELS:
            sr = d[f"sr_{w}"]
            s = sr[armed & np.isfinite(sr)]
            nper[w] = int(s.size)
            if s.size < 5:
                p95[w] = over[w] = np.nan
                continue
            p95[w] = float(np.nanpercentile(s, 95))
            over[w] = 100.0 * float(np.mean(s > SR_TARGET))
            samples[w].append(s)
        if all(not np.isfinite(p95[w]) for w in WHEELS):
            warnings.append(f"{run_name}: too few armed slip samples.")
            continue
        p95arr = [p95[w] for w in WHEELS]
        over_arr = [over[w] for w in WHEELS]
        runs[run_name] = {
            "front_p95_sr": round(float(np.nanmax(p95arr[:2])), 3),
            "rear_p95_sr": round(float(np.nanmax(p95arr[2:])), 3),
            "worst_wheel": WHEELS[int(np.nanargmax(p95arr))],
            "pct_time_over_target": round(float(np.nanmax(over_arr)), 1),
            "armed_samples": int(sum(nper.values())),
        }

    if any(samples[w] for w in WHEELS):
        for w in WHEELS:
            if not samples[w]:
                continue
            s = np.concatenate(samples[w])
            fig.add_trace(
                go.Box(
                    y=s,
                    name=w,
                    marker_color=WHEEL_COLORS[w],
                    line_color=WHEEL_COLORS[w],
                    boxpoints=False,
                    whiskerwidth=0.4,
                    hovertemplate=f"{w}<br>SR=%{{y:.2f}}<extra></extra>",
                )
            )
        fig.add_hline(
            y=SR_TARGET,
            line=dict(color="rgba(255,255,255,0.6)", dash="dash", width=1.3),
            annotation_text="optimum +0.20",
            annotation_position="top left",
            annotation_font_color="#EBEBEB",
        )
        fig.update_yaxes(range=[-0.15, 0.8])
        fig.update_xaxes(categoryorder="array", categoryarray=list(WHEELS))
    else:
        _annotate_empty_tc(fig)
    return fig, {"runs": runs, "warnings": warnings}


def tc_reference_velocity_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """Fig 2 — Reference velocity: does the wheel hold the TC velocity reference?

    Per-wheel actual motor speed vs the TC velocity reference (TC_*_MaxAngVel) inside armed
    segments, both in motor rad/s; on the y=x line the wheel sits exactly at its reference,
    above it the wheel escaped the cap. Leads on the escape rate — the share of armed time
    each axle ran past its reference.
    """
    fig = make_dark_figure(
        "TC reference velocity  ·  wheel speed vs the TC velocity reference",
        "TC velocity reference [rad/s]",
        "Actual wheel speed [rad/s]",
        height=520,
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
        nper: dict[str, int] = {}
        for w in WHEELS:
            cap = d[f"cap_{w}"][armed]
            wv = d[f"wv_{w}"][armed]
            m = np.isfinite(cap) & np.isfinite(wv) & (cap > 1.0)
            nper[w] = int(m.sum())
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
                    hovertemplate=f"{run_name} {w}<br>ref=%{{x:.0f}}<br>actual=%{{y:.0f}} rad/s<extra></extra>",
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
                "front_escape_pct": round(max(over.get("FL", 0.0), over.get("FR", 0.0)), 1),
                "rear_escape_pct": round(max(over.get("RL", 0.0), over.get("RR", 0.0)), 1),
                "front_p95_ratio": round(max(p95r.get("FL", 0.0), p95r.get("FR", 0.0)), 2),
                "rear_p95_ratio": round(max(p95r.get("RL", 0.0), p95r.get("RR", 0.0)), 2),
                "armed_samples": int(sum(nper.values())),
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
                name="holds the reference (y=x)",
            )
        )
    else:
        _annotate_empty_tc(fig)
    return fig, {"runs": runs, "warnings": warnings}


def tc_reference_torque_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """Fig 3 — Reference torque: how hard TC cuts torque to hold the reference.

    Total TC torque cut (TC_*_MaxTrq, ≤0, summed over the four wheels) against the worst
    wheel's slip over the +0.20 target, inside armed segments — the actuation effort vs the
    error driving it. Empty on velocity-mode logs (old car), where the cut is done inside
    the inverter and TC_*_MaxTrq is not recorded; lights up on torque-mode (new-car) logs.
    """
    fig = make_dark_figure(
        "TC reference torque  ·  torque cut vs slip over target",
        "Worst-wheel slip over +0.20 [-]",
        "TC torque cut applied [Nm]",
        height=520,
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
            fig,
            "No torque cut recorded — TC ran in velocity mode (old car); "
            "this figure lights up on new-car torque-mode logs.",
        )
    return fig, {"runs": runs, "warnings": warnings}


def main() -> None:
    """Standalone smoke test of the TC reference figures on CSV_PATH."""
    df = pl.read_csv(CSV_PATH, infer_schema_length=2000)
    dfs = {CSV_PATH: df}
    print(f"TC actuation detected: {tc_mode_for_df(df)}")
    _, kpis = tc_optimal_slip_ratio_fig(dfs)
    print("optimal_slip_ratio", kpis["runs"], kpis["warnings"])
    _, kpis = tc_reference_velocity_fig(dfs)
    print("reference_velocity", kpis["runs"], kpis["warnings"])
    _, kpis = tc_reference_torque_fig(dfs)
    print("reference_torque", kpis["runs"], kpis["warnings"])


if __name__ == "__main__":
    main()
