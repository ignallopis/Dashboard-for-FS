"""powertrain.py
--------------
Powertrain figures for the dashboard's three sub-sections.

Motors & Inverters:
    power_per_wheel_fig(dfs)        -> tuple[go.Figure, dict]
    torque_speed_envelope_fig(dfs)  -> tuple[go.Figure, dict]
    torque_fidelity_fig(dfs)        -> tuple[go.Figure, dict]
    inverter_limits_fig(dfs)        -> tuple[go.Figure, dict]
    pc_function_kpis(df)            -> tuple[list[go.Figure], dict]

Battery:
    energy_per_lap_fig(dfs)         -> tuple[go.Figure, dict]
    soc_per_lap_fig(dfs)            -> tuple[go.Figure, dict]
    voltage_sag_fig(dfs)            -> tuple[go.Figure, dict]
    pack_capacity_fig(dfs)          -> tuple[go.Figure, dict]

Temperatures:
    thermal_evolution_fig(dfs)      -> tuple[go.Figure, dict]
    thermal_headroom_fig(dfs)       -> tuple[go.Figure, dict]

Requires lapcount.py to have been run first.
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from utils import (
    apply_dark_layout,
    make_dark_figure,
    add_lap_scatter,
    add_trend_line,
    cols_to_numpy,
    driver_color,
    ensure_complete_laps_df,
    per_lap_axis,
    unique_laps,
    WHEEL_COLORS,
)

WHEELS = ("FL", "FR", "RL", "RR")

# OT limits from Parameters.m (Motor/Inverter "OT error"); real derating usually
# begins a few °C below these, so headroom-to-limit is conservative.
MOTOR_OT_LIMIT_C = 120.0  # [°C] Parameters.m: Motor OT error
INVERTER_OT_LIMIT_C = 75.0  # [°C] Parameters.m: Inverter OT error
# Motor limits from Parameters.m (docs/context/cat17x_parameters.md).
MOTOR_MAX_TORQUE_NM = 27.5
MOTOR_MAX_SPEED_RADS = 16_000.0 * 2.0 * np.pi / 60.0
# FS-rule accumulator cell temperature limit; NOT in Parameters.m.
BATTERY_TEMP_LIMIT_C = 60.0
# Per-motor share of the 80 kW FS power cap (equal-split approximation).
PER_MOTOR_POWER_CAP_W = 80_000.0 / 4.0
# Plausibility bands: samples outside are sensor glitches -> NaN.
_CELL_V_BAND = (2.0, 4.5)  # [V] single cell
_PACK_V_BAND = (300.0, 650.0)  # [V] accumulator
_TEMP_BAND_C = (-20.0, 150.0)  # [°C] any temperature channel
WEAKEST_CELL_DISCHARGE_FLOOR_V = 3.75  # [V] single-cell discharge reference

# Dark theme (mirrors utils.py for subplot figures)
_BG = "#141417"
_TEXT = "#EBEBEB"
_GRID = "rgba(128,128,128,0.2)"
_AXIS = "#E5E5E5"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _clean_band(a: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Replace values outside a plausibility band (sensor glitches) with NaN."""
    out = np.asarray(a, dtype=float).copy()
    out[~np.isfinite(out) | (out < band[0]) | (out > band[1])] = np.nan
    return out


def _decimate(n: int, max_points: int = 4000) -> slice:
    """Stride slice that keeps at most ~max_points samples."""
    return slice(None, None, max(1, n // max_points))


def _dark_subplots(
    rows: int,
    titles: list[str],
    ylabels: list[str],
    *,
    vertical_spacing: float = 0.08,
) -> go.Figure:
    """Create a make_subplots figure with dark motorsport styling."""
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        subplot_titles=titles,
    )
    # Shared dark styling (bg, font family, legend, dark hover box, modebar);
    # per-row axes are styled below.
    apply_dark_layout(fig, single_axes=False)
    for i in range(1, rows + 1):
        fig.update_xaxes(
            color=_AXIS,
            gridcolor=_GRID,
            linecolor=_AXIS,
            tickcolor=_AXIS,
            showgrid=True,
            row=i,
            col=1,
        )
        fig.update_yaxes(
            title_text=ylabels[i - 1],
            color=_AXIS,
            gridcolor=_GRID,
            linecolor=_AXIS,
            tickcolor=_AXIS,
            showgrid=True,
            row=i,
            col=1,
        )
    for ann in fig.layout.annotations:
        ann.font.color = _TEXT
        ann.font.size = 12
    return fig


def _per_wheel_run_grid(
    runs: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    lap_ticks: np.ndarray | None = None,
) -> go.Figure:
    """2×2 per-wheel grid for multi-run per-lap figures: one line per run.

    *runs* maps run_name -> (x values, {wheel: y values}). Run identity is
    colour (driver_color); wheel identity is the panel position.
    """
    pos = {"FL": (1, 1), "FR": (1, 2), "RL": (2, 1), "RR": (2, 2)}
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=list(WHEELS),
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.07,
        vertical_spacing=0.12,
    )
    apply_dark_layout(fig, single_axes=False)
    for run_name, (x, ys) in runs.items():
        for w, (r, c) in pos.items():
            if w not in ys:
                continue
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=ys[w],
                    mode="lines+markers",
                    name=run_name,
                    legendgroup=run_name,
                    showlegend=(w == "FL"),
                    line=dict(color=driver_color(run_name), width=2),
                    marker=dict(size=6),
                ),
                row=r,
                col=c,
            )
    for _w, (r, c) in pos.items():
        fig.update_xaxes(
            title_text=xlabel if r == 2 else None,
            tickvals=lap_ticks,
            gridcolor=_GRID,
            color=_AXIS,
            row=r,
            col=c,
        )
        fig.update_yaxes(
            title_text=ylabel if c == 1 else None,
            gridcolor=_GRID,
            color=_AXIS,
            row=r,
            col=c,
        )
    fig.update_layout(height=560, title=dict(text=title, font=dict(color=_TEXT, size=14)))
    for ann in fig.layout.annotations:
        ann.font.color = _TEXT
        ann.font.size = 12
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard API  —  functions take dfs dict, return (go.Figure, kpis_dict)
# ═══════════════════════════════════════════════════════════════════════════════


# ── Energy per lap ───────────────────────────────────────────────────────────


def energy_per_lap_fig(
    dfs: dict[str, pl.DataFrame],
    x_mode: str = "laptime",
) -> tuple[go.Figure, dict]:
    """Energy per lap figure + KPIs for all selected runs.

    Returns (figure, kpis) where kpis keys:
        e_mean, e_total, e_cons_mean, e_rec_mean, e_rec_total, cv, r2, p_mean,
        min_e_lap, min_e, max_e_lap, max_e, fastest_lap, fastest_lt,
        table (pl.DataFrame), warnings (list[str]).
    """
    cols_needed = ["TimeStamp", "laps", "Vbat", "Current"]
    laps_all: list[int] = []
    lt_all: list[float] = []
    e_net_all: list[float] = []
    e_cons_all: list[float] = []
    e_rec_all: list[float] = []
    p_all: list[float] = []
    run_all: list[str] = []
    warnings: list[str] = []

    for run_name, df in dfs.items():
        df = ensure_complete_laps_df(df)
        missing = [c for c in cols_needed if c not in df.columns]
        if missing:
            warnings.append(f"{run_name}: missing {missing}")
            continue

        cols = cols_to_numpy(df, ["TimeStamp", "laps", "Vbat", "Current"])
        time = cols["TimeStamp"]
        laps = cols["laps"]
        vbat = cols["Vbat"]
        current = cols["Current"]

        valid = np.isfinite(time) & np.isfinite(laps) & np.isfinite(vbat) & np.isfinite(current)
        time = time[valid] - time[valid][0]
        laps = laps[valid]
        p_kw = (vbat[valid] * current[valid]) / 1000.0

        for lap in unique_laps(laps):
            idx = np.where(laps == lap)[0]
            if len(idx) < 2:
                continue
            lt = time[idx[-1]] - time[idx[0]]
            e_net = float(np.trapezoid(p_kw[idx], time[idx]) / 3600.0)
            e_cons = float(np.trapezoid(np.clip(p_kw[idx], 0.0, None), time[idx]) / 3600.0)
            e_rec = float(np.trapezoid(np.clip(-p_kw[idx], 0.0, None), time[idx]) / 3600.0)
            pa = float(np.nanmean(p_kw[idx]))
            if np.isfinite(lt) and lt > 0 and np.isfinite(e_net):
                laps_all.append(int(lap))
                lt_all.append(float(lt))
                e_net_all.append(e_net)
                e_cons_all.append(e_cons)
                e_rec_all.append(e_rec)
                p_all.append(pa)
                run_all.append(run_name)

    if not e_net_all:
        fig = make_dark_figure(title="Energy per Lap — No data")
        return fig, {"warnings": warnings + ["No valid energy data."]}

    e_arr = np.array(e_net_all)
    e_cons_arr = np.array(e_cons_all)
    e_rec_arr = np.array(e_rec_all)
    lt_arr = np.array(lt_all)
    lp_arr = np.array(laps_all)
    p_arr = np.array(p_all)

    e_mean = float(np.nanmean(e_arr))
    e_cons_mean = float(np.nanmean(e_cons_arr))
    e_rec_mean = float(np.nanmean(e_rec_arr))
    e_std = float(np.nanstd(e_arr))
    cv = 100 * e_std / e_mean if e_mean > 0 else np.nan
    e_total = float(np.nansum(e_arr))
    e_rec_total = float(np.nansum(e_rec_arr))
    e_cons_total = float(np.nansum(e_cons_arr))
    p_mean = float(np.nanmean(p_arr))

    def _r2(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 2:
            return np.nan
        poly = np.polyfit(x, y, 1)
        y_pred = np.polyval(poly, x)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.nanmean(y)) ** 2))
        return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    r2_laptime = _r2(lt_arr, e_arr)
    r2_lap = _r2(lp_arr.astype(float), e_arr)
    r2 = r2_laptime if x_mode == "laptime" else r2_lap

    i_min_e = int(np.argmin(e_arr))
    i_max_e = int(np.argmax(e_arr))
    i_fastest = int(np.argmin(lt_arr))

    x_arr, order, xlabel = per_lap_axis(lp_arr, lt_arr, x_mode)
    fig = _dark_subplots(
        rows=2,
        titles=[
            f"Net Battery Energy vs {'Lap Time' if x_mode == 'laptime' else 'Lap'}  (R² = {r2:.3f})",
            "Consumed vs Recovered Energy",
        ],
        ylabels=["Net energy [kWh]", "Energy [kWh]"],
    )
    add_lap_scatter(fig, x_arr, e_arr[order], lp_arr[order], name="Net", color="#4DB3F2")
    add_trend_line(fig, x_arr, e_arr[order])
    fig.add_trace(
        go.Scatter(
            x=x_arr,
            y=e_cons_arr[order],
            mode="lines+markers",
            name="Consumed",
            line=dict(color="#F27070", width=1.5),
            marker=dict(size=6),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_arr,
            y=e_rec_arr[order],
            mode="lines+markers",
            name="Recovered",
            line=dict(color="#73D973", width=1.5),
            marker=dict(size=6),
        ),
        row=2,
        col=1,
    )
    if x_mode == "laps":
        fig.update_xaxes(tickvals=np.sort(lp_arr.astype(int)), row=1, col=1)
        fig.update_xaxes(tickvals=np.sort(lp_arr.astype(int)), row=2, col=1)
    fig.update_xaxes(title_text=xlabel, row=2, col=1)
    fig.add_hline(
        y=e_mean,
        line=dict(color="rgba(200,200,200,0.4)", dash="dot", width=1),
        row=1,
        col=1,
    )
    fig.add_hrect(
        y0=e_mean - e_std,
        y1=e_mean + e_std,
        fillcolor="rgba(77,179,242,0.08)",
        line_width=0,
        row=1,
        col=1,
    )
    fig.update_layout(
        height=720,
        title_text=f"Energy per Lap {'vs Lap Time' if x_mode == 'laptime' else 'per Lap'}",
        title_font=dict(color=_TEXT, size=14),
    )

    table: dict[str, object] = {
        "Lap": lp_arr,
        "Laptime [s]": np.round(lt_arr, 3),
        "Net energy [kWh]": np.round(e_arr, 5),
        "Consumed [kWh]": np.round(e_cons_arr, 5),
        "Recovered [kWh]": np.round(e_rec_arr, 5),
        "P_avg [kW]": np.round(p_arr, 3),
    }
    if len(dfs) > 1:
        table["Run"] = run_all

    kpis = {
        "e_mean": e_mean,
        "e_total": e_total,
        "e_cons_mean": e_cons_mean,
        "e_cons_total": e_cons_total,
        "e_rec_mean": e_rec_mean,
        "e_rec_total": e_rec_total,
        "cv": cv,
        "r2": r2,
        "r2_lap": r2_lap,
        "r2_laptime": r2_laptime,
        "p_mean": p_mean,
        "min_e_lap": int(lp_arr[i_min_e]),
        "min_e": float(e_arr[i_min_e]),
        "max_e_lap": int(lp_arr[i_max_e]),
        "max_e": float(e_arr[i_max_e]),
        "fastest_lap": int(lp_arr[i_fastest]),
        "fastest_lt": float(lt_arr[i_fastest]),
        "table": pl.DataFrame(table),
        "warnings": warnings,
    }
    return fig, kpis


# ── Power per wheel ──────────────────────────────────────────────────────────


def _wheel_power(df: pl.DataFrame) -> tuple[dict[str, np.ndarray], str] | None:
    """Per-wheel power [W] plus a source label.

    Prefers the electrical `{w}_actualPower`. On CAT18x logs that omit it, falls
    back to **mechanical** power = `{w}_actualTorque × {w}_actualVelocity`
    (Nm·rad/s = W). Returns ``None`` when neither source is available.
    """
    cols = set(df.columns)
    if all(f"{w}_actualPower" in cols for w in WHEELS):
        arr = cols_to_numpy(df, [f"{w}_actualPower" for w in WHEELS])
        return {w: arr[f"{w}_actualPower"] for w in WHEELS}, "Electrical"
    mech = [f"{w}_actualTorque" for w in WHEELS] + [f"{w}_actualVelocity" for w in WHEELS]
    if all(c in cols for c in mech):
        arr = cols_to_numpy(df, mech)
        return {
            w: arr[f"{w}_actualTorque"] * arr[f"{w}_actualVelocity"] for w in WHEELS
        }, "Mechanical"
    return None


def _inverter_input_power(df: pl.DataFrame) -> tuple[np.ndarray, str] | None:
    """Total inverter electrical **input** power [W] (Σ over wheels) + source label.

    Prefers Σ `{w}_actualPower`. On CAT18x logs that omit it, falls back to the
    inverter DC-bus input Σ `{w}_dc_current × {w}_dc_bus_voltage` — which is exactly
    "power reaching the inverters". Returns ``None`` when neither source is present.
    """
    cols = set(df.columns)
    if all(f"{w}_actualPower" in cols for w in WHEELS):
        arr = cols_to_numpy(df, [f"{w}_actualPower" for w in WHEELS])
        p = np.zeros(len(df))
        for w in WHEELS:
            p = p + arr[f"{w}_actualPower"]
        return p, "actualPower"
    dc = [f"{w}_dc_current" for w in WHEELS] + [f"{w}_dc_bus_voltage" for w in WHEELS]
    if all(c in cols for c in dc):
        arr = cols_to_numpy(df, dc)
        p = np.zeros(len(df))
        for w in WHEELS:
            p = p + arr[f"{w}_dc_current"] * _clean_band(arr[f"{w}_dc_bus_voltage"], _PACK_V_BAND)
        return p, "DC bus"
    return None


def power_per_wheel_fig(
    dfs: dict[str, pl.DataFrame],
    x_mode: str = "laps",
) -> tuple[go.Figure, dict]:
    """Power distribution per wheel figure + KPIs.

    Uses the electrical `{w}_actualPower` when logged, else mechanical torque×ω
    (CAT18x logs without `actualPower`); the title reflects which.

    kpis keys: mean_total_kw, fr_pct, lr_pct,
               wheel_mean_kw (dict), wheel_pct (dict),
               table (pl.DataFrame), warnings (list[str]).
    """
    laps_all: list[int] = []
    lt_all: list[float] = []
    pw_all: list[list[float]] = []
    run_all: list[str] = []
    warnings: list[str] = []
    plabel = "Motor"

    for run_name, df in dfs.items():
        df = ensure_complete_laps_df(df)
        resolved = _wheel_power(df)
        if resolved is None:
            warnings.append(
                f"{run_name}: no per-wheel power ({{w}}_actualPower or torque+velocity)"
            )
            continue
        powers, plabel = resolved

        cols = cols_to_numpy(df, ["laps", "laptime"])
        laps = cols["laps"]
        laptime = cols["laptime"]

        for lap in unique_laps(laps):
            idx = laps == lap
            if idx.sum() < 5:
                continue
            pw = [float(np.nanmean(powers[w][idx]) / 1000.0) for w in WHEELS]
            if all(np.isfinite(pw)):
                laps_all.append(int(lap))
                lt_all.append(float(np.nanmax(laptime[idx])))
                pw_all.append(pw)
                run_all.append(run_name)

    if not pw_all:
        fig = make_dark_figure(title="Power Distribution — No data")
        return fig, {"warnings": warnings + ["No valid power data."]}

    lp_arr = np.array(laps_all)
    lt_arr = np.array(lt_all)
    pw_arr = np.array(pw_all)  # (N, 4)

    p_total = np.nansum(pw_arr, axis=1)
    p_front = pw_arr[:, 0] + pw_arr[:, 1]
    p_left = pw_arr[:, 0] + pw_arr[:, 2]
    mean_total = float(np.nanmean(p_total))
    fr_pct = float(np.nanmean(p_front) / mean_total * 100) if mean_total > 0 else np.nan
    lr_pct = float(np.nanmean(p_left) / mean_total * 100) if mean_total > 0 else np.nan

    title = f"Mean {plabel} Power per Wheel vs {'Lap Time' if x_mode == 'laptime' else 'Lap'}"
    run_names = list(dict.fromkeys(run_all))
    if len(run_names) > 1:
        grid_runs: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
        xlabel = "Lap"
        for rn in run_names:
            m = np.array([r == rn for r in run_all])
            x_run, order_run, xlabel = per_lap_axis(lp_arr[m], lt_arr[m], x_mode)
            pw_run = pw_arr[m]
            grid_runs[rn] = (x_run, {w: pw_run[order_run, j] for j, w in enumerate(WHEELS)})
        fig = _per_wheel_run_grid(
            grid_runs,
            title=title,
            xlabel=xlabel,
            ylabel="Power [kW]",
            lap_ticks=np.unique(lp_arr.astype(int)) if x_mode == "laps" else None,
        )
    else:
        x_arr, order, xlabel = per_lap_axis(lp_arr, lt_arr, x_mode)
        fig = make_dark_figure(title=title, xlabel=xlabel, ylabel="Power [kW]")
        for j, w in enumerate(WHEELS):
            fig.add_trace(
                go.Scatter(
                    x=x_arr,
                    y=pw_arr[order, j],
                    mode="lines+markers",
                    name=w,
                    line=dict(color=WHEEL_COLORS[w], width=2),
                    marker=dict(size=7),
                )
            )
        if x_mode == "laps":
            fig.update_xaxes(tickvals=np.sort(lp_arr.astype(int)))

    table: dict[str, object] = {"Lap": lp_arr, "LapTime [s]": np.round(lt_arr, 3)}
    for j, w in enumerate(WHEELS):
        table[f"{w} [kW]"] = np.round(pw_arr[:, j], 3)
    table["Total [kW]"] = np.round(p_total, 3)
    table["F/R %"] = np.round(p_front / p_total * 100, 1)
    if len(dfs) > 1:
        table["Run"] = run_all

    kpis = {
        "mean_total_kw": mean_total,
        "fr_pct": fr_pct,
        "lr_pct": lr_pct,
        "wheel_mean_kw": {w: float(np.nanmean(pw_arr[:, j])) for j, w in enumerate(WHEELS)},
        "wheel_pct": {
            w: float(np.nanmean(pw_arr[:, j]) / mean_total * 100) if mean_total > 0 else np.nan
            for j, w in enumerate(WHEELS)
        },
        "table": pl.DataFrame(table),
        "warnings": warnings,
    }
    return fig, kpis


# ── Inverter load (overload + i²t) ─────────────────────────────────────────────


def _inverter_load(
    df: pl.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str] | None:
    """Per-wheel (overload flag 0/1, load [·], mode) for the inverter-load figure.

    Prefers the real firmware channels `{w}_OverloadActive` + `{w}_IxTLoad` (i²t
    thermal budget, 1.0 = exhausted). On CAT18x logs that omit them, falls back to
    the **AC-current budget**: load = `|actualSCurrent| / max_ac_current`
    (1.0 = at the rated current) and overload = the inverter derating its own budget
    (`available_max_ac_current < max_ac_current`). Returns ``None`` if neither set
    is logged.
    """
    cols = set(df.columns)
    if all(f"{w}_OverloadActive" in cols and f"{w}_IxTLoad" in cols for w in WHEELS):
        arr = cols_to_numpy(
            df, [f"{w}_OverloadActive" for w in WHEELS] + [f"{w}_IxTLoad" for w in WHEELS]
        )
        ov = {w: (arr[f"{w}_OverloadActive"] == 1.0).astype(float) for w in WHEELS}
        load = {w: arr[f"{w}_IxTLoad"] for w in WHEELS}
        return ov, load, "i2t"
    if all(f"{w}_actualSCurrent" in cols and f"{w}_max_ac_current" in cols for w in WHEELS):
        need = [f"{w}_actualSCurrent" for w in WHEELS] + [f"{w}_max_ac_current" for w in WHEELS]
        need += [
            f"{w}_available_max_ac_current"
            for w in WHEELS
            if f"{w}_available_max_ac_current" in cols
        ]
        arr = cols_to_numpy(df, need)
        ov, load = {}, {}
        for w in WHEELS:
            imax = arr[f"{w}_max_ac_current"]
            good = np.isfinite(imax) & (imax > 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                load[w] = np.where(good, np.abs(arr[f"{w}_actualSCurrent"]) / imax, np.nan)
            av = f"{w}_available_max_ac_current"
            ov[w] = (
                np.where(good, arr[av] < imax - 1e-6, False).astype(float)
                if av in cols
                else np.zeros(len(imax))
            )
        return ov, load, "current"
    return None


def inverter_limits_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """How much of each inverter's overload / i²t thermal budget is being used?

    Two per-lap reads: % time the OverloadActive flag is set, and the P95 i²t load
    (IxTLoad, 1.0 = budget exhausted). On CAT18x logs without those channels it
    falls back to the AC-current budget (utilisation `|I|/Imax` + derating); the
    titles/axes reflect which. The front inverters run markedly harder than the
    rear in these logs. kpis: overload_pct / ixt_peak (per wheel, whole run),
    table, warnings. FWActive is flat-zero in these logs (not shown).
    """
    warnings: list[str] = []
    laps_all: list[int] = []
    ov_all: list[list[float]] = []
    ixt_all: list[list[float]] = []
    run_all: list[str] = []
    ov_run: dict[str, list[np.ndarray]] = {w: [] for w in WHEELS}
    ixt_run: dict[str, list[np.ndarray]] = {w: [] for w in WHEELS}
    mode = "i2t"

    for run_name, df in dfs.items():
        df = ensure_complete_laps_df(df)
        resolved = _inverter_load(df)
        if resolved is None:
            warnings.append(
                f"{run_name}: no inverter-load channels "
                "(OverloadActive+IxTLoad or actualSCurrent+max_ac_current)"
            )
            continue
        ov_w, load_w, mode = resolved
        laps = cols_to_numpy(df, ["laps"])["laps"]
        for w in WHEELS:
            ov_run[w].append(ov_w[w])
            ixt_run[w].append(load_w[w])
        for lap in unique_laps(laps):
            idx = laps == lap
            if idx.sum() < 50:
                continue
            ov = [float(np.nanmean(ov_w[w][idx]) * 100.0) for w in WHEELS]
            ix = [float(np.nanpercentile(load_w[w][idx], 95)) for w in WHEELS]
            laps_all.append(int(lap))
            ov_all.append(ov)
            ixt_all.append(ix)
            run_all.append(run_name)

    if not ov_all:
        return make_dark_figure(title="Inverter Limits — No data"), {
            "warnings": warnings + ["No overload / IxT data."]
        }

    lp = np.array(laps_all)
    ov = np.array(ov_all)
    ix = np.array(ixt_all)

    # Labels adapt to the source: real i²t budget vs the CAT18x current-budget fallback.
    if mode == "current":
        fig_title = "Inverter Current-Budget Utilisation"
        y_ov, y_load = "Derating [% lap]", "|I|/Imax P95 [–]"
        t_ov, t_load = "Time Derating per Lap", "Current Use per Lap (P95)"
        col_ov, col_load = "derating [%]", "|I|/Imax P95"
    else:
        fig_title = "Inverter Overload & i²t Utilisation"
        y_ov, y_load = "Overload [% lap]", "IxT P95 [–]"
        t_ov, t_load = "Time in Overload per Lap", "i²t Load per Lap (P95)"
        col_ov, col_load = "overload [%]", "IxT P95"
    lap_ticks = np.sort(np.unique(lp.astype(int)))
    run_names = list(dict.fromkeys(run_all))
    if len(run_names) > 1:
        # Rows = metric (overload / i²t), columns = wheel; one line per run.
        fig = make_subplots(
            rows=2,
            cols=4,
            subplot_titles=list(WHEELS),
            shared_xaxes=True,
            shared_yaxes=True,
            horizontal_spacing=0.045,
            vertical_spacing=0.10,
        )
        apply_dark_layout(fig, single_axes=False)
        for rn in run_names:
            m = np.array([r == rn for r in run_all])
            color = driver_color(rn)
            for j in range(len(WHEELS)):
                for row, vals in ((1, ov), (2, ix)):
                    fig.add_trace(
                        go.Scatter(
                            x=lp[m],
                            y=vals[m][:, j],
                            mode="lines+markers",
                            name=rn,
                            legendgroup=rn,
                            showlegend=(row == 1 and j == 0),
                            line=dict(color=color, width=2),
                            marker=dict(size=5),
                        ),
                        row=row,
                        col=j + 1,
                    )
        for j in range(1, len(WHEELS) + 1):
            fig.add_hline(y=1.0, line=dict(color="#E74C3C", dash="dash", width=1.0), row=2, col=j)
            for row in (1, 2):
                fig.update_xaxes(
                    title_text="Lap" if row == 2 else None,
                    tickvals=lap_ticks,
                    gridcolor=_GRID,
                    color=_AXIS,
                    row=row,
                    col=j,
                )
                fig.update_yaxes(gridcolor=_GRID, color=_AXIS, row=row, col=j)
        fig.update_yaxes(title_text=y_ov, row=1, col=1)
        fig.update_yaxes(title_text=y_load, row=2, col=1)
        fig.update_layout(height=520)
        for ann in fig.layout.annotations:
            ann.font.color = _TEXT
            ann.font.size = 12
    else:
        fig = _dark_subplots(
            rows=2,
            titles=[t_ov, t_load],
            ylabels=[y_ov, y_load],
        )
        for j, w in enumerate(WHEELS):
            fig.add_trace(
                go.Scatter(
                    x=lp,
                    y=ov[:, j],
                    mode="lines+markers",
                    name=w,
                    legendgroup=w,
                    line=dict(color=WHEEL_COLORS[w], width=2),
                    marker=dict(size=6),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=lp,
                    y=ix[:, j],
                    mode="lines+markers",
                    name=w,
                    legendgroup=w,
                    showlegend=False,
                    line=dict(color=WHEEL_COLORS[w], width=2),
                    marker=dict(size=6),
                ),
                row=2,
                col=1,
            )
        fig.add_hline(y=1.0, line=dict(color="#E74C3C", dash="dash", width=1.0), row=2, col=1)
        fig.update_xaxes(title_text="Lap", tickvals=lap_ticks, row=2, col=1)
        fig.update_layout(height=560)
    fig.update_layout(
        title_text=fig_title,
        title_font=dict(color=_TEXT, size=14),
    )

    table: dict[str, object] = {"Lap": lp}
    for j, w in enumerate(WHEELS):
        table[f"{w} {col_ov}"] = np.round(ov[:, j], 1)
        table[f"{w} {col_load}"] = np.round(ix[:, j], 3)
    if len(dfs) > 1:
        table["Run"] = run_all
    kpis = {
        "overload_pct": {
            w: float(np.nanmean(np.concatenate(ov_run[w]) == 1.0) * 100.0)
            for w in WHEELS
            if ov_run[w]
        },
        "ixt_peak": {w: float(np.nanmax(np.concatenate(ixt_run[w]))) for w in WHEELS if ixt_run[w]},
        "table": pl.DataFrame(table),
        "warnings": warnings,
    }
    return fig, kpis


# ── Torque fidelity (actual vs target) ─────────────────────────────────────────


def torque_fidelity_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """Do the inverters deliver the torque they are commanded?

    Per-lap MAE of (actual − target) torque per wheel, on samples with |target| >
    0.5 Nm. A signed mean bias exposes systematic under/over-delivery (the front
    inverters under-deliver slightly in these logs). kpis: mae_nm / bias_nm
    (per wheel), pct_within_1nm, samples, table, warnings.
    """
    warnings: list[str] = []
    laps_all: list[int] = []
    mae_all: list[list[float]] = []
    run_all: list[str] = []
    err_pool: list[np.ndarray] = []
    bias_pool: dict[str, list[np.ndarray]] = {w: [] for w in WHEELS}

    for run_name, df in dfs.items():
        df = ensure_complete_laps_df(df)
        needed = ["laps"] + [f"{w}_{s}" for w in WHEELS for s in ("actualTorque", "targetTorque")]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            warnings.append(f"{run_name}: missing {missing}")
            continue
        cols = cols_to_numpy(df, needed)
        laps = cols["laps"]
        for lap in unique_laps(laps):
            idx = laps == lap
            if idx.sum() < 50:
                continue
            row: list[float] = []
            for w in WHEELS:
                tgt = cols[f"{w}_targetTorque"][idx]
                act = cols[f"{w}_actualTorque"][idx]
                m = np.isfinite(tgt) & np.isfinite(act) & (np.abs(tgt) > 0.5)
                err = act[m] - tgt[m]
                row.append(float(np.nanmean(np.abs(err))) if err.size else np.nan)
                if err.size:
                    err_pool.append(err)
                    bias_pool[w].append(err)
            laps_all.append(int(lap))
            mae_all.append(row)
            run_all.append(run_name)

    if not mae_all:
        return make_dark_figure(title="Torque Fidelity — No data"), {
            "warnings": warnings + ["No valid torque target/actual data."]
        }

    lp = np.array(laps_all)
    mae = np.array(mae_all)
    err_cat = np.concatenate(err_pool)
    title = "Torque Tracking Error per Lap (MAE, |target| > 0.5 Nm)"
    lap_ticks = np.sort(np.unique(lp.astype(int)))
    run_names = list(dict.fromkeys(run_all))
    if len(run_names) > 1:
        grid_runs: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
        for rn in run_names:
            m = np.array([r == rn for r in run_all])
            mae_run = mae[m]
            grid_runs[rn] = (lp[m], {w: mae_run[:, j] for j, w in enumerate(WHEELS)})
        fig = _per_wheel_run_grid(
            grid_runs, title=title, xlabel="Lap", ylabel="MAE [Nm]", lap_ticks=lap_ticks
        )
    else:
        fig = make_dark_figure(title=title, xlabel="Lap", ylabel="MAE [Nm]")
        for j, w in enumerate(WHEELS):
            fig.add_trace(
                go.Scatter(
                    x=lp,
                    y=mae[:, j],
                    mode="lines+markers",
                    name=w,
                    line=dict(color=WHEEL_COLORS[w], width=2),
                    marker=dict(size=6),
                )
            )
        fig.update_xaxes(tickvals=lap_ticks)

    table: dict[str, object] = {"Lap": lp}
    for j, w in enumerate(WHEELS):
        table[f"{w} MAE [Nm]"] = np.round(mae[:, j], 3)
    if len(dfs) > 1:
        table["Run"] = run_all
    kpis = {
        "mae_nm": {w: float(np.nanmean(mae[:, j])) for j, w in enumerate(WHEELS)},
        "bias_nm": {
            w: float(np.nanmean(np.concatenate(bias_pool[w]))) if bias_pool[w] else np.nan
            for w in WHEELS
        },
        "pct_within_1nm": float(np.nanmean(np.abs(err_cat) <= 1.0) * 100.0),
        "samples": int(err_cat.size),
        "table": pl.DataFrame(table),
        "warnings": warnings,
    }
    return fig, kpis


# ── Torque–speed operating map ─────────────────────────────────────────────────


def torque_speed_envelope_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """Where in the torque-speed map does each motor operate, vs its limits?

    Per-wheel scatter of actualTorque vs actualVelocity [rad/s] against the ±27.5 Nm
    motor limit, the 20 kW (=80/4) power hyperbola and the rev-limit clamp. The
    saturation KPIs make it informative regardless of how hard the run was pushed.
    kpis per wheel: torque_p95_nm, speed_p95_rads, pct_torque_saturated,
    pct_rev_limited; plus samples, warnings.
    """
    warnings: list[str] = []
    pos = {"FL": (1, 1), "FR": (1, 2), "RL": (2, 1), "RR": (2, 2)}
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=list(WHEELS),
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    apply_dark_layout(fig, single_axes=False)

    pw_t: dict[str, list[np.ndarray]] = {w: [] for w in WHEELS}
    pw_w: dict[str, list[np.ndarray]] = {w: [] for w in WHEELS}

    for run_name, df in dfs.items():
        df = ensure_complete_laps_df(df)
        needed = [f"{w}_{s}" for w in WHEELS for s in ("actualTorque", "actualVelocity")]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            warnings.append(f"{run_name}: missing {missing}")
            continue
        for w in WHEELS:
            tq = df[f"{w}_actualTorque"].to_numpy().astype(float)
            om = df[f"{w}_actualVelocity"].to_numpy().astype(float)
            ok = np.isfinite(tq) & np.isfinite(om) & (om > 1.0)
            tq, om = tq[ok], om[ok]
            pw_t[w].append(tq)
            pw_w[w].append(om)
            sl = _decimate(len(tq))
            r, c = pos[w]
            fig.add_trace(
                go.Scattergl(
                    x=om[sl],
                    y=tq[sl],
                    mode="markers",
                    name=run_name,
                    legendgroup=run_name,
                    showlegend=(w == "FL"),
                    marker=dict(color=driver_color(run_name), size=3, opacity=0.35),
                ),
                row=r,
                col=c,
            )

    om_line = np.linspace(60.0, MOTOR_MAX_SPEED_RADS, 200)
    hyper = np.minimum(PER_MOTOR_POWER_CAP_W / om_line, MOTOR_MAX_TORQUE_NM * 1.2)
    for w, (r, c) in pos.items():
        for y in (MOTOR_MAX_TORQUE_NM, -MOTOR_MAX_TORQUE_NM):
            fig.add_hline(y=y, line=dict(color="#E74C3C", dash="dash", width=1.0), row=r, col=c)
        fig.add_trace(
            go.Scatter(
                x=om_line,
                y=hyper,
                mode="lines",
                name="20 kW (80/4)",
                legendgroup="cap",
                showlegend=(w == "FL"),
                line=dict(color="rgba(235,235,235,0.55)", dash="dot", width=1.2),
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(
            title_text="Motor speed [rad/s]" if r == 2 else None,
            gridcolor=_GRID,
            color=_AXIS,
            row=r,
            col=c,
        )
        fig.update_yaxes(
            title_text="Torque [Nm]" if c == 1 else None,
            gridcolor=_GRID,
            color=_AXIS,
            row=r,
            col=c,
        )

    rev_limit = 1113.44  # Param_desiredMaximumVelocity clamp [rad/s]
    wheel_kpis: dict[str, dict[str, float]] = {}
    n_total = 0
    for w in WHEELS:
        if not pw_t[w]:
            continue
        tq = np.concatenate(pw_t[w])
        om = np.concatenate(pw_w[w])
        drive = tq > 1.0
        n_total += len(tq)
        wheel_kpis[w] = {
            "torque_p95_nm": float(np.nanpercentile(tq[drive], 95)) if drive.any() else np.nan,
            "speed_p95_rads": float(np.nanpercentile(om, 95)),
            "pct_torque_saturated": float(
                np.nanmean(np.abs(tq) > 0.9 * MOTOR_MAX_TORQUE_NM) * 100.0
            ),
            "pct_rev_limited": float(np.nanmean(om > 0.95 * rev_limit) * 100.0),
        }
    fig.update_layout(
        height=640,
        title=dict(
            text=f"Torque–Speed Operating Map (limits ±{MOTOR_MAX_TORQUE_NM:.1f} Nm, 20 kW)",
            font=dict(color=_TEXT, size=14),
        ),
    )
    for ann in fig.layout.annotations:
        ann.font.color = _TEXT
        ann.font.size = 12
    return fig, {"wheels": wheel_kpis, "samples": n_total, "warnings": warnings}


# ── SoC per lap ────────────────────────────────────────────────────────────────


def soc_per_lap_fig(
    dfs: dict[str, pl.DataFrame],
    x_mode: str = "laps",
) -> tuple[go.Figure, dict]:
    """Battery autonomy: end-of-lap SoC and the per-lap SoC drop.

    SoC is high-resolution and trustworthy in these logs (Ah/%SoC is stable
    run-to-run). kpis: soc_start, soc_end, soc_total_drop, soc_drop_per_lap,
    table, warnings (start/end only meaningful for one continuous run).
    """
    cols_needed = ["laps", "laptime", "SoC"]
    laps_all: list[int] = []
    lt_all: list[float] = []
    soc_all: list[float] = []
    sd_all: list[float] = []
    run_all: list[str] = []
    warnings: list[str] = []
    run_soc_start: list[float] = []
    run_soc_end: list[float] = []

    for run_name, df in dfs.items():
        df = ensure_complete_laps_df(df)
        missing = [c for c in cols_needed if c not in df.columns]
        if missing:
            warnings.append(f"{run_name}: missing {missing}")
            continue
        cols = cols_to_numpy(df, cols_needed)
        laps, laptime, soc_arr = cols["laps"], cols["laptime"], cols["SoC"]
        run_laps = unique_laps(laps)
        if len(run_laps) > 0:
            first = np.where(laps == run_laps[0])[0]
            last = np.where(laps == run_laps[-1])[0]
            if len(first) and len(last):
                run_soc_start.append(float(soc_arr[first[0]]))
                run_soc_end.append(float(soc_arr[last[-1]]))
        for lap in run_laps:
            idx = np.where(laps == lap)[0]
            if len(idx) < 5:
                continue
            end_v = float(soc_arr[idx[-1]])
            if not np.isfinite(end_v):
                continue
            laps_all.append(int(lap))
            lt_all.append(float(np.nanmax(laptime[idx])))
            soc_all.append(end_v)
            sd_all.append(float(soc_arr[idx[0]] - soc_arr[idx[-1]]))
            run_all.append(run_name)

    if not laps_all:
        return make_dark_figure(title="SoC per Lap — No data"), {
            "warnings": warnings + ["No valid SoC data."]
        }

    lp = np.array(laps_all)
    lt = np.array(lt_all)
    soc = np.array(soc_all)
    sd = np.array(sd_all)
    if len(run_soc_start) == 1:
        soc_start, soc_end = run_soc_start[0], run_soc_end[0]
        soc_total_drop = soc_start - soc_end
    else:
        soc_start = soc_end = soc_total_drop = np.nan
        if len(run_soc_start) > 1:
            warnings.append("SoC start/end only meaningful for a single continuous run.")

    x_arr, order, xlabel = per_lap_axis(lp, lt, x_mode)
    fig = _dark_subplots(
        rows=2,
        titles=["End-of-Lap SoC", "SoC Drop per Lap"],
        ylabels=["SoC [%]", "ΔSoC [%]"],
        vertical_spacing=0.16,
    )
    fig.add_trace(
        go.Scatter(
            x=x_arr,
            y=soc[order],
            mode="lines+markers",
            name="SoC",
            line=dict(color="#4DB3F2", width=2),
            marker=dict(size=7),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=x_arr, y=sd[order], name="ΔSoC", marker=dict(color="#F28C40")),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text=xlabel, row=2, col=1)
    if x_mode == "laps":
        fig.update_xaxes(tickvals=np.sort(lp.astype(int)), row=2, col=1)
    fig.update_layout(
        height=620,
        title_text="Battery SoC per Lap",
        title_font=dict(color=_TEXT, size=14),
    )

    table: dict[str, object] = {
        "Lap": lp,
        "LapTime [s]": np.round(lt, 3),
        "SoC [%]": np.round(soc, 1),
        "dSoC [%]": np.round(sd, 2),
    }
    if len(dfs) > 1:
        table["Run"] = run_all
    kpis = {
        "soc_start": soc_start,
        "soc_end": soc_end,
        "soc_total_drop": soc_total_drop,
        "soc_drop_per_lap": float(np.nanmean(sd)),
        "table": pl.DataFrame(table),
        "warnings": warnings,
    }
    return fig, kpis


# ── HV delivery efficiency ─────────────────────────────────────────────────────


def hv_delivery_efficiency_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """How much pack DC power actually reaches the inverters?

    Measured-vs-measured: Σ(inverter electrical power) ÷ (Vbat·Current) under drive
    (P_bat > 3 kW). The shortfall is HV-distribution loss (wiring, contactors, busbars).
    kpis: delivery_eff_median, mean_loss_kw, samples, table, warnings.
    """
    warnings: list[str] = []
    laps_all: list[int] = []
    eff_all: list[float] = []
    loss_all: list[float] = []
    run_all: list[str] = []
    eff_pool: list[np.ndarray] = []
    loss_pool: list[np.ndarray] = []

    for run_name, df in dfs.items():
        df = ensure_complete_laps_df(df)
        needed = ["laps", "Vbat", "Current"]
        missing = [c for c in needed if c not in df.columns]
        resolved = _inverter_input_power(df)
        if missing or resolved is None:
            miss = list(missing)
            if resolved is None:
                miss.append("inverter power (actualPower or dc_current×dc_bus_voltage)")
            warnings.append(f"{run_name}: missing {miss}")
            continue
        p_inv, _src = resolved
        cols = cols_to_numpy(df, needed)
        laps = cols["laps"]
        p_bat = _clean_band(cols["Vbat"], _PACK_V_BAND) * cols["Current"]  # W
        for lap in unique_laps(laps):
            idx = laps == lap
            drive = idx & np.isfinite(p_inv) & np.isfinite(p_bat) & (p_bat > 3000.0)
            if int(drive.sum()) < 50:
                continue
            eff = p_inv[drive] / p_bat[drive]
            loss_kw = (p_bat[drive] - p_inv[drive]) / 1000.0
            eff_all.append(float(np.nanmedian(eff)))
            loss_all.append(float(np.nanmean(loss_kw)))
            laps_all.append(int(lap))
            run_all.append(run_name)
            eff_pool.append(eff)
            loss_pool.append(loss_kw)

    if not eff_all:
        return make_dark_figure(title="HV Delivery Efficiency — No data"), {
            "warnings": warnings + ["No valid drive samples for HV efficiency."]
        }

    lp = np.array(laps_all)
    eff = np.array(eff_all)
    fig = make_dark_figure(
        title="HV Delivery Efficiency per Lap  (Σ inverter P ÷ battery P, drive only)",
        xlabel="Lap",
        ylabel="Delivery efficiency [–]",
    )
    if len(dfs) > 1:
        for run_name in dict.fromkeys(run_all):
            m = np.array([r == run_name for r in run_all])
            fig.add_trace(
                go.Scatter(
                    x=lp[m],
                    y=eff[m],
                    mode="lines+markers",
                    name=run_name,
                    line=dict(color=driver_color(run_name), width=1.5),
                    marker=dict(size=6),
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=lp,
                y=eff,
                mode="lines+markers",
                name="η_delivery",
                line=dict(color="#4DB3F2", width=1.8),
                marker=dict(size=7),
            )
        )
        fig.update_xaxes(tickvals=np.sort(lp.astype(int)))
    fig.add_hline(y=1.0, line=dict(color="rgba(235,235,235,0.4)", dash="dot", width=1.0))

    table: dict[str, object] = {
        "Lap": lp,
        "η delivery": np.round(eff, 4),
        "Mean loss [kW]": np.round(np.array(loss_all), 2),
    }
    if len(dfs) > 1:
        table["Run"] = run_all
    eff_cat = np.concatenate(eff_pool)
    loss_cat = np.concatenate(loss_pool)
    runs: dict[str, dict] = {}
    for run_name in dict.fromkeys(run_all):
        m = [r == run_name for r in run_all]
        eff_r = np.concatenate([p for p, keep in zip(eff_pool, m) if keep])
        loss_r = np.concatenate([p for p, keep in zip(loss_pool, m) if keep])
        runs[run_name] = {
            "delivery_eff_median": float(np.nanmedian(eff_r)),
            "mean_loss_kw": float(np.nanmean(loss_r)),
            "samples": int(eff_r.size),
        }
    kpis = {
        "delivery_eff_median": float(np.nanmedian(eff_cat)),
        "mean_loss_kw": float(np.nanmean(loss_cat)),
        "samples": int(eff_cat.size),
        "runs": runs,
        "table": pl.DataFrame(table),
        "warnings": warnings,
    }
    return fig, kpis


# ── Weakest cell under load ────────────────────────────────────────────────────


def weakest_cell_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """Does the weakest cell approach its discharge floor when current is drawn?

    Scatter of the cleaned minimum cell voltage (Vmin) vs battery current. A safety
    read, not a resistance fit. kpis: vmin_under_load_v (min clean Vmin at I>20 A),
    vmin_at_peak_current_v, samples, warnings.
    """
    warnings: list[str] = []
    runs: dict[str, dict[str, float]] = {}
    fig = make_dark_figure(
        title="Weakest Cell Voltage vs Current",
        xlabel="Battery current [A] (+ = discharge)",
        ylabel="Min cell voltage [V]",
    )
    any_data = False
    for run_name, df in dfs.items():
        df = ensure_complete_laps_df(df)
        missing = [c for c in ("Vmin", "Current") if c not in df.columns]
        if missing:
            warnings.append(f"{run_name}: missing {missing}")
            continue
        vmin = _clean_band(df["Vmin"].to_numpy(), _CELL_V_BAND)
        cur = df["Current"].to_numpy().astype(float)
        ok = np.isfinite(vmin) & np.isfinite(cur)
        vmin, cur = vmin[ok], cur[ok]
        if vmin.size < 100:
            warnings.append(f"{run_name}: not enough clean Vmin samples.")
            continue
        any_data = True
        sl = _decimate(len(vmin))
        fig.add_trace(
            go.Scattergl(
                x=cur[sl],
                y=vmin[sl],
                mode="markers",
                name=run_name,
                marker=dict(color=driver_color(run_name), size=3, opacity=0.30),
            )
        )
        load = cur > 20.0
        under = vmin[load]
        i_peak = int(np.argmax(cur))
        runs[run_name] = {
            "vmin_under_load_v": float(np.nanmin(under)) if under.size else np.nan,
            "vmin_at_peak_current_v": float(vmin[i_peak]),
            "samples": int(vmin.size),
        }
    if not any_data:
        return make_dark_figure(title="Weakest Cell Under Load — No data"), {
            "runs": runs,
            "warnings": warnings + ["No valid cell-voltage data."],
        }
    fig.add_hline(
        y=WEAKEST_CELL_DISCHARGE_FLOOR_V,
        line=dict(color="#E74C3C", dash="dash", width=1.2),
        annotation_text=f"discharge floor {WEAKEST_CELL_DISCHARGE_FLOOR_V:.2f} V",
        annotation_position="bottom right",
    )
    return fig, {"runs": runs, "warnings": warnings}


# ── Thermal evolution + headroom ───────────────────────────────────────────────


def _thermal_lap_stats(
    dfs: dict[str, pl.DataFrame],
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    """Per-lap P95 temps: motors/inverters per wheel + glitch-cleaned battery.

    Returns (data, run_all, warnings) where data has keys laps, laptime, Tm (N,4),
    Ti (N,4), Tb (N,2). All temps are band-cleaned (_TEMP_BAND_C); battery 'Tmax'
    uses lap P99 of the cleaned signal (never nanmax — it glitches to ~600 °C).
    """
    motor_cols = [f"{w}_motorTemperature" for w in WHEELS]
    inv_cols = [f"{w}_inverterTemperature" for w in WHEELS]
    cols_needed = ["laps", "laptime"] + motor_cols + inv_cols + ["Tmax", "Tavg"]
    laps_all: list[int] = []
    lt_all: list[float] = []
    tm_all: list[list[float]] = []
    ti_all: list[list[float]] = []
    tb_all: list[list[float]] = []
    run_all: list[str] = []
    warnings: list[str] = []

    for run_name, df in dfs.items():
        df = ensure_complete_laps_df(df)
        missing = [c for c in cols_needed if c not in df.columns]
        if missing:
            warnings.append(f"{run_name}: missing {missing}")
            continue
        cols = cols_to_numpy(df, cols_needed)
        for c in (*motor_cols, *inv_cols, "Tmax", "Tavg"):
            cols[c] = _clean_band(cols[c], _TEMP_BAND_C)
        laps = cols["laps"]
        for lap in unique_laps(laps):
            idx = laps == lap
            if idx.sum() < 50:
                continue
            tm = [float(np.nanpercentile(cols[f"{w}_motorTemperature"][idx], 95)) for w in WHEELS]
            ti = [
                float(np.nanpercentile(cols[f"{w}_inverterTemperature"][idx], 95)) for w in WHEELS
            ]
            tb = [
                float(np.nanpercentile(cols["Tmax"][idx], 99)),
                float(np.nanpercentile(cols["Tavg"][idx], 95)),
            ]
            if not all(np.isfinite(tm)):
                continue
            laps_all.append(int(lap))
            lt_all.append(float(np.nanmax(cols["laptime"][idx])))
            tm_all.append(tm)
            ti_all.append(ti)
            tb_all.append(tb)
            run_all.append(run_name)

    data = {
        "laps": np.array(laps_all),
        "laptime": np.array(lt_all),
        "Tm": np.array(tm_all),
        "Ti": np.array(ti_all),
        "Tb": np.array(tb_all),
    }
    return data, run_all, warnings


def _thermal_kpis(
    lp: np.ndarray,
    lt: np.ndarray,
    Tm: np.ndarray,
    Ti: np.ndarray,
    Tb: np.ndarray,
    run_all: list[str],
    n_runs: int,
    warnings: list[str],
) -> dict:
    """Whole-run thermal KPIs + per-lap table (shared by single- and multi-run)."""

    def _slope(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else np.nan

    table: dict[str, object] = {"Lap": lp, "LapTime [s]": np.round(lt, 3)}
    for j, w in enumerate(WHEELS):
        table[f"Motor {w} [°C]"] = np.round(Tm[:, j], 1)
        table[f"Inv {w} [°C]"] = np.round(Ti[:, j], 1)
    table["Batt Tmax [°C]"] = np.round(Tb[:, 0], 1)
    table["Batt Tavg [°C]"] = np.round(Tb[:, 1], 1)
    if n_runs > 1:
        table["Run"] = run_all
    return {
        "peak_motor": float(np.nanmax(Tm)),
        "peak_inverter": float(np.nanmax(Ti)),
        "peak_batt_tmax": float(np.nanmax(Tb[:, 0])),
        "motor_thermal_slope": _slope(lp, np.nanmean(Tm, axis=1)),
        "motor_peak_by_wheel": {w: float(np.nanmax(Tm[:, j])) for j, w in enumerate(WHEELS)},
        "table": pl.DataFrame(table),
        "warnings": warnings,
    }


def _thermal_multirun_fig(
    run_names: list[str],
    run_all: list[str],
    lp: np.ndarray,
    lt: np.ndarray,
    Tm: np.ndarray,
    Ti: np.ndarray,
    Tb: np.ndarray,
    x_mode: str,
    lap_ticks: np.ndarray,
) -> go.Figure:
    """Multi-run thermal grid: motors (4 wheels) · inverters (4 wheels) · battery.

    One line per run coloured by driver_color; wheel identity is the column.
    Battery spans the bottom row (Tmax solid · Tavg dot).
    """
    fig = make_subplots(
        rows=3,
        cols=4,
        specs=[[{}, {}, {}, {}], [{}, {}, {}, {}], [{"colspan": 4}, None, None, None]],
        subplot_titles=[*WHEELS, "", "", "", "", "Battery — Tmax (solid) · Tavg (dot)"],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        vertical_spacing=0.13,
        row_heights=[0.3, 0.3, 0.4],
    )
    apply_dark_layout(fig, single_axes=False)
    xlabel = "Lap time [s]" if x_mode == "laptime" else "Lap"
    for rn in run_names:
        m = np.array([r == rn for r in run_all])
        x_run, order_run, _ = per_lap_axis(lp[m], lt[m], x_mode)
        color = driver_color(rn)
        Tm_r, Ti_r, Tb_r = Tm[m][order_run], Ti[m][order_run], Tb[m][order_run]
        for j in range(len(WHEELS)):
            fig.add_trace(
                go.Scatter(
                    x=x_run,
                    y=Tm_r[:, j],
                    mode="lines+markers",
                    name=rn,
                    legendgroup=rn,
                    showlegend=(j == 0),
                    line=dict(color=color, width=2),
                    marker=dict(size=5),
                ),
                row=1,
                col=j + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_run,
                    y=Ti_r[:, j],
                    mode="lines+markers",
                    name=rn,
                    legendgroup=rn,
                    showlegend=False,
                    line=dict(color=color, width=2),
                    marker=dict(size=5),
                ),
                row=2,
                col=j + 1,
            )
        fig.add_trace(
            go.Scatter(
                x=x_run,
                y=Tb_r[:, 0],
                mode="lines+markers",
                name=rn,
                legendgroup=rn,
                showlegend=False,
                line=dict(color=color, width=2),
                marker=dict(size=5),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_run,
                y=Tb_r[:, 1],
                mode="lines",
                name=rn,
                legendgroup=rn,
                showlegend=False,
                line=dict(color=color, dash="dot", width=1.4),
            ),
            row=3,
            col=1,
        )
    ticks = lap_ticks if x_mode == "laps" else None
    for j in range(1, len(WHEELS) + 1):
        fig.add_hline(
            y=MOTOR_OT_LIMIT_C, line=dict(color="#E74C3C", dash="dash", width=1.0), row=1, col=j
        )
        fig.add_hline(
            y=INVERTER_OT_LIMIT_C, line=dict(color="#E74C3C", dash="dash", width=1.0), row=2, col=j
        )
        for row in (1, 2):
            fig.update_xaxes(
                tickvals=ticks,
                gridcolor=_GRID,
                color=_AXIS,
                row=row,
                col=j,
            )
            fig.update_yaxes(gridcolor=_GRID, color=_AXIS, row=row, col=j)
    fig.add_hline(
        y=BATTERY_TEMP_LIMIT_C, line=dict(color="#E74C3C", dash="dash", width=1.0), row=3, col=1
    )
    fig.update_xaxes(title_text=xlabel, tickvals=ticks, gridcolor=_GRID, color=_AXIS, row=3, col=1)
    fig.update_yaxes(gridcolor=_GRID, color=_AXIS, row=3, col=1)
    fig.update_yaxes(title_text="Motor T [°C]", row=1, col=1)
    fig.update_yaxes(title_text="Inverter T [°C]", row=2, col=1)
    fig.update_yaxes(title_text="Battery T [°C]", row=3, col=1)
    fig.update_layout(
        height=980,
        title=dict(
            text="Thermal Evolution per Lap (P95, glitch-cleaned) — limits 120 / 75 / 60 °C",
            font=dict(color=_TEXT, size=14),
        ),
    )
    for ann in fig.layout.annotations:
        ann.font.color = _TEXT
        ann.font.size = 12
    return fig


def thermal_evolution_fig(
    dfs: dict[str, pl.DataFrame],
    x_mode: str = "laps",
) -> tuple[go.Figure, dict]:
    """Motor / inverter / battery temperatures per lap (P95, glitch-cleaned).

    Single run: three stacked subplots (motors/inverters/battery), each with its
    OT-limit line and per-wheel lines. Multi-run: per-wheel small multiples with
    one line per run (driver_color), battery on a wide bottom panel. kpis:
    peak_motor, peak_inverter, peak_batt_tmax, motor_thermal_slope,
    motor_peak_by_wheel, table, warnings.
    """
    data, run_all, warnings = _thermal_lap_stats(dfs)
    if data["laps"].size == 0:
        return make_dark_figure(title="Thermal Evolution — No data"), {
            "warnings": warnings + ["No valid thermal data."]
        }
    lp, lt, Tm, Ti, Tb = data["laps"], data["laptime"], data["Tm"], data["Ti"], data["Tb"]
    run_names = list(dict.fromkeys(run_all))
    lap_ticks = np.sort(np.unique(lp.astype(int)))
    if len(run_names) > 1:
        fig = _thermal_multirun_fig(run_names, run_all, lp, lt, Tm, Ti, Tb, x_mode, lap_ticks)
        return fig, _thermal_kpis(lp, lt, Tm, Ti, Tb, run_all, len(dfs), warnings)
    x_arr, order, xlabel = per_lap_axis(lp, lt, x_mode)

    fig = _dark_subplots(
        rows=3,
        titles=[
            f"Motors (limit {MOTOR_OT_LIMIT_C:.0f} °C)",
            f"Inverters (limit {INVERTER_OT_LIMIT_C:.0f} °C)",
            f"Battery (FS rule {BATTERY_TEMP_LIMIT_C:.0f} °C)",
        ],
        ylabels=["T [°C]", "T [°C]", "T [°C]"],
    )
    for j, w in enumerate(WHEELS):
        fig.add_trace(
            go.Scatter(
                x=x_arr,
                y=Tm[order, j],
                mode="lines+markers",
                name=w,
                legendgroup=w,
                line=dict(color=WHEEL_COLORS[w], width=1.5),
                marker=dict(size=6),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_arr,
                y=Ti[order, j],
                mode="lines+markers",
                name=w,
                legendgroup=w,
                showlegend=False,
                line=dict(color=WHEEL_COLORS[w], width=1.5),
                marker=dict(size=6),
            ),
            row=2,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=x_arr,
            y=Tb[order, 0],
            mode="lines+markers",
            name="Batt Tmax",
            line=dict(color="white", width=2),
            marker=dict(size=6),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_arr,
            y=Tb[order, 1],
            mode="lines",
            name="Batt Tavg",
            line=dict(color="white", dash="dot", width=1.2),
        ),
        row=3,
        col=1,
    )
    for row, limit in ((1, MOTOR_OT_LIMIT_C), (2, INVERTER_OT_LIMIT_C), (3, BATTERY_TEMP_LIMIT_C)):
        fig.add_hline(y=limit, line=dict(color="#E74C3C", dash="dash", width=1.2), row=row, col=1)
    fig.update_xaxes(title_text=xlabel, row=3, col=1)
    if x_mode == "laps":
        fig.update_xaxes(tickvals=np.sort(np.unique(lp.astype(int))), row=3, col=1)
    fig.update_layout(
        height=820,
        title_text="Thermal Evolution per Lap (P95, glitch-cleaned)",
        title_font=dict(color=_TEXT, size=14),
    )
    return fig, _thermal_kpis(lp, lt, Tm, Ti, Tb, run_all, len(dfs), warnings)


def thermal_headroom_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """Margin to the OT limits and heat-soak rate per component.

    Headroom = limit − worst observed lap P95 (worst wheel). Slope = °C/lap of the
    component mean. laps_to_limit = headroom / slope (∞ if slope ≤ 0.05 °C/lap).
    kpis per component (motor/inverter/battery): peak_c, headroom_c, slope_c_per_lap,
    laps_to_limit; plus dt_lr_motor_c / dt_lr_inverter_c (left − right), warnings.
    """
    data, _run_all, warnings = _thermal_lap_stats(dfs)
    if data["laps"].size == 0:
        return make_dark_figure(title="Thermal Headroom — No data"), {
            "warnings": warnings + ["No valid thermal data."]
        }
    lp, Tm, Ti, Tb = data["laps"], data["Tm"], data["Ti"], data["Tb"]

    def _slope(y: np.ndarray) -> float:
        return float(np.polyfit(lp, y, 1)[0]) if len(lp) >= 2 else np.nan

    comps = {
        "Motor": (float(np.nanmax(Tm)), MOTOR_OT_LIMIT_C, _slope(np.nanmean(Tm, axis=1))),
        "Inverter": (float(np.nanmax(Ti)), INVERTER_OT_LIMIT_C, _slope(np.nanmean(Ti, axis=1))),
        "Battery": (float(np.nanmax(Tb[:, 0])), BATTERY_TEMP_LIMIT_C, _slope(Tb[:, 0])),
    }
    names: list[str] = []
    headrooms: list[float] = []
    colors: list[str] = []
    texts: list[str] = []
    comp_kpis: dict[str, dict[str, float]] = {}
    for name, (peak, limit, slope) in comps.items():
        headroom = limit - peak
        laps_to = (
            headroom / slope if (np.isfinite(slope) and slope > 0.05 and headroom > 0) else np.inf
        )
        comp_kpis[name.lower()] = {
            "peak_c": peak,
            "headroom_c": headroom,
            "slope_c_per_lap": slope,
            "laps_to_limit": laps_to,
        }
        names.append(name)
        headrooms.append(headroom)
        colors.append(
            "#F25555"
            if (np.isfinite(laps_to) and laps_to < 8)
            else "#F2C744"
            if (np.isfinite(laps_to) and laps_to < 20)
            else "#73D973"
        )
        lt_txt = "∞" if np.isinf(laps_to) else f"{laps_to:.0f} laps"
        texts.append(f"{headroom:+.1f} °C · {slope:+.2f} °C/lap · {lt_txt} to limit")

    fig = make_dark_figure(
        title="Thermal Headroom to OT Limit (worst wheel, P95)",
        xlabel="Headroom [°C]",
        ylabel="",
    )
    fig.add_trace(
        go.Bar(
            x=headrooms,
            y=names,
            orientation="h",
            marker=dict(color=colors),
            text=texts,
            textposition="auto",
        )
    )
    fig.add_vline(x=0.0, line=dict(color="#E74C3C", dash="dash", width=1.4))
    fig.update_layout(height=360)

    kpis = {
        **comp_kpis,
        "dt_lr_motor_c": float(np.nanmean((Tm[:, 0] + Tm[:, 2]) / 2 - (Tm[:, 1] + Tm[:, 3]) / 2)),
        "dt_lr_inverter_c": float(
            np.nanmean((Ti[:, 0] + Ti[:, 2]) / 2 - (Ti[:, 1] + Ti[:, 3]) / 2)
        ),
        "warnings": warnings,
    }
    return fig, kpis


# ═══════════════════════════════════════════════════════════════════════════════
# Function check  —  is Power Control keeping P_bat under 80 kW (FS rule)?
# ═══════════════════════════════════════════════════════════════════════════════

POWER_CAP_KW = 80.0
POWER_NEAR_CAP_KW = 70.0
OVERSHOOT_MIN_DURATION_S = 0.05


def _count_overshoot_events(over: np.ndarray, dt: float, min_dur: float) -> int:
    """Count contiguous True segments of *over* lasting >= *min_dur*."""
    if not over.any():
        return 0
    min_samples = max(1, int(np.ceil(min_dur / dt)))
    padded = np.concatenate([[False], over.astype(bool), [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return int(np.sum((ends - starts + 1) >= min_samples))


def pc_function_kpis(df: pl.DataFrame) -> tuple[list[go.Figure], dict]:
    """Function-level check for Power Control.

    Pregunta: ¿está PC manteniendo la potencia bajo el techo de 80 kW
    sin dejar potencia en la mesa cuando el piloto pide a fondo?
    """
    df = ensure_complete_laps_df(df)
    needed = ["TimeStamp", "laps", "laptime", "Throttle", "Vbat", "Current"]
    arr = cols_to_numpy(df, needed)
    finite = np.all(np.stack([np.isfinite(arr[c]) for c in needed], axis=1), axis=1)
    arr = {c: v[finite] for c, v in arr.items()}
    keep = arr["laps"] > 0
    arr = {c: v[keep] for c, v in arr.items()}
    if arr["TimeStamp"].size == 0:
        raise ValueError("No valid samples for PC function check.")
    last_lap = unique_laps(arr["laps"]).max()
    keep = arr["laps"] != last_lap
    arr = {c: v[keep] for c, v in arr.items()}

    time_s = arr["TimeStamp"] - arr["TimeStamp"][0]
    diffs = np.diff(time_s)
    valid_dt = diffs[(diffs > 0) & np.isfinite(diffs)]
    dt = float(np.median(valid_dt)) if len(valid_dt) else 0.01

    p_kw = arr["Vbat"] * arr["Current"] / 1000.0  # positive = drain
    over_cap = p_kw > POWER_CAP_KW
    pct_over_cap = float(over_cap.mean() * 100.0)
    n_overshoot = _count_overshoot_events(over_cap, dt, OVERSHOOT_MIN_DURATION_S)

    full = arr["Throttle"] >= 80.0
    near_cap_at_full = (
        float(
            ((p_kw >= POWER_NEAR_CAP_KW) & (p_kw <= POWER_CAP_KW) & full).sum()
            / max(full.sum(), 1)
            * 100.0
        )
        if full.any()
        else np.nan
    )
    peak_kw = float(np.nanmax(p_kw)) if p_kw.size else np.nan

    laps_arr = arr["laps"]
    lap_list = unique_laps(laps_arr)
    peak_per_lap = []
    pct_at_cap_per_lap = []
    for lap in lap_list:
        lm = laps_arr == lap
        if not lm.any():
            peak_per_lap.append(np.nan)
            pct_at_cap_per_lap.append(np.nan)
            continue
        peak_per_lap.append(float(np.nanmax(p_kw[lm])))
        full_lap = full & lm
        if full_lap.any():
            near_lap = (p_kw[full_lap] >= POWER_NEAR_CAP_KW) & (p_kw[full_lap] <= POWER_CAP_KW)
            pct_at_cap_per_lap.append(float(near_lap.mean() * 100.0))
        else:
            pct_at_cap_per_lap.append(np.nan)
    lap_ids = lap_list.astype(int).tolist()

    # ── Fig 1: P_bat vs time with 80 kW reference line ────────────────────────
    fig_p = make_dark_figure(
        title=f"Battery power vs time (cap = {POWER_CAP_KW:.0f} kW)",
        xlabel="Time [s]",
        ylabel="P_bat [kW]",
    )
    fig_p.add_trace(
        go.Scattergl(
            x=time_s,
            y=p_kw,
            mode="lines",
            name="P_bat",
            line=dict(color="#4DB3F2", width=1.0),
        )
    )
    fig_p.add_hline(
        y=POWER_CAP_KW,
        line=dict(color="#E74C3C", dash="dash", width=1.4),
        annotation_text=f"FS rule {POWER_CAP_KW:.0f} kW",
        annotation_position="top right",
    )
    fig_p.add_hline(
        y=POWER_NEAR_CAP_KW,
        line=dict(color="rgba(115, 217, 115, 0.6)", dash="dot", width=1.0),
        annotation_text=f"{POWER_NEAR_CAP_KW:.0f} kW",
        annotation_position="bottom right",
    )

    # ── Fig 2: histogram of P_bat at full throttle ────────────────────────────
    fig_hist = make_dark_figure(
        title="P_bat distribution at full throttle (Throttle ≥ 80 %)",
        xlabel="P_bat [kW]",
        ylabel="Density",
    )
    if full.any():
        fig_hist.add_trace(
            go.Histogram(
                x=p_kw[full],
                name="P_bat | full throttle",
                histnorm="probability density",
                marker=dict(color="#F28C40"),
                opacity=0.85,
                nbinsx=80,
            )
        )
    fig_hist.add_vrect(
        x0=POWER_NEAR_CAP_KW, x1=POWER_CAP_KW, fillcolor="rgba(115, 217, 115, 0.10)", line_width=0
    )
    fig_hist.add_vline(x=POWER_CAP_KW, line=dict(color="#E74C3C", dash="dash", width=1.4))

    # ── Fig 3: peak P_bat per lap ─────────────────────────────────────────────
    fig_peak = make_dark_figure(
        title="Peak P_bat per lap",
        xlabel="Lap",
        ylabel="Peak P_bat [kW]",
    )
    fig_peak.add_trace(
        go.Bar(
            x=lap_ids,
            y=peak_per_lap,
            marker=dict(color="#9B59B6"),
            name="Peak",
            text=[f"{p:.1f}" if np.isfinite(p) else "" for p in peak_per_lap],
            textposition="outside",
        )
    )
    fig_peak.add_hline(y=POWER_CAP_KW, line=dict(color="#E74C3C", dash="dash", width=1.4))

    kpis = {
        "pct_over_cap": pct_over_cap,
        "n_overshoot_events": int(n_overshoot),
        "peak_kw": peak_kw,
        "pct_near_cap_at_full": near_cap_at_full,
        "peak_kw_per_lap": dict(zip(lap_ids, peak_per_lap)),
        "pct_at_cap_per_lap": dict(zip(lap_ids, pct_at_cap_per_lap)),
    }
    return [fig_p, fig_hist, fig_peak], kpis
