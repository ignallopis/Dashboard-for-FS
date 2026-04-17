"""powertrain.py
--------------
Powertrain KPIs:
  1. Energy per lap       (kWh, mean power, trend, correlation)
  2. Power per wheel      (per-wheel distribution, front/rear & left/right balance)
  3. Battery status       (SoC evolution, voltage sag, cell balance)
  4. Thermal evolution    (motor, inverter, battery temps + thermal slope)

Public API:
    Standalone CLI (use CSV_PATH):
        energy_per_lap()      -> go.Figure
        power_per_wheel()     -> go.Figure
        battery_status()      -> go.Figure
        thermal_evolution()   -> go.Figure
        main()                — calls all of the above + .show()

    Dashboard (take dfs dict):
        energy_per_lap_fig(dfs)    -> tuple[go.Figure, dict]
        power_per_wheel_fig(dfs)   -> tuple[go.Figure, dict]
        battery_status_fig(dfs)    -> tuple[go.Figure, dict]
        thermal_evolution_fig(dfs) -> tuple[go.Figure, dict]

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
from scipy.integrate import cumulative_trapezoid

from utils import (
    make_dark_figure,
    add_lap_scatter,
    add_trend_line,
    ensure_complete_laps_df,
    exclude_lap0_and_last_lap,
    per_lap_axis,
    unique_laps,
    WHEEL_COLORS,
)

CSV_PATH = "data/run4_2025-08-24.csv"
WHEELS = ("FL", "FR", "RL", "RR")

# Dark theme (mirrors utils.py for subplot figures)
_BG = "#141417"
_TEXT = "#EBEBEB"
_GRID = "rgba(128,128,128,0.2)"
_AXIS = "#E5E5E5"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load(columns: list[str]) -> dict[str, np.ndarray]:
    df = pl.read_csv(CSV_PATH, columns=columns)
    return {c: df[c].to_numpy().astype(float) for c in columns}


def _dark_subplots(
    rows: int, titles: list[str], ylabels: list[str]
) -> go.Figure:
    """Create a make_subplots figure with dark motorsport styling."""
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=titles,
    )
    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT, size=11),
        legend=dict(
            bgcolor="rgba(20,20,23,0.85)",
            bordercolor="rgba(128,128,128,0.3)",
            font=dict(color=_TEXT),
        ),
    )
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


# ── 1. Energy per lap (standalone) ──────────────────────────────────────────


def energy_per_lap() -> go.Figure:
    d = _load(["TimeStamp", "laps", "Vbat", "Current"])
    time = d["TimeStamp"] - d["TimeStamp"][0]
    laps = d["laps"]

    valid = (
        np.isfinite(time)
        & np.isfinite(laps)
        & np.isfinite(d["Vbat"])
        & np.isfinite(d["Current"])
    )
    time = time[valid]
    laps = laps[valid]
    p_kw = (d["Vbat"][valid] * d["Current"][valid]) / 1000.0
    e_cum = cumulative_trapezoid(p_kw, time, initial=0.0) / 3600.0

    # Filter out lap 0
    mask = laps > 0
    time = time[mask]
    laps = laps[mask]
    p_kw = p_kw[mask]
    e_cum = e_cum[mask]

    lap_list = unique_laps(laps)
    if len(lap_list) >= 2:
        lap_list = lap_list[:-1]

    n = len(lap_list)
    e_lap = np.full(n, np.nan)
    lt_s = np.full(n, np.nan)
    p_avg = np.full(n, np.nan)

    for i, lap in enumerate(lap_list):
        idx = np.where(laps == lap)[0]
        if len(idx) < 2:
            continue
        e_lap[i] = e_cum[idx[-1]] - e_cum[idx[0]]
        lt_s[i] = time[idx[-1]] - time[idx[0]]
        p_avg[i] = np.nanmean(p_kw[idx])

    ok = np.isfinite(e_lap) & np.isfinite(lt_s) & (e_lap > 0)
    e_ok = e_lap[ok]
    lt_ok = lt_s[ok]
    l_ok = lap_list[ok]
    p_ok = p_avg[ok]

    e_mean = np.nanmean(e_ok)
    e_std = np.nanstd(e_ok)
    cv = 100.0 * e_std / e_mean if e_mean > 0 else np.nan
    e_total = np.nansum(e_ok)
    p_mean = np.nanmean(p_ok)
    p_std = np.nanstd(p_ok)

    if len(e_ok) >= 2:
        poly_et = np.polyfit(lt_ok, e_ok, 1)
        slope_e_t = poly_et[0]
        y_pred = np.polyval(poly_et, lt_ok)
        ss_res = np.sum((e_ok - y_pred) ** 2)
        ss_tot = np.sum((e_ok - e_mean) ** 2)
        r2_e_t = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        corr = float(np.corrcoef(lt_ok, e_ok)[0, 1])
        slope_e_lap = np.polyfit(l_ok, e_ok, 1)[0]
    else:
        slope_e_t = r2_e_t = corr = slope_e_lap = np.nan

    i_fastest = int(np.argmin(lt_ok))
    i_slowest = int(np.argmax(lt_ok))
    i_min_e = int(np.argmin(e_ok))
    i_max_e = int(np.argmax(e_ok))

    print("\n===== ENERGY KPIs =====")
    print(f"Mean energy per lap       : {e_mean:.5f} kWh")
    print(f"Std energy per lap        : {e_std:.5f} kWh")
    print(f"Coefficient of variation  : {cv:.2f} %")
    print(f"Total energy consumed     : {e_total:.4f} kWh")
    print(f"Lowest energy lap         : L{int(l_ok[i_min_e])} -> {e_ok.min():.5f} kWh")
    print(f"Highest energy lap        : L{int(l_ok[i_max_e])} -> {e_ok.max():.5f} kWh")
    print(f"Fastest lap               : L{int(l_ok[i_fastest])} -> {lt_ok.min():.3f} s")
    print(f"Slowest lap               : L{int(l_ok[i_slowest])} -> {lt_ok.max():.3f} s")
    print(f"Energy trend              : {slope_e_lap:+.5f} kWh/lap")
    print(f"Energy sensitivity vs t   : {slope_e_t:+.5f} kWh/s")
    print(f"Correlation E_lap vs t    : {corr:.3f}")
    print(f"R² (energy vs laptime)    : {r2_e_t:.3f}")
    print(f"Mean battery power        : {p_mean:.3f} kW")
    print(f"Std battery power         : {p_std:.3f} kW")

    print(
        f'\n{"Lap":>4}  {"LapTime[s]":>10}  {"E_lap[kWh]":>11}  {"P_avg[kW]":>10}'
    )
    for i, (lap, lt, e, pa) in enumerate(zip(l_ok, lt_ok, e_ok, p_ok)):
        tag = ""
        if i == i_fastest:
            tag = "  <- fastest"
        elif i == i_min_e:
            tag = "  <- min energy"
        print(f"{int(lap):>4}  {lt:>10.3f}  {e:>11.5f}  {pa:>10.3f}{tag}")

    fig = make_dark_figure(
        title=f"Energy per Lap vs Lap Time  (R² = {r2_e_t:.3f})",
        xlabel="Lap time [s]",
        ylabel="Energy per lap [kWh]",
    )
    add_lap_scatter(fig, lt_ok, e_ok, l_ok)
    add_trend_line(fig, lt_ok, e_ok)
    fig.add_hline(
        y=e_mean,
        line=dict(color="rgba(200,200,200,0.4)", dash="dot", width=1),
    )
    fig.add_hrect(
        y0=e_mean - e_std,
        y1=e_mean + e_std,
        fillcolor="rgba(77,179,242,0.08)",
        line_width=0,
    )
    return fig


# ── 2. Power per wheel (standalone) ─────────────────────────────────────────


def power_per_wheel() -> go.Figure:
    power_cols = [f"{w}_actualPower" for w in WHEELS]
    d = _load(["laps"] + power_cols)
    d = exclude_lap0_and_last_lap(d)
    laps = d["laps"]
    lap_list = unique_laps(laps)
    n = len(lap_list)

    p_wheel_kw = np.full((n, 4), np.nan)

    for i, lap in enumerate(lap_list):
        idx = laps == lap
        if idx.sum() < 5:
            continue
        for j, w in enumerate(WHEELS):
            p_wheel_kw[i, j] = np.nanmean(d[f"{w}_actualPower"][idx]) / 1000.0

    ok = np.isfinite(p_wheel_kw[:, 0])
    lp = lap_list[ok]
    pw = p_wheel_kw[ok]

    p_total = np.nansum(pw, axis=1)
    p_front = pw[:, 0] + pw[:, 1]
    mean_total = np.nanmean(p_total)
    fr_pct = (
        np.nanmean(p_front) / mean_total * 100 if mean_total > 0 else np.nan
    )
    p_left = pw[:, 0] + pw[:, 2]
    lr_pct = (
        np.nanmean(p_left) / mean_total * 100 if mean_total > 0 else np.nan
    )

    print("\n===== POWER DISTRIBUTION KPIs =====")
    print(f"Mean total power          : {mean_total:.3f} kW")
    print(f"Front/Rear split          : {fr_pct:.1f}% / {100 - fr_pct:.1f}%")
    print(f"Left/Right split          : {lr_pct:.1f}% / {100 - lr_pct:.1f}%")
    for j, w in enumerate(WHEELS):
        w_pct = (
            np.nanmean(pw[:, j]) / mean_total * 100
            if mean_total > 0
            else np.nan
        )
        print(
            f"  {w} mean power            : {np.nanmean(pw[:, j]):.3f} kW ({w_pct:.1f}%)"
        )

    print(f'\n{"Lap":>4}', end="")
    for w in WHEELS:
        print(f'  {w + "[kW]":>9}', end="")
    print(f'  {"Total[kW]":>10}  {"F/R%":>6}')
    for i in range(len(lp)):
        total = np.nansum(pw[i])
        fr = (pw[i, 0] + pw[i, 1]) / total * 100 if total > 0 else np.nan
        print(f"{int(lp[i]):>4}", end="")
        for j in range(4):
            print(f"  {pw[i, j]:>9.3f}", end="")
        print(f"  {total:>10.3f}  {fr:>5.1f}%")

    fig = make_dark_figure(
        title="Mean Electrical Power per Wheel",
        xlabel="Lap",
        ylabel="Power [kW]",
    )
    for j, w in enumerate(WHEELS):
        fig.add_trace(
            go.Scatter(
                x=lp,
                y=pw[:, j],
                mode="lines+markers",
                name=w,
                line=dict(color=WHEEL_COLORS[w], width=1.5),
                marker=dict(size=7),
            )
        )
    fig.update_xaxes(tickvals=lp.astype(int))
    return fig


# ── 3. Battery status (standalone) ──────────────────────────────────────────


def battery_status() -> go.Figure:
    d = _load(
        ["laps", "TimeStamp", "SoC", "Vbat", "Current", "Vmin", "Vmax"]
    )
    d = exclude_lap0_and_last_lap(d)
    laps = d["laps"]
    lap_list = unique_laps(laps)
    n = len(lap_list)

    soc_end = np.full(n, np.nan)
    soc_drop = np.full(n, np.nan)
    vbat_mean_v = np.full(n, np.nan)
    vbat_min_v = np.full(n, np.nan)
    i_mean_a = np.full(n, np.nan)
    cell_spread_v = np.full(n, np.nan)

    for i, lap in enumerate(lap_list):
        idx = np.where(laps == lap)[0]
        if len(idx) < 5:
            continue
        soc_end[i] = d["SoC"][idx[-1]]
        soc_drop[i] = d["SoC"][idx[0]] - d["SoC"][idx[-1]]
        vbat_mean_v[i] = np.nanmean(d["Vbat"][idx])
        vbat_min_v[i] = np.nanmin(d["Vbat"][idx])
        i_mean_a[i] = np.nanmean(d["Current"][idx])
        cell_spread_v[i] = np.nanmean(d["Vmax"][idx] - d["Vmin"][idx])

    ok = np.isfinite(soc_end)
    lp = lap_list[ok]
    soc = soc_end[ok]
    sd = soc_drop[ok]
    vm = vbat_mean_v[ok]
    vn = vbat_min_v[ok]
    im = i_mean_a[ok]
    cs = cell_spread_v[ok]

    soc_start = soc[0] + sd[0] if len(soc) > 0 else np.nan

    print("\n===== BATTERY KPIs =====")
    print(f"SoC start                 : {soc_start:.1f} %")
    print(f"SoC end                   : {soc[-1]:.1f} %")
    print(f"Total SoC consumed        : {np.nansum(sd):.1f} %")
    print(f"Mean SoC drop per lap     : {np.nanmean(sd):.2f} %")
    print(f"Mean battery voltage      : {np.nanmean(vm):.2f} V")
    print(f"Min battery voltage       : {np.nanmin(vn):.2f} V")
    print(
        f"Voltage sag (mean-min)    : {np.nanmean(vm) - np.nanmin(vn):.2f} V"
    )
    print(
        f"Mean cell spread (Vmax-Vmin): {np.nanmean(cs) * 1000:.1f} mV"
    )
    print(f"Max cell spread           : {np.nanmax(cs) * 1000:.1f} mV")
    print(f"Mean current draw         : {np.nanmean(im):.1f} A")

    print(
        f'\n{"Lap":>4}  {"SoC[%]":>7}  {"dSoC[%]":>8}  {"Vbat[V]":>8}  '
        f'{"Vmin[V]":>8}  {"Spread[mV]":>11}'
    )
    for lap, s, drop, v, vmin, sp in zip(lp, soc, sd, vm, vn, cs):
        print(
            f"{int(lap):>4}  {s:>7.1f}  {drop:>8.2f}  {v:>8.2f}  "
            f"{vmin:>8.2f}  {sp * 1000:>11.1f}"
        )

    fig = _dark_subplots(
        rows=3,
        titles=["SoC Evolution", "Battery Voltage", "Cell Voltage Spread"],
        ylabels=["SoC [%]", "Voltage [V]", "Spread [mV]"],
    )
    fig.add_trace(
        go.Scatter(
            x=lp,
            y=soc,
            mode="lines+markers",
            name="SoC",
            line=dict(color="#4DB3F2", width=2),
            marker=dict(size=7),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=lp,
            y=vm,
            mode="lines+markers",
            name="Vbat mean",
            line=dict(color="#73D973", width=1.5),
            marker=dict(size=6),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=lp,
            y=vn,
            mode="lines+markers",
            name="Vbat min",
            line=dict(color="#F28C40", width=1.5, dash="dash"),
            marker=dict(size=6),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=lp,
            y=cs * 1000,
            mode="lines+markers",
            name="Vmax-Vmin",
            line=dict(color="#D973D9", width=1.5),
            marker=dict(size=6),
        ),
        row=3,
        col=1,
    )
    fig.update_xaxes(
        title_text="Lap", tickvals=lp.astype(int), row=3, col=1
    )
    fig.update_layout(
        height=700,
        title_text="Battery Status per Lap",
        title_font=dict(color=_TEXT, size=14),
    )
    return fig


# ── 4. Thermal evolution (standalone) ───────────────────────────────────────


def thermal_evolution() -> go.Figure:
    motor_cols = [f"{w}_motorTemperature" for w in WHEELS]
    inv_cols = [f"{w}_inverterTemperature" for w in WHEELS]
    batt_cols = ["Tmax", "Tavg"]

    d = _load(["laps"] + motor_cols + inv_cols + batt_cols)
    d = exclude_lap0_and_last_lap(d)
    laps = d["laps"]
    lap_list = unique_laps(laps)
    n = len(lap_list)

    T_motor = np.full((n, 4), np.nan)
    T_inv = np.full((n, 4), np.nan)
    T_batt = np.full((n, 2), np.nan)

    for i, lap in enumerate(lap_list):
        idx = laps == lap
        if idx.sum() < 5:
            continue
        for j, w in enumerate(WHEELS):
            T_motor[i, j] = np.nanpercentile(
                d[f"{w}_motorTemperature"][idx], 95
            )
            T_inv[i, j] = np.nanpercentile(
                d[f"{w}_inverterTemperature"][idx], 95
            )
        T_batt[i, 0] = np.nanmax(d["Tmax"][idx])
        T_batt[i, 1] = np.nanpercentile(d["Tavg"][idx], 95)

    ok = np.isfinite(T_motor[:, 0])
    lp = lap_list[ok]
    Tm = T_motor[ok]
    Ti = T_inv[ok]
    Tb = T_batt[ok]

    def slope(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else np.nan

    motor_lr = (Tm[:, 0] + Tm[:, 2]) / 2 - (Tm[:, 1] + Tm[:, 3]) / 2
    inv_lr = (Ti[:, 0] + Ti[:, 2]) / 2 - (Ti[:, 1] + Ti[:, 3]) / 2

    print("\n===== THERMAL KPIs =====")
    print(f"Motor dT Left-Right (mean)  : {np.nanmean(motor_lr):+.2f} C")
    print(
        f"Inverter dT Left-Right (mean): {np.nanmean(inv_lr):+.2f} C"
    )
    for j, w in enumerate(WHEELS):
        print(
            f"  Motor peak P95 ({w})       : {np.nanmax(Tm[:, j]):.1f} C"
        )
    for j, w in enumerate(WHEELS):
        print(
            f"  Inverter peak P95 ({w})     : {np.nanmax(Ti[:, j]):.1f} C"
        )
    print(f"Battery Tmax peak           : {np.nanmax(Tb[:, 0]):.1f} C")
    print(f"Battery Tavg peak P95       : {np.nanmax(Tb[:, 1]):.1f} C")
    print(
        f"Motor avg thermal slope     : "
        f"{slope(lp, np.nanmean(Tm, axis=1)):+.2f} C/lap"
    )
    print(
        f"Inverter avg thermal slope  : "
        f"{slope(lp, np.nanmean(Ti, axis=1)):+.2f} C/lap"
    )
    print(
        f"Battery Tmax slope          : {slope(lp, Tb[:, 0]):+.2f} C/lap"
    )
    for j, w in enumerate(WHEELS):
        mi_delta = np.nanmean(Tm[:, j] - Ti[:, j])
        print(f"  Motor-Inverter dT ({w})   : {mi_delta:+.2f} C")

    fig = make_dark_figure(
        title="Thermal Evolution per Lap (P95)",
        xlabel="Lap",
        ylabel="Temperature [C]",
    )
    for j, w in enumerate(WHEELS):
        fig.add_trace(
            go.Scatter(
                x=lp,
                y=Tm[:, j],
                mode="lines+markers",
                name=f"Motor {w}",
                line=dict(color=WHEEL_COLORS[w], width=1.5),
                marker=dict(size=7),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=lp,
                y=Ti[:, j],
                mode="lines",
                name=f"Inv {w}",
                line=dict(color=WHEEL_COLORS[w], dash="dash", width=1.0),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=lp,
            y=Tb[:, 0],
            mode="lines+markers",
            name="Battery Tmax",
            line=dict(color="white", width=2),
            marker=dict(size=7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=lp,
            y=Tb[:, 1],
            mode="lines",
            name="Battery Tavg",
            line=dict(color="white", dash="dot", width=1.2),
        )
    )
    fig.update_xaxes(tickvals=lp.astype(int))
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

        time = df["TimeStamp"].to_numpy().astype(float)
        laps = df["laps"].to_numpy().astype(float)
        vbat = df["Vbat"].to_numpy().astype(float)
        current = df["Current"].to_numpy().astype(float)

        valid = (
            np.isfinite(time)
            & np.isfinite(laps)
            & np.isfinite(vbat)
            & np.isfinite(current)
        )
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
    e_std  = float(np.nanstd(e_arr))
    cv     = 100 * e_std / e_mean if e_mean > 0 else np.nan
    e_total = float(np.nansum(e_arr))
    e_rec_total = float(np.nansum(e_rec_arr))
    e_cons_total = float(np.nansum(e_cons_arr))
    p_mean  = float(np.nanmean(p_arr))

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

    i_min_e   = int(np.argmin(e_arr))
    i_max_e   = int(np.argmax(e_arr))
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
        "e_mean":      e_mean,
        "e_total":     e_total,
        "e_cons_mean": e_cons_mean,
        "e_cons_total": e_cons_total,
        "e_rec_mean":  e_rec_mean,
        "e_rec_total": e_rec_total,
        "cv":          cv,
        "r2":          r2,
        "r2_lap":      r2_lap,
        "r2_laptime":  r2_laptime,
        "p_mean":      p_mean,
        "min_e_lap":   int(lp_arr[i_min_e]),
        "min_e":       float(e_arr[i_min_e]),
        "max_e_lap":   int(lp_arr[i_max_e]),
        "max_e":       float(e_arr[i_max_e]),
        "fastest_lap": int(lp_arr[i_fastest]),
        "fastest_lt":  float(lt_arr[i_fastest]),
        "table":       pl.DataFrame(table),
        "warnings":    warnings,
    }
    return fig, kpis


# ── Power per wheel ──────────────────────────────────────────────────────────


def power_per_wheel_fig(
    dfs: dict[str, pl.DataFrame],
    x_mode: str = "laps",
) -> tuple[go.Figure, dict]:
    """Power distribution per wheel figure + KPIs.

    kpis keys: mean_total_kw, fr_pct, lr_pct,
               wheel_mean_kw (dict), wheel_pct (dict),
               table (pl.DataFrame), warnings (list[str]).
    """
    power_cols = [f"{w}_actualPower" for w in WHEELS]
    cols_needed = ["laps", "laptime"] + power_cols

    laps_all: list[int] = []
    lt_all: list[float] = []
    pw_all: list[list[float]] = []
    run_all: list[str] = []
    warnings: list[str] = []

    for run_name, df in dfs.items():
        df = ensure_complete_laps_df(df)
        missing = [c for c in cols_needed if c not in df.columns]
        if missing:
            warnings.append(f"{run_name}: missing {missing}")
            continue

        laps = df["laps"].to_numpy().astype(float)
        laptime = df["laptime"].to_numpy().astype(float)
        powers = {
            w: df[f"{w}_actualPower"].to_numpy().astype(float)
            for w in WHEELS
        }

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

    p_total    = np.nansum(pw_arr, axis=1)
    p_front    = pw_arr[:, 0] + pw_arr[:, 1]
    p_left     = pw_arr[:, 0] + pw_arr[:, 2]
    mean_total = float(np.nanmean(p_total))
    fr_pct = float(np.nanmean(p_front) / mean_total * 100) if mean_total > 0 else np.nan
    lr_pct = float(np.nanmean(p_left)  / mean_total * 100) if mean_total > 0 else np.nan

    x_arr, order, xlabel = per_lap_axis(lp_arr, lt_arr, x_mode)
    fig = make_dark_figure(
        title=f"Mean Electrical Power per Wheel vs {'Lap Time' if x_mode == 'laptime' else 'Lap'}",
        xlabel=xlabel,
        ylabel="Power [kW]",
    )
    for j, w in enumerate(WHEELS):
        fig.add_trace(go.Scatter(
            x=x_arr, y=pw_arr[order, j],
            mode="lines+markers", name=w,
            line=dict(color=WHEEL_COLORS[w], width=1.5),
            marker=dict(size=7),
        ))
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
        "fr_pct":        fr_pct,
        "lr_pct":        lr_pct,
        "wheel_mean_kw": {w: float(np.nanmean(pw_arr[:, j])) for j, w in enumerate(WHEELS)},
        "wheel_pct": {
            w: float(np.nanmean(pw_arr[:, j]) / mean_total * 100) if mean_total > 0 else np.nan
            for j, w in enumerate(WHEELS)
        },
        "table":    pl.DataFrame(table),
        "warnings": warnings,
    }
    return fig, kpis


# ── Battery status ───────────────────────────────────────────────────────────


def battery_status_fig(
    dfs: dict[str, pl.DataFrame],
    x_mode: str = "laps",
) -> tuple[go.Figure, dict]:
    """Battery status figure + KPIs.

    kpis keys: soc_start, soc_end, soc_total_drop, voltage_sag,
               cell_spread_mean, soc_drop_per_lap, mean_voltage,
               min_voltage, mean_current, table, warnings.
    """
    cols_needed = ["laps", "laptime", "TimeStamp", "SoC", "Vbat", "Current", "Vmin", "Vmax"]

    laps_all: list[int] = []
    lt_all: list[float] = []
    soc_all: list[float] = []
    sd_all: list[float] = []
    vm_all: list[float] = []
    vn_all: list[float] = []
    vp05_all: list[float] = []
    im_all: list[float] = []
    cs_all: list[float] = []
    sag_all: list[float] = []
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

        laps    = df["laps"].to_numpy().astype(float)
        laptime = df["laptime"].to_numpy().astype(float)
        soc_arr = df["SoC"].to_numpy().astype(float)
        vbat    = df["Vbat"].to_numpy().astype(float)
        current = df["Current"].to_numpy().astype(float)
        vmin    = df["Vmin"].to_numpy().astype(float)
        vmax    = df["Vmax"].to_numpy().astype(float)

        run_laps = unique_laps(laps)
        if len(run_laps) > 0:
            first_idx = np.where(laps == run_laps[0])[0]
            last_idx = np.where(laps == run_laps[-1])[0]
            if len(first_idx) > 0 and len(last_idx) > 0:
                run_soc_start.append(float(soc_arr[first_idx[0]]))
                run_soc_end.append(float(soc_arr[last_idx[-1]]))

        for lap in run_laps:
            idx = np.where(laps == lap)[0]
            if len(idx) < 5:
                continue
            soc_end_val = float(soc_arr[idx[-1]])
            soc_drop    = float(soc_arr[idx[0]] - soc_arr[idx[-1]])
            if not np.isfinite(soc_end_val):
                continue
            laps_all.append(int(lap))
            lt_all.append(float(np.nanmax(laptime[idx])))
            soc_all.append(soc_end_val)
            sd_all.append(soc_drop)
            vm_all.append(float(np.nanmean(vbat[idx])))
            vn_all.append(float(np.nanmin(vbat[idx])))
            vp05_all.append(float(np.nanpercentile(vbat[idx], 5)))
            im_all.append(float(np.nanmean(current[idx])))
            cs_all.append(float(np.nanmean(vmax[idx] - vmin[idx])))
            sag_all.append(
                float(np.nanmean(vbat[idx]) - np.nanpercentile(vbat[idx], 5))
            )
            run_all.append(run_name)

    if not laps_all:
        fig = make_dark_figure(title="Battery Status — No data")
        return fig, {"warnings": warnings + ["No valid battery data."]}

    lp  = np.array(laps_all)
    lt  = np.array(lt_all)
    soc = np.array(soc_all)
    sd  = np.array(sd_all)
    vm  = np.array(vm_all)
    vn  = np.array(vn_all)
    vp05 = np.array(vp05_all)
    im  = np.array(im_all)
    cs  = np.array(cs_all)
    sag = np.array(sag_all)

    if len(run_soc_start) == 1 and len(run_soc_end) == 1:
        soc_start = run_soc_start[0]
        soc_end = run_soc_end[0]
        soc_total_drop = soc_start - soc_end
    else:
        soc_start = np.nan
        soc_end = np.nan
        soc_total_drop = np.nan
        if len(run_soc_start) > 1:
            warnings.append(
                "Battery SoC start/end and total drop are only meaningful for a single continuous run."
            )

    x_arr, order, xlabel = per_lap_axis(lp, lt, x_mode)
    fig = _dark_subplots(
        rows=3,
        titles=[
            "End-of-Lap SoC",
            "Battery Voltage",
            "Cell Voltage Spread",
        ],
        ylabels=["SoC [%]", "Voltage [V]", "Spread [mV]"],
    )
    fig.add_trace(
        go.Scatter(x=x_arr, y=soc[order], mode="lines+markers", name="SoC",
                   line=dict(color="#4DB3F2", width=2), marker=dict(size=7)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x_arr, y=vm[order], mode="lines+markers", name="Vbat mean",
                   line=dict(color="#73D973", width=1.5), marker=dict(size=6)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x_arr, y=vn[order], mode="lines+markers", name="Vbat min",
                   line=dict(color="#F28C40", width=1.5, dash="dash"), marker=dict(size=6)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x_arr, y=(cs * 1000)[order], mode="lines+markers", name="Vmax-Vmin",
                   line=dict(color="#D973D9", width=1.5), marker=dict(size=6)),
        row=3, col=1,
    )
    fig.update_xaxes(title_text=xlabel, row=3, col=1)
    if x_mode == "laps":
        fig.update_xaxes(tickvals=np.sort(lp.astype(int)), row=3, col=1)
    fig.update_layout(
        height=700,
        title_text=f"Battery Status {'vs Lap Time' if x_mode == 'laptime' else 'per Lap'}",
        title_font=dict(color=_TEXT, size=14),
    )

    table: dict[str, object] = {
        "Lap":          lp,
        "LapTime [s]":  np.round(lt, 3),
        "SoC [%]":      np.round(soc, 1),
        "dSoC [%]":     np.round(sd, 2),
        "Vbat [V]":     np.round(vm, 2),
        "Vmin [V]":     np.round(vn, 2),
        "Vbat P05 [V]": np.round(vp05, 2),
        "Voltage sag [V]": np.round(sag, 2),
        "Spread [mV]":  np.round(cs * 1000, 1),
    }
    if len(dfs) > 1:
        table["Run"] = run_all

    kpis = {
        "soc_start":       soc_start,
        "soc_end":         soc_end,
        "soc_total_drop":  soc_total_drop,
        "voltage_sag":     float(np.nanmean(sag)),
        "cell_spread_mean": float(np.nanmean(cs) * 1000),
        "soc_drop_per_lap": float(np.nanmean(sd)),
        "mean_voltage":    float(np.nanmean(vm)),
        "min_voltage":     float(np.nanmin(vn)),
        "mean_current":    float(np.nanmean(im)),
        "table":           pl.DataFrame(table),
        "warnings":        warnings,
    }
    return fig, kpis


# ── Thermal evolution ────────────────────────────────────────────────────────


def thermal_evolution_fig(
    dfs: dict[str, pl.DataFrame],
    x_mode: str = "laps",
) -> tuple[go.Figure, dict]:
    """Thermal evolution figure + KPIs.

    kpis keys: peak_motor, peak_inverter, peak_batt_tmax,
               motor_thermal_slope, motor_peak_by_wheel (dict),
               table (pl.DataFrame), warnings (list[str]).
    """
    motor_cols = [f"{w}_motorTemperature" for w in WHEELS]
    inv_cols   = [f"{w}_inverterTemperature" for w in WHEELS]
    batt_cols  = ["Tmax", "Tavg"]
    cols_needed = ["laps", "laptime"] + motor_cols + inv_cols + batt_cols

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

        laps = df["laps"].to_numpy().astype(float)
        laptime = df["laptime"].to_numpy().astype(float)

        for lap in unique_laps(laps):
            idx = laps == lap
            if idx.sum() < 5:
                continue

            tm = [
                float(
                    np.nanpercentile(
                        df[f"{w}_motorTemperature"].to_numpy()[idx], 95
                    )
                )
                for w in WHEELS
            ]
            ti = [
                float(
                    np.nanpercentile(
                        df[f"{w}_inverterTemperature"].to_numpy()[idx], 95
                    )
                )
                for w in WHEELS
            ]
            tb_max = float(np.nanmax(df["Tmax"].to_numpy()[idx]))
            tb_avg = float(
                np.nanpercentile(df["Tavg"].to_numpy()[idx], 95)
            )

            if not all(np.isfinite(tm)):
                continue

            laps_all.append(int(lap))
            lt_all.append(float(np.nanmax(laptime[idx])))
            tm_all.append(tm)
            ti_all.append(ti)
            tb_all.append([tb_max, tb_avg])
            run_all.append(run_name)

    if not laps_all:
        fig = make_dark_figure(title="Thermal Evolution — No data")
        return fig, {"warnings": warnings + ["No valid thermal data."]}

    lp = np.array(laps_all)
    lt = np.array(lt_all)
    Tm = np.array(tm_all)
    Ti = np.array(ti_all)
    Tb = np.array(tb_all)

    def _slope(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else np.nan

    x_arr, order, xlabel = per_lap_axis(lp, lt, x_mode)
    fig = make_dark_figure(
        title=f"Thermal Evolution {'vs Lap Time' if x_mode == 'laptime' else 'per Lap'} (P95)",
        xlabel=xlabel,
        ylabel="Temperature [°C]",
    )
    for j, w in enumerate(WHEELS):
        fig.add_trace(go.Scatter(
            x=x_arr, y=Tm[order, j], mode="lines+markers", name=f"Motor {w}",
            line=dict(color=WHEEL_COLORS[w], width=1.5), marker=dict(size=7),
        ))
        fig.add_trace(go.Scatter(
            x=x_arr, y=Ti[order, j], mode="lines", name=f"Inv {w}",
            line=dict(color=WHEEL_COLORS[w], dash="dash", width=1.0),
        ))
    fig.add_trace(go.Scatter(
        x=x_arr, y=Tb[order, 0], mode="lines+markers", name="Battery Tmax",
        line=dict(color="white", width=2), marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=x_arr, y=Tb[order, 1], mode="lines", name="Battery Tavg",
        line=dict(color="white", dash="dot", width=1.2),
    ))
    if x_mode == "laps":
        fig.update_xaxes(tickvals=np.sort(lp.astype(int)))

    table: dict[str, object] = {"Lap": lp, "LapTime [s]": np.round(lt, 3)}
    for j, w in enumerate(WHEELS):
        table[f"Motor {w} [°C]"] = np.round(Tm[:, j], 1)
        table[f"Inv {w} [°C]"]   = np.round(Ti[:, j], 1)
    table["Batt Tmax [°C]"] = np.round(Tb[:, 0], 1)
    table["Batt Tavg [°C]"] = np.round(Tb[:, 1], 1)
    if len(dfs) > 1:
        table["Run"] = run_all

    kpis = {
        "peak_motor":          float(np.nanmax(Tm)),
        "peak_inverter":       float(np.nanmax(Ti)),
        "peak_batt_tmax":      float(np.nanmax(Tb[:, 0])),
        "motor_thermal_slope": _slope(lp, np.nanmean(Tm, axis=1)),
        "motor_peak_by_wheel": {w: float(np.nanmax(Tm[:, j])) for j, w in enumerate(WHEELS)},
        "table":               pl.DataFrame(table),
        "warnings":            warnings,
    }
    return fig, kpis


# ── Entry point (standalone CLI) ────────────────────────────────────────────


def main() -> None:
    energy_per_lap().show()
    power_per_wheel().show()
    battery_status().show()
    thermal_evolution().show()


if __name__ == "__main__":
    main()
