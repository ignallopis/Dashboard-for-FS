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
        energy_budget_per_lap_fig(dfs) -> tuple[go.Figure, dict]
        energy_budget_breakdown_fig(dfs, run_name, lap) -> go.Figure
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
    cols_to_numpy,
    ensure_complete_laps_df,
    exclude_lap0_and_last_lap,
    lap_dist_from_gps,
    per_lap_axis,
    unique_laps,
    WHEEL_COLORS,
)
from src.dynamics import (
    A_AERO_M2,
    CD_AERO,
    CL_AERO,
    G_MS2,
    MASS_KG,
    RHO_AIR_KGM3,
)

CSV_PATH = "data/run4_2025-08-24.csv"
WHEELS = ("FL", "FR", "RL", "RR")

# Energy-budget model assumptions. Keep these explicit because they are
# calibration candidates, not measured CAT17x constants.
CRR_TIRE = 0.015
CALPHA_AXLE_TOTAL = 120_000.0  # [N/rad], front + rear axle cornering stiffness assumption

# Dark theme (mirrors utils.py for subplot figures)
_BG = "#141417"
_TEXT = "#EBEBEB"
_GRID = "rgba(128,128,128,0.2)"
_AXIS = "#E5E5E5"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load(columns: list[str]) -> dict[str, np.ndarray]:
    df = pl.read_csv(CSV_PATH, columns=columns)
    return cols_to_numpy(df, columns)


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
        ["laps", "TimeStamp", "SoC", "Vbat", "Current", "Vmin"]
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

    for i, lap in enumerate(lap_list):
        idx = np.where(laps == lap)[0]
        if len(idx) < 5:
            continue
        soc_end[i] = d["SoC"][idx[-1]]
        soc_drop[i] = d["SoC"][idx[0]] - d["SoC"][idx[-1]]
        vbat_mean_v[i] = np.nanmean(d["Vbat"][idx])
        vbat_min_v[i] = np.nanmin(d["Vbat"][idx])
        i_mean_a[i] = np.nanmean(d["Current"][idx])

    ok = np.isfinite(soc_end)
    lp = lap_list[ok]
    soc = soc_end[ok]
    sd = soc_drop[ok]
    vm = vbat_mean_v[ok]
    vn = vbat_min_v[ok]
    im = i_mean_a[ok]

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
    print(f"Mean current draw         : {np.nanmean(im):.1f} A")

    print(
        f'\n{"Lap":>4}  {"SoC[%]":>7}  {"dSoC[%]":>8}  {"Vbat[V]":>8}  '
        f'{"Vmin[V]":>8}'
    )
    for lap, s, drop, v, vmin in zip(lp, soc, sd, vm, vn):
        print(
            f"{int(lap):>4}  {s:>7.1f}  {drop:>8.2f}  {v:>8.2f}  "
            f"{vmin:>8.2f}"
        )

    fig = _dark_subplots(
        rows=2,
        titles=["SoC Evolution", "Battery Voltage"],
        ylabels=["SoC [%]", "Voltage [V]"],
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
    fig.update_xaxes(
        title_text="Lap", tickvals=lp.astype(int), row=2, col=1
    )
    fig.update_layout(
        height=560,
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

        cols = cols_to_numpy(df, ["TimeStamp", "laps", "Vbat", "Current"])
        time = cols["TimeStamp"]
        laps = cols["laps"]
        vbat = cols["Vbat"]
        current = cols["Current"]

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


# ── Energy budget ────────────────────────────────────────────────────────────


_BATTERY_POWER_COLS = (
    "BMS_power",
    "BMS_Power",
    "BatteryPower",
    "Battery_power",
    "P_batt",
)
_VOLT_CURRENT_COLS = (
    ("BMS_voltage", "BMS_current"),
    ("BMS_Voltage", "BMS_Current"),
    ("Vbat", "Current"),
)


def _empty_energy_budget_fig(message: str) -> go.Figure:
    fig = make_dark_figure(
        title="Energy Budget per Lap — No data",
        xlabel="Lap",
        ylabel="Energy [kJ]",
    )
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color=_TEXT, size=13),
    )
    return fig


def _battery_power_w(df: pl.DataFrame) -> tuple[np.ndarray | None, str, list[str]]:
    """Return battery power [W], source label and warnings."""
    for col in _BATTERY_POWER_COLS:
        if col in df.columns:
            return df[col].to_numpy().astype(float), col, []

    for voltage_col, current_col in _VOLT_CURRENT_COLS:
        if voltage_col in df.columns and current_col in df.columns:
            voltage = df[voltage_col].to_numpy().astype(float)
            current = df[current_col].to_numpy().astype(float)
            return voltage * current, f"{voltage_col}*{current_col}", []

    power_cols = [f"{w}_actualPower" for w in WHEELS]
    if all(c in df.columns for c in power_cols):
        p_w = np.zeros(len(df), dtype=float)
        for col in power_cols:
            p_w += df[col].to_numpy().astype(float)
        return p_w, "sum(*_actualPower)", [
            "Battery power unavailable; using summed inverter `*_actualPower` as fallback."
        ]

    return None, "unavailable", ["No battery or inverter power signal found."]


def _energy_budget_lap_rows(
    dfs: dict[str, pl.DataFrame],
    *,
    include_samples: bool = False,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]], list[str]]:
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    run_kpis: dict[str, dict[str, float]] = {}

    for run_name, df in dfs.items():
        try:
            df = ensure_complete_laps_df(df)
        except Exception as exc:
            warnings.append(f"{run_name}: {exc}")
            continue

        speed_col = "VN_vx" if "VN_vx" in df.columns else "Est_vxCOG"
        ay_col = "Filtering_VN_ay" if "Filtering_VN_ay" in df.columns else "VN_ay"
        required = ["TimeStamp", "laps", "laptime", speed_col, ay_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            warnings.append(f"{run_name}: missing {missing}")
            continue

        p_batt_w, power_source, source_warnings = _battery_power_w(df)
        warnings.extend([f"{run_name}: {w}" for w in source_warnings])
        if p_batt_w is None:
            continue

        cols = cols_to_numpy(df, required)
        time_s = cols["TimeStamp"]
        laps = cols["laps"]
        laptime_s = cols["laptime"]
        vx_mps = np.abs(cols[speed_col])
        ay_mps2 = cols[ay_col]
        dist_m = lap_dist_from_gps(df)

        run_rows: list[dict[str, object]] = []
        for lap in unique_laps(laps):
            idx = np.where(laps == lap)[0]
            if len(idx) < 5:
                continue

            valid = (
                np.isfinite(time_s[idx])
                & np.isfinite(vx_mps[idx])
                & np.isfinite(ay_mps2[idx])
                & np.isfinite(p_batt_w[idx])
            )
            if int(valid.sum()) < 5:
                continue

            lap_idx = idx[valid]
            order = np.argsort(time_s[lap_idx])
            lap_idx = lap_idx[order]
            t_s = time_s[lap_idx] - time_s[lap_idx][0]
            if not np.isfinite(t_s[-1]) or t_s[-1] <= 0.0:
                continue

            v = vx_mps[lap_idx]
            ay = ay_mps2[lap_idx]
            p_batt = p_batt_w[lap_idx]

            f_aero_n = 0.5 * RHO_AIR_KGM3 * v**2 * abs(CL_AERO) * A_AERO_M2
            f_drag_n = 0.5 * RHO_AIR_KGM3 * v**2 * CD_AERO * A_AERO_M2
            f_rolling_n = CRR_TIRE * (MASS_KG * G_MS2 + f_aero_n)
            p_drag_w = f_drag_n * v
            p_rolling_w = f_rolling_n * v
            p_corn_w = ((MASS_KG * ay) ** 2 / (2.0 * CALPHA_AXLE_TOTAL)) * v
            p_model_w = p_drag_w + p_rolling_w + p_corn_w

            e_drag_j = float(np.trapezoid(p_drag_w, t_s))
            e_rolling_j = float(np.trapezoid(p_rolling_w, t_s))
            e_corn_j = float(np.trapezoid(p_corn_w, t_s))
            e_model_j = e_drag_j + e_rolling_j + e_corn_j
            e_measured_j = float(np.trapezoid(p_batt, t_s))
            e_regen_j = float(np.trapezoid(np.clip(-p_batt, 0.0, None), t_s))
            e_consumed_j = float(np.trapezoid(np.clip(p_batt, 0.0, None), t_s))
            e_residual_j = e_measured_j - e_model_j

            lt = float(np.nanmax(laptime_s[lap_idx]))
            if not np.isfinite(lt) or lt <= 0.0:
                lt = float(t_s[-1])

            row: dict[str, object] = {
                "run": run_name,
                "lap": int(lap),
                "laptime_s": lt,
                "power_source": power_source,
                "e_drag_j": e_drag_j,
                "e_rolling_j": e_rolling_j,
                "e_corn_j": e_corn_j,
                "e_model_j": e_model_j,
                "e_measured_j": e_measured_j,
                "e_residual_j": e_residual_j,
                "e_regen_j": e_regen_j,
                "e_consumed_j": e_consumed_j,
                "mean_speed_mps": float(np.nanmean(v)),
            }
            if include_samples:
                row["samples"] = {
                    "distance_m": dist_m[lap_idx],
                    "time_s": t_s,
                    "p_drag_w": p_drag_w,
                    "p_rolling_w": p_rolling_w,
                    "p_corn_w": p_corn_w,
                    "p_residual_w": p_batt - p_model_w,
                    "p_batt_w": p_batt,
                    "p_model_w": p_model_w,
                }
            rows.append(row)
            run_rows.append(row)

        if not run_rows:
            warnings.append(f"{run_name}: no valid laps for energy budget.")
            continue

        measured = np.array([float(r["e_measured_j"]) for r in run_rows])
        model = np.array([float(r["e_model_j"]) for r in run_rows])
        residual = np.array([float(r["e_residual_j"]) for r in run_rows])
        regen = np.array([float(r["e_regen_j"]) for r in run_rows])
        drag = np.array([float(r["e_drag_j"]) for r in run_rows])
        corn = np.array([float(r["e_corn_j"]) for r in run_rows])
        measured_pos = np.where(np.abs(measured) > 1e-6, measured, np.nan)
        run_kpis[run_name] = {
            "mean_energy_per_lap_kj": float(np.nanmean(measured) / 1000.0),
            "mean_model_efficiency": float(np.nanmean(model / measured_pos)),
            "mean_residual_kj": float(np.nanmean(residual) / 1000.0),
            "mean_regen_recovery_kj": float(np.nanmean(regen) / 1000.0),
            "mean_drag_share": float(np.nanmean(drag / measured_pos)),
            "mean_corn_share": float(np.nanmean(corn / measured_pos)),
            "laps": float(len(run_rows)),
            "power_source": power_source,
        }
        if np.nanmean(model / measured_pos) > 1.0:
            warnings.append(
                f"{run_name}: model energy exceeds measured battery energy on average; "
                "check current sign or vehicle parameters."
            )

    return rows, run_kpis, warnings


def energy_budget_per_lap_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """Per-lap physical energy model compared with measured battery energy."""
    rows, run_kpis, warnings = _energy_budget_lap_rows(dfs)
    if not rows:
        fig = _empty_energy_budget_fig("No valid energy-budget data.")
        return fig, {"runs": run_kpis, "warnings": warnings + ["No valid energy-budget data."]}

    rows = sorted(rows, key=lambda r: (str(r["run"]), int(r["lap"])))
    n = len(rows)
    centers = np.arange(n, dtype=float)
    x_model = centers - 0.18
    x_meas = centers + 0.18
    width = 0.34

    drag_kj = np.array([float(r["e_drag_j"]) for r in rows]) / 1000.0
    rolling_kj = np.array([float(r["e_rolling_j"]) for r in rows]) / 1000.0
    corn_kj = np.array([float(r["e_corn_j"]) for r in rows]) / 1000.0
    residual_kj = np.array([float(r["e_residual_j"]) for r in rows]) / 1000.0
    model_kj = drag_kj + rolling_kj + corn_kj
    measured_kj = np.array([float(r["e_measured_j"]) for r in rows]) / 1000.0
    laptime_s = np.array([float(r["laptime_s"]) for r in rows])
    run_names = [str(r["run"]) for r in rows]
    laps = np.array([int(r["lap"]) for r in rows])
    power_sources = [str(r["power_source"]) for r in rows]
    ticktext = [
        f"L{lap}<br>{run}" if len(dfs) > 1 else f"L{lap}"
        for lap, run in zip(laps, run_names)
    ]

    palette = ["#4DB3F2", "#F28C40", "#73D973", "#D973D9", "#FFD84D", "#66E0C2"]
    unique_runs = list(dict.fromkeys(run_names))
    run_color = {name: palette[i % len(palette)] for i, name in enumerate(unique_runs)}
    border_colors = [run_color[name] for name in run_names]

    fig = make_dark_figure(
        title="Energy Budget per Lap — Predicted vs Measured",
        xlabel="Lap",
        ylabel="Energy [kJ]",
    )

    custom = np.array(list(zip(run_names, laps, laptime_s, power_sources)), dtype=object)

    def _bar(
        *,
        x: np.ndarray,
        y: np.ndarray,
        base: np.ndarray,
        name: str,
        color: str,
        opacity: float,
        legendgroup: str,
        customdata: np.ndarray = custom,
    ) -> None:
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                base=base,
                width=width,
                name=name,
                legendgroup=legendgroup,
                marker=dict(
                    color=color,
                    opacity=opacity,
                    line=dict(color=border_colors, width=1.2),
                ),
                customdata=customdata,
                hovertemplate=(
                    "Run=%{customdata[0]}<br>"
                    "Lap=%{customdata[1]}<br>"
                    "Laptime=%{customdata[2]:.2f} s<br>"
                    "Power source=%{customdata[3]}<br>"
                    f"{name}=%{{y:.1f}} kJ<extra></extra>"
                ),
            )
        )

    zeros = np.zeros(n)
    _bar(x=x_model, y=drag_kj, base=zeros, name="Model drag", color="#4DB3F2", opacity=0.95, legendgroup="model")
    _bar(x=x_model, y=rolling_kj, base=drag_kj, name="Model rolling", color="#9CA3AF", opacity=0.95, legendgroup="model")
    _bar(x=x_model, y=corn_kj, base=drag_kj + rolling_kj, name="Model cornering scrub", color="#F28C40", opacity=0.95, legendgroup="model")

    _bar(x=x_meas, y=drag_kj, base=zeros, name="Measured: model drag", color="#4DB3F2", opacity=0.35, legendgroup="measured")
    _bar(x=x_meas, y=rolling_kj, base=drag_kj, name="Measured: model rolling", color="#9CA3AF", opacity=0.35, legendgroup="measured")
    _bar(x=x_meas, y=corn_kj, base=drag_kj + rolling_kj, name="Measured: model scrub", color="#F28C40", opacity=0.35, legendgroup="measured")
    _bar(x=x_meas, y=residual_kj, base=model_kj, name="Measured residual", color="#F25555", opacity=0.85, legendgroup="measured")

    y_top = np.maximum(model_kj, measured_kj)
    pad = max(3.0, float(np.nanmax(np.abs(y_top))) * 0.03)
    for x, y, lt in zip(centers, y_top, laptime_s):
        if np.isfinite(y) and np.isfinite(lt):
            fig.add_annotation(
                x=float(x),
                y=float(y + pad),
                text=f"{lt:.1f}s",
                showarrow=False,
                font=dict(color="rgba(235,235,235,0.72)", size=10),
                textangle=-35,
            )

    sources = ", ".join(sorted(set(power_sources)))
    fig.add_annotation(
        x=1.0,
        y=1.03,
        xref="paper",
        yref="paper",
        text=f"Measured source: {sources}",
        showarrow=False,
        xanchor="right",
        font=dict(color="rgba(235,235,235,0.65)", size=11),
    )
    fig.update_layout(
        height=760,
        barmode="overlay",
        bargap=0.25,
        title_font=dict(color=_TEXT, size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    fig.update_xaxes(tickvals=centers, ticktext=ticktext)

    table = pl.DataFrame({
        "Run": run_names,
        "Lap": laps,
        "Laptime [s]": np.round(laptime_s, 3),
        "Measured [kJ]": np.round(measured_kj, 2),
        "Model [kJ]": np.round(model_kj, 2),
        "Residual [kJ]": np.round(residual_kj, 2),
        "Drag [kJ]": np.round(drag_kj, 2),
        "Rolling [kJ]": np.round(rolling_kj, 2),
        "Cornering [kJ]": np.round(corn_kj, 2),
        "Regen [kJ]": np.round(np.array([float(r["e_regen_j"]) for r in rows]) / 1000.0, 2),
        "Consumed [kJ]": np.round(np.array([float(r["e_consumed_j"]) for r in rows]) / 1000.0, 2),
        "Mean speed [m/s]": np.round(np.array([float(r["mean_speed_mps"]) for r in rows]), 2),
        "Power source": power_sources,
    })
    kpis: dict[str, object] = {
        **run_kpis,
        "runs": run_kpis,
        "table": table,
        "warnings": warnings,
        "power_sources": sorted(set(power_sources)),
    }
    return fig, kpis


def energy_budget_breakdown_fig(
    dfs: dict[str, pl.DataFrame],
    run_name: str | None = None,
    lap: int | None = None,
) -> go.Figure:
    """Power-budget breakdown along distance for one lap."""
    rows, _run_kpis, warnings = _energy_budget_lap_rows(dfs, include_samples=True)
    rows_with_samples = [r for r in rows if "samples" in r]
    if not rows_with_samples:
        message = warnings[0] if warnings else "No valid energy-budget lap samples."
        return _empty_energy_budget_fig(message)

    if run_name is not None and lap is not None:
        selected = [
            r for r in rows_with_samples
            if str(r["run"]) == str(run_name) and int(r["lap"]) == int(lap)
        ]
    else:
        first_run = next(iter(dfs.keys()))
        selected = [r for r in rows_with_samples if str(r["run"]) == str(first_run)]

    if not selected:
        selected = rows_with_samples

    row = min(selected, key=lambda r: float(r["laptime_s"]))
    samples = row["samples"]
    assert isinstance(samples, dict)

    dist_m = np.asarray(samples["distance_m"], dtype=float)
    if not np.any(np.isfinite(dist_m)) or float(np.nanmax(dist_m)) <= 1.0:
        dist_m = np.asarray(samples["time_s"], dtype=float)
        xlabel = "Lap time [s]"
    else:
        xlabel = "Distance [m]"

    fig = make_dark_figure(
        title=f"Energy Budget Breakdown — {row['run']} L{int(row['lap'])}",
        xlabel=xlabel,
        ylabel="Power [kW]",
    )
    series = [
        ("Drag", np.asarray(samples["p_drag_w"], dtype=float) / 1000.0, "#4DB3F2"),
        ("Rolling", np.asarray(samples["p_rolling_w"], dtype=float) / 1000.0, "#9CA3AF"),
        ("Cornering scrub", np.asarray(samples["p_corn_w"], dtype=float) / 1000.0, "#F28C40"),
        ("Residual", np.asarray(samples["p_residual_w"], dtype=float) / 1000.0, "#F25555"),
    ]
    for name, values, color in series:
        fig.add_trace(
            go.Scatter(
                x=dist_m,
                y=values,
                mode="lines",
                name=name,
                stackgroup="budget",
                line=dict(color=color, width=1.2),
                hovertemplate=f"{name}: %{{y:.1f}} kW<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=dist_m,
            y=np.asarray(samples["p_batt_w"], dtype=float) / 1000.0,
            mode="lines",
            name="Measured battery",
            line=dict(color="#FFFFFF", width=1.6),
            hovertemplate="Measured battery: %{y:.1f} kW<extra></extra>",
        )
    )
    fig.update_layout(height=520, title_font=dict(color=_TEXT, size=14))
    return fig


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

        cols = cols_to_numpy(df, ["laps", "laptime", *power_cols])
        laps = cols["laps"]
        laptime = cols["laptime"]
        powers = {w: cols[f"{w}_actualPower"] for w in WHEELS}

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
               soc_drop_per_lap, mean_voltage, min_voltage,
               mean_current, table, warnings.
    """
    cols_needed = ["laps", "laptime", "TimeStamp", "SoC", "Vbat", "Current", "Vmin"]

    laps_all: list[int] = []
    lt_all: list[float] = []
    soc_all: list[float] = []
    sd_all: list[float] = []
    vm_all: list[float] = []
    vn_all: list[float] = []
    vp05_all: list[float] = []
    im_all: list[float] = []
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

        cols = cols_to_numpy(df, ["laps", "laptime", "SoC", "Vbat", "Current", "Vmin"])
        laps = cols["laps"]
        laptime = cols["laptime"]
        soc_arr = cols["SoC"]
        vbat = cols["Vbat"]
        current = cols["Current"]
        vmin = cols["Vmin"]

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
        rows=2,
        titles=[
            "End-of-Lap SoC",
            "Battery Voltage",
        ],
        ylabels=["SoC [%]", "Voltage [V]"],
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
    fig.update_xaxes(title_text=xlabel, row=2, col=1)
    if x_mode == "laps":
        fig.update_xaxes(tickvals=np.sort(lp.astype(int)), row=2, col=1)
    fig.update_layout(
        height=560,
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
    }
    if len(dfs) > 1:
        table["Run"] = run_all

    kpis = {
        "soc_start":       soc_start,
        "soc_end":         soc_end,
        "soc_total_drop":  soc_total_drop,
        "voltage_sag":     float(np.nanmean(sag)),
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

        temp_cols = [
            *[f"{w}_motorTemperature" for w in WHEELS],
            *[f"{w}_inverterTemperature" for w in WHEELS],
            "Tmax",
            "Tavg",
        ]
        cols = cols_to_numpy(df, ["laps", "laptime", *temp_cols])
        laps = cols["laps"]
        laptime = cols["laptime"]

        for lap in unique_laps(laps):
            idx = laps == lap
            if idx.sum() < 5:
                continue

            tm = [
                float(
                    np.nanpercentile(cols[f"{w}_motorTemperature"][idx], 95)
                )
                for w in WHEELS
            ]
            ti = [
                float(
                    np.nanpercentile(cols[f"{w}_inverterTemperature"][idx], 95)
                )
                for w in WHEELS
            ]
            tb_max = float(np.nanmax(cols["Tmax"][idx]))
            tb_avg = float(
                np.nanpercentile(cols["Tavg"][idx], 95)
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
        float(((p_kw >= POWER_NEAR_CAP_KW) & (p_kw <= POWER_CAP_KW) & full).sum() / max(full.sum(), 1) * 100.0)
        if full.any() else np.nan
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
    fig_p.add_trace(go.Scattergl(
        x=time_s, y=p_kw, mode="lines", name="P_bat",
        line=dict(color="#4DB3F2", width=1.0),
    ))
    fig_p.add_hline(y=POWER_CAP_KW,
                    line=dict(color="#E74C3C", dash="dash", width=1.4),
                    annotation_text=f"FS rule {POWER_CAP_KW:.0f} kW",
                    annotation_position="top right")
    fig_p.add_hline(y=POWER_NEAR_CAP_KW,
                    line=dict(color="rgba(115, 217, 115, 0.6)", dash="dot", width=1.0),
                    annotation_text=f"{POWER_NEAR_CAP_KW:.0f} kW",
                    annotation_position="bottom right")

    # ── Fig 2: histogram of P_bat at full throttle ────────────────────────────
    fig_hist = make_dark_figure(
        title="P_bat distribution at full throttle (Throttle ≥ 80 %)",
        xlabel="P_bat [kW]",
        ylabel="Density",
    )
    if full.any():
        fig_hist.add_trace(go.Histogram(
            x=p_kw[full], name="P_bat | full throttle",
            histnorm="probability density",
            marker=dict(color="#F28C40"),
            opacity=0.85, nbinsx=80,
        ))
    fig_hist.add_vrect(x0=POWER_NEAR_CAP_KW, x1=POWER_CAP_KW,
                       fillcolor="rgba(115, 217, 115, 0.10)", line_width=0)
    fig_hist.add_vline(x=POWER_CAP_KW,
                       line=dict(color="#E74C3C", dash="dash", width=1.4))

    # ── Fig 3: peak P_bat per lap ─────────────────────────────────────────────
    fig_peak = make_dark_figure(
        title="Peak P_bat per lap",
        xlabel="Lap",
        ylabel="Peak P_bat [kW]",
    )
    fig_peak.add_trace(go.Bar(
        x=lap_ids, y=peak_per_lap,
        marker=dict(color="#9B59B6"),
        name="Peak",
        text=[f"{p:.1f}" if np.isfinite(p) else "" for p in peak_per_lap],
        textposition="outside",
    ))
    fig_peak.add_hline(y=POWER_CAP_KW,
                       line=dict(color="#E74C3C", dash="dash", width=1.4))

    kpis = {
        "pct_over_cap": pct_over_cap,
        "n_overshoot_events": int(n_overshoot),
        "peak_kw": peak_kw,
        "pct_near_cap_at_full": near_cap_at_full,
        "peak_kw_per_lap": dict(zip(lap_ids, peak_per_lap)),
        "pct_at_cap_per_lap": dict(zip(lap_ids, pct_at_cap_per_lap)),
    }
    return [fig_p, fig_hist, fig_peak], kpis


CTRL_TORQUE_ALIASES = {
    "tv": {
        "FL": ("TV_FL_Trq", "tv_fl_torque"),
        "FR": ("TV_FR_Trq", "tv_fr_torque"),
        "RL": ("TV_RL_Trq", "tv_rl_torque"),
        "RR": ("TV_RR_Trq", "tv_rr_torque"),
    },
    "tc": {
        "FL": ("TC_FL_MaxTrq", "tc_fl_torque"),
        "FR": ("TC_FR_MaxTrq", "tc_fr_torque"),
        "RL": ("TC_RL_MaxTrq", "tc_rl_torque"),
        "RR": ("TC_RR_MaxTrq", "tc_rr_torque"),
    },
    "pc": {
        "FL": ("PC_FL_Trq", "pc_fl_torque"),
        "FR": ("PC_FR_Trq", "pc_fr_torque"),
        "RL": ("PC_RL_Trq", "pc_rl_torque"),
        "RR": ("PC_RR_Trq", "pc_rr_torque"),
    },
    "rb": {
        "FL": ("RB_FL_Trq", "rb_fl_torque"),
        "FR": ("RB_FR_Trq", "rb_fr_torque"),
        "RL": ("RB_RL_Trq", "rb_rl_torque"),
        "RR": ("RB_RR_Trq", "rb_rr_torque"),
    },
    "master": {
        "FL": ("Master_frontLeftTrq", "master_fl_torque", "master_front_left_torque"),
        "FR": ("Master_frontRightTrq", "master_fr_torque", "master_front_right_torque"),
        "RL": ("Master_rearLeftTrq", "master_rl_torque", "master_rear_left_torque"),
        "RR": ("Master_rearRightTrq", "master_rr_torque", "master_rear_right_torque"),
    },
    "actual": {
        "FL": ("FL_actualTorque", "fl_actual_torque"),
        "FR": ("FR_actualTorque", "fr_actual_torque"),
        "RL": ("RL_actualTorque", "rl_actual_torque"),
        "RR": ("RR_actualTorque", "rr_actual_torque"),
    },
}


def _first_existing_col(df: pl.DataFrame, aliases: tuple[str, ...]) -> str | None:
    return next((col for col in aliases if col in df.columns), None)


def _series_or_nan(df: pl.DataFrame, aliases: tuple[str, ...]) -> np.ndarray:
    col = _first_existing_col(df, aliases)
    if col is None:
        return np.full(len(df), np.nan, dtype=float)
    return df[col].to_numpy().astype(float)


def _torque_matrix(df: pl.DataFrame, group: str) -> np.ndarray:
    return np.stack([
        _series_or_nan(df, CTRL_TORQUE_ALIASES[group][w]) for w in WHEELS
    ], axis=1)


def pc_master_attribution_figs_kpis(df: pl.DataFrame) -> tuple[list[go.Figure], dict]:
    """Observable PC/Master behaviour: pedal fidelity and power-cap performance loss."""
    df = ensure_complete_laps_df(df)
    if len(df) == 0:
        raise ValueError("No valid samples for PC/Master behaviour metrics.")

    time_s = df["TimeStamp"].to_numpy().astype(float) - float(df["TimeStamp"][0])
    dist_m = lap_dist_from_gps(df)
    ax_col = "Filtering_VN_ax" if "Filtering_VN_ax" in df.columns else "VN_ax"
    ax = df[ax_col].to_numpy().astype(float) if ax_col in df.columns else np.full(len(df), np.nan)
    vx_col = "Est_vxCOG" if "Est_vxCOG" in df.columns else "VN_vx"
    vx = df[vx_col].to_numpy().astype(float) if vx_col in df.columns else np.full(len(df), np.nan)
    command = _series_or_nan(df, ("LLC_Command", "llc_command"))
    throttle = _series_or_nan(df, ("Throttle", "throttle"))
    vbat = _series_or_nan(df, ("Vbat", "v_bat", "V_bat"))
    current = _series_or_nan(df, ("Current", "current"))
    p_bat_kw = vbat * current / 1000.0

    pc = _torque_matrix(df, "pc")
    master = _torque_matrix(df, "master")
    actual = _torque_matrix(df, "actual")
    master_total = np.nansum(master, axis=1)
    actual_total = np.nansum(actual, axis=1)
    pc_total = np.nansum(pc, axis=1)
    actual_track_err = actual_total - master_total
    pc_active = np.any(pc < -1.0e-6, axis=1)
    pc_cut_nm = -np.nansum(np.minimum(pc, 0.0), axis=1)
    accel_mask = (command >= 0.0) & (throttle >= 5.0) & np.isfinite(ax) & (np.abs(vx) >= 4.0)
    full_throttle = accel_mask & (throttle >= 95.0)
    pc_full = full_throttle & pc_active
    no_pc_full = full_throttle & ~pc_active
    ax_loss_pc = (
        float(np.nanmedian(ax[no_pc_full]) - np.nanmedian(ax[pc_full]))
        if pc_full.any() and no_pc_full.any() else np.nan
    )
    pbat_over_78 = np.isfinite(p_bat_kw) & (p_bat_kw >= 78.0)
    cap_near_pct = float(np.nanmean(pbat_over_78[full_throttle]) * 100.0) if full_throttle.any() else np.nan

    def _corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
        valid = mask & np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) < 20:
            return np.nan
        return float(np.corrcoef(x[valid], y[valid])[0, 1])

    pedal_ax_corr = _corr(throttle, ax, accel_mask)
    pedal_torque_corr = _corr(throttle, actual_total, accel_mask)
    pc_cut_ax_corr = _corr(pc_cut_nm, ax, full_throttle)

    order = np.argsort(dist_m)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=("Driver demand", "Car acceleration", "Power cap / PC cut"),
    )
    fig.add_trace(go.Scattergl(
        x=dist_m[order], y=throttle[order], mode="lines", name="Throttle",
        line=dict(color="#F28C40", width=1.1),
    ), row=1, col=1)
    fig.add_trace(go.Scattergl(
        x=dist_m[order], y=command[order] * 100.0, mode="lines", name="LLC command x100",
        line=dict(color="#EBEBEB", width=0.9, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scattergl(
        x=dist_m[order], y=ax[order], mode="markers", name="ax",
        marker=dict(color=pc_cut_nm[order], colorscale="Turbo", size=3, opacity=0.50, colorbar=dict(title="PC cut")),
    ), row=2, col=1)
    fig.add_trace(go.Scattergl(
        x=dist_m[order], y=p_bat_kw[order], mode="lines", name="Pbat",
        line=dict(color="#4DB3F2", width=1.0),
    ), row=3, col=1)
    fig.add_trace(go.Scattergl(
        x=dist_m[order], y=pc_cut_nm[order], mode="lines", name="PC cut",
        line=dict(color="#F28C40", width=1.1, dash="dash"),
    ), row=3, col=1)
    fig.update_layout(
        title=dict(text="PC/Master: pedal fidelity and power-cap performance loss", font=dict(size=14, color="#EBEBEB")),
        paper_bgcolor="#141417",
        plot_bgcolor="#141417",
        font=dict(color="#EBEBEB", size=11),
        height=830,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0.0),
    )
    fig.update_xaxes(title_text="Distance [m]", row=3, col=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(title_text="Demand [%]", row=1, col=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(title_text="ax [m/s²]", row=2, col=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(title_text="kW / Nm", row=3, col=1, gridcolor="rgba(128,128,128,0.2)")

    fig_rel = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.10,
        subplot_titles=("Throttle vs acceleration", "PC cut vs acceleration at full throttle"),
    )
    rel_mask = accel_mask & np.isfinite(throttle) & np.isfinite(ax)
    fig_rel.add_trace(go.Scattergl(
        x=throttle[rel_mask],
        y=ax[rel_mask],
        mode="markers",
        name="Pedal -> ax",
        marker=dict(color=p_bat_kw[rel_mask], colorscale="Turbo", size=4, opacity=0.45, colorbar=dict(title="Pbat")),
    ), row=1, col=1)
    pc_rel = full_throttle & np.isfinite(pc_cut_nm) & np.isfinite(ax)
    fig_rel.add_trace(go.Scattergl(
        x=pc_cut_nm[pc_rel],
        y=ax[pc_rel],
        mode="markers",
        name="PC cut -> ax",
        marker=dict(color=p_bat_kw[pc_rel], colorscale="Turbo", size=4, opacity=0.45, colorbar=dict(title="Pbat")),
    ), row=1, col=2)
    fig_rel.update_layout(
        title=dict(text="PC/Master variable relationships", font=dict(size=14, color="#EBEBEB")),
        paper_bgcolor="#141417",
        plot_bgcolor="#141417",
        font=dict(color="#EBEBEB", size=11),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0.0),
    )
    fig_rel.update_xaxes(title_text="Throttle [%]", row=1, col=1, gridcolor="rgba(128,128,128,0.2)")
    fig_rel.update_yaxes(title_text="ax [m/s²]", row=1, col=1, gridcolor="rgba(128,128,128,0.2)")
    fig_rel.update_xaxes(title_text="PC cut [Nm]", row=1, col=2, gridcolor="rgba(128,128,128,0.2)")
    fig_rel.update_yaxes(title_text="ax [m/s²]", row=1, col=2, gridcolor="rgba(128,128,128,0.2)")

    laps = df["laps"].to_numpy().astype(float)
    lap_rows: list[dict[str, object]] = []
    for lap in unique_laps(laps):
        lm = laps == lap
        if int(lm.sum()) < 20:
            continue
        lem = lm & accel_mask
        lpc = lm & pc_full
        lnopc = lm & no_pc_full
        lap_ax_loss = (
            float(np.nanmedian(ax[lnopc]) - np.nanmedian(ax[lpc]))
            if lpc.any() and lnopc.any() else np.nan
        )
        lap_rows.append({
            "Lap": int(lap),
            "Pedal-ax corr": round(_corr(throttle, ax, lem), 3),
            "Pedal-torque corr": round(_corr(throttle, actual_total, lem), 3),
            "PC active @ WOT [%]": round(float(np.nanmean(pc_active[lm & full_throttle]) * 100.0), 1) if (lm & full_throttle).any() else np.nan,
            "ax loss PC [m/s²]": round(lap_ax_loss, 3) if np.isfinite(lap_ax_loss) else np.nan,
            "Actual/Master MAE [Nm]": round(float(np.nanmean(np.abs(actual_track_err[lm]))), 2),
            "Peak Pbat [kW]": round(float(np.nanmax(p_bat_kw[lm])), 1),
        })

    kpis = {
        "actual_master_mae_nm": float(np.nanmean(np.abs(actual_track_err))),
        "pedal_ax_corr": pedal_ax_corr,
        "pedal_torque_corr": pedal_torque_corr,
        "pc_active_wot_pct": float(np.nanmean(pc_active[full_throttle]) * 100.0) if full_throttle.any() else np.nan,
        "ax_loss_pc_ms2": ax_loss_pc,
        "pc_cut_ax_corr": pc_cut_ax_corr,
        "cap_near_wot_pct": cap_near_pct,
        "peak_pbat_kw": float(np.nanmax(p_bat_kw)) if np.any(np.isfinite(p_bat_kw)) else np.nan,
        "eval_samples": int(accel_mask.sum()),
        "table": pl.DataFrame(lap_rows) if lap_rows else pl.DataFrame(),
        "warnings": [],
    }
    return [fig, fig_rel], kpis


if __name__ == "__main__":
    main()
