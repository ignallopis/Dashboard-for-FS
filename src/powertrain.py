"""powertrain.py
--------------
Powertrain KPIs:
  1. Energy per lap       (kWh, mean power, trend, efficiency correlation)
  2. Thermal evolution    (motor, inverter, battery P95 per lap + thermal slope)

Requires lapcount.py to have been run first.

Usage:
    python powertrain.py
"""
from __future__ import annotations
import numpy as np
import polars as pl
import plotly.graph_objects as go
from scipy.integrate import cumulative_trapezoid

from utils import (
    make_dark_figure, add_lap_scatter, add_trend_line,
    exclude_lap0_and_last_lap, unique_laps,
    WHEEL_COLORS,
)

CSV_PATH = 'data/run4_2025-08-24.csv'


def _load(columns: list[str]) -> dict[str, np.ndarray]:
    df = pl.read_csv(CSV_PATH, columns=columns)
    return {c: df[c].to_numpy().astype(float) for c in columns}


# ── 1. Energy per lap ─────────────────────────────────────────────────────────

def energy_per_lap() -> None:
    d = _load(['TimeStamp', 'laps', 'Vbat', 'Current'])
    time = d['TimeStamp'] - d['TimeStamp'][0]
    laps = d['laps']

    valid = np.isfinite(time) & np.isfinite(laps) & \
            np.isfinite(d['Vbat']) & np.isfinite(d['Current'])

    time   = time[valid]
    laps   = laps[valid]
    p_kw   = (d['Vbat'][valid] * d['Current'][valid]) / 1000.0  # [kW]

    # Cumulative energy [kWh]
    e_cum = cumulative_trapezoid(p_kw, time, initial=0.0) / 3600.0

    # Filter out lap 0
    valid_laps_mask = laps > 0
    time   = time[valid_laps_mask]
    laps   = laps[valid_laps_mask]
    p_kw   = p_kw[valid_laps_mask]
    e_cum  = e_cum[valid_laps_mask]

    lap_list = unique_laps(laps)
    # Remove last lap (may be incomplete)
    if len(lap_list) >= 2:
        lap_list = lap_list[:-1]

    n          = len(lap_list)
    e_lap      = np.full(n, np.nan)
    lt_s       = np.full(n, np.nan)
    p_avg      = np.full(n, np.nan)

    for i, lap in enumerate(lap_list):
        idx = np.where(laps == lap)[0]
        if len(idx) < 2:
            continue
        e_lap[i] = e_cum[idx[-1]] - e_cum[idx[0]]
        lt_s[i]  = time[idx[-1]] - time[idx[0]]
        p_avg[i] = np.nanmean(p_kw[idx])

    ok = np.isfinite(e_lap) & np.isfinite(lt_s) & (e_lap > 0)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    e_ok  = e_lap[ok]
    lt_ok = lt_s[ok]
    l_ok  = lap_list[ok]

    e_mean = np.nanmean(e_ok)
    e_std  = np.nanstd(e_ok)
    cv     = 100.0 * e_std / e_mean if e_mean > 0 else np.nan
    corr   = float(np.corrcoef(lt_ok, e_ok)[0, 1]) if len(e_ok) >= 2 else np.nan

    slope_energy_time = np.polyfit(lt_ok, e_ok, 1)[0] if len(e_ok) >= 2 else np.nan
    slope_energy_lap  = np.polyfit(l_ok,  e_ok, 1)[0] if len(e_ok) >= 2 else np.nan

    print('\n─── Energy KPIs ───')
    print(f'Mean energy per lap       : {e_mean:.4f} kWh')
    print(f'Std energy per lap        : {e_std:.4f} kWh')
    print(f'Coefficient of variation  : {cv:.2f} %')
    print(f'Energy trend (per lap)    : {slope_energy_lap:+.5f} kWh/lap')
    print(f'Energy sensitivity (vs t) : {slope_energy_time:+.5f} kWh/s')
    print(f'Correlation E_lap vs t    : {corr:.3f}')
    print(f'\nBest  lap (energy): Lap {int(l_ok[np.argmin(e_ok)])} → {e_ok.min():.4f} kWh')
    print(f'Worst lap (energy): Lap {int(l_ok[np.argmax(e_ok)])} → {e_ok.max():.4f} kWh')

    print(f'\n{"Lap":>4}  {"LapTime[s]":>10}  {"E_lap[kWh]":>11}  {"P_avg[kW]":>10}')
    for lap, lt, e, pa in zip(l_ok, lt_ok, e_ok, p_avg[ok]):
        print(f'{int(lap):>4}  {lt:>10.3f}  {e:>11.5f}  {pa:>10.3f}')

    fig = make_dark_figure(
        title='Energy per Lap vs Lap Time',
        xlabel='Lap time [s]', ylabel='Energy per lap [kWh]',
    )
    add_lap_scatter(fig, lt_ok, e_ok, l_ok)
    add_trend_line(fig, lt_ok, e_ok)
    fig.show()


# ── 2. Thermal evolution ──────────────────────────────────────────────────────

def thermal_evolution() -> None:
    motor_cols   = [f'{w}_motorTemperature'   for w in ('FL', 'FR', 'RL', 'RR')]
    inv_cols     = [f'{w}_inverterTemperature' for w in ('FL', 'FR', 'RL', 'RR')]
    batt_cols    = ['Tmax', 'Tavg']

    d = _load(['laps'] + motor_cols + inv_cols + batt_cols)
    laps = d['laps']

    # Remove lap 0 and last lap
    valid = laps > 0
    d = {k: v[valid] for k, v in d.items()}
    laps = d['laps']
    all_laps = unique_laps(laps)
    if len(all_laps) >= 2:
        d = {k: v[laps != all_laps.max()] for k, v in d.items()}
        laps = d['laps']

    lap_list = unique_laps(laps)
    n        = len(lap_list)

    # P95 per lap for motors and inverters; max for battery Tmax
    T_motor = np.full((n, 4), np.nan)
    T_inv   = np.full((n, 4), np.nan)
    T_batt  = np.full((n, 2), np.nan)   # [Tmax_max, Tavg_p95]

    wheels = ('FL', 'FR', 'RL', 'RR')

    for i, lap in enumerate(lap_list):
        idx = laps == lap
        if idx.sum() < 5:
            continue
        for j, w in enumerate(wheels):
            T_motor[i, j] = np.nanpercentile(d[f'{w}_motorTemperature'][idx],  95)
            T_inv[i, j]   = np.nanpercentile(d[f'{w}_inverterTemperature'][idx], 95)
        T_batt[i, 0] = np.nanmax(d['Tmax'][idx])
        T_batt[i, 1] = np.nanpercentile(d['Tavg'][idx], 95)

    ok = np.isfinite(T_motor[:, 0])
    lp = lap_list[ok]
    Tm = T_motor[ok]
    Ti = T_inv[ok]
    Tb = T_batt[ok]

    # ── KPIs ─────────────────────────────────────────────────────────────────
    motor_lr_delta = (Tm[:, 0] + Tm[:, 2]) / 2 - (Tm[:, 1] + Tm[:, 3]) / 2
    inv_lr_delta   = (Ti[:, 0] + Ti[:, 2]) / 2 - (Ti[:, 1] + Ti[:, 3]) / 2

    def slope(x, y):
        return np.polyfit(x, y, 1)[0] if len(x) >= 2 else np.nan

    print('\n─── Thermal KPIs ───')
    print(f'Motor ΔT Left−Right (mean)  : {np.nanmean(motor_lr_delta):+.2f} °C')
    print(f'Inverter ΔT Left−Right (mean): {np.nanmean(inv_lr_delta):+.2f} °C')
    for j, w in enumerate(wheels):
        mi_delta = np.nanmean(Tm[:, j] - Ti[:, j])
        print(f'Motor−Inverter ΔT ({w})     : {mi_delta:+.2f} °C')
    print(f'Motor avg thermal slope     : {slope(lp, np.nanmean(Tm, axis=1)):+.2f} °C/lap')
    print(f'Inverter avg thermal slope  : {slope(lp, np.nanmean(Ti, axis=1)):+.2f} °C/lap')
    print(f'Battery Tmax thermal slope  : {slope(lp, Tb[:, 0]):+.2f} °C/lap')

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = make_dark_figure(
        title='Thermal Evolution per Lap',
        xlabel='Lap', ylabel='Temperature [°C]',
    )
    motor_line_styles = ['solid',  'solid',  'solid',  'solid']
    inv_line_styles   = ['dash',   'dash',   'dash',   'dash']

    for j, w in enumerate(wheels):
        fig.add_trace(go.Scatter(
            x=lp, y=Tm[:, j], mode='lines+markers',
            name=f'Motor {w}',
            line=dict(color=WHEEL_COLORS[w], dash=motor_line_styles[j], width=1.5),
            marker=dict(size=7),
        ))
        fig.add_trace(go.Scatter(
            x=lp, y=Ti[:, j], mode='lines',
            name=f'Inv {w}',
            line=dict(color=WHEEL_COLORS[w], dash=inv_line_styles[j], width=1.0),
        ))

    fig.add_trace(go.Scatter(
        x=lp, y=Tb[:, 0], mode='lines+markers',
        name='Battery Tmax',
        line=dict(color='white', width=2),
        marker=dict(size=7),
    ))
    fig.update_xaxes(tickvals=lp.astype(int))
    fig.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    energy_per_lap()
    thermal_evolution()


if __name__ == '__main__':
    main()
