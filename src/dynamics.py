"""dynamics.py
------------
Vehicle dynamics KPIs:
  1. Slip angle efficiency                               (ay / |SA| per wheel)
  2. Understeer angle evolution                          (delta_actual - delta_ideal)

Requires lapcount.py to have been run first.

Usage:
    python dynamics.py
"""
from __future__ import annotations
import numpy as np
import polars as pl
import plotly.graph_objects as go

from utils import (
    make_dark_figure, add_lap_scatter, add_trend_line, add_zero_line,
    keep_min_duration_segments, exclude_lap0_and_last_lap,
    robust_dt, unique_laps,
    WHEEL_COLORS, WHEEL_SYMBOLS,
)

CSV_PATH = 'data/run4_2025-08-24.csv'

# ── Shared filter parameters ──────────────────────────────────────────────────
AY_THRESHOLD        = 2.0    # [m/s²] min |ay| to classify as cornering
STEERING_THRESHOLD  = 0.05   # [rad]  min |steering| for cornering
MIN_SPEED           = 3.0    # [m/s]  min vehicle speed
MIN_CORNER_DURATION = 0.20   # [s]    min cornering event length
MIN_CORNER_SAMPLES  = 50     # per lap

WHEELBASE_EQ        = 1.53   # [m]  equivalent wheelbase for bicycle model
MIN_SA_MEAN_DEG     = 1.0    # [deg] slip angles below this are ignored
MAX_SA_MEAN_DEG     = 15.0   # [deg]
MAX_SA_EFF          = 5.0    # [m/s²/deg] sanity cap on efficiency


# ── Data loading ──────────────────────────────────────────────────────────────

def _load(columns: list[str]) -> dict[str, np.ndarray]:
    df = pl.read_csv(CSV_PATH, columns=columns)
    return {c: df[c].to_numpy().astype(float) for c in columns}


def _base_validity(*arrays: np.ndarray) -> np.ndarray:
    return np.all(np.stack([np.isfinite(a) for a in arrays], axis=1), axis=1)


# ── 1. Slip angle efficiency ──────────────────────────────────────────────────

def slip_angle_efficiency() -> None:
    d = _load(['TimeStamp', 'laps', 'laptime',
               'Filtering_VN_ay', 'Steering', 'VN_vx',
               'Est_SAFL', 'Est_SAFR', 'Est_SARL', 'Est_SARR'])
    d['time'] = d['TimeStamp'] - d['TimeStamp'][0]

    valid = _base_validity(*d.values())
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)

    dt      = robust_dt(d['time'])
    ay      = d['Filtering_VN_ay']
    steering = d['Steering']
    vx      = d['VN_vx']
    laps    = d['laps']
    laptime = d['laptime']

    raw_corner = (np.abs(ay) >= AY_THRESHOLD) & \
                 (np.abs(steering) >= STEERING_THRESHOLD) & \
                 (np.abs(vx) >= MIN_SPEED)
    corner_mask = keep_min_duration_segments(raw_corner, MIN_CORNER_DURATION, dt)

    lap_list  = unique_laps(laps)
    n         = len(lap_list)
    lt_val    = np.full(n, np.nan)
    n_samps   = np.zeros(n, dtype=int)
    ay_mean   = np.full(n, np.nan)

    sa_mean = {w: np.full(n, np.nan) for w in ('FL', 'FR', 'RL', 'RR')}
    sa_eff  = {w: np.full(n, np.nan) for w in ('FL', 'FR', 'RL', 'RR')}

    for i, lap in enumerate(lap_list):
        lm  = laps == lap
        lcm = lm & corner_mask
        n_samps[i] = lcm.sum()
        if lm.any():
            lt_val[i] = laptime[lm].max()
        if n_samps[i] < MIN_CORNER_SAMPLES:
            continue

        ay_mean[i] = np.nanmean(np.abs(ay[lcm]))
        for w in ('FL', 'FR', 'RL', 'RR'):
            sa_deg = np.rad2deg(np.abs(d[f'Est_SA{w}'][lcm]))
            sa_mean[w][i] = np.nanmean(sa_deg)
            if sa_mean[w][i] > MIN_SA_MEAN_DEG:
                sa_eff[w][i] = ay_mean[i] / sa_mean[w][i]

    print('\n─── Slip Angle Efficiency [m/s²/deg] ───')
    header = f"{'Lap':>4}  {'LapTime':>8}" + \
             ''.join(f"  {'Eff_'+w:>10}" for w in ('FL', 'FR', 'RL', 'RR'))
    print(header)
    for i, lap in enumerate(lap_list):
        if n_samps[i] < MIN_CORNER_SAMPLES:
            continue
        vals = ''.join(
            f"  {sa_eff[w][i]:>10.3f}" if np.isfinite(sa_eff[w][i]) else f"  {'—':>10}"
            for w in ('FL', 'FR', 'RL', 'RR')
        )
        print(f'{int(lap):>4}  {lt_val[i]:>8.2f}{vals}')

    # Plot 1: mean SA per wheel vs lap time
    fig1 = make_dark_figure(
        title='Mean Slip Angle per Wheel vs Lap Time',
        xlabel='Lap time [s]', ylabel='Mean |SA| in corners [deg]',
    )
    for w in ('FL', 'FR', 'RL', 'RR'):
        ok = (n_samps >= MIN_CORNER_SAMPLES) & np.isfinite(sa_mean[w]) & \
             (sa_mean[w] > MIN_SA_MEAN_DEG) & (sa_mean[w] < MAX_SA_MEAN_DEG)
        if ok.any():
            add_lap_scatter(fig1, lt_val[ok], sa_mean[w][ok], lap_list[ok],
                            name=w, color=WHEEL_COLORS[w], symbol=WHEEL_SYMBOLS[w])
    fig1.show()

    # Plot 2: SA efficiency vs lateral acceleration
    fig2 = make_dark_figure(
        title='Lateral Load vs Slip Angle Used',
        xlabel='Mean |SA| in corners [deg]',
        ylabel='Mean |ay| in corners [m/s²]',
    )
    for w in ('FL', 'FR', 'RL', 'RR'):
        ok = (n_samps >= MIN_CORNER_SAMPLES) & np.isfinite(sa_mean[w]) & \
             np.isfinite(ay_mean) & (sa_mean[w] > MIN_SA_MEAN_DEG) & \
             (sa_mean[w] < MAX_SA_MEAN_DEG)
        if ok.any():
            add_lap_scatter(fig2, sa_mean[w][ok], ay_mean[ok], lap_list[ok],
                            name=w, color=WHEEL_COLORS[w], symbol=WHEEL_SYMBOLS[w])
    fig2.show()


# ── 2. Understeer angle evolution ─────────────────────────────────────────────

def understeer_angle() -> None:
    """
    Understeer angle = actual steering − ideal steering (bicycle model).

    ideal_steering = L * ay / vx²

    Positive → understeer, negative → oversteer.
    """
    # Prefer Est_vxCOG (kinematic estimate from state estimator)
    try:
        d = _load(['TimeStamp', 'laps', 'laptime',
                   'Steering', 'Filtering_VN_ay', 'Est_vxCOG'])
        vx_key = 'Est_vxCOG'
    except Exception:
        d = _load(['TimeStamp', 'laps', 'laptime',
                   'Steering', 'Filtering_VN_ay', 'VN_vx'])
        vx_key = 'VN_vx'

    d['time'] = d['TimeStamp'] - d['TimeStamp'][0]
    d['vx']   = d.pop(vx_key)

    valid = _base_validity(*d.values()) & (np.abs(d['vx']) >= 4.0)
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)

    dt        = robust_dt(d['time'])
    steering  = d['Steering']
    ay_filt   = d['Filtering_VN_ay']
    vx        = d['vx']
    laps      = d['laps']
    laptime   = d['laptime']

    raw_corner = (np.abs(ay_filt) >= AY_THRESHOLD) & \
                 (np.abs(steering) >= STEERING_THRESHOLD) & \
                 (np.abs(vx) >= 4.0)
    corner_mask = keep_min_duration_segments(raw_corner, MIN_CORNER_DURATION, dt)

    # Bicycle model: δ_ideal = L·ay / vx²
    ideal_steer = WHEELBASE_EQ * ay_filt / (vx ** 2)
    und_rad     = np.abs(steering) - np.abs(ideal_steer)
    und_deg     = np.rad2deg(und_rad)

    lap_list   = unique_laps(laps)
    n          = len(lap_list)
    und_mean   = np.full(n, np.nan)
    lt_val     = np.full(n, np.nan)
    n_samps    = np.zeros(n, dtype=int)

    for i, lap in enumerate(lap_list):
        lm  = laps == lap
        lcm = lm & corner_mask
        n_samps[i] = lcm.sum()
        if lm.any():
            lt_val[i] = laptime[lm].max()
        if n_samps[i] >= MIN_CORNER_SAMPLES:
            und_mean[i] = np.nanmean(und_deg[lcm])

    ok = np.isfinite(und_mean) & np.isfinite(lt_val) & \
         (n_samps >= MIN_CORNER_SAMPLES) & (np.abs(und_mean) < 20.0)

    print('\n─── Understeer Angle per Lap ───')
    print(f"{'Lap':>4}  {'LapTime[s]':>10}  {'Und_mean[deg]':>14}  {'Samples':>8}")
    for lap, lt, um, ns in zip(lap_list[ok], lt_val[ok], und_mean[ok], n_samps[ok]):
        print(f'{int(lap):>4}  {lt:>10.3f}  {um:>14.3f}  {ns:>8d}')

    fig = make_dark_figure(
        title='Average Understeer Angle per Lap',
        xlabel='Lap', ylabel='Mean understeer angle [deg]',
    )
    add_lap_scatter(fig, lap_list[ok], und_mean[ok], lap_list[ok])
    add_trend_line(fig, lap_list[ok], und_mean[ok])
    add_zero_line(fig, lap_list[ok])
    fig.update_xaxes(tickvals=lap_list[ok].astype(int))
    fig.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    slip_angle_efficiency()
    understeer_angle()


if __name__ == '__main__':
    main()
