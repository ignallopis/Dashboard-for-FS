"""tc.py
------
Traction Control (TC) KPIs — slip ratio regulation quality.

All metrics are computed during valid traction phases:
  throttle >= threshold  AND  ax >= threshold  AND  vx >= min_speed
  (optionally restricted to straights: |steering| <= threshold)

KPIs:
  1. Global SR MAE / Bias vs target (SR_target = 0.20)
  2. Per-wheel SR MAE / Bias + imbalance
  3. Time-in-target band [%]
  4. Overslip and underslip percentages
  5. Traction efficiency: mean ax when SR is in target
  6. Worst-wheel overslip

Requires lapcount.py to have been run first.

Usage:
    python tc.py
"""
from __future__ import annotations
import numpy as np
import polars as pl
import plotly.graph_objects as go

from utils import (
    make_dark_figure, add_lap_scatter, add_zero_line,
    keep_min_duration_segments, exclude_lap0_and_last_lap,
    robust_dt, unique_laps,
    WHEEL_COLORS, WHEEL_SYMBOLS,
)

CSV_PATH = 'data/run4_2025-08-24.csv'

# ── TC parameters ─────────────────────────────────────────────────────────────
SR_TARGET            = 0.20    # optimal slip ratio in acceleration
DELTA_SR             = 0.05    # ±band around target
THROTTLE_THRESHOLD   = 10.0   # [%]  min throttle to count as acceleration event
AX_THRESHOLD         = 0.50   # [m/s²]
MIN_SPEED            = 4.0    # [m/s]
STEERING_STRAIGHT    = 0.08   # [rad] |steering| <= this → straight line
USE_STRAIGHT_FILTER  = True   # restrict traction analysis to straights
MIN_EVENT_DURATION   = 0.15   # [s]  min duration for a traction event
MIN_SAMPLES_PER_LAP  = 40


def _load(columns: list[str]) -> dict[str, np.ndarray]:
    df = pl.read_csv(CSV_PATH, columns=columns)
    return {c: df[c].to_numpy().astype(float) for c in columns}


def _ax_signal() -> str:
    """Return column name for longitudinal acceleration."""
    try:
        pl.read_csv(CSV_PATH, columns=['Filtering_VN_ax'], n_rows=1)
        return 'Filtering_VN_ax'
    except Exception:
        return 'VN_ax'


def _vx_signal() -> str:
    """Return column name for vehicle speed."""
    try:
        pl.read_csv(CSV_PATH, columns=['Est_vxCOG'], n_rows=1)
        return 'Est_vxCOG'
    except Exception:
        return 'VN_vx'


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ax_col = _ax_signal()
    vx_col = _vx_signal()

    d = _load(['TimeStamp', 'laps', 'laptime',
               'Throttle', 'Steering', ax_col, vx_col,
               'Est_SRFL', 'Est_SRFR', 'Est_SRRL', 'Est_SRRR'])

    d['time'] = d['TimeStamp'] - d['TimeStamp'][0]
    d['ax']   = d.pop(ax_col)
    d['vx']   = d.pop(vx_col)

    valid = np.all(np.stack([np.isfinite(v) for v in d.values()], axis=1), axis=1)
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)

    dt       = robust_dt(d['time'])
    laps     = d['laps']
    laptime  = d['laptime']
    ax       = d['ax']
    vx       = d['vx']
    throttle = d['Throttle']
    steering = d['Steering']

    sr = {w: d[f'Est_SR{w}'] for w in ('FL', 'FR', 'RL', 'RR')}

    # ── Traction mask ─────────────────────────────────────────────────────────
    raw_traction = (throttle >= THROTTLE_THRESHOLD) & \
                   (ax >= AX_THRESHOLD) & \
                   (np.abs(vx) >= MIN_SPEED)
    if USE_STRAIGHT_FILTER:
        raw_traction &= (np.abs(steering) <= STEERING_STRAIGHT)
    traction_mask = keep_min_duration_segments(raw_traction, MIN_EVENT_DURATION, dt)

    # Pre-compute helpers
    sr_mat         = np.stack([sr[w] for w in ('FL', 'FR', 'RL', 'RR')], axis=1)  # (N,4)
    sr_global      = np.nanmean(sr_mat, axis=1)
    sr_worst       = np.nanmax(sr_mat, axis=1)
    e_mat          = sr_mat - SR_TARGET

    overslip_thr   = SR_TARGET + DELTA_SR
    underslip_thr  = SR_TARGET - DELTA_SR

    in_target_mat  = (sr_mat >= underslip_thr) & (sr_mat <= overslip_thr)   # (N,4) bool
    over_mat       = sr_mat > overslip_thr
    under_mat      = sr_mat < underslip_thr
    in_target_glob = (sr_global >= underslip_thr) & (sr_global <= overslip_thr)
    worst_over     = sr_worst > overslip_thr

    lap_list = unique_laps(laps)
    n        = len(lap_list)

    lt_val        = np.full(n, np.nan)
    trac_samps    = np.zeros(n, dtype=int)
    trac_cover    = np.full(n, np.nan)

    mae_global    = np.full(n, np.nan)
    bias_global   = np.full(n, np.nan)
    mae_worst     = np.full(n, np.nan)
    bias_worst    = np.full(n, np.nan)

    mae_w   = {w: np.full(n, np.nan) for w in ('FL', 'FR', 'RL', 'RR')}
    bias_w  = {w: np.full(n, np.nan) for w in ('FL', 'FR', 'RL', 'RR')}
    imb_mae = np.full(n, np.nan)
    imb_bias = np.full(n, np.nan)

    in_tgt_pct_glob = np.full(n, np.nan)
    in_tgt_pct_w    = {w: np.full(n, np.nan) for w in ('FL', 'FR', 'RL', 'RR')}
    over_pct_glob   = np.full(n, np.nan)
    under_pct_glob  = np.full(n, np.nan)
    over_pct_w      = {w: np.full(n, np.nan) for w in ('FL', 'FR', 'RL', 'RR')}
    worst_over_pct  = np.full(n, np.nan)

    ax_in_tgt_glob  = np.full(n, np.nan)

    for i, lap in enumerate(lap_list):
        lm  = laps == lap
        ltm = lm & traction_mask
        trac_samps[i] = int(ltm.sum())
        if lm.any():
            lt_val[i]    = laptime[lm].max()
            trac_cover[i] = ltm.sum() / lm.sum()
        if trac_samps[i] < MIN_SAMPLES_PER_LAP:
            continue

        # Global MAE / Bias
        e_g = sr_global[ltm] - SR_TARGET
        mae_global[i]  = np.nanmean(np.abs(e_g))
        bias_global[i] = np.nanmean(e_g)
        e_wst = sr_worst[ltm] - SR_TARGET
        mae_worst[i]   = np.nanmean(np.abs(e_wst))
        bias_worst[i]  = np.nanmean(e_wst)

        # Per-wheel MAE / Bias + imbalance
        wheel_maes  = []
        wheel_biases = []
        for j, w in enumerate(('FL', 'FR', 'RL', 'RR')):
            e_w = e_mat[ltm, j]
            mae_w[w][i]  = np.nanmean(np.abs(e_w))
            bias_w[w][i] = np.nanmean(e_w)
            wheel_maes.append(mae_w[w][i])
            wheel_biases.append(bias_w[w][i])
        imb_mae[i]  = max(wheel_maes)  - min(wheel_maes)
        imb_bias[i] = max(wheel_biases) - min(wheel_biases)

        # Time in target
        for j, w in enumerate(('FL', 'FR', 'RL', 'RR')):
            in_tgt_pct_w[w][i] = 100.0 * np.mean(in_target_mat[ltm, j])
        in_tgt_pct_glob[i] = np.mean(list(in_tgt_pct_w[w][i]
                                           for w in ('FL', 'FR', 'RL', 'RR')))

        # Over / under slip
        for j, w in enumerate(('FL', 'FR', 'RL', 'RR')):
            over_pct_w[w][i] = 100.0 * np.mean(over_mat[ltm, j])
        over_pct_glob[i]  = np.mean([over_pct_w[w][i]  for w in ('FL', 'FR', 'RL', 'RR')])
        under_pct_glob[i] = 100.0 * np.mean(np.all(under_mat[ltm], axis=1))
        worst_over_pct[i] = 100.0 * np.mean(worst_over[ltm])

        # Traction efficiency
        ax_in = ax[ltm & in_target_glob]
        if ax_in.size > 0:
            ax_in_tgt_glob[i] = np.nanmean(ax_in)

    # ── Valid masks ───────────────────────────────────────────────────────────
    base_ok  = np.isfinite(lt_val) & (trac_samps >= MIN_SAMPLES_PER_LAP)
    glob_ok  = base_ok & np.isfinite(mae_global)
    wheel_ok = base_ok & np.all(
        np.stack([np.isfinite(mae_w[w]) for w in ('FL', 'FR', 'RL', 'RR')], axis=1),
        axis=1
    )
    eff_ok   = base_ok & np.isfinite(ax_in_tgt_glob)

    # ── Print summary tables ──────────────────────────────────────────────────
    print('\n─── TC: Base diagnostics ───')
    print(f"{'Lap':>4}  {'LapTime[s]':>10}  {'TrcSamples':>11}  {'Coverage':>9}")
    for i in np.where(base_ok)[0]:
        print(f'{int(lap_list[i]):>4}  {lt_val[i]:>10.3f}  '
              f'{trac_samps[i]:>11d}  {trac_cover[i]:>9.3f}')

    print('\n─── TC: Global SR MAE / Bias ───')
    print(f"{'Lap':>4}  {'MAE_global':>10}  {'Bias_global':>11}  "
          f"{'MAE_worst':>10}  {'Bias_worst':>11}")
    for i in np.where(glob_ok)[0]:
        print(f'{int(lap_list[i]):>4}  {mae_global[i]:>10.4f}  '
              f'{bias_global[i]:>+11.4f}  {mae_worst[i]:>10.4f}  '
              f'{bias_worst[i]:>+11.4f}')

    print('\n─── TC: Per-Wheel MAE ───')
    print(f"{'Lap':>4}  {'MAE_FL':>8}  {'MAE_FR':>8}  {'MAE_RL':>8}  "
          f"{'MAE_RR':>8}  {'Imbalance':>10}")
    for i in np.where(wheel_ok)[0]:
        vals = '  '.join(f'{mae_w[w][i]:>8.4f}' for w in ('FL', 'FR', 'RL', 'RR'))
        print(f'{int(lap_list[i]):>4}  {vals}  {imb_mae[i]:>10.4f}')

    print('\n─── TC: Time in Target / Overslip / Underslip [%] ───')
    print(f"{'Lap':>4}  {'InTarget%':>10}  {'Over%':>7}  {'Under%':>8}  {'WorstOver%':>11}")
    for i in np.where(base_ok & np.isfinite(in_tgt_pct_glob))[0]:
        print(f'{int(lap_list[i]):>4}  {in_tgt_pct_glob[i]:>10.2f}  '
              f'{over_pct_glob[i]:>7.2f}  {under_pct_glob[i]:>8.2f}  '
              f'{worst_over_pct[i]:>11.2f}')

    print('\n─── TC: Traction Efficiency (mean ax when SR in target) ───')
    print(f"{'Lap':>4}  {'ax_in_target[m/s²]':>20}")
    for i in np.where(eff_ok)[0]:
        print(f'{int(lap_list[i]):>4}  {ax_in_tgt_glob[i]:>20.4f}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    wheels = ('FL', 'FR', 'RL', 'RR')

    # 1. Global MAE
    if glob_ok.any():
        fig = make_dark_figure('Global SR MAE vs Lap', 'Lap', 'MAE SR global')
        add_lap_scatter(fig, lap_list[glob_ok], mae_global[glob_ok], lap_list[glob_ok])
        fig.update_xaxes(tickvals=lap_list[glob_ok].astype(int))
        fig.show()

    # 2. Global Bias
    if glob_ok.any():
        fig = make_dark_figure('Global SR Bias vs Lap', 'Lap', 'Bias SR global')
        add_lap_scatter(fig, lap_list[glob_ok], bias_global[glob_ok],
                        lap_list[glob_ok], color='#F27070')
        add_zero_line(fig, lap_list[glob_ok])
        fig.update_xaxes(tickvals=lap_list[glob_ok].astype(int))
        fig.show()

    # 3. Per-wheel MAE
    if wheel_ok.any():
        fig = make_dark_figure('Per-Wheel SR MAE vs Lap', 'Lap', 'MAE SR per wheel')
        for w in wheels:
            add_lap_scatter(fig, lap_list[wheel_ok], mae_w[w][wheel_ok],
                            lap_list[wheel_ok], name=w,
                            color=WHEEL_COLORS[w], symbol=WHEEL_SYMBOLS[w])
        fig.update_xaxes(tickvals=lap_list[wheel_ok].astype(int))
        fig.show()

    # 4. Per-wheel Bias
    if wheel_ok.any():
        fig = make_dark_figure('Per-Wheel SR Bias vs Lap', 'Lap', 'Bias SR per wheel')
        for w in wheels:
            add_lap_scatter(fig, lap_list[wheel_ok], bias_w[w][wheel_ok],
                            lap_list[wheel_ok], name=w,
                            color=WHEEL_COLORS[w], symbol=WHEEL_SYMBOLS[w])
        add_zero_line(fig, lap_list[wheel_ok])
        fig.update_xaxes(tickvals=lap_list[wheel_ok].astype(int))
        fig.show()

    # 5. Time in target + overslip
    ok_pct = base_ok & np.isfinite(in_tgt_pct_glob)
    if ok_pct.any():
        fig = make_dark_figure('Time in SR Target vs Lap',
                               'Lap', '% of traction time')
        add_lap_scatter(fig, lap_list[ok_pct], in_tgt_pct_glob[ok_pct],
                        lap_list[ok_pct], name='In target', color='#73D973')
        add_lap_scatter(fig, lap_list[ok_pct], over_pct_glob[ok_pct],
                        lap_list[ok_pct], name='Overslip', color='#F27070',
                        symbol='square')
        add_lap_scatter(fig, lap_list[ok_pct], under_pct_glob[ok_pct],
                        lap_list[ok_pct], name='Underslip', color='#4DB3F2',
                        symbol='diamond')
        fig.update_xaxes(tickvals=lap_list[ok_pct].astype(int))
        fig.show()

    # 6. Traction efficiency
    if eff_ok.any():
        fig = make_dark_figure('Traction Efficiency vs Lap',
                               'Lap', 'Mean ax when SR in target [m/s²]')
        add_lap_scatter(fig, lap_list[eff_ok], ax_in_tgt_glob[eff_ok],
                        lap_list[eff_ok], color='#73D973')
        fig.update_xaxes(tickvals=lap_list[eff_ok].astype(int))
        fig.show()

    # 7. Time-trace: per-wheel SR error during traction phases
    fig = make_dark_figure('Per-Wheel SR Error vs Time (traction only)',
                           'Time [s]', 'SR error (SR − target)')
    t_trac = d['time'][traction_mask]
    for j, w in enumerate(wheels):
        fig.add_trace(go.Scatter(
            x=t_trac,
            y=sr[w][traction_mask] - SR_TARGET,
            mode='lines',
            name=w,
            line=dict(color=WHEEL_COLORS[w], width=1.0),
        ))
    fig.add_hline(y=0, line=dict(color='rgba(200,200,200,0.5)', dash='dash'))
    fig.add_hline(y=DELTA_SR,  line=dict(color='rgba(200,200,200,0.3)', dash='dot'),
                  annotation_text='+band')
    fig.add_hline(y=-DELTA_SR, line=dict(color='rgba(200,200,200,0.3)', dash='dot'),
                  annotation_text='−band')
    fig.show()


if __name__ == '__main__':
    main()
