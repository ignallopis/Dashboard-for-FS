"""tv.py
------
Torque Vectoring (TV) KPIs — yaw tracking and moment distribution quality.

KPIs (all computed during cornering: |ay| >= threshold AND |steering| >= threshold):
  1. Yaw rate error RMSE and bias per lap
  2. Mz error RMSE and bias per lap
  3. Feedback / Feedforward Mz ratio per lap

Requires lapcount.py to have been run first.

Usage:
    python tv.py
"""
from __future__ import annotations
import numpy as np
import polars as pl
import plotly.graph_objects as go

from utils import (
    make_dark_figure, add_lap_scatter, add_trend_line, add_zero_line,
    keep_min_duration_segments, exclude_lap0_and_last_lap,
    robust_dt, unique_laps,
)

CSV_PATH = 'data/run4_2025-08-24.csv'

# ── Cornering filter parameters ───────────────────────────────────────────────
AY_THRESHOLD        = 2.0   # [m/s²]
STEERING_THRESHOLD  = 0.05  # [rad]
MIN_SPEED           = 4.0   # [m/s]
MIN_CORNER_DURATION = 0.20  # [s]
MIN_CORNER_SAMPLES  = 50    # per lap

FF_MIN_FOR_RATIO    = 5.0   # [Nm] minimum FF magnitude to compute FB/FF ratio
EPS_RATIO           = 1e-6


def _load(columns: list[str]) -> dict[str, np.ndarray]:
    df = pl.read_csv(CSV_PATH, columns=columns)
    return {c: df[c].to_numpy().astype(float) for c in columns}


def _vx_signal() -> str:
    try:
        pl.read_csv(CSV_PATH, columns=['Est_vxCOG'], n_rows=1)
        return 'Est_vxCOG'
    except Exception:
        return 'VN_vx'


def _ay_signal() -> str:
    try:
        pl.read_csv(CSV_PATH, columns=['Filtering_VN_ay'], n_rows=1)
        return 'Filtering_VN_ay'
    except Exception:
        return 'VN_ay'


def _corner_mask(ay, steering, vx, dt) -> np.ndarray:
    raw = (np.abs(ay) >= AY_THRESHOLD) & \
          (np.abs(steering) >= STEERING_THRESHOLD) & \
          (np.abs(vx) >= MIN_SPEED)
    return keep_min_duration_segments(raw, MIN_CORNER_DURATION, dt)


def _per_lap_error(err, laps, laptime, corner_mask, lap_list):
    """Compute RMSE and bias of *err* in corners per lap."""
    n     = len(lap_list)
    rmse  = np.full(n, np.nan)
    bias  = np.full(n, np.nan)
    lt    = np.full(n, np.nan)
    nsamp = np.zeros(n, dtype=int)
    cover = np.full(n, np.nan)

    for i, lap in enumerate(lap_list):
        lm  = laps == lap
        lcm = lm & corner_mask
        nsamp[i] = int(lcm.sum())
        if lm.any():
            lt[i]    = laptime[lm].max()
            cover[i] = lcm.sum() / lm.sum()
        if nsamp[i] < MIN_CORNER_SAMPLES:
            continue
        e = err[lcm]
        rmse[i] = np.sqrt(np.nanmean(e ** 2))
        bias[i] = np.nanmean(e)

    return rmse, bias, lt, nsamp, cover


# ── 1. Yaw rate error ─────────────────────────────────────────────────────────

def yaw_rate_error() -> None:
    ay_col = _ay_signal()
    vx_col = _vx_signal()
    d = _load(['TimeStamp', 'laps', 'laptime',
               'TV_errorYawRate', 'Steering', ay_col, vx_col])
    d['time'] = d['TimeStamp'] - d['TimeStamp'][0]
    d['ay']   = d.pop(ay_col)
    d['vx']   = d.pop(vx_col)

    valid = np.all(np.stack([np.isfinite(v) for v in d.values()], axis=1), axis=1)
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)

    dt      = robust_dt(d['time'])
    cm      = _corner_mask(d['ay'], d['Steering'], d['vx'], dt)
    lap_list = unique_laps(d['laps'])

    rmse, bias, lt, nsamp, cover = _per_lap_error(
        d['TV_errorYawRate'], d['laps'], d['laptime'], cm, lap_list
    )

    ok = np.isfinite(rmse) & np.isfinite(lt) & (nsamp >= MIN_CORNER_SAMPLES)

    print('\n─── TV: Yaw Rate Error ───')
    print(f"{'Lap':>4}  {'LapTime[s]':>10}  {'RMSE':>8}  {'Bias':>8}  "
          f"{'CornerSamp':>11}  {'Coverage':>9}")
    for i in np.where(ok)[0]:
        print(f'{int(lap_list[i]):>4}  {lt[i]:>10.3f}  {rmse[i]:>8.4f}  '
              f'{bias[i]:>+8.4f}  {nsamp[i]:>11d}  {cover[i]:>9.3f}')

    # Run chart: RMSE per lap
    fig1 = make_dark_figure('Yaw Rate Tracking Error per Lap',
                            'Lap', 'Yaw rate error RMSE')
    add_lap_scatter(fig1, lap_list[ok], rmse[ok], lap_list[ok])
    fig1.update_xaxes(tickvals=lap_list[ok].astype(int))
    fig1.show()

    # Scatter: error vs ay (shows direction bias)
    valid_lap_mask = np.isin(d['laps'], lap_list[ok])
    scatter_mask   = cm & valid_lap_mask
    ay_s   = d['ay'][scatter_mask]
    err_s  = d['TV_errorYawRate'][scatter_mask]

    fig2 = make_dark_figure('Yaw Rate Error vs Lateral Acceleration',
                            'Lateral acceleration ay [m/s²]',
                            'Yaw rate error')
    fig2.add_trace(go.Scatter(
        x=ay_s, y=err_s, mode='markers',
        marker=dict(color='#4DB3F2', size=3, opacity=0.5),
        name='Samples',
    ))
    n_left  = int((ay_s > 0).sum())
    n_right = int((ay_s < 0).sum())
    fig2.add_annotation(
        x=0.02, y=0.98, xref='paper', yref='paper',
        text=f'Left turns: {n_left}<br>Right turns: {n_right}',
        showarrow=False, align='left',
        font=dict(color='#EBEBEB', size=10),
        bgcolor='rgba(20,20,23,0.8)',
    )
    add_zero_line(fig2, ay_s)
    fig2.show()


# ── 2. Mz error ───────────────────────────────────────────────────────────────

def mz_error() -> None:
    ay_col = _ay_signal()
    vx_col = _vx_signal()
    d = _load(['TimeStamp', 'laps', 'laptime',
               'TV_errorMz', 'Steering', ay_col, vx_col])
    d['time'] = d['TimeStamp'] - d['TimeStamp'][0]
    d['ay']   = d.pop(ay_col)
    d['vx']   = d.pop(vx_col)

    valid = np.all(np.stack([np.isfinite(v) for v in d.values()], axis=1), axis=1)
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)

    dt       = robust_dt(d['time'])
    cm       = _corner_mask(d['ay'], d['Steering'], d['vx'], dt)
    lap_list = unique_laps(d['laps'])

    rmse, bias, lt, nsamp, cover = _per_lap_error(
        d['TV_errorMz'], d['laps'], d['laptime'], cm, lap_list
    )

    ok = np.isfinite(rmse) & np.isfinite(lt) & (nsamp >= MIN_CORNER_SAMPLES)

    print('\n─── TV: Mz Error ───')
    print(f"{'Lap':>4}  {'LapTime[s]':>10}  {'RMSE[Nm]':>10}  {'Bias[Nm]':>10}  "
          f"{'CornerSamp':>11}")
    for i in np.where(ok)[0]:
        print(f'{int(lap_list[i]):>4}  {lt[i]:>10.3f}  {rmse[i]:>10.2f}  '
              f'{bias[i]:>+10.2f}  {nsamp[i]:>11d}')

    # Run chart
    fig1 = make_dark_figure('Mz Tracking Error per Lap',
                            'Lap', 'Mz error RMSE [Nm]')
    add_lap_scatter(fig1, lap_list[ok], rmse[ok], lap_list[ok])
    fig1.update_xaxes(tickvals=lap_list[ok].astype(int))
    fig1.show()

    # Scatter: Mz error vs ay
    valid_lap_mask = np.isin(d['laps'], lap_list[ok])
    scatter_mask   = cm & valid_lap_mask
    ay_s   = d['ay'][scatter_mask]
    err_s  = d['TV_errorMz'][scatter_mask]

    fig2 = make_dark_figure('Mz Error vs Lateral Acceleration',
                            'Lateral acceleration ay [m/s²]',
                            'Mz error [Nm]')
    fig2.add_trace(go.Scatter(
        x=ay_s, y=err_s, mode='markers',
        marker=dict(color='#4DB3F2', size=3, opacity=0.5),
        name='Samples',
    ))
    n_left  = int((ay_s > 0).sum())
    n_right = int((ay_s < 0).sum())
    fig2.add_annotation(
        x=0.02, y=0.98, xref='paper', yref='paper',
        text=f'Left turns: {n_left}<br>Right turns: {n_right}',
        showarrow=False, align='left',
        font=dict(color='#EBEBEB', size=10),
        bgcolor='rgba(20,20,23,0.8)',
    )
    add_zero_line(fig2, ay_s)
    fig2.show()


# ── 3. Feedback / Feedforward Mz ratio ───────────────────────────────────────

def ff_fb_ratio() -> None:
    """FB/FF ratio = mean|MzFB| / mean|MzFF|.

    A ratio close to 0 means the feedforward model is accurate (little correction
    needed). A high ratio means the feedback controller is compensating heavily,
    which indicates a poorly tuned feedforward.
    """
    ay_col = _ay_signal()
    vx_col = _vx_signal()
    d = _load(['TimeStamp', 'laps', 'laptime',
               'TV_feedForwardMz', 'TV_feedBackMz',
               'Steering', ay_col, vx_col])
    d['time'] = d['TimeStamp'] - d['TimeStamp'][0]
    d['ay']   = d.pop(ay_col)
    d['vx']   = d.pop(vx_col)

    valid = np.all(np.stack([np.isfinite(v) for v in d.values()], axis=1), axis=1)
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)

    dt       = robust_dt(d['time'])
    cm       = _corner_mask(d['ay'], d['Steering'], d['vx'], dt)
    lap_list = unique_laps(d['laps'])
    laps     = d['laps']
    laptime  = d['laptime']

    n          = len(lap_list)
    lt_val     = np.full(n, np.nan)
    ff_mean    = np.full(n, np.nan)
    fb_mean    = np.full(n, np.nan)
    ratio      = np.full(n, np.nan)
    fb_share   = np.full(n, np.nan)
    n_samps    = np.zeros(n, dtype=int)
    coverage   = np.full(n, np.nan)

    for i, lap in enumerate(lap_list):
        lm  = laps == lap
        lcm = lm & cm
        n_samps[i] = int(lcm.sum())
        if lm.any():
            lt_val[i]   = laptime[lm].max()
            coverage[i] = lcm.sum() / lm.sum()
        if n_samps[i] < MIN_CORNER_SAMPLES:
            continue

        ff_abs = np.abs(d['TV_feedForwardMz'][lcm])
        fb_abs = np.abs(d['TV_feedBackMz'][lcm])
        ff_mean[i] = np.nanmean(ff_abs)
        fb_mean[i] = np.nanmean(fb_abs)
        total = ff_mean[i] + fb_mean[i] + EPS_RATIO
        fb_share[i] = fb_mean[i] / total
        if ff_mean[i] > FF_MIN_FOR_RATIO:
            ratio[i] = fb_mean[i] / (ff_mean[i] + EPS_RATIO)

    ok = np.isfinite(ratio) & np.isfinite(lt_val) & \
         (n_samps >= MIN_CORNER_SAMPLES) & (ff_mean > FF_MIN_FOR_RATIO)

    print('\n─── TV: Feedback / Feedforward Mz Ratio ───')
    print(f"{'Lap':>4}  {'LapTime[s]':>10}  {'FF_mean[Nm]':>12}  "
          f"{'FB_mean[Nm]':>12}  {'FB/FF':>8}  {'FB_share':>10}")
    for i in np.where(ok)[0]:
        print(f'{int(lap_list[i]):>4}  {lt_val[i]:>10.3f}  '
              f'{ff_mean[i]:>12.2f}  {fb_mean[i]:>12.2f}  '
              f'{ratio[i]:>8.3f}  {fb_share[i]:>10.3f}')

    if not ok.any():
        print('No valid laps for FB/FF ratio.')
        return

    # Run chart: FB/FF ratio per lap
    fig1 = make_dark_figure('Feedback to Feedforward Ratio per Lap',
                            'Lap', 'FB / FF ratio')
    add_lap_scatter(fig1, lap_list[ok], ratio[ok], lap_list[ok])
    fig1.update_xaxes(tickvals=lap_list[ok].astype(int))
    fig1.show()

    # Scatter: FB/FF ratio vs lap time
    fig2 = make_dark_figure('Feedback to Feedforward Ratio vs Lap Time',
                            'Lap time [s]', 'FB / FF ratio')
    add_lap_scatter(fig2, lt_val[ok], ratio[ok], lap_list[ok])
    add_trend_line(fig2, lt_val[ok], ratio[ok])
    fig2.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    yaw_rate_error()
    mz_error()
    ff_fb_ratio()


if __name__ == '__main__':
    main()
