"""lapcount.py
-----------
Detects lap boundaries from GPS data and writes ``laps`` / ``laptime``
columns back to the CSV.

Must be run ONCE before any other analysis script.

Usage:
    python lapcount.py
"""
from __future__ import annotations
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import uniform_filter1d

CSV_PATH = 'data/run4_2025-08-24.csv'

PARAMS = {
    'start_time_cut':  350.4618,  # [s]  ignore everything before this timestamp
    'sample_hz':       100,       # fallback sample rate if dt cannot be computed
    'min_vel':         1.0,       # [m/s] minimum speed to trust as "moving"
    'gate_half_width': 8.0,       # [m]  half-width of the finish line window
    'rearm_distance':  15.0,      # [m]  must move this far away before next crossing counts
    'min_lap_time':    8.0,       # [s]  reject crossings faster than this
    'max_lap_time':    200.0,     # [s]  reject crossings slower than this
    'dir_window':      10,        # samples used to estimate gate tangent direction
    'smooth_window':   5,         # moving-average window for GPS smoothing
    'plot_results':    True,
}


# ── GPS helpers ───────────────────────────────────────────────────────────────

def gps_to_local_xy(lat: np.ndarray, lon: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Convert GPS (lat, lon) to local ENU (x, y) in metres relative to first sample."""
    R    = 6_378_137.0
    lat0 = lat[0]
    lon0 = lon[0]
    x = np.deg2rad(lon - lon0) * R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * R
    return x, y


# ── Lap detection ─────────────────────────────────────────────────────────────

def detect_laps(xg: np.ndarray, yg: np.ndarray,
                speed: np.ndarray, t: np.ndarray,
                params: dict
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Detect lap crossings from smoothed GPS trajectory.

    Returns:
        crossing_idx   – sample indices of each valid crossing
        crossing_times – time [s] of each crossing
        lap_durations  – duration [s] of each completed lap
        finish_x, finish_y – gate centre in local XY
        t_hat          – unit tangent vector of the gate
    """
    N = len(xg)

    moving = np.where(speed > params['min_vel'])[0]
    if len(moving) == 0:
        raise ValueError('No movement found after time cut — check GPS or min_vel.')
    start_idx = int(moving[0])

    # Gate orientation: local tangent at the first moving point
    i1 = max(0, start_idx - params['dir_window'])
    i2 = min(N - 1, start_idx + params['dir_window'])
    dx = xg[i2] - xg[i1]
    dy = yg[i2] - yg[i1]
    norm = np.hypot(dx, dy)
    if norm < 1e-6:
        raise ValueError('Cannot determine finish gate direction from GPS.')
    t_hat = np.array([dx, dy]) / norm
    n_hat = np.array([-t_hat[1], t_hat[0]])

    finish_x = xg[start_idx]
    finish_y = yg[start_idx]

    # Signed distances to the gate line
    signed_dist = (xg - finish_x) * n_hat[0] + (yg - finish_y) * n_hat[1]
    along_dist  = (xg - finish_x) * t_hat[0] + (yg - finish_y) * t_hat[1]

    crossing_idx: list[int]   = []
    crossing_times: list[float] = []
    lap_durations: list[float]  = []
    last_cross_time = 0.0
    armed = False

    for k in range(start_idx + 1, N):
        dist_to_gate = np.hypot(xg[k] - finish_x, yg[k] - finish_y)

        if not armed:
            if dist_to_gate > params['rearm_distance']:
                armed = True
            continue

        crossed    = (signed_dist[k - 1] <= 0) and (signed_dist[k] > 0)
        in_gate    = abs(along_dist[k]) <= params['gate_half_width']
        lap_t_cand = t[k] - last_cross_time
        valid_time = params['min_lap_time'] <= lap_t_cand <= params['max_lap_time']

        if crossed and in_gate and valid_time:
            crossing_idx.append(k)
            crossing_times.append(t[k])
            lap_durations.append(lap_t_cand)
            last_cross_time = t[k]
            armed = False

    return (
        np.array(crossing_idx, dtype=int),
        np.array(crossing_times),
        np.array(lap_durations),
        finish_x, finish_y, t_hat,
    )


def build_lap_samples(crossing_idx: np.ndarray,
                      lap_durations: np.ndarray,
                      N: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Build per-sample ``laps`` and ``laptime`` arrays (NaN before first crossing)."""
    laps_s    = np.full(N, np.nan)
    laptime_s = np.full(N, np.nan)

    if len(crossing_idx) == 0:
        return laps_s, laptime_s

    laps_s[:crossing_idx[0]] = 0  # formation lap → 0

    for i, (cidx, ldur) in enumerate(zip(crossing_idx, lap_durations)):
        end = int(crossing_idx[i + 1]) - 1 if i < len(crossing_idx) - 1 else N - 1
        laps_s[cidx:end + 1]    = i + 1
        laptime_s[cidx:end + 1] = ldur

    return laps_s, laptime_s


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f'Reading {CSV_PATH}…')
    df = pl.read_csv(CSV_PATH)

    time_full = df['TimeStamp'].to_numpy().astype(float)
    keep_mask = time_full >= PARAMS['start_time_cut']

    if not keep_mask.any():
        raise ValueError(
            f"No samples with TimeStamp >= {PARAMS['start_time_cut']} s"
        )

    first_keep = int(np.where(keep_mask)[0][0])
    df_cut = df.filter(pl.Series(keep_mask))
    N      = len(df_cut)

    t_abs = df_cut['TimeStamp'].to_numpy().astype(float)
    t     = t_abs - t_abs[0]

    # Robust dt array (element-wise, for GPS speed computation)
    raw_dt   = np.diff(t)
    valid_dt = raw_dt[(raw_dt > 0) & np.isfinite(raw_dt)]
    dt_fill  = float(np.median(valid_dt)) if len(valid_dt) > 0 else 1.0 / PARAMS['sample_hz']
    dt_arr   = np.concatenate([[0.0],
                               np.where((raw_dt > 0) & np.isfinite(raw_dt),
                                        raw_dt, dt_fill)])

    # GPS → local XY
    lat = df_cut['VN_latitude'].to_numpy().astype(float)
    lon = df_cut['VN_longitude'].to_numpy().astype(float)
    x, y = gps_to_local_xy(lat, lon)

    sw = PARAMS['smooth_window']
    if sw > 1:
        xg = uniform_filter1d(np.where(np.isfinite(x), x, 0.0), size=sw)
        yg = uniform_filter1d(np.where(np.isfinite(y), y, 0.0), size=sw)
    else:
        xg, yg = x.copy(), y.copy()

    # Speed from GPS
    safe_dt   = np.where(dt_arr[1:] > 0, dt_arr[1:], dt_fill)
    vx_gps    = np.concatenate([[0.0], np.diff(xg) / safe_dt])
    vy_gps    = np.concatenate([[0.0], np.diff(yg) / safe_dt])
    speed_gps = np.hypot(vx_gps, vy_gps)

    # Detect laps
    crossing_idx, crossing_times, lap_durations, \
        finish_x, finish_y, t_hat = detect_laps(xg, yg, speed_gps, t, PARAMS)
    lap_count = len(crossing_idx)

    laps_cut, laptime_cut = build_lap_samples(crossing_idx, lap_durations, N)

    # Write back to full DataFrame
    laps_full    = np.full(len(df), np.nan)
    laptime_full = np.full(len(df), np.nan)
    laps_full[keep_mask]    = laps_cut
    laptime_full[keep_mask] = laptime_cut

    df = df.with_columns([
        pl.Series('laps',    laps_full),
        pl.Series('laptime', laptime_full),
    ])
    df.write_csv(CSV_PATH)

    # ── Print results ─────────────────────────────────────────────────────────
    if lap_count > 0:
        print(f"\n{'Lap':>4}  {'Lap Time [s]':>14}")
        print('─' * 20)
        for i, ldur in enumerate(lap_durations):
            print(f"{i + 1:>4}  {ldur:>14.3f}")
    else:
        print('No valid laps detected — check PARAMS (start_time_cut, gate params).')

    print(f'\nCSV updated : {CSV_PATH}')
    print(f'Time cut    : TimeStamp >= {PARAMS["start_time_cut"]} s  (index {first_keep})')
    print(f'Laps found  : {lap_count}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    if PARAMS['plot_results']:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['GPS Track + Finish Gate', 'GPS Speed'],
            vertical_spacing=0.12,
        )

        BG = '#141417'

        fig.add_trace(go.Scatter(x=xg, y=yg, mode='lines',
                                 line=dict(color='#4DB3F2', width=1), name='Track'),
                      row=1, col=1)

        gate_t   = np.linspace(-PARAMS['gate_half_width'] * 1.5,
                                PARAMS['gate_half_width'] * 1.5, 100)
        fig.add_trace(go.Scatter(
            x=finish_x + gate_t * t_hat[0],
            y=finish_y + gate_t * t_hat[1],
            mode='lines', name='Gate',
            line=dict(color='red', dash='dash', width=2)), row=1, col=1)

        fig.add_trace(go.Scatter(x=[finish_x], y=[finish_y], mode='markers',
                                 marker=dict(color='red', size=10), name='Gate centre'),
                      row=1, col=1)

        if lap_count > 0:
            fig.add_trace(go.Scatter(
                x=xg[crossing_idx], y=yg[crossing_idx], mode='markers',
                marker=dict(color='yellow', size=8, symbol='circle-open', line=dict(width=2)),
                name='Crossings'), row=1, col=1)

        fig.add_trace(go.Scatter(x=t, y=speed_gps, mode='lines',
                                 line=dict(color='#EBEBEB', width=1.2), name='Speed'),
                      row=2, col=1)

        fig.add_hline(y=PARAMS['min_vel'],
                      line=dict(color='red', dash='dash'),
                      annotation_text='min_vel', row=2, col=1)

        for ct in crossing_times:
            fig.add_vline(x=ct, line=dict(color='#4DB3F2', dash='dash', width=1),
                          row=2, col=1)

        fig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color='#EBEBEB'),
            title=dict(text='Lap Count — GPS Analysis', font=dict(color='#EBEBEB')),
        )
        for ax in ['xaxis', 'yaxis', 'xaxis2', 'yaxis2']:
            fig.update_layout(**{ax: dict(gridcolor='rgba(128,128,128,0.2)',
                                          linecolor='#E5E5E5', color='#E5E5E5')})
        fig.show()


if __name__ == '__main__':
    main()
