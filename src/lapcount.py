"""lapcount.py
-----------
Detects lap boundaries from GPS data and writes ``laps`` / ``laptime``
columns back to the CSV.

Two ways to use it:

- Import: ``csv_needs_lap_detection(path)`` and ``detect_and_write_laps(path)``
  from the dashboard, so any new CSV in ``data/`` is processed automatically.
- CLI:    ``python src/lapcount.py`` scans ``data/`` and auto-detects laps on
  every CSV that does not have them yet.
"""
from __future__ import annotations
import pathlib
from typing import Any

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import uniform_filter1d

# ── Default detection parameters ──────────────────────────────────────────────
_EARTH_RADIUS_M = 6_378_137.0

AUTO_PARAMS: dict[str, Any] = {
    'sample_hz':       100,       # fallback sample rate if dt cannot be computed
    'min_vel':         10.0,      # [m/s] start gate uses first sample at racing pace
    'gate_half_width': 8.0,       # [m]  half-width of the finish line window
    'rearm_distance':  15.0,      # [m]  must move this far away before next crossing counts
    'min_lap_time':    8.0,       # [s]  reject crossings faster than this
    'max_lap_time':    200.0,     # [s]  reject crossings slower than this
    'dir_window':      10,        # samples used to estimate gate tangent direction
    'smooth_window':   5,         # moving-average window for GPS smoothing
}

# Progressive fallbacks tried when the default params yield no laps.
# Start strict (racing pace + tight gate) and gradually loosen so the gate
# centre lands on the real start/finish line, not on paddock push-off.
_AUTO_FALLBACKS: tuple[dict[str, Any], ...] = (
    {'min_vel': 10.0, 'gate_half_width': 8.0},
    {'min_vel': 8.0,  'gate_half_width': 10.0},
    {'min_vel': 5.0,  'gate_half_width': 15.0},
    {'min_vel': 3.0,  'gate_half_width': 18.0},
    {'min_vel': 1.0,  'gate_half_width': 20.0},
)


# ── GPS helpers ───────────────────────────────────────────────────────────────

def gps_to_local_xy(lat: np.ndarray, lon: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Convert GPS (lat, lon) to local ENU (x, y) in metres relative to first sample."""
    lat0 = lat[0]
    lon0 = lon[0]
    x = np.deg2rad(lon - lon0) * _EARTH_RADIUS_M * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * _EARTH_RADIUS_M
    return x, y


def local_xy_to_gps(
    x_m: np.ndarray,
    y_m: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert local XY coordinates [m] back to GPS latitude/longitude."""
    cos_lat0 = np.cos(np.deg2rad(lat0_deg))
    lat = lat0_deg + np.rad2deg(y_m / _EARTH_RADIUS_M)
    lon = lon0_deg + np.rad2deg(x_m / (_EARTH_RADIUS_M * cos_lat0))
    return lat, lon


def _prepare_detection_inputs(df: pl.DataFrame) -> dict[str, Any]:
    """Build the filtered GPS trajectory and derived speed used by lap detection."""
    required = ('TimeStamp', 'VN_latitude', 'VN_longitude')
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f'missing required columns: {", ".join(missing)}')

    t_abs_full = df['TimeStamp'].to_numpy().astype(float)
    lat_full = df['VN_latitude'].to_numpy().astype(float)
    lon_full = df['VN_longitude'].to_numpy().astype(float)
    gps_valid = (
        np.isfinite(t_abs_full)
        & np.isfinite(lat_full)
        & np.isfinite(lon_full)
        & ((np.abs(lat_full) > 1e-9) | (np.abs(lon_full) > 1e-9))
    )
    valid_idx = np.where(gps_valid)[0]
    if len(valid_idx) < 2:
        raise ValueError('Not enough valid GPS samples for lap detection.')

    t_abs = t_abs_full[valid_idx]
    t = t_abs - t_abs[0]

    raw_dt = np.diff(t)
    valid_dt = raw_dt[(raw_dt > 0) & np.isfinite(raw_dt)]
    dt_fill = float(np.median(valid_dt)) if len(valid_dt) > 0 else 1.0 / AUTO_PARAMS['sample_hz']
    dt_arr = np.concatenate([[0.0], np.where((raw_dt > 0) & np.isfinite(raw_dt), raw_dt, dt_fill)])

    lat = lat_full[valid_idx]
    lon = lon_full[valid_idx]
    x, y = gps_to_local_xy(lat, lon)

    sw = AUTO_PARAMS['smooth_window']
    if sw > 1:
        xg = uniform_filter1d(np.where(np.isfinite(x), x, 0.0), size=sw)
        yg = uniform_filter1d(np.where(np.isfinite(y), y, 0.0), size=sw)
    else:
        xg, yg = x.copy(), y.copy()

    safe_dt = np.where(dt_arr[1:] > 0, dt_arr[1:], dt_fill)
    vx_gps = np.concatenate([[0.0], np.diff(xg) / safe_dt])
    vy_gps = np.concatenate([[0.0], np.diff(yg) / safe_dt])

    return {
        'N': len(df),
        'valid_idx': valid_idx,
        't': t,
        'lat': lat,
        'lon': lon,
        'lat0_deg': float(lat[0]),
        'lon0_deg': float(lon[0]),
        'xg': xg,
        'yg': yg,
        'speed_gps': np.hypot(vx_gps, vy_gps),
    }


def _run_detection_attempts(
    xg: np.ndarray,
    yg: np.ndarray,
    speed_gps: np.ndarray,
    t: np.ndarray,
    params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, dict[str, Any]] | None:
    """Try the configured parameter ladder and return the selected gate metadata."""
    attempts: tuple[dict[str, Any], ...]
    if params is None:
        attempts = _AUTO_FALLBACKS
    else:
        attempts = ({**AUTO_PARAMS, **params},)

    last_result: tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, dict[str, Any]] | None = None
    for override in attempts:
        cand = {**AUTO_PARAMS, **override}
        try:
            result = detect_laps(xg, yg, speed_gps, t, cand)
        except ValueError:
            continue
        last_result = (*result, cand)
        if len(result[0]) > 0 or params is not None:
            return last_result

    return last_result


def lap_detection_gate_from_df(
    df: pl.DataFrame,
    params: dict | None = None,
) -> dict[str, Any] | None:
    """Return the detected lap gate in GPS coordinates for map overlays."""
    prep = _prepare_detection_inputs(df)
    result = _run_detection_attempts(
        prep['xg'], prep['yg'], prep['speed_gps'], prep['t'], params=params,
    )
    if result is None:
        return None

    crossing_idx, crossing_times, lap_durations, finish_x, finish_y, t_hat, used_params = result
    gate_span_m = float(used_params['gate_half_width']) * 1.5
    gate_t = np.array([-gate_span_m, gate_span_m], dtype=float)
    gate_x = finish_x + gate_t * t_hat[0]
    gate_y = finish_y + gate_t * t_hat[1]
    gate_lat, gate_lon = local_xy_to_gps(
        gate_x, gate_y, prep['lat0_deg'], prep['lon0_deg'],
    )
    finish_lat, finish_lon = local_xy_to_gps(
        np.array([finish_x]), np.array([finish_y]),
        prep['lat0_deg'], prep['lon0_deg'],
    )

    return {
        'finish_lat': float(finish_lat[0]),
        'finish_lon': float(finish_lon[0]),
        'gate_lat': gate_lat,
        'gate_lon': gate_lon,
        'crossing_idx': crossing_idx,
        'crossing_times_s': crossing_times,
        'lap_durations_s': lap_durations,
        'gate_half_width_m': float(used_params['gate_half_width']),
        'rearm_distance_m': float(used_params['rearm_distance']),
        'min_vel_mps': float(used_params['min_vel']),
    }


def lap_detection_gate_from_csv(
    csv_path: str | pathlib.Path,
    params: dict | None = None,
) -> dict[str, Any] | None:
    """Load a CSV and return its lap-detection gate for dashboard overlays."""
    return lap_detection_gate_from_df(pl.read_csv(str(csv_path)), params=params)


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
        raise ValueError('No movement found — check GPS or min_vel.')
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


# ── Public API for auto-detection ─────────────────────────────────────────────

def csv_needs_lap_detection(csv_path: str | pathlib.Path) -> bool:
    """Return True if the CSV lacks detected laps (missing column or max <= 0)."""
    path = str(csv_path)
    try:
        head = pl.read_csv(path, n_rows=0)
    except Exception:
        return True
    if 'laps' not in head.columns:
        return True
    try:
        laps = pl.read_csv(path, columns=['laps'])['laps']
    except Exception:
        return True
    max_lap = laps.max()
    if max_lap is None:
        return True
    try:
        return float(max_lap) <= 0.0
    except (TypeError, ValueError):
        return True


def detect_and_write_laps(csv_path: str | pathlib.Path,
                          params: dict | None = None) -> int:
    """Auto-detect laps and overwrite the CSV with ``laps`` / ``laptime`` columns.

    Returns the number of detected laps (0 if detection yielded nothing).
    """
    path = str(csv_path)

    df = pl.read_csv(path)
    prep = _prepare_detection_inputs(df)
    result = _run_detection_attempts(
        prep['xg'], prep['yg'], prep['speed_gps'], prep['t'], params=params,
    )

    crossing_idx_valid = np.array([], dtype=int)
    lap_durations = np.array([])
    if result is not None:
        crossing_idx_valid, _, lap_durations, *_ = result

    laps_valid, laptime_valid = build_lap_samples(
        crossing_idx_valid, lap_durations, len(prep['valid_idx']),
    )
    laps_arr = np.full(int(prep['N']), np.nan)
    laptime_arr = np.full(int(prep['N']), np.nan)
    laps_arr[prep['valid_idx']] = laps_valid
    laptime_arr[prep['valid_idx']] = laptime_valid

    df = df.with_columns([
        pl.Series('laps',    laps_arr),
        pl.Series('laptime', laptime_arr),
    ])
    df.write_csv(path)
    return int(len(crossing_idx_valid))


# ── CLI plot helper (manual debugging) ────────────────────────────────────────

def _plot_detection(csv_path: str) -> None:
    """Re-run detection and show a GPS+speed plot for debugging."""
    df = pl.read_csv(csv_path)
    prep = _prepare_detection_inputs(df)
    result = _run_detection_attempts(
        prep['xg'], prep['yg'], prep['speed_gps'], prep['t'], params=None,
    )
    if result is None:
        raise ValueError('No lap gate could be determined from the GPS trace.')

    crossing_idx, crossing_times, lap_durations, finish_x, finish_y, t_hat, used_params = result

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['GPS Track + Finish Gate', 'GPS Speed'],
        vertical_spacing=0.12,
    )
    BG = '#141417'
    fig.add_trace(go.Scatter(x=prep['xg'], y=prep['yg'], mode='lines',
                             line=dict(color='#4DB3F2', width=1), name='Track'),
                  row=1, col=1)
    gate_t = np.linspace(-used_params['gate_half_width'] * 1.5,
                          used_params['gate_half_width'] * 1.5, 100)
    fig.add_trace(go.Scatter(
        x=finish_x + gate_t * t_hat[0],
        y=finish_y + gate_t * t_hat[1],
        mode='lines', name='Gate',
        line=dict(color='red', dash='dash', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[finish_x], y=[finish_y], mode='markers',
                             marker=dict(color='red', size=10), name='Gate centre'),
                  row=1, col=1)
    if len(crossing_idx) > 0:
        fig.add_trace(go.Scatter(
            x=prep['xg'][crossing_idx], y=prep['yg'][crossing_idx], mode='markers',
            marker=dict(color='yellow', size=8, symbol='circle-open', line=dict(width=2)),
            name='Crossings'), row=1, col=1)
    fig.add_trace(go.Scatter(x=prep['t'], y=prep['speed_gps'], mode='lines',
                             line=dict(color='#EBEBEB', width=1.2), name='Speed'),
                  row=2, col=1)
    fig.add_hline(y=used_params['min_vel'], line=dict(color='red', dash='dash'),
                  annotation_text='min_vel', row=2, col=1)
    for ct in crossing_times:
        fig.add_vline(x=ct, line=dict(color='#4DB3F2', dash='dash', width=1),
                      row=2, col=1)
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color='#EBEBEB'),
        title=dict(text=f'Lap Count — {csv_path}', font=dict(color='#EBEBEB')),
    )
    for ax in ['xaxis', 'yaxis', 'xaxis2', 'yaxis2']:
        fig.update_layout(**{ax: dict(gridcolor='rgba(128,128,128,0.2)',
                                      linecolor='#E5E5E5', color='#E5E5E5')})
    fig.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Scan ``data/`` and auto-detect laps on every CSV that lacks them."""
    data_dir = pathlib.Path(__file__).resolve().parent.parent / 'data'
    csvs = sorted(data_dir.glob('*.csv'))
    if not csvs:
        print(f'No CSV files found in {data_dir}/')
        return

    for path in csvs:
        if not csv_needs_lap_detection(path):
            print(f'{path.name:<40} ok (laps already detected)')
            continue
        try:
            n = detect_and_write_laps(path)
        except Exception as exc:
            print(f'{path.name:<40} FAIL — {exc}')
            continue
        tag = f'{n} laps' if n > 0 else 'no laps detected'
        print(f'{path.name:<40} wrote {tag}')


if __name__ == '__main__':
    main()
