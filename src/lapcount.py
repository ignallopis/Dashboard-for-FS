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
import os
import pathlib
import tempfile
from typing import Any, Literal

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import uniform_filter1d

from utils import cols_to_numpy, read_telemetry_csv

# ── Default detection parameters ──────────────────────────────────────────────
_EARTH_RADIUS_M = 6_378_137.0
_LAPCOUNT_ALGO_VERSION = 4
_LAPCOUNT_VERSION_COL = "lapcount_version"
_LAPCOUNT_MIN_VEL_COL = "lapcount_min_vel_mps"
_LAPCOUNT_GATE_HALF_WIDTH_COL = "lapcount_gate_half_width_m"
_LAPCOUNT_GATE_TIME_COL = "lapcount_gate_time_s"
_LAPCOUNT_DETECTED_LAPS_COL = "lapcount_detected_laps"
_LAPCOUNT_GATE_LON0_COL = "lapcount_gate_lon0_deg"
_LAPCOUNT_GATE_LAT0_COL = "lapcount_gate_lat0_deg"
_LAPCOUNT_GATE_LON1_COL = "lapcount_gate_lon1_deg"
_LAPCOUNT_GATE_LAT1_COL = "lapcount_gate_lat1_deg"
_LAPCOUNT_MODE_COL = "lapcount_mode"

LapCountMode = Literal["circuit", "acceleration", "skidpad"]

AUTO_PARAMS: dict[str, Any] = {
    "sample_hz": 100,  # fallback sample rate if dt cannot be computed
    "min_vel": 10.0,  # [m/s] start gate uses first sample at racing pace
    "gate_half_width": 8.0,  # [m]  half-width of the finish line window
    "rearm_distance": 15.0,  # [m]  must move this far away before next crossing counts
    "min_lap_time": 8.0,  # [s]  reject crossings faster than this
    "max_lap_time": 200.0,  # [s]  reject crossings slower than this
    "dir_window": 10,  # samples used to estimate gate tangent direction
    "smooth_window": 5,  # moving-average window for GPS smoothing
}

ACCELERATION_PARAMS: dict[str, Any] = {
    **AUTO_PARAMS,
    "min_vel": 1.0,  # [m/s] ROS lapcount default
    "dist_max": 3.0,  # [m]   ROS finish-line distance tolerance
    "accel_distance": 75.0,  # [m]   Formula Student acceleration finish
}

SKIDPAD_PARAMS: dict[str, Any] = {
    **AUTO_PARAMS,
    "min_vel": 1.0,  # [m/s] ROS lapcount default
    "rearm_distance": 3.0,  # [m]   equivalent to ROS distMax near-line latch
    "dist_max": 3.0,  # [m]   ROS finish-line distance tolerance
    "min_lap_time": 1.0,  # [s]   real skidpad laps are ~4-7 s; allow some slack
    "max_lap_time": 15.0,  # [s]   anything longer is a transition between attempts
}

# Progressive fallbacks tried when the default params yield no laps.
# Start strict (racing pace + tight gate) and gradually loosen so the gate
# centre lands on the real start/finish line, not on paddock push-off.
_AUTO_FALLBACKS: tuple[dict[str, Any], ...] = (
    {"min_vel": 10.0, "gate_half_width": 8.0},
    {"min_vel": 8.0, "gate_half_width": 10.0},
    {"min_vel": 5.0, "gate_half_width": 15.0},
    {"min_vel": 3.0, "gate_half_width": 18.0},
    {"min_vel": 1.0, "gate_half_width": 20.0},
)


# ── GPS helpers ───────────────────────────────────────────────────────────────


def gps_to_local_xy(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert GPS (lat, lon) to local ENU (x, y) in metres relative to first sample."""
    x, y = gps_to_local_xy_from_origin(lat, lon, float(lat[0]), float(lon[0]))
    return x, y


def gps_to_local_xy_from_origin(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert GPS (lat, lon) to local ENU (x, y) in metres from a given origin."""
    x = np.deg2rad(lon_deg - lon0_deg) * _EARTH_RADIUS_M * np.cos(np.deg2rad(lat0_deg))
    y = np.deg2rad(lat_deg - lat0_deg) * _EARTH_RADIUS_M
    return x, y


def _haversine_distance_m(
    lat0_deg: float,
    lon0_deg: float,
    lat1_deg: float,
    lon1_deg: float,
) -> float:
    """Great-circle distance between two GPS points in metres."""
    lat0 = np.deg2rad(lat0_deg)
    lat1 = np.deg2rad(lat1_deg)
    dlat = lat1 - lat0
    dlon = np.deg2rad(lon1_deg - lon0_deg)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2.0) ** 2
    return float(_EARTH_RADIUS_M * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def _first_finite_value(df: pl.DataFrame, column: str) -> float | None:
    """Return the first finite numeric value from *column*, or None."""
    if column not in df.columns:
        return None
    values = cols_to_numpy(df, [column])[column]
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None
    return float(finite[0])


def _normalise_lapcount_mode(mode: str | None) -> LapCountMode:
    """Map dashboard/event labels to the lapcount strategy used for detection."""
    if mode is None:
        return "circuit"
    key = mode.strip().lower().replace("-", "_").replace(" ", "_")
    aliases: dict[str, LapCountMode] = {
        "auto": "circuit",
        "autox": "circuit",
        "auto_x": "circuit",
        "circuit": "circuit",
        "endurance": "circuit",
        "trackdrive": "circuit",
        "track_drive": "circuit",
        "accel": "acceleration",
        "acceleration": "acceleration",
        "skidpad": "skidpad",
        "skid_pad": "skidpad",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown lapcount mode {mode!r}. Expected one of: {valid}.") from exc


def _trajectory_tangent_at_index(
    xg: np.ndarray,
    yg: np.ndarray,
    idx: int,
    window: int,
) -> np.ndarray:
    """Return the local trajectory tangent near *idx*."""
    N = len(xg)
    i1 = max(0, int(idx) - int(window))
    i2 = min(N - 1, int(idx) + int(window))
    dx = float(xg[i2] - xg[i1])
    dy = float(yg[i2] - yg[i1])
    norm = float(np.hypot(dx, dy))
    if norm < 1e-6:
        raise ValueError("Cannot determine heading direction from GPS.")
    return np.array([dx, dy], dtype=float) / norm


def _gate_lonlat_from_local(
    prep: dict[str, Any],
    finish_x: float,
    finish_y: float,
    t_hat: np.ndarray,
    half_width_m: float,
) -> tuple[float, float, float, float]:
    """Return a local gate line as lon0, lat0, lon1, lat1."""
    gate_t = np.array([-half_width_m, half_width_m], dtype=float)
    gate_x = finish_x + gate_t * t_hat[0]
    gate_y = finish_y + gate_t * t_hat[1]
    gate_lat, gate_lon = local_xy_to_gps(
        gate_x,
        gate_y,
        prep["lat0_deg"],
        prep["lon0_deg"],
    )
    return (
        float(gate_lon[0]),
        float(gate_lat[0]),
        float(gate_lon[1]),
        float(gate_lat[1]),
    )


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
    required = ("TimeStamp", "VN_latitude", "VN_longitude")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {', '.join(missing)}")

    cols = cols_to_numpy(df, ["TimeStamp", "VN_latitude", "VN_longitude"])
    t_abs_full = cols["TimeStamp"]
    lat_full = cols["VN_latitude"]
    lon_full = cols["VN_longitude"]
    gps_valid = (
        np.isfinite(t_abs_full)
        & np.isfinite(lat_full)
        & np.isfinite(lon_full)
        & ((np.abs(lat_full) > 1e-9) | (np.abs(lon_full) > 1e-9))
    )
    valid_idx = np.where(gps_valid)[0]
    if len(valid_idx) < 2:
        raise ValueError("Not enough valid GPS samples for lap detection.")

    t_abs = t_abs_full[valid_idx]
    t = t_abs - t_abs[0]

    raw_dt = np.diff(t)
    valid_dt = raw_dt[(raw_dt > 0) & np.isfinite(raw_dt)]
    dt_fill = float(np.median(valid_dt)) if len(valid_dt) > 0 else 1.0 / AUTO_PARAMS["sample_hz"]
    dt_arr = np.concatenate([[0.0], np.where((raw_dt > 0) & np.isfinite(raw_dt), raw_dt, dt_fill)])

    lat = lat_full[valid_idx]
    lon = lon_full[valid_idx]
    x, y = gps_to_local_xy(lat, lon)

    sw = AUTO_PARAMS["smooth_window"]
    if sw > 1:
        xg = uniform_filter1d(np.where(np.isfinite(x), x, 0.0), size=sw)
        yg = uniform_filter1d(np.where(np.isfinite(y), y, 0.0), size=sw)
    else:
        xg, yg = x.copy(), y.copy()

    safe_dt = np.where(dt_arr[1:] > 0, dt_arr[1:], dt_fill)
    vx_gps = np.concatenate([[0.0], np.diff(xg) / safe_dt])
    vy_gps = np.concatenate([[0.0], np.diff(yg) / safe_dt])

    return {
        "N": len(df),
        "valid_idx": valid_idx,
        "t": t,
        "lat": lat,
        "lon": lon,
        "lat0_deg": float(lat[0]),
        "lon0_deg": float(lon[0]),
        "xg": xg,
        "yg": yg,
        "speed_gps": np.hypot(vx_gps, vy_gps),
    }


def _run_detection_attempts(
    xg: np.ndarray,
    yg: np.ndarray,
    speed_gps: np.ndarray,
    t: np.ndarray,
    params: dict | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float, np.ndarray, dict[str, Any]] | None
):
    """Try the configured parameter ladder and return the best gate metadata.

    With auto params we keep the fallback order as a strictness ladder, but we do
    not stop at the first non-zero result. Some runs produce a few false-acceptable
    crossings with the strict gate and the full lap set with the next fallback, so
    we select the attempt with the highest crossing count and keep the first one on
    ties.
    """
    attempts: tuple[dict[str, Any], ...]
    if params is None:
        attempts = _AUTO_FALLBACKS
    else:
        attempts = ({**AUTO_PARAMS, **params},)

    best_result: (
        tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float, np.ndarray, dict[str, Any]]
        | None
    ) = None
    best_crossings = -1
    for override in attempts:
        cand = {**AUTO_PARAMS, **override}
        try:
            result = detect_laps(xg, yg, speed_gps, t, cand)
        except ValueError:
            continue
        full_result = (*result, cand)
        n_crossings = int(len(result[0]))
        if params is not None:
            return full_result
        if n_crossings > best_crossings:
            best_result = full_result
            best_crossings = n_crossings

    return best_result


def lap_detection_gate_from_df(
    df: pl.DataFrame,
    params: dict | None = None,
) -> dict[str, Any] | None:
    """Return the detected lap gate in GPS coordinates for map overlays."""
    if params is None and _LAPCOUNT_VERSION_COL in df.columns:
        version = _first_finite_value(df, _LAPCOUNT_VERSION_COL)
        if version is not None and int(version) == _LAPCOUNT_ALGO_VERSION:
            lon0 = _first_finite_value(df, _LAPCOUNT_GATE_LON0_COL)
            lat0 = _first_finite_value(df, _LAPCOUNT_GATE_LAT0_COL)
            lon1 = _first_finite_value(df, _LAPCOUNT_GATE_LON1_COL)
            lat1 = _first_finite_value(df, _LAPCOUNT_GATE_LAT1_COL)
            gate_half_width_m = _first_finite_value(df, _LAPCOUNT_GATE_HALF_WIDTH_COL)
            gate_time_s = _first_finite_value(df, _LAPCOUNT_GATE_TIME_COL)
            min_vel_mps = _first_finite_value(df, _LAPCOUNT_MIN_VEL_COL)
            mode_s = (
                str(df[_LAPCOUNT_MODE_COL].drop_nulls()[0])
                if _LAPCOUNT_MODE_COL in df.columns and len(df[_LAPCOUNT_MODE_COL].drop_nulls()) > 0
                else "circuit"
            )
            if None not in (lon0, lat0, lon1, lat1):
                finish_lon = 0.5 * (float(lon0) + float(lon1))
                finish_lat = 0.5 * (float(lat0) + float(lat1))
                if gate_half_width_m is None:
                    gate_half_width_m = 0.5 * _haversine_distance_m(
                        float(lat0),
                        float(lon0),
                        float(lat1),
                        float(lon1),
                    )
                return {
                    "finish_lat": finish_lat,
                    "finish_lon": finish_lon,
                    "gate_lat": np.array([float(lat0), float(lat1)], dtype=float),
                    "gate_lon": np.array([float(lon0), float(lon1)], dtype=float),
                    "crossing_idx": np.array([], dtype=int),
                    "crossing_times_s": np.array([], dtype=float),
                    "lap_durations_s": np.array([], dtype=float),
                    "gate_idx": None,
                    "gate_time_s": gate_time_s,
                    "gate_half_width_m": float(gate_half_width_m),
                    "rearm_distance_m": float(AUTO_PARAMS["rearm_distance"]),
                    "min_vel_mps": min_vel_mps,
                    "lapcount_version": int(version),
                    "lapcount_mode": mode_s,
                }

    prep = _prepare_detection_inputs(df)
    if (
        params is None
        and _LAPCOUNT_MIN_VEL_COL in df.columns
        and _LAPCOUNT_GATE_HALF_WIDTH_COL in df.columns
    ):
        min_vel = _first_finite_value(df, _LAPCOUNT_MIN_VEL_COL)
        gate_half_width = _first_finite_value(df, _LAPCOUNT_GATE_HALF_WIDTH_COL)
        if min_vel is not None and gate_half_width is not None:
            params = {
                "min_vel": float(min_vel),
                "gate_half_width": float(gate_half_width),
            }
    result = _run_detection_attempts(
        prep["xg"],
        prep["yg"],
        prep["speed_gps"],
        prep["t"],
        params=params,
    )
    if result is None:
        return None

    (
        crossing_idx,
        crossing_times,
        lap_durations,
        gate_idx,
        finish_x,
        finish_y,
        t_hat,
        used_params,
    ) = result
    gate_span_m = float(used_params["gate_half_width"]) * 1.5
    gate_t = np.array([-gate_span_m, gate_span_m], dtype=float)
    gate_x = finish_x + gate_t * t_hat[0]
    gate_y = finish_y + gate_t * t_hat[1]
    gate_lat, gate_lon = local_xy_to_gps(
        gate_x,
        gate_y,
        prep["lat0_deg"],
        prep["lon0_deg"],
    )
    finish_lat, finish_lon = local_xy_to_gps(
        np.array([finish_x]),
        np.array([finish_y]),
        prep["lat0_deg"],
        prep["lon0_deg"],
    )

    return {
        "finish_lat": float(finish_lat[0]),
        "finish_lon": float(finish_lon[0]),
        "gate_lat": gate_lat,
        "gate_lon": gate_lon,
        "crossing_idx": crossing_idx,
        "crossing_times_s": crossing_times,
        "lap_durations_s": lap_durations,
        "gate_idx": int(gate_idx),
        "gate_time_s": float(prep["t"][gate_idx]),
        "gate_half_width_m": float(used_params["gate_half_width"]),
        "rearm_distance_m": float(used_params["rearm_distance"]),
        "min_vel_mps": float(used_params["min_vel"]),
        "lapcount_version": int(_LAPCOUNT_ALGO_VERSION),
        "lapcount_mode": "circuit",
    }


def lap_detection_gate_from_csv(
    csv_path: str | pathlib.Path,
    params: dict | None = None,
) -> dict[str, Any] | None:
    """Load a CSV and return its lap-detection gate for dashboard overlays."""
    return lap_detection_gate_from_df(read_telemetry_csv(str(csv_path)), params=params)


# ── Lap detection ─────────────────────────────────────────────────────────────


def detect_laps(
    xg: np.ndarray, yg: np.ndarray, speed: np.ndarray, t: np.ndarray, params: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float, np.ndarray]:
    """Detect lap crossings from smoothed GPS trajectory.

    Returns:
        crossing_idx   – sample indices of each valid crossing
        crossing_times – time [s] of each crossing
        lap_durations  – duration [s] of each completed lap
        gate_idx       – sample index used to anchor the start/finish gate
        finish_x, finish_y – gate centre in local XY
        t_hat          – unit tangent vector of the gate
    """
    N = len(xg)

    moving = np.where(speed > params["min_vel"])[0]
    if len(moving) == 0:
        raise ValueError("No movement found — check GPS or min_vel.")
    start_idx = int(moving[0])

    # Gate orientation: local tangent at the first moving point
    i1 = max(0, start_idx - params["dir_window"])
    i2 = min(N - 1, start_idx + params["dir_window"])
    dx = xg[i2] - xg[i1]
    dy = yg[i2] - yg[i1]
    norm = np.hypot(dx, dy)
    if norm < 1e-6:
        raise ValueError("Cannot determine finish gate direction from GPS.")
    t_hat = np.array([dx, dy]) / norm
    n_hat = np.array([-t_hat[1], t_hat[0]])

    finish_x = xg[start_idx]
    finish_y = yg[start_idx]

    # Signed distances to the gate line
    signed_dist = (xg - finish_x) * n_hat[0] + (yg - finish_y) * n_hat[1]
    along_dist = (xg - finish_x) * t_hat[0] + (yg - finish_y) * t_hat[1]

    crossing_idx: list[int] = []
    crossing_times: list[float] = []
    lap_durations: list[float] = []
    last_cross_time = float(t[start_idx])
    armed = False

    for k in range(start_idx + 1, N):
        dist_to_gate = np.hypot(xg[k] - finish_x, yg[k] - finish_y)

        if not armed:
            if dist_to_gate > params["rearm_distance"]:
                armed = True
            continue

        crossed = (signed_dist[k - 1] <= 0) and (signed_dist[k] > 0)
        in_gate = abs(along_dist[k]) <= params["gate_half_width"]
        lap_t_cand = t[k] - last_cross_time
        valid_time = params["min_lap_time"] <= lap_t_cand <= params["max_lap_time"]

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
        start_idx,
        finish_x,
        finish_y,
        t_hat,
    )


def build_lap_samples(
    gate_idx: int, crossing_idx: np.ndarray, lap_durations: np.ndarray, N: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-sample ``laps`` and ``laptime`` arrays from the detected gate.

    Samples before the gate anchor are labelled as lap 0. Complete laps start at
    the gate anchor (lap 1) and each subsequent crossing starts the next lap.
    Samples after the last detected crossing are left NaN because that trailing
    segment is an incomplete lap.
    """
    laps_s = np.full(N, np.nan)
    laptime_s = np.full(N, np.nan)

    if N <= 0:
        return laps_s, laptime_s

    gate_idx = int(np.clip(gate_idx, 0, N - 1))
    laps_s[:gate_idx] = 0

    if len(crossing_idx) == 0:
        laps_s[gate_idx:] = 0
        return laps_s, laptime_s

    start = gate_idx
    for lap_id, (cidx, ldur) in enumerate(zip(crossing_idx, lap_durations), start=1):
        end = int(np.clip(cidx, start, N))
        if end > start:
            laps_s[start:end] = float(lap_id)
            laptime_s[start:end] = float(ldur)
        start = end

    return laps_s, laptime_s


def build_manual_lap_samples(
    crossing_idx: np.ndarray,
    lap_durations: np.ndarray,
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-sample laps from a user-defined gate line.

    The gate exists independently of the telemetry start, so lap 0 spans from
    the file start until the first crossing. Full laps begin at the first
    accepted crossing and end at the next one. Pairs whose duration is NaN
    (e.g. transitions between two skidpad attempts) are skipped and their
    samples remain NaN; the surviving laps are numbered consecutively.
    """
    laps_s = np.full(N, np.nan)
    laptime_s = np.full(N, np.nan)
    if N <= 0:
        return laps_s, laptime_s
    if len(crossing_idx) == 0:
        laps_s[:] = 0
        return laps_s, laptime_s

    first_cross = int(np.clip(crossing_idx[0], 0, N))
    laps_s[:first_cross] = 0

    if len(crossing_idx) < 2 or len(lap_durations) == 0:
        return laps_s, laptime_s

    next_lap_id = 1
    for i in range(min(len(crossing_idx) - 1, len(lap_durations))):
        ldur = float(lap_durations[i])
        if not np.isfinite(ldur):
            continue
        start = int(np.clip(crossing_idx[i], 0, N))
        end = int(np.clip(crossing_idx[i + 1], start, N))
        if end > start:
            laps_s[start:end] = float(next_lap_id)
            laptime_s[start:end] = ldur
        next_lap_id += 1

    return laps_s, laptime_s


def _find_acceleration_launch(
    speed: np.ndarray,
    t: np.ndarray,
    params: dict,
) -> int:
    """Return the sample index where the 75 m sprint actually launches.

    Acceleration CSVs typically include paddock idling before the real sprint,
    and raw GPS-derived speed has enough noise during standstill to exceed
    ``min_vel`` momentarily. The first such crossing would lock the launch
    onto random GPS jitter and place the finish line in the wrong direction.
    Instead, find the first *sustained* motion segment: smoothed speed must
    stay above ``min_vel`` for at least ``launch_sustain_s`` seconds and the
    segment peak must exceed ``launch_peak_threshold`` so paddock pushes are
    rejected.
    """
    if len(speed) == 0:
        raise ValueError("No GPS samples available — cannot find launch.")

    if len(t) > 1:
        dt = float(np.median(np.diff(t)))
        if dt <= 0.0 or not np.isfinite(dt):
            dt = 1.0 / AUTO_PARAMS["sample_hz"]
    else:
        dt = 1.0 / AUTO_PARAMS["sample_hz"]

    default_smooth = max(5, int(round(0.5 / dt)))
    smooth_n = max(5, int(params.get("launch_smooth_samples", default_smooth)))
    sustain_s = float(params.get("launch_sustain_s", 2.0))
    peak_thresh = float(params.get("launch_peak_threshold", 10.0))
    min_samples = max(1, int(round(sustain_s / dt)))
    min_vel = float(params["min_vel"])

    sp_smooth = uniform_filter1d(speed.astype(float, copy=False), size=smooth_n)
    moving = sp_smooth > min_vel

    in_run = False
    seg_start = -1
    seg_peak = -1.0
    for i, is_moving in enumerate(moving):
        if is_moving:
            v = float(sp_smooth[i])
            if not in_run:
                in_run = True
                seg_start = i
                seg_peak = v
            elif v > seg_peak:
                seg_peak = v
            continue
        if in_run:
            seg_end = i - 1
            if (seg_end - seg_start) >= min_samples and seg_peak >= peak_thresh:
                return seg_start
            in_run = False
            seg_start = -1
            seg_peak = -1.0
    if in_run:
        seg_end = len(moving) - 1
        if (seg_end - seg_start) >= min_samples and seg_peak >= peak_thresh:
            return seg_start

    raise ValueError(
        "No sustained sprint found — smoothed GPS speed never stayed above "
        f"{min_vel:.2f} m/s for {sustain_s:.1f}s with a peak above "
        f"{peak_thresh:.1f} m/s."
    )


def detect_acceleration_run(
    xg: np.ndarray,
    yg: np.ndarray,
    speed: np.ndarray,
    t: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float, float, np.ndarray]:
    """Detect the 75 m Formula Student acceleration run.

    Launch is the start of the first sustained sprint (see
    ``_find_acceleration_launch``). The finish is the sample where the
    integrated path length from launch first reaches ``accel_distance``;
    path-length is used instead of a fixed straight-line finish so that minor
    steering corrections do not displace the gate. The finish gate geometry
    returned is centred on the car position at the 75 m mark with its tangent
    perpendicular to the local heading there — useful for map overlays.
    """
    start_idx = _find_acceleration_launch(speed, t, params)

    seg_lengths = np.hypot(np.diff(xg[start_idx:]), np.diff(yg[start_idx:]))
    cumpath = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    accel_distance = float(params["accel_distance"])
    if cumpath[-1] < accel_distance:
        raise ValueError(
            f"Run too short — path length from launch only reaches "
            f"{cumpath[-1]:.1f} m, need {accel_distance:.1f} m."
        )
    rel_idx = int(np.searchsorted(cumpath, accel_distance))
    rel_idx = int(np.clip(rel_idx, 1, len(cumpath) - 1))
    finish_idx = start_idx + rel_idx

    finish_x = float(xg[finish_idx])
    finish_y = float(yg[finish_idx])
    win = int(params["dir_window"])
    i1 = max(start_idx, finish_idx - win)
    i2 = min(len(xg) - 1, finish_idx + win)
    h_dx = float(xg[i2] - xg[i1])
    h_dy = float(yg[i2] - yg[i1])
    h_norm = float(np.hypot(h_dx, h_dy))
    if h_norm < 1e-3:
        h_dx = finish_x - float(xg[start_idx])
        h_dy = finish_y - float(yg[start_idx])
        h_norm = float(np.hypot(h_dx, h_dy))
        if h_norm < 1e-3:
            raise ValueError("Cannot determine heading at acceleration finish.")
    heading_hat = np.array([h_dx, h_dy], dtype=float) / h_norm
    t_hat = np.array([-heading_hat[1], heading_hat[0]], dtype=float)

    duration_s = float(t[finish_idx] - t[start_idx])
    if duration_s <= 0.0 or not np.isfinite(duration_s):
        raise ValueError("Acceleration run has invalid duration.")

    return (
        np.array([finish_idx], dtype=int),
        np.array([float(t[finish_idx])], dtype=float),
        np.array([duration_s], dtype=float),
        start_idx,
        finish_idx,
        finish_x,
        finish_y,
        t_hat,
    )


def build_acceleration_samples(
    start_idx: int,
    finish_idx: int,
    duration_s: float,
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one timed segment for acceleration: lap 1 is start to 75 m."""
    laps_s = np.full(N, np.nan)
    laptime_s = np.full(N, np.nan)
    if N <= 0:
        return laps_s, laptime_s

    start = int(np.clip(start_idx, 0, N - 1))
    end = int(np.clip(finish_idx, start + 1, N))
    laps_s[:start] = 0.0
    laps_s[start:end] = 1.0
    laptime_s[start:end] = float(duration_s)
    return laps_s, laptime_s


def _manual_gate_geometry(
    prep: dict[str, Any],
    gate_line_lonlat: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float, np.ndarray, float]:
    """Convert a manual gate line in GPS to local XY geometry for one run."""
    gate_lon = np.array([gate_line_lonlat[0][0], gate_line_lonlat[1][0]], dtype=float)
    gate_lat = np.array([gate_line_lonlat[0][1], gate_line_lonlat[1][1]], dtype=float)
    if not np.all(np.isfinite(gate_lon)) or not np.all(np.isfinite(gate_lat)):
        raise ValueError("Manual gate line must contain finite GPS coordinates.")

    gate_x, gate_y = gps_to_local_xy_from_origin(
        gate_lat,
        gate_lon,
        prep["lat0_deg"],
        prep["lon0_deg"],
    )
    dx = float(gate_x[1] - gate_x[0])
    dy = float(gate_y[1] - gate_y[0])
    norm = float(np.hypot(dx, dy))
    if norm < 2.0:
        raise ValueError("Manual gate line is too short to define a finish line.")
    t_hat = np.array([dx, dy], dtype=float) / norm
    finish_x = float((gate_x[0] + gate_x[1]) * 0.5)
    finish_y = float((gate_y[0] + gate_y[1]) * 0.5)
    return finish_x, finish_y, t_hat, norm * 0.5


def detect_laps_from_manual_gate(
    xg: np.ndarray,
    yg: np.ndarray,
    t: np.ndarray,
    finish_x: float,
    finish_y: float,
    t_hat: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect laps from a user-defined finish line."""
    n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)
    signed_dist = (xg - finish_x) * n_hat[0] + (yg - finish_y) * n_hat[1]
    along_dist = (xg - finish_x) * t_hat[0] + (yg - finish_y) * t_hat[1]

    crossing_idx: list[int] = []
    crossing_times: list[float] = []
    lap_durations: list[float] = []
    last_cross_time: float | None = None
    armed = False

    for k in range(1, len(xg)):
        dist_to_gate = float(np.hypot(xg[k] - finish_x, yg[k] - finish_y))
        if not armed:
            if dist_to_gate > params["rearm_distance"]:
                armed = True
            continue

        crossed = (signed_dist[k - 1] <= 0.0) and (signed_dist[k] > 0.0)
        in_gate = abs(float(along_dist[k])) <= float(params["gate_half_width"])
        if not (crossed and in_gate):
            continue

        if last_cross_time is None:
            crossing_idx.append(k)
            crossing_times.append(float(t[k]))
            last_cross_time = float(t[k])
            armed = False
            continue

        lap_t_cand = float(t[k] - last_cross_time)
        valid_time = params["min_lap_time"] <= lap_t_cand <= params["max_lap_time"]
        if not valid_time:
            continue

        crossing_idx.append(k)
        crossing_times.append(float(t[k]))
        lap_durations.append(lap_t_cand)
        last_cross_time = float(t[k])
        armed = False

    return (
        np.array(crossing_idx, dtype=int),
        np.array(crossing_times, dtype=float),
        np.array(lap_durations, dtype=float),
    )


def detect_skidpad_laps_from_gate(
    xg: np.ndarray,
    yg: np.ndarray,
    t: np.ndarray,
    finish_x: float,
    finish_y: float,
    t_hat: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect skidpad crossings from a centre gate, matching ROS mission 1.

    ``lap_durations`` is kept aligned with ``crossing_idx`` (length =
    ``len(crossing_idx) - 1``) by writing ``np.nan`` for gaps that fall outside
    ``[min_lap_time, max_lap_time]``. ``build_manual_lap_samples`` interprets
    NaN as "skip this segment", so transitions between consecutive skidpad
    attempts are not persisted as fake laps.
    """
    n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)
    signed_dist = (xg - finish_x) * n_hat[0] + (yg - finish_y) * n_hat[1]

    crossing_idx: list[int] = []
    crossing_times: list[float] = []
    lap_durations: list[float] = []
    last_cross_time: float | None = None
    changed = True

    for k in range(1, len(xg)):
        crossed = (signed_dist[k - 1] < 0.0) and (signed_dist[k] > 0.0)
        near_centre = float(np.hypot(xg[k] - finish_x, yg[k] - finish_y)) < float(
            params["dist_max"]
        )

        if (not changed) and crossed and near_centre:
            crossing_idx.append(k)
            crossing_times.append(float(t[k]))
            changed = True

            if last_cross_time is not None:
                lap_t_cand = float(t[k] - last_cross_time)
                valid_time = params["min_lap_time"] <= lap_t_cand <= params["max_lap_time"]
                lap_durations.append(lap_t_cand if valid_time else np.nan)
            last_cross_time = float(t[k])
        else:
            changed = False

    return (
        np.array(crossing_idx, dtype=int),
        np.array(crossing_times, dtype=float),
        np.array(lap_durations, dtype=float),
    )


def _run_manual_gate_detection(
    prep: dict[str, Any],
    gate_line_lonlat: tuple[tuple[float, float], tuple[float, float]],
    params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, dict[str, Any]]:
    """Detect laps from a user-defined GPS gate line, trying both orientations."""
    base_params = {**AUTO_PARAMS, **(params or {})}
    finish_x, finish_y, t_hat, gate_half_width = _manual_gate_geometry(prep, gate_line_lonlat)
    gate_params = {**base_params, "gate_half_width": float(gate_half_width)}

    best_result: (
        tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, dict[str, Any]] | None
    ) = None
    best_laps = -1
    best_crossings = -1
    for cand_t_hat in (t_hat, -t_hat):
        crossing_idx, crossing_times, lap_durations = detect_laps_from_manual_gate(
            prep["xg"],
            prep["yg"],
            prep["t"],
            finish_x,
            finish_y,
            cand_t_hat,
            gate_params,
        )
        n_laps = int(len(lap_durations))
        n_crossings = int(len(crossing_idx))
        if (n_laps > best_laps) or (n_laps == best_laps and n_crossings > best_crossings):
            best_result = (
                crossing_idx,
                crossing_times,
                lap_durations,
                finish_x,
                finish_y,
                cand_t_hat,
                gate_params,
            )
            best_laps = n_laps
            best_crossings = n_crossings

    if best_result is None:
        raise ValueError("Manual gate line could not be evaluated.")
    return best_result


def _run_skidpad_detection(
    prep: dict[str, Any],
    gate_line_lonlat: tuple[tuple[float, float], tuple[float, float]],
    params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, dict[str, Any]]:
    """Detect skidpad laps from the user-provided centre line."""
    base_params = {**SKIDPAD_PARAMS, **(params or {})}
    finish_x, finish_y, t_hat, gate_half_width = _manual_gate_geometry(prep, gate_line_lonlat)
    gate_params = {**base_params, "gate_half_width": float(gate_half_width)}

    best_result: (
        tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, dict[str, Any]] | None
    ) = None
    best_laps = -1
    best_crossings = -1
    for cand_t_hat in (t_hat, -t_hat):
        crossing_idx, crossing_times, lap_durations = detect_skidpad_laps_from_gate(
            prep["xg"],
            prep["yg"],
            prep["t"],
            finish_x,
            finish_y,
            cand_t_hat,
            gate_params,
        )
        n_laps = int(np.sum(np.isfinite(lap_durations)))
        n_crossings = int(len(crossing_idx))
        if (n_laps > best_laps) or (n_laps == best_laps and n_crossings > best_crossings):
            best_result = (
                crossing_idx,
                crossing_times,
                lap_durations,
                finish_x,
                finish_y,
                cand_t_hat,
                gate_params,
            )
            best_laps = n_laps
            best_crossings = n_crossings

    if best_result is None:
        raise ValueError("Skidpad gate line could not be evaluated.")
    return best_result


def _skidpad_window_xy(
    df: pl.DataFrame, prep: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pick the local-XY samples that look like the skidpad-active portion.

    Filters by sustained |ay| (the figure-8 corners) when ay is logged, and
    falls back to "moving" samples otherwise. Returns None when there is not
    enough data to estimate a gate.
    """
    speed = prep["speed_gps"]
    if "Filtering_VN_ay" in df.columns:
        ay = np.abs(df["Filtering_VN_ay"].to_numpy().astype(float)[prep["valid_idx"]])
        ay_smooth = uniform_filter1d(np.where(np.isfinite(ay), ay, 0.0), size=200)
        win = ay_smooth > 3.0
        if int(win.sum()) < 500:
            win = ay_smooth > 2.0
    else:
        win = speed > 5.0
    if int(win.sum()) < 300:
        return None
    return prep["xg"][win], prep["yg"][win]


def estimate_skidpad_gate_from_gps(
    df: pl.DataFrame,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Estimate a skidpad centre-gate line from GPS via a local search.

    Strategy: anchor on the mean of the high-|ay| samples, derive a starting
    orientation from PCA (the gate is perpendicular to the long axis of the
    figure-8), then sweep a small grid of positions and orientations and pick
    the gate whose detected lap_durations are most numerous and consistent.
    Returns ``((lon0, lat0), (lon1, lat1))`` or ``None`` when no plausible
    gate could be found.
    """
    try:
        prep = _prepare_detection_inputs(df)
    except Exception:
        return None
    win = _skidpad_window_xy(df, prep)
    if win is None:
        return None
    x_in, y_in = win
    midx0, midy0 = float(np.mean(x_in)), float(np.mean(y_in))
    cov = np.cov(x_in - midx0, y_in - midy0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, int(np.argmax(eigvals))]
    base_angle = float(np.arctan2(-pc1[0], pc1[1]))  # perpendicular to PC1

    pos_offsets = np.linspace(-6.0, 6.0, 5)
    angle_offsets = np.deg2rad([-20.0, -10.0, 0.0, 10.0, 20.0])
    half_m = 5.0

    best_score = (-1, -1.0)
    best_gate: tuple[tuple[float, float], tuple[float, float]] | None = None
    for dx in pos_offsets:
        for dy in pos_offsets:
            cx, cy = midx0 + dx, midy0 + dy
            for ang_off in angle_offsets:
                theta = base_angle + ang_off
                t_hat = np.array([np.cos(theta), np.sin(theta)], dtype=float)
                gx = np.array([cx + half_m * t_hat[0], cx - half_m * t_hat[0]])
                gy = np.array([cy + half_m * t_hat[1], cy - half_m * t_hat[1]])
                gate_lat, gate_lon = local_xy_to_gps(
                    gx,
                    gy,
                    prep["lat0_deg"],
                    prep["lon0_deg"],
                )
                gate = (
                    (float(gate_lon[0]), float(gate_lat[0])),
                    (float(gate_lon[1]), float(gate_lat[1])),
                )
                try:
                    _, _, lap_durations, *_ = _run_skidpad_detection(prep, gate)
                except Exception:
                    continue
                durs = np.asarray(lap_durations, dtype=float)
                plausible = durs[(durs >= 3.0) & (durs <= 15.0)]
                n = int(len(plausible))
                if n >= 2:
                    cv = float(np.std(plausible) / max(np.mean(plausible), 1e-3))
                    consistency = 1.0 / (1.0 + cv)
                else:
                    consistency = 0.0
                score = (n, consistency)
                if score > best_score:
                    best_score = score
                    best_gate = gate

    if best_score[0] <= 0:
        return None
    return best_gate


def _write_csv_atomic(df: pl.DataFrame, csv_path: str | pathlib.Path) -> None:
    """Write *df* to *csv_path* atomically to avoid partial CSV corruption."""
    path = pathlib.Path(csv_path)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp_path = pathlib.Path(tmp_name)
    try:
        df.write_csv(str(tmp_path))
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


# ── Public API for auto-detection ─────────────────────────────────────────────


def csv_needs_lap_detection(csv_path: str | pathlib.Path) -> bool:
    """Return True if the CSV lacks detected laps or uses a stale detector."""
    path = str(csv_path)
    try:
        head = read_telemetry_csv(path, n_rows=1)
    except Exception:
        return True
    required_cols = {"laps", "laptime", _LAPCOUNT_VERSION_COL}
    if not required_cols.issubset(head.columns):
        return True
    try:
        cols = read_telemetry_csv(path, columns=["laps", _LAPCOUNT_VERSION_COL])
    except Exception:
        return True
    laps = cols["laps"]
    max_lap = laps.max()
    if max_lap is None:
        return True
    try:
        if float(max_lap) <= 0.0:
            return True
    except (TypeError, ValueError):
        return True
    version_s = cols[_LAPCOUNT_VERSION_COL].drop_nulls()
    if len(version_s) == 0:
        return True
    try:
        version = int(float(version_s[0]))
    except (TypeError, ValueError):
        return True
    return version != _LAPCOUNT_ALGO_VERSION


def detect_and_write_laps(
    csv_path: str | pathlib.Path,
    params: dict | None = None,
    gate_line_lonlat: tuple[tuple[float, float], tuple[float, float]] | None = None,
    mode: str | None = None,
) -> int:
    """Auto-detect laps and overwrite the CSV with ``laps`` / ``laptime`` columns.

    Returns the number of detected laps (0 if detection yielded nothing).
    """
    path = str(csv_path)
    lap_mode = _normalise_lapcount_mode(mode)

    df = read_telemetry_csv(path)
    prep = _prepare_detection_inputs(df)

    crossing_idx_valid = np.array([], dtype=int)
    lap_durations = np.array([], dtype=float)
    gate_idx_valid = 0
    used_params = {**AUTO_PARAMS}
    gate_time_s = np.nan
    gate_lon0 = np.nan
    gate_lat0 = np.nan
    gate_lon1 = np.nan
    gate_lat1 = np.nan
    detected_laps = 0

    if lap_mode == "acceleration":
        accel_params = {**ACCELERATION_PARAMS, **(params or {})}
        (
            crossing_idx_valid,
            _crossing_times,
            lap_durations,
            start_idx_valid,
            finish_idx_valid,
            finish_x,
            finish_y,
            t_hat,
        ) = detect_acceleration_run(
            prep["xg"],
            prep["yg"],
            prep["speed_gps"],
            prep["t"],
            accel_params,
        )
        duration_s = float(lap_durations[0]) if len(lap_durations) > 0 else np.nan
        laps_valid, laptime_valid = build_acceleration_samples(
            start_idx_valid,
            finish_idx_valid,
            duration_s,
            len(prep["valid_idx"]),
        )
        used_params = accel_params
        gate_time_s = float(prep["t"][start_idx_valid])
        gate_lon0, gate_lat0, gate_lon1, gate_lat1 = _gate_lonlat_from_local(
            prep,
            finish_x,
            finish_y,
            t_hat,
            float(used_params["dist_max"]),
        )
        detected_laps = int(len(lap_durations))
    elif lap_mode == "skidpad":
        # Try the explicit (manual) gate first when supplied; if it yields no
        # plausible laps, fall back to GPS-based auto-estimation. This keeps
        # user-drawn gates authoritative while rescuing CSVs whose stored
        # gate is mis-positioned (e.g. copied from another test).
        chosen_gate = gate_line_lonlat
        skidpad_result = None
        if chosen_gate is not None:
            try:
                skidpad_result = _run_skidpad_detection(prep, chosen_gate, params=params)
            except Exception:
                skidpad_result = None
            if skidpad_result is not None and int(np.sum(np.isfinite(skidpad_result[2]))) < 2:
                skidpad_result = None
        if skidpad_result is None:
            auto_gate = estimate_skidpad_gate_from_gps(df)
            if auto_gate is None:
                if chosen_gate is None:
                    raise ValueError(
                        "Skidpad lap detection needs a centre-gate line and "
                        "GPS auto-estimation could not find one. Draw the "
                        "gate manually in the dashboard."
                    )
                # Stick with the user gate even if it produced 0 laps so the
                # CSV records what was attempted; downstream UI surfaces 0 laps.
                skidpad_result = _run_skidpad_detection(prep, chosen_gate, params=params)
            else:
                chosen_gate = auto_gate
                skidpad_result = _run_skidpad_detection(prep, chosen_gate, params=params)
        crossing_idx_valid, _, lap_durations, _fx, _fy, _t_hat, used_params = skidpad_result
        laps_valid, laptime_valid = build_manual_lap_samples(
            crossing_idx_valid,
            lap_durations,
            len(prep["valid_idx"]),
        )
        gate_lon0 = float(chosen_gate[0][0])
        gate_lat0 = float(chosen_gate[0][1])
        gate_lon1 = float(chosen_gate[1][0])
        gate_lat1 = float(chosen_gate[1][1])
        used_params = {**used_params, "min_vel": np.nan}
        detected_laps = int(np.sum(np.isfinite(lap_durations)))
    elif gate_line_lonlat is None:
        result = _run_detection_attempts(
            prep["xg"],
            prep["yg"],
            prep["speed_gps"],
            prep["t"],
            params=params,
        )
        if result is not None:
            (
                crossing_idx_valid,
                _crossing_times,
                lap_durations,
                gate_idx_valid,
                finish_x,
                finish_y,
                t_hat,
                used_params,
            ) = result
            gate_time_s = float(prep["t"][gate_idx_valid])
            gate_lon0, gate_lat0, gate_lon1, gate_lat1 = _gate_lonlat_from_local(
                prep,
                finish_x,
                finish_y,
                t_hat,
                float(used_params["gate_half_width"]),
            )
        laps_valid, laptime_valid = build_lap_samples(
            gate_idx_valid,
            crossing_idx_valid,
            lap_durations,
            len(prep["valid_idx"]),
        )
        detected_laps = int(len(lap_durations))
    else:
        manual_result = _run_manual_gate_detection(prep, gate_line_lonlat, params=params)
        crossing_idx_valid, _, lap_durations, _fx, _fy, _t_hat, used_params = manual_result
        laps_valid, laptime_valid = build_manual_lap_samples(
            crossing_idx_valid,
            lap_durations,
            len(prep["valid_idx"]),
        )
        gate_lon0 = float(gate_line_lonlat[0][0])
        gate_lat0 = float(gate_line_lonlat[0][1])
        gate_lon1 = float(gate_line_lonlat[1][0])
        gate_lat1 = float(gate_line_lonlat[1][1])
        used_params = {**used_params, "min_vel": np.nan}
        detected_laps = int(len(lap_durations))

    laps_arr = np.full(int(prep["N"]), np.nan)
    laptime_arr = np.full(int(prep["N"]), np.nan)
    laps_arr[prep["valid_idx"]] = laps_valid
    laptime_arr[prep["valid_idx"]] = laptime_valid

    df = df.with_columns(
        [
            pl.Series("laps", laps_arr),
            pl.Series("laptime", laptime_arr),
            pl.Series(
                _LAPCOUNT_VERSION_COL,
                np.full(int(prep["N"]), _LAPCOUNT_ALGO_VERSION, dtype=np.int32),
            ),
            pl.Series(
                _LAPCOUNT_MIN_VEL_COL, np.full(int(prep["N"]), float(used_params["min_vel"]))
            ),
            pl.Series(
                _LAPCOUNT_GATE_HALF_WIDTH_COL,
                np.full(int(prep["N"]), float(used_params["gate_half_width"])),
            ),
            pl.Series(_LAPCOUNT_GATE_TIME_COL, np.full(int(prep["N"]), gate_time_s)),
            pl.Series(
                _LAPCOUNT_DETECTED_LAPS_COL, np.full(int(prep["N"]), detected_laps, dtype=np.int32)
            ),
            pl.Series(_LAPCOUNT_GATE_LON0_COL, np.full(int(prep["N"]), gate_lon0)),
            pl.Series(_LAPCOUNT_GATE_LAT0_COL, np.full(int(prep["N"]), gate_lat0)),
            pl.Series(_LAPCOUNT_GATE_LON1_COL, np.full(int(prep["N"]), gate_lon1)),
            pl.Series(_LAPCOUNT_GATE_LAT1_COL, np.full(int(prep["N"]), gate_lat1)),
            pl.Series(_LAPCOUNT_MODE_COL, np.full(int(prep["N"]), lap_mode)),
        ]
    )
    _write_csv_atomic(df, path)
    return detected_laps


# ── CLI plot helper (manual debugging) ────────────────────────────────────────


def _plot_detection(csv_path: str) -> None:
    """Re-run detection and show a GPS+speed plot for debugging."""
    df = read_telemetry_csv(csv_path)
    prep = _prepare_detection_inputs(df)
    result = _run_detection_attempts(
        prep["xg"],
        prep["yg"],
        prep["speed_gps"],
        prep["t"],
        params=None,
    )
    if result is None:
        raise ValueError("No lap gate could be determined from the GPS trace.")

    (
        crossing_idx,
        crossing_times,
        lap_durations,
        gate_idx,
        finish_x,
        finish_y,
        t_hat,
        used_params,
    ) = result

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["GPS Track + Finish Gate", "GPS Speed"],
        vertical_spacing=0.12,
    )
    BG = "#141417"
    fig.add_trace(
        go.Scatter(
            x=prep["xg"],
            y=prep["yg"],
            mode="lines",
            line=dict(color="#4DB3F2", width=1),
            name="Track",
        ),
        row=1,
        col=1,
    )
    gate_t = np.linspace(
        -used_params["gate_half_width"] * 1.5, used_params["gate_half_width"] * 1.5, 100
    )
    fig.add_trace(
        go.Scatter(
            x=finish_x + gate_t * t_hat[0],
            y=finish_y + gate_t * t_hat[1],
            mode="lines",
            name="Gate",
            line=dict(color="red", dash="dash", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[finish_x],
            y=[finish_y],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Gate centre",
        ),
        row=1,
        col=1,
    )
    if len(crossing_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=prep["xg"][crossing_idx],
                y=prep["yg"][crossing_idx],
                mode="markers",
                marker=dict(color="yellow", size=8, symbol="circle-open", line=dict(width=2)),
                name="Crossings",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=prep["t"],
            y=prep["speed_gps"],
            mode="lines",
            line=dict(color="#EBEBEB", width=1.2),
            name="Speed",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=used_params["min_vel"],
        line=dict(color="red", dash="dash"),
        annotation_text="min_vel",
        row=2,
        col=1,
    )
    fig.add_vline(
        x=float(prep["t"][gate_idx]), line=dict(color="red", dash="dot", width=1), row=2, col=1
    )
    for ct in crossing_times:
        fig.add_vline(x=ct, line=dict(color="#4DB3F2", dash="dash", width=1), row=2, col=1)
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color="#EBEBEB"),
        title=dict(text=f"Lap Count — {csv_path}", font=dict(color="#EBEBEB")),
    )
    for ax in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
        fig.update_layout(
            **{ax: dict(gridcolor="rgba(128,128,128,0.2)", linecolor="#E5E5E5", color="#E5E5E5")}
        )
    fig.show()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Scan ``data/`` and auto-detect laps on every CSV that lacks them."""
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSV files found in {data_dir}/")
        return

    for path in csvs:
        if not csv_needs_lap_detection(path):
            print(f"{path.name:<40} ok (laps already detected)")
            continue
        try:
            n = detect_and_write_laps(path)
        except Exception as exc:
            print(f"{path.name:<40} FAIL — {exc}")
            continue
        tag = f"{n} laps" if n > 0 else "no laps detected"
        print(f"{path.name:<40} wrote {tag}")


if __name__ == "__main__":
    main()
