from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from utils import available_laps, cols_to_numpy, lap_dist_from_gps

from src.driver import _classify_phases


@dataclass(frozen=True)
class Sector:
    index: int
    kind: str
    s_start_m: float
    s_end_m: float
    turn_id: int | None


@dataclass(frozen=True)
class SectorLapStats:
    lap_id: int
    sector_index: int
    duration_s: float
    samples: int
    n_throttle: int
    n_braking: int
    n_coasting: int
    n_plausibility: int


_METRIC_SPECS: tuple[tuple[str, str, int], ...] = (
    ("LapTime [s]", "lap_time_s", 2),
    ("Throttle [%]", "throttle_pct", 1),
    ("Braking [%]", "braking_pct", 1),
    ("Coasting [%]", "coasting_pct", 1),
    ("Plausibility [%]", "plausibility_pct", 1),
)


def _is_valid_sector(start_m: float, end_m: float) -> bool:
    return np.isfinite(start_m) and np.isfinite(end_m) and end_m > start_m


def _analysis_lap_ids(df: pl.DataFrame) -> list[int]:
    lap_ids = available_laps(df).astype(int)
    if lap_ids.size == 0:
        return []
    max_lap = int(np.max(lap_ids))
    return [int(lap_id) for lap_id in lap_ids.tolist() if int(lap_id) < max_lap]


def _metric_nan_dict() -> dict[str, float]:
    return {
        "lap_time_s": np.nan,
        "throttle_pct": np.nan,
        "braking_pct": np.nan,
        "coasting_pct": np.nan,
        "plausibility_pct": np.nan,
    }


def _display_value(metrics: dict[str, float] | None, key: str, decimals: int) -> float:
    if metrics is None:
        return np.nan
    value = float(metrics.get(key, np.nan))
    if not np.isfinite(value):
        return np.nan
    return round(value, decimals)


def valid_analysis_laps(df: pl.DataFrame) -> list[int]:
    return _analysis_lap_ids(df)


def fastest_valid_lap(df: pl.DataFrame) -> int | None:
    lap_times = whole_lap_metrics_by_lap(df)
    best_lap: int | None = None
    best_time_s = np.inf
    for lap_id, metrics in lap_times.items():
        lap_time_s = float(metrics.get("lap_time_s", np.nan))
        if np.isfinite(lap_time_s) and lap_time_s < best_time_s:
            best_time_s = lap_time_s
            best_lap = int(lap_id)
    return best_lap


def whole_lap_metrics_by_lap(df: pl.DataFrame) -> dict[int, dict[str, float]]:
    metrics_by_lap: dict[int, dict[str, float]] = {}
    for lap_id in _analysis_lap_ids(df):
        metrics = whole_lap_metrics(df, lap_id)
        if metrics is not None:
            metrics_by_lap[int(lap_id)] = metrics
    return metrics_by_lap


def build_sectors(turns, lap_end_m: float) -> list[Sector]:
    ordered_turns = sorted(turns, key=lambda turn: float(turn.s_entry_m))
    sectors: list[Sector] = []
    sector_index = 0
    cursor_m = 0.0

    if not ordered_turns:
        if _is_valid_sector(0.0, lap_end_m):
            sectors.append(
                Sector(
                    index=sector_index,
                    kind="straight",
                    s_start_m=0.0,
                    s_end_m=float(lap_end_m),
                    turn_id=None,
                )
            )
        return sectors

    for turn in ordered_turns:
        s_entry_m = float(turn.s_entry_m)
        s_exit_m = float(turn.s_exit_m)
        if _is_valid_sector(cursor_m, s_entry_m):
            sectors.append(
                Sector(
                    index=sector_index,
                    kind="straight",
                    s_start_m=float(cursor_m),
                    s_end_m=s_entry_m,
                    turn_id=None,
                )
            )
            sector_index += 1
        if _is_valid_sector(s_entry_m, s_exit_m):
            sectors.append(
                Sector(
                    index=sector_index,
                    kind="corner",
                    s_start_m=s_entry_m,
                    s_end_m=s_exit_m,
                    turn_id=int(turn.turn_id),
                )
            )
            sector_index += 1
        cursor_m = max(cursor_m, s_exit_m)

    if _is_valid_sector(cursor_m, lap_end_m):
        sectors.append(
            Sector(
                index=sector_index,
                kind="straight",
                s_start_m=float(cursor_m),
                s_end_m=float(lap_end_m),
                turn_id=None,
            )
        )

    return sectors


def lap_end_distance(df: pl.DataFrame, lap_id: int) -> float:
    if "laps" not in df.columns:
        return np.nan
    laps = df["laps"].to_numpy().astype(float)
    s_lap_m = lap_dist_from_gps(df)
    mask = (laps == float(lap_id)) & np.isfinite(s_lap_m)
    if not np.any(mask):
        return np.nan
    return float(np.nanmax(s_lap_m[mask]))


def per_lap_sector_stats(
    df: pl.DataFrame,
    sectors: list[Sector],
) -> dict[int, list[SectorLapStats]]:
    required_cols = {"laps", "laptime", "TimeStamp", "Throttle", "Brake"}
    if any(col not in df.columns for col in required_cols):
        return {}

    cols = cols_to_numpy(df, ["laps", "laptime", "TimeStamp", "Throttle", "Brake"])
    laps = cols["laps"]
    laptime = cols["laptime"]
    time_s = cols["TimeStamp"]
    throttle = cols["Throttle"]
    brake = cols["Brake"]
    s_lap_m = lap_dist_from_gps(df)

    out: dict[int, list[SectorLapStats]] = {}
    for lap_id in _analysis_lap_ids(df):
        lap_mask = laps == float(lap_id)
        if not np.any(lap_mask):
            continue
        if not np.any(np.isfinite(laptime[lap_mask])):
            continue
        lap_stats: list[SectorLapStats] = []
        for sector in sectors:
            sector_mask = (
                lap_mask
                & np.isfinite(s_lap_m)
                & np.isfinite(time_s)
                & np.isfinite(throttle)
                & np.isfinite(brake)
                & (s_lap_m >= float(sector.s_start_m))
                & (s_lap_m < float(sector.s_end_m))
            )
            if not np.any(sector_mask):
                lap_stats.append(
                    SectorLapStats(
                        lap_id=int(lap_id),
                        sector_index=int(sector.index),
                        duration_s=np.nan,
                        samples=0,
                        n_throttle=0,
                        n_braking=0,
                        n_coasting=0,
                        n_plausibility=0,
                    )
                )
                continue

            phase = _classify_phases(throttle[sector_mask], brake[sector_mask])
            duration_s = float(np.nanmax(time_s[sector_mask]) - np.nanmin(time_s[sector_mask]))
            lap_stats.append(
                SectorLapStats(
                    lap_id=int(lap_id),
                    sector_index=int(sector.index),
                    duration_s=duration_s,
                    samples=int(sector_mask.sum()),
                    n_throttle=int(np.sum(phase == "ACCELERATING")),
                    n_braking=int(np.sum(phase == "BRAKING")),
                    n_coasting=int(np.sum(phase == "COASTING")),
                    n_plausibility=int(np.sum(phase == "PLAUSIBILITY")),
                )
            )
        out[int(lap_id)] = lap_stats
    return out


def potential_lap(
    sectors: list[Sector],
    per_lap: dict[int, list[SectorLapStats]],
) -> dict[str, float]:
    _ = sectors
    winners: list[SectorLapStats] = []
    if not per_lap:
        return _metric_nan_dict()

    sector_indices = sorted(
        {int(stat.sector_index) for lap_stats in per_lap.values() for stat in lap_stats}
    )
    for sector_index in sector_indices:
        candidates = [
            stat
            for lap_stats in per_lap.values()
            for stat in lap_stats
            if int(stat.sector_index) == sector_index and np.isfinite(stat.duration_s)
        ]
        if not candidates:
            return _metric_nan_dict()
        winners.append(min(candidates, key=lambda stat: float(stat.duration_s)))

    total_duration_s = float(sum(float(stat.duration_s) for stat in winners))
    if not np.isfinite(total_duration_s) or total_duration_s <= 0.0:
        return _metric_nan_dict()

    throttle_weighted_s = 0.0
    braking_weighted_s = 0.0
    coasting_weighted_s = 0.0
    plausibility_weighted_s = 0.0
    for stat in winners:
        if stat.samples <= 0:
            return _metric_nan_dict()
        duration_s = float(stat.duration_s)
        throttle_weighted_s += duration_s * float(stat.n_throttle) / float(stat.samples)
        braking_weighted_s += duration_s * float(stat.n_braking) / float(stat.samples)
        coasting_weighted_s += duration_s * float(stat.n_coasting) / float(stat.samples)
        plausibility_weighted_s += duration_s * float(stat.n_plausibility) / float(stat.samples)

    return {
        "lap_time_s": total_duration_s,
        "throttle_pct": 100.0 * throttle_weighted_s / total_duration_s,
        "braking_pct": 100.0 * braking_weighted_s / total_duration_s,
        "coasting_pct": 100.0 * coasting_weighted_s / total_duration_s,
        "plausibility_pct": 100.0 * plausibility_weighted_s / total_duration_s,
    }


def whole_lap_metrics(df: pl.DataFrame, lap_id: int) -> dict[str, float] | None:
    required_cols = {"laps", "laptime", "Throttle", "Brake"}
    if any(col not in df.columns for col in required_cols):
        return None

    cols = cols_to_numpy(df, ["laps", "laptime", "Throttle", "Brake"])
    laps = cols["laps"]
    laptime = cols["laptime"]
    throttle = cols["Throttle"]
    brake = cols["Brake"]

    lap_mask = laps == float(lap_id)
    if not np.any(lap_mask):
        return None

    valid = lap_mask & np.isfinite(throttle) & np.isfinite(brake)
    if not np.any(valid):
        return None

    phase = _classify_phases(throttle[valid], brake[valid])
    samples = int(valid.sum())
    lap_time_s = (
        float(np.nanmax(laptime[lap_mask])) if np.any(np.isfinite(laptime[lap_mask])) else np.nan
    )
    return {
        "lap_time_s": lap_time_s,
        "throttle_pct": 100.0 * float(np.sum(phase == "ACCELERATING")) / float(samples),
        "braking_pct": 100.0 * float(np.sum(phase == "BRAKING")) / float(samples),
        "coasting_pct": 100.0 * float(np.sum(phase == "COASTING")) / float(samples),
        "plausibility_pct": 100.0 * float(np.sum(phase == "PLAUSIBILITY")) / float(samples),
    }


def csv_metrics_summary(
    df: pl.DataFrame, sectors: list[Sector]
) -> dict[str, dict[str, float] | None]:
    metrics_by_lap = whole_lap_metrics_by_lap(df)
    if not metrics_by_lap:
        return {"best": None, "avg": None, "potential": None}

    best_lap_id = min(
        metrics_by_lap,
        key=lambda lap_id: float(metrics_by_lap[lap_id].get("lap_time_s", np.inf)),
    )
    avg_metrics: dict[str, float] = {}
    for _label, key, _decimals in _METRIC_SPECS:
        values = np.array(
            [float(metrics[key]) for metrics in metrics_by_lap.values()],
            dtype=float,
        )
        avg_metrics[key] = float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan

    return {
        "best": metrics_by_lap[int(best_lap_id)],
        "avg": avg_metrics,
        "potential": potential_lap(sectors, per_lap_sector_stats(df, sectors)),
    }


def build_metrics_table(
    p1_label: str,
    p2_label: str | None,
    ref_metrics: dict[str, float] | None,
    cmp_metrics: dict[str, float] | None,
    p1_summary: dict[str, dict[str, float] | None] | None,
    p2_summary: dict[str, dict[str, float] | None] | None,
    ref_lap_id: int,
    cmp_lap_id: int,
) -> pl.DataFrame:
    _ = (p1_label, p2_label, ref_lap_id, cmp_lap_id)
    p1_summary = p1_summary or {"best": None, "avg": None, "potential": None}
    p2_summary = p2_summary or {"best": None, "avg": None, "potential": None}

    rows: list[dict[str, float | str]] = []
    for metric_label, metric_key, decimals in _METRIC_SPECS:
        row: dict[str, float | str] = {
            "Metric": metric_label,
            "Ref": _display_value(ref_metrics, metric_key, decimals),
            "Cmp": _display_value(cmp_metrics, metric_key, decimals),
            "Pot P1": _display_value(p1_summary.get("potential"), metric_key, decimals),
            "Best P1": _display_value(p1_summary.get("best"), metric_key, decimals),
            "Avg P1": _display_value(p1_summary.get("avg"), metric_key, decimals),
        }
        if p2_label is not None:
            row["Pot P2"] = _display_value(p2_summary.get("potential"), metric_key, decimals)
            row["Best P2"] = _display_value(p2_summary.get("best"), metric_key, decimals)
            row["Avg P2"] = _display_value(p2_summary.get("avg"), metric_key, decimals)
        rows.append(row)

    if p2_label is None:
        return pl.DataFrame(rows).select(["Metric", "Ref", "Cmp", "Pot P1", "Best P1", "Avg P1"])

    return pl.DataFrame(rows).select(
        ["Metric", "Ref", "Cmp", "Pot P1", "Pot P2", "Best P1", "Best P2", "Avg P1", "Avg P2"]
    )
