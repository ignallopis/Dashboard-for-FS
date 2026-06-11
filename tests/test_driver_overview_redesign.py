"""Smoke + sanity test: Driver > Overview redesign.

Run: PYTHONPATH=src:. ./.venv/bin/python tests/test_driver_overview_redesign.py
"""

import numpy as np
import polars as pl

import src.driver as drv

CSVS = ("data/Cerpa_FSG.csv", "data/Martinez_FSG.csv")


def _load(path: str) -> tuple[str, pl.DataFrame]:
    name = path.split("/")[-1].removesuffix(".csv")
    return name, pl.read_csv(path)


def test_run_speed_stats() -> None:
    for path in CSVS:
        name, df = _load(path)
        stats = drv.run_speed_stats(df)
        assert set(stats) >= {"v_max_kmh", "v_avg_kmh", "samples"}, stats
        assert stats["samples"] > 0, f"{name}: no samples"
        # plausible FS-circuit band
        assert 0.0 < stats["v_avg_kmh"] <= stats["v_max_kmh"] <= 130.0, (
            f"{name}: implausible speeds {stats}"
        )
        print(f"{name:16s} run_speed_stats  OK  {stats}")


if __name__ == "__main__":
    test_run_speed_stats()
    print("ALL OK")
