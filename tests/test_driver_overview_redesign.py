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


def test_run_phase_distribution_fig() -> None:
    for path in CSVS:
        name, df = _load(path)
        fig, kpis = drv.run_phase_distribution_fig({name: df})
        assert fig is not None, f"{name}: no figure"
        assert isinstance(kpis, dict)
        pct = kpis[name]["pct"]
        assert set(pct) == set(drv._PHASE_ORDER), pct
        total = sum(pct.values())
        assert abs(total - 100.0) < 0.5, f"{name}: phases sum to {total}, not ~100"
        assert kpis[name]["samples"] > 0
        print(f"{name:16s} run_phase_distribution  OK  {pct}")


def test_multi_run_builds() -> None:
    dfs = {name: df for name, df in (_load(p) for p in CSVS)}
    pfig, pk = drv.run_phase_distribution_fig(dfs)
    assert pfig is not None and len(pk) == 2, pk
    print("multi-run phase build  OK")


if __name__ == "__main__":
    test_run_speed_stats()
    test_run_phase_distribution_fig()
    test_multi_run_builds()
    print("ALL OK")
