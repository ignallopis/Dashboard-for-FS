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
    mfig, mk = drv.fastest_lap_speed_map_fig(dfs)
    assert mfig is not None and len(mk) == 2, mk
    for name, df in (_load(p) for p in CSVS):
        lap_id, laptime_s = drv._fastest_valid_lap(df)
        assert lap_id > 0
        cons = drv.lap_consistency_stats({name: df})
        best = float(cons["Best [s]"][0])
        assert abs(laptime_s - best) < 1e-3, f"{name}: fastest {laptime_s} != Best {best}"
    print("multi-run build  OK")


def test_per_lap_overview_table() -> None:
    dfs = {name: df for name, df in (_load(p) for p in CSVS)}
    tbl = drv.per_lap_overview_table(dfs)
    assert not tbl.is_empty()
    assert "Lap" in tbl.columns
    # one laptime + vmax + vavg column per run
    for name in dfs:
        label = drv._run_display_name(name)
        assert any(label in c and "laptime" in c for c in tbl.columns), tbl.columns
        assert any(label in c and "v_max" in c for c in tbl.columns), tbl.columns
        assert any(label in c and "v_avg" in c for c in tbl.columns), tbl.columns
    print(f"per_lap_overview_table  OK  cols={tbl.columns}")


def test_scorecard_deleted() -> None:
    attrs = (
        "driver_" + "scorecard",
        "_scorecard_" + "verdict",
        "_SCORECARD_" + "SPEC",
    )
    for attr in attrs:
        assert not hasattr(drv, attr), f"{attr} should be deleted"
    print("scorecard deleted  OK")


if __name__ == "__main__":
    test_run_speed_stats()
    test_run_phase_distribution_fig()
    test_multi_run_builds()
    test_per_lap_overview_table()
    test_scorecard_deleted()
    print("ALL OK")
