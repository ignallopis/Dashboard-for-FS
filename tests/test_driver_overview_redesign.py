"""Smoke + sanity test: Driver > Overview redesign.

Run: PYTHONPATH=src:. ./.venv/bin/python tests/test_driver_overview_redesign.py
"""

import polars as pl

import src.cornering as corn
import src.driver as drv
import src.lap_sectors as lsec
from utils import style_sector_times_table

CSVS = ("data/Cerpa_FSG.csv", "data/Martinez_FSG.csv")


def _load(path: str) -> tuple[str, pl.DataFrame]:
    name = path.split("/")[-1].removesuffix(".csv")
    return name, pl.read_csv(path)


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
    for _name, df in (_load(p) for p in CSVS):
        lap_id, _laptime_s = drv._fastest_valid_lap(df)
        assert lap_id > 0
    print("multi-run build  OK")


def test_sector_times_matrix() -> None:
    dfs = {name: df for name, df in (_load(p) for p in CSVS)}
    geo_run = min(
        dfs,
        key=lambda k: min(
            (m["lap_time_s"] for m in lsec.whole_lap_metrics_by_lap(dfs[k]).values()),
            default=float("inf"),
        ),
    )
    fast = lsec.fastest_valid_lap(dfs[geo_run])
    assert fast is not None
    d = corn.compute_radius_curvature(dfs[geo_run])
    turns = corn.detect_turns_on_lap(d, geo_run, int(fast))
    assert turns, "no corners detected on geometry lap"
    sectors = lsec.build_sectors(turns, lsec.lap_end_distance(dfs[geo_run], int(fast)))
    tbl = lsec.sector_times_matrix(dfs, sectors)
    assert not tbl.is_empty()
    assert {"Run", "Lap", "Lap time [s]"} <= set(tbl.columns)
    seg_cols = [c for c in tbl.columns if c not in ("Run", "Lap", "Lap time [s]")]
    assert len(seg_cols) == len(sectors)
    assert len(seg_cols) == len(lsec.sector_labels(sectors))
    # sum of segments ≈ lap time (sector edges quantised at 100 Hz)
    chk = tbl.with_columns(pl.sum_horizontal(seg_cols).alias("sum_s")).drop_nulls("Lap time [s]")
    diff = (chk["sum_s"] - chk["Lap time [s]"]).abs()
    assert float(diff.max()) < 1.5, f"segment sum drifts from lap time: {float(diff.max()):.2f}s"
    print(f"sector_times_matrix  OK  shape={tbl.shape} max|Σseg-lap|={float(diff.max()):.2f}s")


def test_sector_times_table_colours_ranked_values() -> None:
    tbl = pl.DataFrame(
        {
            "Lap": [1, 1, 1, 1],
            "Lap time [s]": [10.0, 11.0, 13.0, 15.0],
            "T1 [s]": [4.0, 5.0, 7.0, 9.0],
        }
    )

    html = style_sector_times_table(tbl).to_html().lower()

    assert "background-color: #8e5aa8" in html  # best
    assert "background-color: #2a9d8f" in html  # second best
    assert "background-color: #d9a441" in html  # middle of the ramp
    assert "background-color: #c75d5d" in html  # worst
    print("sector_times_table colour ranking  OK")


def test_sector_times_table_colours_are_independent_per_run() -> None:
    tbl = pl.DataFrame(
        {
            "Run": ["A", "A", "A", "B", "B", "B"],
            "Lap": [1, 2, 3, 1, 2, 3],
            "Lap time [s]": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
        }
    )

    styler = style_sector_times_table(tbl)._compute()
    ctx = styler.ctx
    lap_time_col = 2

    assert ("background-color", "#8E5AA8") in ctx[(0, lap_time_col)]
    assert ("background-color", "#C75D5D") in ctx[(2, lap_time_col)]
    assert ("background-color", "#8E5AA8") in ctx[(3, lap_time_col)]
    assert ("background-color", "#C75D5D") in ctx[(5, lap_time_col)]
    print("sector_times_table per-run colour ranking  OK")


def test_old_overview_tables_deleted() -> None:
    for attr in ("run_speed_stats", "per_lap_overview_table", "lap_consistency_stats"):
        assert not hasattr(drv, attr), f"{attr} should be deleted"
    print("old overview tables deleted  OK")


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
    test_run_phase_distribution_fig()
    test_multi_run_builds()
    test_sector_times_matrix()
    test_sector_times_table_colours_ranked_values()
    test_sector_times_table_colours_are_independent_per_run()
    test_old_overview_tables_deleted()
    test_scorecard_deleted()
    print("ALL OK")
