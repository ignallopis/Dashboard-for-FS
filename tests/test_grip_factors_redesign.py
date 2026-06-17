"""Smoke + sanity test: Dynamics > Grip Factors redesign (unified filters).

Covers the rebuilt grip overview: unified phase filters (radius corner / brake /
corner-exit), the g-g diagram, combined-G track map, breakdown bar + radar,
single- and multi-category evolution, and grip utilisation. Verifies the shared
corner detector matches dynamics' wrapper.

Run: PYTHONPATH=src:. ./.venv/bin/python tests/test_grip_factors_redesign.py
"""

import glob

import numpy as np
import plotly.graph_objects as go
import polars as pl

import src.gripfactor as gf
import src.dynamics as dyn
import utils


def _discover_csvs() -> tuple[str, ...]:
    """Circuit/endurance CSVs in data/ (exclude acceleration/skidpad events).

    Works whatever the local naming is (Cerpa_FSG/Martinez_FSG or FSG_A/FSG_B).
    """
    cands = sorted(p for p in glob.glob("data/*.csv") if not p.endswith(("_acc.csv", "_skpd.csv")))
    assert cands, "no circuit CSVs found under data/"
    return tuple(cands[:2])


CSVS = _discover_csvs()
GRIP_CATS = ("Overall", "Cornering", "Braking", "Traction")


def _load(path: str) -> pl.DataFrame:
    return pl.read_csv(path)


def _check_shared_corner_detector(df: pl.DataFrame) -> None:
    d = utils.ensure_complete_laps_df(df)
    ax = d["Filtering_VN_ax"].to_numpy().astype(float)
    ay = d["Filtering_VN_ay"].to_numpy().astype(float)
    vx = d["VN_vx"].to_numpy().astype(float)
    dt = utils.robust_dt(d["TimeStamp"].to_numpy().astype(float))
    m_u, _ = utils.radius_corner_mask(vx, ay, dt)
    m_d, _ = dyn._radius_corner_mask(vx, ay, dt, radius_threshold_m=60.0)
    assert np.array_equal(m_u, m_d), "utils vs dynamics corner mask diverged"
    assert utils.MU_TIRE == dyn.MU_TIRE == 1.70, "tyre μ not single-sourced"
    print("shared corner detector + μ      OK")


def _check_grip_factor_kpis(path: str) -> None:
    df = _load(path)
    k = gf.grip_factor_kpis(df)
    assert not k["table"].is_empty(), f"{path}: empty grip-factor table"
    assert k["valid_laps"] > 0, f"{path}: no valid laps"
    for cat in GRIP_CATS:
        v = k["means"][cat]
        assert np.isfinite(v) and v > 0, f"{path}: bad {cat} mean {v}"
    # physical sanity: Overall ≥ Braking (combined ≥ single axis), Traction modest
    assert k["means"]["Overall"] >= k["means"]["Braking"] - 1e-6, "Overall < Braking?"
    assert 0.0 < k["means"]["Traction"] < k["means"]["Cornering"], "Traction not < Cornering"
    assert k["fastest_lap"] is not None, f"{path}: no fastest lap"
    print(
        f"grip_factor_kpis {path:18s} OK  means={ {c: round(k['means'][c], 2) for c in GRIP_CATS} }"
    )


def _check_utilisation(path: str) -> None:
    df = _load(path)
    u = gf.grip_utilization_kpis(df)
    assert not u.get("warnings"), f"{path}: util warnings {u.get('warnings')}"
    assert u["envelope_g"] > 0, f"{path}: bad envelope"
    assert 0 <= u["utilization_pct"] <= 100, "utilisation out of range"
    assert 0 <= u["time_at_limit_pct"] <= 100, "TAL out of range"
    for ph in ("Braking", "Cornering", "Traction"):
        v = u["phase_time_at_limit_pct"][ph]
        assert np.isfinite(v) and 0 <= v <= 100, f"{path}: bad phase TAL {ph}={v}"
    print(
        f"grip_utilization {path:18s} OK  env={u['envelope_g']:.2f} util={u['utilization_pct']:.0f}%"
    )


def _check_figures_single(path: str) -> None:
    df = _load(path)
    k = gf.grip_factor_kpis(df)
    fig, kpis = gf.gg_scatter_fig({path: df})
    assert isinstance(fig, go.Figure) and len(fig.data) > 0, "g-g empty"
    assert not kpis.get("warnings"), kpis.get("warnings")
    map_laps = [int(l) for l in k["table"]["Lap"].to_list()]
    mp = gf.combined_g_track_map_fig(df, map_laps)
    assert isinstance(mp, go.Figure) and len(mp.data) == 1, "map should be one trace"
    assert mp.data[0].marker.showscale, "combined-G map has no colourbar"
    assert "average" in (mp.layout.title.text or ""), "map title should note the lap average"
    # default (no laps arg) must also work and average over every lap
    assert len(gf.combined_g_track_map_fig(df).data) == 1, "default-laps map failed"
    assert isinstance(gf.grip_factor_bar_fig(k["means"]), go.Figure), "bar fail"
    assert len(gf.grip_factor_evolution_fig(k["table"]).data) > 0, "evolution empty"
    assert len(gf.grip_utilization_fig({path: df}).data) > 0, "util fig empty"
    print(f"single-run figures {path:16s} OK")


def _check_figures_multi() -> None:
    dfs = {p: _load(p) for p in CSVS}
    fig, kpis = gf.gg_scatter_fig(dfs)
    assert len(kpis["runs"]) == len(dfs), "g-g multi missing runs"
    tables = {p: gf.grip_factor_kpis(df)["table"] for p, df in dfs.items()}
    for cat in GRIP_CATS:
        fe = gf.grip_factor_evolution_multi_fig(tables, category=cat)
        assert len(fe.data) == len(dfs), f"multi evo [{cat}]: one line per run expected"
    assert len(gf.grip_factor_radar_fig(tables).data) == len(dfs), "radar runs"
    assert len(gf.grip_utilization_fig(dfs).data) == len(dfs), "util bars per run"
    print(f"multi-run figures               OK  runs={[p.split('/')[-1] for p in CSVS]}")


def main() -> None:
    df0 = _load(CSVS[0])
    _check_shared_corner_detector(df0)
    for p in CSVS:
        _check_grip_factor_kpis(p)
        _check_utilisation(p)
        _check_figures_single(p)
    _check_figures_multi()
    print("\nALL GRIP-FACTOR REDESIGN CHECKS PASSED")


if __name__ == "__main__":
    main()
