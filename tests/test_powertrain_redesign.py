"""Smoke test: every Powertrain figure builds on real circuit data.

Run: PYTHONPATH=src:. ./.venv/bin/python tests/test_powertrain_redesign.py
"""

import polars as pl

import src.powertrain as pt

FIG_FNS = [
    pt.energy_per_lap_fig,
    pt.power_per_wheel_fig,
    pt.inverter_limits_fig,
    pt.torque_fidelity_fig,
    pt.torque_speed_envelope_fig,
    pt.soc_per_lap_fig,
    pt.hv_delivery_efficiency_fig,
    pt.weakest_cell_fig,
    pt.thermal_evolution_fig,
    pt.thermal_headroom_fig,
]

CSVS = ("data/Cerpa_FSG.csv", "data/Martinez_FSG.csv")


def main() -> None:
    assert not hasattr(pt, "battery_status_fig"), "battery_status_fig should be deleted"
    assert not hasattr(pt, "endurance_projection"), "endurance_projection should be deleted"

    for path in CSVS:
        name = path.split("/")[-1].removesuffix(".csv")
        dfs = {name: pl.read_csv(path)}
        for fn in FIG_FNS:
            fig, kpis = fn(dfs)
            assert fig is not None, f"{path}: {fn.__name__} returned no figure"
            assert isinstance(kpis, dict), f"{path}: {fn.__name__} kpis not a dict"
            print(f"{path:24s} {fn.__name__:30s} OK  warnings={kpis.get('warnings', [])}")

        # glitch guard: battery peak must be physical after cleaning
        _f, tk = pt.thermal_evolution_fig(dfs)
        assert tk["peak_batt_tmax"] < 80.0, (
            f"{path}: battery Tmax still glitched ({tk['peak_batt_tmax']})"
        )
        # HV efficiency must be physical
        _f, hk = pt.hv_delivery_efficiency_fig(dfs)
        assert 0.7 < hk["delivery_eff_median"] < 1.0, (
            f"{path}: HV eff implausible ({hk['delivery_eff_median']})"
        )

    # two-run pooling must not crash
    dfs2 = {p.split("/")[-1].removesuffix(".csv"): pl.read_csv(p) for p in CSVS}
    for fn in FIG_FNS:
        fig, _kpis = fn(dfs2)
        assert fig is not None, f"multi-run: {fn.__name__}"
    print("ALL OK")


if __name__ == "__main__":
    main()
