"""Smoke test: every Dynamics-Setup figure builds on real circuit data.

Run: PYTHONPATH=src:. ./.venv/bin/python tests/test_dynamics_setup_redesign.py
"""

import polars as pl

import src.dynamics as dyn

CSVS = ("data/Cerpa_FSG.csv", "data/Martinez_FSG.csv")


def main() -> None:
    # Removals (spec §Removals)
    assert not hasattr(dyn, "spring_velocity_histogram_figs"), "spring hist should be deleted"
    assert not hasattr(dyn, "aero_load_heave_fig"), "aero heave fig should be deleted"
    # Kept helper (used by braking/traction ideal curves)
    assert hasattr(dyn, "_aero_front_fraction")

    for path in CSVS:
        name = path.split("/")[-1].removesuffix(".csv")
        df = pl.read_csv(path)
        dfs = {name: df}

        # F1 static Fz: median-based, MASS_KG-derived design load
        fig, k = dyn.static_fz_reference_fig(df)
        assert fig is not None and k["samples"] > 0
        assert abs(k["design_corner_n"] - dyn.MASS_KG / 4.0 * dyn.G_MPS2) < 1e-6
        print(f"{name:14s} static_fz            OK samples={k['samples']}")

        # F3 LLTD per-lap
        fig, k = dyn.lltd_mid_corner_per_lap_fig(dfs)
        assert k["runs"], "no per-lap LLTD rows"
        print(f"{name:14s} lltd_per_lap         OK laps={k['runs'][name]['laps']}")

        # F4 LLTD vs |ay|
        fig, k = dyn.lateral_load_transfer_fig(df)
        assert k["samples"] > 0
        print(f"{name:14s} lltd_vs_ay           OK samples={k['samples']}")

        # F5 roll gradient: new per-axle samples KPIs
        fig, k = dyn.roll_gradient_fig(df)
        assert "front_samples" in k and "rear_samples" in k
        assert k["calibrated"] == dyn.DAMPER_CALIBRATED
        print(
            f"{name:14s} roll_gradient        OK samples F/R="
            f"{k['front_samples']}/{k['rear_samples']} calibrated={k['calibrated']}"
        )

        # F6 pitch gradient
        fig, k = dyn.pitch_gradient_fig(df)
        assert k["brake_samples"] > 0 and k["accel_samples"] > 0
        print(
            f"{name:14s} pitch_gradient       OK samples B/A="
            f"{k['brake_samples']}/{k['accel_samples']}"
        )

        # F8 damper histograms: rod velocity + samples KPI
        figs, k = dyn.damper_histogram_figs(df, phase="all")
        assert figs and "samples" in k
        assert all(k["samples"][w] > 0 for w in ("FL", "FR", "RL", "RR"))
        assert set(k["quad_share"]["FL"]) == {"HSR", "LSR", "LSB", "HSB"}
        print(f"{name:14s} damper_histograms    OK samples={k['samples']}")

    print("ALL OK")


if __name__ == "__main__":
    main()
