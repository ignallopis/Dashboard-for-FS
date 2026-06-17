"""Smoke + synthetic test: Controls > RB controller-quality redesign.

Real logs (FSG_A/B) have DEAD RB channels, so the three figures must render gracefully
empty there. A synthetic DataFrame with LIVE RB channels exercises the math.

Run: PYTHONPATH=src:. ./.venv/bin/python tests/test_controls_rb_controller_quality.py
"""

import glob

import numpy as np
import plotly.graph_objects as go
import polars as pl

import src.rb as rb

FIGS = (rb.rb_current_fidelity_fig, rb.rb_regen_distribution_fig, rb.rb_saturation_fig)


def _real_logs() -> list[str]:
    return sorted(p for p in glob.glob("data/*.csv") if not p.endswith("_skpd.csv"))


def _check_graceful_empty() -> None:
    for path in _real_logs():
        dfs = {path: pl.read_csv(path, infer_schema_length=3000)}
        for fn in FIGS:
            fig, k = fn(dfs)
            assert isinstance(fig, go.Figure), f"{path} {fn.__name__}: not a Figure"
            assert k["runs"] == {}, f"{path} {fn.__name__}: expected empty runs (dead channels)"
        print(f"graceful-empty OK  {path}")


def _synthetic_live_df(n_per_lap: int = 400) -> pl.DataFrame:
    """4 laps so the complete-laps filter keeps laps 1–2. RB channels live during braking."""
    laps, laptime = [], []
    for lp in range(4):
        laps += [lp] * n_per_lap
        laptime += (
            [float("nan")] * n_per_lap if lp == 0 else list(np.linspace(10.0, 11.0, n_per_lap))
        )
    n = len(laps)
    rng = np.random.default_rng(0)
    cmd = np.full(n, 0.3)
    limiter = np.zeros(n)
    target = np.full(n, -20.0)
    cur = np.zeros(n)
    rbf = np.zeros(n)
    rbr = np.zeros(n)
    # braking bursts: ramp target to -100, track it, regen torque present, front-biased split
    for s in range(60, n - 80, 220):
        e = s + 60
        cmd[s:e] = -0.5
        limiter[s:e] = 1.0
        ramp = np.linspace(-40.0, -100.0, e - s)
        target[s:e] = ramp
        cur[s:e] = ramp + rng.normal(0.0, 2.0, e - s)  # achieved tracks target within ~2 A
        rbf[s:e] = 600.0  # front regen torque [Nm]
        rbr[s:e] = 400.0  # rear regen torque [Nm]
    return pl.DataFrame(
        {
            "laps": [float(x) for x in laps],
            "laptime": laptime,
            "LLC_Command": cmd,
            "Est_vxCOG": np.full(n, 15.0),
            "RB_enableIntensityLimiter": limiter,
            "RB_intensityTarget": target,
            "Current": cur,
            "RB_F_TotalTrq": rbf,
            "RB_R_TotalTrq": rbr,
            "Param_desiredMaximumRegenTorque": np.full(n, 905.0),
            "Est_FZFL": np.full(n, 900.0),
            "Est_FZFR": np.full(n, 900.0),  # front load share = 1800/3000 = 60%
            "Est_FZRL": np.full(n, 600.0),
            "Est_FZRR": np.full(n, 600.0),
        }
    )


def _check_synthetic() -> None:
    dfs = {"synthetic": _synthetic_live_df()}

    _, k1 = rb.rb_current_fidelity_fig(dfs)
    v1 = k1["runs"]["synthetic"]
    assert v1["samples"] > 0, "R1: no samples on synthetic live data"
    assert v1["track_error_med_a"] <= 5.0, (
        f"R1: tracking error too high ({v1['track_error_med_a']})"
    )
    assert 50.0 <= v1["within_band_pct"] <= 100.0, f"R1: within-band {v1['within_band_pct']}"
    print("R1 synthetic OK ", v1)

    _, k3 = rb.rb_regen_distribution_fig(dfs)
    v3 = k3["runs"]["synthetic"]
    assert v3["samples"] > 0, "R3: no samples"
    assert abs(v3["front_regen_share_median"] - 60.0) <= 2.0, f"R3: front regen share {v3}"
    assert abs(v3["front_load_share_median"] - 60.0) <= 2.0, f"R3: front load share {v3}"
    print("R3 synthetic OK ", v3)

    _, k4 = rb.rb_saturation_fig(dfs)
    v4 = k4["runs"]["synthetic"]
    assert v4["samples"] > 0, "R4: no samples"
    assert 0.0 <= v4["pct_saturated"] <= 100.0, f"R4: pct_saturated {v4}"
    assert v4["pct_current_limited"] > 0.0, f"R4: expected some current-limited time {v4}"
    print("R4 synthetic OK ", v4)


def main() -> None:
    _check_graceful_empty()
    _check_synthetic()
    print("\nALL RB CONTROLLER-QUALITY CHECKS PASSED")


if __name__ == "__main__":
    main()
