"""Smoke + sanity test: Control > TV KPI redesign (multi-run overlay).

Run: PYTHONPATH=src:. ./.venv/bin/python tests/test_tv_control_redesign.py
"""

import plotly.graph_objects as go
import polars as pl

import src.tv as tv

CSVS = ("data/Cerpa_FSG.csv", "data/Martinez_FSG.csv")


def _load_all() -> dict[str, pl.DataFrame]:
    return {p.split("/")[-1].removesuffix(".csv"): pl.read_csv(p) for p in CSVS}


def _check_single_fig(fn, label: str, kpi_keys: tuple[str, ...]) -> None:
    dfs = _load_all()
    fig, kpis = fn(dfs)
    assert isinstance(fig, go.Figure), f"{label}: returned no Figure"
    assert len(fig.data) > 0, f"{label}: empty figure"
    runs = kpis.get("runs", {})
    assert len(runs) == len(dfs), f"{label}: expected {len(dfs)} runs, got {len(runs)}"
    for name, vals in runs.items():
        assert int(vals.get("corner_samples", 0)) > 0, f"{label}/{name}: no corner samples"
        for k in kpi_keys:
            assert k in vals, f"{label}/{name}: missing KPI {k}"
    print(f"{label:28s} OK  runs={list(runs)}")


def _check_balance() -> None:
    dfs = _load_all()
    figs, kpis = tv.tv_intended_balance_figs_kpis(dfs)
    assert isinstance(figs, list) and len(figs) == 2, "B1 expected 2 figures"
    assert all(isinstance(f, go.Figure) for f in figs), "B1 non-Figure"
    runs = kpis.get("runs", {})
    assert len(runs) == len(dfs), f"B1: expected {len(dfs)} runs"
    for name, vals in runs.items():
        for k in ("median_intended_balance", "peak_intended_balance", "balance_sign"):
            assert k in vals, f"B1/{name}: missing {k}"
        assert int(vals.get("corner_samples", 0)) > 0
    # the per-corner figure must overlay one grouped Bar trace per run
    bar_traces = [t for t in figs[1].data if isinstance(t, go.Bar)]
    assert len(bar_traces) == len(dfs), f"B1: expected {len(dfs)} bar traces, got {len(bar_traces)}"
    print(
        f"{'B1 intended balance':28s} OK  "
        + " ".join(f"{n}:med={v['median_intended_balance']:+.1f}%" for n, v in runs.items())
    )


if __name__ == "__main__":
    _check_single_fig(
        tv.tv_yaw_tracking_fig,
        "A1 yaw tracking",
        ("tracking_rmse", "tracking_slope", "tracking_r2"),
    )
    _check_single_fig(
        tv.tv_pi_loop_health_fig,
        "A2 PI loop health",
        ("effective_gain", "gain_r2", "ringing_rate"),
    )
    _check_single_fig(
        tv.tv_authority_utilisation_fig,
        "A4 authority/util",
        ("util_p95", "mz_tracking_rmse"),
    )
    _check_balance()
    _check_single_fig(
        tv.tv_sideslip_stability_fig,
        "B2 sideslip beta",
        ("beta_p95_deg", "beta_peak_g"),
    )
    print("ALL TV REDESIGN TESTS PASSED")
