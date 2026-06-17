"""Regression tests for dashboard CSV labels.

Run: PYTHONPATH=src:. ./.venv/bin/python tests/test_dashboard_csv_labels.py
"""

import plotly.graph_objects as go

import src.dashboard as dash


def test_csv_selector_label_keeps_full_filename() -> None:
    assert dash._format_csv_file_option("Martinez_FSG.csv") == "Martinez_FSG.csv"
    assert dash._format_csv_file_option("Cerpa_FSG.csv") == "Cerpa_FSG.csv"


def test_single_trace_per_csv_legend_uses_full_filenames_without_row_split() -> None:
    dash.st.session_state["selected_csv_files"] = ["Cerpa_FSG.csv", "Martinez_FSG.csv"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Cerpa_FSG"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[1, 0], name="Martinez_FSG"))

    rows = dash._split_legend_by_csv_rows(fig)

    assert rows == 1
    assert [trace.name for trace in fig.data] == ["Cerpa_FSG.csv", "Martinez_FSG.csv"]
    assert fig.layout.legend.title.text is None
    assert getattr(fig.layout, "legend2", None) is None


def test_setup_overlay_merges_per_run_figures_into_one_plot() -> None:
    results = {}
    for run_name, offset in (("Cerpa_FSG.csv", 0), ("Martinez_FSG.csv", 1)):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[offset, offset + 1], name="Front samples"))
        results[run_name] = (fig, {"samples": 2})

    merged = dash._overlay_setup_single_figure(results)

    assert merged is not None
    assert len(merged.data) == 2
    assert {trace.legendgroup for trace in merged.data} == {"Cerpa_FSG.csv", "Martinez_FSG.csv"}
    assert [trace.name for trace in merged.data] == [
        "Cerpa_FSG.csv · Front samples",
        "Martinez_FSG.csv · Front samples",
    ]


if __name__ == "__main__":
    test_csv_selector_label_keeps_full_filename()
    test_single_trace_per_csv_legend_uses_full_filenames_without_row_split()
    test_setup_overlay_merges_per_run_figures_into_one_plot()
    print("ALL OK")
