from __future__ import annotations

import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import plotly.graph_objects as go

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import src.dashboard as dash
import src.driver as drv
from utils import (
    PLOT_AXIS_TITLE_STANDOFF,
    PLOT_FONT_SIZE,
    PLOT_HOVER_FONT_SIZE,
    apply_dark_layout,
)


class DashboardReadabilityTest(unittest.TestCase):
    def test_plotly_theme_uses_document_readable_body_text(self) -> None:
        fig = apply_dark_layout(
            go.Figure(),
            title="Power Distribution per Wheel",
            xlabel="Distance [m]",
            ylabel="Speed [km/h]",
        )

        self.assertGreaterEqual(PLOT_FONT_SIZE, 22)
        self.assertGreaterEqual(PLOT_HOVER_FONT_SIZE, 20)
        self.assertGreaterEqual(fig.layout.font.size, PLOT_FONT_SIZE)
        self.assertGreaterEqual(fig.layout.legend.font.size, PLOT_FONT_SIZE)
        self.assertGreaterEqual(fig.layout.hoverlabel.font.size, PLOT_HOVER_FONT_SIZE)
        self.assertGreaterEqual(fig.layout.title.font.size, PLOT_FONT_SIZE)
        self.assertGreaterEqual(fig.layout.xaxis.title.font.size, PLOT_FONT_SIZE)
        self.assertGreaterEqual(fig.layout.xaxis.tickfont.size, PLOT_FONT_SIZE)
        self.assertGreaterEqual(fig.layout.yaxis.title.font.size, PLOT_FONT_SIZE)
        self.assertGreaterEqual(fig.layout.yaxis.tickfont.size, PLOT_FONT_SIZE)
        self.assertGreaterEqual(fig.layout.yaxis.title.standoff, PLOT_AXIS_TITLE_STANDOFF)
        self.assertTrue(fig.layout.yaxis.automargin)
        self.assertGreaterEqual(fig.layout.xaxis.title.standoff, PLOT_AXIS_TITLE_STANDOFF)
        self.assertTrue(fig.layout.xaxis.automargin)

    def test_plotly_render_wrapper_enforces_axis_title_spacing(self) -> None:
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=58, r=20, t=40, b=40),
            xaxis=dict(title=dict(text="Distance [m]")),
            yaxis=dict(title=dict(text="Front vertical-load share [-]")),
        )

        dash._enforce_readable_plot_fonts(fig)

        self.assertGreaterEqual(fig.layout.yaxis.title.standoff, PLOT_AXIS_TITLE_STANDOFF)
        self.assertTrue(fig.layout.yaxis.automargin)
        self.assertGreaterEqual(fig.layout.xaxis.title.standoff, PLOT_AXIS_TITLE_STANDOFF)
        self.assertTrue(fig.layout.xaxis.automargin)

    def test_plotly_render_wrapper_raises_subplot_titles(self) -> None:
        fig = apply_dark_layout(go.Figure(), title="Existing title")
        fig.update_layout(
            title=dict(font=dict(size=12)),
            annotations=[
                dict(text="FL", x=0.25, y=1.0, showarrow=False, font=dict(size=10)),
                dict(text="FR", x=0.75, y=1.0, showarrow=False),
            ],
        )

        dash._enforce_readable_plot_fonts(fig)

        self.assertGreaterEqual(fig.layout.title.font.size, PLOT_FONT_SIZE)
        for annotation in fig.layout.annotations:
            self.assertGreaterEqual(annotation.font.size, PLOT_FONT_SIZE)

    def test_plotly_render_wrapper_preserves_explicit_trace_text_size(self) -> None:
        fig = apply_dark_layout(go.Figure(), title="Track map")
        fig.add_trace(
            go.Scatter(
                x=[1],
                y=[1],
                mode="markers+text",
                text=["T1"],
                textfont=dict(size=10),
            )
        )

        dash._enforce_readable_plot_fonts(fig)

        self.assertEqual(fig.data[0].textfont.size, 10)

    def test_lap_comparison_track_map_keeps_title_and_legend_clear(self) -> None:
        comp = {
            "ref_longitude": np.array([8.564, 8.565, 8.566]),
            "ref_latitude": np.array([49.330, 49.331, 49.330]),
            "cmp_longitude": np.array([8.564, 8.565, 8.566]),
            "cmp_latitude": np.array([49.330, 49.331, 49.330]),
            "loss_rate_ms_10m": np.array([-20.0, 0.0, 30.0]),
            "s_m": np.array([0.0, 10.0, 20.0]),
            "ref_label": "Potential lap",
            "cmp_label": "FSG_A.csv L3",
        }
        turns = [
            SimpleNamespace(
                turn_id=1,
                apex_lng=8.565,
                apex_lat=49.331,
                s_apex_m=10.0,
            )
        ]

        with patch.object(drv, "lap_comparison_arrays", return_value=comp):
            fig = drv.lap_comparison_track_fig(
                {},
                "Potential lap",
                -1,
                "FSG_A.csv",
                3,
                turns=turns,
                active_turn_ids={1},
            )
        with patch.object(dash.st, "plotly_chart", return_value=None):
            dash._plotly_chart(fig)

        self.assertGreaterEqual(fig.layout.margin.t, 130)

    def test_plotly_legend_margin_leaves_room_for_readable_titles(self) -> None:
        fig = apply_dark_layout(go.Figure(), title="Power Distribution")
        fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], name="Run A"))
        fig.add_trace(go.Scatter(x=[1, 2], y=[2, 1], name="Run B"))
        fig.update_layout(
            annotations=[dict(text="FL", x=0.25, y=1.0, showarrow=False, font=dict(size=10))]
        )

        dash._enforce_readable_plot_fonts(fig)
        dash._place_legend_above_plot(fig)

        self.assertGreaterEqual(fig.layout.legend.y, 1.08)
        self.assertGreaterEqual(fig.layout.margin.t, 115)

    def test_plotly_render_wrapper_styles_split_csv_legend_rows(self) -> None:
        dash.st.session_state["selected_csv_files"] = ["FSG_A.csv", "FSG_B.csv"]
        fig = apply_dark_layout(go.Figure(), title="Battery SoC per Lap")
        fig.add_trace(go.Scatter(x=[1, 2], y=[90, 85], name="FSG_A.csv · SoC"))
        fig.add_trace(go.Bar(x=[1, 2], y=[5, 5], name="FSG_A.csv · ΔSoC"))
        fig.add_trace(go.Scatter(x=[1, 2], y=[40, 35], name="FSG_B.csv · SoC"))
        fig.add_trace(go.Bar(x=[1, 2], y=[4, 4], name="FSG_B.csv · ΔSoC"))

        with patch.object(dash.st, "plotly_chart", return_value=None):
            dash._plotly_chart(fig)

        self.assertGreaterEqual(fig.layout.legend.font.size, PLOT_FONT_SIZE)
        self.assertGreaterEqual(fig.layout.legend.title.font.size, PLOT_FONT_SIZE)
        self.assertIsNotNone(getattr(fig.layout, "legend2", None))
        self.assertGreaterEqual(fig.layout.legend2.font.size, PLOT_FONT_SIZE)
        self.assertGreaterEqual(fig.layout.legend2.title.font.size, PLOT_FONT_SIZE)
        row_gap = abs(fig.layout.legend2.y - fig.layout.legend.y)
        self.assertGreaterEqual(row_gap, 0.11)
        self.assertLessEqual(row_gap, 0.13)
        self.assertLessEqual(fig.layout.margin.t, 135)

    def test_streamlit_global_css_uses_document_readable_body_text(self) -> None:
        css = dash._GLOBAL_STYLE_CSS

        self.assertIn("font-size: 1.1rem;", css)
        self.assertIn("font-size: 1.15rem;", css)
        self.assertIn("font-size: 1.9rem;", css)
        self.assertIn("font-size: 17px;", css)

    def test_streamlit_global_css_offsets_plotly_modebar_below_title(self) -> None:
        css = dash._GLOBAL_STYLE_CSS

        self.assertIn(".js-plotly-plot .modebar-container", css)
        self.assertIn("top: 88px !important;", css)


if __name__ == "__main__":
    unittest.main()
