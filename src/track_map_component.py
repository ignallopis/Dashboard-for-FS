from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).resolve().parent / "track_map_component"
_track_map_component = components.declare_component(
    "track_map_component",
    path=str(_COMPONENT_DIR),
)


def render_track_map_component(
    figure_json: str,
    *,
    height_px: int,
    key: str,
    draw_enabled: bool = True,
) -> dict[str, Any]:
    """Render the interactive track map component and return its latest event.

    ``draw_enabled=False`` hides the line-drawing tools (keeping lasso/click),
    so a map can stay interactive for curve selection without offering gate
    drawing.
    """
    return _track_map_component(
        figure_json=figure_json,
        height_px=int(height_px),
        draw_enabled=bool(draw_enabled),
        default={
            "event_id": 0,
            "selection_indices": [],
            "clicked_turn_id": None,
            "line": None,
            "line_event": False,
            "fullscreen_event": False,
        },
        key=key,
    )


def serialize_figure(fig: Any) -> str:
    """Serialize a Plotly figure to JSON for the custom track-map component."""
    return fig.to_json()
