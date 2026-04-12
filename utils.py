"""Shared utilities for CAT17x data analysis (Formula Student 4WD Electric)."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

# ── Dark theme constants ──────────────────────────────────────────────────────
_BG   = '#141417'
_TEXT = '#EBEBEB'
_GRID = 'rgba(128,128,128,0.2)'
_AXIS = '#E5E5E5'

# Per-wheel colours: FL=blue, FR=orange, RL=green, RR=purple
WHEEL_COLORS  = {'FL': '#4DB3F2', 'FR': '#F28C40', 'RL': '#73D973', 'RR': '#D973D9'}
WHEEL_SYMBOLS = {'FL': 'circle',  'FR': 'square',  'RL': 'triangle-up', 'RR': 'diamond'}


# ── Figure helpers ────────────────────────────────────────────────────────────

def make_dark_figure(title: str = '', xlabel: str = '', ylabel: str = '') -> go.Figure:
    """Return a Plotly Figure with dark motorsport styling."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=_TEXT)),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT, size=11),
        xaxis=dict(title=xlabel, color=_AXIS, gridcolor=_GRID,
                   linecolor=_AXIS, tickcolor=_AXIS, showgrid=True),
        yaxis=dict(title=ylabel, color=_AXIS, gridcolor=_GRID,
                   linecolor=_AXIS, tickcolor=_AXIS, showgrid=True),
        legend=dict(bgcolor='rgba(20,20,23,0.85)',
                    bordercolor='rgba(128,128,128,0.3)',
                    font=dict(color=_TEXT)),
    )
    return fig


def add_lap_scatter(fig: go.Figure, x: np.ndarray, y: np.ndarray,
                    lap_ids: np.ndarray, name: str = '',
                    color: str = '#4DB3F2', symbol: str = 'circle',
                    size: int = 10) -> None:
    """Add scatter trace with lap number labels."""
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers+text',
        name=name,
        marker=dict(color=color, symbol=symbol, size=size, line=dict(width=0)),
        text=[f'  {int(l)}' for l in lap_ids],
        textposition='middle right',
        textfont=dict(color=_TEXT, size=10),
    ))


def add_trend_line(fig: go.Figure, x: np.ndarray, y: np.ndarray,
                   color: str = '#F28C40', dash: str = 'dash') -> None:
    """Add a linear regression line to *fig*."""
    if len(x) < 2:
        return
    p     = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    fig.add_trace(go.Scatter(
        x=x_fit, y=np.polyval(p, x_fit),
        mode='lines', name='Trend',
        line=dict(color=color, dash=dash, width=1.6),
        showlegend=False,
    ))


def add_zero_line(fig: go.Figure, x: np.ndarray) -> None:
    """Add a horizontal dashed reference line at y=0."""
    fig.add_hline(y=0, line=dict(color='rgba(200,200,200,0.5)',
                                 dash='dash', width=1.2))


# ── Data helpers ──────────────────────────────────────────────────────────────

def keep_min_duration_segments(mask: np.ndarray,
                                min_duration: float,
                                dt: float) -> np.ndarray:
    """Remove boolean segments shorter than *min_duration* seconds.

    Args:
        mask:         Boolean event array.
        min_duration: Minimum segment duration [s].
        dt:           Sample interval [s].

    Returns:
        Filtered boolean array (same shape as *mask*).
    """
    clean = np.zeros(len(mask), dtype=bool)
    if not np.any(mask):
        return clean
    min_samples = max(1, int(np.ceil(min_duration / dt)))
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    d      = np.diff(padded.astype(np.int8))
    starts = np.where(d ==  1)[0]
    ends   = np.where(d == -1)[0] - 1
    for s, e in zip(starts, ends):
        if e - s + 1 >= min_samples:
            clean[s:e + 1] = True
    return clean


def exclude_lap0_and_last_lap(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Filter out formation lap (laps <= 0) and the last (incomplete) lap.

    *data* must contain key ``'laps'``.
    Raises ``ValueError`` if fewer than 2 valid laps remain.
    """
    laps  = data['laps']
    valid = laps > 0
    filt  = {k: v[valid] for k, v in data.items()}

    all_laps = np.unique(filt['laps'][np.isfinite(filt['laps'])])
    if len(all_laps) < 2:
        raise ValueError(
            'Not enough valid laps after excluding lap 0. '
            'Run lapcount.py first.'
        )
    keep = filt['laps'] != all_laps.max()
    return {k: v[keep] for k, v in filt.items()}


def robust_dt(time: np.ndarray) -> float:
    """Return median sample interval [s], ignoring gaps and NaNs."""
    diffs = np.diff(time)
    valid = diffs[(diffs > 0) & np.isfinite(diffs)]
    if len(valid) == 0:
        raise ValueError('Cannot compute dt: no positive time step found.')
    return float(np.median(valid))


def unique_laps(laps: np.ndarray) -> np.ndarray:
    """Sorted unique lap IDs, NaN excluded."""
    u = np.unique(laps)
    return u[np.isfinite(u)]
