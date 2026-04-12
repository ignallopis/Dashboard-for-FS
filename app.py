"""Interactive telemetry dashboard — CAT17x.

Run with:
    streamlit run app.py

Layout:
    Left (55%): 3 distance plots stacked — Brake/Throttle | Steering/VN_vx | ax/ay
    Right (45%): Track map (top) + GG diagram (bottom)
    Bottom (left column): lap selector

Cross-chart: box-select (drag) in any left plot → yellow band or line in all left
plots + yellow dots on map. Hover shows all signals at that distance (unified tooltip).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from utils import exclude_lap0_and_last_lap, make_dark_figure, unique_laps


DATA_DIR = Path(__file__).parent / "data"

LOAD_COLUMNS = [
    "laps", "laptime",
    "Filtering_VN_ax", "Filtering_VN_ay",
    "VN_latitude", "VN_longitude",
    "VN_vx",
    "Brake", "Throttle", "Steering",
]

# Fixed colour per signal variable (independent of lap)
SIG_COLORS = {
    "throttle": "#73D973",   # green
    "brake":    "#D94F4F",   # red
    "steering": "#4DB3F2",   # blue
    "vx":       "#F28C40",   # orange
    "ax":       "#FFD700",   # gold
    "ay":       "#00BFBF",   # teal
}
# Lap distinguished by dash pattern cycling
DASH_CYCLE = ["solid", "dash", "dot", "dashdot"]

PURPLE_FASTEST = "rgb(170, 60, 230)"
_AXIS = "#E5E5E5"
_TEXT = "#EBEBEB"
_YELLOW      = "rgba(255, 220, 0, 0.9)"
_YELLOW_BAND = "rgba(255, 220, 0, 0.10)"

# Box-select narrower than this (metres) is treated as a point click → vline
CLICK_THR_M = 30.0

# Heights
_H_LEFT  = 250
_H_RIGHT = 390


# ── Data loading ──────────────────────────────────────────────────────────────

def _per_lap_distance(lat: np.ndarray, lng: np.ndarray, laps: np.ndarray) -> np.ndarray:
    """Haversine cumulative distance [m] per lap, reset to 0 at each lap start."""
    R = 6_371_000.0
    dist = np.zeros(len(lat))
    for lap_id in np.unique(laps[np.isfinite(laps)]):
        idx = np.where(laps == lap_id)[0]
        if len(idx) < 2:
            continue
        lat_r = np.radians(lat[idx])
        lng_r = np.radians(lng[idx])
        dlat = np.diff(lat_r)
        dlng = np.diff(lng_r)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(lat_r[:-1]) * np.cos(lat_r[1:]) * np.sin(dlng / 2) ** 2)
        inc = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        dist[idx] = np.concatenate([[0.0], np.cumsum(inc)])
    return dist


@st.cache_data(show_spinner=False)
def load_run(path: str) -> dict[str, np.ndarray]:
    df = pl.read_csv(path, columns=LOAD_COLUMNS)
    data: dict[str, np.ndarray] = {
        "laps":     df["laps"].to_numpy().astype(float),
        "laptime":  df["laptime"].to_numpy().astype(float),
        "ax":       df["Filtering_VN_ax"].to_numpy().astype(float),
        "ay":       df["Filtering_VN_ay"].to_numpy().astype(float),
        "lat":      df["VN_latitude"].to_numpy().astype(float),
        "lng":      df["VN_longitude"].to_numpy().astype(float),
        "vx":       df["VN_vx"].to_numpy().astype(float),
        "brake":    df["Brake"].to_numpy().astype(float),
        "throttle": df["Throttle"].to_numpy().astype(float),
        "steering": df["Steering"].to_numpy().astype(float),
    }
    finite = np.all(np.stack([np.isfinite(v) for v in data.values()]), axis=0)
    data = {k: v[finite] for k, v in data.items()}
    data = exclude_lap0_and_last_lap(data)
    data["dist"] = _per_lap_distance(data["lat"], data["lng"], data["laps"])
    return data


def per_lap_laptimes(data: dict[str, np.ndarray]) -> dict[int, float]:
    laps = data["laps"]
    lt   = data["laptime"]
    return {int(l): float(lt[laps == l].max()) for l in unique_laps(laps)}


# ── Colour gradient (GG diagram only) ────────────────────────────────────────

import plotly.colors as pc

def build_color_map(entries: list[tuple[str, int, float]]) -> dict[tuple[str, int], str]:
    if not entries:
        return {}
    ordered = sorted(entries, key=lambda e: e[2])
    n = len(ordered)
    colors: dict[tuple[str, int], str] = {}
    colors[(ordered[0][0], ordered[0][1])] = PURPLE_FASTEST
    if n == 1:
        return colors
    positions = [1.0 - (i / (n - 1)) for i in range(n)]
    scale = pc.sample_colorscale("RdYlGn", positions)
    for i in range(1, n):
        colors[(ordered[i][0], ordered[i][1])] = scale[i]
    return colors


# ── Selection helpers ─────────────────────────────────────────────────────────

def _get_pts(event) -> list:
    try:
        return event["selection"]["points"] or []
    except (TypeError, KeyError, AttributeError):
        return []


def _has_selection(event) -> bool:
    """True if the event contains any box coordinates or selected points."""
    try:
        sel = event["selection"]
        boxes = sel.get("box", [])
        if boxes and boxes[0].get("x"):
            return True
        return bool(sel.get("points", []))
    except (TypeError, KeyError, AttributeError):
        return False


def extract_zone_mask(event, n_points: int) -> tuple[np.ndarray, bool]:
    """(mask, is_active) from a track-map box/lasso event (uses point_index)."""
    pts = _get_pts(event)
    if not pts:
        return np.ones(n_points, dtype=bool), False
    mask = np.zeros(n_points, dtype=bool)
    for p in pts:
        idx = p.get("point_index") or p.get("pointIndex")
        if idx is not None and 0 <= idx < n_points:
            mask[idx] = True
    if not mask.any():
        return np.ones(n_points, dtype=bool), False
    return mask, True


def extract_gg_pool_indices(event) -> np.ndarray | None:
    """Extract pool indices from a GG selection event (reads customdata)."""
    pts = _get_pts(event)
    if not pts:
        return None
    indices = []
    for p in pts:
        cd = p.get("customdata")
        if cd is not None:
            if isinstance(cd, (list, tuple, np.ndarray)):
                indices.append(int(cd[0]))
            else:
                indices.append(int(cd))
    return np.array(indices, dtype=int) if indices else None


def dist_range_from_event(event) -> tuple[float, float] | None:
    """Extract (d_min, d_max) from a distance-plot selection event, or None.

    Prefers box-shape coordinates (reliable for line traces), falls back to
    selected-points x-values.
    """
    try:
        sel = event["selection"]
    except (TypeError, KeyError, AttributeError):
        return None

    # Box shape coordinates (most reliable for line-mode traces)
    try:
        boxes = sel.get("box", [])
        if boxes:
            xs = boxes[0].get("x", [])
            if len(xs) >= 2:
                return (float(min(xs)), float(max(xs)))
    except Exception:
        pass

    # Fall back to individual selected-point x-values
    pts = sel.get("points", [])
    if not pts:
        return None
    xs = [p["x"] for p in pts if p.get("x") is not None]
    return (float(min(xs)), float(max(xs))) if xs else None


# ── Distance-plot helper ──────────────────────────────────────────────────────

def add_dist_traces(
    fig: go.Figure,
    pool_dist: np.ndarray,
    pool_sig1: np.ndarray,
    pool_sig2: np.ndarray | None,
    pool_run: np.ndarray,
    pool_lap: np.ndarray,
    entries: list[tuple[str, int, float]],
    visible_keys: set[tuple[str, int]],
    single_csv: bool,
    sig1_label: str,
    sig1_color: str,
    sig2_label: str = "",
    sig2_color: str = "",
    sig2_yaxis: str = "y",
    extra_mask: np.ndarray | None = None,
) -> None:
    """Add one solid/dashed trace per lap for sig1 (and optionally sig2).

    Colour encodes the signal variable; dash pattern cycles across laps.
    Hover tooltip shows: <value> (L<lap_id>).
    """
    visible_entries = [
        (run, lap, lt)
        for run, lap, lt in sorted(entries, key=lambda e: e[2])
        if (run, lap) in visible_keys
    ]
    for i, (run, lap, lt) in enumerate(visible_entries):
        smask = (pool_run == run) & (pool_lap == lap)
        if extra_mask is not None:
            smask = smask & extra_mask
        if not smask.any():
            continue
        dash = DASH_CYCLE[i % len(DASH_CYCLE)]
        x     = pool_dist[smask]
        s1    = pool_sig1[smask]
        order = np.argsort(x)

        fig.add_trace(go.Scattergl(
            x=x[order], y=s1[order],
            mode="lines",
            name=sig1_label,
            showlegend=False,
            line=dict(color=sig1_color, width=1.5, dash=dash),
            hovertemplate=f"%{{y:.2f}} (L{lap})<extra></extra>",
        ))
        if pool_sig2 is not None and sig2_label:
            s2 = pool_sig2[smask]
            fig.add_trace(go.Scattergl(
                x=x[order], y=s2[order],
                mode="lines",
                name=sig2_label,
                showlegend=False,
                yaxis=sig2_yaxis,
                line=dict(color=sig2_color, width=1.5, dash=dash),
                hovertemplate=f"%{{y:.2f}} (L{lap})<extra></extra>",
            ))


def _right_yaxis(title: str) -> dict:
    return dict(
        title=title, overlaying="y", side="right",
        showgrid=False, color=_AXIS, linecolor=_AXIS, tickcolor=_AXIS,
        tickfont=dict(color=_TEXT), title_font=dict(color=_TEXT),
    )


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="CAT17x — Telemetry",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("CAT17x — Telemetry Dashboard")

    # ── File selection ────────────────────────────────────────────────────────
    csv_files = sorted(p.name for p in DATA_DIR.glob("*.csv"))
    if not csv_files:
        st.error(f"No CSV files found in `{DATA_DIR}`.")
        return

    st.sidebar.header("Runs")
    selected_files = [
        f for f in csv_files
        if st.sidebar.checkbox(f, value=(f == csv_files[0]), key=f"csv_{f}")
    ]
    if not selected_files:
        st.warning("Select at least one run from the sidebar.")
        return

    runs: dict[str, dict[str, np.ndarray]] = {}
    for fname in selected_files:
        try:
            runs[fname] = load_run(str(DATA_DIR / fname))
        except Exception as exc:
            st.warning(f"Skipping `{fname}`: {exc}")
    if not runs:
        st.error("No runs could be loaded.")
        return

    # ── Pool arrays ───────────────────────────────────────────────────────────
    ax_c, ay_c, lat_c, lng_c, dist_c = [], [], [], [], []
    brk_c, thr_c, ste_c, vx_c = [], [], [], []
    run_c, lap_c = [], []
    entries: list[tuple[str, int, float]] = []

    for run_name, d in runs.items():
        for lap, lt_val in per_lap_laptimes(d).items():
            entries.append((run_name, lap, lt_val))
        n = len(d["ax"])
        ax_c.append(d["ax"]);      ay_c.append(d["ay"])
        lat_c.append(d["lat"]);    lng_c.append(d["lng"])
        dist_c.append(d["dist"]);  vx_c.append(d["vx"])
        brk_c.append(d["brake"]);  thr_c.append(d["throttle"])
        ste_c.append(d["steering"])
        run_c.append(np.full(n, run_name))
        lap_c.append(d["laps"].astype(int))

    pool_ax   = np.concatenate(ax_c)
    pool_ay   = np.concatenate(ay_c)
    pool_lat  = np.concatenate(lat_c)
    pool_lng  = np.concatenate(lng_c)
    pool_dist = np.concatenate(dist_c)
    pool_vx   = np.concatenate(vx_c)
    pool_brk  = np.concatenate(brk_c)
    pool_thr  = np.concatenate(thr_c)
    pool_ste  = np.concatenate(ste_c)
    pool_run  = np.concatenate(run_c)
    pool_lap  = np.concatenate(lap_c)

    if not entries:
        st.warning("No valid laps in the selected runs.")
        return

    # Pre-compute GG range (locked axes, never auto-zoom)
    _gg_max  = float(max(np.abs(pool_ay).max(), np.abs(pool_ax).max())) * 1.1
    gg_range = [-_gg_max, _gg_max]

    color_map  = build_color_map(entries)   # used only for GG diagram
    single_csv = len(selected_files) == 1
    ui_rev     = "|".join(selected_files)

    # Lap labels — sorted fastest → slowest
    entries_by_time = sorted(entries, key=lambda e: e[2])
    if single_csv:
        lap_labels = [f"L{l}  ({t:.2f}s)" for _, l, t in entries_by_time]
    else:
        lap_labels = [f"{r} · L{l}  ({t:.2f}s)" for r, l, t in entries_by_time]
    label_to_key = {lbl: (r, l) for lbl, (r, l, _) in zip(lap_labels, entries_by_time)}

    # ── Read previous selection events from session_state ─────────────────────
    prev_gg_event = st.session_state.get("_gg_event")

    prev_cross_event = st.session_state.get("_cross_event")
    last_cross_chart = st.session_state.get("_last_cross_chart", "")
    _prev_cross_ver  = st.session_state.get("_cross_ver", 0)
    _prev_gg_ver     = st.session_state.get("_gg_ver", 0)
    cross_range      = dist_range_from_event(prev_cross_event)
    cross_active     = cross_range is not None
    if cross_active:
        d_min, d_max = cross_range
        cross_is_line = (d_max - d_min) < CLICK_THR_M

    # ── Container / column structure ──────────────────────────────────────────
    col_left, col_right = st.columns([11, 9])

    with col_left:
        cont_left_plots = st.container()   # visual top
        cont_left_sel   = st.container()   # visual bottom (lap selector)

    with col_right:
        cont_right = st.container()

    # ── Lap selector (fills bottom of left column) ────────────────────────────
    with cont_left_sel:
        # Colour legend matching GG diagram colours (fastest → slowest)
        color_spans = " &nbsp;".join(
            f'<span style="color:{color_map[(r, l)]}">■</span> '
            f'<span style="color:#ccc">{lbl}</span>'
            for lbl, (r, l, _) in zip(lap_labels, entries_by_time)
        )
        st.markdown(color_spans, unsafe_allow_html=True)
        selected_labels = st.multiselect(
            "Laps", options=lap_labels, default=lap_labels,
        )
    visible_keys = {label_to_key[lbl] for lbl in selected_labels}
    if not visible_keys:
        st.warning("Select at least one lap.")
        return

    # Visible mask for map highlights
    visible_mask = np.zeros(len(pool_run), dtype=bool)
    for (run, lap) in visible_keys:
        visible_mask |= (pool_run == run) & (pool_lap == lap)

    base_kwargs = dict(
        pool_run=pool_run, pool_lap=pool_lap,
        entries=entries, visible_keys=visible_keys,
        single_csv=single_csv,
    )

    # ── Right column: track map + GG (filled first for zone_mask) ────────────
    with cont_right:
        # ── Track map ─────────────────────────────────────────────────────────
        st.markdown("**Track** — drag to filter GG zone | yellow = selected section")
        track_fig = make_dark_figure(xlabel="Longitude [deg]", ylabel="Latitude [deg]")

        # Base track (gray, all laps)
        track_fig.add_trace(go.Scattergl(
            x=pool_lng, y=pool_lat,
            mode="markers",
            marker=dict(size=2, color="rgba(180,180,180,0.55)"),
            showlegend=False, hoverinfo="skip",
        ))

        # Yellow: cross-chart distance selection → visible laps only
        if cross_active:
            if cross_is_line:
                d_mid = (d_min + d_max) / 2
                map_mask = visible_mask & (np.abs(pool_dist - d_mid) < 5.0)
            else:
                map_mask = visible_mask & (pool_dist >= d_min) & (pool_dist <= d_max)
            if map_mask.any():
                track_fig.add_trace(go.Scattergl(
                    x=pool_lng[map_mask], y=pool_lat[map_mask],
                    mode="markers",
                    marker=dict(size=5, color=_YELLOW),
                    showlegend=False, hoverinfo="skip",
                ))

        # Yellow: GG lasso → track highlight (1-frame delay, uses customdata)
        if prev_gg_event is not None:
            gg_idx = extract_gg_pool_indices(prev_gg_event)
            if gg_idx is not None and len(gg_idx) > 0:
                track_fig.add_trace(go.Scattergl(
                    x=pool_lng[gg_idx], y=pool_lat[gg_idx],
                    mode="markers",
                    marker=dict(size=4, color=_YELLOW),
                    showlegend=False, hoverinfo="skip",
                ))

        track_fig.update_layout(
            height=_H_RIGHT,
            dragmode="select",
            uirevision=ui_rev,
            margin=dict(l=60, r=10, t=30, b=60),
        )
        event_track = st.plotly_chart(
            track_fig, use_container_width=True, theme=None,
            key="track_" + ui_rev,
            on_select="rerun", selection_mode=("box", "lasso"),
        )

        zone_mask, zone_active = extract_zone_mask(event_track, len(pool_ax))

        # ── GG diagram (filtered by track zone) ───────────────────────────────
        st.markdown(
            "**GG Diagram** — drag to highlight on map"
            + (f"  ·  zone: {int(zone_mask.sum())} pts" if zone_active else "")
        )
        gg_fig = make_dark_figure(
            xlabel="Filtering_VN_ay [m/s²]",
            ylabel="Filtering_VN_ax [m/s²]",
        )
        for run, lap, lt in sorted(entries, key=lambda e: e[2]):
            if (run, lap) not in visible_keys:
                continue
            smask = (pool_run == run) & (pool_lap == lap) & zone_mask
            if not smask.any():
                continue
            lap_name = f"L{lap} ({lt:.2f}s)" if single_csv else f"{run}·L{lap} ({lt:.2f}s)"
            pool_idx = np.where(smask)[0]
            gg_fig.add_trace(go.Scattergl(
                x=pool_ay[smask], y=pool_ax[smask],
                mode="markers", name=lap_name,
                marker=dict(size=3, color=color_map[(run, lap)], opacity=0.75),
                customdata=pool_idx.reshape(-1, 1),
                hovertemplate=(
                    f"{lap_name}<br>ay=%{{x:.2f}} m/s²<br>ax=%{{y:.2f}} m/s²<extra></extra>"
                ),
            ))

        gg_fig.update_layout(
            height=_H_RIGHT,
            uirevision=ui_rev,
            showlegend=False,
            margin=dict(l=60, r=10, t=30, b=60),
            xaxis=dict(range=gg_range, autorange=False),
            yaxis=dict(range=gg_range, autorange=False, scaleanchor="x", scaleratio=1),
        )
        event_gg = st.plotly_chart(
            gg_fig, use_container_width=True, theme=None,
            key="gg_" + ui_rev,
            on_select="rerun", selection_mode=("box", "lasso"),
        )
        if _has_selection(event_gg) and event_gg is not prev_gg_event:
            st.session_state["_gg_event"] = event_gg
            st.session_state["_gg_ver"] = st.session_state.get("_gg_ver", 0) + 1
        elif not _has_selection(event_gg) and prev_gg_event is not None:
            st.session_state.pop("_gg_event", None)
            st.session_state["_gg_ver"] = st.session_state.get("_gg_ver", 0) + 1

    # Pass zone_mask to left charts so they filter by track selection
    base_kwargs["extra_mask"] = zone_mask if zone_active else None

    # ── Left column: 3 distance plots ────────────────────────────────────────
    def _make_left_fig(ylabel: str, xlabel: str = "Distance [m]") -> go.Figure:
        fig = make_dark_figure(xlabel=xlabel, ylabel=ylabel)
        fig.update_layout(
            height=_H_LEFT, uirevision=ui_rev, showlegend=False,
            margin=dict(l=60, r=10, t=30, b=40),
            hovermode="x unified",
            dragmode="select",
        )
        # Spike line: visual cursor on hover
        fig.update_xaxes(
            showspikes=True, spikemode="across",
            spikedash="solid", spikecolor="rgba(200,200,200,0.5)",
            spikethickness=1,
        )
        return fig

    def _add_cross_mark(fig: go.Figure) -> None:
        """Add yellow vline (click) or vrect (range) for cross-chart selection."""
        if not cross_active:
            return
        if cross_is_line:
            fig.add_vline(
                x=(d_min + d_max) / 2,
                line=dict(color=_YELLOW, dash="solid", width=1.5),
            )
        else:
            fig.add_vrect(
                x0=d_min, x1=d_max,
                fillcolor=_YELLOW_BAND, layer="below", line_width=0,
            )

    def _store_cross(event, chart_id: str) -> None:
        if _has_selection(event):
            st.session_state["_cross_event"]      = event
            st.session_state["_last_cross_chart"] = chart_id
            st.session_state["_cross_ver"] = st.session_state.get("_cross_ver", 0) + 1
        elif last_cross_chart == chart_id:
            st.session_state.pop("_cross_event", None)
            st.session_state.pop("_last_cross_chart", None)
            st.session_state["_cross_ver"] = st.session_state.get("_cross_ver", 0) + 1

    with cont_left_plots:
        # ── Plot 1: Throttle (green, solid/dash/…) & Brake (red) ─────────────
        st.markdown(
            f'<span style="color:{SIG_COLORS["throttle"]}">■ Throttle</span> &nbsp;'
            f'<span style="color:{SIG_COLORS["brake"]}">■ Brake</span> &nbsp;'
            f'<span style="font-size:0.8em;color:#888">'
            f'solid=L1 · dash=L2 · dot=L3 · dashdot=L4+</span>',
            unsafe_allow_html=True,
        )
        bt_fig = _make_left_fig(ylabel="[%]")
        add_dist_traces(
            bt_fig, pool_dist, pool_thr, pool_brk,
            sig1_label="Throttle", sig1_color=SIG_COLORS["throttle"],
            sig2_label="Brake",    sig2_color=SIG_COLORS["brake"],
            **base_kwargs,
        )
        _add_cross_mark(bt_fig)
        event_bt = st.plotly_chart(
            bt_fig, use_container_width=True, theme=None,
            key="bt_" + ui_rev,
            on_select="rerun", selection_mode=("box",),
        )
        _store_cross(event_bt, "bt")

        # ── Plot 2: Steering (blue, left y) & VN_vx (orange, right y) ────────
        st.markdown(
            f'<span style="color:{SIG_COLORS["steering"]}">■ Steering</span> &nbsp;'
            f'<span style="color:{SIG_COLORS["vx"]}">■ VN_vx</span>',
            unsafe_allow_html=True,
        )
        mid_fig = _make_left_fig(ylabel="Steering [rad]")
        mid_fig.update_layout(
            yaxis2=_right_yaxis("VN_vx [m/s]"),
            margin=dict(l=60, r=60, t=30, b=40),
        )
        add_dist_traces(
            mid_fig, pool_dist, pool_ste, pool_vx,
            sig1_label="Steering", sig1_color=SIG_COLORS["steering"],
            sig2_label="VN_vx",    sig2_color=SIG_COLORS["vx"],
            sig2_yaxis="y2",
            **base_kwargs,
        )
        _add_cross_mark(mid_fig)
        event_mid = st.plotly_chart(
            mid_fig, use_container_width=True, theme=None,
            key="mid_" + ui_rev,
            on_select="rerun", selection_mode=("box",),
        )
        _store_cross(event_mid, "mid")

        # ── Plot 3: ax (gold) & ay (teal) ─────────────────────────────────────
        st.markdown(
            f'<span style="color:{SIG_COLORS["ax"]}">■ ax</span> &nbsp;'
            f'<span style="color:{SIG_COLORS["ay"]}">■ ay</span>',
            unsafe_allow_html=True,
        )
        bot_fig = _make_left_fig(ylabel="[m/s²]")
        add_dist_traces(
            bot_fig, pool_dist, pool_ax, pool_ay,
            sig1_label="ax", sig1_color=SIG_COLORS["ax"],
            sig2_label="ay", sig2_color=SIG_COLORS["ay"],
            **base_kwargs,
        )
        _add_cross_mark(bot_fig)
        event_bot = st.plotly_chart(
            bot_fig, use_container_width=True, theme=None,
            key="bot_" + ui_rev,
            on_select="rerun", selection_mode=("box",),
        )
        _store_cross(event_bot, "bot")

    # ── Force one extra rerun so map + all charts pick up new state ─────────
    _cross_changed = st.session_state.get("_cross_ver", 0) != _prev_cross_ver
    _gg_changed    = st.session_state.get("_gg_ver", 0) != _prev_gg_ver
    if _cross_changed or _gg_changed:
        st.rerun()


if __name__ == "__main__":
    main()
