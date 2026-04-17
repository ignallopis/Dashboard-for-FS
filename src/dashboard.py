"""CAT17x — Telemetry Dashboard

Entry point:  streamlit run src/dashboard.py

This is the only file that calls st.plotly_chart() or any other st.* rendering
functions.  All src/ modules return go.Figure objects (and kpis dicts) and never
render themselves.
"""
from __future__ import annotations

import copy
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pathlib import Path

import numpy as np
import polars as pl
import plotly.graph_objects as go
import streamlit as st

import streamlit.components.v1 as components

import src.powertrain as pt
import src.dynamics as dyn
import src.tc as tc
import src.tv as tv
import src.rb as rb
import src.driver as drv
import src.gripfactor as gf
import src.lapcount as lapcount
import src.videoanalysis as va
from utils import WHEEL_COLORS, available_laps, load_data, select_laps_df

DATA_DIR = Path(__file__).parent.parent / "data"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RUN_COLORS = ("#4DB3F2", "#F28C40", "#73D973", "#F27070", "#D973D9", "#F2C94C")
TRACE_DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")
TRACE_SYMBOLS = ("circle", "square", "diamond", "triangle-up", "x", "cross")
FileSignature = tuple[int, int]
_PL_HASH_FUNCS = {pl.DataFrame: lambda _df: 0}


# ── Data loading ──────────────────────────────────────────────────────────────

def _file_signature(path: Path) -> FileSignature:
    """Return a cache-busting signature for *path* based on its current stat()."""
    stat = path.stat()
    return (int(stat.st_mtime_ns), int(stat.st_size))


@st.cache_resource(show_spinner="Loading run...")
def load_run(path: str, file_signature: FileSignature) -> pl.DataFrame:
    """Load a CSV run through the shared project loader, keeping all laps."""
    _ = file_signature
    return load_data(path, complete_laps_only=False)


@st.cache_data(show_spinner=False)
def load_lap_gate(path: str, file_signature: FileSignature) -> dict | None:
    """Load lap-detection gate metadata from the raw CSV for map overlays."""
    _ = file_signature
    return lapcount.lap_detection_gate_from_csv(path)


@st.cache_data(show_spinner=False)
def csv_needs_lap_detection_cached(
    path: str,
    file_signature: FileSignature,
) -> bool:
    """Return whether *path* still needs GPS lap detection."""
    _ = file_signature
    return lapcount.csv_needs_lap_detection(path)


def _autodetect_laps(data_dir: Path) -> None:
    """Run lap detection on any CSV in *data_dir* that doesn't have it yet."""
    modified = False
    for path in sorted(data_dir.glob("*.csv")):
        try:
            file_signature = _file_signature(path)
            if not csv_needs_lap_detection_cached(str(path), file_signature):
                continue
        except Exception as exc:
            st.sidebar.warning(f"`{path.name}`: cannot inspect — {exc}")
            continue
        with st.spinner(f"Detecting laps in {path.name}..."):
            try:
                n = lapcount.detect_and_write_laps(path)
            except Exception as exc:
                st.sidebar.warning(f"`{path.name}`: lap detection failed — {exc}")
                continue
        modified = True
        if n > 0:
            st.sidebar.info(f"`{path.name}`: detected {n} laps")
        else:
            st.sidebar.warning(f"`{path.name}`: no laps detected from GPS")
    if modified:
        load_run.clear()
        load_lap_gate.clear()
        csv_needs_lap_detection_cached.clear()
        _driver_summary_cached.clear()
        _driver_throttle_histogram_fig_cached.clear()
        _driver_full_throttle_time_fig_cached.clear()
        _driver_throttle_speed_fig_cached.clear()
        _driver_braking_effort_fig_cached.clear()
        _driver_braking_aggressiveness_fig_cached.clear()
        _driver_brake_release_smoothness_fig_cached.clear()
        _driver_steering_smoothness_fig_cached.clear()
        _driver_corner_curvature_fig_cached.clear()
        _driver_circuit_map_fig_cached.clear()
        _driver_circuit_map_stats_cached.clear()


def _fmt(value: float | int, pattern: str) -> str:
    """Format finite numeric values, fallback to n/a."""
    try:
        if not np.isfinite(value):
            return "n/a"
    except TypeError:
        return "n/a"
    return format(value, pattern)


def _select_per_lap_axis(key: str, default: str) -> str:
    """Dashboard selector for per-lap chart x-axis."""
    labels = ["Lap", "Lap time [s]"]
    default_index = 0 if default == "laps" else 1
    choice = st.radio(
        "X-axis",
        options=labels,
        index=default_index,
        horizontal=True,
        key=key,
    )
    return "laps" if choice == "Lap" else "laptime"


def _format_lap_label(lap_id: int) -> str:
    """Human-readable label for sidebar and plot selectors."""
    lap_int = int(lap_id)
    return "Lap 0 (formation)" if lap_int == 0 else f"Lap {lap_int}"


def _lap_signature(df: pl.DataFrame) -> str:
    """Compact signature of the currently selected laps in *df*."""
    return ",".join(str(int(lap)) for lap in available_laps(df))


def _run_color_map(run_names: list[str]) -> dict[str, str]:
    """Stable color per run for multi-run overlays."""
    return {
        run_name: RUN_COLORS[i % len(RUN_COLORS)]
        for i, run_name in enumerate(run_names)
    }


def _wheel_token(trace_name: str) -> str | None:
    """Extract wheel ID from a trace name when present."""
    for wheel in ("FL", "FR", "RL", "RR"):
        if trace_name == wheel or f" {wheel}" in trace_name or trace_name.endswith(f"- {wheel}"):
            return wheel
    return None


def _style_trace_for_run(
    trace: go.BaseTraceType,
    run_name: str,
    run_color: str,
    variant_idx: int,
) -> go.BaseTraceType:
    """Clone *trace* and restyle it for a specific run overlay."""
    out = copy.deepcopy(trace)
    base_name = getattr(out, "name", "") or ""
    out.name = f"{run_name} · {base_name}" if base_name else run_name
    out.legendgroup = run_name

    dash = TRACE_DASHES[variant_idx % len(TRACE_DASHES)]
    symbol = TRACE_SYMBOLS[variant_idx % len(TRACE_SYMBOLS)]
    wheel = _wheel_token(base_name)
    trace_color = WHEEL_COLORS[wheel] if wheel is not None else run_color

    if hasattr(out, "line") and out.line is not None:
        out.line.color = trace_color
        if getattr(out, "mode", "") != "markers":
            out.line.dash = dash
    if hasattr(out, "marker") and out.marker is not None:
        out.marker.color = trace_color
        if wheel is not None:
            out.marker.line.color = run_color
            out.marker.line.width = 2
        elif getattr(out, "type", "") != "bar":
            out.marker.symbol = symbol
    if hasattr(out, "textfont") and out.textfont is not None and wheel is not None:
        out.textfont.color = run_color
    return out


def _overlay_figures(figs_by_run: dict[str, list[go.Figure]]) -> list[go.Figure]:
    """Overlay equally-indexed figures from multiple runs into one figure list."""
    if not figs_by_run:
        return []

    run_names = list(figs_by_run.keys())
    run_colors = _run_color_map(run_names)
    fig_count = len(next(iter(figs_by_run.values())))
    merged: list[go.Figure] = []

    for fig_idx in range(fig_count):
        base = go.Figure(next(iter(figs_by_run.values()))[fig_idx])
        base.data = ()
        variant_map: dict[str, int] = {}
        tickvals_by_axis: dict[str, set[float]] = {}

        for run_name in run_names:
            fig = figs_by_run[run_name][fig_idx]
            for trace_idx, trace in enumerate(fig.data):
                variant_key = getattr(trace, "name", None) or f"trace_{trace_idx}"
                if variant_key not in variant_map:
                    variant_map[variant_key] = len(variant_map)
                base.add_trace(
                    _style_trace_for_run(
                        trace,
                        run_name,
                        run_colors[run_name],
                        variant_map[variant_key],
                    )
                )

            for axis_name in fig.layout:
                if not str(axis_name).startswith("xaxis"):
                    continue
                tickvals = getattr(fig.layout[axis_name], "tickvals", None)
                if tickvals is None:
                    continue
                tickvals_by_axis.setdefault(str(axis_name), set()).update(float(v) for v in tickvals)

        for axis_name, tickvals in tickvals_by_axis.items():
            if tickvals:
                base.layout[axis_name].tickvals = sorted(tickvals)
        merged.append(base)

    return merged


def _concat_run_tables(run_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Concatenate per-run tables, adding a `Run` column when needed."""
    tables: list[pl.DataFrame] = []
    for run_name, table in run_tables.items():
        if "Run" not in table.columns:
            table = table.with_columns(pl.lit(run_name).alias("Run"))
        tables.append(table)
    return pl.concat(tables, how="vertical_relaxed") if tables else pl.DataFrame()


def _show_summary_table(rows: list[dict[str, object]]) -> None:
    """Render a compact per-run summary table when multiple CSVs are loaded."""
    if rows:
        st.dataframe(pl.DataFrame(rows), use_container_width=True, hide_index=True)


def _driver_run_tokens(
    dfs: dict[str, pl.DataFrame],
    file_signatures: dict[str, FileSignature],
) -> tuple[tuple[str, FileSignature, str], ...]:
    """Stable cache tokens for the current driver-tab runs and selected laps."""
    return tuple(
        (
            run_name,
            file_signatures[run_name],
            _lap_signature(df),
        )
        for run_name, df in dfs.items()
    )


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_summary_cached(
    df: pl.DataFrame,
    run_token: tuple[str, FileSignature, str],
) -> dict:
    """Cached wrapper for per-run driver summary metrics."""
    _ = run_token
    return drv.driver_summary(df)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_throttle_histogram_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> go.Figure:
    """Cached wrapper for the driver throttle histogram."""
    _ = run_tokens
    return drv.throttle_histogram_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_full_throttle_time_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> go.Figure:
    """Cached wrapper for full-throttle per-lap figure."""
    _ = run_tokens
    return drv.full_throttle_time_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_throttle_speed_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> go.Figure:
    """Cached wrapper for throttle-speed per-lap figure."""
    _ = run_tokens
    return drv.throttle_speed_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_braking_effort_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> go.Figure:
    """Cached wrapper for braking-effort figure."""
    _ = run_tokens
    return drv.braking_effort_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_braking_aggressiveness_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> go.Figure:
    """Cached wrapper for braking-aggressiveness per-lap figure."""
    _ = run_tokens
    return drv.braking_aggressiveness_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_brake_release_smoothness_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> go.Figure:
    """Cached wrapper for brake-release smoothness per-lap figure."""
    _ = run_tokens
    return drv.brake_release_smoothness_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_steering_smoothness_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> go.Figure:
    """Cached wrapper for steering-smoothness figure."""
    _ = run_tokens
    return drv.steering_smoothness_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_corner_curvature_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> go.Figure:
    """Cached wrapper for corner-curvature per-lap figure."""
    _ = run_tokens
    return drv.corner_curvature_fig(dfs, x_mode=x_mode)


def _driver_circuit_map_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    selected_map: tuple[tuple[str, int], ...],
) -> go.Figure:
    """Cached wrapper for driver circuit map figure."""
    _ = run_tokens
    return drv.circuit_map_fig(dfs, list(selected_map))


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_circuit_map_stats_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    selected_map: tuple[tuple[str, int], ...],
) -> pl.DataFrame:
    """Cached wrapper for driver circuit map phase tables."""
    _ = run_tokens
    return drv.circuit_map_stats(dfs, list(selected_map))


# ── Tab renderers ─────────────────────────────────────────────────────────────

def _tab_powertrain(dfs: dict[str, pl.DataFrame]) -> None:
    # ── Energy ───────────────────────────────────────────────────────────────
    st.subheader("Energy per Lap")
    energy_x_mode = _select_per_lap_axis("pt_energy_axis", default="laptime")
    try:
        if len(dfs) == 1:
            fig, kpis = pt.energy_per_lap_fig(dfs, x_mode=energy_x_mode)
            for w in kpis.get("warnings", []):
                st.warning(w)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean net / lap",     f"{kpis['e_mean']:.4f} kWh")
            c2.metric("Total net",          f"{kpis['e_total']:.3f} kWh")
            c3.metric("Coefficient of Variation", f"{kpis['cv']:.1f}%")
            c4.metric(
                f"R² (Enet vs {'laptime' if energy_x_mode == 'laptime' else 'lap'})",
                f"{kpis['r2']:.3f}",
            )
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Mean battery power", f"{kpis['p_mean']:.1f} kW")
            c6.metric("Consumed / lap",     f"{kpis['e_cons_mean']:.4f} kWh")
            c7.metric("Recovered / lap",    f"{kpis['e_rec_mean']:.4f} kWh")
            c8.metric("Fastest lap",        f"L{kpis['fastest_lap']} — {kpis['fastest_lt']:.2f} s")
            c9, c10, c11, c12 = st.columns(4)
            c9.metric("Total consumed",     f"{kpis['e_cons_total']:.3f} kWh")
            c10.metric("Total recovered",   f"{kpis['e_rec_total']:.3f} kWh")
            c11.metric("Min net lap",       f"L{kpis['min_e_lap']} — {kpis['min_e']:.4f} kWh")
            c12.metric("Max net lap",       f"L{kpis['max_e_lap']} — {kpis['max_e']:.4f} kWh")
            st.plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {
                run_name: pt.energy_per_lap_fig({run_name: df}, x_mode=energy_x_mode)
                for run_name, df in dfs.items()
            }
            for _run_name, (_fig, kpis) in run_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(w)
            _show_summary_table([
                {
                    "Run": run_name,
                    "Mean net / lap [kWh]": round(kpis["e_mean"], 4),
                    "Total net [kWh]": round(kpis["e_total"], 3),
                    "Consumed / lap [kWh]": round(kpis["e_cons_mean"], 4),
                    "Recovered / lap [kWh]": round(kpis["e_rec_mean"], 4),
                    "Mean battery power [kW]": round(kpis["p_mean"], 1),
                    "Fastest lap": int(kpis["fastest_lap"]),
                    "Fastest lt [s]": round(kpis["fastest_lt"], 2),
                }
                for run_name, (_fig, kpis) in run_results.items()
            ])
            st.plotly_chart(
                _overlay_figures({run_name: [fig] for run_name, (fig, _kpis) in run_results.items()})[0],
                use_container_width=True,
                theme=None,
            )
            with st.expander("Per-lap data"):
                st.dataframe(
                    _concat_run_tables({run_name: kpis["table"] for run_name, (_fig, kpis) in run_results.items()}),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.error(f"Energy KPIs unavailable: {exc}")

    st.divider()

    # ── Power per wheel ───────────────────────────────────────────────────────
    st.subheader("Power Distribution per Wheel")
    power_x_mode = _select_per_lap_axis("pt_power_axis", default="laps")
    try:
        if len(dfs) == 1:
            fig, kpis = pt.power_per_wheel_fig(dfs, x_mode=power_x_mode)
            for w in kpis.get("warnings", []):
                st.warning(w)
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean total power", f"{kpis['mean_total_kw']:.1f} kW")
            c2.metric("Front / Rear",
                      f"{kpis['fr_pct']:.1f}% / {100 - kpis['fr_pct']:.1f}%")
            c3.metric("Left / Right",
                      f"{kpis['lr_pct']:.1f}% / {100 - kpis['lr_pct']:.1f}%")
            c4, c5, c6, c7 = st.columns(4)
            for w, col in zip(("FL", "FR", "RL", "RR"), [c4, c5, c6, c7]):
                col.metric(w,
                           f"{kpis['wheel_mean_kw'][w]:.2f} kW"
                           f"  ({kpis['wheel_pct'][w]:.1f}%)")
            st.plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {
                run_name: pt.power_per_wheel_fig({run_name: df}, x_mode=power_x_mode)
                for run_name, df in dfs.items()
            }
            for _run_name, (_fig, kpis) in run_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(w)
            _show_summary_table([
                {
                    "Run": run_name,
                    "Mean total power [kW]": round(kpis["mean_total_kw"], 1),
                    "Front share [%]": round(kpis["fr_pct"], 1),
                    "Left share [%]": round(kpis["lr_pct"], 1),
                    "FL [kW]": round(kpis["wheel_mean_kw"]["FL"], 2),
                    "FR [kW]": round(kpis["wheel_mean_kw"]["FR"], 2),
                    "RL [kW]": round(kpis["wheel_mean_kw"]["RL"], 2),
                    "RR [kW]": round(kpis["wheel_mean_kw"]["RR"], 2),
                }
                for run_name, (_fig, kpis) in run_results.items()
            ])
            st.plotly_chart(
                _overlay_figures({run_name: [fig] for run_name, (fig, _kpis) in run_results.items()})[0],
                use_container_width=True,
                theme=None,
            )
            with st.expander("Per-lap data"):
                st.dataframe(
                    _concat_run_tables({run_name: kpis["table"] for run_name, (_fig, kpis) in run_results.items()}),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.error(f"Power KPIs unavailable: {exc}")

    st.divider()

    # ── Battery status ────────────────────────────────────────────────────────
    st.subheader("Battery Status")
    battery_x_mode = _select_per_lap_axis("pt_battery_axis", default="laps")
    try:
        if len(dfs) == 1:
            fig, kpis = pt.battery_status_fig(dfs, x_mode=battery_x_mode)
            for w in kpis.get("warnings", []):
                st.warning(w)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("SoC start",         f"{_fmt(kpis['soc_start'], '.1f')}%")
            c2.metric("SoC end",           f"{_fmt(kpis['soc_end'], '.1f')}%",
                      delta=(f"-{_fmt(kpis['soc_total_drop'], '.1f')}%" if np.isfinite(kpis["soc_total_drop"]) else None))
            c3.metric("Voltage sag",        f"{kpis['voltage_sag']:.1f} V")
            c4.metric("Cell spread (mean)", f"{kpis['cell_spread_mean']:.1f} mV")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Mean SoC drop / lap", f"{kpis['soc_drop_per_lap']:.2f}%")
            c6.metric("Mean voltage",        f"{kpis['mean_voltage']:.1f} V")
            c7.metric("Min voltage",         f"{kpis['min_voltage']:.1f} V")
            c8.metric("Mean current",        f"{kpis['mean_current']:.1f} A")
            st.plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {
                run_name: pt.battery_status_fig({run_name: df}, x_mode=battery_x_mode)
                for run_name, df in dfs.items()
            }
            for _run_name, (_fig, kpis) in run_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(w)
            _show_summary_table([
                {
                    "Run": run_name,
                    "Mean dSoC / lap [%]": round(kpis["soc_drop_per_lap"], 2),
                    "Voltage sag [V]": round(kpis["voltage_sag"], 1),
                    "Cell spread [mV]": round(kpis["cell_spread_mean"], 1),
                    "Mean voltage [V]": round(kpis["mean_voltage"], 1),
                    "Min voltage [V]": round(kpis["min_voltage"], 1),
                    "Mean current [A]": round(kpis["mean_current"], 1),
                }
                for run_name, (_fig, kpis) in run_results.items()
            ])
            st.plotly_chart(
                _overlay_figures({run_name: [fig] for run_name, (fig, _kpis) in run_results.items()})[0],
                use_container_width=True,
                theme=None,
            )
            with st.expander("Per-lap data"):
                st.dataframe(
                    _concat_run_tables({run_name: kpis["table"] for run_name, (_fig, kpis) in run_results.items()}),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.error(f"Battery KPIs unavailable: {exc}")

    st.divider()

    # ── Thermal evolution ─────────────────────────────────────────────────────
    st.subheader("Thermal Evolution")
    thermal_x_mode = _select_per_lap_axis("pt_thermal_axis", default="laps")
    try:
        if len(dfs) == 1:
            fig, kpis = pt.thermal_evolution_fig(dfs, x_mode=thermal_x_mode)
            for w in kpis.get("warnings", []):
                st.warning(w)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Peak motor (P95)",    f"{kpis['peak_motor']:.1f} °C")
            c2.metric("Peak inverter (P95)", f"{kpis['peak_inverter']:.1f} °C")
            c3.metric("Peak battery Tmax",   f"{kpis['peak_batt_tmax']:.1f} °C")
            c4.metric("Motor thermal slope", f"{kpis['motor_thermal_slope']:+.2f} °C/lap")
            c5, c6, c7, c8 = st.columns(4)
            for w, col in zip(("FL", "FR", "RL", "RR"), [c5, c6, c7, c8]):
                col.metric(f"Motor {w} peak",
                           f"{kpis['motor_peak_by_wheel'][w]:.1f} °C")
            st.plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {
                run_name: pt.thermal_evolution_fig({run_name: df}, x_mode=thermal_x_mode)
                for run_name, df in dfs.items()
            }
            for _run_name, (_fig, kpis) in run_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(w)
            _show_summary_table([
                {
                    "Run": run_name,
                    "Peak motor [°C]": round(kpis["peak_motor"], 1),
                    "Peak inverter [°C]": round(kpis["peak_inverter"], 1),
                    "Peak batt Tmax [°C]": round(kpis["peak_batt_tmax"], 1),
                    "Motor slope [°C/lap]": round(kpis["motor_thermal_slope"], 2),
                }
                for run_name, (_fig, kpis) in run_results.items()
            ])
            st.plotly_chart(
                _overlay_figures({run_name: [fig] for run_name, (fig, _kpis) in run_results.items()})[0],
                use_container_width=True,
                theme=None,
            )
            with st.expander("Per-lap data"):
                st.dataframe(
                    _concat_run_tables({run_name: kpis["table"] for run_name, (_fig, kpis) in run_results.items()}),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.error(f"Thermal KPIs unavailable: {exc}")


def _tab_dynamics(dfs: dict[str, pl.DataFrame]) -> None:
    dyn_section = st.segmented_control(
        "Dynamics section",
        options=["Cornering", "Grip Factors"],
        default="Cornering",
        required=True,
        key="dyn_subsection",
        label_visibility="collapsed",
        width="stretch",
    )

    if dyn_section == "Grip Factors":
        _render_dynamics_grip_factors(dfs)
        return

    _render_dynamics_cornering(dfs)


def _render_dynamics_cornering(dfs: dict[str, pl.DataFrame]) -> None:
    # ── Interactive: track map + GG + driver traces ───────────────────────────
    st.subheader("GG Diagram & Driver Traces")
    st.caption(
        "Track map phases: red = braking, yellow = cornering, green = straight. "
        "White dashed gate = lap detection. Blue highlight = selected section."
    )
    lap_gates: dict[str, dict] = {}
    for run_name in dfs:
        try:
            run_path = DATA_DIR / run_name
            gate = load_lap_gate(str(run_path), _file_signature(run_path))
        except Exception:
            gate = None
        if gate is not None:
            lap_gates[run_name] = gate
    pool, entries = dyn.pool_arrays_from_dfs(dfs)
    if not pool or not entries:
        st.warning("Missing required columns (GPS, ax/ay, Brake, Throttle, Steering).")
    else:
        single_csv = len(dfs) == 1
        ui_rev = "|".join(
            f"{run_name}:{_lap_signature(df)}"
            for run_name, df in sorted(dfs.items())
        )
        color_map  = dyn.build_color_map(entries)

        # Clear stale cross/GG events when the active CSV set changes
        _prev_ui_rev = st.session_state.get("_dyn_ui_rev", "")
        if _prev_ui_rev != ui_rev:
            for _k in ("_dyn_gg_event", "_dyn_cross_event", "_dyn_last_cross",
                       "_dyn_cross_ver", "_dyn_gg_ver"):
                st.session_state.pop(_k, None)
            st.session_state["_dyn_ui_rev"] = ui_rev

        # Read previous events from session state
        prev_gg_event    = st.session_state.get("_dyn_gg_event")
        prev_cross_event = st.session_state.get("_dyn_cross_event")
        last_cross_chart = st.session_state.get("_dyn_last_cross", "")
        _prev_cross_ver  = st.session_state.get("_dyn_cross_ver", 0)
        _prev_gg_ver     = st.session_state.get("_dyn_gg_ver", 0)

        cross_range = dyn.dist_range_from_event(prev_cross_event)
        gg_idx      = dyn.extract_gg_pool_indices(prev_gg_event)

        col_left, col_right = st.columns([11, 9])
        with col_left:
            cont_left_plots = st.container()
            cont_left_sel   = st.container()

        # Lap selector (bottom of left column) — one column per run, fastest → slowest
        with cont_left_sel:
            run_entries_dyn: dict[str, list[tuple[str, int, float]]] = {}
            for _r, _l, _t in entries:
                run_entries_dyn.setdefault(_r, []).append((_r, _l, _t))
            for _rn in run_entries_dyn:
                run_entries_dyn[_rn].sort(key=lambda e: e[2] if np.isfinite(e[2]) else 1e9)

            sel_cols_dyn = st.columns(len(run_entries_dyn))
            visible_keys: set[tuple[str, int]] = set()
            for _col, (_rn, _run_ents) in zip(sel_cols_dyn, run_entries_dyn.items()):
                _rcmap = dyn.build_color_map(_run_ents)
                _rlabels = [
                    f"L{l}  ({t:.2f}s)" if np.isfinite(t) else f"L{l}"
                    for _, l, t in _run_ents
                ]
                _rl2key = {lbl: (r, l) for lbl, (r, l, _) in zip(_rlabels, _run_ents)}
                with _col:
                    st.markdown(f"**{_rn}**")
                    st.markdown(" &nbsp;".join(
                        f'<span style="color:{_rcmap.get((r, l), "#ccc")}">■</span>'
                        f' <span style="color:#ccc">{lbl}</span>'
                        for lbl, (r, l, _) in zip(_rlabels, _run_ents)
                    ), unsafe_allow_html=True)
                    _sel = st.multiselect(
                        _rn,
                        options=_rlabels,
                        default=_rlabels,
                        key=f"dyn_lap_sel_{_rn}_{ui_rev}",
                        label_visibility="collapsed",
                    )
                    visible_keys.update(_rl2key[lbl] for lbl in _sel)
        if not visible_keys:
            st.warning("Select at least one lap.")
            return

        visible_mask = np.zeros(len(pool["run"]), dtype=bool)
        for (run, lap) in visible_keys:
            visible_mask |= (pool["run"] == run) & (pool["lap"] == lap)

        # Right column: track map + GG
        with col_right:
            st.markdown(
                "**Track** — drag to filter GG zone | white dashed = lap detection | "
                "blue = selected section"
            )
            track_fig = dyn.track_map_fig(
                pool, visible_mask, cross_range, gg_idx, ui_rev, lap_gates=lap_gates,
            )
            event_track = st.plotly_chart(
                track_fig, use_container_width=True, theme=None,
                key="dyn_track_" + ui_rev,
                on_select="rerun", selection_mode=("box", "lasso"),
            )
            zone_mask, zone_active = dyn.extract_zone_mask(event_track, len(pool["ax"]))
            _vk_str = sorted(f"{r}:{l}" for r, l in visible_keys)
            gg_key_suffix = (
                "|".join(_vk_str)
                + f"|zone:{int(zone_active)}|pts:{int(zone_mask.sum())}"
            )
            gg_ui_rev = (
                ui_rev
                + "|gg|"
                + ",".join(_vk_str)
            )

            st.markdown(
                "**GG Diagram** — drag to highlight on map"
                + (f"  ·  zone: {int(zone_mask.sum())} pts" if zone_active else "")
            )
            gg_fig = dyn.gg_diagram_fig(
                pool, entries, visible_keys, color_map,
                zone_mask, single_csv, gg_ui_rev, [],
            )
            event_gg = st.plotly_chart(
                gg_fig, use_container_width=True, theme=None,
                key="dyn_gg_" + ui_rev + "|" + gg_key_suffix,
                on_select="rerun", selection_mode=("box", "lasso"),
            )
            if dyn.has_selection(event_gg) and event_gg is not prev_gg_event:
                st.session_state["_dyn_gg_event"] = event_gg
                st.session_state["_dyn_gg_ver"]   = _prev_gg_ver + 1
            elif not dyn.has_selection(event_gg) and prev_gg_event is not None:
                st.session_state.pop("_dyn_gg_event", None)
                st.session_state["_dyn_gg_ver"] = _prev_gg_ver + 1

        extra_mask = zone_mask if zone_active else None

        def _store_cross(event, chart_id: str) -> None:
            if dyn.has_selection(event):
                st.session_state["_dyn_cross_event"] = event
                st.session_state["_dyn_last_cross"]  = chart_id
                st.session_state["_dyn_cross_ver"]   = _prev_cross_ver + 1
            elif last_cross_chart == chart_id:
                st.session_state.pop("_dyn_cross_event", None)
                st.session_state.pop("_dyn_last_cross", None)
                st.session_state["_dyn_cross_ver"] = _prev_cross_ver + 1

        SIG = dyn.SIG_COLORS
        dist_kwargs = dict(
            pool=pool, entries=entries, visible_keys=visible_keys,
            extra_mask=extra_mask, ui_rev=ui_rev, cross_range=cross_range,
        )

        with cont_left_plots:
            # CSS to collapse Streamlit vertical gaps between stacked charts
            st.markdown(
                '<style>'
                '[data-st-key="dyn_stacked_charts"] > div { gap: 0 !important; }'
                '</style>'
                '<span style="font-size:0.8em;color:#888">'
                'solid=L1 · dash=L2 · dot=L3 · dashdot=L4+</span>',
                unsafe_allow_html=True,
            )

            stacked = st.container(key="dyn_stacked_charts")
            with stacked:
                bt_fig = dyn.dist_plot_fig(
                    sig1_key="thr", sig2_key="brk", ylabel="[%]",
                    sig1_label="Throttle", sig1_color=SIG["throttle"],
                    sig2_label="Brake",    sig2_color=SIG["brake"],
                    sig2_yaxis="y", compact="top",
                    **dist_kwargs,
                )
                event_bt = st.plotly_chart(
                    bt_fig, use_container_width=True, theme=None,
                    key="dyn_bt_" + ui_rev,
                    on_select="rerun", selection_mode=("box",),
                )
                _store_cross(event_bt, "bt")

                mid_fig = dyn.dist_plot_fig(
                    sig1_key="ste", sig2_key="vx", ylabel="Steering [deg]",
                    sig1_label="Steering [deg]", sig1_color=SIG["steering"],
                    sig2_label="VN_vx",    sig2_color=SIG["vx"],
                    sig2_yaxis="y2",       right_yaxis_title="",
                    compact="middle",
                    **dist_kwargs,
                )
                event_mid = st.plotly_chart(
                    mid_fig, use_container_width=True, theme=None,
                    key="dyn_mid_" + ui_rev,
                    on_select="rerun", selection_mode=("box",),
                )
                _store_cross(event_mid, "mid")

                bot_fig = dyn.dist_plot_fig(
                    sig1_key="ax", sig2_key="ay", ylabel="[m/s²]",
                    sig1_label="ax", sig1_color=SIG["ax"],
                    sig2_label="ay", sig2_color=SIG["ay"],
                    sig2_yaxis="y", compact="bottom",
                    **dist_kwargs,
                )
                event_bot = st.plotly_chart(
                    bot_fig, use_container_width=True, theme=None,
                    key="dyn_bot_" + ui_rev,
                    on_select="rerun", selection_mode=("box",),
                )
                _store_cross(event_bot, "bot")

        # Force rerun so map + all charts pick up new state (1-frame lag fix)
        _cross_changed = st.session_state.get("_dyn_cross_ver", 0) != _prev_cross_ver
        _gg_changed    = st.session_state.get("_dyn_gg_ver", 0) != _prev_gg_ver
        if _cross_changed or _gg_changed:
            st.rerun()

    st.divider()

    # ── Per-run analysis: slip angle + understeer ─────────────────────────────
    st.subheader("Slip Angle Efficiency")
    slip_x_mode = _select_per_lap_axis("dyn_slip_axis", default="laptime")
    try:
        if len(dfs) == 1:
            run_name, df = next(iter(dfs.items()))
            kpis = dyn.slip_angle_efficiency_kpis(df)
            for w in kpis.get("warnings", []):
                st.warning(w)
            if not kpis.get("warnings"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Valid laps", str(kpis["valid_laps"]))
                c2.metric("Mean corner ay", f"{_fmt(kpis['mean_corner_ay'], '.2f')} m/s²")
                c3.metric("Best mean efficiency", f"{_fmt(kpis['best_eff'], '.3f')} m/s²/deg")
                c4.metric("Fastest valid lap", f"L{kpis['fastest_lap']} — {_fmt(kpis['fastest_lt'], '.2f')} s")
                c5, c6, c7, c8 = st.columns(4)
                for w, col in zip(("FL", "FR", "RL", "RR"), [c5, c6, c7, c8]):
                    col.metric(f"Eff {w}", f"{_fmt(kpis['eff_mean_by_wheel'][w], '.3f')} m/s²/deg")
            for fig in dyn.slip_angle_efficiency_figs(df, x_mode=slip_x_mode):
                st.plotly_chart(fig, use_container_width=True, theme=None)
            if not kpis.get("warnings"):
                with st.expander("Per-lap data"):
                    st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {}
            for run_name, df in dfs.items():
                run_results[run_name] = (
                    dyn.slip_angle_efficiency_figs(df, x_mode=slip_x_mode),
                    dyn.slip_angle_efficiency_kpis(df),
                )
                for w in run_results[run_name][1].get("warnings", []):
                    st.warning(f"{run_name}: {w}")
            _show_summary_table([
                {
                    "Run": run_name,
                    "Valid laps": kpis["valid_laps"],
                    "Mean corner ay [m/s²]": round(kpis["mean_corner_ay"], 2),
                    "Best mean efficiency": round(kpis["best_eff"], 3),
                    "Fastest lap": int(kpis["fastest_lap"]),
                    "Fastest lt [s]": round(kpis["fastest_lt"], 2),
                }
                for run_name, (_figs, kpis) in run_results.items()
                if not kpis.get("warnings")
            ])
            for fig in _overlay_figures({run_name: figs for run_name, (figs, _kpis) in run_results.items()}):
                st.plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(
                    _concat_run_tables({
                        run_name: kpis["table"]
                        for run_name, (_figs, kpis) in run_results.items()
                        if not kpis.get("warnings")
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.warning(f"Slip angle efficiency: {exc}")

    st.divider()

    st.subheader("Understeer Angle")
    understeer_x_mode = _select_per_lap_axis("dyn_understeer_axis", default="laps")
    try:
        if len(dfs) == 1:
            run_name, df = next(iter(dfs.items()))
            kpis = dyn.understeer_angle_kpis(df)
            for w in kpis.get("warnings", []):
                st.warning(w)
            if not kpis.get("warnings"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Valid laps", str(kpis["valid_laps"]))
                c2.metric("Mean understeer", f"{_fmt(kpis['mean_understeer'], '.2f')} deg")
                c3.metric("Min / Max", f"{_fmt(kpis['min_understeer'], '.2f')} / {_fmt(kpis['max_understeer'], '.2f')} deg")
                c4.metric("Fastest valid lap", f"L{kpis['fastest_lap']} — {_fmt(kpis['fastest_lt'], '.2f')} s")
            fig = dyn.understeer_angle_fig(df, x_mode=understeer_x_mode)
            st.plotly_chart(fig, use_container_width=True, theme=None)
            if not kpis.get("warnings"):
                with st.expander("Per-lap data"):
                    st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {}
            for run_name, df in dfs.items():
                run_results[run_name] = (
                    [dyn.understeer_angle_fig(df, x_mode=understeer_x_mode)],
                    dyn.understeer_angle_kpis(df),
                )
                for w in run_results[run_name][1].get("warnings", []):
                    st.warning(f"{run_name}: {w}")
            _show_summary_table([
                {
                    "Run": run_name,
                    "Valid laps": kpis["valid_laps"],
                    "Mean understeer [deg]": round(kpis["mean_understeer"], 2),
                    "Min [deg]": round(kpis["min_understeer"], 2),
                    "Max [deg]": round(kpis["max_understeer"], 2),
                    "Fastest lap": int(kpis["fastest_lap"]),
                    "Fastest lt [s]": round(kpis["fastest_lt"], 2),
                }
                for run_name, (_figs, kpis) in run_results.items()
                if not kpis.get("warnings")
            ])
            st.plotly_chart(
                _overlay_figures({run_name: figs for run_name, (figs, _kpis) in run_results.items()})[0],
                use_container_width=True,
                theme=None,
            )
            with st.expander("Per-lap data"):
                st.dataframe(
                    _concat_run_tables({
                        run_name: kpis["table"]
                        for run_name, (_figs, kpis) in run_results.items()
                        if not kpis.get("warnings")
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.warning(f"Understeer angle: {exc}")


def _render_dynamics_grip_factors(dfs: dict[str, pl.DataFrame]) -> None:
    """Per-lap grip factor analysis adapted to FS (no aero grip)."""
    st.subheader("Grip Factors")
    st.caption(
        "Mean grip per category over each lap, in G units "
        "(Buurman methodology, FS-adapted thresholds). "
        "Channels are independent: a sample can count for both cornering and "
        "braking, or cornering and traction. "
        "Aero grip omitted: FS speeds are too low to isolate downforce cleanly."
    )

    # ── Auto-estimate thresholds from data ─────────────────────────────
    data_key = tuple(sorted(dfs.keys()))
    if st.session_state.get("_gf_data_key") != data_key:
        st.session_state["_gf_data_key"] = data_key
        combined_df = pl.concat(list(dfs.values()), how="diagonal_relaxed")
        est = gf.estimate_thresholds(combined_df)
        st.session_state["gf_overall"] = est.overall_combined_g
        st.session_state["gf_corner_ay"] = est.cornering_ay_g
        st.session_state["gf_braking"] = est.braking_ax_g
        st.session_state["gf_traction_ax"] = est.traction_ax_g
        st.session_state["gf_traction_ay"] = est.traction_ay_g
        st.session_state["gf_min_samples"] = est.min_samples

    # ── Boundary conditions ──────────────────────────────────────────────
    with st.expander("Boundary conditions [G]"):
        st.caption(
            "Auto-estimated from the loaded data. "
            "Traction uses positive ax plus lateral load to isolate corner exit, "
            "which is more meaningful in FS than straight-line accel."
        )
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown(
                '<b style="color:#FFD700">Overall</b> '
                "— grip-limited combined acceleration",
                unsafe_allow_html=True,
            )
            overall = st.number_input(
                "Combined G >=",
                min_value=0.20, max_value=2.00, step=0.05,
                key="gf_overall", format="%.2f",
            )
        with bc2:
            st.markdown(
                '<b style="color:#D94F4F">Braking</b> '
                "— hard deceleration / trail-braking",
                unsafe_allow_html=True,
            )
            braking = st.number_input(
                "Deceleration G >=",
                min_value=0.20, max_value=2.00, step=0.05,
                key="gf_braking", format="%.2f",
            )
        bc3, bc4 = st.columns(2)
        with bc3:
            st.markdown(
                '<b style="color:#00BFBF">Cornering</b> '
                "— sustained lateral load",
                unsafe_allow_html=True,
            )
            cornering = st.number_input(
                "Lateral G >=",
                min_value=0.10, max_value=2.00, step=0.05,
                key="gf_corner_ay", format="%.2f",
            )
        with bc4:
            st.markdown(
                '<b style="color:#73D973">Traction</b> '
                "— corner exit under power",
                unsafe_allow_html=True,
            )
            traction_ax = st.number_input(
                "Acceleration G >=",
                min_value=0.10, max_value=2.00, step=0.05,
                key="gf_traction_ax", format="%.2f",
            )
            traction_ay = st.number_input(
                "Lateral G >=",
                min_value=0.10, max_value=2.00, step=0.05,
                key="gf_traction_ay", format="%.2f",
            )
        min_samples = st.number_input(
            "Min. samples per category per lap",
            min_value=5, max_value=500, step=5,
            key="gf_min_samples",
        )

    thresholds = gf.GripThresholds(
        overall_combined_g=overall,
        cornering_ay_g=cornering,
        braking_ax_g=braking,
        traction_ax_g=traction_ax,
        traction_ay_g=traction_ay,
        min_samples=min_samples,
    )

    try:
        if len(dfs) == 1:
            run_name, df = next(iter(dfs.items()))
            kpis = gf.grip_factor_kpis(df, thresholds)
            for w in kpis.get("warnings", []):
                st.warning(w)

            # ── Track maps (top — tune thresholds here) ──────────────
            if not kpis.get("warnings") and not kpis["table"].is_empty():
                lap_options = [int(l) for l in kpis["table"]["Lap"].to_list()]
                fastest = kpis.get("fastest_lap")
                default_idx = (
                    lap_options.index(int(fastest))
                    if fastest is not None and int(fastest) in lap_options
                    else 0
                )
                selected_lap = st.selectbox(
                    "Track map — lap",
                    options=lap_options,
                    index=default_idx,
                    format_func=lambda l: f"L{int(l)}",
                    key="dyn_gf_map_lap",
                )
                map_fig = gf.grip_factor_track_maps_fig(
                    df, int(selected_lap), thresholds,
                )
                st.plotly_chart(map_fig, use_container_width=True, theme=None)

            # ── KPI metrics ──────────────────────────────────────────
            if not kpis.get("warnings"):
                m = kpis["means"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Overall",   f"{_fmt(m['Overall'],   '.2f')} G")
                c2.metric("Cornering", f"{_fmt(m['Cornering'], '.2f')} G")
                c3.metric("Braking",   f"{_fmt(m['Braking'],   '.2f')} G")
                c4.metric("Traction",  f"{_fmt(m['Traction'],  '.2f')} G")
                c5, c6 = st.columns(2)
                c5.metric("Valid laps", str(kpis["valid_laps"]))
                if kpis["fastest_lap"] is not None:
                    c6.metric(
                        "Fastest valid lap",
                        f"L{kpis['fastest_lap']} — {_fmt(kpis['fastest_lt'], '.2f')} s",
                    )

            # ── Evolution chart ──────────────────────────────────────
            gf_x_mode = _select_per_lap_axis("dyn_gf_axis", default="laps")
            evo_fig = gf.grip_factor_evolution_fig(
                kpis["table"], x_mode=gf_x_mode,
            )
            st.plotly_chart(evo_fig, use_container_width=True, theme=None)

            if not kpis.get("warnings"):
                with st.expander("Per-lap grip factors"):
                    st.dataframe(
                        kpis["table"],
                        use_container_width=True,
                        hide_index=True,
                    )

        # ── Multi-CSV ────────────────────────────────────────────────
        else:
            run_results: dict[str, dict] = {}
            for run_name, df in dfs.items():
                run_results[run_name] = gf.grip_factor_kpis(df, thresholds)
                for w in run_results[run_name].get("warnings", []):
                    st.warning(f"{run_name}: {w}")
            valid_results = {
                rn: k for rn, k in run_results.items()
                if not k.get("warnings")
            }
            if not valid_results:
                return

            # ── Track maps (top) ─────────────────────────────────────
            map_runs = [
                rn for rn, k in valid_results.items()
                if not k["table"].is_empty()
            ]
            if map_runs:
                mc1, mc2 = st.columns([2, 1])
                with mc1:
                    map_run = st.selectbox(
                        "Track map — run",
                        options=map_runs,
                        key="dyn_gf_map_run_multi",
                    )
                run_table = valid_results[map_run]["table"]
                map_lap_opts = [int(l) for l in run_table["Lap"].to_list()]
                fastest = valid_results[map_run].get("fastest_lap")
                def_idx = (
                    map_lap_opts.index(int(fastest))
                    if fastest is not None and int(fastest) in map_lap_opts
                    else 0
                )
                with mc2:
                    map_lap = st.selectbox(
                        "Track map — lap",
                        options=map_lap_opts,
                        index=def_idx,
                        format_func=lambda l: f"L{int(l)}",
                        key="dyn_gf_map_lap_multi",
                    )
                map_fig = gf.grip_factor_track_maps_fig(
                    dfs[map_run], int(map_lap), thresholds,
                )
                st.plotly_chart(map_fig, use_container_width=True, theme=None)

            # ── Summary table ────────────────────────────────────────
            _show_summary_table([
                {
                    "Run": rn,
                    "Valid laps": k["valid_laps"],
                    "Overall [G]":   round(k["means"]["Overall"],   3),
                    "Cornering [G]": round(k["means"]["Cornering"], 3),
                    "Braking [G]":   round(k["means"]["Braking"],   3),
                    "Traction [G]":  round(k["means"]["Traction"],  3),
                    "Fastest lap": k["fastest_lap"],
                    "Fastest lt [s]": (
                        round(k["fastest_lt"], 2)
                        if np.isfinite(k["fastest_lt"]) else None
                    ),
                }
                for rn, k in valid_results.items()
            ])

            # ── Evolution chart (category colours) ───────────────────
            gf_x_mode = _select_per_lap_axis("dyn_gf_axis", default="laps")
            evo_fig = gf.grip_factor_evolution_multi_fig(
                {rn: k["table"] for rn, k in valid_results.items()},
                x_mode=gf_x_mode,
            )
            st.plotly_chart(evo_fig, use_container_width=True, theme=None)

            # ── Radar ────────────────────────────────────────────────
            radar_fig = gf.grip_factor_radar_fig({
                rn: k["table"] for rn, k in valid_results.items()
            })
            st.plotly_chart(radar_fig, use_container_width=True, theme=None)

            with st.expander("Per-lap grip factors"):
                st.dataframe(
                    _concat_run_tables({
                        rn: k["table"] for rn, k in valid_results.items()
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.warning(f"Grip factors: {exc}")


def _tab_tc(dfs: dict[str, pl.DataFrame]) -> None:
    st.subheader("Traction Control KPIs")
    tc_x_mode = _select_per_lap_axis("tc_axis", default="laps")
    try:
        if len(dfs) == 1:
            run_name, df = next(iter(dfs.items()))
            figs, kpis = tc.tc_figs_kpis(df, x_mode=tc_x_mode)
            for w in kpis.get("warnings", []):
                st.warning(w)
            if not kpis.get("warnings"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Valid laps", str(kpis["valid_laps"]))
                c2.metric("Global SR MAE", _fmt(kpis["mean_global_mae"], ".4f"))
                c3.metric("Global SR bias", _fmt(kpis["mean_global_bias"], "+.4f"))
                c4.metric("Traction coverage", f"{_fmt(kpis['mean_traction_coverage_pct'], '.1f')}%")
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("In target", f"{_fmt(kpis['mean_in_target_pct'], '.1f')}%")
                c6.metric("Overslip", f"{_fmt(kpis['mean_overslip_pct'], '.1f')}%")
                c7.metric("Underslip", f"{_fmt(kpis['mean_underslip_pct'], '.1f')}%")
                c8.metric("ax in target", f"{_fmt(kpis['mean_ax_in_target'], '.2f')} m/s²")
                c9, c10, c11, c12 = st.columns(4)
                for w, col in zip(("FL", "FR", "RL", "RR"), [c9, c10, c11, c12]):
                    col.metric(f"MAE {w}", _fmt(kpis["mae_by_wheel"][w], ".4f"))
            for fig in figs:
                st.plotly_chart(fig, use_container_width=True, theme=None)
            if not kpis.get("warnings"):
                with st.expander("Per-lap data"):
                    st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {run_name: tc.tc_figs_kpis(df, x_mode=tc_x_mode) for run_name, df in dfs.items()}
            for run_name, (_figs, kpis) in run_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(f"{run_name}: {w}")
            _show_summary_table([
                {
                    "Run": run_name,
                    "Valid laps": kpis["valid_laps"],
                    "Global SR MAE": round(kpis["mean_global_mae"], 4),
                    "Global SR bias": round(kpis["mean_global_bias"], 4),
                    "Traction coverage [%]": round(kpis["mean_traction_coverage_pct"], 1),
                    "In target [%]": round(kpis["mean_in_target_pct"], 1),
                    "Overslip [%]": round(kpis["mean_overslip_pct"], 1),
                    "Underslip [%]": round(kpis["mean_underslip_pct"], 1),
                    "ax in target [m/s²]": round(kpis["mean_ax_in_target"], 2),
                }
                for run_name, (_figs, kpis) in run_results.items()
                if not kpis.get("warnings")
            ])
            for fig in _overlay_figures({run_name: figs for run_name, (figs, _kpis) in run_results.items()}):
                st.plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(
                    _concat_run_tables({
                        run_name: kpis["table"]
                        for run_name, (_figs, kpis) in run_results.items()
                        if not kpis.get("warnings")
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.error(f"TC KPIs unavailable: {exc}")


def _tab_tv(dfs: dict[str, pl.DataFrame]) -> None:
    st.subheader("Torque Vectoring KPIs")
    tv_x_mode = _select_per_lap_axis("tv_axis", default="laps")
    try:
        if len(dfs) == 1:
            run_name, df = next(iter(dfs.items()))
            figs, kpis = tv.tv_figs_kpis(df, x_mode=tv_x_mode)
            for w in kpis.get("warnings", []):
                st.warning(w)
            if not kpis.get("warnings"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Valid laps", str(kpis["valid_laps"]))
                c2.metric("Yaw RMSE", _fmt(kpis["mean_yaw_rmse"], ".4f"))
                c3.metric("Yaw bias", _fmt(kpis["mean_yaw_bias"], "+.4f"))
                c4.metric("Corner coverage", f"{_fmt(kpis['mean_corner_coverage_pct'], '.1f')}%")
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Mz RMSE", f"{_fmt(kpis['mean_mz_rmse'], '.2f')} Nm")
                c6.metric("Mz bias", f"{_fmt(kpis['mean_mz_bias'], '+.2f')} Nm")
                c7.metric("FB / FF ratio", _fmt(kpis["mean_ratio"], ".3f"))
                c8.metric("FB share", _fmt(kpis["mean_fb_share"], ".3f"))
            for fig in figs:
                st.plotly_chart(fig, use_container_width=True, theme=None)
            if not kpis.get("warnings"):
                with st.expander("Per-lap data"):
                    st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {run_name: tv.tv_figs_kpis(df, x_mode=tv_x_mode) for run_name, df in dfs.items()}
            for run_name, (_figs, kpis) in run_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(f"{run_name}: {w}")
            _show_summary_table([
                {
                    "Run": run_name,
                    "Valid laps": kpis["valid_laps"],
                    "Yaw RMSE": round(kpis["mean_yaw_rmse"], 4),
                    "Yaw bias": round(kpis["mean_yaw_bias"], 4),
                    "Mz RMSE [Nm]": round(kpis["mean_mz_rmse"], 2),
                    "Mz bias [Nm]": round(kpis["mean_mz_bias"], 2),
                    "FB / FF ratio": round(kpis["mean_ratio"], 3),
                    "FB share": round(kpis["mean_fb_share"], 3),
                }
                for run_name, (_figs, kpis) in run_results.items()
                if not kpis.get("warnings")
            ])
            for fig in _overlay_figures({run_name: figs for run_name, (figs, _kpis) in run_results.items()}):
                st.plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(
                    _concat_run_tables({
                        run_name: kpis["table"]
                        for run_name, (_figs, kpis) in run_results.items()
                        if not kpis.get("warnings")
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.error(f"TV KPIs unavailable: {exc}")


def _tab_driver(
    dfs: dict[str, pl.DataFrame],
    file_signatures: dict[str, FileSignature],
    available_videos: dict[str, str],
) -> None:
    driver_run_tokens = _driver_run_tokens(dfs, file_signatures)
    summaries = _collect_driver_summaries(dfs, driver_run_tokens)

    _render_driver_circuit_map_section(dfs, file_signatures, driver_run_tokens)
    st.divider()

    drv_section = st.segmented_control(
        "Driver section",
        options=["Throttle", "Brake", "Steering", "Video Analysis"],
        default="Throttle",
        required=True,
        key="driver_subsection",
        label_visibility="collapsed",
        width="stretch",
    )

    if drv_section == "Throttle":
        _render_driver_throttle_subtab(dfs, summaries, driver_run_tokens)
    elif drv_section == "Brake":
        _render_driver_brake_subtab(dfs, summaries, driver_run_tokens)
    elif drv_section == "Steering":
        _render_driver_steering_subtab(dfs, summaries, driver_run_tokens)
    elif drv_section == "Video Analysis":
        _render_driver_video_subtab(dfs, file_signatures, available_videos)

    if drv_section in ("Throttle", "Brake", "Steering") and summaries:
        with st.expander("Per-lap data"):
            for run_name, s in summaries.items():
                if s.get("valid_laps", 0) == 0:
                    continue
                if len(dfs) > 1:
                    st.markdown(f"**{run_name}**")
                st.dataframe(s["table"], use_container_width=True, hide_index=True)


def _collect_driver_summaries(
    dfs: dict[str, pl.DataFrame],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> dict[str, dict]:
    """Compute (and cache) per-run driver summaries; surface warnings inline."""
    summaries: dict[str, dict] = {}
    for run_name, df in dfs.items():
        try:
            summaries[run_name] = _driver_summary_cached(
                df,
                next(token for token in driver_run_tokens if token[0] == run_name),
            )
        except Exception as exc:
            st.error(f"Driver KPIs unavailable for `{run_name}`: {exc}")

    for run_name, s in summaries.items():
        for w in s.get("warnings", []):
            st.warning(f"{run_name}: {w}")
    return summaries


def _render_driver_circuit_map_section(
    dfs: dict[str, pl.DataFrame],
    file_signatures: dict[str, FileSignature],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> None:
    """Render the Circuit Map overview that lives above the Driver subtabs."""
    st.subheader("Circuit Map")
    st.caption(
        "■ Green = accelerating (Throttle ≥ 5 %, Brake < 5 %)  "
        "■ Red = braking (Brake ≥ 5 % and dominant)  "
        "■ Grey = coasting (both < 5 %)  "
        "■ White = plausibility (both ≥ 5 % and Throttle > Brake)"
    )

    try:
        # Build per-run entries: run_name → [(run, lap_id, laptime), ...] fastest first
        run_entries: dict[str, list[tuple[str, int, float]]] = {}
        for run_name, df in dfs.items():
            try:
                laps_arr = available_laps(df)
            except Exception:
                continue
            laps_col = df["laps"].to_numpy().astype(float)
            lt_col   = (
                df["laptime"].to_numpy().astype(float)
                if "laptime" in df.columns
                else np.full(len(df), np.nan)
            )
            entries: list[tuple[str, int, float]] = []
            for lap_id in laps_arr.tolist():
                lm = laps_col == float(lap_id)
                lt = (
                    float(np.nanmax(lt_col[lm]))
                    if lm.any() and np.any(np.isfinite(lt_col[lm]))
                    else np.nan
                )
                entries.append((run_name, int(lap_id), lt))
            run_entries[run_name] = sorted(
                entries, key=lambda e: e[2] if np.isfinite(e[2]) else 1e9
            )

        if not run_entries:
            st.warning("No valid laps found.")
        else:
            map_ui_rev = "|".join(
                f"{rn}:{_lap_signature(df)}" for rn, df in sorted(dfs.items())
            )

            selected_map: list[tuple[str, int]] = []
            with st.form(f"drv_map_form_{map_ui_rev}", clear_on_submit=False):
                # Keep lap picking local to this form so every click does not rerun
                # the whole Driver tab and all its KPI figures.
                sel_cols = st.columns(len(run_entries))
                for col, (run_name, entries) in zip(sel_cols, run_entries.items()):
                    color_map = dyn.build_color_map(entries)
                    lap_labels = [
                        f"L{l}  ({t:.2f} s)" if np.isfinite(t) else f"L{l}"
                        for _, l, t in entries
                    ]
                    label_to_key = {
                        lbl: (r, l)
                        for lbl, (r, l, _) in zip(lap_labels, entries)
                    }
                    with col:
                        st.markdown(f"**{run_name}**")
                        color_spans = " &nbsp;".join(
                            f'<span style="color:{color_map.get((r, l), "#ccc")}">■</span>'
                            f' <span style="color:#ccc">{lbl}</span>'
                            for lbl, (r, l, _) in zip(lap_labels, entries)
                        )
                        st.markdown(color_spans, unsafe_allow_html=True)
                        sel_labels = st.multiselect(
                            run_name,
                            options=lap_labels,
                            default=lap_labels,
                            key=f"drv_map_sel_{run_name}_{map_ui_rev}",
                            label_visibility="collapsed",
                        )
                        selected_map.extend(label_to_key[lbl] for lbl in sel_labels)
                st.form_submit_button(
                    "Apply map lap selection",
                    use_container_width=True,
                )

            if not selected_map:
                st.warning("Select at least one lap.")
            else:
                selected_map_key = tuple(sorted(selected_map))
                map_fig = _driver_circuit_map_fig_cached(
                    dfs,
                    driver_run_tokens,
                    selected_map_key,
                )
                st.plotly_chart(map_fig, use_container_width=True, theme=None)

                # Tables: one column per run, aligned below each circuit panel
                if len(run_entries) > 1:
                    tbl_cols = st.columns(len(run_entries))
                    for col, run_name in zip(tbl_cols, run_entries.keys()):
                        run_selected = [(rn, lap) for rn, lap in selected_map if rn == run_name]
                        if not run_selected:
                            continue
                        stats = _driver_circuit_map_stats_cached(
                            {run_name: dfs[run_name]},
                            tuple(
                                token for token in driver_run_tokens
                                if token[0] == run_name
                            ),
                            tuple(sorted(run_selected)),
                        )
                        if not stats.is_empty():
                            with col:
                                st.dataframe(stats, use_container_width=True, hide_index=True)
                else:
                    stats_df = _driver_circuit_map_stats_cached(
                        dfs,
                        driver_run_tokens,
                        selected_map_key,
                    )
                    if not stats_df.is_empty():
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(f"Circuit map unavailable: {exc}")


def _render_driver_throttle_subtab(
    dfs: dict[str, pl.DataFrame],
    summaries: dict[str, dict],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> None:
    st.subheader("Driver Throttle Summary")
    valid_rows = [
        {
            "Run": run_name,
            "Mean throttle [%]": round(s["mean_throttle_pct"], 1),
            "Full throttle / lap [s]": round(s["mean_full_t"], 2),
            "Full throttle [%]": round(s["mean_full_pct"], 1),
            "Off throttle [%]": round(s["mean_off_pct"], 1),
            "Median |dTP/dt| [%/s]": round(s["mean_speed"], 2),
            "Peak lap |dTP/dt| [%/s]": round(s["max_speed"], 1),
        }
        for run_name, s in summaries.items()
        if s.get("valid_laps", 0) > 0
    ]
    if valid_rows:
        st.dataframe(pl.DataFrame(valid_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Throttle Position Histogram")
    try:
        fig = _driver_throttle_histogram_fig_cached(dfs, driver_run_tokens)
        st.plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Throttle histogram unavailable: {exc}")

    st.divider()
    st.subheader("Full Throttle Time per Lap")
    full_x_mode = _select_per_lap_axis("driver_full_axis", default="laps")
    try:
        fig = _driver_full_throttle_time_fig_cached(dfs, driver_run_tokens, full_x_mode)
        st.plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Full throttle time unavailable: {exc}")

    st.divider()
    st.subheader("Throttle Speed per Lap")
    speed_x_mode = _select_per_lap_axis("driver_speed_axis", default="laps")
    try:
        fig = _driver_throttle_speed_fig_cached(dfs, driver_run_tokens, speed_x_mode)
        st.plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Throttle speed unavailable: {exc}")


def _render_driver_brake_subtab(
    dfs: dict[str, pl.DataFrame],
    summaries: dict[str, dict],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> None:
    st.subheader("Driver Brake Summary")
    brake_rows = [
        {
            "Run": run_name,
            "Mean aggressiveness [%/s]": round(s["mean_brake_aggr"], 1),
            "Peak lap aggressiveness [%/s]": round(s["peak_brake_aggr"], 1),
            "Mean release smoothness [%/s]": round(s["mean_brake_release"], 1),
            "Peak lap release smoothness [%/s]": round(s["peak_brake_release"], 1),
        }
        for run_name, s in summaries.items()
        if s.get("valid_laps", 0) > 0
    ]
    if brake_rows:
        st.dataframe(pl.DataFrame(brake_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Braking Effort")
    try:
        fig = _driver_braking_effort_fig_cached(dfs, driver_run_tokens)
        st.plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Braking effort unavailable: {exc}")

    st.divider()
    st.subheader("Braking Aggressiveness per Lap")
    brake_aggr_x_mode = _select_per_lap_axis("driver_brake_aggr_axis", default="laps")
    try:
        fig = _driver_braking_aggressiveness_fig_cached(
            dfs, driver_run_tokens, brake_aggr_x_mode,
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Braking aggressiveness unavailable: {exc}")

    st.divider()
    st.subheader("Brake Release Smoothness per Lap")
    brake_release_x_mode = _select_per_lap_axis("driver_brake_release_axis", default="laps")
    try:
        fig = _driver_brake_release_smoothness_fig_cached(
            dfs, driver_run_tokens, brake_release_x_mode,
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Brake release smoothness unavailable: {exc}")


def _render_driver_steering_subtab(
    dfs: dict[str, pl.DataFrame],
    summaries: dict[str, dict],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> None:
    st.subheader("Driver Steering Summary")
    steering_rows = [
        {
            "Run": run_name,
            "Mean smoothness [rad/s]": round(s["mean_steering_smoothness"], 4),
            "Peak lap smoothness [rad/s]": round(s["max_steering_smoothness"], 4),
            "Mean curvature [1/m]": round(s["mean_curvature"], 5),
            "Peak lap curvature [1/m]": round(s["max_curvature"], 5),
        }
        for run_name, s in summaries.items()
        if s.get("valid_laps", 0) > 0
    ]
    if steering_rows:
        st.dataframe(pl.DataFrame(steering_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Steering Smoothness")
    steering_x_mode = _select_per_lap_axis("driver_steering_axis", default="laps")
    try:
        fig = _driver_steering_smoothness_fig_cached(
            dfs, driver_run_tokens, steering_x_mode,
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Steering smoothness unavailable: {exc}")

    st.divider()
    st.subheader("Corner Curvature per Lap")
    curvature_x_mode = _select_per_lap_axis("driver_curvature_axis", default="laps")
    try:
        fig = _driver_corner_curvature_fig_cached(
            dfs, driver_run_tokens, curvature_x_mode,
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Corner curvature unavailable: {exc}")


def _render_driver_video_subtab(
    dfs: dict[str, pl.DataFrame],
    file_signatures: dict[str, FileSignature],
    available_videos: dict[str, str],
) -> None:
    """Render the synced onboard-video + telemetry component(s) for the run(s)."""
    st.caption(
        "Onboard video synced with Throttle, Brake, Steering and VN_vx. "
        "The video plays continuously across all laps; the charts always show "
        "the lap that the video is currently in. Click on a chart to seek the "
        "video. Use the offset slider if the video does not start exactly at "
        "the same instant as the telemetry."
    )

    items = list(dfs.items())
    if not items:
        st.warning("No runs to display.")
        return

    layout = "horizontal" if len(items) == 1 else "vertical"
    height_px = 760 if len(items) == 1 else 940
    columns = st.columns(len(items)) if len(items) > 1 else [st.container()]

    for col, (run_name, df) in zip(columns, items):
        with col:
            if len(items) > 1:
                st.markdown(f"**{run_name}**")
            try:
                payload = va.build_video_payload(df)
            except KeyError as exc:
                st.error(f"`{run_name}`: {exc}")
                continue
            except Exception as exc:
                st.error(f"`{run_name}`: video payload failed — {exc}")
                continue

            video_url = va.video_url_for_csv(run_name, available_videos)
            if video_url is None:
                st.info(
                    f"No onboard video found for `{run_name}`. "
                    f"Drop a file at `videos/{Path(run_name).stem}.mp4` to enable sync."
                )

            sig = file_signatures.get(run_name)
            sig_token = f"{sig[0]}_{sig[1]}" if sig else "0_0"
            component_id = f"va_{Path(run_name).stem}_{sig_token}"
            html = va.build_video_component_html(
                component_id=component_id,
                video_url=video_url,
                payload=payload,
                height_px=height_px,
                layout=layout,
            )
            components.html(html, height=height_px + 20, scrolling=False)


def _tab_rb(dfs: dict[str, pl.DataFrame]) -> None:
    st.subheader("Regenerative Braking KPIs")
    rb_x_mode = _select_per_lap_axis("rb_axis", default="laps")
    try:
        if len(dfs) == 1:
            run_name, df = next(iter(dfs.items()))
            figs, kpis = rb.rb_figs_kpis(df, x_mode=rb_x_mode)
            for note in kpis.get("notes", []):
                st.info(note)
            for w in kpis.get("warnings", []):
                st.warning(w)
            if not kpis.get("warnings"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Valid laps", str(kpis["valid_laps"]))
                c2.metric("SR MAE", _fmt(kpis["mean_sr_mae"], ".4f"))
                c3.metric("SR bias", _fmt(kpis["mean_sr_bias"], "+.4f"))
                c4.metric("RB active / brake", f"{_fmt(kpis['mean_rb_cover_pct'], '.1f')}%")
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("In target", f"{_fmt(kpis['mean_in_target_pct'], '.1f')}%")
                c6.metric("Overslip", f"{_fmt(kpis['mean_overslip_pct'], '.1f')}%")
                c7.metric("Underslip", f"{_fmt(kpis['mean_underslip_pct'], '.1f')}%")
                c8.metric("RB intensity", _fmt(kpis["mean_intensity_target"], ".3f"))
            for fig in figs:
                st.plotly_chart(fig, use_container_width=True, theme=None)
            if not kpis.get("warnings"):
                with st.expander("Per-lap data"):
                    st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {run_name: rb.rb_figs_kpis(df, x_mode=rb_x_mode) for run_name, df in dfs.items()}
            for run_name, (_figs, kpis) in run_results.items():
                for note in kpis.get("notes", []):
                    st.info(f"{run_name}: {note}")
                for w in kpis.get("warnings", []):
                    st.warning(f"{run_name}: {w}")
            _show_summary_table([
                {
                    "Run": run_name,
                    "Valid laps": kpis["valid_laps"],
                    "SR MAE": round(kpis["mean_sr_mae"], 4),
                    "SR bias": round(kpis["mean_sr_bias"], 4),
                    "RB active / brake [%]": round(kpis["mean_rb_cover_pct"], 1),
                    "In target [%]": round(kpis["mean_in_target_pct"], 1),
                    "Overslip [%]": round(kpis["mean_overslip_pct"], 1),
                    "Underslip [%]": round(kpis["mean_underslip_pct"], 1),
                    "RB intensity": round(kpis["mean_intensity_target"], 3),
                }
                for run_name, (_figs, kpis) in run_results.items()
                if not kpis.get("warnings")
            ])
            for fig in _overlay_figures({run_name: figs for run_name, (figs, _kpis) in run_results.items()}):
                st.plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(
                    _concat_run_tables({
                        run_name: kpis["table"]
                        for run_name, (_figs, kpis) in run_results.items()
                        if not kpis.get("warnings")
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.error(f"RB KPIs unavailable: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="CAT17x — Telemetry",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.markdown("### CAT17x")
    st.sidebar.caption("Formula Student Telemetry")
    st.sidebar.divider()

    csv_files = sorted(p.name for p in DATA_DIR.glob("*.csv"))
    if not csv_files:
        st.error(f"No CSV files found in `{DATA_DIR}`.")
        return

    _autodetect_laps(DATA_DIR)
    available_videos = va.ensure_static_videos(REPO_ROOT, SCRIPT_DIR)

    selected_files = [
        f for f in csv_files
        if st.sidebar.checkbox(f, value=(f == csv_files[0]), key=f"csv_{f}")
    ]
    if not selected_files:
        st.sidebar.warning("Select at least one run.")
        return

    file_signatures = {
        fname: _file_signature(DATA_DIR / fname)
        for fname in selected_files
    }

    raw_dfs: dict[str, pl.DataFrame] = {}
    for fname in selected_files:
        try:
            raw_dfs[fname] = load_run(
                str(DATA_DIR / fname),
                file_signatures[fname],
            )
        except Exception as exc:
            st.sidebar.warning(f"Skipping `{fname}`: {exc}")
    if not raw_dfs:
        st.error("No runs could be loaded.")
        return

    st.sidebar.divider()
    st.sidebar.markdown("### Laps")
    st.sidebar.caption("These lap selections are applied to every tab.")

    dfs: dict[str, pl.DataFrame] = {}
    for fname, raw_df in raw_dfs.items():
        try:
            lap_options = available_laps(raw_df).tolist()
        except Exception as exc:
            st.sidebar.warning(f"`{fname}`: cannot list laps — {exc}")
            continue

        selected_laps = st.sidebar.multiselect(
            fname,
            options=lap_options,
            default=lap_options,
            key=f"laps_{fname}",
            format_func=_format_lap_label,
        )
        if not selected_laps:
            st.sidebar.warning(f"`{fname}`: select at least one lap.")
            continue

        try:
            dfs[fname] = select_laps_df(raw_df, selected_laps)
        except Exception as exc:
            st.sidebar.warning(f"Skipping `{fname}`: {exc}")

    if not dfs:
        st.error("No runs remain after lap selection.")
        return

    section_renderers = {
        "Dynamics": _tab_dynamics,
        "Powertrain": _tab_powertrain,
        "TC": _tab_tc,
        "TV": _tab_tv,
        "RB": _tab_rb,
        "Driver": _tab_driver,
    }

    # `st.tabs` renders every tab on each rerun, which hurts multi-run loading.
    active_section = st.segmented_control(
        "Section",
        options=list(section_renderers),
        default="Dynamics",
        required=True,
        key="dashboard_section",
        label_visibility="collapsed",
        width="stretch",
    )
    if active_section is not None:
        if active_section == "Driver":
            section_renderers[active_section](
                dfs,
                {run_name: file_signatures[run_name] for run_name in dfs},
                available_videos,
            )
        else:
            section_renderers[active_section](dfs)


if __name__ == "__main__":
    main()
