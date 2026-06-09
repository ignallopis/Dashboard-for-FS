"""gripfactor.py
----------------
Per-lap grip factors (Buurman / motorsport-data-acquisition methodology),
adapted to a Formula Student 4WD electric car.

Four grip categories — aero grip is intentionally omitted because FS speeds
are too low to isolate a clean downforce effect:

  • Overall   : mean combined |G| in grip-limited samples
  • Cornering : mean |ay| when lateral G exceeds the cornering threshold
  • Braking   : mean |ax| when longitudinal G is below the braking threshold
  • Traction  : mean  ax  when ax > 0 and lateral G is still high enough

These are independent math channels, not mutually exclusive phase labels.
That matches the original methodology better: one sample can contribute to
both the cornering and traction grip factors during corner exit, or to both
cornering and braking during trail-braking.

For Formula Student, the traction channel intentionally keeps a lateral-G
condition. That excludes most straight-line acceleration, which is often
limited by inverter/current/power rather than tyre grip.

Inputs use the pre-filtered acceleration channels ``Filtering_VN_ax`` and
``Filtering_VN_ay`` (m/s²) and convert them to G internally so every metric is
reported in the same unit as the reference book.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import (
    COMPLETE_LAPS_MARKER,
    driver_color,
    ensure_complete_laps_df,
    exclude_lap0_and_last_lap,
    make_dark_figure,
    per_lap_axis,
    unique_laps,
    cols_to_numpy,
)

G = 9.80665  # [m/s²]

GRIP_CATEGORIES: tuple[str, ...] = ("Overall", "Cornering", "Braking", "Traction")
GRIP_COLORS: dict[str, str] = {
    "Overall":   "#FFD700",
    "Cornering": "#00BFBF",
    "Braking":   "#D94F4F",
    "Traction":  "#73D973",
}

_REQUIRED_COLS: tuple[str, ...] = (
    "TimeStamp", "laps", "laptime",
    "Filtering_VN_ax", "Filtering_VN_ay",
)

# A sample counts as "at the limit" when its combined |G| reaches this fraction
# of the run's grip envelope (P95 of combined |G|).
LIMIT_FRAC = 0.90
_UTIL_PHASES: tuple[str, ...] = ("Braking", "Cornering", "Traction")


@dataclass(frozen=True)
class GripThresholds:
    """Boundary conditions for the grip-factor math channels.

    Channels are independent, so thresholds should be easy to interpret on the
    raw accelerations rather than on derived sector geometry.
    """
    overall_combined_g: float = 0.80
    cornering_ay_g:     float = 0.50
    braking_ax_g:       float = 0.60
    traction_ax_g:      float = 0.15
    traction_ay_g:      float = 0.35
    min_samples:        int   = 25


def estimate_thresholds(df: pl.DataFrame) -> GripThresholds:
    """Estimate boundary conditions from the run's G distribution.

    Uses P95 of each acceleration axis as the car's approximate capability on
    this circuit, then scales the thresholds down to FS-appropriate trigger
    levels. Traction keeps a comparatively low positive-ax threshold because we
    still require simultaneous lateral load, which isolates corner-exit usage.
    """
    try:
        d = _from_df(df)
        d = exclude_lap0_and_last_lap(d)
    except (KeyError, ValueError):
        return GripThresholds()

    ax_g = d["Filtering_VN_ax"] / G
    ay_g = d["Filtering_VN_ay"] / G
    ok = np.isfinite(ax_g) & np.isfinite(ay_g)
    if ok.sum() < 200:
        return GripThresholds()

    ax_ok = ax_g[ok]
    ay_ok = ay_g[ok]
    combined = np.sqrt(ax_ok ** 2 + ay_ok ** 2)

    peak_combined = float(np.percentile(combined, 95))
    peak_lat      = float(np.percentile(np.abs(ay_ok), 95))

    decel = ax_ok[ax_ok < -0.05]
    peak_brk = (
        float(np.percentile(np.abs(decel), 95))
        if len(decel) > 50 else peak_combined
    )

    accel = ax_ok[ax_ok > 0.05]
    peak_acc = (
        float(np.percentile(accel, 95))
        if len(accel) > 50 else peak_combined * 0.4
    )

    return GripThresholds(
        overall_combined_g=round(float(np.clip(0.70 * peak_combined, 0.60, 1.30)), 2),
        cornering_ay_g=round(float(np.clip(0.45 * peak_lat, 0.35, 0.90)), 2),
        braking_ax_g=round(float(np.clip(0.60 * peak_brk, 0.35, 0.90)), 2),
        traction_ax_g=round(float(np.clip(0.35 * peak_acc, 0.10, 0.40)), 2),
        traction_ay_g=round(float(np.clip(0.30 * peak_lat, 0.25, 0.70)), 2),
    )


def _from_df(df: pl.DataFrame) -> dict[str, np.ndarray]:
    df = ensure_complete_laps_df(df)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for grip factors: {missing}")
    cols = list(_REQUIRED_COLS)
    if COMPLETE_LAPS_MARKER in df.columns:
        cols.append(COMPLETE_LAPS_MARKER)
    return cols_to_numpy(df, cols)


def _phase_masks(
    ax_g: np.ndarray,
    ay_g: np.ndarray,
    t: GripThresholds,
) -> dict[str, np.ndarray]:
    """Sample-level masks for each grip category. ``ax_g``/``ay_g`` in G."""
    finite = np.isfinite(ax_g) & np.isfinite(ay_g)
    combined = np.sqrt(ax_g ** 2 + ay_g ** 2)

    return {
        "Overall":   finite & (combined >= t.overall_combined_g),
        "Cornering": finite & (np.abs(ay_g) >= t.cornering_ay_g),
        "Braking":   finite & (ax_g <= -t.braking_ax_g),
        "Traction":  finite
                     & (ax_g >= t.traction_ax_g)
                     & (np.abs(ay_g) >= t.traction_ay_g),
    }


def _value_for_category(
    category: str,
    ax_g: np.ndarray,
    ay_g: np.ndarray,
) -> np.ndarray:
    """Per-sample value used to compute the mean grip factor (in G)."""
    if category == "Overall":
        return np.sqrt(ax_g ** 2 + ay_g ** 2)
    if category == "Cornering":
        return np.abs(ay_g)
    if category == "Braking":
        return np.abs(ax_g)
    if category == "Traction":
        return ax_g
    raise ValueError(f"Unknown grip category: {category}")


def _per_lap_table(
    d: dict[str, np.ndarray],
    t: GripThresholds,
) -> pl.DataFrame:
    """Build the per-lap grip-factor table. Empty if no laps available."""
    d = exclude_lap0_and_last_lap(d)
    laps    = d["laps"]
    laptime = d["laptime"]
    ax_g    = d["Filtering_VN_ax"] / G
    ay_g    = d["Filtering_VN_ay"] / G

    masks = _phase_masks(ax_g, ay_g, t)
    lap_list = unique_laps(laps).astype(int)
    if lap_list.size == 0:
        return pl.DataFrame()

    rows: list[dict[str, object]] = []
    for lap in lap_list:
        lm = laps == lap
        if not lm.any():
            continue
        row: dict[str, object] = {
            "Lap": int(lap),
            "LapTime [s]": round(float(laptime[lm].max()), 3),
        }
        for cat in GRIP_CATEGORIES:
            mm = lm & masks[cat]
            n = int(mm.sum())
            row[f"{cat} samples"] = n
            if n >= t.min_samples:
                vals = _value_for_category(cat, ax_g[mm], ay_g[mm])
                row[f"GF {cat}"] = round(float(np.nanmean(vals)), 3)
            else:
                row[f"GF {cat}"] = None
        rows.append(row)

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def grip_factor_kpis(
    df: pl.DataFrame,
    thresholds: GripThresholds | None = None,
) -> dict:
    """Dashboard KPIs for grip factors. Returns means, fastest lap, table."""
    t = thresholds or GripThresholds()
    d = _from_df(df)
    table = _per_lap_table(d, t)
    if table.is_empty():
        return {
            "valid_laps": 0,
            "means": {c: float("nan") for c in GRIP_CATEGORIES},
            "fastest_lap": None,
            "fastest_lt": float("nan"),
            "table": pl.DataFrame(),
            "warnings": ["No valid laps for grip factor computation."],
        }

    means: dict[str, float] = {}
    for cat in GRIP_CATEGORIES:
        col = table[f"GF {cat}"].drop_nulls()
        means[cat] = float(col.mean()) if len(col) > 0 else float("nan")

    laps_with_overall = table.filter(
        pl.col("GF Overall").is_not_null() & pl.col("LapTime [s]").is_not_null()
    )
    if laps_with_overall.is_empty():
        fastest_lap: int | None = None
        fastest_lt = float("nan")
    else:
        fastest = laps_with_overall.sort("LapTime [s]").row(0, named=True)
        fastest_lap = int(fastest["Lap"])
        fastest_lt = float(fastest["LapTime [s]"])

    return {
        "valid_laps": int(table["GF Overall"].drop_nulls().len()),
        "means": means,
        "fastest_lap": fastest_lap,
        "fastest_lt": fastest_lt,
        "table": table,
        "warnings": [],
    }


def grip_utilization_kpis(df: pl.DataFrame) -> dict:
    """Friction-circle utilisation: how hard the driver works the grip envelope.

    ``envelope_g`` = P95 of combined |G| over valid laps (the car's grip ceiling
    on this circuit). Returns:
      • utilization_pct       : mean combined |G| as a % of the envelope
      • time_at_limit_pct     : % of samples with combined |G| >= LIMIT_FRAC·env
      • phase_time_at_limit_pct: the same, split by Braking / Cornering / Traction
    Self-normalising (no speed channel needed). A driver that "leaves grip on the
    table" shows lower utilisation and less time at the limit.
    """
    try:
        d = _from_df(df)
        d = exclude_lap0_and_last_lap(d)
    except (KeyError, ValueError) as exc:
        return {"warnings": [str(exc)]}

    ax_g = d["Filtering_VN_ax"] / G
    ay_g = d["Filtering_VN_ay"] / G
    ok = np.isfinite(ax_g) & np.isfinite(ay_g)
    if int(ok.sum()) < 200:
        return {"warnings": ["Not enough samples for grip utilisation."]}

    ax_g = ax_g[ok]
    ay_g = ay_g[ok]
    combined = np.sqrt(ax_g ** 2 + ay_g ** 2)
    envelope = float(np.percentile(combined, 95))
    if not np.isfinite(envelope) or envelope <= 0.0:
        return {"warnings": ["Could not estimate grip envelope."]}

    near = combined >= LIMIT_FRAC * envelope
    t = estimate_thresholds(df)
    masks = _phase_masks(ax_g, ay_g, t)
    phase_tal = {
        cat: (float(np.mean(near[masks[cat]]) * 100.0) if masks[cat].any() else float("nan"))
        for cat in _UTIL_PHASES
    }

    return {
        "envelope_g": round(envelope, 3),
        "limit_frac": LIMIT_FRAC,
        "utilization_pct": round(float(np.mean(combined) / envelope * 100.0), 1),
        "time_at_limit_pct": round(float(np.mean(near) * 100.0), 1),
        "phase_time_at_limit_pct": {
            cat: (round(v, 1) if np.isfinite(v) else float("nan"))
            for cat, v in phase_tal.items()
        },
        "samples": int(ok.sum()),
        "warnings": [],
    }


def grip_utilization_fig(dfs: dict[str, pl.DataFrame]) -> go.Figure:
    """Grouped bars of time-at-limit % (Overall + per phase) per driver."""
    categories = ["Overall", *_UTIL_PHASES]
    fig = make_dark_figure(
        title=f"Time at the Limit  ·  combined |G| ≥ {int(LIMIT_FRAC * 100)}% of grip envelope",
        xlabel="",
        ylabel="Time at the limit [%]",
    )
    any_bar = False
    for run_name, df in dfs.items():
        k = grip_utilization_kpis(df)
        if k.get("warnings"):
            continue
        phase = k["phase_time_at_limit_pct"]
        ys = [k["time_at_limit_pct"], *(phase.get(cat, float("nan")) for cat in _UTIL_PHASES)]
        fig.add_trace(go.Bar(
            x=categories,
            y=ys,
            name=run_name.rsplit("/", 1)[-1].removesuffix(".csv"),
            marker=dict(color=driver_color(run_name)),
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        any_bar = True
    if any_bar:
        fig.update_layout(barmode="group")
    return fig


# Design-target tyre friction, used only to draw a reference circle on the g-g
# overview. Mirrors dynamics.MU_TIRE; kept local to avoid a cross-module import.
_DESIGN_MU_CIRCLE = 1.70


def _phase_for_scatter(
    ax_g: np.ndarray, ay_g: np.ndarray, t: GripThresholds,
) -> dict[str, np.ndarray]:
    """Assign each sample ONE colour group for the g-g cloud.

    Phase masks overlap (trail-braking is both braking and cornering), so the
    scatter needs a single label per point. Priority longitudinal-first reads the
    cloud as: lower half braking, upper-with-lateral traction, pure lateral
    cornering, the rest straight-line.
    """
    masks = _phase_masks(ax_g, ay_g, t)
    braking = masks["Braking"]
    traction = masks["Traction"] & ~braking
    cornering = masks["Cornering"] & ~braking & ~traction
    finite = np.isfinite(ax_g) & np.isfinite(ay_g)
    straight = finite & ~(braking | traction | cornering)
    return {"Braking": braking, "Traction": traction, "Cornering": cornering, "Straight": straight}


def _envelope_circle(radius: float, n: int = 181) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) of a circle of the given radius, centred at the origin."""
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    return radius * np.cos(theta), radius * np.sin(theta)


def gg_scatter_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]:
    """g-g overview: lateral vs longitudinal acceleration cloud.

    X = ay [g] (signed), Y = ax [g] (signed: +accel / −braking). A single run is
    coloured by phase (braking / cornering / traction / straight); multiple runs
    are coloured by run identity. A dashed ring marks each run's P95 combined-|G|
    grip envelope; the dotted circle is the design-μ reference. Equal-scaled axes
    so the friction circle reads true.
    """
    fig = make_dark_figure(
        title="g-g Diagram  ·  Lateral vs Longitudinal Acceleration",
        xlabel="Lateral acceleration ay [g]",
        ylabel="Longitudinal acceleration ax [g]  (+accel / −braking)",
    )
    warnings: list[str] = []
    runs: dict[str, dict] = {}
    max_r = _DESIGN_MU_CIRCLE
    single = len(dfs) == 1
    phase_colors = {
        "Braking":   GRIP_COLORS["Braking"],
        "Cornering": GRIP_COLORS["Cornering"],
        "Traction":  GRIP_COLORS["Traction"],
        "Straight":  "#7A7A7A",
    }

    for run_name, df in dfs.items():
        try:
            d = exclude_lap0_and_last_lap(_from_df(df))
        except (KeyError, ValueError) as exc:
            warnings.append(f"{run_name}: {exc}")
            continue
        ax_g = d["Filtering_VN_ax"] / G
        ay_g = d["Filtering_VN_ay"] / G
        ok = np.isfinite(ax_g) & np.isfinite(ay_g)
        if int(ok.sum()) < 100:
            warnings.append(f"{run_name}: not enough samples for g-g.")
            continue
        ax_g = ax_g[ok]
        ay_g = ay_g[ok]
        combined = np.sqrt(ax_g ** 2 + ay_g ** 2)
        envelope = float(np.percentile(combined, 95))
        peak = float(np.percentile(combined, 99.5))
        max_r = max(max_r, peak)
        stride = max(1, int(np.ceil(ax_g.size / 9000)))

        if single:
            groups = _phase_for_scatter(ax_g, ay_g, estimate_thresholds(df))
            for label, mask in groups.items():
                if not mask.any():
                    continue
                fig.add_trace(go.Scattergl(
                    x=ay_g[mask][::stride], y=ax_g[mask][::stride],
                    mode="markers",
                    marker=dict(color=phase_colors[label], size=3, opacity=0.45),
                    name=label,
                    hovertemplate="ay=%{x:.2f} g<br>ax=%{y:.2f} g<extra>" + label + "</extra>",
                ))
            ring_color = "#EBEBEB"
        else:
            color = driver_color(run_name)
            fig.add_trace(go.Scattergl(
                x=ay_g[::stride], y=ax_g[::stride],
                mode="markers",
                marker=dict(color=color, size=3, opacity=0.40),
                name=run_name.rsplit("/", 1)[-1].removesuffix(".csv"),
                hovertemplate="ay=%{x:.2f} g<br>ax=%{y:.2f} g<extra></extra>",
            ))
            ring_color = color

        cx, cy = _envelope_circle(envelope)
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="lines",
            line=dict(color=ring_color, width=2.0, dash="dash"),
            name=f"P95 envelope {envelope:.2f} g",
            showlegend=single,
            hoverinfo="skip",
        ))
        runs[run_name] = {
            "envelope_g": round(envelope, 3),
            "peak_combined_g": round(peak, 3),
            "samples": int(ok.sum()),
        }

    mx, my = _envelope_circle(_DESIGN_MU_CIRCLE)
    fig.add_trace(go.Scatter(
        x=mx, y=my, mode="lines",
        line=dict(color="rgba(235,235,235,0.45)", width=1.4, dash="dot"),
        name=f"Design μ={_DESIGN_MU_CIRCLE:.2f} g",
        hoverinfo="skip",
    ))
    lim = max_r * 1.08
    fig.add_hline(y=0.0, line=dict(color="rgba(200,200,200,0.35)", dash="dot", width=1))
    fig.add_vline(x=0.0, line=dict(color="rgba(200,200,200,0.35)", dash="dot", width=1))
    fig.update_layout(height=640, margin=dict(l=70, r=40, t=55, b=65), hovermode="closest")
    fig.update_xaxes(range=[-lim, lim])
    fig.update_yaxes(range=[-lim, lim], scaleanchor="x", scaleratio=1)
    return fig, {"runs": runs, "warnings": warnings}


def grip_factor_evolution_fig(
    table: pl.DataFrame,
    x_mode: str = "laps",
) -> go.Figure:
    """Line chart of each grip category vs lap (or lap time). Book Fig. 8.2."""
    fig = make_dark_figure(
        title="Grip Factor Evolution",
        xlabel="Lap" if x_mode == "laps" else "Lap time [s]",
        ylabel="Grip factor [G]",
    )
    if table.is_empty():
        return fig

    cols = cols_to_numpy(table, ["Lap", "LapTime [s]", *[f"GF {cat}" for cat in GRIP_CATEGORIES]])
    laps = cols["Lap"].astype(int)
    laptime = cols["LapTime [s]"]
    x_arr, order, _xlabel = per_lap_axis(laps, laptime, x_mode)
    sorted_laps = laps[order]

    for cat in GRIP_CATEGORIES:
        ys = cols[f"GF {cat}"][order]
        ok = np.isfinite(ys)
        if not ok.any():
            continue
        fig.add_trace(go.Scatter(
            x=x_arr[ok],
            y=ys[ok],
            mode="lines+markers",
            name=cat,
            line=dict(color=GRIP_COLORS[cat], width=1.8),
            marker=dict(size=7, color=GRIP_COLORS[cat]),
            text=[str(int(l)) for l in sorted_laps[ok]],
            hovertemplate=f"{cat} %{{y:.3f}} G (L%{{text}})<extra></extra>",
        ))

    if x_mode == "laps":
        fig.update_xaxes(tickvals=sorted(set(laps.tolist())))
    return fig


def grip_factor_track_maps_fig(
    df: pl.DataFrame,
    lap: int,
    thresholds: GripThresholds | None = None,
) -> go.Figure:
    """Four mini track maps (1 row × 4 cols), one per grip category.

    Base layer is the full lap in grey; samples that fall inside each category's
    mask are overlaid in the category colour. Per-circuit visual sanity check
    while tuning the boundary conditions.
    """
    t = thresholds or GripThresholds()

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=list(GRIP_CATEGORIES),
        horizontal_spacing=0.03,
    )
    fig.update_layout(
        paper_bgcolor="#141417",
        plot_bgcolor="#141417",
        font=dict(color="#EBEBEB", size=11),
        margin=dict(l=10, r=10, t=50, b=20),
        height=320,
        showlegend=False,
    )

    needed = ("laps", "VN_latitude", "VN_longitude",
              "Filtering_VN_ax", "Filtering_VN_ay")
    missing = [c for c in needed if c not in df.columns]
    if missing:
        fig.add_annotation(
            text=f"Missing columns: {missing}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#EBEBEB", size=12),
        )
        return fig

    cols = cols_to_numpy(df, [
        "laps",
        "VN_latitude",
        "VN_longitude",
        "Filtering_VN_ax",
        "Filtering_VN_ay",
    ])
    laps_arr = cols["laps"]
    lap_mask = laps_arr == float(lap)
    if not lap_mask.any():
        fig.add_annotation(
            text=f"Lap {lap} not found",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#EBEBEB", size=12),
        )
        return fig

    lat = cols["VN_latitude"]
    lng = cols["VN_longitude"]
    ax_g = cols["Filtering_VN_ax"] / G
    ay_g = cols["Filtering_VN_ay"] / G

    valid = lap_mask & np.isfinite(lat) & np.isfinite(lng)
    if not valid.any():
        fig.add_annotation(
            text=f"Lap {lap}: no valid GPS samples",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#EBEBEB", size=12),
        )
        return fig

    masks = _phase_masks(ax_g, ay_g, t)
    n_lap = int(valid.sum())
    lat_v = lat[valid]
    lng_v = lng[valid]

    for col_idx, cat in enumerate(GRIP_CATEGORIES, start=1):
        fig.add_trace(
            go.Scattergl(
                x=lng_v, y=lat_v, mode="markers",
                marker=dict(size=2, color="rgba(150,150,150,0.30)"),
                hoverinfo="skip", showlegend=False,
            ),
            row=1, col=col_idx,
        )
        m = valid & masks[cat]
        n = int(m.sum())
        if n > 0:
            fig.add_trace(
                go.Scattergl(
                    x=lng[m], y=lat[m], mode="markers",
                    marker=dict(size=4, color=GRIP_COLORS[cat], opacity=0.85),
                    hoverinfo="skip", showlegend=False,
                ),
                row=1, col=col_idx,
            )
        pct = 100.0 * n / max(n_lap, 1)
        fig.layout.annotations[col_idx - 1].text = (
            f"{cat} — {n} pts ({pct:.1f}%)"
        )

    for i in range(1, 5):
        sfx = "" if i == 1 else str(i)
        fig.update_layout(**{
            f"xaxis{sfx}": dict(showgrid=False, zeroline=False,
                                showticklabels=False, showline=False),
            f"yaxis{sfx}": dict(showgrid=False, zeroline=False,
                                showticklabels=False, showline=False,
                                scaleanchor=f"x{sfx}", scaleratio=1.0),
        })
    for ann in fig.layout.annotations:
        ann.font.color = "#EBEBEB"
    return fig


_RUN_DASHES: tuple[str, ...] = ("solid", "dash", "dot", "dashdot", "longdash")


def grip_factor_evolution_multi_fig(
    tables_by_run: dict[str, pl.DataFrame],
    x_mode: str = "laps",
) -> go.Figure:
    """Multi-run evolution: **category colours**, run distinguished by dash.

    Unlike the generic ``_overlay_figures`` helper (which assigns one colour per
    run), this keeps Overall = gold, Cornering = cyan, etc. and uses dash
    patterns to tell runs apart.
    """
    fig = make_dark_figure(
        title="Grip Factor Evolution",
        xlabel="Lap" if x_mode == "laps" else "Lap time [s]",
        ylabel="Grip factor [G]",
    )
    if not tables_by_run:
        return fig

    all_laps: set[int] = set()
    for run_idx, (run_name, table) in enumerate(tables_by_run.items()):
        if table.is_empty():
            continue
        dash = _RUN_DASHES[run_idx % len(_RUN_DASHES)]
        cols = cols_to_numpy(table, ["Lap", "LapTime [s]", *[f"GF {cat}" for cat in GRIP_CATEGORIES]])
        laps = cols["Lap"].astype(int)
        laptime = cols["LapTime [s]"]
        x_arr, order, _ = per_lap_axis(laps, laptime, x_mode)
        sorted_laps = laps[order]
        all_laps.update(laps.tolist())

        for cat in GRIP_CATEGORIES:
            ys = cols[f"GF {cat}"][order]
            ok = np.isfinite(ys)
            if not ok.any():
                continue
            fig.add_trace(go.Scatter(
                x=x_arr[ok],
                y=ys[ok],
                mode="lines+markers",
                name=f"{run_name} · {cat}",
                legendgroup=cat,
                line=dict(color=GRIP_COLORS[cat], width=1.8, dash=dash),
                marker=dict(size=7, color=GRIP_COLORS[cat]),
                text=[str(int(l)) for l in sorted_laps[ok]],
                hovertemplate=(
                    f"{run_name} · {cat} "
                    "%{y:.3f} G (L%{text})<extra></extra>"
                ),
            ))

    if x_mode == "laps" and all_laps:
        fig.update_xaxes(tickvals=sorted(all_laps))
    return fig


def grip_factor_radar_fig(
    tables_by_run: dict[str, pl.DataFrame],
) -> go.Figure:
    """Radar comparing the average grip factor per category across runs.

    Book Fig. 8.3 — one filled polygon per CSV.
    """
    fig = make_dark_figure(title="Grip Factor (Average) — Radar")
    fig.update_layout(
        polar=dict(
            bgcolor="#141417",
            angularaxis=dict(
                color="#E5E5E5",
                gridcolor="rgba(128,128,128,0.25)",
                linecolor="rgba(128,128,128,0.4)",
                rotation=90,
                direction="clockwise",
            ),
            radialaxis=dict(
                color="#E5E5E5",
                gridcolor="rgba(128,128,128,0.25)",
                linecolor="rgba(128,128,128,0.4)",
                tickformat=".2f",
            ),
        ),
        showlegend=True,
    )

    theta = list(GRIP_CATEGORIES) + [GRIP_CATEGORIES[0]]
    for run_name, table in tables_by_run.items():
        if table.is_empty():
            continue
        means: list[float] = []
        for cat in GRIP_CATEGORIES:
            col = table[f"GF {cat}"].drop_nulls()
            means.append(float(col.mean()) if len(col) > 0 else 0.0)
        r = means + [means[0]]
        color = driver_color(run_name)
        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta,
            fill="toself",
            name=run_name,
            line=dict(color=color, width=1.8),
            opacity=0.55,
        ))
    return fig
