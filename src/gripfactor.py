"""gripfactor.py
----------------
Per-lap grip factors (Buurman / motorsport-data-acquisition methodology),
adapted to a Formula Student 4WD electric car.

Four grip categories — aero grip is intentionally omitted because FS speeds
are too low to isolate a clean downforce effect:

  • Overall   : mean combined |G| over grip-limited samples (braking ∪ corner)
  • Cornering : mean |ay| over radius-detected corners
  • Braking   : mean |ax| over the braking phase
  • Traction  : mean  ax  over corner-exit samples (in a corner with ax > 0)

The phase gating reuses the **same detectors as the rest of the dashboard** so
the grip categories agree with the Braking / Cornering sections:

  • Corner   : path radius ``R = vx²/|ay| < 60 m`` (``utils.radius_corner_mask``,
               the Lap-Analysis curvature logic).
  • Braking  : ``Filtering_VN_ax < −1 m/s²`` AND ``Brake > 5`` (brake pressure).
  • Traction : corner AND ``Filtering_VN_ax > 0`` — corner exit. The lateral
               condition (it must be a corner) excludes straight-line
               acceleration, which on an FS electric car is usually
               inverter/power-limited rather than tyre-grip-limited.

Categories overlap (corner exit is both cornering and traction; trail-braking is
both braking and cornering) — they are independent math channels, not mutually
exclusive labels. Inputs use the pre-filtered acceleration channels
``Filtering_VN_ax`` / ``Filtering_VN_ay`` (m/s²) and convert to G internally so
every metric reads in the same unit as the reference book.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go

from utils import (
    COMPLETE_LAPS_MARKER,
    BRAKE_DECEL_MIN_MPS2,
    BRAKE_PRESS_MIN,
    MU_TIRE,
    driver_color,
    ensure_complete_laps_df,
    exclude_lap0_and_last_lap,
    lap_dist_from_gps,
    make_dark_figure,
    per_lap_axis,
    radius_corner_mask,
    robust_dt,
    unique_laps,
    cols_to_numpy,
)

G = 9.80665  # [m/s²]

# Minimum samples a category needs in a lap before its grip factor is trusted.
MIN_SAMPLES = 25

GRIP_CATEGORIES: tuple[str, ...] = ("Overall", "Cornering", "Braking", "Traction")
GRIP_COLORS: dict[str, str] = {
    "Overall": "#FFD700",
    "Cornering": "#00BFBF",
    "Braking": "#D94F4F",
    "Traction": "#73D973",
}

_REQUIRED_COLS: tuple[str, ...] = (
    "TimeStamp",
    "laps",
    "laptime",
    "Filtering_VN_ax",
    "Filtering_VN_ay",
    "VN_vx",
    "Brake",
)

# A sample counts as "at the limit" when its combined |G| reaches this fraction
# of the run's grip envelope (P95 of combined |G|).
LIMIT_FRAC = 0.90
_UTIL_PHASES: tuple[str, ...] = ("Braking", "Cornering", "Traction")

# Shared height for the three side-by-side Grip Overview figures so the row
# reads as one aligned panel (map · g-g · grip-by-phase).
_OVERVIEW_FIG_HEIGHT = 460


def _from_df(df: pl.DataFrame) -> dict[str, np.ndarray]:
    df = ensure_complete_laps_df(df)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for grip factors: {missing}")
    cols = list(_REQUIRED_COLS)
    if COMPLETE_LAPS_MARKER in df.columns:
        cols.append(COMPLETE_LAPS_MARKER)
    return cols_to_numpy(df, cols)


def _phase_masks(d: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    """Sample-level masks for each grip category, using the shared detectors.

    Operates on the **full, time-ordered** arrays in ``d`` (the corner detector
    smooths over time, so it must not run on a pre-filtered subset). Masks:

      • Corner   = ``utils.radius_corner_mask`` (R < 60 m), the Lap-Analysis logic.
      • Braking  = ``ax < −1 m/s²`` AND ``Brake > 5``.
      • Traction = corner AND ``ax > 0`` (corner exit).
      • Overall  = braking ∪ corner (the grip-limited samples).

    Categories overlap on purpose (corner exit is both cornering and traction).
    """
    ax = d["Filtering_VN_ax"]
    ay = d["Filtering_VN_ay"]
    vx = d["VN_vx"]
    brake = d["Brake"]
    finite = np.isfinite(ax) & np.isfinite(ay)

    corner, _radius_m = radius_corner_mask(vx, ay, dt)
    corner = corner & finite
    braking = finite & (ax < -BRAKE_DECEL_MIN_MPS2) & (brake > BRAKE_PRESS_MIN)
    traction = corner & (ax > 0.0)
    overall = braking | corner
    return {"Overall": overall, "Cornering": corner, "Braking": braking, "Traction": traction}


def _value_for_category(
    category: str,
    ax_g: np.ndarray,
    ay_g: np.ndarray,
) -> np.ndarray:
    """Per-sample value used to compute the mean grip factor (in G)."""
    if category == "Overall":
        return np.sqrt(ax_g**2 + ay_g**2)
    if category == "Cornering":
        return np.abs(ay_g)
    if category == "Braking":
        return np.abs(ax_g)
    if category == "Traction":
        return ax_g
    raise ValueError(f"Unknown grip category: {category}")


def _per_lap_table(
    d: dict[str, np.ndarray],
    min_samples: int = MIN_SAMPLES,
) -> pl.DataFrame:
    """Build the per-lap grip-factor table. Empty if no laps available."""
    d = exclude_lap0_and_last_lap(d)
    laps = d["laps"]
    laptime = d["laptime"]
    ax_g = d["Filtering_VN_ax"] / G
    ay_g = d["Filtering_VN_ay"] / G
    dt = robust_dt(d["TimeStamp"])

    masks = _phase_masks(d, dt)
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
            if n >= min_samples:
                vals = _value_for_category(cat, ax_g[mm], ay_g[mm])
                row[f"GF {cat}"] = round(float(np.nanmean(vals)), 3)
            else:
                row[f"GF {cat}"] = None
        rows.append(row)

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def grip_factor_kpis(
    df: pl.DataFrame,
    min_samples: int = MIN_SAMPLES,
) -> dict:
    """Dashboard KPIs for grip factors. Returns means, fastest lap, table."""
    d = _from_df(df)
    table = _per_lap_table(d, min_samples)
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

    # Phase masks need the full, time-ordered arrays (the corner detector smooths
    # over time); combine with `ok` afterwards so every array indexes the same.
    dt = robust_dt(d["TimeStamp"])
    masks = _phase_masks(d, dt)

    combined_all = np.sqrt(ax_g**2 + ay_g**2)
    combined = combined_all[ok]
    envelope = float(np.percentile(combined, 95))
    if not np.isfinite(envelope) or envelope <= 0.0:
        return {"warnings": ["Could not estimate grip envelope."]}

    near_all = ok & (combined_all >= LIMIT_FRAC * envelope)
    phase_tal = {}
    for cat in _UTIL_PHASES:
        m = masks[cat] & ok
        phase_tal[cat] = float(np.mean(near_all[m]) * 100.0) if m.any() else float("nan")

    return {
        "envelope_g": round(envelope, 3),
        "limit_frac": LIMIT_FRAC,
        "utilization_pct": round(float(np.mean(combined) / envelope * 100.0), 1),
        "time_at_limit_pct": round(float(np.mean(near_all[ok]) * 100.0), 1),
        "phase_time_at_limit_pct": {
            cat: (round(v, 1) if np.isfinite(v) else float("nan")) for cat, v in phase_tal.items()
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
        fig.add_trace(
            go.Bar(
                x=categories,
                y=ys,
                name=run_name.rsplit("/", 1)[-1].removesuffix(".csv"),
                marker=dict(color=driver_color(run_name)),
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            )
        )
        any_bar = True
    if any_bar:
        fig.update_layout(barmode="group")
    return fig


def _phase_for_scatter(d: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    """Assign each sample ONE colour group for the g-g cloud.

    Phase masks overlap (trail-braking is both braking and cornering), so the
    scatter needs a single label per point. Priority braking → traction →
    cornering reads the cloud as: lower half braking, upper-with-lateral
    traction, the rest of the corners pure cornering, everything else straight.
    Returns full-length masks (the caller combines them with its finite filter).
    """
    masks = _phase_masks(d, dt)
    braking = masks["Braking"]
    traction = masks["Traction"] & ~braking
    cornering = masks["Cornering"] & ~braking & ~traction
    finite = np.isfinite(d["Filtering_VN_ax"]) & np.isfinite(d["Filtering_VN_ay"])
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
    grip envelope. Equal-scaled axes so the friction circle reads true.
    """
    fig = make_dark_figure(
        title="g-g diagram",
        xlabel="Lateral ay [g]",
        ylabel="Longitudinal ax [g]",
    )
    warnings: list[str] = []
    runs: dict[str, dict] = {}
    max_r = MU_TIRE
    single = len(dfs) == 1
    phase_colors = {
        "Braking": GRIP_COLORS["Braking"],
        "Cornering": GRIP_COLORS["Cornering"],
        "Traction": GRIP_COLORS["Traction"],
        "Straight": "#7A7A7A",
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
        combined = np.sqrt(ax_g[ok] ** 2 + ay_g[ok] ** 2)
        envelope = float(np.percentile(combined, 95))
        peak = float(np.percentile(combined, 99.5))
        max_r = max(max_r, peak)
        stride = max(1, int(np.ceil(int(ok.sum()) / 9000)))

        if single:
            dt = robust_dt(d["TimeStamp"])
            groups = _phase_for_scatter(d, dt)
            for label, mask in groups.items():
                m = mask & ok
                if not m.any():
                    continue
                fig.add_trace(
                    go.Scattergl(
                        x=ay_g[m][::stride],
                        y=ax_g[m][::stride],
                        mode="markers",
                        marker=dict(color=phase_colors[label], size=3, opacity=0.45),
                        name=label,
                        hovertemplate="ay=%{x:.2f} g<br>ax=%{y:.2f} g<extra>" + label + "</extra>",
                    )
                )
            ring_color = "#EBEBEB"
        else:
            color = driver_color(run_name)
            fig.add_trace(
                go.Scattergl(
                    x=ay_g[ok][::stride],
                    y=ax_g[ok][::stride],
                    mode="markers",
                    marker=dict(color=color, size=3, opacity=0.40),
                    name=run_name.rsplit("/", 1)[-1].removesuffix(".csv"),
                    hovertemplate="ay=%{x:.2f} g<br>ax=%{y:.2f} g<extra></extra>",
                )
            )
            ring_color = color

        cx, cy = _envelope_circle(envelope)
        fig.add_trace(
            go.Scatter(
                x=cx,
                y=cy,
                mode="lines",
                line=dict(color=ring_color, width=2.0, dash="dash"),
                name=f"P95 envelope {envelope:.2f} g",
                showlegend=single,
                hoverinfo="skip",
            )
        )
        runs[run_name] = {
            "envelope_g": round(envelope, 3),
            "peak_combined_g": round(peak, 3),
            "samples": int(ok.sum()),
        }

    lim = max_r * 1.08
    fig.add_hline(y=0.0, line=dict(color="rgba(200,200,200,0.35)", dash="dot", width=1))
    fig.add_vline(x=0.0, line=dict(color="rgba(200,200,200,0.35)", dash="dot", width=1))
    fig.update_layout(
        height=_OVERVIEW_FIG_HEIGHT,
        margin=dict(l=58, r=22, t=48, b=52),
        title=dict(y=0.98, yanchor="top"),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=0.89,
            yanchor="top",
            font=dict(size=11),
        ),
        hovermode="closest",
    )
    fig.update_xaxes(range=[-lim, lim], dtick=1.0)
    fig.update_yaxes(
        range=[-lim, lim],
        domain=[0.0, 0.78],
        dtick=1.0,
        scaleanchor="x",
        scaleratio=1,
    )
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
        fig.add_trace(
            go.Scatter(
                x=x_arr[ok],
                y=ys[ok],
                mode="lines+markers",
                name=cat,
                line=dict(color=GRIP_COLORS[cat], width=1.8),
                marker=dict(size=7, color=GRIP_COLORS[cat]),
                text=[str(int(l)) for l in sorted_laps[ok]],
                hovertemplate=f"{cat} %{{y:.3f}} G (L%{{text}})<extra></extra>",
            )
        )

    if x_mode == "laps":
        fig.update_xaxes(tickvals=sorted(set(laps.tolist())))
    return fig


def combined_g_track_map_fig(df: pl.DataFrame, laps: list[int] | None = None) -> go.Figure:
    """Track map coloured by the **stint-average** combined |G| = √(ax² + ay²) [g].

    The spatial view of *where* the car develops grip: bright = high combined
    acceleration (hard corners, braking zones), dark = low (straights). Instead
    of one lap, every (valid) lap is aligned by normalised track progress and the
    combined |G| is averaged per track-position bin, then drawn over one reference
    lap's GPS path. Averaging cancels single-lap noise (traffic, a missed apex)
    and shows the grip the car *typically* develops at each point of the circuit.

    ``laps`` optionally restricts the average to a set of lap numbers (e.g. the
    grip-factor valid laps); ``None`` uses every lap > 0. GPS from
    ``VN_latitude`` / ``VN_longitude``.
    """
    fig = make_dark_figure(title="Combined-G track map", xlabel="", ylabel="")
    fig.update_layout(height=_OVERVIEW_FIG_HEIGHT, margin=dict(l=10, r=10, t=48, b=10))

    def _msg(text: str) -> go.Figure:
        fig.add_annotation(
            text=text,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#EBEBEB", size=12),
        )
        return fig

    needed = ("laps", "VN_latitude", "VN_longitude", "Filtering_VN_ax", "Filtering_VN_ay")
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return _msg(f"Missing columns: {missing}")

    cols = cols_to_numpy(df, list(needed))
    lap_ids_all = cols["laps"]
    lat = cols["VN_latitude"]
    lng = cols["VN_longitude"]
    combined = np.sqrt((cols["Filtering_VN_ax"] / G) ** 2 + (cols["Filtering_VN_ay"] / G) ** 2)
    dist = lap_dist_from_gps(df)

    finite = np.isfinite(lat) & np.isfinite(lng) & np.isfinite(combined) & np.isfinite(dist)
    if not finite.any():
        return _msg("No valid GPS samples")

    # Laps to average over: caller-supplied set, else every lap > 0.
    if laps is not None:
        want = {int(l) for l in laps}
        use_laps = [l for l in unique_laps(lap_ids_all) if int(l) in want]
    else:
        use_laps = [l for l in unique_laps(lap_ids_all) if l > 0]
    if not use_laps:
        return _msg("No valid laps to average")

    # Bin every lap by normalised track progress (0..1) and average combined |G|
    # per bin across laps. Track the lap with the most clean samples as the
    # reference path to draw.
    n_bins = 400
    sum_g = np.zeros(n_bins)
    cnt = np.zeros(n_bins, dtype=int)
    n_used = 0
    ref_lap: float | None = None
    ref_samples = -1

    def _progress_bins(idx: np.ndarray) -> np.ndarray | None:
        lap_dist = dist[idx]
        lap_len = float(np.nanmax(lap_dist)) if lap_dist.size else 0.0
        if not np.isfinite(lap_len) or lap_len < 10.0:
            return None
        progress = np.clip(lap_dist / lap_len, 0.0, 1.0)
        return np.clip((progress * (n_bins - 1)).astype(int), 0, n_bins - 1)

    for lap_id in use_laps:
        idx = np.where((lap_ids_all == lap_id) & finite)[0]
        if len(idx) < 20:
            continue
        bin_idx = _progress_bins(idx)
        if bin_idx is None:
            continue
        np.add.at(sum_g, bin_idx, combined[idx])
        np.add.at(cnt, bin_idx, 1)
        n_used += 1
        if len(idx) > ref_samples:
            ref_samples, ref_lap = len(idx), lap_id

    if n_used == 0 or ref_lap is None:
        return _msg("Not enough GPS to build the map")

    mean_g = np.full(n_bins, np.nan)
    np.divide(sum_g, cnt, out=mean_g, where=cnt > 0)

    # Draw the reference lap's path, colouring each point by the cross-lap mean
    # combined |G| at its track position.
    ref_idx = np.where((lap_ids_all == ref_lap) & finite)[0]
    ref_bins = _progress_bins(ref_idx)
    ref_color = mean_g[ref_bins]
    drawable = np.isfinite(ref_color)
    ref_idx, ref_color = ref_idx[drawable], ref_color[drawable]

    fig.update_layout(title=f"Combined-G track map  ·  {n_used}-lap average")
    fig.add_trace(
        go.Scattergl(
            x=lng[ref_idx],
            y=lat[ref_idx],
            mode="markers",
            marker=dict(
                size=6,
                color=ref_color,
                colorscale="Turbo",
                cmin=0.0,
                showscale=True,
                colorbar=dict(
                    title=dict(text="|G| [g]", side="right"),
                    orientation="h",
                    thickness=10,
                    len=0.6,
                    x=0.5,
                    xanchor="center",
                    y=-0.02,
                    yanchor="top",
                ),
            ),
            hovertemplate="mean combined |G| = %{marker.color:.2f} g<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, showline=False)
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        showline=False,
        scaleanchor="x",
        scaleratio=1.0,
    )
    return fig


def grip_factor_evolution_multi_fig(
    tables_by_run: dict[str, pl.DataFrame],
    x_mode: str = "laps",
    category: str = "Overall",
) -> go.Figure:
    """Multi-run evolution of **one** grip-factor category, one line per run.

    Compares runs on a single factor (default Overall) to avoid a 4×N overlay;
    the radar carries the all-category cross-run comparison. Run identity via
    ``driver_color``.
    """
    if category not in GRIP_CATEGORIES:
        category = "Overall"
    fig = make_dark_figure(
        title=f"{category} Grip Factor — Evolution",
        xlabel="Lap" if x_mode == "laps" else "Lap time [s]",
        ylabel=f"{category} grip factor [g]",
    )
    if not tables_by_run:
        return fig

    col = f"GF {category}"
    all_laps: set[int] = set()
    for run_name, table in tables_by_run.items():
        if table.is_empty():
            continue
        cols = cols_to_numpy(table, ["Lap", "LapTime [s]", col])
        laps = cols["Lap"].astype(int)
        laptime = cols["LapTime [s]"]
        x_arr, order, _ = per_lap_axis(laps, laptime, x_mode)
        sorted_laps = laps[order]
        all_laps.update(laps.tolist())

        ys = cols[col][order]
        ok = np.isfinite(ys)
        if not ok.any():
            continue
        color = driver_color(run_name)
        fig.add_trace(
            go.Scatter(
                x=x_arr[ok],
                y=ys[ok],
                mode="lines+markers",
                name=run_name.rsplit("/", 1)[-1].removesuffix(".csv"),
                line=dict(color=color, width=1.8),
                marker=dict(size=7, color=color),
                text=[str(int(l)) for l in sorted_laps[ok]],
                hovertemplate=(
                    f"{category} %{{y:.3f}} g (L%{{text}})<extra>"
                    + run_name.rsplit("/", 1)[-1].removesuffix(".csv")
                    + "</extra>"
                ),
            )
        )

    if x_mode == "laps" and all_laps:
        fig.update_xaxes(tickvals=sorted(all_laps))
    return fig


def grip_factor_bar_fig(means: dict[str, float]) -> go.Figure:
    """Single-run bar of the four grip factors (in g), one bar per category.

    The at-a-glance breakdown of which grip dimension (braking / cornering /
    traction) the car+driver develop most, against Overall.
    """
    fig = make_dark_figure(
        title="Grip by phase",
        xlabel="",
        ylabel="Grip factor [g]",
    )
    cats = list(GRIP_CATEGORIES)
    ys = [float(means.get(c, float("nan"))) for c in cats]
    fig.add_trace(
        go.Bar(
            x=cats,
            y=ys,
            marker=dict(color=[GRIP_COLORS[c] for c in cats]),
            text=[f"{v:.2f}" if np.isfinite(v) else "" for v in ys],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}: %{y:.2f} g<extra></extra>",
        )
    )
    ymax = max((v for v in ys if np.isfinite(v)), default=1.0)
    fig.update_layout(height=_OVERVIEW_FIG_HEIGHT, margin=dict(l=58, r=22, t=48, b=52))
    fig.update_yaxes(range=[0, ymax * 1.18])
    return fig


def grip_factor_radar_fig(
    tables_by_run: dict[str, pl.DataFrame],
) -> go.Figure:
    """Radar comparing the average grip factor per category across runs.

    Book Fig. 8.3 — one filled polygon per CSV.
    """
    fig = make_dark_figure(title="Grip by phase · radar")
    fig.update_layout(
        height=_OVERVIEW_FIG_HEIGHT,
        margin=dict(l=50, r=50, t=76, b=40),
        polar=dict(
            bgcolor="#141417",
            domain=dict(y=[0.0, 0.84]),
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
                angle=90,
                tickformat=".2f",
            ),
        ),
        showlegend=True,
    )
    fig.update_layout(legend=dict(y=0.96, yanchor="top"))

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
        fig.add_trace(
            go.Scatterpolar(
                r=r,
                theta=theta,
                fill="toself",
                name=run_name,
                line=dict(color=color, width=1.8),
                opacity=0.55,
            )
        )
    return fig
