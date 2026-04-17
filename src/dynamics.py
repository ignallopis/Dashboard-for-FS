"""dynamics.py
------------
Vehicle dynamics KPIs:
  1. Slip angle efficiency                               (ay / |SA| per wheel)
  2. Understeer angle evolution                          (delta_actual - delta_ideal)

Usage:
    python src/dynamics.py                    — standalone CLI (loads from CSV_PATH)
    slip_angle_efficiency_figs(df)            — dashboard (takes polars DataFrame)
    understeer_angle_fig(df)                  — dashboard (takes polars DataFrame)
"""
from __future__ import annotations
import numpy as np
import polars as pl
import plotly.graph_objects as go

from utils import (
    COMPLETE_LAPS_MARKER,
    make_dark_figure, add_lap_scatter, add_trend_line, add_zero_line,
    ensure_complete_laps_df,
    keep_min_duration_segments, exclude_lap0_and_last_lap,
    robust_dt, unique_laps, phase_masks_for_map, per_lap_axis,
    WHEEL_COLORS, WHEEL_SYMBOLS,
)

CSV_PATH = 'data/run4_2025-08-24.csv'

# ── Shared filter parameters ──────────────────────────────────────────────────
AY_THRESHOLD        = 2.0    # [m/s²] min |ay| to classify as cornering
STEERING_THRESHOLD  = 0.05   # [rad]  min |steering| for cornering
MIN_SPEED           = 3.0    # [m/s]  min vehicle speed
MIN_CORNER_DURATION = 0.20   # [s]    min cornering event length
MIN_CORNER_SAMPLES  = 50     # per lap

WHEELBASE_EQ        = 1.53   # [m]  equivalent wheelbase for bicycle model
MIN_SA_MEAN_DEG     = 1.0    # [deg] slip angles below this are ignored
MAX_SA_MEAN_DEG     = 15.0   # [deg]
MAX_SA_EFF          = 5.0    # [m/s²/deg] sanity cap on efficiency

_SA_COLS = ['TimeStamp', 'laps', 'laptime',
            'Filtering_VN_ay', 'Steering', 'VN_vx',
            'Est_SAFL', 'Est_SAFR', 'Est_SARL', 'Est_SARR']

_US_COLS_COG = ['TimeStamp', 'laps', 'laptime',
                'Steering', 'Filtering_VN_ay', 'Est_vxCOG']
_US_COLS_VX  = ['TimeStamp', 'laps', 'laptime',
                'Steering', 'Filtering_VN_ay', 'VN_vx']


# ── Data loading ──────────────────────────────────────────────────────────────

def _load(columns: list[str]) -> dict[str, np.ndarray]:
    df = pl.read_csv(CSV_PATH, columns=columns)
    return {c: df[c].to_numpy().astype(float) for c in columns}


def _from_df(df: pl.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    df = ensure_complete_laps_df(df)
    cols = list(columns)
    if COMPLETE_LAPS_MARKER in df.columns and COMPLETE_LAPS_MARKER not in cols:
        cols.append(COMPLETE_LAPS_MARKER)
    return {c: df[c].to_numpy().astype(float) for c in cols}


def _base_validity(*arrays: np.ndarray) -> np.ndarray:
    return np.all(np.stack([np.isfinite(a) for a in arrays], axis=1), axis=1)


def _display_laps(lap_ids: np.ndarray) -> np.ndarray:
    """Return lap IDs as displayed in dashboard figures and tables."""
    return np.asarray(lap_ids, dtype=int)


# ── 1. Slip angle efficiency ──────────────────────────────────────────────────

def _compute_slip_angle(
    d: dict[str, np.ndarray],
    x_mode: str = 'laptime',
) -> tuple[go.Figure, go.Figure, np.ndarray, dict, np.ndarray, np.ndarray, dict, np.ndarray]:
    """Core computation for slip angle efficiency.

    Returns (fig1, fig2, lap_list, sa_eff, lt_val, n_samps, sa_mean, ay_mean).
    """
    d['time'] = d['TimeStamp'] - d['TimeStamp'][0]

    valid = _base_validity(*d.values())
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)

    dt       = robust_dt(d['time'])
    ay       = d['Filtering_VN_ay']
    steering = d['Steering']
    vx       = d['VN_vx']
    laps     = d['laps']
    laptime  = d['laptime']

    if '__corner_mask' in d:
        corner_mask = d['__corner_mask'].astype(bool) & (np.abs(vx) >= MIN_SPEED)
    else:
        raw_corner = (
            (np.abs(ay) >= AY_THRESHOLD)
            & (np.abs(steering) >= STEERING_THRESHOLD)
            & (np.abs(vx) >= MIN_SPEED)
        )
        corner_mask = keep_min_duration_segments(raw_corner, MIN_CORNER_DURATION, dt)

    lap_list = unique_laps(laps)
    n        = len(lap_list)
    lt_val   = np.full(n, np.nan)
    n_samps  = np.zeros(n, dtype=int)
    ay_mean  = np.full(n, np.nan)

    sa_mean = {w: np.full(n, np.nan) for w in ('FL', 'FR', 'RL', 'RR')}
    sa_eff  = {w: np.full(n, np.nan) for w in ('FL', 'FR', 'RL', 'RR')}

    for i, lap in enumerate(lap_list):
        lm  = laps == lap
        lcm = lm & corner_mask
        n_samps[i] = lcm.sum()
        if lm.any():
            lt_val[i] = laptime[lm].max()
        if n_samps[i] < MIN_CORNER_SAMPLES:
            continue

        ay_mean[i] = np.nanmean(np.abs(ay[lcm]))
        for w in ('FL', 'FR', 'RL', 'RR'):
            sa_deg = np.rad2deg(np.abs(d[f'Est_SA{w}'][lcm]))
            sa_mean[w][i] = np.nanmean(sa_deg)
            if sa_mean[w][i] > MIN_SA_MEAN_DEG:
                sa_eff[w][i] = ay_mean[i] / sa_mean[w][i]

    # Plot 1: mean SA per wheel vs selected lap axis
    fig1 = make_dark_figure(
        title=f"Mean Slip Angle per Wheel vs {'Lap Time' if x_mode == 'laptime' else 'Lap'}",
        xlabel='Lap time [s]' if x_mode == 'laptime' else 'Lap',
        ylabel='Mean |SA| in corners [deg]',
    )
    for w in ('FL', 'FR', 'RL', 'RR'):
        ok = (
            (n_samps >= MIN_CORNER_SAMPLES)
            & np.isfinite(sa_mean[w])
            & (sa_mean[w] > MIN_SA_MEAN_DEG)
            & (sa_mean[w] < MAX_SA_MEAN_DEG)
        )
        if ok.any():
            lap_disp = _display_laps(lap_list[ok])
            x_arr, order, _xlabel = per_lap_axis(lap_disp, lt_val[ok], x_mode)
            add_lap_scatter(fig1, x_arr, sa_mean[w][ok][order], lap_disp[order],
                            name=w, color=WHEEL_COLORS[w], symbol=WHEEL_SYMBOLS[w])
    if x_mode == 'laps':
        valid_laps = _display_laps(lap_list[n_samps >= MIN_CORNER_SAMPLES])
        if valid_laps.size > 0:
            fig1.update_xaxes(tickvals=np.sort(valid_laps.astype(int)))

    # Plot 2: SA efficiency vs lateral acceleration
    fig2 = make_dark_figure(
        title='Lateral Load vs Slip Angle Used',
        xlabel='Mean |SA| in corners [deg]',
        ylabel='Mean |ay| in corners [m/s²]',
    )
    for w in ('FL', 'FR', 'RL', 'RR'):
        ok = (
            (n_samps >= MIN_CORNER_SAMPLES)
            & np.isfinite(sa_mean[w])
            & np.isfinite(ay_mean)
            & (sa_mean[w] > MIN_SA_MEAN_DEG)
            & (sa_mean[w] < MAX_SA_MEAN_DEG)
        )
        if ok.any():
            add_lap_scatter(fig2, sa_mean[w][ok], ay_mean[ok], _display_laps(lap_list[ok]),
                            name=w, color=WHEEL_COLORS[w], symbol=WHEEL_SYMBOLS[w])

    return fig1, fig2, lap_list, sa_eff, lt_val, n_samps, sa_mean, ay_mean


def slip_angle_efficiency() -> list[go.Figure]:
    """CLI version: loads from CSV, prints KPIs, returns figures."""
    d = _load(_SA_COLS)
    fig1, fig2, lap_list, sa_eff, lt_val, n_samps, sa_mean, ay_mean = _compute_slip_angle(d)

    print('\n─── Slip Angle Efficiency [m/s²/deg] ───')
    header = (
        f"{'Lap':>4}  {'LapTime':>8}"
        + ''.join(f"  {'Eff_' + w:>10}" for w in ('FL', 'FR', 'RL', 'RR'))
    )
    print(header)
    for i, lap in enumerate(lap_list):
        if n_samps[i] < MIN_CORNER_SAMPLES:
            continue
        vals = ''.join(
            f"  {sa_eff[w][i]:>10.3f}" if np.isfinite(sa_eff[w][i]) else f"  {'—':>10}"
            for w in ('FL', 'FR', 'RL', 'RR')
        )
        print(f'{int(lap):>4}  {lt_val[i]:>8.2f}{vals}')

    return [fig1, fig2]


def slip_angle_efficiency_figs(
    df: pl.DataFrame,
    corner_mask: np.ndarray | None = None,
    x_mode: str = 'laptime',
) -> list[go.Figure]:
    """Dashboard version: takes a polars DataFrame, returns figures (no print)."""
    d = _from_df(df, _SA_COLS)
    if corner_mask is not None:
        d['__corner_mask'] = corner_mask.astype(float)
    fig1, fig2, *_ = _compute_slip_angle(d, x_mode=x_mode)
    return [fig1, fig2]


def slip_angle_efficiency_kpis(
    df: pl.DataFrame,
    corner_mask: np.ndarray | None = None,
) -> dict:
    """Dashboard KPIs for slip angle efficiency."""
    d = _from_df(df, _SA_COLS)
    if corner_mask is not None:
        d['__corner_mask'] = corner_mask.astype(float)
    _fig1, _fig2, lap_list, sa_eff, lt_val, n_samps, sa_mean, ay_mean = _compute_slip_angle(d)

    valid_mask = (
        (n_samps >= MIN_CORNER_SAMPLES)
        & np.isfinite(lt_val)
        & np.isfinite(ay_mean)
        & np.all(
            np.stack([np.isfinite(sa_eff[w]) for w in ("FL", "FR", "RL", "RR")], axis=1),
            axis=1,
        )
    )
    if not valid_mask.any():
        return {"warnings": ["No valid laps for slip-angle KPIs."]}

    mean_eff = np.nanmean(
        np.stack([sa_eff[w][valid_mask] for w in ("FL", "FR", "RL", "RR")], axis=1),
        axis=1,
    )
    best_idx = int(np.nanargmax(mean_eff))
    valid_laps = lap_list[valid_mask]
    valid_laps_disp = _display_laps(valid_laps)
    valid_lt = lt_val[valid_mask]
    valid_ay = ay_mean[valid_mask]

    table = {
        "Lap": valid_laps_disp.astype(int),
        "LapTime [s]": np.round(valid_lt, 3),
        "Corner samples": n_samps[valid_mask].astype(int),
        "Mean |ay| [m/s²]": np.round(valid_ay, 3),
    }
    for w in ("FL", "FR", "RL", "RR"):
        table[f"Mean |SA| {w} [deg]"] = np.round(sa_mean[w][valid_mask], 3)
        table[f"Eff {w} [m/s²/deg]"] = np.round(sa_eff[w][valid_mask], 4)
    table["Mean Eff [m/s²/deg]"] = np.round(mean_eff, 4)

    return {
        "valid_laps": int(valid_mask.sum()),
        "mean_corner_ay": float(np.nanmean(valid_ay)),
        "fastest_lap": int(valid_laps_disp[int(np.nanargmin(valid_lt))]),
        "fastest_lt": float(np.nanmin(valid_lt)),
        "best_eff_lap": int(valid_laps_disp[best_idx]),
        "best_eff": float(mean_eff[best_idx]),
        "eff_mean_by_wheel": {
            w: float(np.nanmean(sa_eff[w][valid_mask])) for w in ("FL", "FR", "RL", "RR")
        },
        "table": pl.DataFrame(table),
        "warnings": [],
    }


# ── 2. Understeer angle evolution ─────────────────────────────────────────────

def _compute_understeer(
    d: dict[str, np.ndarray],
    x_mode: str = 'laps',
) -> tuple[go.Figure, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Core computation for understeer angle.

    Returns (fig, lap_list, und_mean, lt_val, n_samps, ok_mask).
    """
    d['time'] = d['TimeStamp'] - d['TimeStamp'][0]

    valid = _base_validity(*d.values()) & (np.abs(d['vx']) >= 4.0)
    d = {k: v[valid] for k, v in d.items()}
    d = exclude_lap0_and_last_lap(d)

    dt       = robust_dt(d['time'])
    steering = d['Steering']
    ay_filt  = d['Filtering_VN_ay']
    vx       = d['vx']
    laps     = d['laps']
    laptime  = d['laptime']

    if '__corner_mask' in d:
        corner_mask = d['__corner_mask'].astype(bool) & (np.abs(vx) >= 4.0)
    else:
        raw_corner = (
            (np.abs(ay_filt) >= AY_THRESHOLD)
            & (np.abs(steering) >= STEERING_THRESHOLD)
            & (np.abs(vx) >= 4.0)
        )
        corner_mask = keep_min_duration_segments(raw_corner, MIN_CORNER_DURATION, dt)

    # Bicycle model: δ_ideal = L·ay / vx²
    ideal_steer = WHEELBASE_EQ * ay_filt / (vx ** 2)
    und_rad     = np.abs(steering) - np.abs(ideal_steer)
    und_deg     = np.rad2deg(und_rad)

    lap_list = unique_laps(laps)
    n        = len(lap_list)
    und_mean = np.full(n, np.nan)
    lt_val   = np.full(n, np.nan)
    n_samps  = np.zeros(n, dtype=int)

    for i, lap in enumerate(lap_list):
        lm  = laps == lap
        lcm = lm & corner_mask
        n_samps[i] = lcm.sum()
        if lm.any():
            lt_val[i] = laptime[lm].max()
        if n_samps[i] >= MIN_CORNER_SAMPLES:
            und_mean[i] = np.nanmean(und_deg[lcm])

    ok = (
        np.isfinite(und_mean)
        & np.isfinite(lt_val)
        & (n_samps >= MIN_CORNER_SAMPLES)
        & (np.abs(und_mean) < 20.0)
    )

    lap_disp = _display_laps(lap_list[ok]) if ok.any() else np.array([], dtype=int)
    x_arr, order, xlabel = per_lap_axis(lap_disp, lt_val[ok], x_mode) if ok.any() else (np.array([]), np.array([], dtype=int), 'Lap')
    fig = make_dark_figure(
        title=f"Average Understeer Angle vs {'Lap Time' if x_mode == 'laptime' else 'Lap'}",
        xlabel=xlabel, ylabel='Mean understeer angle [deg]',
    )
    if ok.any():
        add_lap_scatter(fig, x_arr, und_mean[ok][order], lap_disp[order])
        add_trend_line(fig, x_arr, und_mean[ok][order])
        add_zero_line(fig, x_arr)
        if x_mode == 'laps':
            fig.update_xaxes(tickvals=np.sort(lap_disp.astype(int)))

    return fig, lap_list, und_mean, lt_val, n_samps, ok


def understeer_angle() -> go.Figure:
    """CLI version: loads from CSV, prints KPIs, returns figure."""
    try:
        d = _load(_US_COLS_COG)
        d['vx'] = d.pop('Est_vxCOG')
    except Exception:
        d = _load(_US_COLS_VX)
        d['vx'] = d.pop('VN_vx')

    fig, lap_list, und_mean, lt_val, n_samps, ok = _compute_understeer(d)

    print('\n─── Understeer Angle per Lap ───')
    print(f"{'Lap':>4}  {'LapTime[s]':>10}  {'Und_mean[deg]':>14}  {'Samples':>8}")
    for lap, lt, um, ns in zip(lap_list[ok], lt_val[ok], und_mean[ok], n_samps[ok]):
        print(f'{int(lap):>4}  {lt:>10.3f}  {um:>14.3f}  {ns:>8d}')

    return fig


def understeer_angle_fig(
    df: pl.DataFrame,
    corner_mask: np.ndarray | None = None,
    x_mode: str = 'laps',
) -> go.Figure:
    """Dashboard version: takes a polars DataFrame, returns figure (no print)."""
    try:
        d = _from_df(df, _US_COLS_COG)
        d['vx'] = d.pop('Est_vxCOG')
    except (KeyError, Exception):
        d = _from_df(df, _US_COLS_VX)
        d['vx'] = d.pop('VN_vx')

    if corner_mask is not None:
        d['__corner_mask'] = corner_mask.astype(float)
    fig, *_ = _compute_understeer(d, x_mode=x_mode)
    return fig


def understeer_angle_kpis(
    df: pl.DataFrame,
    corner_mask: np.ndarray | None = None,
) -> dict:
    """Dashboard KPIs for understeer angle."""
    try:
        d = _from_df(df, _US_COLS_COG)
        d["vx"] = d.pop("Est_vxCOG")
    except (KeyError, Exception):
        d = _from_df(df, _US_COLS_VX)
        d["vx"] = d.pop("VN_vx")

    if corner_mask is not None:
        d['__corner_mask'] = corner_mask.astype(float)
    _fig, lap_list, und_mean, lt_val, n_samps, ok = _compute_understeer(d)
    if not ok.any():
        return {"warnings": ["No valid laps for understeer KPIs."]}

    valid_laps = lap_list[ok]
    valid_laps_disp = _display_laps(valid_laps)
    valid_und = und_mean[ok]
    valid_lt = lt_val[ok]
    valid_samples = n_samps[ok]

    table = pl.DataFrame({
        "Lap": valid_laps_disp.astype(int),
        "LapTime [s]": np.round(valid_lt, 3),
        "Mean understeer [deg]": np.round(valid_und, 3),
        "Corner samples": valid_samples.astype(int),
    })

    return {
        "valid_laps": int(ok.sum()),
        "mean_understeer": float(np.nanmean(valid_und)),
        "min_understeer": float(np.nanmin(valid_und)),
        "max_understeer": float(np.nanmax(valid_und)),
        "fastest_lap": int(valid_laps_disp[int(np.nanargmin(valid_lt))]),
        "fastest_lt": float(np.nanmin(valid_lt)),
        "mean_corner_samples": float(np.nanmean(valid_samples)),
        "table": table,
        "warnings": [],
    }


# ── 3. Interactive pilot / GG view ────────────────────────────────────────────

_PILOT_COLS = [
    "laps", "laptime",
    "Filtering_VN_ax", "Filtering_VN_ay",
    "VN_latitude", "VN_longitude",
    "VN_vx", "Brake", "Throttle", "Steering",
]

SIG_COLORS = {
    "throttle": "#73D973",
    "brake":    "#D94F4F",
    "steering": "#4DB3F2",
    "vx":       "#F28C40",
    "ax":       "#FFD700",
    "ay":       "#00BFBF",
}
_DASH_CYCLE     = ["solid", "dash", "dot", "dashdot"]
_PURPLE_FASTEST = "rgb(170, 60, 230)"
_YELLOW         = "rgba(255, 220, 0, 0.9)"
_YELLOW_BAND    = "rgba(255, 220, 0, 0.10)"
_MAP_HIGHLIGHT  = "rgba(77, 179, 242, 0.95)"
_LAP_GATE_LINE  = "rgba(240, 240, 240, 0.95)"
_LAP_GATE_CENTRE = "#FFFFFF"
_CLICK_THR_M    = 30.0   # box narrower than this [m] → treated as vline
_H_LEFT         = 250
_H_RIGHT        = 390
_AXIS_COL       = "#E5E5E5"
_TEXT_COL       = "#EBEBEB"
_PHASE_STYLES   = [
    ("STRAIGHT", "#73D973", 3, "Straight"),
    ("CORNER",   "#F2D44D", 4, "Corner"),
    ("BRAKE",    "#F27070", 4, "Braking"),
]


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
        dlat  = np.diff(lat_r)
        dlng  = np.diff(lng_r)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(lat_r[:-1]) * np.cos(lat_r[1:]) * np.sin(dlng / 2) ** 2)
        inc = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        dist[idx] = np.concatenate([[0.0], np.cumsum(inc)])
    return dist


def pool_arrays_from_dfs(
    dfs: dict[str, pl.DataFrame],
) -> tuple[dict[str, np.ndarray], list[tuple[str, int, float]]]:
    """Pool multiple polars DataFrames into flat numpy arrays for plotting.

    Assumes DataFrames are already filtered (laps > 0, last lap excluded).
    Returns (pool, entries) where pool keys are:
        ax, ay, lat, lng, dist, vx, brk, thr, ste, phase, run (str), lap (int)
    and entries is list of (run_name, lap_id, laptime).
    """
    ax_c, ay_c, lat_c, lng_c, dist_c = [], [], [], [], []
    brk_c, thr_c, ste_c, vx_c = [], [], [], []
    phase_c: list[np.ndarray] = []
    run_c: list[np.ndarray] = []
    lap_c: list[np.ndarray] = []
    entries: list[tuple[str, int, float]] = []

    for run_name, df in dfs.items():
        df = ensure_complete_laps_df(df)
        if any(c not in df.columns for c in _PILOT_COLS):
            continue

        laps    = df["laps"].to_numpy().astype(float)
        laptime = df["laptime"].to_numpy().astype(float)
        ax      = df["Filtering_VN_ax"].to_numpy().astype(float)
        ay      = df["Filtering_VN_ay"].to_numpy().astype(float)
        lat     = df["VN_latitude"].to_numpy().astype(float)
        lng     = df["VN_longitude"].to_numpy().astype(float)
        vx      = df["VN_vx"].to_numpy().astype(float)
        brk     = df["Brake"].to_numpy().astype(float)
        thr     = df["Throttle"].to_numpy().astype(float)
        ste     = np.rad2deg(df["Steering"].to_numpy().astype(float))
        phase_masks = phase_masks_for_map(df)
        phase = np.full(len(df), "STRAIGHT", dtype=object)
        phase[phase_masks["BRAKE"]] = "BRAKE"
        phase[phase_masks["CORNER"]] = "CORNER"

        finite = np.all(np.stack([
            np.isfinite(laps), np.isfinite(laptime),
            np.isfinite(ax), np.isfinite(ay),
            np.isfinite(lat), np.isfinite(lng),
            np.isfinite(vx), np.isfinite(brk),
            np.isfinite(thr), np.isfinite(ste),
        ]), axis=0)

        laps    = laps[finite].astype(int)
        laptime = laptime[finite]
        ax  = ax[finite];   ay  = ay[finite]
        lat = lat[finite];  lng = lng[finite]
        vx  = vx[finite];   brk = brk[finite]
        thr = thr[finite];  ste = ste[finite]
        phase = phase[finite]

        laps_disp = laps.astype(int)
        for lap_id in np.unique(laps):
            lm = laps == lap_id
            entries.append((run_name, int(lap_id), float(laptime[lm].max())))

        dist = _per_lap_distance(lat, lng, laps.astype(float))
        n = len(ax)
        ax_c.append(ax);      ay_c.append(ay)
        lat_c.append(lat);    lng_c.append(lng)
        dist_c.append(dist);  vx_c.append(vx)
        brk_c.append(brk);    thr_c.append(thr)
        ste_c.append(ste)
        phase_c.append(phase)
        run_c.append(np.full(n, run_name))
        lap_c.append(laps_disp)

    if not ax_c:
        return {}, []

    return {
        "ax":   np.concatenate(ax_c),
        "ay":   np.concatenate(ay_c),
        "lat":  np.concatenate(lat_c),
        "lng":  np.concatenate(lng_c),
        "dist": np.concatenate(dist_c),
        "vx":   np.concatenate(vx_c),
        "brk":  np.concatenate(brk_c),
        "thr":  np.concatenate(thr_c),
        "ste":  np.concatenate(ste_c),
        "phase": np.concatenate(phase_c),
        "run":  np.concatenate(run_c),
        "lap":  np.concatenate(lap_c),
    }, entries


def build_color_map(
    entries: list[tuple[str, int, float]],
) -> dict[tuple[str, int], str]:
    """Map (run_name, lap_id) → color. Purple = fastest, RdYlGn gradient for rest."""
    import plotly.colors as pc
    if not entries:
        return {}
    ordered = sorted(entries, key=lambda e: e[2])
    n = len(ordered)
    colors: dict[tuple[str, int], str] = {
        (ordered[0][0], ordered[0][1]): _PURPLE_FASTEST
    }
    if n == 1:
        return colors
    positions = [1.0 - (i / (n - 1)) for i in range(n)]
    scale = pc.sample_colorscale("RdYlGn", positions)
    for i in range(1, n):
        colors[(ordered[i][0], ordered[i][1])] = scale[i]
    return colors


def gg_axis_range(
    pool: dict[str, np.ndarray],
    visible_mask: np.ndarray,
    zone_mask: np.ndarray | None = None,
) -> list[float]:
    """Return a symmetric GG range from currently visible samples."""
    mask = visible_mask.copy()
    if zone_mask is not None:
        mask &= zone_mask

    if not np.any(mask):
        mask = visible_mask

    if not np.any(mask):
        mask = np.ones(len(pool["ax"]), dtype=bool)

    ax_vis = pool["ax"][mask]
    ay_vis = pool["ay"][mask]
    finite = np.isfinite(ax_vis) & np.isfinite(ay_vis)
    if not np.any(finite):
        return [-1.0, 1.0]

    gg_max = float(max(np.max(np.abs(ax_vis[finite])), np.max(np.abs(ay_vis[finite]))))
    gg_max = max(gg_max * 1.1, 1.0)
    return [-gg_max, gg_max]


def _gg_lap_label(run: str, lap: int, single_csv: bool) -> str:
    """Compact lap label for GG quadrant annotations."""
    return f"L{lap}" if single_csv else f"{run}·L{lap}"


def _gg_quadrant_counts(ay_mps2: np.ndarray, ax_mps2: np.ndarray) -> dict[str, int]:
    """Count points in each GG quadrant using the plotted axes sign convention."""
    return {
        "Q1 (+ay, +ax)": int(((ay_mps2 >= 0.0) & (ax_mps2 >= 0.0)).sum()),
        "Q2 (-ay, +ax)": int(((ay_mps2 < 0.0) & (ax_mps2 >= 0.0)).sum()),
        "Q3 (-ay, -ax)": int(((ay_mps2 < 0.0) & (ax_mps2 < 0.0)).sum()),
        "Q4 (+ay, -ax)": int(((ay_mps2 >= 0.0) & (ax_mps2 < 0.0)).sum()),
    }


def _add_gg_quadrant_annotations(
    fig: go.Figure,
    pool: dict[str, np.ndarray],
    entries: list[tuple[str, int, float]],
    visible_keys: set[tuple[str, int]],
    zone_mask: np.ndarray,
    single_csv: bool,
) -> None:
    """Annotate each GG quadrant with per-lap point counts for the visible data."""
    quadrant_lines = {
        "Q1 (+ay, +ax)": [],
        "Q2 (-ay, +ax)": [],
        "Q3 (-ay, -ax)": [],
        "Q4 (+ay, -ax)": [],
    }

    for run, lap, _lt in sorted(entries, key=lambda e: e[2]):
        if (run, lap) not in visible_keys:
            continue
        smask = (pool["run"] == run) & (pool["lap"] == lap) & zone_mask
        if not smask.any():
            continue
        lap_name = _gg_lap_label(run, lap, single_csv)
        counts = _gg_quadrant_counts(pool["ay"][smask], pool["ax"][smask])
        for quadrant, count in counts.items():
            quadrant_lines[quadrant].append(f"{lap_name}: {count}")

    annotation_specs = [
        ("Q2 (-ay, +ax)", 0.02, 0.98, "left", "top"),
        ("Q1 (+ay, +ax)", 0.98, 0.98, "right", "top"),
        ("Q3 (-ay, -ax)", 0.02, 0.02, "left", "bottom"),
        ("Q4 (+ay, -ax)", 0.98, 0.02, "right", "bottom"),
    ]
    for quadrant, x_pos, y_pos, x_anchor, y_anchor in annotation_specs:
        lines = quadrant_lines[quadrant]
        if not lines:
            continue
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=x_pos,
            y=y_pos,
            xanchor=x_anchor,
            yanchor=y_anchor,
            showarrow=False,
            align="left" if x_anchor == "left" else "right",
            text=f"<b>{quadrant}</b><br>" + "<br>".join(lines),
            font=dict(size=9, color=_TEXT_COL),
            bgcolor="rgba(20,20,23,0.82)",
            bordercolor="rgba(128,128,128,0.35)",
            borderwidth=1,
            borderpad=4,
        )


def has_selection(event) -> bool:
    """True if a Streamlit plotly event contains a box or lasso selection."""
    try:
        sel = event["selection"]
        boxes = sel.get("box", [])
        if boxes and boxes[0].get("x"):
            return True
        return bool(sel.get("points", []))
    except (TypeError, KeyError, AttributeError):
        return False


def _selection_index(value) -> int | None:
    """Extract a single pool index from Plotly/Streamlit selection payloads."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value) if np.isfinite(value) else None
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return _selection_index(value.flat[0])
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _selection_index(value[0])
    if isinstance(value, dict):
        for key in ("value", "point_index", "pointIndex", "0"):
            if key in value:
                idx = _selection_index(value[key])
                if idx is not None:
                    return idx
        for nested in value.values():
            idx = _selection_index(nested)
            if idx is not None:
                return idx
    return None


def _customdata_from_indices(indices: np.ndarray) -> list[list[int]]:
    """Convert pool indices to JSON-stable customdata payloads."""
    return [[int(idx)] for idx in np.asarray(indices, dtype=int).tolist()]


def extract_zone_mask(event, n_points: int) -> tuple[np.ndarray, bool]:
    """Extract zone mask from a track-map selection event."""
    try:
        pts = event["selection"]["points"] or []
    except (TypeError, KeyError, AttributeError):
        return np.ones(n_points, dtype=bool), False
    if not pts:
        return np.ones(n_points, dtype=bool), False
    mask = np.zeros(n_points, dtype=bool)
    for p in pts:
        idx = _selection_index(p.get("customdata"))
        if idx is None:
            idx = _selection_index(p.get("point_index"))
        if idx is None:
            idx = _selection_index(p.get("pointIndex"))
        if idx is not None and 0 <= idx < n_points:
            mask[idx] = True
    if not mask.any():
        return np.ones(n_points, dtype=bool), False
    return mask, True


def extract_gg_pool_indices(event) -> np.ndarray | None:
    """Extract pool indices from a GG selection event (reads customdata)."""
    try:
        pts = event["selection"]["points"] or []
    except (TypeError, KeyError, AttributeError):
        return None
    if not pts:
        return None
    indices = []
    for p in pts:
        idx = _selection_index(p.get("customdata"))
        if idx is not None:
            indices.append(idx)
    return np.array(indices, dtype=int) if indices else None


def dist_range_from_event(event) -> tuple[float, float] | None:
    """Extract (d_min, d_max) [m] from a distance-plot selection event, or None."""
    try:
        sel = event["selection"]
    except (TypeError, KeyError, AttributeError):
        return None
    try:
        boxes = sel.get("box", [])
        if boxes:
            xs = boxes[0].get("x", [])
            if len(xs) >= 2:
                return (float(min(xs)), float(max(xs)))
    except Exception:
        pass
    pts = sel.get("points", [])
    if not pts:
        return None
    xs = [p["x"] for p in pts if p.get("x") is not None]
    return (float(min(xs)), float(max(xs))) if xs else None


def _add_dist_traces(
    fig: go.Figure,
    pool: dict[str, np.ndarray],
    sig1_key: str,
    sig2_key: str | None,
    entries: list[tuple[str, int, float]],
    visible_keys: set[tuple[str, int]],
    sig1_label: str,
    sig1_color: str,
    sig2_label: str = "",
    sig2_color: str = "",
    sig2_yaxis: str = "y",
    extra_mask: np.ndarray | None = None,
) -> None:
    visible_entries = [
        (run, lap, lt) for run, lap, lt in sorted(entries, key=lambda e: e[2])
        if (run, lap) in visible_keys
    ]
    for i, (run, lap, _lt) in enumerate(visible_entries):
        smask = (pool["run"] == run) & (pool["lap"] == lap)
        if extra_mask is not None:
            smask = smask & extra_mask
        if not smask.any():
            continue
        dash  = _DASH_CYCLE[i % len(_DASH_CYCLE)]
        x     = pool["dist"][smask]
        s1    = pool[sig1_key][smask]
        order = np.argsort(x)
        fig.add_trace(go.Scattergl(
            x=x[order], y=s1[order], mode="lines",
            name=sig1_label, showlegend=False,
            line=dict(color=sig1_color, width=1.5, dash=dash),
            hovertemplate=f"%{{y:.2f}} (L{lap})<extra></extra>",
        ))
        if sig2_key and sig2_label:
            s2 = pool[sig2_key][smask]
            fig.add_trace(go.Scattergl(
                x=x[order], y=s2[order], mode="lines",
                name=sig2_label, showlegend=False, yaxis=sig2_yaxis,
                line=dict(color=sig2_color, width=1.5, dash=dash),
                hovertemplate=f"%{{y:.2f}} (L{lap})<extra></extra>",
            ))


def track_map_fig(
    pool: dict[str, np.ndarray],
    visible_mask: np.ndarray,
    cross_range: tuple[float, float] | None,
    gg_idx: np.ndarray | None,
    ui_rev: str,
    lap_gates: dict[str, dict[str, object]] | None = None,
) -> go.Figure:
    """Track map coloured by phase, with optional highlights for linked selections."""
    fig = make_dark_figure(xlabel="Longitude [deg]", ylabel="Latitude [deg]")
    for phase, color, size, label in _PHASE_STYLES:
        phase_mask = visible_mask & (pool["phase"] == phase)
        if not phase_mask.any():
            continue
        pool_idx = np.where(phase_mask)[0]
        fig.add_trace(go.Scattergl(
            x=pool["lng"][phase_mask], y=pool["lat"][phase_mask], mode="markers",
            name=label,
            marker=dict(size=size, color=color, opacity=0.85),
            customdata=_customdata_from_indices(pool_idx),
            hovertemplate=(
                f"{label}<br>lon=%{{x:.6f}}"
                f"<br>lat=%{{y:.6f}}<extra></extra>"
            ),
        ))
    if lap_gates:
        visible_runs = np.unique(pool["run"][visible_mask])
        multi_run = len(visible_runs) > 1
        for run_name in visible_runs:
            gate = lap_gates.get(str(run_name))
            if not gate:
                continue
            gate_lon = np.asarray(gate["gate_lon"], dtype=float)
            gate_lat = np.asarray(gate["gate_lat"], dtype=float)
            finish_lon = float(gate["finish_lon"])
            finish_lat = float(gate["finish_lat"])
            gate_half_width_m = float(gate["gate_half_width_m"])
            gate_name = f"Lap detection · {run_name}" if multi_run else "Lap detection"
            fig.add_trace(go.Scattergl(
                x=gate_lon, y=gate_lat, mode="lines",
                name=gate_name,
                line=dict(color=_LAP_GATE_LINE, width=2, dash="dash"),
                hovertemplate=(
                    f"{gate_name}<br>half width={gate_half_width_m:.1f} m"
                    "<extra></extra>"
                ),
            ))
            fig.add_trace(go.Scattergl(
                x=[finish_lon], y=[finish_lat], mode="markers",
                marker=dict(
                    size=11, color=_LAP_GATE_CENTRE, symbol="x",
                    line=dict(color=_LAP_GATE_LINE, width=1.5),
                ),
                showlegend=False,
                hovertemplate=(
                    f"{gate_name} centre"
                    f"<br>lon={finish_lon:.6f}"
                    f"<br>lat={finish_lat:.6f}<extra></extra>"
                ),
            ))
    if cross_range is not None:
        d_min, d_max = cross_range
        if (d_max - d_min) < _CLICK_THR_M:
            map_mask = visible_mask & (np.abs(pool["dist"] - (d_min + d_max) / 2) < 5.0)
        else:
            map_mask = visible_mask & (pool["dist"] >= d_min) & (pool["dist"] <= d_max)
        if map_mask.any():
            pool_idx = np.where(map_mask)[0]
            fig.add_trace(go.Scattergl(
                x=pool["lng"][map_mask], y=pool["lat"][map_mask], mode="markers",
                marker=dict(size=6, color=_MAP_HIGHLIGHT),
                customdata=_customdata_from_indices(pool_idx),
                showlegend=False, hoverinfo="skip",
            ))
    if gg_idx is not None and len(gg_idx) > 0:
        fig.add_trace(go.Scattergl(
            x=pool["lng"][gg_idx], y=pool["lat"][gg_idx], mode="markers",
            marker=dict(size=5, color=_MAP_HIGHLIGHT),
            customdata=_customdata_from_indices(np.asarray(gg_idx)),
            showlegend=False, hoverinfo="skip",
        ))
    fig.update_layout(
        height=_H_RIGHT, dragmode="select", uirevision=ui_rev,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1.0,
        ),
        margin=dict(l=60, r=10, t=50, b=60),
    )
    return fig


def gg_diagram_fig(
    pool: dict[str, np.ndarray],
    entries: list[tuple[str, int, float]],
    visible_keys: set[tuple[str, int]],
    color_map: dict[tuple[str, int], str],
    zone_mask: np.ndarray,
    single_csv: bool,
    ui_rev: str,
    gg_range: list[float],
) -> go.Figure:
    """GG diagram coloured by lap, optionally filtered by track zone."""
    fig = make_dark_figure(
        xlabel="Filtering_VN_ay [m/s²]",
        ylabel="Filtering_VN_ax [m/s²]",
    )
    for run, lap, lt in sorted(entries, key=lambda e: e[2]):
        if (run, lap) not in visible_keys:
            continue
        smask = (pool["run"] == run) & (pool["lap"] == lap) & zone_mask
        if not smask.any():
            continue
        lap_name = f"L{lap} ({lt:.2f}s)" if single_csv else f"{run}·L{lap} ({lt:.2f}s)"
        pool_idx = np.where(smask)[0]
        fig.add_trace(go.Scattergl(
            x=pool["ay"][smask], y=pool["ax"][smask],
            mode="markers", name=lap_name,
            marker=dict(size=3, color=color_map[(run, lap)], opacity=0.75),
            customdata=_customdata_from_indices(pool_idx),
            hovertemplate=(
                f"{lap_name}<br>ay=%{{x:.2f}} m/s²"
                f"<br>ax=%{{y:.2f}} m/s²<extra></extra>"
            ),
        ))
    fig.add_vline(x=0.0, line=dict(color="rgba(200,200,200,0.35)", dash="dot", width=1))
    fig.add_hline(y=0.0, line=dict(color="rgba(200,200,200,0.35)", dash="dot", width=1))
    _add_gg_quadrant_annotations(fig, pool, entries, visible_keys, zone_mask, single_csv)
    fig.update_layout(
        height=_H_RIGHT, uirevision=ui_rev, showlegend=False,
        margin=dict(l=60, r=10, t=30, b=60),
        xaxis=dict(autorange=True),
        yaxis=dict(autorange=True, scaleanchor="x", scaleratio=1),
    )
    return fig


def dist_plot_fig(
    pool: dict[str, np.ndarray],
    sig1_key: str,
    sig2_key: str | None,
    entries: list[tuple[str, int, float]],
    visible_keys: set[tuple[str, int]],
    extra_mask: np.ndarray | None,
    ylabel: str,
    sig1_label: str,
    sig1_color: str,
    sig2_label: str,
    sig2_color: str,
    sig2_yaxis: str,
    ui_rev: str,
    cross_range: tuple[float, float] | None,
    right_yaxis_title: str = "",
    compact: str = "",
) -> go.Figure:
    """One distance plot with optional cross-chart highlight band/vline.

    *compact* controls tight vertical stacking:
      - ``"top"``    — first chart: no x-axis labels, small bottom margin
      - ``"middle"`` — middle chart: no x-axis labels, minimal top+bottom
      - ``"bottom"`` — last chart: keep x-axis, minimal top margin
      - ``""``       — legacy (default spacing)
    """
    show_xaxis = compact not in ("top", "middle")
    has_secondary_y = sig2_yaxis == "y2"
    xlabel = "Distance [m]" if show_xaxis else ""
    fig = make_dark_figure(xlabel=xlabel, ylabel=ylabel)

    if compact == "top":
        margin = dict(l=60, r=60 if right_yaxis_title else 10, t=24, b=4)
    elif compact == "middle":
        margin = dict(l=60, r=60 if right_yaxis_title else 10, t=4, b=4)
    elif compact == "bottom":
        margin = dict(l=60, r=60 if right_yaxis_title else 10, t=4, b=32)
    else:
        margin = dict(l=60, r=60 if right_yaxis_title else 10, t=30, b=40)

    fig.update_layout(
        height=_H_LEFT, uirevision=ui_rev, showlegend=False,
        margin=margin,
        hovermode="x unified", dragmode="select",
    )

    if compact in ("top", "middle"):
        fig.update_xaxes(showticklabels=False, title_text="")

    # Signal legend as annotation inside the plot area
    if compact:
        legend_parts = [
            f'<span style="color:{sig1_color}">■</span> {sig1_label}'
        ]
        if sig2_label:
            legend_parts.append(
                f'<span style="color:{sig2_color}">■</span> {sig2_label}'
            )
        fig.add_annotation(
            text="&nbsp;&nbsp;&nbsp;&nbsp;".join(legend_parts),
            xref="paper", yref="paper", x=0.0, y=1.0,
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=11, color="#EBEBEB"),
            bgcolor="rgba(20,20,23,0.7)",
        )

    fig.update_xaxes(
        showspikes=True, spikemode="across",
        spikedash="solid", spikecolor="rgba(200,200,200,0.5)", spikethickness=1,
    )
    if has_secondary_y:
        fig.update_layout(yaxis2=dict(
            title=right_yaxis_title,
            overlaying="y",
            side="right",
            showgrid=False,
            showticklabels=bool(right_yaxis_title),
            ticks="outside" if right_yaxis_title else "",
            showline=bool(right_yaxis_title),
            color=_AXIS_COL, linecolor=_AXIS_COL, tickcolor=_AXIS_COL,
            tickfont=dict(color=_TEXT_COL), title_font=dict(color=_TEXT_COL),
        ))
    _add_dist_traces(
        fig, pool, sig1_key, sig2_key, entries, visible_keys,
        sig1_label, sig1_color, sig2_label, sig2_color, sig2_yaxis, extra_mask,
    )
    if cross_range is not None:
        d_min, d_max = cross_range
        if (d_max - d_min) < _CLICK_THR_M:
            fig.add_vline(x=(d_min + d_max) / 2,
                          line=dict(color=_YELLOW, dash="solid", width=1.5))
        else:
            fig.add_vrect(x0=d_min, x1=d_max,
                          fillcolor=_YELLOW_BAND, layer="below", line_width=0)
    return fig


def phase_map_fig(df: pl.DataFrame) -> go.Figure:
    """GPS track map coloured by adaptive phase detection.

    BRAKE = red, CORNER = yellow, STRAIGHT = green.
    Thresholds are derived from the run's data distribution via phase_masks_for_map().
    """
    fig = make_dark_figure(
        title="Phase Map — Adaptive Filters",
        xlabel="Longitude [deg]",
        ylabel="Latitude [deg]",
    )

    gps_cols = ("VN_latitude", "VN_longitude")
    if any(c not in df.columns for c in gps_cols):
        fig.add_annotation(
            text="GPS columns (VN_latitude / VN_longitude) not available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#EBEBEB", size=13),
        )
        return fig

    lat   = df["VN_latitude"].to_numpy().astype(float)
    lng   = df["VN_longitude"].to_numpy().astype(float)
    valid = np.isfinite(lat) & np.isfinite(lng)
    if not valid.any():
        return fig

    masks = phase_masks_for_map(df)

    # Base layer: full track outline in grey
    fig.add_trace(go.Scattergl(
        x=lng[valid], y=lat[valid], mode="markers",
        marker=dict(size=2, color="rgba(180,180,180,0.25)"),
        showlegend=False, hoverinfo="skip",
    ))

    # Paint BRAKE first and CORNER after it so overlapping samples read as curve.
    for phase, color, size, label in [("BRAKE", "#F27070", 4, "Braking"),
                                      ("STRAIGHT", "#73D973", 3, "Straight"),
                                      ("CORNER", "#F2D44D", 4, "Corner")]:
        m = masks[phase] & valid
        if m.any():
            fig.add_trace(go.Scattergl(
                x=lng[m], y=lat[m], mode="markers",
                name=label,
                marker=dict(size=size, color=color),
                showlegend=True,
            ))

    fig.update_layout(
        height=520,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1.0,
        ),
        margin=dict(l=60, r=10, t=50, b=60),
    )
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    for fig in slip_angle_efficiency():
        fig.show()
    understeer_angle().show()


if __name__ == '__main__':
    main()
