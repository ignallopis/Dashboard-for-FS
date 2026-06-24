"""CAT17x — Telemetry Dashboard

Entry point:  streamlit run src/dashboard.py

This is the only file that calls st.plotly_chart() or any other st.* rendering
functions.  All src/ modules return go.Figure objects (and kpis dicts) and never
render themselves.
"""

from __future__ import annotations

import base64
import copy
import html
import json
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
import src.cornering as corn
import src.tc as tc
import src.tv as tv
import src.rb as rb
import src.driver as drv
import src.gripfactor as gf
import src.acceleration as accel
import src.skidpad as skidpad
import src.lap_sectors as lsec
import src.lapcount as lapcount
import src.track_map_component as tmc
import src.videoanalysis as va

from utils import (
    BRAND_BLUE_600,
    FONT_DISPLAY,
    FONT_FAMILY,
    FONT_MONO,
    PLOT_AXIS_TITLE_STANDOFF,
    PLOT_FONT_SIZE,
    PLOT_HOVER_FONT_SIZE,
    WHEEL_COLORS,
    _SURFACE,
    _SURFACE_BORDER,
    _TEXT_MUTED,
    available_laps,
    cols_to_numpy,
    driver_color,
    enrich_run_df,
    load_data,
    select_laps_df,
    set_run_colors,
    style_metrics_table,
    style_sector_times_table,
)

DATA_DIR = Path(__file__).parent.parent / "data"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
APP_LOGO_PATH = REPO_ROOT / "assets" / "bcn-emotorsport-logo-blue.png"
TRACE_DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")
TRACE_SYMBOLS = ("circle", "square", "diamond", "triangle-up", "x", "cross")
POTENTIAL_LAP_RUN = "__potential_lap__"
POTENTIAL_LAP_ID = 1
FileSignature = tuple[int, int]

# Central tooltip text for KPI columns / cards. Keyed by the exact display
# label so the same explanation appears wherever a metric is shown (tables and
# st.metric cards). Columns/metrics not listed simply render without a tooltip.
METRIC_HELP: dict[str, str] = {
    # TC — one figure per controller reference (redesign 2026-06-21)
    "TC front p95 SR": "95th-percentile front-wheel slip while armed; above +0.20 means the fronts run past the grip optimum.",
    "TC rear p95 SR": "95th-percentile rear-wheel slip while armed; compare with the front to see which axle TC fights.",
    "TC time over target [%]": "Share of TC-armed time the wheel's slip ratio sits above +0.20 — how persistently it rides past the grip optimum (lower is better).",
    "TC escape [%]": "Share of armed time the wheel speed exceeded its TC velocity reference — the wheel escaping the cap (lower is better).",
    "TC p95 speed-to-ref": "95th-percentile of wheel speed ÷ velocity reference while armed; ≈1 means riding the reference, >1 means escaping it.",
    "TC p95 cut [Nm]": "95th-percentile torque cut applied while armed (torque-mode/new-car logs). Empty on velocity-mode (old-car) logs.",
    # RB — controller-quality redesign (2026-06-17)
    "RB current track error": "Median |achieved − target| pack current during braking [A]; lower = the PI holds its current target tighter.",
    "RB current within band": "% of braking samples where achieved current is within ±5 A of the target; higher = tighter tracking.",
    "RB current overshoot": "P95 of achieved/target current; >1 = the loop overshoots its target.",
    "RB front regen share": "Median front share of regen (motor) braking torque under braking — the regen the RB system sends to the front axle. Excludes hydraulic.",
    "RB regen Δ vs load": "Median (front regen share − front vertical-load share) in pp. 0 = regen splits exactly with load; >0 = front over-regen'd vs its load.",
    "RB regen-load corr": "Correlation of front regen share with front load share under braking. Near 1 = regen tracks load (RB's design intent).",
    "RB time saturated": "% of armed braking time the controller is at a limit (current or torque) and can't recover more.",
    "RB current-limited": "% of braking time pinned at the battery-current ceiling (RB_intensityTarget).",
    "RB torque-limited": "% of braking time pinned at the regen-torque ceiling (Param_desiredMaximumRegenTorque).",
    # RB — regen-as-delivered redesign (2026-06-20)
    "RB capture ratio": "Median per-event regen capture: electrical energy returned to the battery ÷ kinetic energy shed braking. <1 by design (friction brakes + drag take the rest); battery-bus, not motor-shaft.",
    "RB capture ratio overall": "Run-total recovered electrical energy ÷ total kinetic energy shed braking — the aggregate capture across all braking events.",
    "RB p95 regen current": "95th-percentile delivered pack regen current while braking [A] — the robust peak recovery the battery sees.",
    "RB current authority": "P95 regen current as a share of the 80 A battery cap; <100% = recovery headroom (battery never the limit), ≈100% = battery-limited.",
    "RB time at cap": "Share of braking time delivered regen current sits at/above the 80 A battery cap — how often the battery is the binding limit.",
    "Traction eff. grip [%]": "P95 achieved ax over the aero-scaled tyre-grip limit, max across speed bins in the grip-limited regime. 100% = riding the tyre limit; the gap mixes wheelspin (see TC) and torque shaping. On circuit logs this is low because acceleration is rarely traction-limited — read it comparatively run-to-run.",
    "Grip-limited [%]": "Share of acceleration samples where tyre grip (not power) is the binding limit (low speed).",
    "Power-limited [%]": "Share of acceleration samples where the 80 kW power cap is the binding limit (high speed).",
    "Peak drive [kN]": "P95 total drive force (sum of Est_FX over four wheels) in acceleration.",
    "Torque-limited [%]": "Share of acceleration samples capped by the motor-torque ceiling (very low speed / launch).",
    "Util front [-]": "Front-axle traction utilisation Fx/(mu*Fz): 1.0 = front axle at its grip limit.",
    "Util rear [-]": "Rear-axle traction utilisation Fx/(mu*Fz): 1.0 = rear axle at its grip limit.",
    "Limiting axle": "Axle with the higher peak utilisation = the one that saturates grip first under drive.",
    "Rear bias [%]": "Measured share of drive force on the rear axle.",
    "Ideal rear [%]": "Load-proportional ideal rear share given the rearward load shift under acceleration.",
    # Generic
    "Run": "Telemetry run / driver.",
    "Valid laps": "Laps that passed the lap-validity filter and were used.",
    "Samples": "Number of 100 Hz samples behind this aggregate.",
    "Fastest lap": "Lap number with the lowest lap time in this run.",
    "Fastest lt [s]": "Lowest lap time in this run [s].",
    # TV · control — one figure per TV reference (cascade order)
    "Yaw tracking RMSE [rad/s]": "RMSE between real yaw rate (VN_gz) and TV's desired yaw rate, in corners. Lower = the car follows the TV target better.",
    "Tracking slope": "Robust slope of real vs target yaw rate. 1.0 = perfect; <1 = the car under-rotates relative to what the TV asks; >1 = over-rotates.",
    "FF share (median)": "Median |Mz_ff| / (|Mz_ff| + |Mz_fb|) in corners (magnitude share, 0–1). High = the feedforward map carries the yaw command (anticipation); low = the PI feedback is left correcting.",
    "FB-led [%]": "Share of corner samples where feedback dominates (FF share < 0.5). High = the FF map under-delivers and the loop constantly corrects — retune the FF look-up.",
    "Effective PI gain [Nm·s/rad]": "Robust slope of feedback moment vs yaw error — the loop's effective response. Mixes P and the integral term (NOT pure Kp); negative would flag wrong-sign feedback.",
    "Ringing rate [1/s]": "Sign changes of the feedback moment per second of cornering (jitter-filtered, per corner segment). High = the PI loop is oscillating / under-damped.",
    "Moment utilisation p95": "P95 of |desired Mz| / |Mz limit| in corners (1.0 = saturated). Low = the TV runs far from its yaw-moment authority limit (plenty of headroom).",
    "Mz delivery ratio": "Median |actual Mz| / |desired Mz| in corners (≈1 = the allocator delivers the commanded yaw moment; actual is a Bz·T reconstruction from the wheel torques).",
    "Fx envelope-use p95": "P95 of |desired Fx| / |Fx limit| over the moving lap (1.0 = the wheels can give no more drive/brake force). High = the car is longitudinal-force-limited; as cornering builds, the QP trades this force for yaw moment (figure 4).",
    "Time at Fx limit [%]": "Share of moving-lap samples with Fx envelope-use ≥ 0.95 — how often longitudinal force is the binding constraint (vs yaw moment, which rarely is).",
    # Driver · throttle
    "Mean throttle [%]": "Average throttle position over the lap. Higher = more time on power.",
    "Full throttle / lap [s]": "Seconds per lap above the full-throttle threshold.",
    "Full throttle [%]": "Share of lap time at full throttle (TP > 95 %).",
    "Off throttle [%]": "Share of lap time fully off throttle.",
    "Median |dTP/dt| [%/s]": "Typical throttle application rate. Higher = snappier pedal.",
    "Peak lap |dTP/dt| [%/s]": "Fastest throttle application seen in a lap.",
    # Driver · brake
    "Mean aggressiveness [%/s]": "Average brake-apply rate (|dBrake/dt| while pressing).",
    "Peak lap aggressiveness [%/s]": "Hardest brake application in a lap.",
    "Mean release smoothness [%/s]": "Average brake-release rate. Lower = smoother trail-off.",
    "Peak lap release smoothness [%/s]": "Fastest brake release in a lap.",
    # Driver · steering
    "Mean smoothness [deg]": "Average steering reversal magnitude. Lower = smoother hands.",
    "Peak lap smoothness [deg]": "Largest steering reversal in a lap.",
    "Mean integral [deg*m]": "Distance-integral of |steering| per lap. Higher = more total steering demand.",
    "Peak lap integral [deg*m]": "Highest per-lap steering integral.",
    "Mean curvature [1/m]": "Average driven-path curvature. Higher = tighter line.",
    "Peak lap curvature [1/m]": "Tightest driven curvature in a lap.",
    # Dynamics · cornering / understeer
    "Mean understeer [deg]": "Mean steady-state understeer angle. Positive = understeer.",
    "Min [deg]": "Minimum per-lap understeer angle.",
    "Max [deg]": "Maximum per-lap understeer angle.",
    "Entry balance": "Front−rear slip angle on turn-in [deg]. Positive = understeer (front slips more).",
    "Steady balance": "Front−rear slip angle at the apex/steady-state [deg]. Positive = understeer.",
    "Exit balance": "Front−rear slip angle on corner exit [deg]. Positive = understeer.",
    "Entry→exit shift": "Exit minus entry balance [deg]: how the car's balance migrates through the corner.",
    "Peak ay (L)": "P95 lateral g sustained in left turns (envelope max).",
    "Peak ay (R)": "P95 lateral g sustained in right turns (envelope max).",
    "L/R asymmetry": "Peak lateral g left minus right [g]: setup/track asymmetry.",
    "Lat util front": "Front-axle lateral utilisation |Fy|/(mu*Fz): 1.0 = front at its lateral grip limit.",
    "Lat util rear": "Rear-axle lateral utilisation |Fy|/(mu*Fz): 1.0 = rear at its lateral grip limit.",
    "Limiting axle (lat)": "Axle with higher peak lateral utilisation = saturates grip first in cornering.",
    # Dynamics · braking force balance
    "Front bias [%]": "Front-axle share of braking force. ~60–67 % is the model target.",
    "Bias std [pp]": "Lap-to-lap spread of front bias [percentage points]. Lower = more repeatable.",
    "RMS to ideal [N]": "RMS distance of measured regen from the ideal load-proportional curve.",
    "Rear overbiased [%]": "Share of braking samples where the rear axle exceeds its ideal share.",
    "Peak brake [g]": "Peak longitudinal deceleration [g].",
    "Best max braking [g]": "Most negative per-lap braking point in the selected laps. More negative = harder peak decel.",
    "Mean max braking [g]": "Average of the per-lap Max Braking G values across the selected laps.",
    "Raising avg end [g]": "Final cumulative mean of Max Braking G across laps. Useful to compare runs without overreacting to one lap.",
    "Best lap": "Lap with the most negative Max Braking G in the selected laps.",
    "Mean brake samples": "Average count of braking samples per lap behind the Max Braking G metric.",
    "Peak P95 decel": "Highest P95 deceleration reached in any speed bin [g] - the car's robust braking capability, ignoring single-sample spikes.",
    "Peak decel speed": "Speed bin [m/s] where that peak P95 deceleration occurs.",
    "Gap to max decel": "Peak P95 decel minus the 1.79 g vehicle max deceleration [g]. Negative = below the car's estimated limit.",
    "Max decel use": "Peak P95 decel as a percentage of the 1.79 g vehicle max deceleration.",
    "Blend threshold [%]": "Brake-pedal % where front hydraulic line pressure (BSEFront) first crosses 1 bar — the regen→hydraulic handover. Below it, braking is regen-only.",
    "Front peak P [bar]": "P95 front hydraulic line pressure (BSEFront) while braking — the robust peak the front circuit reaches.",
    "Rear peak P [bar]": "P95 rear hydraulic line pressure (BSERear) while braking.",
    "Brake front bias": "Mean front share of total braking force |Est_FX| (regen + hydraulic estimate). The measured front/rear split.",
    "Brake front bias ideal": "Mean front share the load-proportional ideal would use at the same decel and speed — the target split.",
    "Brake bias error": "Measured minus ideal front bias. Positive = front over-braked (front locks first); negative = rear over-braked.",
    "Dist to ideal brake [N]": "RMS distance from each braking sample to the ideal load-proportional curve in the front/rear force plane. Lower = closer to the optimal split.",
    "Peak combined brake [g]": "P95 of total braking force / weight — the robust peak combined deceleration the axles demand.",
    "Brake util front": "Median front-axle brake grip utilisation |Fx|/(mu*Fz). 1.0 = at the longitudinal grip limit.",
    "Brake util rear": "Median rear-axle brake grip utilisation |Fx|/(mu*Fz). 1.0 = at the longitudinal grip limit.",
    "Limiting axle (brake)": "Axle whose P95 brake utilisation is higher - the end closest to locking under braking.",
    "Front-rear util gap": "Front minus rear P95 brake utilisation. Positive = front runs closer to its grip limit (shift bias rearward); negative = rear over-braked.",
    "Brake slip front": "Median front-axle braking slip ratio (Est_SR). More negative = more slip; -0.20 optimal, -0.30 lock-up onset.",
    "Brake slip rear": "Median rear-axle braking slip ratio (Est_SR). More negative = more slip.",
    "Axle nearer lock-up": "Axle whose P5 braking slip is more negative - the end kinematically closest to lock-up.",
    # Dynamics · acceleration force balance
    "Peak accel [g]": "Peak longitudinal acceleration [g].",
    "Drive slip front": "Median front-axle drive slip ratio (Est_SR) on near-straight throttle. More positive = more slip; +0.20 optimal.",
    "Drive slip rear": "Median rear-axle drive slip ratio (Est_SR) on near-straight throttle. More positive = more slip; +0.20 optimal.",
    "Axle more slip": "Axle whose P95 drive slip is higher - the end kinematically spinning most under throttle.",
    # Dynamics · setup (LLTD)
    "Mean LLTD [%]": "Front share of lateral load transfer, mid-corner average.",
    "Min [%]": "Minimum per-lap LLTD front share.",
    "Max [%]": "Maximum per-lap LLTD front share.",
    "Span [pp]": "Per-lap LLTD spread [percentage points]. Large span = setup/behaviour change.",
    "Mean samples": "Mean mid-corner samples per lap behind the LLTD figure.",
    # Grip factors
    "Overall [G]": "Mean combined |G| over grip-limited samples (braking ∪ corner) [g] (Buurman method).",
    "Cornering [G]": "Mean |ay| over radius-detected corners (R<60 m) [g].",
    "Braking [G]": "Mean |ax| over the braking phase (ax<−1 m/s² & Brake>5) [g].",
    "Traction [G]": "Mean ax over corner-exit samples (in a corner with ax>0) [g].",
    # Powertrain
    "Mean net / lap [kWh]": "Net battery energy per lap (consumed − recovered).",
    "Total net [kWh]": "Net battery energy across selected laps.",
    "Consumed / lap [kWh]": "Energy drawn from the battery per lap.",
    "Recovered / lap [kWh]": "Energy regenerated per lap.",
    "Mean battery power [kW]": "Average DC battery power over the run.",
    "Overload [%]": "Share of run time with the inverter's OverloadActive flag set.",
    "IxT peak": "Peak i²t thermal load of the inverter (1.0 = budget exhausted).",
    "Torque P95 [Nm]": "95th-percentile delivered motor torque (limit 27.5 Nm).",
    "Speed P95 [rad/s]": "95th-percentile motor angular velocity.",
    "Torque sat [%]": "Share of drive samples with |torque| > 90% of the 27.5 Nm limit.",
    "Rev limited [%]": "Share of samples with motor speed > 95% of the rev-limit clamp.",
    "Mean dSoC / lap [%]": "Mean state-of-charge drop per lap.",
    "SoC start [%]": "State of charge at the start of the run.",
    "SoC end [%]": "State of charge at the end of the run.",
    "Min cell under load [V]": "Lowest plausible cell voltage while discharging > 20 A.",
    "Cell V at peak I [V]": "Minimum cell voltage at the moment of peak battery current.",
    "Peak batt Tmax [°C]": "Glitch-cleaned P99 of the hottest battery cell temperature.",
    "Motor slope [°C/lap]": "Per-lap heat-soak rate of the motor mean temperature.",
    # TC
    "Target met": "Whether slip-ratio tracking met the ±target band overall.",
    "In target [%]": "Share of armed samples with SR inside the target band. Higher is better.",
    "Median SR": "Median slip ratio while the controller is armed.",
    "Target gap [%]": "Median distance of SR from the +0.20 acceleration target.",
    "Failure mode": "Dominant tracking error (under / over target).",
    "Too low SR [%]": "Armed samples below the target band.",
    "Too high SR [%]": "Armed samples above the target band (overslip).",
    "TC response [%]": "Share of overslip events where TC cut torque in response.",
    "Worst wheel": "Wheel with the worst slip-ratio tracking.",
    # Dynamics · braking lock-up (relocated from RB, 2026-06-15)
    "Lock-up time [s]": "Total time per run all four wheels spend past the −0.30 lock-up line while braking. Higher = more lock-up.",
    "Lock-up events": "Count of sustained (≥0.05 s) braking lock-up segments across wheels.",
    "Worst SR": "Most negative Est_SR reached inside a sustained braking lock-up segment (−1 = fully locked).",
    # TV
    "Yaw RMSE": "RMS yaw-rate tracking error vs target [rad/s]. Lower is better.",
    "Yaw bias": "Mean signed yaw-rate error [rad/s]. ~0 = unbiased.",
    "Mz RMSE [Nm]": "RMS yaw-moment tracking error. Lower is better.",
    "Mz bias [Nm]": "Mean signed yaw-moment error.",
    "FB / FF ratio": "Feedback-to-feedforward magnitude ratio. High = controller working hard.",
    "FB share": "Share of total Mz coming from feedback.",
    # Driver · trail-braking / grip utilisation
    "Trail-braking overlap [%]": "Share of braking time with simultaneous steering input (brake carried into the corner).",
    "Track Speed Distribution": "Share of lap time at each speed (|VN_vx|→km/h) over all valid laps, normalised to %. A fingerprint of the track: a fat high-speed tail means aerodynamics play a bigger role here. One curve per run; pick which to show.",
    "Envelope [G]": "Grip ceiling on this circuit = P95 of combined |G| over valid laps.",
    "Utilisation [%]": "Mean combined |G| as a % of the grip envelope. Higher = working the tyres harder.",
    "Time at limit [%]": "Share of samples within 90% of the grip envelope.",
    "Braking TAL [%]": "Time at the limit during braking-phase samples.",
    "Cornering TAL [%]": "Time at the limit during cornering-phase samples.",
    "Traction TAL [%]": "Time at the limit during traction-phase samples.",
    # Controls intervention map
    "TV active [%]": "Share of lap samples where Torque Vectoring applies a yaw moment.",
    "TC active [%]": "Share of lap samples where Traction Control cuts drive torque.",
    "RB active [%]": "Share of lap samples where Regenerative Braking blends regen.",
    "TV saturated [%]": "Share of TV-active samples where |Mz| is within 95% of its limit (controller maxed).",
    "TC @ full throttle [%]": "Share of full-throttle samples where TC is cutting torque (driver vs control conflict).",
    "RB regen front bias": "Mean front share of measured REGEN braking force (motor torque only, hydraulic brakes unmeasured). Not the car's total brake balance.",
    "RB regen bias std": "Spread (pp) of the regen front-bias across braking samples - how consistent the regen split is.",
    "RB regen RMS to ideal": "RMS distance [N] from the measured regen force point to the ideal load-proportional braking curve at that speed.",
    "RB regen rear overbiased": "Share of braking samples where measured rear regen force exceeds the ideal rear force at that front force.",
    "RB regen peak brake": "Peak combined (front+rear) regen braking force expressed as deceleration [g].",
}
_PL_HASH_FUNCS = {pl.DataFrame: lambda _df: 0}
_TELEMETRY_REQUIRED_HEADER_COLS = {"TimeStamp"}
_TELEMETRY_SIGNAL_HEADER_COLS = {"laps", "VN_vx", "VN_latitude", "VN_longitude"}


# ── Data loading ──────────────────────────────────────────────────────────────


def _file_signature(path: Path) -> FileSignature:
    """Return a cache-busting signature for *path* based on its current stat()."""
    stat = path.stat()
    return (int(stat.st_mtime_ns), int(stat.st_size))


def _is_telemetry_csv(path: Path) -> bool:
    """Fast header-only check to ignore support CSVs stored under data/."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            header = fh.readline().strip().split(",")
    except OSError:
        return False
    header_cols = set(header)
    return _TELEMETRY_REQUIRED_HEADER_COLS.issubset(header_cols) and bool(
        _TELEMETRY_SIGNAL_HEADER_COLS & header_cols
    )


def _telemetry_csv_paths(data_dir: Path) -> list[Path]:
    """Return dashboard-loadable telemetry CSVs, excluding lookup/support files."""
    return sorted(path for path in data_dir.glob("*.csv") if _is_telemetry_csv(path))


def _format_csv_file_option(fname: str) -> str:
    """Show the complete CSV filename in run selectors."""
    return fname


def _video_dir_signature(repo_root: Path) -> tuple[tuple[str, int, int], ...]:
    """Return a lightweight cache token for available onboard videos."""
    videos_dir = repo_root / "videos"
    if not videos_dir.is_dir():
        return ()
    out: list[tuple[str, int, int]] = []
    for path in sorted(videos_dir.glob("*.mp4")):
        try:
            stat = path.stat()
        except OSError:
            continue
        out.append((path.name, int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(out)


@st.cache_resource(show_spinner="Loading run...")
def load_run(path: str, file_signature: FileSignature) -> pl.DataFrame:
    """Load a CSV run through the shared project loader, keeping all laps."""
    _ = file_signature
    return enrich_run_df(load_data(path, complete_laps_only=False))


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


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _select_laps_df_cached(
    df: pl.DataFrame,
    run_token: tuple[str, FileSignature],
    lap_ids: tuple[int, ...],
) -> pl.DataFrame:
    """Filter selected laps once per run/lap selection instead of every rerun."""
    _ = run_token
    return select_laps_df(df, list(lap_ids))


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _available_laps_cached(
    df: pl.DataFrame,
    run_token: tuple[str, FileSignature],
) -> tuple[int, ...]:
    """Cached lap IDs for sidebar selectors."""
    _ = run_token
    return tuple(int(lap) for lap in available_laps(df).tolist())


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _lap_laptimes_cached(
    df: pl.DataFrame,
    run_token: tuple[str, FileSignature],
) -> dict[int, float]:
    """Cached laptime labels for selectors."""
    _ = run_token
    return _lap_laptimes(df)


@st.cache_resource(show_spinner=False)
def _video_server_cached(
    repo_root: str,
    video_signature: tuple[tuple[str, int, int], ...],
) -> va.VideoServerInfo:
    """Start/reuse the video HTTP server; refresh only when video files change."""
    _ = video_signature
    return va.ensure_video_server(Path(repo_root))


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _video_payload_cached(
    df: pl.DataFrame,
    run_token: tuple[str, FileSignature],
) -> dict:
    """Build the heavy Video Analysis JSON payload once per raw CSV version."""
    _ = run_token
    return va.build_video_payload(df)


def _clear_data_caches() -> None:
    """Clear cached data derived from CSV contents."""
    cached_funcs = (
        load_run,
        load_lap_gate,
        csv_needs_lap_detection_cached,
        _select_laps_df_cached,
        _available_laps_cached,
        _lap_laptimes_cached,
        _video_server_cached,
        _video_payload_cached,
        _pt_energy_per_lap_fig_cached,
        _pt_power_per_wheel_fig_cached,
        _pt_soc_per_lap_fig_cached,
        _pt_thermal_evolution_fig_cached,
        _pt_inverter_limits_fig_cached,
        _pt_torque_fidelity_fig_cached,
        _pt_torque_speed_envelope_fig_cached,
        _pt_hv_delivery_efficiency_fig_cached,
        _pt_weakest_cell_fig_cached,
        _pt_thermal_headroom_fig_cached,
        _dyn_decel_envelope_fig_cached,
        _dyn_brake_blending_fig_cached,
        _dyn_ideal_braking_curve_fig_cached,
        _dyn_ideal_brake_distribution_fig_cached,
        _dyn_axle_brake_utilisation_fig_cached,
        _dyn_axle_brake_slip_fig_cached,
        _driver_max_braking_g_per_lap_fig_cached,
        _skidpad_fig_cached,
        _accel_fig_cached,
        _dyn_ideal_traction_curve_fig_cached,
        _dyn_accel_envelope_fig_cached,
        _dyn_traction_slip_curve_fig_cached,
        _dyn_axle_traction_utilisation_fig_cached,
        _dyn_cornering_balance_phase_fig_cached,
        _dyn_lateral_grip_envelope_fig_cached,
        _dyn_axle_lateral_utilisation_fig_cached,
        _driver_summary_cached,
        _driver_throttle_histogram_fig_cached,
        _driver_full_throttle_time_fig_cached,
        _driver_throttle_speed_fig_cached,
        _driver_braking_aggressiveness_fig_cached,
        _driver_brake_release_smoothness_fig_cached,
        _driver_steering_smoothness_fig_cached,
        _driver_steering_integral_fig_cached,
        _driver_steering_stability_fig_cached,
        _driver_corner_curvature_fig_cached,
        _driver_lap_time_progression_fig_cached,
        _driver_lap_time_distribution_fig_cached,
        _driver_run_phase_distribution_fig_cached,
        _driver_fastest_lap_speed_map_fig_cached,
        _driver_speed_distribution_fig_cached,
        _driver_sector_times_matrix_cached,
        _driver_cornering_turns_cached,
        _driver_cornering_metrics_cached,
        _driver_fastest_lap_cached,
        _driver_lap_sectors_cached,
        _driver_csv_sector_summary_cached,
        _driver_whole_lap_metrics_cached,
        _driver_potential_lap_cached,
    )
    for func in cached_funcs:
        clear = getattr(func, "clear", None)
        if callable(clear):
            clear()


def _consume_track_component_event(
    track_event: dict[str, object] | None,
    *,
    pool_len: int,
    event_state_key: str,
) -> None:
    """Persist track-map lasso/manual-line interactions from the custom component."""
    if not isinstance(track_event, dict):
        return

    event_id = int(track_event.get("event_id", 0) or 0)
    last_event_id = int(st.session_state.get(event_state_key, -1))
    if event_id <= last_event_id:
        return
    st.session_state[event_state_key] = event_id

    selection_indices = np.asarray(track_event.get("selection_indices", []), dtype=int)
    valid_sel = selection_indices[(selection_indices >= 0) & (selection_indices < pool_len)]
    st.session_state["_dyn_track_selection_indices"] = np.unique(valid_sel).tolist()

    if bool(track_event.get("fullscreen_event", False)):
        st.session_state["_dyn_track_open_fullscreen"] = True
        st.rerun()

    if not bool(track_event.get("line_event", False)):
        return

    line_payload = track_event.get("line")
    if isinstance(line_payload, dict):
        try:
            manual_gate_line = (
                (float(line_payload["x0"]), float(line_payload["y0"])),
                (float(line_payload["x1"]), float(line_payload["y1"])),
            )
        except (KeyError, TypeError, ValueError):
            manual_gate_line = st.session_state.get("_dyn_manual_gate_line")
        else:
            st.session_state["_dyn_manual_gate_line"] = manual_gate_line
            # Stamp each draw with the originating track-component event_id
            # so downstream consumers can detect "already used" gates.
            st.session_state["_dyn_manual_gate_event_id"] = event_id
            st.session_state.pop("_dyn_manual_gate_consumed_event_id", None)
    else:
        st.session_state.pop("_dyn_manual_gate_line", None)
        st.session_state.pop("_dyn_manual_gate_event_id", None)
        st.session_state.pop("_dyn_manual_gate_consumed_event_id", None)
    st.rerun()


def _consume_lap_turn_click_event(
    track_event: dict[str, object] | None,
    *,
    all_turn_ids: list[int],
    included_state_key: str,
    event_state_key: str,
) -> bool:
    """Toggle a Lap Analysis turn when the track-map component reports a click."""
    if not isinstance(track_event, dict):
        return False

    event_id = int(track_event.get("event_id", 0) or 0)
    last_event_id = int(st.session_state.get(event_state_key, -1))
    if event_id <= last_event_id:
        return False
    st.session_state[event_state_key] = event_id

    clicked = track_event.get("clicked_turn_id")
    if clicked is None:
        return False
    try:
        turn_id = int(clicked)
    except (TypeError, ValueError):
        return False
    valid_turns = {int(tid) for tid in all_turn_ids}
    if turn_id not in valid_turns:
        return False

    included = {
        int(tid)
        for tid in st.session_state.get(included_state_key, all_turn_ids)
        if int(tid) in valid_turns
    }
    if turn_id in included:
        included.remove(turn_id)
    else:
        included.add(turn_id)
    st.session_state[included_state_key] = sorted(included)
    return True


def _add_manual_gate_line_to_fig(
    fig: go.Figure,
    manual_gate_line: tuple[tuple[float, float], tuple[float, float]] | None,
) -> go.Figure:
    """Overlay the manual finish-line preview on a longitude/latitude figure."""
    if manual_gate_line is None:
        return fig
    (x0, y0), (x1, y1) = manual_gate_line
    fig.add_shape(
        type="line",
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        xref="x",
        yref="y",
        line=dict(color="#4DB3F2", width=3),
    )
    fig.add_trace(
        go.Scattergl(
            x=[x0, x1],
            y=[y0, y1],
            mode="markers",
            marker=dict(color="#4DB3F2", size=8, symbol="circle"),
            name="Manual finish line",
            hovertemplate="Manual finish line<extra></extra>",
            showlegend=False,
        )
    )
    return fig


def _lap_gates_from_run_tokens(
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> dict[str, dict]:
    """Load lapcount finish-gate metadata for the selected CSV runs."""
    lap_gates: dict[str, dict] = {}
    for run_name, file_signature, _lap_token in run_tokens:
        try:
            gate = load_lap_gate(str(DATA_DIR / run_name), file_signature)
        except Exception:
            continue
        if not gate:
            continue
        try:
            gate_lon = np.asarray(gate["gate_lon"], dtype=float)
            gate_lat = np.asarray(gate["gate_lat"], dtype=float)
            finish_lon = float(gate["finish_lon"])
            finish_lat = float(gate["finish_lat"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            gate_lon.size == 2
            and gate_lat.size == 2
            and np.all(np.isfinite(gate_lon))
            and np.all(np.isfinite(gate_lat))
            and np.isfinite(finish_lon)
            and np.isfinite(finish_lat)
        ):
            lap_gates[run_name] = gate
    return lap_gates


def _add_lap_detection_gates_to_fig(
    fig: go.Figure,
    lap_gates: dict[str, dict],
) -> go.Figure:
    """Overlay lapcount-detected finish lines on a longitude/latitude figure."""
    if not lap_gates:
        return fig

    multi_run = len(lap_gates) > 1
    for run_name, gate in lap_gates.items():
        gate_lon = np.asarray(gate["gate_lon"], dtype=float)
        gate_lat = np.asarray(gate["gate_lat"], dtype=float)
        finish_lon = float(gate["finish_lon"])
        finish_lat = float(gate["finish_lat"])
        gate_half_width_m = float(gate.get("gate_half_width_m", np.nan))
        mode = str(gate.get("lapcount_mode", "circuit"))
        color = driver_color(run_name) if multi_run else "#F2F2F2"
        is_autocross = mode == "autocross"
        gate_kind = "start" if is_autocross else "finish"
        label = (
            f"Lapcount {gate_kind} · {Path(run_name).stem}"
            if multi_run
            else f"Lapcount {gate_kind}"
        )

        fig.add_trace(
            go.Scattergl(
                x=gate_lon,
                y=gate_lat,
                mode="lines",
                name=label,
                line=dict(color=color, width=2.5, dash="dash"),
                hovertemplate=(
                    f"{label}<br>mode={mode}<br>half width={gate_half_width_m:.1f} m<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=[finish_lon],
                y=[finish_lat],
                mode="markers",
                marker=dict(
                    size=10,
                    color=color,
                    symbol="x",
                    line=dict(color="#FFFFFF", width=1.5),
                ),
                showlegend=False,
                hovertemplate=(
                    f"{label} centre<br>lon={finish_lon:.6f}<br>lat={finish_lat:.6f}<extra></extra>"
                ),
            )
        )
        # Autocross also stores a finish line (gate2); draw it distinctly.
        if is_autocross and gate.get("gate2_lon") is not None:
            g2_lon = np.asarray(gate["gate2_lon"], dtype=float)
            g2_lat = np.asarray(gate["gate2_lat"], dtype=float)
            g2_label = (
                f"Lapcount finish · {Path(run_name).stem}" if multi_run else "Lapcount finish"
            )
            fig.add_trace(
                go.Scattergl(
                    x=g2_lon,
                    y=g2_lat,
                    mode="lines",
                    name=g2_label,
                    line=dict(color=color, width=2.5, dash="dot"),
                    hovertemplate=f"{g2_label}<extra></extra>",
                )
            )
    return fig


@st.dialog("Lap Analysis Phase Map", width="large")
def _render_lap_phase_fullscreen_dialog(
    phase_fig: go.Figure,
    *,
    turn_ids: list[int],
    included_turns_key: str,
    event_state_key: str,
) -> None:
    """Render the Lap Analysis phase map in a large dialog."""
    st.caption("Click a curve to include/exclude it from Lap Analysis.")
    phase_event = tmc.render_track_map_component(
        tmc.serialize_figure(phase_fig),
        height_px=760,
        key=f"drv_lap_phase_map_fullscreen_{event_state_key}",
        draw_enabled=False,
    )
    if _consume_lap_turn_click_event(
        phase_event,
        all_turn_ids=turn_ids,
        included_state_key=included_turns_key,
        event_state_key=f"{event_state_key}_fullscreen",
    ):
        st.rerun()


def _peek_persisted_mode_and_gate(
    csv_path: Path,
) -> tuple[str | None, tuple[tuple[float, float], tuple[float, float]] | None, bool]:
    """Read the previously written event mode, centre gate and manual flag.

    Returns (mode_str_or_None, gate_or_None, manual_gate). ``mode_str`` is
    ``None`` for circuit/auto so the caller can fall back to plain auto
    detection. ``manual_gate`` is True when the persisted gate came from a
    user-drawn line (lapcount stores ``min_vel = NaN`` for manual circuit and
    skidpad paths) and must therefore survive a re-detection.
    """
    candidate_cols = (
        "lapcount_mode",
        "lapcount_gate_lon0_deg",
        "lapcount_gate_lat0_deg",
        "lapcount_gate_lon1_deg",
        "lapcount_gate_lat1_deg",
        "lapcount_min_vel_mps",
    )
    try:
        header = pl.read_csv(str(csv_path), n_rows=0).columns
    except Exception:
        return None, None, False
    available = [c for c in candidate_cols if c in header]
    if "lapcount_mode" not in available:
        return None, None, False
    try:
        df = pl.read_csv(str(csv_path), columns=available)
    except Exception:
        return None, None, False
    mode = _current_event_mode_label(df)
    mode_str: str | None = None if mode == "Endurance" else mode.lower()
    gate = _stored_gate_line(df)
    manual = False
    if "lapcount_min_vel_mps" in available:
        vals = df["lapcount_min_vel_mps"].drop_nulls()
        if len(vals) > 0:
            try:
                manual = not np.isfinite(float(vals[0]))
            except (TypeError, ValueError):
                manual = False
    return mode_str, gate, manual


def _autodetect_laps(data_dir: Path) -> None:
    """Run lap detection on any CSV in *data_dir* that doesn't have it yet.

    When a CSV already carries a previously chosen event mode (e.g. skidpad
    after a manual override), preserve that mode and the saved gate so a
    lapcount algorithm bump cannot silently demote the session back to
    circuit.
    """
    modified = False
    for path in _telemetry_csv_paths(data_dir):
        try:
            file_signature = _file_signature(path)
            if not csv_needs_lap_detection_cached(str(path), file_signature):
                continue
        except Exception as exc:
            st.sidebar.warning(f"`{path.name}`: cannot inspect — {exc}")
            continue
        persisted_mode, persisted_gate, manual_gate = _peek_persisted_mode_and_gate(path)
        with st.spinner(f"Detecting laps in {path.name}..."):
            try:
                if persisted_mode == "skidpad":
                    # Pass the stored gate (may be None); detect_and_write_laps
                    # falls back to GPS-based auto-estimation when the gate is
                    # missing or stale.
                    n = lapcount.detect_and_write_laps(
                        path,
                        mode="skidpad",
                        gate_line_lonlat=persisted_gate,
                    )
                elif persisted_mode == "acceleration":
                    n = lapcount.detect_and_write_laps(path, mode="acceleration")
                elif persisted_mode == "autocross":
                    # Preserve both user-drawn lines across an algorithm bump.
                    g2_cols = (
                        "lapcount_gate2_lon0_deg",
                        "lapcount_gate2_lat0_deg",
                        "lapcount_gate2_lon1_deg",
                        "lapcount_gate2_lat1_deg",
                    )
                    header = pl.read_csv(str(path), n_rows=0).columns
                    present = [c for c in g2_cols if c in header]
                    finish_gate = (
                        _stored_autocross_finish_line(pl.read_csv(str(path), columns=present))
                        if len(present) == len(g2_cols)
                        else None
                    )
                    if persisted_gate is None or finish_gate is None:
                        raise ValueError("autocross lines missing — redraw start & finish")
                    n = lapcount.detect_and_write_laps(
                        path,
                        mode="autocross",
                        gate_line_lonlat=persisted_gate,
                        finish_gate_line_lonlat=finish_gate,
                    )
                elif manual_gate and persisted_gate is not None:
                    # Circuit run with a user-drawn finish line. Preserve the
                    # manual gate; otherwise auto-fit would silently relocate
                    # the start/finish line on every algorithm version bump.
                    n = lapcount.detect_and_write_laps(
                        path,
                        gate_line_lonlat=persisted_gate,
                    )
                else:
                    n = lapcount.detect_and_write_laps(path)
            except Exception as exc:
                st.sidebar.warning(f"`{path.name}`: lap detection failed — {exc}")
                continue
        modified = True
        if persisted_mode:
            mode_label = persisted_mode
        elif manual_gate and persisted_gate is not None:
            mode_label = "circuit · manual gate"
        else:
            mode_label = "auto"
        if n > 0:
            st.sidebar.info(f"`{path.name}`: detected {n} laps ({mode_label})")
        else:
            st.sidebar.warning(f"`{path.name}`: no laps detected from GPS ({mode_label})")
    if modified:
        _clear_data_caches()


_EVENT_MODE_LABELS: tuple[str, ...] = ("Endurance", "Autocross", "Acceleration", "Skidpad")
_EVENT_MODE_TO_LABEL: dict[str, str] = {
    "circuit": "Endurance",
    "auto": "Endurance",
    "endurance": "Endurance",
    "autocross": "Autocross",
    "acceleration": "Acceleration",
    "accel": "Acceleration",
    "skidpad": "Skidpad",
}


def _current_event_mode_label(raw_df: pl.DataFrame | None) -> str:
    """Return the human-readable event mode currently stored in the CSV."""
    if raw_df is None or "lapcount_mode" not in raw_df.columns:
        return "Endurance"
    values = raw_df["lapcount_mode"].drop_nulls()
    if len(values) == 0:
        return "Endurance"
    return _EVENT_MODE_TO_LABEL.get(str(values[0]).strip().lower(), "Endurance")


def _fresh_manual_gate_line() -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return the session manual gate only if it has not been consumed yet.

    A single drawn line carries the originating ``event_id`` of the track
    component event. Each consuming action (mode switch, "Apply To All CSVs",
    ...) marks the same event_id as consumed. Subsequent consumers see the
    stamp matches and treat the gate as stale, which prevents e.g. a circuit
    finish-line drawn for a prior "Apply To All CSVs" from silently being
    reused as a skidpad centre-gate when the event-mode selector changes.
    """
    gate = st.session_state.get("_dyn_manual_gate_line")
    event_id = st.session_state.get("_dyn_manual_gate_event_id")
    consumed = st.session_state.get("_dyn_manual_gate_consumed_event_id")
    if gate is None or event_id is None or event_id == consumed:
        return None
    return gate


def _mark_manual_gate_consumed() -> None:
    """Record that the current manual gate has been used by some action."""
    event_id = st.session_state.get("_dyn_manual_gate_event_id")
    if event_id is not None:
        st.session_state["_dyn_manual_gate_consumed_event_id"] = event_id


def _stored_gate_line(
    raw_df: pl.DataFrame | None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Recover a previously written gate line (lon, lat) from CSV metadata."""
    if raw_df is None:
        return None
    cols = (
        "lapcount_gate_lon0_deg",
        "lapcount_gate_lat0_deg",
        "lapcount_gate_lon1_deg",
        "lapcount_gate_lat1_deg",
    )
    if not all(c in raw_df.columns for c in cols):
        return None
    try:
        lon0 = float(raw_df[cols[0]].drop_nulls()[0])
        lat0 = float(raw_df[cols[1]].drop_nulls()[0])
        lon1 = float(raw_df[cols[2]].drop_nulls()[0])
        lat1 = float(raw_df[cols[3]].drop_nulls()[0])
    except (IndexError, ValueError, TypeError):
        return None
    if not all(np.isfinite(v) for v in (lon0, lat0, lon1, lat1)):
        return None
    return ((lon0, lat0), (lon1, lat1))


def _stored_autocross_finish_line(
    raw_df: pl.DataFrame | None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Recover the autocross finish line (gate2) from CSV metadata."""
    if raw_df is None:
        return None
    cols = (
        "lapcount_gate2_lon0_deg",
        "lapcount_gate2_lat0_deg",
        "lapcount_gate2_lon1_deg",
        "lapcount_gate2_lat1_deg",
    )
    if not all(c in raw_df.columns for c in cols):
        return None
    try:
        lon0 = float(raw_df[cols[0]].drop_nulls()[0])
        lat0 = float(raw_df[cols[1]].drop_nulls()[0])
        lon1 = float(raw_df[cols[2]].drop_nulls()[0])
        lat1 = float(raw_df[cols[3]].drop_nulls()[0])
    except (IndexError, ValueError, TypeError):
        return None
    if not all(np.isfinite(v) for v in (lon0, lat0, lon1, lat1)):
        return None
    return ((lon0, lat0), (lon1, lat1))


def _redetect_with_event_mode(
    csv_path: Path,
    label: str,
    *,
    gate_line: tuple[tuple[float, float], tuple[float, float]] | None,
) -> tuple[bool, str]:
    """Re-run lapcount with the user-selected event mode. Returns (ok, message)."""
    label = (label or "Endurance").strip()
    if label == "Endurance":
        try:
            n = lapcount.detect_and_write_laps(str(csv_path))
        except Exception as exc:
            return False, f"`{csv_path.name}`: endurance detection failed — {exc}"
        return True, f"`{csv_path.name}`: {n} laps (endurance)"
    if label == "Acceleration":
        try:
            n = lapcount.detect_and_write_laps(str(csv_path), mode="acceleration")
        except Exception as exc:
            return False, f"`{csv_path.name}`: acceleration detection failed — {exc}"
        return True, f"`{csv_path.name}`: {n} laps (acceleration)"
    if label == "Skidpad":
        # gate_line may be None — detect_and_write_laps falls back to a GPS-
        # based auto-gate when no gate is provided or the supplied gate yields
        # no plausible laps.
        try:
            n = lapcount.detect_and_write_laps(
                str(csv_path),
                mode="skidpad",
                gate_line_lonlat=gate_line,
            )
        except Exception as exc:
            return False, f"`{csv_path.name}`: skidpad detection failed — {exc}"
        suffix = "skidpad" if gate_line is not None else "skidpad, auto-gate"
        return True, f"`{csv_path.name}`: {n} laps ({suffix})"
    if label == "Autocross":
        # Autocross needs two manually drawn lines; it is handled by the
        # dedicated two-step panel, not the selectbox auto-redetect path.
        return True, f"`{csv_path.name}`: switched to Autocross — draw start & finish lines."
    return False, f"`{csv_path.name}`: unknown mode {label!r}"


def _detect_autocross_lap(
    csv_path: Path,
    start_line: tuple[tuple[float, float], tuple[float, float]],
    finish_line: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[bool, str]:
    """Run autocross detection with an explicit start and finish line."""
    try:
        n = lapcount.detect_and_write_laps(
            str(csv_path),
            mode="autocross",
            gate_line_lonlat=start_line,
            finish_gate_line_lonlat=finish_line,
        )
    except Exception as exc:
        return False, f"`{csv_path.name}`: autocross detection failed — {exc}"
    if n < 1:
        return False, f"`{csv_path.name}`: no autocross lap found between the two lines."
    return True, f"`{csv_path.name}`: autocross lap detected."


def _render_autocross_panel(fname: str, raw_df: pl.DataFrame | None) -> None:
    """Two-step start/finish line panel for a file in Autocross mode."""
    start_key = f"_autocross_start_{fname}"
    finish_key = f"_autocross_finish_{fname}"

    # Seed session slots from previously persisted lines on first render, but
    # only when the CSV was already detected as autocross — otherwise a circuit
    # file's finish line would wrongly pre-fill the start slot.
    already_autocross = _current_event_mode_label(raw_df) == "Autocross"
    if start_key not in st.session_state:
        st.session_state[start_key] = _stored_gate_line(raw_df) if already_autocross else None
    if finish_key not in st.session_state:
        st.session_state[finish_key] = (
            _stored_autocross_finish_line(raw_df) if already_autocross else None
        )

    start_line = st.session_state.get(start_key)
    finish_line = st.session_state.get(finish_key)

    st.sidebar.caption(
        f"`{fname}` · Autocross — draw a line on the Overview › Circuit map, "
        "then assign it as start or finish."
    )
    st.sidebar.markdown(
        f"- Start line: {'✓' if start_line is not None else '—'}\n"
        f"- Finish line: {'✓' if finish_line is not None else '—'}"
    )

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("Set start line", key=f"set_start_{fname}"):
        fresh = _fresh_manual_gate_line()
        if fresh is None:
            st.sidebar.error("Draw a line on the Overview › Circuit map first.")
        else:
            st.session_state[start_key] = fresh
            _mark_manual_gate_consumed()
            st.rerun()
    if col_b.button("Set finish line", key=f"set_finish_{fname}"):
        fresh = _fresh_manual_gate_line()
        if fresh is None:
            st.sidebar.error("Draw a line on the Overview › Circuit map first.")
        else:
            st.session_state[finish_key] = fresh
            _mark_manual_gate_consumed()
            st.rerun()

    ready = start_line is not None and finish_line is not None
    if st.sidebar.button(
        "Detect autocross lap", key=f"detect_autocross_{fname}", disabled=not ready
    ):
        with st.spinner("Detecting autocross lap..."):
            ok, msg = _detect_autocross_lap(DATA_DIR / fname, start_line, finish_line)
        (st.sidebar.success if ok else st.sidebar.error)(msg)
        if ok:
            _clear_data_caches()
            st.rerun()


def _render_event_mode_selector(
    selected_files: list[str],
    raw_dfs: dict[str, pl.DataFrame],
) -> None:
    """Sidebar selector to override the lapcount event mode per CSV."""
    if not selected_files:
        return
    st.sidebar.divider()
    st.sidebar.markdown("### Event mode")
    st.sidebar.caption(
        "How lapcount segments laps. Skidpad needs a centre-gate line drawn "
        "on the Driver › Overview › Circuit map."
    )
    pending_changes: list[tuple[Path, str]] = []
    for fname in selected_files:
        raw_df = raw_dfs.get(fname)
        current_label = _current_event_mode_label(raw_df)
        widget_key = f"event_mode_{fname}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = current_label
        elif st.session_state[widget_key] not in _EVENT_MODE_LABELS:
            st.session_state[widget_key] = current_label

        selected_label = st.sidebar.selectbox(
            fname,
            options=_EVENT_MODE_LABELS,
            key=widget_key,
        )
        if selected_label == "Autocross":
            # Autocross is driven by an explicit two-step line panel, not the
            # auto-redetect-on-change path the other modes use.
            _render_autocross_panel(fname, raw_df)
            continue
        if selected_label != current_label:
            pending_changes.append((DATA_DIR / fname, selected_label))

    if not pending_changes:
        return

    messages: list[tuple[bool, str]] = []
    any_success = False
    # Only forward the session manual gate when it has not yet been used by
    # another action (e.g. a previous "Apply To All CSVs" that consumed it
    # as a circuit finish-line). The gate's event_id encodes "this draw";
    # a user who wants the same line re-applied to a different mode must
    # redraw it. Do NOT fall back to the CSV's persisted gate for any
    # target mode either: gates persisted from circuit/auto runs are
    # finish-lines, not centre-gates, and would silently feed the wrong
    # geometry to skidpad detection.
    fresh_gate = _fresh_manual_gate_line()
    consumed_fresh_gate = False
    with st.spinner("Re-detecting laps with the new event mode..."):
        for csv_path, label in pending_changes:
            gate_line = fresh_gate
            ok, msg = _redetect_with_event_mode(csv_path, label, gate_line=gate_line)
            messages.append((ok, msg))
            if ok:
                any_success = True
                if gate_line is not None:
                    consumed_fresh_gate = True
            else:
                st.session_state[f"event_mode_{csv_path.name}"] = _current_event_mode_label(
                    raw_dfs.get(csv_path.name)
                )
    if consumed_fresh_gate:
        _mark_manual_gate_consumed()
    for ok, msg in messages:
        (st.sidebar.success if ok else st.sidebar.error)(msg)
    if any_success:
        _clear_data_caches()
        st.rerun()


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


def _section_per_lap_axis(section_key: str, default: str = "laps") -> str:
    """Render one per-lap x-axis selector for a whole section and return its mode.

    Sections with several stacked per-lap charts (e.g. Powertrain) used to repeat
    an identical "X-axis" radio above every plot. This renders a single control
    at the top of the section; all charts in that section then read the shared
    mode, so the axis stays consistent and the UI is far less cluttered.
    """
    return _select_per_lap_axis(f"axis_{section_key}", default)


def _format_lap_label(lap_id: int) -> str:
    """Human-readable label for sidebar and plot selectors."""
    lap_int = int(lap_id)
    return "Lap 0 (formation)" if lap_int == 0 else f"Lap {lap_int}"


def _lap_laptimes(df: pl.DataFrame) -> dict[int, float]:
    """Return max detected laptime per lap for selector labels."""
    if "laps" not in df.columns or "laptime" not in df.columns:
        return {}
    cols = cols_to_numpy(df, ["laps", "laptime"])
    laps = cols["laps"]
    laptime = cols["laptime"]
    out: dict[int, float] = {}
    for lap_id in available_laps(df).tolist():
        mask = laps == float(lap_id)
        if mask.any() and np.any(np.isfinite(laptime[mask])):
            out[int(lap_id)] = float(np.nanmax(laptime[mask]))
        else:
            out[int(lap_id)] = np.nan
    return out


def _format_lap_with_laptime(lap_id: int, lap_times: dict[int, float]) -> str:
    """Human-readable lap label including laptime when available."""
    base = _format_lap_label(lap_id)
    lt = lap_times.get(int(lap_id), np.nan)
    return f"{base} ({lt:.2f} s)" if np.isfinite(lt) else base


def _lap_signature(df: pl.DataFrame) -> str:
    """Compact signature of the currently selected laps in *df*."""
    return ",".join(str(int(lap)) for lap in available_laps(df))


def _run_color_map(run_names: list[str]) -> dict[str, str]:
    """Stable color per run for multi-run overlays (shared driver-identity palette)."""
    return {run_name: driver_color(run_name) for run_name in run_names}


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
    run_idx: int,
    variant_idx: int,
) -> go.BaseTraceType:
    """Clone *trace* and restyle it for a specific run overlay."""
    out = copy.deepcopy(trace)
    base_name = getattr(out, "name", "") or ""
    out.name = f"{run_name} · {base_name}" if base_name else run_name
    out.legendgroup = run_name

    dash = TRACE_DASHES[variant_idx % len(TRACE_DASHES)]
    symbol = TRACE_SYMBOLS[variant_idx % len(TRACE_SYMBOLS)]
    run_dash = TRACE_DASHES[run_idx % len(TRACE_DASHES)]
    run_symbol = TRACE_SYMBOLS[run_idx % len(TRACE_SYMBOLS)]
    wheel = _wheel_token(base_name)
    trace_color = WHEEL_COLORS[wheel] if wheel is not None else run_color
    trace_type = getattr(out, "type", "") or ""
    run_label = Path(run_name).stem
    is_ideal_share_line = base_name == "ideal (Fz = braking)"
    is_fz_reference_line = base_name == "Fz-proportional reference"
    mode = getattr(out, "mode", "") or ""
    is_wheel_line = (
        wheel is not None
        and trace_type in {"scatter", "scattergl"}
        and "lines" in mode
        and "markers" not in mode
    )

    if (
        hasattr(out, "line")
        and out.line is not None
        and not is_ideal_share_line
        and not is_fz_reference_line
    ):
        out.line.color = trace_color
        if trace_type != "box" and mode != "markers":
            out.line.dash = run_dash if is_wheel_line else dash
    elif hasattr(out, "line") and out.line is not None and is_fz_reference_line:
        out.line.dash = run_dash
    if hasattr(out, "marker") and out.marker is not None:
        out.marker.color = trace_color
        if wheel is not None:
            out.marker.line.color = run_color
            out.marker.line.width = 2
            if trace_type in {"scatter", "scattergl"}:
                out.marker.symbol = run_symbol
        elif getattr(out, "type", "") not in {"bar", "histogram", "barpolar", "bar3d"}:
            out.marker.symbol = symbol
    if hasattr(out, "textfont") and out.textfont is not None and wheel is not None:
        out.textfont.color = run_color

    x_vals = getattr(out, "x", None)
    if (
        wheel is not None
        and x_vals is not None
        and len(x_vals) > 0
        and all(isinstance(x_val, str) for x_val in x_vals)
    ):
        out.x = [f"{run_label} · {str(x_val)}" for x_val in x_vals]
    elif (
        trace_type == "box"
        and x_vals is not None
        and len(x_vals) > 0
        and all(isinstance(x_val, str) for x_val in x_vals)
    ):
        out.x = np.full(len(x_vals), run_label)
    elif (
        trace_type == "bar"
        and x_vals is not None
        and len(x_vals) > 0
        and all(isinstance(x_val, str) for x_val in x_vals)
    ):
        out.x = [f"{run_label} · {str(x_val)}" for x_val in x_vals]
    elif base_name == "Lockup threshold" and getattr(out, "x", None) is not None:
        out.x = [f"{run_label} · {str(x)}" for x in out.x]
    elif base_name == "Yaw instability threshold" and getattr(out, "x", None) is not None:
        out.x = [run_label for _ in out.x]
    elif base_name == "F/R summary" and getattr(out, "x", None) is not None:
        out.x = [f"{run_label} · FR" for _ in out.x]
    return out


def _figure_has_visible_legend(fig: go.Figure) -> bool:
    """Return True when Plotly is expected to render legend entries."""
    if fig.layout.showlegend is False:
        return False
    if fig.layout.showlegend is True:
        return True

    candidates = []
    for trace in fig.data:
        if getattr(trace, "showlegend", None) is False:
            continue
        if getattr(trace, "visible", None) is False:
            continue
        candidates.append(trace)
        if getattr(trace, "showlegend", None) is True:
            return True
    return len(candidates) > 1


def _selected_legend_runs() -> list[tuple[str, str]]:
    """Return selected CSV names and stems used to split multi-run legends."""
    try:
        selected_files = list(st.session_state.get("selected_csv_files", []))
    except Exception:
        selected_files = []
    return [(fname, Path(fname).stem) for fname in selected_files]


def _trace_legend_run(
    trace_name: str, selected_runs: list[tuple[str, str]]
) -> tuple[str, str] | None:
    """Find the selected CSV represented by a trace name."""
    if not trace_name:
        return None
    for fname, stem in selected_runs:
        labels = (fname, stem)
        for label in labels:
            if (
                trace_name == label
                or trace_name.startswith(f"{label} · ")
                or trace_name.startswith(f"{label} - ")
                or trace_name.startswith(f"{label} — ")
                or trace_name.endswith(f"· {label}")
                or trace_name.endswith(f"- {label}")
                or trace_name.endswith(f"— {label}")
            ):
                return fname, stem
    return None


def _trim_trace_run_prefix(trace: go.BaseTraceType, run_file: str, run_stem: str) -> None:
    """Keep per-CSV legend rows compact by removing the repeated run prefix."""
    name = getattr(trace, "name", "") or ""
    for prefix in (run_file, run_stem):
        for separator in (" · ", " - ", " — "):
            full_prefix = f"{prefix}{separator}"
            if name.startswith(full_prefix):
                trace.name = name.removeprefix(full_prefix)
                return


def _split_legend_by_csv_rows(fig: go.Figure) -> int:
    """Place selected CSV traces in one horizontal legend row per CSV."""
    selected_runs = _selected_legend_runs()
    if len(selected_runs) < 2:
        return 1

    used_runs: list[tuple[str, str]] = []
    assigned_run_by_trace: dict[int, tuple[str, str]] = {}
    trace_count_by_run: dict[tuple[str, str], int] = {}
    for idx, trace in enumerate(fig.data):
        if getattr(trace, "showlegend", None) is False:
            continue
        run = _trace_legend_run(str(getattr(trace, "name", "") or ""), selected_runs)
        if run is None:
            continue
        assigned_run_by_trace[idx] = run
        trace_count_by_run[run] = trace_count_by_run.get(run, 0) + 1
        if run not in used_runs:
            used_runs.append(run)

    if len(used_runs) < 2:
        return 1

    if all(trace_count_by_run.get(run, 0) == 1 for run in used_runs):
        for idx, (run_file, _run_stem) in assigned_run_by_trace.items():
            fig.data[idx].name = run_file
            fig.data[idx].legendgroup = run_file
        return 1

    legend_ids = ["legend"] + [f"legend{idx}" for idx in range(2, len(used_runs) + 1)]
    row_step = 0.12
    bottom_row_y = 1.075 if len(fig.layout.annotations) > 0 else 1.015
    y_positions = [
        bottom_row_y + row_step * (len(used_runs) - 1 - idx) for idx in range(len(used_runs))
    ]
    for row_idx, ((run_file, run_stem), legend_id, y_pos) in enumerate(
        zip(used_runs, legend_ids, y_positions)
    ):
        legend_layout = dict(
            title=dict(text=run_file),
            orientation="h",
            yanchor="bottom",
            y=y_pos,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(20,20,23,0.85)",
            bordercolor="rgba(128,128,128,0.3)",
            font=dict(size=PLOT_FONT_SIZE, family=FONT_FAMILY),
            title_font=dict(size=PLOT_FONT_SIZE, family=FONT_FAMILY),
            traceorder="normal",
        )
        if row_idx == 0:
            fig.update_layout(legend=legend_layout)
        else:
            fig.update_layout(**{legend_id: legend_layout})

    for idx, trace in enumerate(fig.data):
        run = assigned_run_by_trace.get(idx)
        if run is None:
            continue
        legend_idx = used_runs.index(run)
        trace.legend = legend_ids[legend_idx]
        _trim_trace_run_prefix(trace, *run)
    return len(used_runs)


def _place_legend_above_plot(
    fig: go.Figure,
    *,
    preserve_legend: bool = False,
) -> go.Figure:
    """Move Plotly trace selectors above charts so they do not steal plot width."""
    if preserve_legend:
        return fig

    has_plot_annotations = len(fig.layout.annotations) > 0
    fig.update_layout(
        title=dict(y=0.985, yref="container", yanchor="top", pad=dict(b=12)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12 if has_plot_annotations else 1.0,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(20,20,23,0.85)",
            bordercolor="rgba(128,128,128,0.3)",
        ),
    )
    if _figure_has_visible_legend(fig):
        legend_rows = _split_legend_by_csv_rows(fig)
        current_top = fig.layout.margin.t if fig.layout.margin.t is not None else 55
        # Figures with no in-figure title (section header rendered by Streamlit)
        # only need room for the legend rows — not the ~80px a title would take.
        has_title = bool((fig.layout.title.text or "").strip())
        if has_title:
            # Title is anchored to the container top; the legend rows are anchored
            # just above the plot. Keep the top margin only as tall as title +
            # legend rows actually need, so the legend/plot rise to sit right under
            # the title instead of leaving a dead band between them.
            required_top = (
                (130 if legend_rows == 1 else 150 + 34 * max(0, legend_rows - 2))
                if has_plot_annotations
                else (95 if legend_rows == 1 else 125 + 34 * max(0, legend_rows - 2))
            )
        else:
            required_top = (
                (105 if legend_rows == 1 else 125 + 28 * max(0, legend_rows - 2))
                if has_plot_annotations
                else (75 if legend_rows == 1 else 100 + 28 * max(0, legend_rows - 2))
            )
        if current_top < required_top:
            fig.update_layout(margin=dict(t=required_top))
    return fig


def _font_size_or_min(current_size: object, minimum_size: int) -> int:
    """Return *current_size* unless it is missing or smaller than *minimum_size*."""
    try:
        value = int(current_size)
    except (TypeError, ValueError):
        return minimum_size
    return max(value, minimum_size)


def _enforce_readable_plot_fonts(
    fig: go.Figure,
    *,
    preserve_legend: bool = False,
) -> go.Figure:
    """Raise non-title Plotly text to document-readable sizes before rendering."""
    fig.update_layout(
        font=dict(
            size=_font_size_or_min(fig.layout.font.size, PLOT_FONT_SIZE),
            family=fig.layout.font.family or FONT_FAMILY,
        ),
        title=dict(
            font=dict(
                size=_font_size_or_min(fig.layout.title.font.size, PLOT_FONT_SIZE),
                family=fig.layout.title.font.family or FONT_DISPLAY,
            )
        ),
        hoverlabel=dict(
            font=dict(
                size=_font_size_or_min(fig.layout.hoverlabel.font.size, PLOT_HOVER_FONT_SIZE),
                family=fig.layout.hoverlabel.font.family or FONT_FAMILY,
            )
        ),
    )
    fig.update_xaxes(
        tickfont=dict(size=PLOT_FONT_SIZE, family=FONT_FAMILY),
        title_font=dict(size=PLOT_FONT_SIZE, family=FONT_FAMILY),
        title_standoff=PLOT_AXIS_TITLE_STANDOFF,
        automargin=True,
    )
    fig.update_yaxes(
        tickfont=dict(size=PLOT_FONT_SIZE, family=FONT_FAMILY),
        title_font=dict(size=PLOT_FONT_SIZE, family=FONT_FAMILY),
        title_standoff=PLOT_AXIS_TITLE_STANDOFF,
        automargin=True,
    )

    if not preserve_legend:
        layout_json = fig.layout.to_plotly_json()
        legend_keys = ["legend"] + sorted(
            key for key in layout_json if key.startswith("legend") and key != "legend"
        )
        for legend_key in legend_keys:
            legend = getattr(fig.layout, legend_key, None)
            if legend is None:
                continue
            font_size = _font_size_or_min(getattr(legend.font, "size", None), PLOT_FONT_SIZE)
            title_font_size = _font_size_or_min(
                getattr(getattr(legend.title, "font", None), "size", None), PLOT_FONT_SIZE
            )
            fig.update_layout(
                **{
                    legend_key: dict(
                        font=dict(size=font_size, family=FONT_FAMILY),
                        title=dict(font=dict(size=title_font_size, family=FONT_FAMILY)),
                    )
                }
            )

    for trace in fig.data:
        textfont = getattr(trace, "textfont", None)
        if textfont is not None:
            if getattr(textfont, "size", None) is None:
                textfont.size = PLOT_FONT_SIZE
            if not getattr(textfont, "family", None):
                textfont.family = FONT_FAMILY

    for annotation in fig.layout.annotations:
        annotation.font.size = _font_size_or_min(
            getattr(annotation.font, "size", None), PLOT_FONT_SIZE
        )
        if not getattr(annotation.font, "family", None):
            annotation.font.family = FONT_FAMILY

    try:
        fig.update_coloraxes(
            colorbar=dict(
                tickfont=dict(size=PLOT_FONT_SIZE, family=FONT_FAMILY),
                title=dict(font=dict(size=PLOT_FONT_SIZE, family=FONT_FAMILY)),
            )
        )
    except ValueError:
        pass
    return fig


def _plotly_chart(
    fig: go.Figure,
    *args,
    preserve_legend: bool = False,
    **kwargs,
):
    """Render a Plotly figure through the dashboard's single chart wrapper."""
    _place_legend_above_plot(fig, preserve_legend=preserve_legend)
    _enforce_readable_plot_fonts(fig, preserve_legend=preserve_legend)
    return st.plotly_chart(fig, *args, **kwargs)


def _render_lap_detail_chart(
    fig: go.Figure,
    *,
    key: str,
    fullscreen_figures: dict[str, go.Figure] | None = None,
    fullscreen_selected: str | None = None,
    fullscreen_track_figure: go.Figure | None = None,
    fullscreen_gg_figures: dict[str, go.Figure] | None = None,
) -> None:
    """Render Lap Analysis detail with Video-Analysis-style crosshair labels."""
    fig_json = fig.to_json()
    fullscreen_payload = {
        str(label): json.loads(corner_fig.to_json())
        for label, corner_fig in (fullscreen_figures or {}).items()
    }
    if not fullscreen_payload:
        fullscreen_payload = {str(fullscreen_selected or "Current corner"): json.loads(fig_json)}
    fullscreen_track_payload = (
        json.loads(fullscreen_track_figure.to_json())
        if fullscreen_track_figure is not None
        else None
    )
    fullscreen_gg_payload = {
        str(label): json.loads(gg_fig.to_json())
        for label, gg_fig in (fullscreen_gg_figures or {}).items()
    }
    height_px = int(fig.layout.height or 900)
    component_id = f"lap_detail_{abs(hash(key))}"
    wrapper_id = f"{component_id}_wrap"
    component_id_js = json.dumps(component_id)
    wrapper_id_js = json.dumps(wrapper_id)
    fullscreen_selected_js = json.dumps(str(fullscreen_selected or next(iter(fullscreen_payload))))
    plotly_cdn_js = json.dumps(va._PLOTLY_CDN)
    escaped_fig_json = html.escape(fig_json, quote=False)
    escaped_fullscreen_json = html.escape(
        json.dumps(fullscreen_payload, ensure_ascii=False, allow_nan=False),
        quote=False,
    )
    escaped_fullscreen_track_json = html.escape(
        json.dumps(fullscreen_track_payload, ensure_ascii=False, allow_nan=False),
        quote=False,
    )
    escaped_fullscreen_gg_json = html.escape(
        json.dumps(fullscreen_gg_payload, ensure_ascii=False, allow_nan=False),
        quote=False,
    )
    component_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src={plotly_cdn_js}></script>
<style>
  html, body {{ margin:0; padding:0; background:#141417; overflow:hidden; }}
  #{wrapper_id} {{
    position:relative; width:100%; height:{height_px}px; background:#141417;
    overflow:hidden; border:1px solid rgba(255,255,255,0.08); border-radius:8px;
  }}
  .lap_detail_shell {{
    width:100%; height:100%; display:grid; grid-template-columns: minmax(0, 1fr);
    grid-template-rows: minmax(0, 1fr);
  }}
  .lap_detail_main {{ min-width:0; min-height:0; }}
  #{component_id} {{ width:100%; height:100%; }}
  .lap_detail_side {{
    display:none; min-width:0; min-height:0; grid-template-rows:minmax(0, 1fr) minmax(0, 1fr);
    gap:14px; padding:52px 12px 12px 0;
  }}
  .lap_detail_side_plot {{
    width:100%; height:100%; min-height:260px; background:#141417;
  }}
  #{wrapper_id}:fullscreen {{
    width:100vw; height:100vh; background:#141417; border-radius:0; border:none;
  }}
  #{wrapper_id}:fullscreen .lap_detail_shell {{
    grid-template-columns: minmax(0, 1.08fr) minmax(520px, 0.92fr);
    gap:16px; padding:12px;
  }}
  #{wrapper_id}:fullscreen .lap_detail_side {{ display:grid; }}
  .lap_detail_toolbar {{
    position:absolute; z-index:20; top:8px; left:10px;
    display:flex; align-items:center; gap:6px; padding:6px 8px;
    background:rgba(20,20,23,0.88); border:1px solid rgba(255,255,255,0.14);
    border-radius:6px; color:#EBEBEB; font:12px -apple-system, "Segoe UI", sans-serif;
  }}
  .lap_detail_toolbar button,
  .lap_detail_toolbar select {{
    background:#111318; color:#EBEBEB; border:1px solid rgba(255,255,255,0.18);
    border-radius:5px; padding:3px 6px; font-size:12px;
  }}
  .lap_detail_toolbar label {{ display:none; align-items:center; gap:6px; }}
  #{wrapper_id}:fullscreen .lap_detail_toolbar label {{ display:flex; }}
</style>
</head>
<body>
<script id="{component_id}_json" type="application/json">{escaped_fig_json}</script>
<script id="{component_id}_fullscreen_json" type="application/json">{escaped_fullscreen_json}</script>
<script id="{component_id}_fullscreen_track_json" type="application/json">{escaped_fullscreen_track_json}</script>
<script id="{component_id}_fullscreen_gg_json" type="application/json">{escaped_fullscreen_gg_json}</script>
<div id="{wrapper_id}">
  <div class="lap_detail_toolbar">
    <button id="{component_id}_fullscreen_btn" type="button">Full screen</button>
    <label>Inspect corner
      <select id="{component_id}_corner_select"></select>
    </label>
  </div>
  <div class="lap_detail_shell">
    <div class="lap_detail_main">
      <div id="{component_id}"></div>
    </div>
    <div class="lap_detail_side">
      <div id="{component_id}_track" class="lap_detail_side_plot"></div>
      <div id="{component_id}_gg" class="lap_detail_side_plot"></div>
    </div>
  </div>
</div>
<script>
(function() {{
  const CID = {component_id_js};
  const WID = {wrapper_id_js};
  const SELECTED_CORNER = {fullscreen_selected_js};
  const wrap = document.getElementById(WID);
  const gd = document.getElementById(CID);
  const trackGd = document.getElementById(CID + "_track");
  const ggGd = document.getElementById(CID + "_gg");
  const cornerSelect = document.getElementById(CID + "_corner_select");
  const fullscreenButton = document.getElementById(CID + "_fullscreen_btn");
  const fullscreenFigures = JSON.parse(document.getElementById(CID + "_fullscreen_json").textContent);
  const fullscreenTrackFigure = JSON.parse(document.getElementById(CID + "_fullscreen_track_json").textContent);
  const fullscreenGgFigures = JSON.parse(document.getElementById(CID + "_fullscreen_gg_json").textContent);
  let fig = JSON.parse(document.getElementById(CID + "_json").textContent);
  let baseShapes = [];
  let baseAnnotations = [];
  let selectedDistanceM = null;
  let currentTrackFig = null;
  let currentGgFig = null;
  let cursorRaf = 0;
  let pendingCursorPoint = null;
  let lastCursorXVal = null;

  const fullscreenIcon = {{
    width: 1000,
    height: 1000,
    path: "M60 360V60H360V160H160V360Z M640 60H940V360H840V160H640Z M160 640V840H360V940H60V640Z M840 640V840H640V940H940V640Z",
  }};
  function targetHeight() {{
    return document.fullscreenElement === wrap ? window.innerHeight - 24 : {height_px};
  }}
  function resizePlotForMode() {{
    Plotly.relayout(gd, {{ height: targetHeight(), autosize: true }});
    if (trackGd && trackGd.data) Plotly.Plots.resize(trackGd);
    if (ggGd && ggGd.data) Plotly.Plots.resize(ggGd);
  }}
  function toggleFullscreen() {{
    if (document.fullscreenElement === wrap) {{
      document.exitFullscreen();
      return;
    }}
    if (wrap.requestFullscreen) wrap.requestFullscreen();
  }}
  const config = {{
    displaylogo: false,
    responsive: true,
    scrollZoom: true,
    doubleClick: "reset+autosize",
    modeBarButtonsToRemove: [
      "lasso2d", "select2d", "toggleSpikelines",
      "hoverClosestCartesian", "hoverCompareCartesian"
    ],
    modeBarButtonsToAdd: [{{
      name: "fullscreen",
      title: "Full screen charts",
      icon: fullscreenIcon,
      click: toggleFullscreen,
    }}],
  }};

  function prepareFigure(nextFig) {{
    fig = JSON.parse(JSON.stringify(nextFig));
    fig.layout = fig.layout || {{}};
    fig.layout.height = targetHeight();
    fig.layout.autosize = true;
    fig.layout.margin = Object.assign({{}}, fig.layout.margin || {{}}, {{ autoexpand: false }});
    fig.layout.showlegend = false;
    fig.layout.hovermode = false;
    fig.layout.dragmode = "pan";
    fig.data = Array.isArray(fig.data) ? fig.data : [];
    fig.data.forEach(trace => {{
      trace.hoverinfo = "skip";
      trace.hovertemplate = null;
    }});
    baseShapes = Array.isArray(fig.layout.shapes) ? fig.layout.shapes.slice() : [];
    baseAnnotations = Array.isArray(fig.layout.annotations) ? fig.layout.annotations.slice() : [];
    pendingCursorPoint = null;
    lastCursorXVal = null;
    if (cursorRaf) {{
      cancelAnimationFrame(cursorRaf);
      cursorRaf = 0;
    }}
  }}

  function populateCornerSelect() {{
    const labels = Object.keys(fullscreenFigures);
    cornerSelect.innerHTML = "";
    labels.forEach(label => {{
      const option = document.createElement("option");
      option.value = label;
      option.textContent = label;
      cornerSelect.appendChild(option);
    }});
    cornerSelect.value = fullscreenFigures[SELECTED_CORNER] ? SELECTED_CORNER : labels[0];
    cornerSelect.disabled = labels.length <= 1;
  }}

  function sidePlotConfig() {{
    return {{
      displaylogo: false,
      responsive: true,
      scrollZoom: true,
      displayModeBar: false,
    }};
  }}

  function customColumn(raw, col) {{
    if (Array.isArray(raw)) {{
      return raw.map(value => Array.isArray(value) ? Number(value[col || 0]) : Number(value));
    }}
    if (!raw || typeof raw !== "object") return [];
    const flat = arrayFrom(raw);
    if (!flat.length) return [];
    const shape = Array.isArray(raw.shape)
      ? raw.shape
      : (typeof raw.shape === "string" ? raw.shape.split(",").map(value => Number(value.trim())) : []);
    const width = shape.length >= 2 ? Number(shape[1]) : 1;
    if (width <= 1) return flat.map(value => Number(value));
    const out = [];
    for (let i = col || 0; i < flat.length; i += width) out.push(Number(flat[i]));
    return out;
  }}

  function nearestPointInTrace(trace, distanceM) {{
    if (!trace || !isFinite(distanceM)) return null;
    if (trace.visible === false || trace.visible === "legendonly") return null;
    const xArr = arrayFrom(trace.x);
    const yArr = arrayFrom(trace.y);
    const sArr = customColumn(trace.customdata, 0);
    const n = Math.min(xArr.length, yArr.length, sArr.length);
    let best = null;
    let bestDiff = Infinity;
    for (let i = 0; i < n; i++) {{
      const x = Number(xArr[i]);
      const y = Number(yArr[i]);
      const s = Number(sArr[i]);
      if (!isFinite(x) || !isFinite(y) || !isFinite(s)) continue;
      const diff = Math.abs(s - distanceM);
      if (diff < bestDiff) {{
        bestDiff = diff;
        best = {{ x, y, s }};
      }}
    }}
    return best;
  }}

  function nearestByDistanceTrace(data, distanceM) {{
    if (!Array.isArray(data) || !isFinite(distanceM)) return null;
    let best = null;
    let bestDiff = Infinity;
    data.forEach(trace => {{
      const point = nearestPointInTrace(trace, distanceM);
      if (!point) return;
      const diff = Math.abs(point.s - distanceM);
      if (diff < bestDiff) {{
        bestDiff = diff;
        best = point;
      }}
    }});
    return best;
  }}

  function nearestByVisibleTrace(data, distanceM) {{
    if (!Array.isArray(data) || !isFinite(distanceM)) return [];
    const points = [];
    data.forEach((trace, idx) => {{
      const point = nearestPointInTrace(trace, distanceM);
      if (!point) return;
      points.push({{
        x: point.x,
        y: point.y,
        s: point.s,
        traceIndex: idx,
        traceName: String(trace.name || ""),
      }});
    }});
    return points;
  }}

  function markerTrace(point, name, color) {{
    if (!point) return null;
    return {{
      x: [point.x],
      y: [point.y],
      mode: "markers",
      name: name,
      showlegend: false,
      hovertemplate: "Selected point<br>Distance %{{customdata[0]:.1f}} m<extra></extra>",
      customdata: [[point.s]],
      marker: {{
        color: color || "#FFFFFF",
        size: 13,
        symbol: "circle",
        line: {{ color: "#111318", width: 2 }},
      }},
    }};
  }}

  function figureWithSelection(baseFig, markerName) {{
    if (!baseFig) return {{ data: [], layout: {{}} }};
    const nextData = Array.isArray(baseFig.data) ? baseFig.data.slice() : [];
    const marker = markerTrace(
      nearestByDistanceTrace(nextData, selectedDistanceM),
      markerName,
      "#FFFFFF",
    );
    if (marker) nextData.push(marker);
    return {{ data: nextData, layout: baseFig.layout || {{}} }};
  }}

  function ggFigureWithSelection(baseFig) {{
    if (!baseFig) return {{ data: [], layout: {{}} }};
    const nextData = Array.isArray(baseFig.data) ? baseFig.data.slice() : [];
    const markerColors = ["#FFFFFF", "#73D973"];
    nearestByVisibleTrace(nextData, selectedDistanceM).forEach((point, idx) => {{
      const markerName = point.traceName
        ? ("Selected GG point · " + point.traceName)
        : ("Selected GG point " + String(idx + 1));
      const marker = markerTrace(
        point,
        markerName,
        markerColors[idx] || "#FFFFFF",
      );
      if (marker) nextData.push(marker);
    }});
    return {{ data: nextData, layout: baseFig.layout || {{}} }};
  }}

  function renderSelectionMarkers() {{
    if (trackGd && currentTrackFig) {{
      const nextTrack = figureWithSelection(currentTrackFig, "Selected car position");
      Plotly.react(trackGd, nextTrack.data, nextTrack.layout, sidePlotConfig()).then(() => {{
        Plotly.Plots.resize(trackGd);
      }});
    }}
    if (ggGd && currentGgFig) {{
      const nextGg = ggFigureWithSelection(currentGgFig);
      Plotly.react(ggGd, nextGg.data, nextGg.layout, sidePlotConfig()).then(() => {{
        Plotly.Plots.resize(ggGd);
      }});
    }}
  }}

  function renderTrackFigure() {{
    if (!trackGd || !fullscreenTrackFigure) return;
    const trackFig = JSON.parse(JSON.stringify(fullscreenTrackFigure));
    trackFig.layout = trackFig.layout || {{}};
    trackFig.layout.autosize = true;
    currentTrackFig = trackFig;
    const nextTrack = figureWithSelection(currentTrackFig, "Selected car position");
    Plotly.react(trackGd, nextTrack.data, nextTrack.layout, sidePlotConfig()).then(() => {{
      Plotly.Plots.resize(trackGd);
    }});
  }}

  function renderGgFigure(label) {{
    if (!ggGd) return;
    const ggFig = fullscreenGgFigures[label];
    if (!ggFig) {{
      Plotly.purge(ggGd);
      ggGd.innerHTML = "";
      return;
    }}
    const nextFig = JSON.parse(JSON.stringify(ggFig));
    nextFig.layout = nextFig.layout || {{}};
    nextFig.layout.autosize = true;
    currentGgFig = nextFig;
    const nextGg = ggFigureWithSelection(currentGgFig);
    Plotly.react(ggGd, nextGg.data, nextGg.layout, sidePlotConfig()).then(() => {{
      Plotly.Plots.resize(ggGd);
    }});
  }}

  function changeCorner(label) {{
    const nextFig = fullscreenFigures[label];
    if (!nextFig) return;
    prepareFigure(nextFig);
    Plotly.react(gd, fig.data, fig.layout, config).then(resizePlotForMode);
    renderGgFigure(label);
  }}

  function arrayFrom(raw) {{
    if (Array.isArray(raw)) return raw;
    if (!raw || typeof raw !== "object" || typeof raw.bdata !== "string") return [];
    const dtype = String(raw.dtype || "");
    let bytes;
    try {{
      const binary = atob(raw.bdata);
      bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    }} catch (e) {{
      return [];
    }}
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    let typed;
    try {{
      if (dtype === "f8") typed = new Float64Array(buffer);
      else if (dtype === "f4") typed = new Float32Array(buffer);
      else if (dtype === "i4") typed = new Int32Array(buffer);
      else if (dtype === "u4") typed = new Uint32Array(buffer);
      else if (dtype === "i2") typed = new Int16Array(buffer);
      else if (dtype === "u2") typed = new Uint16Array(buffer);
      else if (dtype === "i1") typed = new Int8Array(buffer);
      else if (dtype === "u1") typed = new Uint8Array(buffer);
      else if (dtype === "i8") typed = new BigInt64Array(buffer);
      else if (dtype === "u8") typed = new BigUint64Array(buffer);
      else return [];
    }} catch (e) {{
      return [];
    }}
    return Array.from(typed, value => Number(value));
  }}

  function axisLayoutName(axisRef, prefix) {{
    const ref = axisRef || prefix;
    return ref === prefix ? prefix + "axis" : prefix + "axis" + ref.slice(1);
  }}

  function screenYFromData(yref, yVal) {{
    const fullLayout = gd._fullLayout;
    if (!fullLayout) return null;
    const axis = fullLayout[axisLayoutName(yref, "y")];
    if (!axis || typeof axis.l2p !== "function") return null;
    const pixel = Number(axis.l2p(yVal));
    return isFinite(pixel) ? pixel : null;
  }}

  function annotationOffsets(entries, nearRightEdge) {{
    const yShifts = Array(entries.length).fill(0);
    const xShifts = Array(entries.length).fill(nearRightEdge ? -6 : 6);
    const groups = new Map();
    entries.forEach((entry, idx) => {{
      const screenY = screenYFromData(entry.yref, entry.y);
      const key = entry.yref || "y";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push({{ idx, screenY }});
    }});
    const minGapPx = 24;
    const xStepPx = 42;

    function applyCluster(cluster) {{
      if (!cluster.length) return;
      if (cluster.length === 1) {{
        yShifts[cluster[0].idx] = 0;
        return;
      }}
      const center = cluster.reduce((sum, item) => sum + item.screenY, 0) / cluster.length;
      const start = center - (minGapPx * (cluster.length - 1)) / 2;
      cluster.forEach((item, order) => {{
        yShifts[item.idx] = start + order * minGapPx - item.screenY;
        xShifts[item.idx] = (nearRightEdge ? -6 : 6) + (nearRightEdge ? -1 : 1) * order * xStepPx;
      }});
    }}

    groups.forEach(group => {{
      const valid = group
        .filter(item => item.screenY !== null && isFinite(item.screenY))
        .sort((a, b) => a.screenY - b.screenY);
      let cluster = [];
      valid.forEach(item => {{
        if (!cluster.length) {{
          cluster = [item];
          return;
        }}
        if (item.screenY - cluster[cluster.length - 1].screenY < minGapPx) {{
          cluster.push(item);
          return;
        }}
        applyCluster(cluster);
        cluster = [item];
      }});
      applyCluster(cluster);
    }});
    return {{ yShifts, xShifts }};
  }}

  function plotAreaXFromEvent(event) {{
    const fl = gd._fullLayout;
    if (!fl || !fl.xaxis || !Array.isArray(fl.xaxis.range)) return null;
    const rect = gd.getBoundingClientRect();
    const margin = fl.margin || {{}};
    const width = Number.isFinite(fl.width) ? fl.width : rect.width;
    const height = Number.isFinite(fl.height) ? fl.height : rect.height;
    const leftMargin = Number.isFinite(margin.l) ? margin.l : 0;
    const rightMargin = Number.isFinite(margin.r) ? margin.r : 0;
    const topMargin = Number.isFinite(margin.t) ? margin.t : 0;
    const bottomMargin = Number.isFinite(margin.b) ? margin.b : 0;
    const plotWidth = width - leftMargin - rightMargin;
    const plotHeight = height - topMargin - bottomMargin;
    const domain = Array.isArray(fl.xaxis.domain) ? fl.xaxis.domain : [0, 1];
    const x0Px = leftMargin + domain[0] * plotWidth;
    const x1Px = leftMargin + domain[1] * plotWidth;
    const y0Px = topMargin;
    const y1Px = topMargin + plotHeight;
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    if (px < x0Px || px > x1Px || py < y0Px || py > y1Px) return null;
    const frac = Math.max(0, Math.min(1, (px - x0Px) / Math.max(1, x1Px - x0Px)));
    const r0 = Number(fl.xaxis.range[0]);
    const r1 = Number(fl.xaxis.range[1]);
    if (!isFinite(r0) || !isFinite(r1)) return null;
    return r0 + frac * (r1 - r0);
  }}

  function valueAtX(xArr, yArr, xVal) {{
    xArr = arrayFrom(xArr);
    yArr = arrayFrom(yArr);
    if (!Array.isArray(xArr) || !Array.isArray(yArr)) return NaN;
    const n = Math.min(xArr.length, yArr.length);
    let first = -1;
    let last = -1;
    for (let i = 0; i < n; i++) {{
      const x = Number(xArr[i]);
      const y = Number(yArr[i]);
      if (isFinite(x) && isFinite(y)) {{
        if (first < 0) first = i;
        last = i;
      }}
    }}
    if (first < 0) return NaN;
    const xFirst = Number(xArr[first]);
    const xLast = Number(xArr[last]);
    if (xVal <= xFirst) return Number(yArr[first]);
    if (xVal >= xLast) return Number(yArr[last]);
    for (let i = first; i < last; i++) {{
      const x0 = Number(xArr[i]);
      const x1 = Number(xArr[i + 1]);
      const y0 = Number(yArr[i]);
      const y1 = Number(yArr[i + 1]);
      if (!isFinite(x0) || !isFinite(x1) || !isFinite(y0) || !isFinite(y1)) continue;
      const inside = (x0 <= xVal && xVal <= x1) || (x1 <= xVal && xVal <= x0);
      if (!inside) continue;
      if (x1 === x0) return y0;
      const f = (xVal - x0) / (x1 - x0);
      return y0 + f * (y1 - y0);
    }}
    return NaN;
  }}

  function distanceAtGraphX(xVal) {{
    if (!isFinite(xVal)) return NaN;
    for (const trace of fig.data) {{
      if (trace.visible === false || trace.visible === "legendonly") continue;
      const xArr = arrayFrom(trace.x);
      const sArr = customColumn(trace.customdata, 0);
      const n = Math.min(xArr.length, sArr.length);
      if (n < 2) continue;
      let first = -1;
      let last = -1;
      for (let i = 0; i < n; i++) {{
        if (isFinite(Number(xArr[i])) && isFinite(Number(sArr[i]))) {{
          if (first < 0) first = i;
          last = i;
        }}
      }}
      if (first < 0) continue;
      const xFirst = Number(xArr[first]);
      const xLast = Number(xArr[last]);
      if (xVal <= xFirst) return Number(sArr[first]);
      if (xVal >= xLast) return Number(sArr[last]);
      for (let i = first; i < last; i++) {{
        const x0 = Number(xArr[i]);
        const x1 = Number(xArr[i + 1]);
        const s0 = Number(sArr[i]);
        const s1 = Number(sArr[i + 1]);
        if (!isFinite(x0) || !isFinite(x1) || !isFinite(s0) || !isFinite(s1)) continue;
        const inside = (x0 <= xVal && xVal <= x1) || (x1 <= xVal && xVal <= x0);
        if (!inside) continue;
        if (x1 === x0) return s0;
        const f = (xVal - x0) / (x1 - x0);
        return s0 + f * (s1 - s0);
      }}
    }}
    return xVal;
  }}

  function colorForTrace(trace) {{
    if (trace && trace.line && trace.line.color) return trace.line.color;
    if (trace && trace.marker && trace.marker.color) return trace.marker.color;
    return "#FFFFFF";
  }}

  function formatValue(value) {{
    if (!isFinite(value)) return "";
    const absValue = Math.abs(value);
    if (absValue >= 100) return value.toFixed(1);
    if (absValue >= 10) return value.toFixed(2);
    return value.toFixed(3);
  }}

  function subplotRefs() {{
    const refs = [];
    fig.data.forEach(trace => {{
      const yref = trace.yaxis || "y";
      const xref = trace.xaxis || "x";
      if (!refs.some(item => item.yref === yref)) refs.push({{ xref, yref }});
    }});
    refs.sort((a, b) => {{
      const ay = gd._fullLayout[axisLayoutName(a.yref, "y")];
      const by = gd._fullLayout[axisLayoutName(b.yref, "y")];
      const ad = ay && Array.isArray(ay.domain) ? ay.domain[1] : 0;
      const bd = by && Array.isArray(by.domain) ? by.domain[1] : 0;
      return bd - ad;
    }});
    return refs;
  }}

  function renderCursor(clientX, clientY) {{
    const xVal = plotAreaXFromEvent({{ clientX, clientY }});
    if (xVal === null || !isFinite(xVal)) return;
    if (lastCursorXVal !== null && Math.abs(lastCursorXVal - xVal) < 1e-6) return;
    lastCursorXVal = xVal;
    const shapes = baseShapes.slice();
    subplotRefs().forEach(ref => {{
      shapes.push({{
        type: "line",
        xref: ref.xref,
        yref: ref.yref + " domain",
        x0: xVal,
        x1: xVal,
        y0: 0,
        y1: 1,
        line: {{ color: "rgba(255,255,255,0.85)", width: 1, dash: "dot" }},
      }});
    }});

    const annotations = baseAnnotations.slice();
    const fullX = gd._fullLayout && gd._fullLayout.xaxis;
    const xrange = fullX && Array.isArray(fullX.range) ? fullX.range : null;
    const xFrac = xrange && isFinite(Number(xrange[0])) && isFinite(Number(xrange[1])) && Number(xrange[1]) !== Number(xrange[0])
      ? (xVal - Number(xrange[0])) / (Number(xrange[1]) - Number(xrange[0]))
      : 0.5;
    const nearRightEdge = xFrac > 0.88;
    const entries = [];
    fig.data.forEach(trace => {{
      if (trace.visible === false || trace.visible === "legendonly") return;
      const y = valueAtX(trace.x, trace.y, xVal);
      if (!isFinite(y)) return;
      const color = colorForTrace(trace);
      const xref = trace.xaxis || "x";
      const yref = trace.yaxis || "y";
      entries.push({{
        x: xVal,
        y: y,
        xref: xref,
        yref: yref,
        text: formatValue(y),
        color: color,
      }});
    }});
    const offsets = annotationOffsets(entries, nearRightEdge);
    entries.forEach((entry, idx) => {{
      annotations.push({{
        x: entry.x,
        y: entry.y,
        xref: entry.xref,
        yref: entry.yref,
        text: entry.text,
        showarrow: false,
        xanchor: nearRightEdge ? "right" : "left",
        yanchor: "middle",
        xshift: offsets.xShifts[idx],
        yshift: offsets.yShifts[idx],
        bgcolor: "rgba(20,20,23,0.96)",
        bordercolor: entry.color,
        borderwidth: 1,
        borderpad: 2,
        font: {{ color: entry.color, size: 11 }},
      }});
    }});
    Plotly.relayout(gd, {{ shapes: shapes, annotations: annotations }});
  }}

  function updateCursor(event) {{
    pendingCursorPoint = {{
      clientX: event.clientX,
      clientY: event.clientY,
    }};
    if (cursorRaf) return;
    cursorRaf = requestAnimationFrame(() => {{
      cursorRaf = 0;
      const point = pendingCursorPoint;
      pendingCursorPoint = null;
      if (!point) return;
      renderCursor(point.clientX, point.clientY);
    }});
  }}

  function selectCursor(event) {{
    if (event.button !== 0) return;
    const xVal = plotAreaXFromEvent(event);
    if (xVal === null || !isFinite(xVal)) return;
    const distanceM = distanceAtGraphX(xVal);
    if (!isFinite(distanceM)) return;
    selectedDistanceM = distanceM;
    renderCursor(event.clientX, event.clientY);
    renderSelectionMarkers();
  }}

  function clearCursor() {{
    if (selectedDistanceM !== null) return;
    pendingCursorPoint = null;
    lastCursorXVal = null;
    if (cursorRaf) {{
      cancelAnimationFrame(cursorRaf);
      cursorRaf = 0;
    }}
    Plotly.relayout(gd, {{ shapes: baseShapes, annotations: baseAnnotations }});
  }}

  populateCornerSelect();
  prepareFigure(fig);
  Plotly.newPlot(gd, fig.data, fig.layout, config).then(() => {{
    gd.addEventListener("mousemove", updateCursor);
    gd.addEventListener("mousedown", selectCursor);
    gd.addEventListener("mouseleave", clearCursor);
    document.addEventListener("fullscreenchange", resizePlotForMode);
    window.addEventListener("resize", resizePlotForMode);
    cornerSelect.addEventListener("change", event => changeCorner(event.target.value));
    fullscreenButton.addEventListener("click", toggleFullscreen);
    renderTrackFigure();
    renderGgFigure(cornerSelect.value);
  }});
}})();
</script>
</body>
</html>"""
    components.html(component_html, height=height_px + 8, scrolling=False)


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

        for run_idx, run_name in enumerate(run_names):
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
                        run_idx,
                        variant_map[variant_key],
                    )
                )

            for axis_name in fig.layout:
                if not str(axis_name).startswith("xaxis"):
                    continue
                tickvals = getattr(fig.layout[axis_name], "tickvals", None)
                if tickvals is None:
                    continue
                tickvals_by_axis.setdefault(str(axis_name), set()).update(
                    float(v) for v in tickvals
                )

        for axis_name, tickvals in tickvals_by_axis.items():
            if tickvals:
                base.layout[axis_name].tickvals = sorted(tickvals)
        if any(getattr(t, "type", "") == "histogram" for t in base.data):
            base.update_layout(barmode="overlay")
            for trace in base.data:
                if getattr(trace, "type", "") == "histogram":
                    trace.opacity = min(getattr(trace, "opacity", 1.0) or 1.0, 0.65)
        merged.append(base)

    return merged


def _overlay_setup_single_figure(
    results_by_run: dict[str, tuple[go.Figure, dict]],
) -> go.Figure | None:
    """Merge one Setup figure per run into a single comparison plot."""
    if not results_by_run:
        return None
    merged = _overlay_figures(
        {run_name: [fig] for run_name, (fig, _kpis) in results_by_run.items()}
    )
    if not merged:
        return None
    fig = merged[0]
    fig.update_xaxes(autorange=True)
    fig.update_yaxes(autorange=True)
    return fig


def _concat_run_tables(run_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Concatenate per-run tables, adding a `Run` column when needed."""
    tables: list[pl.DataFrame] = []
    for run_name, table in run_tables.items():
        if "Run" not in table.columns:
            table = table.with_columns(pl.lit(run_name).alias("Run"))
        tables.append(table)
    return pl.concat(tables, how="vertical_relaxed") if tables else pl.DataFrame()


def _kpi_column_config(columns) -> dict:
    """Tooltip config for KPI table columns, sourced from METRIC_HELP."""
    return {
        col: st.column_config.Column(help=METRIC_HELP[col]) for col in columns if col in METRIC_HELP
    }


def _render_summary_df(df: pl.DataFrame) -> None:
    """Render a KPI summary dataframe with per-column tooltips."""
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=_kpi_column_config(df.columns),
    )


def _show_summary_table(rows: list[dict[str, object]]) -> None:
    """Render a compact per-run summary table when multiple CSVs are loaded."""
    if rows:
        _render_summary_df(pl.DataFrame(rows))


def _full_dataframe_height(row_count: int) -> int:
    """Height that lets Streamlit show every dataframe row without inner scroll."""
    header_px = 38
    row_px = 35
    border_px = 3
    return header_px + row_px * max(1, int(row_count)) + border_px


def _run_cache_tokens(dfs: dict[str, pl.DataFrame]) -> tuple[tuple[str, FileSignature, str], ...]:
    """Stable cache tokens for current runs and selected laps."""
    file_signatures = st.session_state.get("_run_file_signatures", {})
    return tuple(
        (
            run_name,
            file_signatures.get(run_name, (0, 0)),
            _lap_signature(df),
        )
        for run_name, df in dfs.items()
    )


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
def _pt_energy_per_lap_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return pt.energy_per_lap_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _pt_power_per_wheel_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return pt.power_per_wheel_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _pt_soc_per_lap_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return pt.soc_per_lap_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _pt_thermal_evolution_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return pt.thermal_evolution_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _pt_inverter_limits_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return pt.inverter_limits_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _pt_torque_fidelity_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return pt.torque_fidelity_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _pt_torque_speed_envelope_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return pt.torque_speed_envelope_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _pt_hv_delivery_efficiency_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return pt.hv_delivery_efficiency_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _pt_weakest_cell_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return pt.weakest_cell_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _pt_thermal_headroom_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return pt.thermal_headroom_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_decel_envelope_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.decel_envelope_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_brake_blending_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.brake_blending_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_ideal_brake_distribution_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.ideal_brake_distribution_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_axle_brake_utilisation_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.axle_brake_utilisation_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_axle_brake_slip_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.axle_brake_slip_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_wheel_lockup_per_lap_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.wheel_lockup_per_lap_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_wheel_lockup_track_map_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    run_name: str,
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.wheel_lockup_track_map_fig(dfs[run_name])


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_ideal_braking_curve_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.ideal_braking_curve_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_max_braking_g_per_lap_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return drv.max_braking_g_per_lap_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _skidpad_fig_cached(
    metric: str,
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    """Cached wrapper for multi-run skidpad event KPI plots."""
    _ = run_tokens
    funcs = {
        "event_time": skidpad.event_time_summary_fig,
        "lateral_g": skidpad.lateral_g_fig,
        "driven_radius": skidpad.driven_radius_fig,
        "balance": skidpad.balance_fig,
        "understeer_gradient": skidpad.understeer_gradient_fig,
        "tv_intervention": skidpad.tv_intervention_fig,
        "lateral_load_dist": skidpad.lateral_load_dist_fig,
        "lr_asymmetry": skidpad.lr_asymmetry_fig,
        "gps_figure8": skidpad.gps_figure8_fig,
        # Compatibility with the first Events implementation.
        "lateral_g_hist": skidpad.lateral_g_fig,
        "radius_hist": skidpad.driven_radius_fig,
        "understeer": skidpad.understeer_gradient_fig,
        "slip_vs_ay": skidpad.balance_fig,
        "yaw_vs_ay": skidpad.yaw_rate_vs_ay_fig,
        "lltd_scatter": skidpad.lateral_load_dist_fig,
    }
    if metric not in funcs:
        raise KeyError(f"Unknown skidpad metric: {metric}")
    return funcs[metric](dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _accel_fig_cached(
    metric: str,
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    """Cached wrapper for acceleration event figures."""
    _ = run_tokens
    funcs = {
        "power_vx": accel.power_dc_vs_vx_scatter_fig,
        "sr_fr_balance": accel.sr_front_rear_scatter_fig,
        "ax_vx": accel.ax_vs_vx_envelope_fig,
        "motor_tq_rpm": accel.motor_torque_vs_wheel_speed_fig,
        "sr_fx_wheel": accel.sr_vs_fx_per_wheel_fig,
    }
    if metric not in funcs:
        raise KeyError(f"Unknown acceleration metric: {metric}")
    return funcs[metric](dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_ideal_traction_curve_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.ideal_traction_curve_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_accel_envelope_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.accel_envelope_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_traction_slip_curve_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.traction_slip_curve_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_axle_traction_utilisation_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.axle_traction_utilisation_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_cornering_balance_phase_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.cornering_balance_phase_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_lateral_grip_envelope_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.lateral_grip_envelope_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _dyn_axle_lateral_utilisation_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens
    return dyn.axle_lateral_utilisation_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _gf_gg_scatter_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    _ = run_tokens, "gg-horizontal-legend-v2"
    return gf.gg_scatter_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _gf_utilization_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> go.Figure:
    _ = run_tokens
    return gf.grip_utilization_fig(dfs)


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


def _brake_application_turns_from_signature(turns_signature: tuple) -> list:
    """Rebuild TurnDef list from a cache signature."""
    return [
        corn.TurnDef(
            turn_id=int(t[0]),
            s_entry_m=float(t[1]),
            s_apex_m=float(t[2]),
            s_exit_m=float(t[3]),
            apex_lat=float(t[4]),
            apex_lng=float(t[5]),
            lat=np.array([], dtype=float),
            lng=np.array([], dtype=float),
        )
        for t in turns_signature
    ]


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
def _driver_combined_brake_steer_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> go.Figure:
    """Cached wrapper for combined braking & steering per-lap figure."""
    _ = run_tokens
    return drv.combined_brake_steer_fig(dfs, x_mode=x_mode)


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
def _driver_steering_integral_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> go.Figure:
    """Cached wrapper for steering-integral figure."""
    _ = run_tokens
    return drv.steering_integral_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_steering_stability_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    turns_signature: tuple,
) -> go.Figure:
    """Cached wrapper for steering-stability box-plot figure."""
    _ = run_tokens
    turns = [
        corn.TurnDef(
            turn_id=int(t[0]),
            s_entry_m=float(t[1]),
            s_apex_m=float(t[2]),
            s_exit_m=float(t[3]),
            apex_lat=float(t[4]),
            apex_lng=float(t[5]),
            lat=np.array([], dtype=float),
            lng=np.array([], dtype=float),
        )
        for t in turns_signature
    ]
    return drv.steering_stability_fig(dfs, turns=turns)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_corner_curvature_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
    x_mode: str,
) -> go.Figure:
    """Cached wrapper for corner-curvature per-lap figure."""
    _ = run_tokens
    return drv.corner_curvature_fig(dfs, x_mode=x_mode)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_lap_time_progression_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> go.Figure:
    """Cached wrapper for the lap-time progression figure."""
    _ = run_tokens
    return drv.lap_time_progression_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_lap_time_distribution_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> go.Figure:
    """Cached wrapper for the lap-time distribution box plot."""
    _ = run_tokens
    return drv.lap_time_distribution_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_run_phase_distribution_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    """Cached wrapper for the Overview pedal-phase distribution bar."""
    _ = run_tokens
    return drv.run_phase_distribution_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_fastest_lap_speed_map_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    """Cached wrapper for the Overview fastest-lap speed maps."""
    _ = run_tokens
    return drv.fastest_lap_speed_map_fig(dfs)


@st.cache_resource(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_speed_distribution_fig_cached(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[go.Figure, dict]:
    """Cached wrapper for the Overview track speed-distribution figure."""
    _ = run_tokens
    return drv.speed_distribution_fig(dfs)


def _is_skidpad_lap(df: pl.DataFrame, lap_id: int) -> bool:
    """True if the given lap was logged in skidpad event mode."""
    if "lapcount_mode" not in df.columns or "laps" not in df.columns:
        return False
    sub = df.filter(pl.col("laps") == int(lap_id))
    if sub.is_empty():
        return False
    values = sub["lapcount_mode"].drop_nulls()
    if len(values) == 0:
        return False
    return str(values[0]).strip().lower() == "skidpad"


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_cornering_turns_cached(
    dfs,
    driver_run_tokens,
    R_thr_m,
    min_dur_s,
    merge_gap_m,
    ref_run,
    ref_lap,
) -> list:
    """Cached turn detection for the selected cornering reference lap."""
    _ = driver_run_tokens
    d = corn.compute_radius_curvature(dfs[ref_run])
    if _is_skidpad_lap(dfs[ref_run], int(ref_lap)):
        return corn.detect_skidpad_turn_on_lap(d, int(ref_lap))
    return corn.detect_turns_on_lap(
        d,
        ref_run,
        int(ref_lap),
        R_thr_m=float(R_thr_m),
        min_dur_s=float(min_dur_s),
        merge_gap_m=float(merge_gap_m),
    )


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_cornering_metrics_cached(
    dfs,
    driver_run_tokens,
    turns_signature,
) -> pl.DataFrame:
    """Cached cornering metrics projected onto all selected driver laps."""
    _ = driver_run_tokens
    turns = [
        corn.TurnDef(
            turn_id=int(t[0]),
            s_entry_m=float(t[1]),
            s_apex_m=float(t[2]),
            s_exit_m=float(t[3]),
            apex_lat=float(t[4]),
            apex_lng=float(t[5]),
            lat=np.array([], dtype=float),
            lng=np.array([], dtype=float),
        )
        for t in turns_signature
    ]
    return corn.compute_turn_metrics(dfs, turns)


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_fastest_lap_cached(
    dfs,
    driver_run_tokens,
    run_name: str,
) -> int | None:
    """Cached fastest valid lap for per-CSV sector summaries."""
    _ = driver_run_tokens
    return lsec.fastest_valid_lap(dfs[run_name])


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_lap_sectors_cached(
    dfs,
    driver_run_tokens,
    run_name: str,
    turns_signature: tuple,
    lap_end_m_token: float,
) -> list[lsec.Sector]:
    """Cached sectorization for one run's fastest valid lap."""
    _ = (dfs, driver_run_tokens, run_name, lap_end_m_token)
    turns = [
        corn.TurnDef(
            turn_id=int(t[0]),
            s_entry_m=float(t[1]),
            s_apex_m=float(t[2]),
            s_exit_m=float(t[3]),
            apex_lat=float(t[4]),
            apex_lng=float(t[5]),
            lat=np.array([], dtype=float),
            lng=np.array([], dtype=float),
        )
        for t in turns_signature
    ]
    return lsec.build_sectors(turns, float(lap_end_m_token))


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_csv_sector_summary_cached(
    dfs,
    driver_run_tokens,
    run_name: str,
    sectors_token: tuple,
) -> dict[str, dict[str, float] | None]:
    """Cached best/avg/potential metrics for one CSV."""
    _ = driver_run_tokens
    sectors = [
        lsec.Sector(
            index=int(token[0]),
            kind=str(token[1]),
            s_start_m=float(token[2]),
            s_end_m=float(token[3]),
            turn_id=None if int(token[4]) < 0 else int(token[4]),
        )
        for token in sectors_token
    ]
    return lsec.csv_metrics_summary(dfs[run_name], sectors)


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_sector_times_matrix_cached(
    dfs,
    driver_run_tokens,
    sectors_token: tuple,
) -> pl.DataFrame:
    """Cached per-lap sector-time matrix across all loaded runs."""
    _ = driver_run_tokens
    sectors = [
        lsec.Sector(
            index=int(token[0]),
            kind=str(token[1]),
            s_start_m=float(token[2]),
            s_end_m=float(token[3]),
            turn_id=None if int(token[4]) < 0 else int(token[4]),
        )
        for token in sectors_token
    ]
    return lsec.sector_times_matrix(dfs, sectors)


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_whole_lap_metrics_cached(
    dfs,
    driver_run_tokens,
    run_name: str,
    lap_id: int,
) -> dict[str, float] | None:
    """Cached whole-lap phase metrics for the selected reference/comparison lap."""
    _ = driver_run_tokens
    return lsec.whole_lap_metrics(dfs[run_name], int(lap_id))


@st.cache_data(show_spinner=False, hash_funcs=_PL_HASH_FUNCS)
def _driver_potential_lap_cached(
    dfs,
    driver_run_tokens,
    sectors_token: tuple,
) -> tuple[pl.DataFrame, dict[str, object]] | None:
    """Cached stitched potential lap built from the loaded Driver CSVs."""
    _ = driver_run_tokens
    sectors = [
        lsec.Sector(
            index=int(token[0]),
            kind=str(token[1]),
            s_start_m=float(token[2]),
            s_end_m=float(token[3]),
            turn_id=None if int(token[4]) < 0 else int(token[4]),
        )
        for token in sectors_token
    ]
    return drv.potential_lap_from_sectors(dfs, sectors)


# ── Function-level checks (per-controller "is it doing its job?") ─────────────


# Per-chart captions for the Controls figure stacks (TC/RB return figure lists
# under a single section heading, so each chart needs its own one-line gloss).
# Keyed by a substring of the Plotly title so the caption survives reordering.
_CONTROLS_FIG_CAPTIONS: dict[str, str] = {
    # TC — Slip Ratio Tracking
    "SR distribution while TC armed": "How often the wheels sit in the target slip-ratio band while TC is working — taller near +0.20 is better.",
    "Slip ratio vs longitudinal acceleration": "Slip ratio against the grip it buys — the cloud should sit near +0.20 where acceleration peaks.",
    "Maximum slip ratio by wheel": "The worst slip spike each wheel reaches when TC steps in — lower is more controlled.",
    "TC slip error balance by wheel": "Whether TC errs above or below target, wheel by wheel — centred on zero is balanced.",
    "Overslip response": "Once a wheel oversteps the target, how quickly TC brings it back down.",
    "SR vs delivered wheel torque": "Slip ratio against the torque actually delivered — shows TC trading torque for grip.",
    # RB — main braking/regen stack
    "Slip ratio vs longitudinal deceleration": "Braking slip against the decel it produces — the cloud should sit near the −0.20 target.",
    "Minimum slip ratio by wheel": "The deepest (most locked) slip each wheel reaches under braking — closer to 0 is safer.",
    "Wheel locking time per lap": "Seconds per lap each wheel spends near lock-up — lower is better.",
    "Braking torque vs vertical load by wheel": "Regen braking torque against the load on each wheel — should follow load, with the front carrying more.",
    # RB — function check
    "SR distribution while braking": "How often braking slip sits in the target band — taller near −0.20 is better.",
    "Energy recovered vs braking effort per lap": "Energy regenerated against how hard the lap was braked — higher and steeper means more recovery.",
}


def _controls_fig_caption(fig: go.Figure) -> str | None:
    """Plain-language caption for a Controls stack figure, matched by title."""
    title = (fig.layout.title.text or "") if fig.layout.title else ""
    for key, caption in _CONTROLS_FIG_CAPTIONS.items():
        if key in title:
            return caption
    return None


def _plot_controls_stack(figs) -> None:
    """Render a Controls figure stack, each chart preceded by its caption."""
    for fig in figs:
        caption = _controls_fig_caption(fig)
        if caption:
            st.caption(caption)
        _plotly_chart(fig, use_container_width=True, theme=None)


def _render_rb_function_check(dfs: dict[str, pl.DataFrame]) -> None:
    st.divider()
    st.subheader("RB — Slip Ratio & Energy Recovery")
    st.caption("Is RB holding SR ≈ −0.20 under braking and recovering meaningful energy?")
    all_figs: dict[str, list[go.Figure]] = {}
    rows = []
    for run_name, df in dfs.items():
        try:
            figs, kpis = rb.rb_function_kpis(df)
        except Exception as exc:
            st.warning(f"{run_name}: RB function check unavailable: {exc}")
            continue
        all_figs[run_name] = figs
        rows.append(
            {
                "Run": Path(run_name).stem,
                "In target [%]": round(kpis["pct_in_target"], 1),
                "Lock-up risk [%]": round(kpis["pct_lockup_risk"], 1),
                "Recovered total [Wh]": round(kpis["energy_recovered_wh_total"], 1),
                "Recovered / lap [Wh]": round(kpis["energy_recovered_wh_median_lap"], 1),
                "Regen coverage [%]": round(kpis["regen_coverage_pct"], 1),
            }
        )
    if rows:
        _show_summary_table(rows)
    _plot_controls_stack(_overlay_figures(all_figs))


def _render_pc_function_check(dfs: dict[str, pl.DataFrame]) -> None:
    st.divider()
    st.subheader("PC — Battery Power Cap")
    st.caption("Is P_bat staying under the 80 kW cap and exploiting it near full throttle?")
    all_figs: dict[str, list[go.Figure]] = {}
    rows = []
    for run_name, df in dfs.items():
        try:
            figs, kpis = pt.pc_function_kpis(df)
        except Exception as exc:
            st.warning(f"{run_name}: PC function check unavailable: {exc}")
            continue
        all_figs[run_name] = [
            fig for fig in figs if "Battery power vs time" not in (fig.layout.title.text or "")
        ]
        rows.append(
            {
                "Run": Path(run_name).stem,
                "Over cap [%]": round(kpis["pct_over_cap"], 2),
                "Overshoot events": int(kpis["n_overshoot_events"]),
                "Peak Pbat [kW]": round(kpis["peak_kw"], 1),
                "Near cap @ full [%]": round(kpis["pct_near_cap_at_full"], 1),
            }
        )
    if rows:
        _show_summary_table(rows)
    for fig in _overlay_figures(all_figs):
        _plotly_chart(fig, use_container_width=True, theme=None)


# ── Tab renderers ─────────────────────────────────────────────────────────────


def _tab_powertrain(dfs: dict[str, pl.DataFrame]) -> None:
    pt_section = st.segmented_control(
        "Powertrain section",
        options=["Motors & Inverters", "Battery", "Temperatures"],
        default="Motors & Inverters",
        required=True,
        key="pt_subsection",
        label_visibility="collapsed",
        width="stretch",
    )
    # One per-lap x-axis control shared by every per-lap chart in the tab.
    pt_x_mode = _section_per_lap_axis("powertrain", default="laps")
    run_tokens = _run_cache_tokens(dfs)

    if pt_section == "Battery":
        _render_pt_battery(dfs, run_tokens, pt_x_mode)
    elif pt_section == "Temperatures":
        _render_pt_temperatures(dfs, run_tokens, pt_x_mode)
    else:
        _render_pt_motors(dfs, run_tokens, pt_x_mode)


def _render_pt_motors(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple,
    pt_x_mode: str,
) -> None:
    # ── Power per wheel ───────────────────────────────────────────────────────
    st.subheader("Power Distribution per Wheel")
    st.caption("Mean motor power per wheel, with Front/Rear and Left/Right splits.")
    try:
        if len(dfs) == 1:
            fig, kpis = _pt_power_per_wheel_fig_cached(dfs, run_tokens, pt_x_mode)
            for w in kpis.get("warnings", []):
                st.warning(w)
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean total power", f"{kpis['mean_total_kw']:.1f} kW")
            c2.metric("Front / Rear", f"{kpis['fr_pct']:.1f}% / {100 - kpis['fr_pct']:.1f}%")
            c3.metric("Left / Right", f"{kpis['lr_pct']:.1f}% / {100 - kpis['lr_pct']:.1f}%")
            c4, c5, c6, c7 = st.columns(4)
            for w, col in zip(("FL", "FR", "RL", "RR"), [c4, c5, c6, c7]):
                col.metric(w, f"{kpis['wheel_mean_kw'][w]:.2f} kW  ({kpis['wheel_pct'][w]:.1f}%)")
            _plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {
                run_name: _pt_power_per_wheel_fig_cached(
                    {run_name: df}, _run_cache_tokens({run_name: df}), pt_x_mode
                )
                for run_name, df in dfs.items()
            }
            for _run_name, (_fig, kpis) in run_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(w)
            _show_summary_table(
                [
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
                ]
            )
            fig, kpis = _pt_power_per_wheel_fig_cached(dfs, run_tokens, pt_x_mode)
            _plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Power KPIs unavailable: {exc}")

    st.divider()

    # ── Inverter load ─────────────────────────────────────────────────────────
    st.subheader("Inverter Load (overload & i²t)")
    st.caption("How hard each inverter runs vs its thermal budget (overload & i²t).")
    try:
        if len(dfs) == 1:
            fig, kpis = _pt_inverter_limits_fig_cached(dfs, run_tokens)
            for w in kpis.get("warnings", []):
                st.warning(w)
            ov, ixt = kpis.get("overload_pct", {}), kpis.get("ixt_peak", {})
            if ov:
                cols = st.columns(4)
                for w, col in zip(("FL", "FR", "RL", "RR"), cols):
                    col.metric(
                        f"{w} overload",
                        f"{ov.get(w, float('nan')):.1f}%",
                        delta=f"IxT {ixt.get(w, float('nan')):.2f}",
                        delta_color="off",
                    )
            _plotly_chart(fig, use_container_width=True, theme=None)
            if "table" in kpis:
                with st.expander("Per-lap data"):
                    st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {
                run_name: _pt_inverter_limits_fig_cached(
                    {run_name: df}, _run_cache_tokens({run_name: df})
                )
                for run_name, df in dfs.items()
            }
            for _run_name, (_fig, kpis) in run_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(w)
            _show_summary_table(
                [
                    {
                        "Run": run_name,
                        **{
                            f"{w} overload [%]": round(
                                kpis.get("overload_pct", {}).get(w, float("nan")), 1
                            )
                            for w in ("FL", "FR", "RL", "RR")
                        },
                        "Worst IxT": round(
                            max(kpis.get("ixt_peak", {}).values(), default=float("nan")), 3
                        ),
                    }
                    for run_name, (_fig, kpis) in run_results.items()
                ]
            )
            fig, kpis = _pt_inverter_limits_fig_cached(dfs, run_tokens)
            _plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Inverter-load KPIs unavailable: {exc}")

    st.divider()

    # ── Torque fidelity ───────────────────────────────────────────────────────
    st.subheader("Torque Fidelity")
    st.caption("Tracking error between commanded and delivered motor torque (|target| > 0.5 Nm).")
    try:
        if len(dfs) == 1:
            fig, kpis = _pt_torque_fidelity_fig_cached(dfs, run_tokens)
            for w in kpis.get("warnings", []):
                st.warning(w)
            mae, bias = kpis.get("mae_nm", {}), kpis.get("bias_nm", {})
            if mae:
                cols = st.columns(4)
                for w, col in zip(("FL", "FR", "RL", "RR"), cols):
                    col.metric(
                        f"{w} MAE",
                        f"{mae.get(w, float('nan')):.2f} Nm",
                        delta=f"bias {bias.get(w, float('nan')):+.2f}",
                        delta_color="off",
                    )
                st.metric("Within ±1 Nm", f"{kpis.get('pct_within_1nm', float('nan')):.1f}%")
            _plotly_chart(fig, use_container_width=True, theme=None)
            if "table" in kpis:
                with st.expander("Per-lap data"):
                    st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {
                run_name: _pt_torque_fidelity_fig_cached(
                    {run_name: df}, _run_cache_tokens({run_name: df})
                )
                for run_name, df in dfs.items()
            }
            for _run_name, (_fig, kpis) in run_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(w)
            _show_summary_table(
                [
                    {
                        "Run": run_name,
                        **{
                            f"{w} MAE [Nm]": round(kpis.get("mae_nm", {}).get(w, float("nan")), 2)
                            for w in ("FL", "FR", "RL", "RR")
                        },
                        "Within ±1 Nm [%]": round(kpis.get("pct_within_1nm", float("nan")), 1),
                    }
                    for run_name, (_fig, kpis) in run_results.items()
                ]
            )
            fig, kpis = _pt_torque_fidelity_fig_cached(dfs, run_tokens)
            _plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Torque-fidelity KPIs unavailable: {exc}")

    st.divider()

    # ── Torque–speed envelope ─────────────────────────────────────────────────
    st.subheader("Torque–Speed Operating Map")
    st.caption("Where each motor operates vs the 27.5 Nm / 20 kW / rev limits.")
    try:
        fig, kpis = _pt_torque_speed_envelope_fig_cached(dfs, run_tokens)
        for w in kpis.get("warnings", []):
            st.warning(w)
        wheels = kpis.get("wheels", {})
        if wheels:
            _show_summary_table(
                [
                    {
                        "Wheel": w,
                        "Torque P95 [Nm]": round(d["torque_p95_nm"], 1),
                        "Speed P95 [rad/s]": round(d["speed_p95_rads"], 0),
                        "Torque sat [%]": round(d["pct_torque_saturated"], 1),
                        "Rev limited [%]": round(d["pct_rev_limited"], 2),
                    }
                    for w, d in wheels.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Torque–speed KPIs unavailable: {exc}")

    _render_pc_function_check(dfs)


def _render_pt_battery(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple,
    pt_x_mode: str,
) -> None:
    # ── Energy per lap ────────────────────────────────────────────────────────
    st.subheader("Energy per Lap")
    st.caption("Net battery energy per lap (consumed − recovered).")
    try:
        if len(dfs) == 1:
            fig, kpis = _pt_energy_per_lap_fig_cached(dfs, run_tokens, pt_x_mode)
            for w in kpis.get("warnings", []):
                st.warning(w)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean net / lap", f"{kpis['e_mean']:.4f} kWh")
            c2.metric("Total net", f"{kpis['e_total']:.3f} kWh")
            c3.metric("Coefficient of Variation", f"{kpis['cv']:.1f}%")
            c4.metric(
                f"R² (Enet vs {'laptime' if pt_x_mode == 'laptime' else 'lap'})",
                f"{kpis['r2']:.3f}",
            )
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Mean battery power", f"{kpis['p_mean']:.1f} kW")
            c6.metric("Consumed / lap", f"{kpis['e_cons_mean']:.4f} kWh")
            c7.metric("Recovered / lap", f"{kpis['e_rec_mean']:.4f} kWh")
            c8.metric("Fastest lap", f"L{kpis['fastest_lap']} — {kpis['fastest_lt']:.2f} s")
            _plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {
                run_name: _pt_energy_per_lap_fig_cached(
                    {run_name: df}, _run_cache_tokens({run_name: df}), pt_x_mode
                )
                for run_name, df in dfs.items()
            }
            for _run_name, (_fig, kpis) in run_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(w)
            _show_summary_table(
                [
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
                ]
            )
            _plotly_chart(
                _overlay_figures(
                    {run_name: [fig] for run_name, (fig, _kpis) in run_results.items()}
                )[0],
                use_container_width=True,
                theme=None,
            )
            with st.expander("Per-lap data"):
                st.dataframe(
                    _concat_run_tables(
                        {run_name: kpis["table"] for run_name, (_fig, kpis) in run_results.items()}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.error(f"Energy KPIs unavailable: {exc}")

    st.divider()

    # ── SoC per lap ───────────────────────────────────────────────────────────
    st.subheader("SoC per Lap")
    st.caption("End-of-lap state of charge and the per-lap SoC drop (autonomy).")
    try:
        if len(dfs) == 1:
            fig, kpis = _pt_soc_per_lap_fig_cached(dfs, run_tokens, pt_x_mode)
            for w in kpis.get("warnings", []):
                st.warning(w)
            c1, c2, c3 = st.columns(3)
            c1.metric("SoC start", f"{_fmt(kpis['soc_start'], '.1f')}%")
            c2.metric(
                "SoC end",
                f"{_fmt(kpis['soc_end'], '.1f')}%",
                delta=(
                    f"-{_fmt(kpis['soc_total_drop'], '.1f')}%"
                    if np.isfinite(kpis["soc_total_drop"])
                    else None
                ),
            )
            c3.metric("Mean SoC drop / lap", f"{kpis['soc_drop_per_lap']:.2f}%")
            _plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {
                run_name: _pt_soc_per_lap_fig_cached(
                    {run_name: df}, _run_cache_tokens({run_name: df}), pt_x_mode
                )
                for run_name, df in dfs.items()
            }
            for _run_name, (_fig, kpis) in run_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(w)
            _show_summary_table(
                [
                    {
                        "Run": run_name,
                        "Mean dSoC / lap [%]": round(kpis["soc_drop_per_lap"], 2),
                        "SoC start [%]": round(kpis["soc_start"], 1),
                        "SoC end [%]": round(kpis["soc_end"], 1),
                    }
                    for run_name, (_fig, kpis) in run_results.items()
                ]
            )
            _plotly_chart(
                _overlay_figures(
                    {run_name: [fig] for run_name, (fig, _kpis) in run_results.items()}
                )[0],
                use_container_width=True,
                theme=None,
            )
    except Exception as exc:
        st.error(f"SoC KPIs unavailable: {exc}")

    st.divider()

    # ── HV delivery efficiency ────────────────────────────────────────────────
    st.subheader("HV Delivery Efficiency")
    st.caption(
        "Fraction of pack DC power reaching the inverters (Σ inverter P ÷ battery P, drive only)."
    )
    try:
        fig, kpis = _pt_hv_delivery_efficiency_fig_cached(dfs, run_tokens)
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if runs:
            _show_summary_table(
                [
                    {
                        "Run": name,
                        "Median delivery η [%]": round(d["delivery_eff_median"] * 100, 1),
                        "Mean HV loss [kW]": round(d["mean_loss_kw"], 2),
                        "Samples": d["samples"],
                    }
                    for name, d in runs.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
        if "table" in kpis:
            with st.expander("Per-lap data"):
                st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"HV-efficiency KPIs unavailable: {exc}")

    st.divider()

    # ── Weakest cell under load ───────────────────────────────────────────────
    st.subheader("Weakest Cell Under Load")
    st.caption("Does the minimum cell voltage approach its discharge floor when current is drawn?")
    try:
        fig, kpis = _pt_weakest_cell_fig_cached(dfs, run_tokens)
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if len(runs) == 1:
            d = next(iter(runs.values()))
            c1, c2 = st.columns(2)
            c1.metric("Min cell under load", f"{_fmt(d['vmin_under_load_v'], '.2f')} V")
            c2.metric("Cell V at peak current", f"{_fmt(d['vmin_at_peak_current_v'], '.2f')} V")
        elif runs:
            _show_summary_table(
                [
                    {
                        "Run": name,
                        "Min cell under load [V]": round(d["vmin_under_load_v"], 2),
                        "Cell V at peak I [V]": round(d["vmin_at_peak_current_v"], 2),
                    }
                    for name, d in runs.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Weakest-cell KPIs unavailable: {exc}")


def _render_pt_temperatures(
    dfs: dict[str, pl.DataFrame],
    run_tokens: tuple,
    pt_x_mode: str,
) -> None:
    # ── Thermal evolution ─────────────────────────────────────────────────────
    st.subheader("Thermal Evolution")
    st.caption(
        "Motor, inverter and battery temperatures per lap (P95, glitch-cleaned), "
        "each with its OT limit."
    )
    try:
        if len(dfs) == 1:
            fig, kpis = _pt_thermal_evolution_fig_cached(dfs, run_tokens, pt_x_mode)
            for w in kpis.get("warnings", []):
                st.warning(w)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Peak motor (P95)", f"{kpis['peak_motor']:.1f} °C")
            c2.metric("Peak inverter (P95)", f"{kpis['peak_inverter']:.1f} °C")
            c3.metric("Peak battery Tmax", f"{kpis['peak_batt_tmax']:.1f} °C")
            c4.metric("Motor thermal slope", f"{kpis['motor_thermal_slope']:+.2f} °C/lap")
            c5, c6, c7, c8 = st.columns(4)
            for w, col in zip(("FL", "FR", "RL", "RR"), [c5, c6, c7, c8]):
                col.metric(f"Motor {w} peak", f"{kpis['motor_peak_by_wheel'][w]:.1f} °C")
            _plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
        else:
            run_results = {
                run_name: _pt_thermal_evolution_fig_cached(
                    {run_name: df}, _run_cache_tokens({run_name: df}), pt_x_mode
                )
                for run_name, df in dfs.items()
            }
            for _run_name, (_fig, kpis) in run_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(w)
            _show_summary_table(
                [
                    {
                        "Run": run_name,
                        "Peak motor [°C]": round(kpis["peak_motor"], 1),
                        "Peak inverter [°C]": round(kpis["peak_inverter"], 1),
                        "Peak batt Tmax [°C]": round(kpis["peak_batt_tmax"], 1),
                        "Motor slope [°C/lap]": round(kpis["motor_thermal_slope"], 2),
                    }
                    for run_name, (_fig, kpis) in run_results.items()
                ]
            )
            fig, kpis = _pt_thermal_evolution_fig_cached(dfs, run_tokens, pt_x_mode)
            _plotly_chart(fig, use_container_width=True, theme=None)
            with st.expander("Per-lap data"):
                st.dataframe(kpis["table"], use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Thermal KPIs unavailable: {exc}")

    st.divider()

    # ── Headroom & heat soak ──────────────────────────────────────────────────
    st.subheader("Headroom & Heat Soak")
    st.caption("Margin to each OT limit, the per-lap heat-soak slope and laps-to-limit.")
    try:
        fig, kpis = _pt_thermal_headroom_fig_cached(dfs, run_tokens)
        for w in kpis.get("warnings", []):
            st.warning(w)

        def _l2l(v: float) -> str:
            return "∞" if not np.isfinite(v) else f"{v:.0f}"

        if "motor" in kpis:
            dt_lr = {
                "motor": f"{kpis['dt_lr_motor_c']:+.2f}",
                "inverter": f"{kpis['dt_lr_inverter_c']:+.2f}",
                "battery": "—",
            }
            _show_summary_table(
                [
                    {
                        "Component": comp.capitalize(),
                        "Headroom [°C]": round(kpis[comp]["headroom_c"], 1),
                        "Heat-soak [°C/lap]": round(kpis[comp]["slope_c_per_lap"], 2),
                        "Laps to limit": _l2l(kpis[comp]["laps_to_limit"]),
                        "ΔT L−R [°C]": dt_lr[comp],
                    }
                    for comp in ("motor", "inverter", "battery")
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Headroom KPIs unavailable: {exc}")


def _tab_dynamics(dfs: dict[str, pl.DataFrame]) -> None:
    dyn_section = st.segmented_control(
        "Dynamics section",
        options=["Grip Factors", "Braking", "Cornering", "Acceleration", "Setup"],
        default="Grip Factors",
        required=True,
        key="dyn_subsection",
        label_visibility="collapsed",
        width="stretch",
    )

    if dyn_section == "Braking":
        _render_dynamics_braking(dfs)
        return
    if dyn_section == "Acceleration":
        _render_dynamics_acceleration(dfs)
        return
    if dyn_section == "Setup":
        _render_dynamics_setup_signatures(dfs)
        return
    if dyn_section == "Cornering":
        _render_dynamics_cornering(dfs)
        return

    _render_dynamics_grip_factors(dfs)


def _tab_events(dfs: dict[str, pl.DataFrame]) -> None:
    any_event_run = any(
        accel.is_acceleration_run(df) or skidpad.is_skidpad_run(df) for df in dfs.values()
    )
    if not any_event_run:
        st.info(
            "**Events** covers the Acceleration and Skidpad disciplines. The "
            "loaded run(s) are endurance/circuit telemetry (`lapcount_mode` is "
            "neither `acceleration` nor `skidpad`), so there is nothing to plot "
            "here. Load an `*_acc` or `*_skpd` run to use this section."
        )
        return
    section = st.segmented_control(
        "Events section",
        options=["Skidpad", "Acceleration"],
        default="Skidpad",
        required=True,
        key="events_subsection",
        label_visibility="collapsed",
        width="stretch",
    )
    if section == "Acceleration":
        _render_events_acceleration(dfs)
        return
    _render_events_skidpad(dfs)


def _filter_acceleration_mode_df(df: pl.DataFrame) -> pl.DataFrame:
    """Return only samples tagged as acceleration, preserving the input schema."""
    if "lapcount_mode" not in df.columns or len(df) == 0:
        return df.filter(pl.Series("__accel_mask", np.zeros(len(df), dtype=bool)))
    mask = np.array([str(v).lower() == "acceleration" for v in df["lapcount_mode"].to_list()])
    return df.filter(pl.Series("__accel_mask", mask))


_ACCEL_KPI_SPEC: list[tuple[str, str, str, bool]] = [
    # (display, kpi key, format, lower_is_better)
    ("Time [s]", "event_time_s", ".3f", True),
    ("Peak vx [km/h]", "peak_vx_kmh", ".1f", False),
    ("Mean ax [G]", "mean_ax_g", ".3f", False),
    ("Peak ax [G]", "peak_ax_g", ".3f", False),
    ("t 0–15 m [s]", "t_0_15m_s", ".3f", True),
    ("t 15–45 m [s]", "t_15_45m_s", ".3f", True),
    ("t 45–75 m [s]", "t_45_75m_s", ".3f", True),
    ("t→30 km/h [s]", "t_to_30kmh_s", ".3f", True),
    ("t→60 km/h [s]", "t_to_60kmh_s", ".3f", True),
    ("t→100 km/h [s]", "t_to_100kmh_s", ".3f", True),
    ("Launch ax 0–0.5 s [G]", "launch_ax_g_05s", ".3f", False),
    ("Launch SR peak", "launch_sr_peak", ".3f", True),
    ("Throttle rise 10→90 % [s]", "throttle_rise_time_s", ".3f", True),
    ("Mean ax 0–30 m [G]", "mean_ax_g_traction", ".3f", False),
    ("SR in band on throttle [%]", "pct_sr_in_band_on_throttle", ".1f", False),
    ("Wheelspin events", "wheelspin_events", ".0f", True),
    ("P_DC peak [kW]", "p_dc_peak_kw", ".1f", False),
    ("P_DC mean [kW]", "p_dc_mean_kw", ".1f", False),
    ("P_DC in 70–80 kW [%]", "pct_time_p_dc_70_80kw", ".1f", False),
    ("P_DC > 80 kW [%]", "pct_time_p_dc_over_80kw", ".2f", True),
    ("V_DC sag [%]", "v_dc_sag_pct", ".1f", True),
    ("I_DC peak [A]", "i_dc_peak_a", ".0f", False),
    ("Energy DC [kJ]", "energy_dc_kj", ".1f", True),
    ("F/R torque split [%]", "fr_torque_split_pct", ".1f", False),
    ("L/R imbalance front [%]", "lr_imbalance_front_pct", ".1f", True),
    ("L/R imbalance rear [%]", "lr_imbalance_rear_pct", ".1f", True),
    ("Full throttle [%]", "pct_full_thr", ".1f", False),
    ("SR MAE", "sr_mae_global", ".3f", True),
    ("SR in band [%]", "pct_all_in_band", ".1f", False),
    ("Any overslip [%]", "pct_any_overslip", ".1f", True),
]

# Hierarchy over the ~30 acceleration KPIs: a short headline set up front, the
# rest grouped into drill-down expanders so the summary no longer dumps as one
# flat 30-metric block. union(headline, all groups) covers every KPI key.
_ACCEL_HEADLINE_KEYS: tuple[str, ...] = (
    "event_time_s",
    "peak_vx_kmh",
    "t_to_100kmh_s",
    "mean_ax_g",
    "pct_sr_in_band_on_throttle",
    "p_dc_peak_kw",
)
_ACCEL_KPI_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Phase splits",
        (
            "t_0_15m_s",
            "t_15_45m_s",
            "t_45_75m_s",
            "t_to_30kmh_s",
            "t_to_60kmh_s",
            "t_to_100kmh_s",
        ),
    ),
    (
        "Launch",
        (
            "launch_ax_g_05s",
            "launch_sr_peak",
            "throttle_rise_time_s",
        ),
    ),
    (
        "Traction & TC",
        (
            "mean_ax_g_traction",
            "pct_sr_in_band_on_throttle",
            "wheelspin_events",
            "sr_mae_global",
            "pct_all_in_band",
            "pct_any_overslip",
        ),
    ),
    (
        "Power (DC bus)",
        (
            "p_dc_peak_kw",
            "p_dc_mean_kw",
            "pct_time_p_dc_70_80kw",
            "pct_time_p_dc_over_80kw",
            "v_dc_sag_pct",
            "i_dc_peak_a",
            "energy_dc_kj",
        ),
    ),
    (
        "Drivetrain & misc",
        (
            "peak_ax_g",
            "fr_torque_split_pct",
            "lr_imbalance_front_pct",
            "lr_imbalance_rear_pct",
            "pct_full_thr",
        ),
    ),
)
_ACCEL_SPEC_BY_KEY: dict[str, tuple[str, str, bool]] = {
    key: (display, fmt, lb) for display, key, fmt, lb in _ACCEL_KPI_SPEC
}


def _accel_kpi_cards(k: dict, keys: tuple[str, ...]) -> None:
    """Render the given KPI keys as st.metric cards, in rows of four."""
    items = [key for key in keys if key in _ACCEL_SPEC_BY_KEY]
    for i in range(0, len(items), 4):
        cols = st.columns(4)
        for col, key in zip(cols, items[i : i + 4]):
            display, fmt, _lb = _ACCEL_SPEC_BY_KEY[key]
            value = k.get(key)
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                col.metric(display, format(float(value), fmt))
            else:
                col.metric(display, "—")


def _accel_kpi_table(kpis_by_run: dict[str, dict], keys: tuple[str, ...]) -> None:
    """Render the given KPI keys as a styled per-run comparison table."""
    run_keys = list(kpis_by_run.keys())
    table_rows: list[dict] = []
    lower_better_dict: dict[str, bool] = {}
    for key in keys:
        if key not in _ACCEL_SPEC_BY_KEY:
            continue
        display, _fmt_spec, lb = _ACCEL_SPEC_BY_KEY[key]
        row: dict[str, object] = {"Metric": display}
        for rn in run_keys:
            val = kpis_by_run[rn].get(key)
            row[Path(rn).stem] = (
                float(val)
                if isinstance(val, (int, float)) and np.isfinite(float(val))
                else float("nan")
            )
        table_rows.append(row)
        lower_better_dict[display] = lb
    if not table_rows:
        return
    metrics_df = pl.DataFrame(table_rows)
    try:
        st.dataframe(
            style_metrics_table(metrics_df, lower_better=lower_better_dict),
            use_container_width=True,
            hide_index=True,
        )
    except Exception as exc:
        st.warning(f"Styled metrics table unavailable: {exc}")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)


def _render_events_acceleration(dfs: dict[str, pl.DataFrame]) -> None:
    accel_dfs: dict[str, pl.DataFrame] = {}
    for run_name, df in dfs.items():
        if not accel.is_acceleration_run(df):
            continue
        filtered = _filter_acceleration_mode_df(df)
        if not filtered.is_empty():
            accel_dfs[run_name] = filtered

    if not accel_dfs:
        st.warning("No acceleration data in selected runs (lapcount_mode != 'acceleration').")
        return

    run_tokens = _run_cache_tokens(accel_dfs)

    # ── Per-run KPIs ──────────────────────────────────────────────────────────
    st.subheader("Acceleration Summary")
    kpis_by_run: dict[str, dict] = {}
    for run_name, df in accel_dfs.items():
        kpis = accel.summary_kpis(df)
        for w in kpis.get("warnings", []):
            st.warning(f"{Path(run_name).stem}: {w}")
        if "event_time_s" not in kpis:
            continue
        kpis_by_run[run_name] = kpis

    if not kpis_by_run:
        return

    st.caption("Per-run acceleration-event metrics — event time, speeds and slip ratios.")
    if len(kpis_by_run) == 1:
        run_name = next(iter(kpis_by_run))
        k = kpis_by_run[run_name]
        _accel_kpi_cards(k, _ACCEL_HEADLINE_KEYS)
        for group_name, group_keys in _ACCEL_KPI_GROUPS:
            with st.expander(group_name):
                _accel_kpi_cards(k, group_keys)
        worst = k.get("worst_wheel", "")
        if worst:
            st.caption(f"Worst SR wheel: **{worst}**")
    else:
        _accel_kpi_table(kpis_by_run, _ACCEL_HEADLINE_KEYS)
        for group_name, group_keys in _ACCEL_KPI_GROUPS:
            with st.expander(group_name):
                _accel_kpi_table(kpis_by_run, group_keys)

    # ── Diagnostic scatter figures ────────────────────────────────────────────
    _ACCEL_FIGURES = [
        (
            "power_vx",
            "DC Power vs Speed  ·  80 kW regulatory limit",
            "DC battery power vs speed against the 80 kW regulatory limit.",
        ),
        (
            "ax_vx",
            "Acceleration Envelope  ·  Longitudinal G vs Speed",
            "Longitudinal g achieved across the speed range during the run.",
        ),
        (
            "motor_tq_rpm",
            "Powertrain Operating Cloud  ·  Motor Torque vs Wheel Speed",
            "Where the motors operate on the torque–speed plane.",
        ),
        (
            "sr_fr_balance",
            "Traction Balance  ·  Front vs Rear Slip Ratio",
            "Front vs rear slip ratio — how traction splits across the axles.",
        ),
        (
            "sr_fx_wheel",
            "TC Operating Point  ·  SR vs Fx per Wheel",
            "Per-wheel slip ratio vs drive force — the TC operating point.",
        ),
    ]
    for metric, title, caption in _ACCEL_FIGURES:
        st.divider()
        st.subheader(title)
        st.caption(caption)
        try:
            fig, fig_kpis = _accel_fig_cached(metric, accel_dfs, run_tokens)
        except Exception as exc:
            st.error(f"{title} unavailable: {exc}")
            continue
        for w in fig_kpis.get("warnings", []):
            st.warning(w)
        if fig is not None:
            _plotly_chart(fig, use_container_width=True, theme=None)


def _render_events_skidpad(dfs: dict[str, pl.DataFrame]) -> None:
    skidpad_dfs: dict[str, pl.DataFrame] = {}
    for run_name, df in dfs.items():
        if not skidpad.is_skidpad_run(df):
            continue
        filtered = _filter_skidpad_mode_df(df)
        if not filtered.is_empty():
            skidpad_dfs[run_name] = filtered

    if not skidpad_dfs:
        st.warning("No skidpad data in selected runs (lapcount_mode != 'skidpad').")
        return

    _render_skidpad_top_kpis(skidpad_dfs)

    for metric, title, caption in (
        (
            "event_time",
            "Event Time Summary",
            "Timed lap per direction and the left/right time asymmetry.",
        ),
        ("lateral_g", "Lateral G", "Sustained lateral g held around each skidpad circle."),
        ("driven_radius", "Driven Radius", "Driven radius vs the skidpad reference radius."),
        ("balance", "Balance", "Steering vs lateral g — the car's understeer/oversteer balance."),
    ):
        st.divider()
        _render_skidpad_plot(metric, title, skidpad_dfs, caption=caption)

    st.divider()
    st.subheader("Setup Signatures")
    st.caption("Setup-sensitive skidpad signatures: understeer gradient and lateral load split.")
    _render_skidpad_plot(
        "understeer_gradient",
        "Understeer Gradient",
        skidpad_dfs,
        caption="Steering demand vs lateral g — the steady-state understeer gradient.",
    )

    load_dfs = {name: df for name, df in skidpad_dfs.items() if skidpad.has_load_signals(df)}
    if load_dfs:
        _render_skidpad_plot(
            "lateral_load_dist",
            "Lateral Load Distribution",
            load_dfs,
            caption="Front share of lateral load transfer on the skidpad.",
        )
    else:
        st.info("LLTD skipped: missing `Est_FZFL/FR/RL/RR` load signals.")

    _render_skidpad_plot(
        "lr_asymmetry",
        "Right-Left Asymmetry",
        skidpad_dfs,
        caption="Left vs right circle — setup or track asymmetry.",
    )

    st.divider()
    _render_skidpad_plot(
        "gps_figure8",
        "GPS Figure-8",
        skidpad_dfs,
        caption="GPS trace of the two skidpad circles.",
        show_table=False,
    )

    tv_dfs = {name: df for name, df in skidpad_dfs.items() if skidpad.has_tv_signals(df)}
    if tv_dfs:
        st.divider()
        _render_skidpad_plot(
            "tv_intervention",
            "TV Intervention",
            tv_dfs,
            caption="Torque-vectoring activity around the skidpad.",
        )
    else:
        st.info("TV intervention skipped: missing TV torque or `TV_errorYawRate` signals.")


def _filter_skidpad_mode_df(df: pl.DataFrame) -> pl.DataFrame:
    """Return only samples tagged as skidpad, preserving the input schema."""
    if "lapcount_mode" not in df.columns or len(df) == 0:
        return df.filter(pl.Series("__skidpad_mask", np.zeros(len(df), dtype=bool)))
    mask = np.array([str(value).lower() == "skidpad" for value in df["lapcount_mode"].to_list()])
    return df.filter(pl.Series("__skidpad_mask", mask))


def _render_skidpad_plot(
    metric: str,
    title: str,
    skidpad_dfs: dict[str, pl.DataFrame],
    *,
    caption: str = "",
    show_table: bool = True,
) -> None:
    st.subheader(title)
    if caption:
        st.caption(caption)
    run_tokens = _run_cache_tokens(skidpad_dfs)
    try:
        fig, kpis = _skidpad_fig_cached(metric, skidpad_dfs, run_tokens)
    except Exception as exc:
        st.error(f"{title.lower()} unavailable: {exc}")
        return
    for warning in kpis.get("warnings", []):
        st.warning(warning)
    _plotly_chart(fig, use_container_width=True, theme=None)
    table = kpis.get("table")
    if show_table and isinstance(table, pl.DataFrame) and not table.is_empty():
        with st.expander("Per-run data"):
            st.dataframe(table, use_container_width=True, hide_index=True)


def _render_skidpad_top_kpis(skidpad_dfs: dict[str, pl.DataFrame]) -> None:
    st.subheader("Skidpad Summary")
    st.caption(
        "Timed skidpad circles only (warm-up excluded); the fastest lap per direction is "
        "the official run."
    )
    values: dict[str, dict[str, float]] = {
        run_name: {
            "event_time_s": np.nan,
            "lr_asymmetry_s": np.nan,
            "ay_sustained_mean_g": np.nan,
            "radius_error_m": np.nan,
            "understeer_angle_deg": np.nan,
        }
        for run_name in skidpad_dfs
    }

    metric_keys = {
        "event_time": ("event_time_s", "LR_asymmetry_s"),
        "lateral_g": ("ay_sustained_mean_g",),
        "driven_radius": ("radius_error_m",),
        "understeer_gradient": ("understeer_angle_deg",),
    }
    for metric, keys in metric_keys.items():
        try:
            _fig, kpis = _skidpad_fig_cached(metric, skidpad_dfs, _run_cache_tokens(skidpad_dfs))
        except Exception:
            continue
        for run_name, run_vals in kpis.get("runs", {}).items():
            for key in keys:
                target = "lr_asymmetry_s" if key == "LR_asymmetry_s" else key
                values[run_name][target] = run_vals.get(key, np.nan)

    for run_name in skidpad_dfs:
        run_values = values[run_name]
        if len(skidpad_dfs) > 1:
            st.markdown(f"**{Path(run_name).stem}**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Event Time", f"{_fmt(run_values['event_time_s'], '.3f')} s")
        c2.metric("L/R Asym", f"{_fmt(run_values['lr_asymmetry_s'], '.3f')} s")
        c3.metric("ay medio (estac.)", f"{_fmt(run_values['ay_sustained_mean_g'], '.2f')} g")
        c4.metric("R Error", f"{_fmt(run_values['radius_error_m'], '+.2f')} m")
        c5.metric("Subviraje", f"{_fmt(run_values['understeer_angle_deg'], '+.2f')} deg")


def _render_dynamics_braking(dfs: dict[str, pl.DataFrame]) -> None:
    """Vehicle dynamic behavior under braking: capability, balance, load transfer, limits."""
    single = len(dfs) == 1
    tokens = _run_cache_tokens(dfs)

    # 1 - Decel Envelope ------------------------------------------------------
    st.subheader("Decel Envelope  ·  Longitudinal G vs Speed")
    st.caption(
        "P95 deceleration envelope by 1 m/s speed bin vs the 1.79 g vehicle max deceleration."
    )
    try:
        fig, kpis = _dyn_decel_envelope_fig_cached(dfs, tokens)
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if single and runs:
            v = next(iter(runs.values()))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Peak P95 decel [g]",
                f"{_fmt(v.get('peak_decel_p95_g', np.nan), '.3f')}",
                help=METRIC_HELP["Peak P95 decel"],
            )
            c2.metric(
                "Peak speed [m/s]",
                f"{_fmt(v.get('speed_at_peak_mps', np.nan), '.1f')}",
                help=METRIC_HELP["Peak decel speed"],
            )
            c3.metric(
                "Gap to max decel [g]",
                f"{_fmt(v.get('gap_to_design_g', np.nan), '+.3f')}",
                help=METRIC_HELP["Gap to max decel"],
            )
            c4.metric(
                "Max decel use [%]",
                f"{_fmt(v.get('pct_design_decel', np.nan), '.1f')}%",
                help=METRIC_HELP["Max decel use"],
            )
        elif runs:
            _show_summary_table(
                [
                    {
                        "Run": run_name,
                        "Peak p95 decel [g]": round(vals.get("peak_decel_p95_g", np.nan), 3),
                        "Peak speed [m/s]": round(vals.get("speed_at_peak_mps", np.nan), 1),
                        "Gap [g]": round(vals.get("gap_to_design_g", np.nan), 3),
                        "Max decel use [%]": round(vals.get("pct_design_decel", np.nan), 1),
                        "Samples": int(vals.get("samples", 0)),
                    }
                    for run_name, vals in runs.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Decel envelope unavailable: {exc}")

    st.divider()

    # 2 - Brake Blending ------------------------------------------------------
    st.subheader("Brake Blending  ·  Hydraulic Pressure vs Pedal")
    st.caption(
        "Median front/rear line pressure (bar) by pedal % — below the blend threshold "
        "braking is regen-only; above it the hydraulics engage."
    )
    try:
        fig, kpis = _dyn_brake_blending_fig_cached(dfs, tokens)
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if single and runs:
            v = next(iter(runs.values()))
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Blend threshold [%]",
                f"{_fmt(v.get('blend_threshold_pct', np.nan), '.0f')}",
                help=METRIC_HELP["Blend threshold [%]"],
            )
            c2.metric(
                "Front peak P [bar]",
                f"{_fmt(v.get('front_peak_pressure_bar', np.nan), '.1f')}",
                help=METRIC_HELP["Front peak P [bar]"],
            )
            c3.metric(
                "Rear peak P [bar]",
                f"{_fmt(v.get('rear_peak_pressure_bar', np.nan), '.1f')}",
                help=METRIC_HELP["Rear peak P [bar]"],
            )
        elif runs:
            _show_summary_table(
                [
                    {
                        "Run": rn,
                        "Blend threshold [%]": round(v.get("blend_threshold_pct", np.nan), 0),
                        "Front peak [bar]": round(v.get("front_peak_pressure_bar", np.nan), 1),
                        "Rear peak [bar]": round(v.get("rear_peak_pressure_bar", np.nan), 1),
                        "Samples": int(v.get("samples", 0)),
                    }
                    for rn, v in runs.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Brake blending unavailable: {exc}")

    st.divider()

    # 3 - Ideal Brake Distribution --------------------------------------------
    st.subheader("Ideal Brake Distribution  ·  Total Fx Front vs Rear")
    st.caption(
        "Total front vs rear braking force vs the load-proportional ideal (coloured by speed) "
        "— above the curve = rear over-braked, below = front over-braked."
    )
    try:
        fig, kpis = _dyn_ideal_brake_distribution_fig_cached(dfs, tokens)
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if single and runs:
            v = next(iter(runs.values()))
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(
                "Front bias",
                f"{_fmt(v.get('front_bias_mean', np.nan) * 100.0, '.1f')} %",
                help=METRIC_HELP["Brake front bias"],
            )
            c2.metric(
                "Front bias (ideal)",
                f"{_fmt(v.get('front_bias_ideal_mean', np.nan) * 100.0, '.1f')} %",
                help=METRIC_HELP["Brake front bias ideal"],
            )
            c3.metric(
                "Bias error",
                f"{_fmt(v.get('bias_error_mean', np.nan) * 100.0, '+.1f')} pp",
                help=METRIC_HELP["Brake bias error"],
            )
            c4.metric(
                "Dist to ideal [N]",
                f"{_fmt(v.get('rms_dist_to_ideal_N', np.nan), '.0f')}",
                help=METRIC_HELP["Dist to ideal brake [N]"],
            )
            c5.metric(
                "Peak combined [g]",
                f"{_fmt(v.get('peak_combined_brake_g', np.nan), '.2f')}",
                help=METRIC_HELP["Peak combined brake [g]"],
            )
        elif runs:
            _show_summary_table(
                [
                    {
                        "Run": rn,
                        "Front bias [%]": round(v.get("front_bias_mean", np.nan) * 100.0, 1),
                        "Ideal bias [%]": round(v.get("front_bias_ideal_mean", np.nan) * 100.0, 1),
                        "Bias error [pp]": round(v.get("bias_error_mean", np.nan) * 100.0, 1),
                        "Dist to ideal [N]": round(v.get("rms_dist_to_ideal_N", np.nan), 0),
                        "Peak combined [g]": round(v.get("peak_combined_brake_g", np.nan), 2),
                        "Samples": int(v.get("samples", 0)),
                    }
                    for rn, v in runs.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Ideal brake distribution unavailable: {exc}")

    st.divider()

    # 4 - Per-axle Brake Utilisation ------------------------------------------
    st.subheader("Per-axle Brake Utilisation  ·  |Fx| / (mu·Fz)")
    st.caption(
        "Front vs rear brake-grip utilisation |Fx|/(μ·Fz) by decel — the axle reaching 1.0 "
        "first locks first."
    )
    try:
        fig, kpis = _dyn_axle_brake_utilisation_fig_cached(dfs, tokens)
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if single and runs:
            v = next(iter(runs.values()))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Brake util front",
                f"{_fmt(v['util_front_median'], '.3f')}",
                help=METRIC_HELP["Brake util front"],
            )
            c2.metric(
                "Brake util rear",
                f"{_fmt(v['util_rear_median'], '.3f')}",
                help=METRIC_HELP["Brake util rear"],
            )
            c3.metric(
                "Front-rear gap (p95)",
                f"{_fmt(v['util_gap_p95'], '+.3f')}",
                help=METRIC_HELP["Front-rear util gap"],
            )
            c4.metric(
                "Limiting axle",
                v.get("limiting_axle", "-"),
                help=METRIC_HELP["Limiting axle (brake)"],
            )
        elif runs:
            _show_summary_table(
                [
                    {
                        "Run": rn,
                        "Util front (med)": round(v.get("util_front_median", np.nan), 3),
                        "Util rear (med)": round(v.get("util_rear_median", np.nan), 3),
                        "Util front (p95)": round(v.get("util_front_p95", np.nan), 3),
                        "Util rear (p95)": round(v.get("util_rear_p95", np.nan), 3),
                        "Gap p95 (F-R)": round(v.get("util_gap_p95", np.nan), 3),
                        "Limiting axle": v.get("limiting_axle", "-"),
                        "Samples": int(v.get("samples", 0)),
                    }
                    for rn, v in runs.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Brake utilisation unavailable: {exc}")

    st.divider()

    # 5 - Per-axle Braking Slip -----------------------------------------------
    st.subheader("Per-axle Braking Slip  ·  Est_SR vs Decel")
    st.caption(
        "Front vs rear kinematic braking slip (Est_SR) — the cross-check of the force "
        "utilisation above."
    )
    try:
        fig, kpis = _dyn_axle_brake_slip_fig_cached(dfs, tokens)
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if single and runs:
            v = next(iter(runs.values()))
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Brake slip front",
                f"{_fmt(v['sr_front_median'], '+.3f')}",
                help=METRIC_HELP["Brake slip front"],
            )
            c2.metric(
                "Brake slip rear",
                f"{_fmt(v['sr_rear_median'], '+.3f')}",
                help=METRIC_HELP["Brake slip rear"],
            )
            c3.metric(
                "Axle nearer lock-up",
                v.get("axle_nearer_lockup", "-"),
                help=METRIC_HELP["Axle nearer lock-up"],
            )
        elif runs:
            _show_summary_table(
                [
                    {
                        "Run": rn,
                        "Slip front (med)": round(v.get("sr_front_median", np.nan), 4),
                        "Slip rear (med)": round(v.get("sr_rear_median", np.nan), 4),
                        "Slip front (p5)": round(v.get("sr_front_p5", np.nan), 4),
                        "Slip rear (p5)": round(v.get("sr_rear_p5", np.nan), 4),
                        "Nearer lock-up": v.get("axle_nearer_lockup", "-"),
                        "Samples": int(v.get("samples", 0)),
                    }
                    for rn, v in runs.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Braking slip unavailable: {exc}")

    st.divider()

    # 6 - Wheel locking time (relocated from Controls › RB) -------------------
    st.subheader("Wheel Locking Time  ·  Est_SR < −0.30 under braking")
    lock_view = st.segmented_control(
        "Wheel lock-up view",
        options=["Per lap", "Track map"],
        default="Per lap",
        key="dyn_wheel_lockup_view",
        label_visibility="collapsed",
    )
    if lock_view == "Track map":
        st.caption(
            "Where each wheel crosses the −0.30 lock-up line while braking (colour = wheel)."
        )
        results: dict[str, tuple[go.Figure, dict]] = {}
        for rn in dfs:
            try:
                results[rn] = _dyn_wheel_lockup_track_map_fig_cached(dfs, tokens, rn)
            except Exception as exc:  # noqa: BLE001
                st.error(f"{Path(rn).stem}: wheel lock-up unavailable: {exc}")
        for rn, (_fig, kpis) in results.items():
            for w in kpis.get("warnings", []):
                st.warning(f"{Path(rn).stem}: {w}")

        # KPI block above the maps: cards for one run, table for several.
        valid = {rn: k for rn, (_f, k) in results.items() if k.get("time_per_wheel")}
        if single and valid:
            kpis = next(iter(valid.values()))
            events = sum(kpis.get("events_per_wheel", {}).values())
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Lock-up time [s]",
                f"{_fmt(kpis['lockup_total_time_s'], '.2f')}",
                help=METRIC_HELP["Lock-up time [s]"],
            )
            c2.metric("Lock-up events", str(int(events)), help=METRIC_HELP["Lock-up events"])
            c3.metric(
                "Worst SR",
                f"{_fmt(kpis['worst_min_sr'], '+.3f')}",
                help=METRIC_HELP["Worst SR"],
            )
        elif valid:
            _show_summary_table(
                [
                    {
                        "Run": rn,
                        "Lock-up time [s]": round(k.get("lockup_total_time_s", np.nan), 3),
                        "Lock-up events": int(sum(k.get("events_per_wheel", {}).values())),
                        "Worst SR": round(k.get("worst_min_sr", np.nan), 3),
                        "Laps": int(k.get("laps", 0)),
                    }
                    for rn, k in valid.items()
                ]
            )

        # Maps side by side, one column per run.
        if results:
            map_cols = st.columns(len(results))
            for col, (rn, (fig, _k)) in zip(map_cols, results.items()):
                with col:
                    if not single:
                        st.markdown(f"**{Path(rn).stem}**")
                    _plotly_chart(fig, use_container_width=True, theme=None)
        return

    st.caption(
        "Per-corner time past the −0.30 lock-up line while braking, by lap (top-down car view)."
    )
    try:
        fig, kpis = _dyn_wheel_lockup_per_lap_fig_cached(dfs, tokens)
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if single and runs:
            v = next(iter(runs.values()))
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Lock-up time [s]",
                f"{_fmt(v['lockup_total_time_s'], '.2f')}",
                help=METRIC_HELP["Lock-up time [s]"],
            )
            c2.metric(
                "Lock-up events",
                str(int(v["lockup_events_total"])),
                help=METRIC_HELP["Lock-up events"],
            )
            c3.metric(
                "Worst SR", f"{_fmt(v['worst_min_sr'], '+.3f')}", help=METRIC_HELP["Worst SR"]
            )
        elif runs:
            _show_summary_table(
                [
                    {
                        "Run": rn,
                        "Lock-up time [s]": round(v.get("lockup_total_time_s", np.nan), 3),
                        "Lock-up events": int(v.get("lockup_events_total", 0)),
                        "Worst SR": round(v.get("worst_min_sr", np.nan), 3),
                        "Laps": int(v.get("laps", 0)),
                    }
                    for rn, v in runs.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Wheel lock-up unavailable: {exc}")


def _render_dynamics_acceleration(dfs: dict[str, pl.DataFrame]) -> None:
    """Vehicle traction performance: capability, limiting regime and the 4WD axle story."""
    single = len(dfs) == 1
    tokens = _run_cache_tokens(dfs)

    # 1 · Accel Envelope ------------------------------------------------------
    st.subheader("Accel Envelope  ·  Longitudinal G vs Speed")
    try:
        fig, kpis = _dyn_accel_envelope_fig_cached(dfs, tokens)
        crossover = kpis.get("crossover_mps", float("nan"))
        st.caption(
            "P95 ax by speed bin vs the grip-μ and 80 kW power references; "
            f"grip→power crossover ≈ {_fmt(crossover, '.1f')} m/s."
        )
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if single and runs:
            v = next(iter(runs.values()))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Peak ax (P95)", f"{_fmt(v['peak_ax_g'], '.2f')} g")
            c2.metric(
                "Limited by",
                f"{_fmt(v['pct_grip_limited'], '.0f')}% grip / {_fmt(v['pct_power_limited'], '.0f')}% power",
            )
            c3.metric("Traction eff. (grip)", f"{_fmt(v['traction_efficiency_grip_pct'], '.0f')} %")
            c4.metric("Samples", f"{int(v['samples'])}")
        elif runs:
            _show_summary_table(
                [
                    {
                        "Run": r,
                        "Peak ax [g]": round(v["peak_ax_g"], 3),
                        "Grip-limited [%]": round(v["pct_grip_limited"], 1),
                        "Power-limited [%]": round(v["pct_power_limited"], 1),
                        "Traction eff. grip [%]": round(v["traction_efficiency_grip_pct"], 1),
                        "Samples": int(v["samples"]),
                    }
                    for r, v in runs.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Accel envelope unavailable: {exc}")

    st.divider()

    # 2 · Traction Slip Curve -------------------------------------------------
    st.subheader("Traction Slip Curve  ·  Est_SR vs longitudinal grip")
    st.caption(
        "Front vs rear kinematic drive slip (Est_SR) as grip builds — the on-throttle "
        "mirror of the braking-slip view, against the +0.20 optimum."
    )
    try:
        fig, kpis = _dyn_traction_slip_curve_fig_cached(dfs, tokens)
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if single and runs:
            v = next(iter(runs.values()))
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Drive slip front",
                f"{_fmt(v['sr_front_median'], '+.3f')}",
                help=METRIC_HELP["Drive slip front"],
            )
            c2.metric(
                "Drive slip rear",
                f"{_fmt(v['sr_rear_median'], '+.3f')}",
                help=METRIC_HELP["Drive slip rear"],
            )
            c3.metric(
                "Axle more slip",
                v.get("axle_more_slip", "-"),
                help=METRIC_HELP["Axle more slip"],
            )
        elif runs:
            _show_summary_table(
                [
                    {
                        "Run": r,
                        "Slip front (med)": round(v.get("sr_front_median", np.nan), 4),
                        "Slip rear (med)": round(v.get("sr_rear_median", np.nan), 4),
                        "Slip front (p95)": round(v.get("sr_front_p95", np.nan), 4),
                        "Slip rear (p95)": round(v.get("sr_rear_p95", np.nan), 4),
                        "More slip": v.get("axle_more_slip", "-"),
                        "Samples": int(v.get("samples", 0)),
                    }
                    for r, v in runs.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Traction slip curve unavailable: {exc}")

    st.divider()

    # 3 · Per-axle Traction Utilisation --------------------------------------
    st.subheader("Per-axle Traction Utilisation  ·  Fx / (μ·Fz)")
    st.caption(
        "Front vs rear drive-grip utilisation Fx/(μ·Fz) — the axle reaching 1.0 first "
        "limits traction."
    )
    try:
        fig, kpis = _dyn_axle_traction_utilisation_fig_cached(dfs, tokens)
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if single and runs:
            v = next(iter(runs.values()))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Util front (med)", f"{_fmt(v['util_front_median'], '.2f')}")
            c2.metric("Util rear (med)", f"{_fmt(v['util_rear_median'], '.2f')}")
            c3.metric("Limiting axle", str(v["limiting_axle"]).title())
            c4.metric("Samples", f"{int(v['samples'])}")
        elif runs:
            _show_summary_table(
                [
                    {
                        "Run": r,
                        "Util front [-]": round(v["util_front_median"], 3),
                        "Util rear [-]": round(v["util_rear_median"], 3),
                        "Limiting axle": str(v["limiting_axle"]).title(),
                        "Samples": int(v["samples"]),
                    }
                    for r, v in runs.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Axle utilisation unavailable: {exc}")

    st.divider()

    # 4 · Drive Distribution vs Load Transfer --------------------------------
    st.subheader("Drive Distribution vs Load Transfer")
    st.caption(
        "Whether the front/rear drive split follows the rearward load shift under "
        "acceleration (maximum 4WD traction)."
    )
    try:
        fig, kpis = _dyn_ideal_traction_curve_fig_cached(dfs, tokens)
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if single and runs:
            v = next(iter(runs.values()))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rear bias (real)", f"{_fmt(v['rear_bias_mean'] * 100.0, '.1f')} %")
            c2.metric("Rear bias (ideal)", f"{_fmt(v['rear_bias_ideal_mean'] * 100.0, '.1f')} %")
            c3.metric("Peak accel (P95)", f"{_fmt(v['peak_combined_accel_g'], '.2f')} g")
            c4.metric("Samples", f"{int(v['samples'])}")
        elif runs:
            _show_summary_table(
                [
                    {
                        "Run": r,
                        "Rear bias [%]": round(v["rear_bias_mean"] * 100.0, 1),
                        "Ideal rear [%]": round(v["rear_bias_ideal_mean"] * 100.0, 1),
                        "Peak accel [g]": round(v["peak_combined_accel_g"], 3),
                        "Samples": int(v["samples"]),
                    }
                    for r, v in runs.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Drive distribution unavailable: {exc}")


def _render_dynamics_setup_signatures(dfs: dict[str, pl.DataFrame]) -> None:
    """Setup signatures grouped by adjustable knob: masses -> roll split -> ride
    heights -> dampers (ordered along the load chain)."""
    multi = len(dfs) > 1
    st.caption(
        "Camber and toe leave no signature in this telemetry — those setup calls go via "
        "driver feedback and tyre wear."
    )
    if not dyn.DAMPER_CALIBRATED:
        st.caption(
            ":gear: Damper pots are uncalibrated raw counts (the `Damp*` channel has no "
            "documented unit in Variables_CSV). Roll/pitch gradients and damper velocities "
            "show shape only; set DAMPER_COUNTS_PER_MM / DAMPER_MOTION_RATIO in "
            "src/dynamics.py to unlock absolute deg/g and mm/s."
        )

    # 1 - MASSES (static) - knob: ballast / corner weighting ----------------------
    st.divider()
    st.subheader("① Masses  ·  Static Fz Reference")
    st.caption(
        "Per-corner **calculated** Fz over straight low-input samples vs the 706 N/corner "
        "design (green ≤±5%, yellow ≤±10%, red >±10%)."
    )
    static_results: dict[str, tuple[go.Figure, dict]] = {}
    for run_name, df in dfs.items():
        try:
            static_results[run_name] = dyn.static_fz_reference_fig(df)
        except Exception as exc:
            st.error(f"Static Fz reference unavailable ({Path(run_name).stem}): {exc}")
    if static_results:
        if not multi:
            fig, kpis = next(iter(static_results.values()))
            for w in kpis.get("warnings", []):
                st.warning(w)
            corners_data = kpis.get("corners", {})
            c1, c2, c3, c4 = st.columns(4)
            for col, corner in zip([c1, c2, c3, c4], ["FL", "FR", "RL", "RR"]):
                cd = corners_data.get(corner, {})
                col.metric(
                    corner,
                    f"{_fmt(cd.get('measured_n', float('nan')), '.0f')} N",
                    f"{_fmt(cd.get('deviation_pct', float('nan')), '+.1f')} % vs design",
                )
            c1, c2 = st.columns(2)
            fs = kpis.get("front_share_pct", float("nan"))
            c1.metric("Front share", f"{_fmt(fs, '.1f')} %", f"{_fmt(fs - 50.0, '+.1f')} % vs 50%")
            c2.metric("Samples", str(kpis.get("samples", 0)))
            _plotly_chart(fig, use_container_width=True, theme=None)
        else:
            for rn, (_fig, kpis) in static_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(f"{Path(rn).stem}: {w}")
            _show_summary_table(
                [
                    {
                        "Run": Path(rn).stem,
                        "Front share [%]": _fmt(kpis.get("front_share_pct", float("nan")), ".1f"),
                        "FL dev [%]": _fmt(
                            kpis.get("corners", {}).get("FL", {}).get("deviation_pct", np.nan),
                            "+.1f",
                        ),
                        "FR dev [%]": _fmt(
                            kpis.get("corners", {}).get("FR", {}).get("deviation_pct", np.nan),
                            "+.1f",
                        ),
                        "RL dev [%]": _fmt(
                            kpis.get("corners", {}).get("RL", {}).get("deviation_pct", np.nan),
                            "+.1f",
                        ),
                        "RR dev [%]": _fmt(
                            kpis.get("corners", {}).get("RR", {}).get("deviation_pct", np.nan),
                            "+.1f",
                        ),
                        "Samples": kpis.get("samples", 0),
                    }
                    for rn, (_fig, kpis) in static_results.items()
                ]
            )
            fig_static = _overlay_setup_single_figure(static_results)
            if fig_static is not None:
                _plotly_chart(fig_static, use_container_width=True, theme=None)

    # 2 - ROLL SPLIT - knob: ARB / springs ----------------------------------------
    st.divider()
    st.subheader("② Roll Split  ·  LLTD Mid-Corner Avg per Lap")
    st.caption(
        "Front share of lateral load transfer, mid-corner average per lap, vs the "
        "roll-stiffness split "
        f"({dyn.KROLLF_NMRAD / (dyn.KROLLF_NMRAD + dyn.KROLLR_NMRAD) * 100.0:.1f}% front)."
    )
    lltd_x_mode = _select_per_lap_axis("dyn_setup_lltd_axis", default="laps")
    try:
        fig, kpis = dyn.lltd_mid_corner_per_lap_fig(dfs, x_mode=lltd_x_mode)
        for w in kpis.get("warnings", []):
            st.warning(w)
        rows = [
            {
                "Run": Path(run_name).stem,
                "Laps": int(vals.get("laps", 0)),
                "Mean LLTD [%]": round(vals.get("lltd_mean_pct", np.nan), 3),
                "Min [%]": round(vals.get("lltd_min_pct", np.nan), 3),
                "Max [%]": round(vals.get("lltd_max_pct", np.nan), 3),
                "Span [pp]": round(vals.get("lltd_span_pct_points", np.nan), 4),
                "Mean samples": round(vals.get("samples_mean", np.nan), 0),
            }
            for run_name, vals in kpis.get("runs", {}).items()
        ]
        if rows:
            _show_summary_table(rows)
        _plotly_chart(fig, use_container_width=True, theme=None)
        table = kpis.get("table")
        if table is not None and not table.is_empty():
            with st.expander("Per-lap LLTD data"):
                st.dataframe(
                    table.with_columns(
                        [
                            pl.col("LapTime [s]").round(3),
                            pl.col("LLTD mid-corner avg [%]").round(4),
                            pl.col("LLTD mid-corner median [%]").round(4),
                            pl.col("LLTD mid-corner span [pp]").round(5),
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.error(f"LLTD mid-corner unavailable: {exc}")

    st.subheader("② Roll Split  ·  LLTD Balance vs Lateral g")
    st.caption(
        "Front LLTD per corner sample vs the roll-stiffness target "
        f"({dyn.KROLLF_NMRAD / (dyn.KROLLF_NMRAD + dyn.KROLLR_NMRAD) * 100.0:.1f}% front), "
        "by |ay| band."
    )
    if len(dfs) == 1:
        _run_name, df_single = next(iter(dfs.items()))
        try:
            fig, kpis = dyn.lateral_load_transfer_fig(df_single)
            for w in kpis.get("warnings", []):
                st.warning(w)
            geom = kpis.get("geom_ltd_front_mean", np.nan)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Median front LTD", f"{_fmt(kpis.get('ltd_front_median') * 100.0, '.1f')} %")
            c2.metric("Theoretical", f"{_fmt(kpis.get('ltd_theoretical') * 100.0, '.1f')} %")
            c3.metric("Deviation", f"{_fmt(kpis.get('deviation_pct'), '+.1f')} %")
            c4.metric("Samples", str(kpis.get("samples", 0)))
            c5.metric(
                "Geometry LTD",
                f"{_fmt(geom * 100.0, '.1f')} %" if np.isfinite(geom) else "—",
                help="Roll-spring geometry cross-check of front LTD (independent of Est_FZ).",
            )
            _plotly_chart(fig, use_container_width=True, theme=None)
        except Exception as exc:
            st.warning(f"LTD unavailable: {exc}")
    else:
        ltd_results: dict[str, tuple[go.Figure, dict]] = {}
        for run_name, df_single in dfs.items():
            try:
                fig, kpis = dyn.lateral_load_transfer_fig(df_single)
                ltd_results[run_name] = (fig, kpis)
            except Exception as exc:
                st.warning(f"LTD unavailable ({Path(run_name).stem}): {exc}")
        if ltd_results:
            for rn, (_f, kpis) in ltd_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(f"{Path(rn).stem}: {w}")
            _show_summary_table(
                [
                    {
                        "Run": Path(rn).stem,
                        "Median front LTD [%]": _fmt(
                            kpis.get("ltd_front_median", float("nan")) * 100.0, ".1f"
                        ),
                        "Theoretical [%]": _fmt(
                            kpis.get("ltd_theoretical", float("nan")) * 100.0, ".1f"
                        ),
                        "Deviation [%]": _fmt(kpis.get("deviation_pct"), "+.1f"),
                        "Samples": kpis.get("samples", 0),
                    }
                    for rn, (_f, kpis) in ltd_results.items()
                ]
            )
            fig_ltd = _overlay_setup_single_figure(ltd_results)
            if fig_ltd is not None:
                _plotly_chart(fig_ltd, use_container_width=True, theme=None)

    st.subheader("② Roll Split  ·  Roll Gradient vs Theory")
    st.caption(
        "Radius-filtered roll angle vs lateral g; front vs rear gradient is the "
        "roll-stiffness balance you tune with the ARBs (theory ≈ 0.52 deg/g)."
    )
    roll_results: dict[str, tuple[go.Figure, dict]] = {}
    for run_name, df in dfs.items():
        try:
            roll_results[run_name] = dyn.roll_gradient_fig(df)
        except Exception as exc:
            st.error(f"Roll gradient unavailable ({Path(run_name).stem}): {exc}")
    if roll_results:
        if not multi:
            fig, kpis = next(iter(roll_results.values()))
            for w in kpis.get("warnings", []):
                st.warning(w)
            if kpis.get("calibrated"):
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Theory", f"{_fmt(kpis.get('theoretical_deg_per_g'), '.2f')} deg/g")
                c2.metric(
                    "Front grad", f"{_fmt(kpis.get('front_gradient_deg_per_g'), '+.2f')} deg/g"
                )
                c3.metric("Front dev", f"{_fmt(kpis.get('front_deviation_pct'), '+.1f')} %")
                c4.metric("Rear grad", f"{_fmt(kpis.get('rear_gradient_deg_per_g'), '+.2f')} deg/g")
                c5.metric("Rear dev", f"{_fmt(kpis.get('rear_deviation_pct'), '+.1f')} %")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Theory", f"{_fmt(kpis.get('theoretical_deg_per_g'), '.2f')} deg/g")
                c2.metric(
                    "Front grad", f"{_fmt(kpis.get('front_gradient_deg_per_g'), '+.2f')} deg/g"
                )
                c3.metric("Rear grad", f"{_fmt(kpis.get('rear_gradient_deg_per_g'), '+.2f')} deg/g")
                st.caption(
                    "Uncalibrated - roll is derived from raw damper counts, so only the "
                    "shape is meaningful and the rear gradient collapses; deviation-vs-theory "
                    "is hidden until calibration constants are set."
                )
            _plotly_chart(fig, use_container_width=True, theme=None)
        else:
            for rn, (_fig, kpis) in roll_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(f"{Path(rn).stem}: {w}")
            _show_summary_table(
                [
                    {
                        "Run": Path(rn).stem,
                        "Theory [deg/g]": _fmt(kpis.get("theoretical_deg_per_g"), ".2f"),
                        "Front grad [deg/g]": _fmt(kpis.get("front_gradient_deg_per_g"), "+.2f"),
                        "Rear grad [deg/g]": _fmt(kpis.get("rear_gradient_deg_per_g"), "+.2f"),
                        "Front samples": kpis.get("front_samples", 0),
                        "Rear samples": kpis.get("rear_samples", 0),
                    }
                    for rn, (_fig, kpis) in roll_results.items()
                ]
            )
            fig_roll = _overlay_setup_single_figure(roll_results)
            if fig_roll is not None:
                _plotly_chart(fig_roll, use_container_width=True, theme=None)

    # 3 - RIDE HEIGHTS - knob: pushrod / ride height ------------------------------
    st.divider()
    st.subheader("③ Ride Heights  ·  Pitch Gradient (Braking vs Acceleration)")
    st.caption(
        "Suspension pitch vs longitudinal g, split braking/acceleration — read dive/squat "
        "magnitude and the brake-vs-accel asymmetry."
    )
    pitch_results: dict[str, tuple[go.Figure, dict]] = {}
    for run_name, df in dfs.items():
        try:
            pitch_results[run_name] = dyn.pitch_gradient_fig(df)
        except Exception as exc:
            st.error(f"Pitch gradient unavailable ({Path(run_name).stem}): {exc}")
    if pitch_results:
        if not multi:
            fig, kpis = next(iter(pitch_results.values()))
            for w in kpis.get("warnings", []):
                st.warning(w)
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Brake gradient", f"{_fmt(kpis.get('brake_gradient_deg_per_g'), '+.2f')} deg/g"
            )
            c2.metric(
                "Accel gradient", f"{_fmt(kpis.get('accel_gradient_deg_per_g'), '+.2f')} deg/g"
            )
            c3.metric("Calibrated", "yes" if kpis.get("calibrated") else "no")
            _plotly_chart(fig, use_container_width=True, theme=None)
        else:
            for rn, (_fig, kpis) in pitch_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(f"{Path(rn).stem}: {w}")
            _show_summary_table(
                [
                    {
                        "Run": Path(rn).stem,
                        "Brake grad [deg/g]": _fmt(kpis.get("brake_gradient_deg_per_g"), "+.2f"),
                        "Accel grad [deg/g]": _fmt(kpis.get("accel_gradient_deg_per_g"), "+.2f"),
                        "Brake samples": kpis.get("brake_samples", 0),
                        "Accel samples": kpis.get("accel_samples", 0),
                        "Calibrated": "yes" if kpis.get("calibrated") else "no",
                    }
                    for rn, (_fig, kpis) in pitch_results.items()
                ]
            )
            fig_pitch = _overlay_setup_single_figure(pitch_results)
            if fig_pitch is not None:
                _plotly_chart(fig_pitch, use_container_width=True, theme=None)

    # 4 - DAMPERS - knob: clicks --------------------------------------------------
    st.divider()
    st.subheader("④ Dampers  ·  Velocity Histograms by Phase")
    st.caption("Damper-**rod** velocity distribution per wheel, split LSB/LSR/HSB/HSR at ±25 mm/s.")
    phase_label = st.segmented_control(
        "Damper phase",
        options=["ALL", "BRAKE", "CORNER", "ACCEL", "STRAIGHT"],
        default="ALL",
        required=True,
        key="dyn_damper_phase",
        label_visibility="collapsed",
    )
    phase = str(phase_label).lower()
    damper_figs: dict[str, list[go.Figure]] = {}
    damper_kpis: dict[str, dict] = {}
    for run_name, df in dfs.items():
        try:
            figs, kpis = dyn.damper_histogram_figs(df, phase=phase)
            damper_figs[run_name] = figs
            damper_kpis[run_name] = kpis
        except Exception as exc:
            st.error(f"Damper histograms unavailable ({Path(run_name).stem}): {exc}")
    if damper_figs:
        if not multi:
            figs = next(iter(damper_figs.values()))
            kpis = next(iter(damper_kpis.values()))
            for w in kpis.get("warnings", []):
                st.warning(w)
            front_bump = kpis.get("bump_share_by_axle", {}).get("front", float("nan"))
            rear_bump = kpis.get("bump_share_by_axle", {}).get("rear", float("nan"))
            sample_vals = list(kpis.get("samples", {}).values())
            c1, c2, c3 = st.columns(3)
            c1.metric("Front bump share", f"{_fmt(front_bump * 100.0, '.1f')} %")
            c2.metric("Rear bump share", f"{_fmt(rear_bump * 100.0, '.1f')} %")
            c3.metric("Samples/wheel", str(min(sample_vals) if sample_vals else 0))
            for fig in figs:
                _plotly_chart(fig, use_container_width=True, theme=None)
        else:
            for rn, kpis in damper_kpis.items():
                for w in kpis.get("warnings", []):
                    st.warning(f"{Path(rn).stem}: {w}")
            _show_summary_table(
                [
                    {
                        "Run": Path(rn).stem,
                        "Front bump share [%]": _fmt(
                            kpis.get("bump_share_by_axle", {}).get("front", np.nan) * 100.0,
                            ".1f",
                        ),
                        "Rear bump share [%]": _fmt(
                            kpis.get("bump_share_by_axle", {}).get("rear", np.nan) * 100.0,
                            ".1f",
                        ),
                        "Samples/wheel": min(kpis.get("samples", {}).values())
                        if kpis.get("samples")
                        else 0,
                    }
                    for rn, kpis in damper_kpis.items()
                ]
            )
            for fig in _overlay_figures(damper_figs):
                _plotly_chart(fig, use_container_width=True, theme=None)

    if multi:
        st.divider()


def _render_manual_gate_editor(
    ui_rev: str,
) -> tuple[tuple[tuple[float, float], tuple[float, float]] | None, bool]:
    """Render compact finish-line actions and return the current manual gate."""
    preview_line = st.session_state.get("_dyn_manual_gate_line")
    open_fullscreen = False

    cols = st.columns([1, 1.15, 1.2, 1.1])
    if cols[0].button(
        "Reset",
        key=f"dyn_gate_reset_{ui_rev}",
        use_container_width=True,
    ):
        for key in (
            "_dyn_manual_gate_line",
            "_dyn_manual_gate_event_id",
            "_dyn_manual_gate_consumed_event_id",
            "_dyn_manual_gate_result",
            "_dyn_track_last_event_id",
        ):
            st.session_state.pop(key, None)
        st.rerun()
    apply_disabled = preview_line is None
    if cols[1].button(
        "Apply To All CSVs",
        key=f"dyn_gate_apply_{ui_rev}",
        use_container_width=True,
        disabled=apply_disabled,
    ):
        assert preview_line is not None
        updated: list[str] = []
        with st.spinner("Recomputing laps from manual finish line..."):
            for path in _telemetry_csv_paths(DATA_DIR):
                n = lapcount.detect_and_write_laps(path, gate_line_lonlat=preview_line)
                updated.append(f"{path.name}: {n} laps")
        _clear_data_caches()
        _mark_manual_gate_consumed()
        st.session_state["_dyn_manual_gate_result"] = updated
        st.rerun()

    if cols[2].button(
        "Restore Auto",
        key=f"dyn_gate_restore_auto_{ui_rev}",
        use_container_width=True,
    ):
        st.session_state.pop("_dyn_manual_gate_line", None)
        st.session_state.pop("_dyn_manual_gate_event_id", None)
        st.session_state.pop("_dyn_manual_gate_consumed_event_id", None)
        updated: list[str] = []
        with st.spinner("Restoring automatic lap detection..."):
            for path in _telemetry_csv_paths(DATA_DIR):
                n = lapcount.detect_and_write_laps(path)
                updated.append(f"{path.name}: {n} laps")
        _clear_data_caches()
        st.session_state["_dyn_manual_gate_result"] = updated
        st.rerun()
    if cols[3].button(
        "Full Screen",
        key=f"dyn_gate_fullscreen_{ui_rev}",
        use_container_width=True,
    ):
        st.session_state["_dyn_track_open_fullscreen"] = True
        open_fullscreen = True
    result_lines = st.session_state.get("_dyn_manual_gate_result")
    if result_lines:
        st.caption(" | ".join(result_lines))
    return preview_line, open_fullscreen


@st.dialog("Circuit — Gate Lines", width="large")
def _render_circuit_gate_fullscreen_dialog(circuit_fig: go.Figure) -> None:
    """Full-screen interactive circuit map for drawing lapcount gate lines."""
    st.caption(
        "Draw a line to set a lapcount gate; assign it from the Event mode panel in the sidebar."
    )
    event = tmc.render_track_map_component(
        tmc.serialize_figure(circuit_fig),
        height_px=760,
        key="drv_circuit_gate_fullscreen",
    )
    _consume_track_component_event(
        event,
        pool_len=0,
        event_state_key="drv_circuit_gate_manual_fullscreen",
    )


def _render_circuit_gate_map(
    dfs: dict[str, pl.DataFrame],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> None:
    """Interactive Overview circuit map — the place to draw lapcount gate lines.

    Renders the fastest-lap speed map as a draw-enabled track component plus the
    manual-gate action buttons, so start/finish/centre lines are drawn here (and
    consumed by the sidebar Event mode panel) instead of in Lap Analysis.
    """
    manual_gate_line, open_fullscreen = _render_manual_gate_editor("driver_overview_circuit")
    if manual_gate_line is not None:
        st.caption(
            "Manual line: "
            f"({manual_gate_line[0][0]:.6f}, {manual_gate_line[0][1]:.6f}) -> "
            f"({manual_gate_line[1][0]:.6f}, {manual_gate_line[1][1]:.6f})"
        )
    base_fig, _mk = _driver_fastest_lap_speed_map_fig_cached(dfs, driver_run_tokens)
    # Copy the cached figure before overlaying gates so reruns don't accumulate
    # shapes on the shared cache_resource object.
    fig = go.Figure(base_fig)
    lap_gates = _lap_gates_from_run_tokens(driver_run_tokens)
    fig = _add_lap_detection_gates_to_fig(fig, lap_gates)
    fig = _add_manual_gate_line_to_fig(fig, manual_gate_line)
    event = tmc.render_track_map_component(
        tmc.serialize_figure(fig),
        height_px=430,
        key="drv_circuit_gate_map",
    )
    _consume_track_component_event(
        event,
        pool_len=0,
        event_state_key="drv_circuit_gate_manual",
    )
    if (
        open_fullscreen
        or bool(st.session_state.get("_dyn_track_open_fullscreen", False))
        or bool(event.get("fullscreen_event", False))
    ):
        st.session_state["_dyn_track_open_fullscreen"] = False
        _render_circuit_gate_fullscreen_dialog(fig)


def _render_dynamics_cornering(dfs: dict[str, pl.DataFrame]) -> None:
    """Cornering vehicle behavior: lateral capability, which axle limits, and
    balance from three independent physics (slip angle, forces, steering)."""
    single = len(dfs) == 1
    tokens = _run_cache_tokens(dfs)

    # 1 · Lateral grip envelope (capability) ---------------------------------
    st.subheader("Lateral Grip Envelope  ·  |ay| vs Speed (L/R)")
    st.caption(
        "P95 lateral g by speed bin vs the aero-scaled grip-μ reference, split left vs right turns."
    )
    try:
        fig, kpis = _dyn_lateral_grip_envelope_fig_cached(dfs, tokens)
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if single and runs:
            v = next(iter(runs.values()))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Peak ay (L)",
                f"{_fmt(v['peak_ay_left_g'], '.2f')} g",
                help=METRIC_HELP["Peak ay (L)"],
            )
            c2.metric(
                "Peak ay (R)",
                f"{_fmt(v['peak_ay_right_g'], '.2f')} g",
                help=METRIC_HELP["Peak ay (R)"],
            )
            c3.metric(
                "L/R asymmetry",
                f"{_fmt(v['ay_asymmetry_g'], '+.2f')} g",
                help=METRIC_HELP["L/R asymmetry"],
            )
            c4.metric(
                "Samples (L/R)",
                f"{int(v['samples_left'])}/{int(v['samples_right'])}",
                help=METRIC_HELP["Samples"],
            )
        elif runs:
            _show_summary_table(
                [
                    {
                        "Run": r,
                        "Peak ay L [g]": round(v["peak_ay_left_g"], 2),
                        "Peak ay R [g]": round(v["peak_ay_right_g"], 2),
                        "L/R asymmetry": round(v["ay_asymmetry_g"], 2),
                        "Samples L": int(v["samples_left"]),
                        "Samples R": int(v["samples_right"]),
                    }
                    for r, v in runs.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Lateral grip envelope unavailable: {exc}")

    st.divider()

    # 2 · Per-axle lateral utilisation (which axle limits) -------------------
    st.subheader("Per-axle Lateral Utilisation  ·  |Fy| / (μ·Fz)")
    st.caption(
        "Front vs rear lateral-grip utilisation |Fy|/(μ·Fz) — the axle reaching 1.0 first "
        "limits (front = understeer)."
    )
    try:
        fig, kpis = _dyn_axle_lateral_utilisation_fig_cached(dfs, tokens)
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if single and runs:
            v = next(iter(runs.values()))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Lat util front",
                f"{_fmt(v['util_front_median'], '.2f')}",
                help=METRIC_HELP["Lat util front"],
            )
            c2.metric(
                "Lat util rear",
                f"{_fmt(v['util_rear_median'], '.2f')}",
                help=METRIC_HELP["Lat util rear"],
            )
            c3.metric(
                "Limiting axle (lat)",
                str(v["limiting_axle"]).title(),
                help=METRIC_HELP["Limiting axle (lat)"],
            )
            c4.metric("Samples", f"{int(v['samples'])}", help=METRIC_HELP["Samples"])
        elif runs:
            _show_summary_table(
                [
                    {
                        "Run": r,
                        "Lat util front [-]": round(v["util_front_median"], 3),
                        "Lat util rear [-]": round(v["util_rear_median"], 3),
                        "Limiting axle (lat)": str(v["limiting_axle"]).title(),
                        "Samples": int(v["samples"]),
                    }
                    for r, v in runs.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Lateral axle utilisation unavailable: {exc}")

    st.divider()

    # 3 · Balance vs corner phase (slip-angle) -------------------------------
    st.subheader("Balance vs Corner Phase  ·  Understeer / Oversteer")
    st.caption(
        "Front−rear slip angle (Est_SA, deg) over entry → steady → exit — positive = "
        "understeer, negative = oversteer."
    )
    try:
        fig, kpis = _dyn_cornering_balance_phase_fig_cached(dfs, tokens)
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if single and runs:
            v = next(iter(runs.values()))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Entry balance",
                f"{_fmt(v['balance_entry_deg'], '+.2f')}°",
                help=METRIC_HELP["Entry balance"],
            )
            c2.metric(
                "Steady balance",
                f"{_fmt(v['balance_steady_deg'], '+.2f')}°",
                help=METRIC_HELP["Steady balance"],
            )
            c3.metric(
                "Exit balance",
                f"{_fmt(v['balance_exit_deg'], '+.2f')}°",
                help=METRIC_HELP["Exit balance"],
            )
            c4.metric(
                "Entry→exit shift",
                f"{_fmt(v['balance_shift_deg'], '+.2f')}°",
                help=METRIC_HELP["Entry→exit shift"],
            )
        elif runs:
            _show_summary_table(
                [
                    {
                        "Run": r,
                        "Entry [deg]": round(v["balance_entry_deg"], 2),
                        "Steady [deg]": round(v["balance_steady_deg"], 2),
                        "Exit [deg]": round(v["balance_exit_deg"], 2),
                        "Shift [deg]": round(v["balance_shift_deg"], 2),
                        "Corners": int(v["corners"]),
                        "Samples": int(v["samples"]),
                    }
                    for r, v in runs.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Balance vs corner phase unavailable: {exc}")

    st.divider()

    # 4 · Understeer Angle (per-lap, steering) -------------------------------
    st.subheader("Understeer Angle")
    st.caption(
        "Mean per-lap understeer angle: `Steering` (potentiometer, rad) vs the Ackermann "
        "ideal δ = L·ay/vx², over R<60 m corners."
    )
    understeer_x_mode = _select_per_lap_axis("dyn_understeer_axis", default="laps")
    try:
        if len(dfs) == 1:
            _run_name, df = next(iter(dfs.items()))
            kpis = dyn.understeer_angle_kpis(df)
            for w in kpis.get("warnings", []):
                st.warning(w)
            if not kpis.get("warnings"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Valid laps", str(kpis["valid_laps"]))
                c2.metric("Mean understeer", f"{_fmt(kpis['mean_understeer'], '.2f')} deg")
                c3.metric(
                    "Min / Max",
                    f"{_fmt(kpis['min_understeer'], '.2f')} / {_fmt(kpis['max_understeer'], '.2f')} deg",
                )
                c4.metric(
                    "Fastest valid lap",
                    f"L{kpis['fastest_lap']} - {_fmt(kpis['fastest_lt'], '.2f')} s",
                )
            fig = dyn.understeer_angle_fig(df, x_mode=understeer_x_mode)
            _plotly_chart(fig, use_container_width=True, theme=None)
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
            _show_summary_table(
                [
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
                ]
            )
            _plotly_chart(
                _overlay_figures(
                    {run_name: figs for run_name, (figs, _kpis) in run_results.items()}
                )[0],
                use_container_width=True,
                theme=None,
            )
            with st.expander("Per-lap data"):
                st.dataframe(
                    _concat_run_tables(
                        {
                            run_name: kpis["table"]
                            for run_name, (_figs, kpis) in run_results.items()
                            if not kpis.get("warnings")
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.warning(f"Understeer angle unavailable: {exc}")

    st.divider()

    # 5 · Steering vs ay · US/OS curve (steering gradient) -------------------
    st.subheader("Steering vs Lateral Acceleration  ·  US/OS Curve")
    st.caption(
        "Steady-state `Steering` (potentiometer, rad) vs the Ackermann ideal δ = L·ay/vx²; "
        "the low-|ay| slope is the understeer gradient."
    )
    if len(dfs) == 1:
        _run_name, df_single = next(iter(dfs.items()))
        try:
            fig, kpis = dyn.steering_vs_ay_fig(df_single)
            for w in kpis.get("warnings", []):
                st.warning(w)
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "US gradient", f"{_fmt(kpis.get('understeer_gradient_deg_per_g'), '+.2f')} deg/g"
            )
            c2.metric("Median vx", f"{_fmt(kpis.get('vx_median_mps'), '.1f')} m/s")
            c3.metric("Samples", str(kpis.get("samples", 0)))
            _plotly_chart(fig, use_container_width=True, theme=None)
        except Exception as exc:
            st.warning(f"Steering vs ay unavailable: {exc}")
    else:
        steer_results: dict[str, tuple[go.Figure, dict]] = {}
        for run_name, df_single in dfs.items():
            try:
                fig, kpis = dyn.steering_vs_ay_fig(df_single)
                steer_results[run_name] = (fig, kpis)
            except Exception as exc:
                st.warning(f"Steering vs ay unavailable ({Path(run_name).stem}): {exc}")
        if steer_results:
            for rn, (_f, kpis) in steer_results.items():
                for w in kpis.get("warnings", []):
                    st.warning(f"{Path(rn).stem}: {w}")
            _show_summary_table(
                [
                    {
                        "Run": Path(rn).stem,
                        "US gradient [deg/g]": _fmt(
                            kpis.get("understeer_gradient_deg_per_g"), "+.2f"
                        ),
                        "Median vx [m/s]": _fmt(kpis.get("vx_median_mps"), ".1f"),
                        "Samples": kpis.get("samples", 0),
                    }
                    for rn, (_f, kpis) in steer_results.items()
                ]
            )
            merged = _overlay_figures({Path(rn).stem: [f] for rn, (f, _k) in steer_results.items()})
            _plotly_chart(merged[0], use_container_width=True, theme=None)


def _render_dynamics_grip_factors(dfs: dict[str, pl.DataFrame]) -> None:
    """Grip overview (FS-adapted Buurman grip factors).

    Landing view for Dynamics: how much combined grip the car develops (g-g
    envelope + where on track), how it splits by phase, whether it holds across
    the stint, and how hard the driver works it. Phase gating reuses the
    dashboard's shared detectors (radius corner / brake / corner-exit). Aero grip
    is omitted: FS speeds are too low to isolate downforce cleanly.
    """
    single = len(dfs) == 1
    tokens = _run_cache_tokens(dfs)

    util_by_run = {rn: gf.grip_utilization_kpis(df) for rn, df in dfs.items()}

    # Per-run grip-factor tables — shared by every block below.
    run_results: dict[str, dict] = {}
    for run_name, df in dfs.items():
        try:
            run_results[run_name] = gf.grip_factor_kpis(df)
        except Exception as exc:
            st.warning(f"{Path(run_name).stem}: {exc}")
    for rn, k in run_results.items():
        for w in k.get("warnings", []):
            st.caption(f"{Path(rn).stem}: {w}")
    valid_results = {
        rn: k
        for rn, k in run_results.items()
        if not k.get("warnings") and not k["table"].is_empty()
    }

    # ── Capability overview: map · g-g, side by side ─────────────────────────
    st.subheader("Grip Overview")
    st.caption(
        "**Where** the car develops grip (combined-|G| map) and its "
        "combined-acceleration **envelope** (g-g)."
    )
    try:
        gg_fig, gg_kpis = _gf_gg_scatter_fig_cached(dfs, tokens)
    except Exception as exc:
        gg_fig, gg_kpis = None, {"runs": {}}
        st.error(f"g-g diagram unavailable: {exc}")
    for w in gg_kpis.get("warnings", []):
        st.warning(w)

    # ── KPI strip (envelope / utilisation + grip-factor means), above the figures ──
    if single:
        rn, k = next(iter(util_by_run.items()))
        gk = gg_kpis.get("runs", {}).get(rn, {})
        if not k.get("warnings"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Grip envelope", f"{k['envelope_g']:.2f} G", help=METRIC_HELP["Envelope [G]"])
            c2.metric(
                "Peak combined",
                f"{_fmt(gk.get('peak_combined_g'), '.2f')} G",
                help="P99.5 of combined |G|.",
            )
            c3.metric(
                "Utilisation", f"{k['utilization_pct']:.1f} %", help=METRIC_HELP["Utilisation [%]"]
            )
            c4.metric(
                "Time at limit",
                f"{k['time_at_limit_pct']:.1f} %",
                help=METRIC_HELP["Time at limit [%]"],
            )
        gfk = valid_results.get(rn)
        if gfk is not None:
            m = gfk["means"]
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Overall", f"{_fmt(m['Overall'], '.2f')} G", help=METRIC_HELP["Overall [G]"])
            d2.metric(
                "Cornering", f"{_fmt(m['Cornering'], '.2f')} G", help=METRIC_HELP["Cornering [G]"]
            )
            d3.metric("Braking", f"{_fmt(m['Braking'], '.2f')} G", help=METRIC_HELP["Braking [G]"])
            d4.metric(
                "Traction", f"{_fmt(m['Traction'], '.2f')} G", help=METRIC_HELP["Traction [G]"]
            )
            d5.metric(
                "Valid laps",
                str(gfk["valid_laps"]),
                help=(
                    f"Fastest L{gfk['fastest_lap']} · {_fmt(gfk['fastest_lt'], '.2f')} s"
                    if gfk["fastest_lap"] is not None
                    else None
                ),
            )
    elif valid_results:
        # One row per run: identity → grip per phase → envelope/utilisation → fastest.
        rows = []
        for rn, k in valid_results.items():
            u = util_by_run.get(rn, {})
            has_util = bool(u) and not u.get("warnings")
            rows.append(
                {
                    "Run": Path(rn).stem,
                    "Valid laps": k["valid_laps"],
                    "Overall [G]": round(k["means"]["Overall"], 3),
                    "Cornering [G]": round(k["means"]["Cornering"], 3),
                    "Braking [G]": round(k["means"]["Braking"], 3),
                    "Traction [G]": round(k["means"]["Traction"], 3),
                    "Envelope [G]": round(u["envelope_g"], 2) if has_util else None,
                    "Peak [G]": gg_kpis.get("runs", {}).get(rn, {}).get("peak_combined_g"),
                    "Utilisation [%]": u["utilization_pct"] if has_util else None,
                    "Time at limit [%]": u["time_at_limit_pct"] if has_util else None,
                    "Fastest lap": k["fastest_lap"],
                    "Fastest lt [s]": (
                        round(k["fastest_lt"], 2) if np.isfinite(k["fastest_lt"]) else None
                    ),
                }
            )
        _show_summary_table(rows)

    # Two figures below the KPI strip. The track map averages combined |G| over
    # every valid lap; single run → no picker, several runs → pick which map to show.
    map_run = None
    if valid_results:
        if single:
            map_run = next(iter(valid_results))
        else:
            _, ctrl = st.columns([2, 1])
            map_run = ctrl.selectbox(
                "Track-map run",
                options=list(valid_results.keys()),
                format_func=lambda rn: Path(rn).stem,
                key="dyn_gf_map_run_multi",
            )

    overview_cols = st.columns(2)

    # 1 · Combined-G track map (where grip is developed, averaged over the stint)
    with overview_cols[0]:
        if valid_results and map_run is not None:
            map_laps = [int(l) for l in valid_results[map_run]["table"]["Lap"].to_list()]
            _plotly_chart(
                gf.combined_g_track_map_fig(dfs[map_run], map_laps),
                use_container_width=True,
                theme=None,
            )

    # 2 · g-g envelope (capability)
    with overview_cols[1]:
        if gg_fig is not None:
            _plotly_chart(
                gg_fig,
                use_container_width=True,
                theme=None,
                preserve_legend=True,
            )

    if not valid_results:
        return

    # ── 4 · Phase level + evolution (consistency over the stint) ──────────────
    st.divider()
    st.subheader("Grip Factor Evolution")
    st.caption("Mean grip per phase next to each lap's grip-factor evolution.")
    evo_col, phase_col = st.columns([2, 1])
    with evo_col:
        if single:
            gf_x_mode = _select_per_lap_axis("dyn_gf_axis", default="laps")
            rn, k = next(iter(valid_results.items()))
            _plotly_chart(
                gf.grip_factor_evolution_fig(k["table"], x_mode=gf_x_mode),
                use_container_width=True,
                theme=None,
            )
        else:
            axis_ctrl_col, factor_ctrl_col = st.columns([1, 2])
            with axis_ctrl_col:
                gf_x_mode = _select_per_lap_axis("dyn_gf_axis", default="laps")
            with factor_ctrl_col:
                evo_cat = st.segmented_control(
                    "Grip factor",
                    options=list(gf.GRIP_CATEGORIES),
                    default="Overall",
                    required=True,
                    key="dyn_gf_evo_cat",
                )
            _plotly_chart(
                gf.grip_factor_evolution_multi_fig(
                    {rn: k["table"] for rn, k in valid_results.items()},
                    x_mode=gf_x_mode,
                    category=evo_cat,
                ),
                use_container_width=True,
                theme=None,
            )
    with phase_col:
        if single:
            _rn, _k = next(iter(valid_results.items()))
            _plotly_chart(gf.grip_factor_bar_fig(_k["means"]), use_container_width=True, theme=None)
        else:
            _plotly_chart(
                gf.grip_factor_radar_fig({rn: k["table"] for rn, k in valid_results.items()}),
                use_container_width=True,
                theme=None,
            )

    with st.expander("Per-lap grip factors"):
        if single:
            rn, k = next(iter(valid_results.items()))
            st.dataframe(k["table"], use_container_width=True, hide_index=True)
        else:
            st.dataframe(
                _concat_run_tables({rn: k["table"] for rn, k in valid_results.items()}),
                use_container_width=True,
                hide_index=True,
            )

    # ── 5 · Grip utilisation / time-at-limit (driver usage) ───────────────────
    st.divider()
    st.subheader("Grip Utilisation  ·  Time at the Limit")
    st.caption(
        "Share of samples within "
        f"{int(gf.LIMIT_FRAC * 100)}% of each run's P95 combined-|G| envelope, by phase "
        "(lower = grip left on the table)."
    )
    valid_util = {rn: k for rn, k in util_by_run.items() if not k.get("warnings")}
    if valid_util:
        if single:
            _rn, k = next(iter(valid_util.items()))
            ph = k["phase_time_at_limit_pct"]
            u1, u2, u3 = st.columns(3)
            u1.metric(
                "Utilisation", f"{k['utilization_pct']:.1f}%", help=METRIC_HELP["Utilisation [%]"]
            )
            u2.metric(
                "Time at limit",
                f"{k['time_at_limit_pct']:.1f}%",
                help=METRIC_HELP["Time at limit [%]"],
            )
            u3.metric(
                "Brake / Corner / Traction TAL",
                f"{_fmt(ph.get('Braking'), '.0f')} / {_fmt(ph.get('Cornering'), '.0f')} / "
                f"{_fmt(ph.get('Traction'), '.0f')} %",
            )
        else:
            _show_summary_table(
                [
                    {
                        "Run": Path(rn).stem,
                        "Envelope [G]": round(k["envelope_g"], 2),
                        "Utilisation [%]": k["utilization_pct"],
                        "Time at limit [%]": k["time_at_limit_pct"],
                        "Braking TAL [%]": k["phase_time_at_limit_pct"].get("Braking"),
                        "Cornering TAL [%]": k["phase_time_at_limit_pct"].get("Cornering"),
                        "Traction TAL [%]": k["phase_time_at_limit_pct"].get("Traction"),
                    }
                    for rn, k in valid_util.items()
                ]
            )
        _plotly_chart(_gf_utilization_fig_cached(dfs, tokens), use_container_width=True, theme=None)


def _tab_tc(dfs: dict[str, pl.DataFrame]) -> None:
    st.subheader("TC — one figure per controller reference")
    st.caption(
        "Following the traction-control pipeline: the optimal slip-ratio setpoint, the "
        "velocity reference it computes, and the torque it cuts to hold it."
    )
    single = len(dfs) == 1

    def _render_kpis(kpis: dict, specs: list[tuple[str, str, str, str | None]]) -> None:
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if not runs:
            return
        if single:
            vals = next(iter(runs.values()))
            cols = st.columns(len(specs))
            for col, (label, key, fmt, helpkey) in zip(cols, specs):
                v = vals.get(key)
                text = (
                    v
                    if fmt in ("str", "int") and isinstance(v, str)
                    else ("—" if v is None else (str(int(v)) if fmt == "int" else _fmt(v, fmt)))
                )
                col.metric(label, text, help=METRIC_HELP[helpkey] if helpkey else None)
        else:
            rows = []
            for run_name, vals in runs.items():
                row: dict[str, object] = {"Run": Path(run_name).stem}
                for label, key, fmt, _h in specs:
                    v = vals.get(key)
                    if fmt == "str":
                        row[label] = v
                    elif fmt == "int":
                        row[label] = int(v) if v is not None else None
                    else:
                        row[label] = _fmt(v, fmt)
                rows.append(row)
            _show_summary_table(rows)

    # 1 — Optimal slip ratio (setpoint)
    st.markdown("##### 1 · Optimal slip ratio — where the tyre operates vs +0.20")
    st.caption(
        "Per-wheel slip reached while TC is armed, vs the +0.20 grip optimum — an "
        "operating-point read."
    )
    try:
        fig, kpis = tc.tc_optimal_slip_ratio_fig(dfs)
        _render_kpis(
            kpis,
            [
                ("Front p95 SR", "front_p95_sr", ".3f", "TC front p95 SR"),
                ("Rear p95 SR", "rear_p95_sr", ".3f", "TC rear p95 SR"),
                ("Worst wheel", "worst_wheel", "str", None),
                ("Time over +0.20 [%]", "pct_time_over_target", ".1f", "TC time over target [%]"),
                ("Armed samples", "armed_samples", "int", None),
            ],
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.warning(f"TC optimal slip ratio unavailable: {exc}")

    st.divider()
    # 2 — Reference velocity
    st.markdown("##### 2 · Reference velocity — does the wheel hold the reference?")
    st.caption(
        "Actual wheel speed against the TC velocity reference, with the y=x line where the "
        "wheel exactly holds it."
    )
    try:
        fig, kpis = tc.tc_reference_velocity_fig(dfs)
        _render_kpis(
            kpis,
            [
                ("Front escape [%]", "front_escape_pct", ".1f", "TC escape [%]"),
                ("Rear escape [%]", "rear_escape_pct", ".1f", "TC escape [%]"),
                ("Front p95 speed/ref", "front_p95_ratio", ".2f", "TC p95 speed-to-ref"),
                ("Rear p95 speed/ref", "rear_p95_ratio", ".2f", "TC p95 speed-to-ref"),
                ("Armed samples", "armed_samples", "int", None),
            ],
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.warning(f"TC reference velocity unavailable: {exc}")

    st.divider()
    # 3 — Reference torque (placeholder until torque-mode logs)
    st.markdown("##### 3 · Reference torque — how hard it cuts torque")
    st.caption(
        "Torque cut against the slip over target — the TC actuation effort, empty on "
        "velocity-mode (old-car) logs."
    )
    try:
        fig, kpis = tc.tc_reference_torque_fig(dfs)
        _render_kpis(
            kpis,
            [
                ("p95 cut [Nm]", "p95_cut_nm", ".1f", "TC p95 cut [Nm]"),
                ("Cut samples", "cut_samples", "int", None),
            ],
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.warning(f"TC reference torque unavailable: {exc}")


def _tab_tv(dfs: dict[str, pl.DataFrame]) -> None:
    st.subheader("TV — one figure per controller reference")
    st.caption(
        "Following the torque-vectoring cascade: the yaw-rate goal, the feedforward and PI "
        "halves of the yaw command, the yaw moment it delivers, and the longitudinal force it trades against."
    )
    single = len(dfs) == 1

    def _render_kpis(kpis: dict, specs: list[tuple[str, str, str, str | None]]) -> None:
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if not runs:
            return
        if single:
            vals = next(iter(runs.values()))
            cols = st.columns(len(specs))
            for col, (label, key, fmt, helpkey) in zip(cols, specs):
                v = vals.get(key)
                text = (
                    v
                    if fmt in ("str", "int") and isinstance(v, str)
                    else (str(int(v)) if fmt == "int" else _fmt(v, fmt))
                )
                col.metric(label, text, help=METRIC_HELP[helpkey] if helpkey else None)
        else:
            rows = []
            for run_name, vals in runs.items():
                row: dict[str, object] = {"Run": Path(run_name).stem}
                for label, key, fmt, _h in specs:
                    v = vals.get(key)
                    if fmt == "str":
                        row[label] = v
                    elif fmt == "int":
                        row[label] = int(v) if v is not None else None
                    else:
                        row[label] = _fmt(v, fmt)
                rows.append(row)
            _show_summary_table(rows)

    st.markdown("##### 1 · Yaw-rate reference — does the car reach the yaw rate TV asks?")
    st.caption(
        "Commanded vs measured yaw rate — on the diagonal, the car turns as much as TV asks."
    )
    try:
        fig, kpis = tv.tv_yaw_tracking_fig(dfs)
        _render_kpis(
            kpis,
            [
                ("Yaw tracking RMSE [rad/s]", "tracking_rmse", ".3f", "Yaw tracking RMSE [rad/s]"),
                ("Tracking slope", "tracking_slope", ".2f", "Tracking slope"),
                ("Tracking R²", "tracking_r2", ".2f", None),
            ],
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.warning(f"Yaw tracking unavailable: {exc}")

    st.divider()
    st.markdown(
        "##### 2 · Feedforward — how much of the yaw command is anticipation vs correction?"
    )
    st.caption(
        "How much of the yaw-moment command is carried by the feedforward map vs the PI feedback."
    )
    try:
        fig, kpis = tv.tv_feedforward_share_fig(dfs)
        _render_kpis(
            kpis,
            [
                ("FF share (median)", "median_ff_share", ".2f", "FF share (median)"),
                ("FB-led [%]", "fb_led_pct", ".0f", "FB-led [%]"),
                ("Corner samples", "corner_samples", "int", None),
            ],
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.warning(f"Feedforward share unavailable: {exc}")

    st.divider()
    st.markdown("##### 3 · PI controller — effective gain & ringing")
    st.caption("How hard the feedback loop pushes against yaw error, and whether it oscillates.")
    try:
        fig, kpis = tv.tv_pi_loop_health_fig(dfs)
        _render_kpis(
            kpis,
            [
                (
                    "Effective PI gain [Nm·s/rad]",
                    "effective_gain",
                    ".0f",
                    "Effective PI gain [Nm·s/rad]",
                ),
                ("Ringing rate [1/s]", "ringing_rate", ".2f", "Ringing rate [1/s]"),
                ("Gain R²", "gain_r2", ".2f", None),
            ],
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.warning(f"PI loop health unavailable: {exc}")

    st.divider()
    st.markdown("##### 4 · Yaw moment — authority used and moment delivered")
    st.caption(
        "How much of the yaw-moment limit the TV demands, and whether the wheels deliver the commanded moment."
    )
    try:
        fig, kpis = tv.tv_authority_utilisation_fig(dfs)
        _render_kpis(
            kpis,
            [
                ("Moment utilisation p95", "util_p95", ".2f", "Moment utilisation p95"),
                ("Mz delivery ratio", "delivery_ratio_p50", ".2f", "Mz delivery ratio"),
                ("Corner samples", "corner_samples", "int", None),
            ],
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.warning(f"Moment authority unavailable: {exc}")

    st.divider()
    st.markdown("##### 5 · Longitudinal force — does TV run out of drive/brake force?")
    st.caption("Share of the longitudinal-force envelope the TV demands over the lap.")
    try:
        fig, kpis = tv.tv_fx_envelope_fig(dfs)
        _render_kpis(
            kpis,
            [
                ("Fx envelope-use p95", "fx_use_p95", ".2f", "Fx envelope-use p95"),
                ("Time at Fx limit [%]", "time_at_limit_pct", ".0f", "Time at Fx limit [%]"),
                ("Moving samples", "moving_samples", "int", None),
            ],
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.warning(f"Longitudinal force unavailable: {exc}")


def _tab_driver(
    dfs: dict[str, pl.DataFrame],
    file_signatures: dict[str, FileSignature],
    video_server: va.VideoServerInfo,
    raw_dfs: dict[str, pl.DataFrame],
    csv_files: list[str],
) -> None:
    driver_run_tokens = _driver_run_tokens(dfs, file_signatures)

    drv_section = st.segmented_control(
        "Driver section",
        options=["Overview", "Lap Analysis", "Throttle", "Brake", "Steering", "Video Analysis"],
        default="Overview",
        required=True,
        key="driver_subsection",
        label_visibility="collapsed",
        width="stretch",
    )

    if drv_section == "Overview":
        _render_driver_overview_subtab(dfs, driver_run_tokens)
        return
    if drv_section == "Lap Analysis":
        _render_driver_lap_analysis_subtab(dfs, driver_run_tokens)
        return
    if drv_section == "Video Analysis":
        _render_driver_video_subtab(raw_dfs, file_signatures, video_server, csv_files)
        return

    summaries = _collect_driver_summaries(dfs, driver_run_tokens)
    if drv_section == "Throttle":
        _render_driver_throttle_subtab(dfs, summaries, driver_run_tokens)
    elif drv_section == "Brake":
        _render_driver_brake_subtab(dfs, summaries, driver_run_tokens)
    elif drv_section == "Steering":
        _render_driver_steering_subtab(dfs, summaries, driver_run_tokens)

    if summaries:
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


def _render_driver_throttle_subtab(
    dfs: dict[str, pl.DataFrame],
    summaries: dict[str, dict],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> None:
    st.subheader("Driver Throttle Summary")
    st.caption(
        "Per-run throttle averages — higher full-throttle and lower off-throttle usually "
        "mean more aggressive driving."
    )
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
        _render_summary_df(pl.DataFrame(valid_rows))

    st.divider()
    st.subheader("Throttle Position Histogram")
    st.caption(
        "Distribution of throttle position samples per lap (bimodal at 0/100% = on/off "
        "style; spread = modulated)."
    )
    try:
        fig = _driver_throttle_histogram_fig_cached(dfs, driver_run_tokens)
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Throttle histogram unavailable: {exc}")

    st.divider()
    st.subheader("Full Throttle Time per Lap")
    st.caption("Seconds per lap with throttle above the full-throttle threshold.")
    full_x_mode = _select_per_lap_axis("driver_full_axis", default="laps")
    try:
        fig = _driver_full_throttle_time_fig_cached(dfs, driver_run_tokens, full_x_mode)
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Full throttle time unavailable: {exc}")

    st.divider()
    st.subheader("Throttle Speed per Lap")
    st.caption(
        "How fast the driver opens or closes the throttle each lap — higher means snappier pedal work."
    )
    speed_x_mode = _select_per_lap_axis("driver_speed_axis", default="laps")
    try:
        fig = _driver_throttle_speed_fig_cached(dfs, driver_run_tokens, speed_x_mode)
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Throttle speed unavailable: {exc}")


def _driver_brake_zone_turns(
    dfs: dict[str, pl.DataFrame],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> tuple[tuple, str]:
    """Return (turns_signature, caption_label) for braking-zone box-plot figures.

    Picks the run with the fastest lap as the reference for turn detection and
    builds the same signature consumed by `_driver_steering_stability_fig_cached`.
    """
    brake_candidates: list[tuple[float, str, int]] = []
    for run_name, df in dfs.items():
        fastest_lap = _driver_fastest_lap_cached(dfs, driver_run_tokens, run_name)
        if fastest_lap is None:
            continue
        lap_time_s = _lap_laptimes(df).get(int(fastest_lap), np.nan)
        brake_candidates.append(
            (
                float(lap_time_s) if np.isfinite(lap_time_s) else np.inf,
                run_name,
                int(fastest_lap),
            )
        )
    if not brake_candidates:
        return (), ""
    _best_time_s, brake_ref_run, brake_ref_lap = min(
        brake_candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    balanced = _LAP_ANALYSIS_PRESETS["Balanced"]
    brake_R_thr_m = _curvature_thr_1pkm_to_R_thr_m(balanced["curvature_thr_1pkm"])
    brake_turns = _driver_cornering_turns_cached(
        dfs,
        driver_run_tokens,
        brake_R_thr_m,
        balanced["min_dur_s"],
        balanced["corner_merge_gap_m"],
        brake_ref_run,
        int(brake_ref_lap),
    )
    brake_turns_signature = _cornering_turns_signature(brake_turns)
    label = ""
    if brake_turns_signature:
        label = f"Turn labels from {Path(brake_ref_run).stem} L{int(brake_ref_lap)}."
    return brake_turns_signature, label


def _render_driver_brake_subtab(
    dfs: dict[str, pl.DataFrame],
    summaries: dict[str, dict],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> None:
    st.subheader("Driver Brake Summary")
    st.caption(
        "Per-run brake aggressiveness, release smoothness, trail-braking overlap and peak decel."
    )
    # Trail-braking overlap and per-lap max braking g are folded into this one
    # section table so the sub-tab keeps a single summary (their figures follow).
    trail_overlap: dict[str, float] = {}
    for run_name, df in dfs.items():
        k = drv.trail_braking_kpis(df)
        for w in k.get("warnings", []):
            st.caption(f"{Path(run_name).stem}: {w}")
        if not k.get("warnings"):
            trail_overlap[run_name] = k["trail_overlap_pct"]
    try:
        max_g_fig, max_g_kpis = _driver_max_braking_g_per_lap_fig_cached(dfs, driver_run_tokens)
    except Exception as exc:
        max_g_fig, max_g_kpis = None, {"warnings": [f"Max braking G unavailable: {exc}"]}
    max_g_runs = max_g_kpis.get("runs", {})

    brake_rows = [
        {
            "Run": run_name,
            "Mean aggressiveness [%/s]": round(s["mean_brake_aggr"], 1),
            "Peak lap aggressiveness [%/s]": round(s["peak_brake_aggr"], 1),
            "Mean release smoothness [%/s]": round(s["mean_brake_release"], 1),
            "Peak lap release smoothness [%/s]": round(s["peak_brake_release"], 1),
            "Trail-braking overlap [%]": (
                round(trail_overlap[run_name], 1) if run_name in trail_overlap else None
            ),
            "Mean max braking [g]": (
                round(max_g_runs[run_name].get("mean_max_braking_g", np.nan), 3)
                if run_name in max_g_runs
                else None
            ),
        }
        for run_name, s in summaries.items()
        if s.get("valid_laps", 0) > 0
    ]
    if brake_rows:
        _render_summary_df(pl.DataFrame(brake_rows))

    st.divider()
    st.subheader("Combined Braking & Steering")
    st.caption(
        "Share of each lap with the brake pressed (>5%) and the wheel turned (>5°) at once — brake carried into the corner."
    )
    combined_bs_x_mode = _select_per_lap_axis("driver_brake_steer_axis", default="laps")
    try:
        fig = _driver_combined_brake_steer_fig_cached(
            dfs,
            driver_run_tokens,
            combined_bs_x_mode,
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Combined braking & steering unavailable: {exc}")

    st.divider()
    st.subheader("Braking Aggressiveness per Lap")
    st.caption("How fast the driver presses the brake each lap — higher means a sharper hit.")
    brake_aggr_x_mode = _select_per_lap_axis("driver_brake_aggr_axis", default="laps")
    try:
        fig = _driver_braking_aggressiveness_fig_cached(
            dfs,
            driver_run_tokens,
            brake_aggr_x_mode,
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Braking aggressiveness unavailable: {exc}")

    st.divider()
    st.subheader("Brake Release Smoothness per Lap")
    st.caption(
        "How fast the driver lifts off the brake each lap — higher means a more abrupt release."
    )
    brake_release_x_mode = _select_per_lap_axis("driver_brake_release_axis", default="laps")
    try:
        fig = _driver_brake_release_smoothness_fig_cached(
            dfs,
            driver_run_tokens,
            brake_release_x_mode,
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Brake release smoothness unavailable: {exc}")

    st.divider()
    st.subheader("Max Braking G per Lap")
    st.caption(
        "The hardest sustained braking reached each lap, in g (more negative = harder decel)."
    )
    for w in max_g_kpis.get("warnings", []):
        st.warning(w)
    if max_g_fig is not None:
        _plotly_chart(max_g_fig, use_container_width=True, theme=None)


def _render_driver_steering_subtab(
    dfs: dict[str, pl.DataFrame],
    summaries: dict[str, dict],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> None:
    st.subheader("Driver Steering Summary")
    st.caption(
        "Per-run steering smoothness, integral and curvature (high integral = high steering "
        "demand)."
    )
    steering_rows = [
        {
            "Run": run_name,
            "Mean smoothness [deg]": round(s["mean_steering_smoothness"], 4),
            "Peak lap smoothness [deg]": round(s["max_steering_smoothness"], 4),
            "Mean integral [deg*m]": round(s["mean_steering_integral"], 1),
            "Peak lap integral [deg*m]": round(s["max_steering_integral"], 1),
            "Mean curvature [1/m]": round(s["mean_curvature"], 5),
            "Peak lap curvature [1/m]": round(s["max_curvature"], 5),
        }
        for run_name, s in summaries.items()
        if s.get("valid_laps", 0) > 0
    ]
    if steering_rows:
        _render_summary_df(pl.DataFrame(steering_rows))

    st.divider()
    st.subheader("Steering Integral")
    st.caption(
        "Total steering used per lap, weighted by distance — higher means more steering demand."
    )
    steering_integral_x_mode = _select_per_lap_axis(
        "driver_steering_integral_axis",
        default="laps",
    )
    try:
        fig = _driver_steering_integral_fig_cached(
            dfs,
            driver_run_tokens,
            steering_integral_x_mode,
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Steering integral unavailable: {exc}")

    st.divider()
    st.subheader("Steering Smoothness")
    st.caption(
        "Mean deviation of steering from a 1.0 s smoothed trace — higher = more "
        "high-frequency corrections."
    )
    steering_x_mode = _select_per_lap_axis("driver_steering_axis", default="laps")
    try:
        fig = _driver_steering_smoothness_fig_cached(
            dfs,
            driver_run_tokens,
            steering_x_mode,
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Steering smoothness unavailable: {exc}")

    st.divider()
    st.subheader("Steering Stability under Braking")
    st.caption(
        "How much the driver saws at the wheel while braking in a straight line — higher means less stable."
    )
    try:
        steer_turns_signature, steer_turn_label = _driver_brake_zone_turns(
            dfs,
            driver_run_tokens,
        )
        if steer_turn_label:
            st.caption(steer_turn_label)
        fig = _driver_steering_stability_fig_cached(
            dfs,
            driver_run_tokens,
            steer_turns_signature,
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Steering stability unavailable: {exc}")

    st.divider()
    st.subheader("Corner Curvature per Lap")
    st.caption(
        "How tight a line the driver takes each lap, from the GPS path — higher means a tighter line."
    )
    curvature_x_mode = _select_per_lap_axis("driver_curvature_axis", default="laps")
    try:
        fig = _driver_corner_curvature_fig_cached(
            dfs,
            driver_run_tokens,
            curvature_x_mode,
        )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Corner curvature unavailable: {exc}")


def _cornering_turns_signature(turns: list[corn.TurnDef]) -> tuple:
    """Hashable turn definition for Streamlit cache invalidation."""
    return tuple(
        (
            int(t.turn_id),
            round(float(t.s_entry_m), 3),
            round(float(t.s_apex_m), 3),
            round(float(t.s_exit_m), 3),
            round(float(t.apex_lat), 8),
            round(float(t.apex_lng), 8),
        )
        for t in turns
    )


def _driver_lap_compare_options(
    dfs: dict[str, pl.DataFrame],
) -> list[tuple[str, int, float]]:
    options: list[tuple[str, int, float]] = []
    for run_name, df in dfs.items():
        lap_times = _lap_laptimes(df)
        for lap_id in available_laps(df).tolist():
            options.append((run_name, int(lap_id), lap_times.get(int(lap_id), np.nan)))
    return sorted(
        options,
        key=lambda item: (
            np.inf if not np.isfinite(item[2]) else item[2],
            item[0],
            item[1],
        ),
    )


def _format_driver_lap_compare_option(option: tuple[str, int, float]) -> str:
    run_name, lap_id, lap_time_s = option
    if run_name == POTENTIAL_LAP_RUN:
        base = "Potential lap"
        return f"{base} ({lap_time_s:.2f} s)" if np.isfinite(lap_time_s) else base
    base = f"{Path(run_name).stem} - L{int(lap_id)}"
    return f"{base} ({lap_time_s:.2f} s)" if np.isfinite(lap_time_s) else base


def _driver_default_compare_indices(
    lap_options: list[tuple[str, int, float]],
) -> tuple[int, int]:
    """Return default (reference, compared) indices for lap A/B selectors."""
    ref_default = 0
    cmp_default = 1 if len(lap_options) > 1 else 0
    for idx, option in enumerate(lap_options):
        if option[0] != lap_options[ref_default][0]:
            cmp_default = idx
            break
    return ref_default, cmp_default


def _format_cornering_turn_option(
    turn_id: int,
    labels: dict[int, str],
) -> str:
    return labels.get(int(turn_id), f"Turn {int(turn_id)}")


_LAP_ANALYSIS_PRESETS: dict[str, dict[str, float]] = {
    "Balanced": {
        "curvature_thr_1pkm": 16.7,
        "min_dur_s": 0.5,
        "corner_merge_gap_m": 4.0,
    },
    "Tighter corners": {
        "curvature_thr_1pkm": 22.0,
        "min_dur_s": 0.4,
        "corner_merge_gap_m": 2.0,
    },
    "Open corners": {
        "curvature_thr_1pkm": 12.5,
        "min_dur_s": 0.7,
        "corner_merge_gap_m": 6.0,
    },
}

_LAP_ANALYSIS_ZONE_DEFAULTS: dict[str, float] = {
    "brake_threshold_pct": 5.0,
    "throttle_threshold_pct": 40.0,
    "min_zone_m": 10.0,
    "zone_merge_gap_m": 6.0,
}


def _curvature_thr_1pkm_to_R_thr_m(curvature_thr_1pkm: float) -> float:
    """Convert user-facing curvature threshold [1/km] to radius [m]."""
    return 1000.0 / max(float(curvature_thr_1pkm), 1.0e-6)


def _lap_analysis_settings_ui() -> dict[str, float]:
    """Compact detection controls for Lap Analysis."""
    preset = st.selectbox(
        "Detection",
        options=[*_LAP_ANALYSIS_PRESETS.keys(), "Manual"],
        index=0,
        key="drv_lap_detection_preset",
    )
    values = {**_LAP_ANALYSIS_ZONE_DEFAULTS, **_LAP_ANALYSIS_PRESETS["Balanced"]}
    if preset in _LAP_ANALYSIS_PRESETS:
        values.update(_LAP_ANALYSIS_PRESETS[preset])
        st.caption(
            f"{preset}: r≥{values['curvature_thr_1pkm']:.1f} 1/km, "
            f"corner ≥{values['min_dur_s']:.1f} s, "
            f"merge ≤{values['corner_merge_gap_m']:.0f} m"
        )
        return values

    with st.expander("Manual corner detection", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            values["curvature_thr_1pkm"] = st.slider(
                "Curvature threshold r [1/km]",
                8.0,
                35.0,
                values["curvature_thr_1pkm"],
                0.5,
                key="drv_lap_curvature_thr",
            )
        with c2:
            values["min_dur_s"] = st.slider(
                "Min corner duration [s]",
                0.2,
                2.0,
                values["min_dur_s"],
                0.1,
                key="drv_lap_corner_min_dur",
            )
        with c3:
            values["corner_merge_gap_m"] = st.slider(
                "Corner merge gap [m]",
                0.0,
                30.0,
                values["corner_merge_gap_m"],
                1.0,
                key="drv_lap_corner_merge_gap",
            )
    return values


def _render_driver_overview_subtab(
    dfs: dict[str, pl.DataFrame],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> None:
    """Synthesis landing view: circuit + driving phases, pace, sector times."""
    # ── 1. Circuit map + driving phases ───────────────────────────────────────
    top = st.columns(2)
    with top[0]:
        st.subheader("Circuit")
        st.caption(
            "Overall-fastest valid lap, coloured by speed [km/h]. Draw a line here to set a "
            "lapcount gate (start/finish/centre), then assign it from the sidebar Event mode panel."
        )
        try:
            _render_circuit_gate_map(dfs, driver_run_tokens)
        except Exception as exc:
            st.error(f"Circuit map unavailable: {exc}")
    with top[1]:
        st.subheader("Driving Phases")
        st.caption(
            "Share of lap time the driver spent accelerating, braking, coasting, or "
            "in a plausibility state (both pedals, throttle-dominant)."
        )
        try:
            fig, _pk = _driver_run_phase_distribution_fig_cached(dfs, driver_run_tokens)
            _plotly_chart(fig, use_container_width=True, theme=None)
        except Exception as exc:
            st.error(f"Driving phases unavailable: {exc}")

    st.divider()

    # ── 2. Track speed distribution ───────────────────────────────────────────
    st.subheader("Track Speed Distribution")
    st.caption("Share of lap time at each speed over all valid laps — a fingerprint of the track.")
    try:
        fig, _sk = _driver_speed_distribution_fig_cached(dfs, driver_run_tokens)
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Speed distribution unavailable: {exc}")

    st.divider()

    # ── 3. Pace ───────────────────────────────────────────────────────────────
    st.subheader("Pace")
    st.caption(
        "Lap-time progression through the stint and its distribution — pace and consistency."
    )
    try:
        fig = _driver_lap_time_progression_fig_cached(dfs, driver_run_tokens)
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Lap time progression unavailable: {exc}")
    try:
        fig = _driver_lap_time_distribution_fig_cached(dfs, driver_run_tokens)
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"Lap time distribution unavailable: {exc}")

    st.divider()

    # ── 4. Lap & sector times ─────────────────────────────────────────────────
    st.subheader("Lap & Sector Times")
    st.caption("Seconds per lap in each track segment (T = corner, S = straight).")
    try:
        geo_candidates: list[tuple[float, str, int]] = []
        for run_name, df in dfs.items():
            fastest = _driver_fastest_lap_cached(dfs, driver_run_tokens, run_name)
            if fastest is None:
                continue
            lt = _lap_laptimes(df).get(int(fastest), np.nan)
            geo_candidates.append(
                (float(lt) if np.isfinite(lt) else np.inf, run_name, int(fastest))
            )
        if not geo_candidates:
            st.info("No valid laps to sectorise.")
        else:
            _geo_lt, geo_run, geo_lap = min(geo_candidates)
            turns = _driver_cornering_turns_cached(
                dfs, driver_run_tokens, 60.0, 0.5, 8.0, geo_run, geo_lap
            )
            lap_end_m = lsec.lap_end_distance(dfs[geo_run], geo_lap)
            sectors = _driver_lap_sectors_cached(
                dfs,
                driver_run_tokens,
                geo_run,
                _cornering_turns_signature(turns),
                round(float(lap_end_m), 1) if np.isfinite(lap_end_m) else np.nan,
            )
            sectors_token = tuple(
                (
                    int(s.index),
                    str(s.kind),
                    round(float(s.s_start_m), 2),
                    round(float(s.s_end_m), 2),
                    int(s.turn_id) if s.turn_id is not None else -1,
                )
                for s in sectors
            )
            tbl = _driver_sector_times_matrix_cached(dfs, driver_run_tokens, sectors_token)
            if tbl.is_empty():
                st.info("No valid laps to list.")
            else:
                st.dataframe(
                    style_sector_times_table(tbl),
                    use_container_width=True,
                    hide_index=True,
                    height=_full_dataframe_height(tbl.height),
                )
                st.caption(f"Segments from {Path(geo_run).stem} L{geo_lap}: {len(turns)} corners.")
    except Exception as exc:
        st.warning(f"Lap & sector times unavailable: {exc}")


def _render_driver_lap_analysis_subtab(
    dfs: dict[str, pl.DataFrame],
    driver_run_tokens: tuple[tuple[str, FileSignature, str], ...],
) -> None:
    """A/B lap analysis anchored to geometric corner phases.

    Corners come from curvature (r = 1/R) on the fastest selected lap. Apex is
    a point marker; only Entry, Exit and Straight are distance phases.
    """
    st.subheader("Lap Analysis")
    st.caption(
        "A/B comparison anchored to detected corners. Apex is a point; "
        "distance phases are Entry, Exit and Straight."
    )

    lap_options = _driver_lap_compare_options(dfs)
    if len(lap_options) < 2:
        st.warning("Select at least two laps to compare.")
        return

    with st.expander("Corner detection settings", expanded=False):
        detection_values = _lap_analysis_settings_ui()

    R_thr_m = _curvature_thr_1pkm_to_R_thr_m(detection_values["curvature_thr_1pkm"])
    potential_turns: list = []
    potential_candidates: list[tuple[float, str, int]] = []
    for run_name, df in dfs.items():
        fastest_lap = _driver_fastest_lap_cached(dfs, driver_run_tokens, run_name)
        if fastest_lap is None:
            continue
        lap_time_s = _lap_laptimes(df).get(int(fastest_lap), np.nan)
        potential_candidates.append(
            (
                float(lap_time_s) if np.isfinite(lap_time_s) else np.inf,
                run_name,
                int(fastest_lap),
            )
        )
    if potential_candidates:
        _best_time_s, potential_geometry_run, potential_geometry_lap = min(
            potential_candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
    else:
        potential_geometry_run = str(lap_options[0][0])
        potential_geometry_lap = int(lap_options[0][1])
    potential_result: tuple[pl.DataFrame, dict[str, object]] | None = None
    try:
        potential_turns = _driver_cornering_turns_cached(
            dfs,
            driver_run_tokens,
            R_thr_m,
            detection_values["min_dur_s"],
            detection_values["corner_merge_gap_m"],
            potential_geometry_run,
            potential_geometry_lap,
        )
        potential_lap_end_m = lsec.lap_end_distance(
            dfs[potential_geometry_run],
            potential_geometry_lap,
        )
        potential_sectors = _driver_lap_sectors_cached(
            dfs,
            driver_run_tokens,
            potential_geometry_run,
            _cornering_turns_signature(potential_turns),
            round(float(potential_lap_end_m), 1) if np.isfinite(potential_lap_end_m) else np.nan,
        )
        potential_sectors_token = tuple(
            (
                int(sector.index),
                str(sector.kind),
                round(float(sector.s_start_m), 2),
                round(float(sector.s_end_m), 2),
                int(sector.turn_id) if sector.turn_id is not None else -1,
            )
            for sector in potential_sectors
        )
        potential_result = _driver_potential_lap_cached(
            dfs,
            driver_run_tokens,
            potential_sectors_token,
        )
    except Exception as exc:
        st.warning(f"Potential lap unavailable: {exc}")

    ref_options = list(lap_options)
    if potential_result is not None:
        _potential_df, potential_meta = potential_result
        ref_options = [
            (
                POTENTIAL_LAP_RUN,
                POTENTIAL_LAP_ID,
                float(potential_meta.get("lap_time_s", np.nan)),
            ),
            *lap_options,
        ]

    ref_default, _unused_cmp_default = _driver_default_compare_indices(ref_options)
    cmp_default = 0
    if ref_options[ref_default][0] != POTENTIAL_LAP_RUN:
        for idx, option in enumerate(lap_options):
            if option[0] != ref_options[ref_default][0] or option[1] != ref_options[ref_default][1]:
                cmp_default = idx
                break

    lc1, lc2 = st.columns(2)
    with lc1:
        ref_run, ref_lap, _ref_lt = st.selectbox(
            "Reference lap",
            options=ref_options,
            index=ref_default,
            format_func=_format_driver_lap_compare_option,
            key="drv_lap_cmp_ref",
        )
    with lc2:
        cmp_run, cmp_lap, _cmp_lt = st.selectbox(
            "Compared lap",
            options=lap_options,
            index=cmp_default,
            format_func=_format_driver_lap_compare_option,
            key="drv_lap_cmp_cmp",
        )

    analysis_dfs = dict(dfs)
    if potential_result is not None:
        potential_df, _potential_meta = potential_result
        analysis_dfs[POTENTIAL_LAP_RUN] = potential_df

    try:
        summary = drv.lap_comparison_summary(
            analysis_dfs, ref_run, int(ref_lap), cmp_run, int(cmp_lap)
        )
    except Exception as exc:
        st.error(f"Lap comparison unavailable: {exc}")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Reference",
        summary["ref_label"],
        f"{summary['ref_lap_time_s']:.3f} s",
        help="Baseline lap the comparison is measured against (often the synthetic potential lap).",
    )
    k2.metric(
        "Compared",
        summary["cmp_label"],
        f"{summary['cmp_lap_time_s']:.3f} s",
        help="Lap being analysed against the reference.",
    )
    k3.metric(
        "Net Δt",
        f"{summary['total_delta_s']:+.3f} s",
        help="Net time difference vs reference over the lap. Positive = slower than reference.",
    )
    k4.metric(
        "Lost / Gained",
        f"{summary['gross_lost_s']:.3f} / {summary['gross_gained_s']:.3f} s",
        help="Total time lost / gained summed across all corners (gross, before they net out).",
    )

    active_turns: list = []
    selected_turn_set: set[int] = set()
    turn_ids: list[int] = []
    included_turns_key = ""
    click_event_key = ""
    try:
        if ref_run == POTENTIAL_LAP_RUN:
            turns = potential_turns
            corner_run = potential_geometry_run
            corner_lap = potential_geometry_lap
        else:
            ref_lt = float(summary["ref_lap_time_s"])
            cmp_lt = float(summary["cmp_lap_time_s"])
            use_ref_for_corners = np.isfinite(ref_lt) and (
                not np.isfinite(cmp_lt) or ref_lt <= cmp_lt
            )
            corner_run = ref_run if use_ref_for_corners else cmp_run
            corner_lap = int(ref_lap if use_ref_for_corners else cmp_lap)
            turns = _driver_cornering_turns_cached(
                dfs,
                driver_run_tokens,
                R_thr_m,
                detection_values["min_dur_s"],
                detection_values["corner_merge_gap_m"],
                corner_run,
                corner_lap,
            )
        if turns:
            if ref_run == POTENTIAL_LAP_RUN:
                st.caption(
                    "Potential reference built from fastest sectors across loaded CSVs. "
                    f"Corner geometry from {Path(corner_run).stem} L{corner_lap}: "
                    f"{len(turns)} turns."
                )
            else:
                st.caption(
                    f"Corners detected on the fastest selected lap "
                    f"({corner_run} L{corner_lap}): {len(turns)} turns."
                )
            turn_ids = [int(t.turn_id) for t in turns]
            turn_labels = {
                int(t.turn_id): (
                    f"T{int(t.turn_id)} "
                    f"({t.s_entry_m:.0f}-{t.s_exit_m:.0f} m, apex {t.s_apex_m:.0f} m)"
                )
                for t in turns
            }
            detected_turn_token = "-".join(
                f"{int(turn.turn_id)}:"
                f"{round(float(turn.s_entry_m))}:"
                f"{round(float(turn.s_apex_m))}:"
                f"{round(float(turn.s_exit_m))}"
                for turn in turns
            )
            included_turns_key = (
                f"drv_lap_included_turns_{corner_run}_{int(corner_lap)}_{detected_turn_token}"
            )
            click_event_key = (
                f"drv_lap_turn_click_event_{corner_run}_{int(corner_lap)}_{detected_turn_token}"
            )
            if included_turns_key not in st.session_state:
                st.session_state[included_turns_key] = turn_ids
            selected_turn_set = {
                int(turn_id)
                for turn_id in st.session_state.get(included_turns_key, turn_ids)
                if int(turn_id) in set(turn_ids)
            }
            with st.expander("Curves included in Lap Analysis", expanded=False):
                st.caption("Click a highlighted curve on the map to include/exclude it.")
                if st.button(
                    "Include all curves",
                    key=f"drv_lap_include_all_{corner_run}_{int(corner_lap)}_{detected_turn_token}",
                ):
                    st.session_state[included_turns_key] = turn_ids
                    selected_turn_set = set(turn_ids)
                    st.rerun()
                included_names = [
                    _format_cornering_turn_option(turn_id, turn_labels)
                    for turn_id in turn_ids
                    if turn_id in selected_turn_set
                ]
                st.caption(f"{len(selected_turn_set)} / {len(turn_ids)} detected curves included")
                if included_names:
                    st.write(", ".join(included_names))
                else:
                    st.write("No curves selected.")
            active_turns = [turn for turn in turns if int(turn.turn_id) in selected_turn_set]
            if not active_turns:
                st.warning("Select at least one curve to run Lap Analysis.")
    except Exception as exc:
        st.warning(f"Corner detection unavailable: {exc}")
        turns = []
        active_turns = []

    brake_thr = float(detection_values["brake_threshold_pct"])

    phase_table = pl.DataFrame()
    phase_bounds: list = []
    map_phase_bounds = drv.compute_lap_analysis_corner_phases(turns) if turns else []
    lap_gates = _lap_gates_from_run_tokens(driver_run_tokens)

    if active_turns:
        try:
            phase_table, phase_bounds = drv.corner_phase_delta_table(
                analysis_dfs,
                ref_run,
                int(ref_lap),
                cmp_run,
                int(cmp_lap),
                turns=active_turns,
                apex_half_window_m=5.0,
                brake_threshold_pct=brake_thr,
            )
        except Exception as exc:
            st.warning(f"Corner phase breakdown unavailable: {exc}")

    def _whole_lap_metrics_for_option(
        run_name: str,
        lap_id: int,
    ) -> dict[str, float] | None:
        if run_name == POTENTIAL_LAP_RUN:
            return lsec.whole_lap_metrics(analysis_dfs[run_name], int(lap_id))
        return _driver_whole_lap_metrics_cached(
            dfs,
            driver_run_tokens,
            run_name,
            int(lap_id),
        )

    def _sector_summary_for_run(run_name: str) -> dict[str, dict[str, float] | None] | None:
        fastest_lap = _driver_fastest_lap_cached(dfs, driver_run_tokens, run_name)
        if fastest_lap is None:
            return None
        run_turns = _driver_cornering_turns_cached(
            dfs,
            driver_run_tokens,
            R_thr_m,
            detection_values["min_dur_s"],
            detection_values["corner_merge_gap_m"],
            run_name,
            int(fastest_lap),
        )
        if turns:
            run_turns = [turn for turn in run_turns if int(turn.turn_id) in selected_turn_set]
        turns_signature = _cornering_turns_signature(run_turns)
        lap_end_m = lsec.lap_end_distance(dfs[run_name], int(fastest_lap))
        sectors = _driver_lap_sectors_cached(
            dfs,
            driver_run_tokens,
            run_name,
            turns_signature,
            round(float(lap_end_m), 1) if np.isfinite(lap_end_m) else np.nan,
        )
        sectors_token = tuple(
            (
                int(sector.index),
                str(sector.kind),
                round(float(sector.s_start_m), 2),
                round(float(sector.s_end_m), 2),
                int(sector.turn_id) if sector.turn_id is not None else -1,
            )
            for sector in sectors
        )
        return _driver_csv_sector_summary_cached(
            dfs,
            driver_run_tokens,
            run_name,
            sectors_token,
        )

    top_left, top_right = st.columns([1.15, 0.85])
    with top_left:
        if map_phase_bounds:
            try:
                # Gate-line drawing lives in Driver › Overview › Circuit now; this
                # map stays interactive only for clicking curves in/out.
                phase_fig = drv.lap_phase_track_fig(
                    analysis_dfs,
                    ref_run,
                    int(ref_lap),
                    cmp_run,
                    int(cmp_lap),
                    phases=map_phase_bounds,
                    active_turn_ids=selected_turn_set if turns else None,
                )
                phase_fig = _add_lap_detection_gates_to_fig(phase_fig, lap_gates)
                phase_fig_json = tmc.serialize_figure(phase_fig)
                phase_event = tmc.render_track_map_component(
                    phase_fig_json,
                    height_px=430,
                    key=f"drv_lap_phase_map_click_{ref_run}_{ref_lap}_{cmp_run}_{cmp_lap}",
                    draw_enabled=False,
                )
                if bool(st.session_state.get("_dyn_track_open_fullscreen", False)) or bool(
                    phase_event.get("fullscreen_event", False)
                ):
                    st.session_state["_dyn_track_open_fullscreen"] = False
                    _render_lap_phase_fullscreen_dialog(
                        phase_fig,
                        turn_ids=turn_ids,
                        included_turns_key=included_turns_key,
                        event_state_key=click_event_key,
                    )
                if turns and _consume_lap_turn_click_event(
                    phase_event,
                    all_turn_ids=turn_ids,
                    included_state_key=included_turns_key,
                    event_state_key=click_event_key,
                ):
                    st.rerun()
            except Exception as exc:
                st.warning(f"Phase map unavailable: {exc}")
        else:
            st.info("Corner phase map available once corners are detected.")

        top_runs = list(dfs.keys())[:2]
        p1_run = top_runs[0] if top_runs else None
        p2_run = top_runs[1] if len(top_runs) > 1 else None
        if p1_run is not None:
            p_caption = f"P1 = {Path(p1_run).stem}"
            if p2_run is not None:
                p_caption += f" | P2 = {Path(p2_run).stem}"
            st.caption(p_caption)
        if turns and not active_turns:
            st.info("Metrics table available once at least one curve is included.")
        elif p1_run is None:
            st.info("Metrics table unavailable without loaded CSVs.")
        else:
            try:
                p1_summary = _sector_summary_for_run(p1_run)
                p2_summary = _sector_summary_for_run(p2_run) if p2_run is not None else None
                metrics_table = lsec.build_metrics_table(
                    p1_run,
                    p2_run,
                    _whole_lap_metrics_for_option(ref_run, int(ref_lap)),
                    _whole_lap_metrics_for_option(cmp_run, int(cmp_lap)),
                    p1_summary,
                    p2_summary,
                    int(ref_lap),
                    int(cmp_lap),
                )
                st.dataframe(
                    style_metrics_table(
                        metrics_table,
                        lower_better={
                            "LapTime [s]": True,
                            "Throttle [%]": False,
                            "Braking [%]": True,
                            "Coasting [%]": True,
                            "Plausibility [%]": True,
                        },
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception as exc:
                st.warning(f"Metrics table unavailable: {exc}")

    with top_right:
        if phase_table.is_empty():
            st.info("Where is the time? is available once at least one curve is included.")
        else:
            _plotly_chart(
                drv.corner_phase_delta_fig(phase_table),
                use_container_width=True,
                theme=None,
                key=f"drv_corner_phase_bars_{ref_run}_{ref_lap}_{cmp_run}_{cmp_lap}",
            )

    # 2) Per-corner analysis: signal traces (left) + GG + Δt chart (right).
    if phase_bounds and not phase_table.is_empty():
        phase_by_id = {int(p.turn_id): p for p in phase_bounds}
        turn_by_id = {int(t.turn_id): t for t in active_turns}
        total_by_id = {
            int(row["Turn"]): float(row["Total [s]"]) for row in phase_table.iter_rows(named=True)
        }
        corner_options = sorted(
            phase_by_id.keys(),
            key=lambda tid: -abs(total_by_id.get(tid, 0.0)),
        )
        sel_corner = st.selectbox(
            "Inspect corner",
            options=corner_options,
            format_func=lambda tid: f"T{int(tid)}   Δt {total_by_id.get(int(tid), 0.0):+.3f} s",
            key=f"drv_corner_detail_sel_{ref_run}_{ref_lap}_{cmp_run}_{cmp_lap}",
        )
        detail_ref_run, detail_ref_lap = ref_run, int(ref_lap)

        detail_col, right_col = st.columns([1.0, 1.0])
        track_overview_fig: go.Figure | None = None
        track_overview_error: str | None = None
        try:
            track_overview_fig = drv.lap_comparison_track_fig(
                analysis_dfs,
                detail_ref_run,
                int(detail_ref_lap),
                cmp_run,
                int(cmp_lap),
                turns=turns,
                active_turn_ids=selected_turn_set if turns else None,
            )
            track_overview_fig = _add_lap_detection_gates_to_fig(track_overview_fig, lap_gates)
        except Exception as exc:
            track_overview_error = str(exc)

        # ── Left: signal selector controls + corner detail trace ──────────────
        with detail_col:
            signal_options = list(drv.LAP_SIGNAL_OPTIONS)
            hidden_signal = "__hidden__"
            signal_select_options = [hidden_signal, *signal_options]
            default_signal_keys = [
                "delta_s",
                "vx_mps",
                "throttle_pct",
                "brake_pct",
                "steering_deg",
                "ay_mps2",
            ]

            def _format_lap_signal_option(signal_key: str) -> str:
                if signal_key == hidden_signal:
                    return "Hide"
                return drv.LAP_SIGNAL_OPTIONS[signal_key]["label"]

            control_cols = st.columns([0.9, 1.2, 1.2, 1.2])
            with control_cols[0]:
                corner_detail_x_axis = st.selectbox(
                    "X-axis",
                    options=["distance", "time"],
                    format_func=lambda mode: "Distance" if mode == "distance" else "Time",
                    key=f"drv_corner_detail_x_axis_{ref_run}_{ref_lap}_{cmp_run}_{cmp_lap}",
                )
            selected_signal_keys: list[str] = []
            for slot_idx, default_key in enumerate(default_signal_keys[:3]):
                with control_cols[slot_idx + 1]:
                    selected_signal = st.selectbox(
                        f"Plot {slot_idx + 1}",
                        options=signal_select_options,
                        index=signal_select_options.index(default_key),
                        format_func=_format_lap_signal_option,
                        key=(
                            f"drv_corner_detail_signal_{slot_idx}_"
                            f"{ref_run}_{ref_lap}_{cmp_run}_{cmp_lap}"
                        ),
                    )
                if selected_signal != hidden_signal:
                    selected_signal_keys.append(str(selected_signal))
            lower_control_cols = st.columns(3)
            for slot_idx, default_key in enumerate(default_signal_keys[3:], start=3):
                with lower_control_cols[slot_idx - 3]:
                    selected_signal = st.selectbox(
                        f"Plot {slot_idx + 1}",
                        options=signal_select_options,
                        index=signal_select_options.index(default_key),
                        format_func=_format_lap_signal_option,
                        key=(
                            f"drv_corner_detail_signal_{slot_idx}_"
                            f"{ref_run}_{ref_lap}_{cmp_run}_{cmp_lap}"
                        ),
                    )
                if selected_signal != hidden_signal:
                    selected_signal_keys.append(str(selected_signal))

            try:
                detail_fig = drv.corner_detail_fig(
                    analysis_dfs,
                    detail_ref_run,
                    int(detail_ref_lap),
                    cmp_run,
                    int(cmp_lap),
                    turn_by_id[int(sel_corner)],
                    phases=phase_by_id[int(sel_corner)],
                    signal_keys=selected_signal_keys,
                    x_axis_mode=corner_detail_x_axis,
                )
                fullscreen_figures = {}
                fullscreen_gg_figures: dict[str, go.Figure] = {}
                fullscreen_selected = (
                    f"T{int(sel_corner)}   Δt {total_by_id.get(int(sel_corner), 0.0):+.3f} s"
                )
                for corner_id in corner_options:
                    corner_label = (
                        f"T{int(corner_id)}   Δt {total_by_id.get(int(corner_id), 0.0):+.3f} s"
                    )
                    if int(corner_id) == int(sel_corner):
                        fullscreen_figures[corner_label] = detail_fig
                        continue
                    fullscreen_figures[corner_label] = drv.corner_detail_fig(
                        analysis_dfs,
                        ref_run,
                        int(ref_lap),
                        cmp_run,
                        int(cmp_lap),
                        turn_by_id[int(corner_id)],
                        phases=phase_by_id[int(corner_id)],
                        signal_keys=selected_signal_keys,
                        x_axis_mode=corner_detail_x_axis,
                    )
                for corner_id in corner_options:
                    corner_label = (
                        f"T{int(corner_id)}   Δt {total_by_id.get(int(corner_id), 0.0):+.3f} s"
                    )
                    try:
                        fullscreen_gg_figures[corner_label] = drv.corner_gg_fig(
                            analysis_dfs,
                            ref_run,
                            int(ref_lap),
                            cmp_run,
                            int(cmp_lap),
                            phase_by_id[int(corner_id)],
                        )
                    except Exception:
                        continue
                _render_lap_detail_chart(
                    detail_fig,
                    key=(
                        f"drv_corner_detail_{detail_ref_run}_{detail_ref_lap}_{cmp_run}_{cmp_lap}_"
                        f"{int(sel_corner)}_{corner_detail_x_axis}_"
                        f"{'_'.join(selected_signal_keys) or 'empty'}"
                    ),
                    fullscreen_figures=fullscreen_figures,
                    fullscreen_selected=fullscreen_selected,
                    fullscreen_track_figure=track_overview_fig,
                    fullscreen_gg_figures=fullscreen_gg_figures,
                )
            except Exception as exc:
                st.warning(f"Corner detail unavailable: {exc}")

        # ── Right: track map + GG diagram ────────────────────────────────────
        with right_col:
            if track_overview_fig is not None:
                _plotly_chart(
                    track_overview_fig,
                    use_container_width=True,
                    theme=None,
                    key=f"drv_lap_track_{detail_ref_run}_{detail_ref_lap}_{cmp_run}_{cmp_lap}",
                )
            elif track_overview_error is not None:
                st.warning(f"Track map unavailable: {track_overview_error}")
            try:
                _plotly_chart(
                    drv.corner_gg_fig(
                        analysis_dfs,
                        detail_ref_run,
                        int(detail_ref_lap),
                        cmp_run,
                        int(cmp_lap),
                        phase_by_id[int(sel_corner)],
                    ),
                    use_container_width=True,
                    theme=None,
                    key=(
                        f"drv_corner_gg_{detail_ref_run}_{detail_ref_lap}_"
                        f"{cmp_run}_{cmp_lap}_{sel_corner}"
                    ),
                )
            except Exception as exc:
                st.warning(f"GG diagram unavailable: {exc}")

    # Lap-time progression, consistency stats and distribution now live in the
    # always-on Driver › Overview sub-tab (no longer hidden behind a checkbox).
    st.caption(
        "Lap-time progression, variability/consistency stats and the lap-time "
        "distribution are in **Driver › Overview**."
    )


def _render_driver_video_subtab(
    raw_dfs: dict[str, pl.DataFrame],
    file_signatures: dict[str, FileSignature],
    video_server: va.VideoServerInfo,
    csv_files: list[str],
) -> None:
    """Render the synced onboard-video workspace for one selected run."""
    st.caption(
        "Analyse one pilot at a time. The video runs continuously across the "
        "whole file, while the map and telemetry automatically switch to the "
        "current lap. Click a telemetry trace to seek the video."
    )

    run_names = list(raw_dfs)
    if not run_names:
        st.warning("No runs to display.")
        return

    selector_key = "driver_video_selected_run"
    if st.session_state.get(selector_key) not in run_names:
        st.session_state[selector_key] = run_names[0]

    selected_run = st.selectbox(
        "Pilot",
        options=run_names,
        key=selector_key,
        format_func=lambda run_name: Path(run_name).stem,
    )
    raw_df = raw_dfs[selected_run]

    compare_payload: dict | None = None
    compare_lap_id: int | None = None
    compare_sig_token = "none"
    compare_enabled = st.checkbox(
        "Compare with another lap",
        value=False,
        key="driver_video_compare_enabled",
    )
    if compare_enabled:
        default_compare_idx = csv_files.index(selected_run) if selected_run in csv_files else 0
        vc1, vc2 = st.columns([2, 1])
        with vc1:
            compare_file = st.selectbox(
                "Comparison CSV",
                options=csv_files,
                index=default_compare_idx,
                key="driver_video_compare_file",
                format_func=lambda fname: Path(fname).stem,
            )

        try:
            compare_sig = _file_signature(DATA_DIR / compare_file)
            compare_raw_df = (
                raw_dfs[compare_file]
                if compare_file in raw_dfs
                else load_run(str(DATA_DIR / compare_file), compare_sig)
            )
            compare_token = (compare_file, compare_sig)
            compare_lap_options = list(_available_laps_cached(compare_raw_df, compare_token))
            if len(compare_lap_options) > 1:
                compare_lap_options = compare_lap_options[:-1]
            compare_lap_times = _lap_laptimes_cached(compare_raw_df, compare_token)
        except Exception as exc:
            st.warning(f"`{compare_file}`: comparison unavailable — {exc}")
        else:
            if compare_lap_options:
                with vc2:
                    compare_lap_id = int(
                        st.selectbox(
                            "Comparison lap",
                            options=compare_lap_options,
                            key=f"driver_video_compare_lap_{compare_file}",
                            format_func=lambda lap, times=compare_lap_times: (
                                _format_lap_with_laptime(lap, times)
                            ),
                        )
                    )
                try:
                    compare_payload = _video_payload_cached(compare_raw_df, compare_token)
                except Exception as exc:
                    st.warning(f"`{compare_file}`: comparison payload failed — {exc}")
                    compare_payload = None
                    compare_lap_id = None
                else:
                    compare_sig_token = f"{compare_sig[0]}_{compare_sig[1]}_L{compare_lap_id}"
            else:
                st.warning(f"`{compare_file}`: no valid comparison laps.")

    try:
        selected_sig = file_signatures.get(selected_run, _file_signature(DATA_DIR / selected_run))
        payload = _video_payload_cached(raw_df, (selected_run, selected_sig))
    except KeyError as exc:
        st.error(f"`{selected_run}`: {exc}")
        return
    except Exception as exc:
        st.error(f"`{selected_run}`: video payload failed — {exc}")
        return

    video_url = va.video_url_for_csv(selected_run, video_server.available_videos)
    video_diag = va.video_diagnostics_for_csv(
        selected_run,
        video_server.diagnostics,
    )
    if video_url is None:
        st.info(
            f"No onboard video found for `{selected_run}`. "
            f"Drop a file at `videos/{Path(selected_run).stem}.mp4` to enable sync."
        )
    elif video_diag is not None:
        for warning in video_diag.warnings:
            st.warning(f"`{Path(selected_run).stem}.mp4`: {warning}")

    sig = file_signatures.get(selected_run)
    sig_token = f"{sig[0]}_{sig[1]}" if sig else "0_0"
    component_id = f"va_{Path(selected_run).stem}_{sig_token}_{compare_sig_token}"
    html = va.build_video_component_html(
        component_id=component_id,
        video_url=video_url,
        video_server_port=video_server.port,
        payload=payload,
        compare_payload=compare_payload,
        compare_lap_id=compare_lap_id,
        height_px=920,
    )
    components.html(html, height=940, scrolling=False)


def _tab_rb(dfs: dict[str, pl.DataFrame]) -> None:
    """Controls › RB — regen as delivered, from live channels.

    RB's own published channels (RB_*_Trq, RB_Enable) read zero on the available logs (logging
    artifact). The CAT17x brakes with regen only, so these figures read regen as actually
    delivered from the LIVE battery + motor channels: F/R torque distribution (R1, the
    brake-balance plane), capture efficiency (R2), and the regen-current characteristic vs the
    battery cap (R3).
    """
    st.subheader("RB — Regenerative Braking")
    st.caption(
        "Regen as actually delivered (battery + motor channels): how it splits torque, how much energy it recovers, and how hard it pulls."
    )
    st.caption(
        "RB's own published channels (RB_*_Trq, RB_Enable) read zero on these logs — a logging artifact; the CAT17x brakes with regen only, so the figures read it from the live battery + motor-torque channels."
    )
    single = len(dfs) == 1

    # R1 — Regen torque distribution (reuse the Dynamics brake-balance plane) ----
    with st.container(key="rb_h_distribution"):
        st.markdown("##### Regen torque distribution — front vs rear")
    st.caption(
        "Front vs rear regen braking force vs the load-proportional ideal — how regen split the torque."
    )
    try:
        fig, kpis = dyn.ideal_braking_curve_fig(dfs)
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if single and runs:
            v = next(iter(runs.values()))
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Front bias",
                f"{_fmt(v.get('front_bias_mean', np.nan) * 100.0, '.1f')} %",
                help=METRIC_HELP["Brake front bias"],
            )
            c2.metric(
                "Peak combined [g]",
                f"{_fmt(v.get('peak_combined_brake_g', np.nan), '.2f')}",
                help=METRIC_HELP["Peak combined brake [g]"],
            )
            c3.metric("Samples", f"{int(v.get('samples', 0))}")
        elif runs:
            _show_summary_table(
                [
                    {
                        "Run": Path(rn).stem,
                        "Front bias [%]": round(v.get("front_bias_mean", np.nan) * 100.0, 1),
                        "Peak combined [g]": round(v.get("peak_combined_brake_g", np.nan), 2),
                        "Samples": int(v.get("samples", 0)),
                    }
                    for rn, v in runs.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.warning(f"Regen distribution unavailable: {exc}")

    st.divider()
    # R2 — Capture ratio (recovery efficiency) ----------------------------------
    with st.container(key="rb_h_capture"):
        st.markdown("##### Capture ratio — recovered ÷ braking energy")
    st.caption(
        "Per braking event, the fraction of shed kinetic energy returned to the battery (the rest goes to the friction brakes + drag)."
    )
    try:
        fig, kpis = rb.rb_capture_ratio_fig(dfs)
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if single and runs:
            v = next(iter(runs.values()))
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Capture (median)",
                f"{_fmt(v['capture_ratio_median'], '.2f')}",
                help=METRIC_HELP["RB capture ratio"],
            )
            c2.metric(
                "Capture (overall)",
                f"{_fmt(v['capture_ratio_overall'], '.2f')}",
                help=METRIC_HELP["RB capture ratio overall"],
            )
            c3.metric("Events", f"{int(v.get('events', 0))}")
        elif runs:
            _show_summary_table(
                [
                    {
                        "Run": Path(rn).stem,
                        "Capture med": v.get("capture_ratio_median"),
                        "Capture overall": v.get("capture_ratio_overall"),
                        "Events": int(v.get("events", 0)),
                    }
                    for rn, v in runs.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.warning(f"Capture ratio unavailable: {exc}")

    st.divider()
    # R3 — Regen vs brake-demand (characteristic + 80 A ceiling) ----------------
    with st.container(key="rb_h_map"):
        st.markdown("##### Regen vs brake-demand — pull and the 80 A ceiling")
    st.caption(
        "Delivered regen current vs deceleration, coloured by speed, against the 80 A battery cap (below it = recovery headroom)."
    )
    try:
        fig, kpis = rb.rb_regen_brake_map_fig(dfs)
        for w in kpis.get("warnings", []):
            st.warning(w)
        runs = kpis.get("runs", {})
        if single and runs:
            v = next(iter(runs.values()))
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "P95 regen current",
                f"{_fmt(v['p95_regen_current_a'], '.0f')} A",
                help=METRIC_HELP["RB p95 regen current"],
            )
            c2.metric(
                "Current authority",
                f"{_fmt(v['current_authority_pct'], '.0f')}%",
                help=METRIC_HELP["RB current authority"],
            )
            c3.metric(
                "Time at cap",
                f"{_fmt(v['pct_at_cap'], '.1f')}%",
                help=METRIC_HELP["RB time at cap"],
            )
        elif runs:
            _show_summary_table(
                [
                    {
                        "Run": Path(rn).stem,
                        "P95 regen [A]": v.get("p95_regen_current_a"),
                        "Authority [%]": v.get("current_authority_pct"),
                        "At cap [%]": v.get("pct_at_cap"),
                        "Samples": int(v.get("samples", 0)),
                    }
                    for rn, v in runs.items()
                ]
            )
        _plotly_chart(fig, use_container_width=True, theme=None)
    except Exception as exc:
        st.warning(f"Regen vs brake-demand unavailable: {exc}")

    st.divider()


def _tab_controls(dfs: dict[str, pl.DataFrame]) -> None:
    if st.session_state.get("controls_subsection") not in {"TC", "TV", "RB"}:
        st.session_state["controls_subsection"] = "TC"

    controls_section = st.segmented_control(
        "Controls section",
        options=["TC", "TV", "RB"],
        default="TC",
        required=True,
        key="controls_subsection",
        label_visibility="collapsed",
        width="stretch",
    )

    # Keyed container per sub-tab → unique element path, so Streamlit doesn't reuse
    # headings across TC/TV/RB and leak stale auto-anchors between them.
    if controls_section == "TC":
        with st.container(key="controls_tc"):
            _tab_tc(dfs)
    elif controls_section == "TV":
        with st.container(key="controls_tv"):
            _tab_tv(dfs)
    elif controls_section == "RB":
        with st.container(key="controls_rb"):
            _tab_rb(dfs)


# ── Main ──────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def _logo_data_uri(path: str, file_signature: FileSignature) -> str | None:
    """Read and base64-encode the brand logo once; cached across reruns."""
    _ = file_signature  # cache-busts when the file changes
    try:
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:image/png;base64,{encoded}"


_GLOBAL_STYLE_CSS = """
<style>
  /* Fonts (Archivo / IBM Plex Sans / IBM Plex Mono) are loaded AND applied via
     the native theme config in .streamlit/config.toml — injected font-family
     rules lose specificity to Streamlit's own heading/body styles. Only the
     data-testid rules below (which DO win) live here. */
  [data-testid="stCaptionContainer"], .stCaption {
    color: __MUTED__;
    font-size: 1.1rem;
    line-height: 1.45;
  }
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li,
  [data-testid="stWidgetLabel"],
  [data-testid="stSelectbox"] label,
  [data-testid="stMultiSelect"] label,
  [data-testid="stNumberInput"] label,
  [data-testid="stCheckbox"] label,
  [data-testid="stRadio"] label {
    font-size: 1.1rem;
    line-height: 1.45;
  }
  [data-testid="stDataFrame"],
  [data-testid="stTable"],
  [data-testid="stDataFrame"] *,
  [data-testid="stTable"] * {
    font-size: 17px;
  }

  /* KPI metrics → elevated surface cards with depth (upgrades every st.metric). */
  [data-testid="stMetric"] {
    background: __SURFACE__;
    border: 1px solid __SURFACE_BORDER__;
    border-left: 3px solid __ACCENT__;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.35);
  }
  [data-testid="stMetricLabel"] {
    font-family: __DISPLAY__;
    color: __MUTED__;
    font-size: 1.15rem;
    letter-spacing: 0.2px;
  }
  /* Big numbers: mono + tabular figures so digits align in a column. */
  [data-testid="stMetricValue"] {
    font-family: __MONO__;
    font-size: 1.9rem;
    font-variant-numeric: tabular-nums;
    font-feature-settings: "tnum" 1;
  }
  [data-testid="stMetricDelta"] {
    font-size: 1.1rem;
  }
  /* Best-effort tabular figures for data tables (DOM-rendered cells). */
  [data-testid="stDataFrame"], [data-testid="stTable"] {
    font-variant-numeric: tabular-nums;
    font-feature-settings: "tnum" 1;
  }
  /* Plotly's floating zoom toolbar appears on hover; keep it above dense plots. */
  [data-testid="stPlotlyChart"] .js-plotly-plot .modebar-container,
  .stPlotlyChart .js-plotly-plot .modebar-container {
    top: 88px !important;
  }
  [data-testid="stButtonGroup"] [data-baseweb="button-group"] button[data-testid="stBaseButton-segmented_controlActive"],
  [data-testid="stButtonGroup"] [data-baseweb="button-group"] button[kind="segmented_controlActive"] {
    color: __ACCENT__ !important;
    border-color: __ACCENT__ !important;
    background-color: rgba(0, 59, 116, 0.10) !important;
    box-shadow: inset 0 0 0 1px __ACCENT__ !important;
  }
  [data-testid="stButtonGroup"] [data-baseweb="button-group"] button,
  [data-testid="stBaseButton-secondary"],
  [data-testid="stBaseButton-primary"],
  [data-testid="stSidebar"] [data-baseweb="select"] *,
  [data-baseweb="select"] *,
  [data-baseweb="popover"] * {
    font-size: 1.1rem;
  }
  [data-testid="stButtonGroup"] [data-baseweb="button-group"] button[data-testid="stBaseButton-segmented_controlActive"] *,
  [data-testid="stButtonGroup"] [data-baseweb="button-group"] button[kind="segmented_controlActive"] *,
  [data-testid="stButtonGroup"] [data-baseweb="button-group"] button[data-testid="stBaseButton-segmented_controlActive"] svg,
  [data-testid="stButtonGroup"] [data-baseweb="button-group"] button[kind="segmented_controlActive"] svg {
    color: __ACCENT__ !important;
    fill: __ACCENT__ !important;
    stroke: __ACCENT__ !important;
  }
  [data-testid="stButtonGroup"] [data-baseweb="button-group"] button[data-testid="stBaseButton-segmented_controlActive"]:hover,
  [data-testid="stButtonGroup"] [data-baseweb="button-group"] button[kind="segmented_controlActive"]:hover,
  [data-testid="stButtonGroup"] [data-baseweb="button-group"] button[data-testid="stBaseButton-segmented_controlActive"]:focus,
  [data-testid="stButtonGroup"] [data-baseweb="button-group"] button[kind="segmented_controlActive"]:focus,
  [data-testid="stButtonGroup"] [data-baseweb="button-group"] button[data-testid="stBaseButton-segmented_controlActive"]:active,
  [data-testid="stButtonGroup"] [data-baseweb="button-group"] button[kind="segmented_controlActive"]:active {
    color: __ACCENT__ !important;
    border-color: __ACCENT__ !important;
    background-color: rgba(0, 59, 116, 0.16) !important;
  }
</style>
"""


def _render_dashboard_header() -> None:
    """Render the dashboard brand and author credit."""
    st.markdown(
        _GLOBAL_STYLE_CSS.replace("__BODY__", FONT_FAMILY)
        .replace("__DISPLAY__", FONT_DISPLAY)
        .replace("__MONO__", FONT_MONO)
        .replace("__SURFACE__", _SURFACE)
        .replace("__SURFACE_BORDER__", _SURFACE_BORDER)
        .replace("__ACCENT__", BRAND_BLUE_600)
        .replace("__MUTED__", _TEXT_MUTED),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
          .cat17x-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.25rem;
            padding: 0.45rem 0 1.05rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.10);
            margin-bottom: 1rem;
          }
          .cat17x-topbar__logo {
            width: min(430px, 58vw);
            max-height: 92px;
            object-fit: contain;
          }
          .cat17x-topbar__credit {
            color: #F2F4F8;
            font-size: 0.95rem;
            font-weight: 600;
            letter-spacing: 0;
            text-align: right;
            white-space: nowrap;
          }
          .cat17x-topbar__credit span {
            display: block;
            color: rgba(242, 244, 248, 0.64);
            font-size: 0.72rem;
            font-weight: 500;
            margin-top: 0.16rem;
          }
          @media (max-width: 900px) {
            .cat17x-topbar {
              align-items: flex-start;
              flex-direction: column;
              gap: 0.75rem;
            }
            .cat17x-topbar__logo {
              width: min(100%, 360px);
            }
            .cat17x-topbar__credit {
              text-align: left;
              white-space: normal;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    logo_uri = (
        _logo_data_uri(str(APP_LOGO_PATH), _file_signature(APP_LOGO_PATH))
        if APP_LOGO_PATH.exists()
        else None
    )
    if logo_uri is not None:
        logo_html = f'<img class="cat17x-topbar__logo" src="{logo_uri}" alt="BCN eMotorsport">'
    else:
        logo_html = '<div class="cat17x-topbar__credit">BCN eMotorsport</div>'
    st.markdown(
        f"""
        <div class="cat17x-topbar">
          {logo_html}
          <div class="cat17x-topbar__credit">
            Ignacio Llopis
            <span>Copyright</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="CAT17x — Telemetry",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _render_dashboard_header()

    st.sidebar.markdown("### CAT17x")
    st.sidebar.caption("Formula Student Telemetry")
    st.sidebar.divider()

    csv_files = [p.name for p in _telemetry_csv_paths(DATA_DIR)]
    if not csv_files:
        st.error(f"No CSV files found in `{DATA_DIR}`.")
        return

    _autodetect_laps(DATA_DIR)
    video_server = _video_server_cached(
        str(REPO_ROOT),
        _video_dir_signature(REPO_ROOT),
    )
    if video_server.error:
        st.sidebar.error(f"Video assets unavailable: {video_server.error}")

    st.sidebar.markdown("### Runs")
    selected_files = st.sidebar.multiselect(
        "Telemetry CSV files",
        options=csv_files,
        default=csv_files[:2],
        key="selected_csv_files",
        format_func=_format_csv_file_option,
        placeholder="Choose one or more CSV files",
    )
    st.sidebar.caption(
        f"{len(selected_files)} of {len(csv_files)} runs selected. "
        "Any number of runs can be compared."
    )
    if not selected_files:
        st.sidebar.warning("Select at least one run.")
        return

    file_signatures = {fname: _file_signature(DATA_DIR / fname) for fname in selected_files}

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

    _render_event_mode_selector(selected_files, raw_dfs)

    st.sidebar.divider()
    st.sidebar.markdown("### Laps")
    st.sidebar.caption("These lap selections are applied to every tab.")

    dfs: dict[str, pl.DataFrame] = {}
    run_source_files: dict[str, str] = {}
    run_file_signatures: dict[str, FileSignature] = {}
    any_run_has_detectable_laps = False
    for fname in selected_files:
        raw_df = raw_dfs.get(fname)
        if raw_df is None:
            continue
        run_token = (fname, file_signatures[fname])
        try:
            lap_options = list(_available_laps_cached(raw_df, run_token))
        except Exception as exc:
            st.sidebar.warning(f"`{fname}`: cannot list laps — {exc}")
            continue
        if lap_options:
            any_run_has_detectable_laps = True

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
            dfs[fname] = _select_laps_df_cached(
                raw_df,
                run_token,
                tuple(int(lap) for lap in selected_laps),
            )
            run_source_files[fname] = fname
            run_file_signatures[fname] = file_signatures[fname]
        except Exception as exc:
            st.sidebar.warning(f"Skipping `{fname}`: {exc}")

    if not dfs:
        if any_run_has_detectable_laps:
            # Laps ARE detected — the selection just collapsed to empty (e.g. a
            # single-lap autocross run whose sidebar lap selection got cleared,
            # so Streamlit keeps the stale empty value over the `default`). Do
            # NOT show the gate-drawing Circuit map here: that wrongly implies
            # detection is needed and reads as if the detected lap was erased.
            st.info("Select at least one lap in the sidebar to view the analysis.")
            return
        # No run has selectable laps yet. This is the expected bootstrap state
        # for manual modes (Autocross / Skidpad / manual finish line): laps only
        # appear once gate lines are drawn. Rather than abort the whole app —
        # which would hide the very map needed to draw them — render the Circuit
        # draw map from the raw GPS so the user can draw lines here and then
        # assign/detect from the sidebar Event mode panel.
        st.warning(
            "No laps detected yet for the selected run(s). For a manual mode "
            "(Autocross / Skidpad / manual finish line), draw the gate line(s) on the "
            "map below, then assign and detect from the **Event mode** panel in the sidebar."
        )
        raw_tokens = tuple((fname, file_signatures[fname], "raw") for fname in raw_dfs)
        st.subheader("Circuit")
        try:
            _render_circuit_gate_map(raw_dfs, raw_tokens)
        except Exception as exc:
            st.error(f"Circuit map unavailable: {exc}")
        return

    # Seed the shared driver-identity colour map so every tab/figure paints a
    # given run with the same colour, regardless of load order.
    set_run_colors(dfs.keys())

    st.session_state["_run_source_files"] = run_source_files
    st.session_state["_run_file_signatures"] = run_file_signatures

    section_renderers = {
        "Driver": _tab_driver,
        "Dynamics": _tab_dynamics,
        "Powertrain": _tab_powertrain,
        "Controls": _tab_controls,
        "Events": _tab_events,
    }

    if "dashboard_section" not in st.session_state:
        st.session_state["dashboard_section"] = "Driver"
    elif st.session_state["dashboard_section"] in {"TC", "TV", "RB"}:
        st.session_state["controls_subsection"] = st.session_state["dashboard_section"]
        st.session_state["dashboard_section"] = "Controls"
    elif st.session_state["dashboard_section"] not in section_renderers:
        st.session_state["dashboard_section"] = "Driver"

    # `st.tabs` renders every tab on each rerun, which hurts multi-run loading.
    active_section = st.segmented_control(
        "Section",
        options=list(section_renderers),
        default="Driver",
        required=True,
        key="dashboard_section",
        label_visibility="collapsed",
        width="stretch",
    )
    if active_section is not None:
        if active_section == "Driver":
            section_renderers[active_section](
                dfs,
                run_file_signatures,
                video_server,
                {run_name: raw_dfs[run_name] for run_name in dfs if run_name in raw_dfs},
                csv_files,
            )
        else:
            section_renderers[active_section](dfs)


if __name__ == "__main__":
    main()
