# Codex Agent — CAT17x

## Project
Formula Student 4WD electric telemetry analysis.
One motor per wheel (FL, FR, RL, RR).
No hay frenada hidraulica,solo regenerativa
Active systems: Torque Vectoring (TV), Traction Control (TC), Regenerative Braking (RB).
Optimal slip ratio: +0.20 acceleration (TC), −0.20 braking (RB).


## Data loading
- Entry point: `utils.load_data(path)` — handles filtering and adds `dt_s` column;
  raises `ValueError` if the CSV has no valid laps (trigger `lapcount` first).
- Lap detection: `src.lapcount.csv_needs_lap_detection(path)` + `detect_and_write_laps(path)`
  auto-detect laps from GPS and overwrite the CSV with `laps` / `laptime` columns.
  The dashboard runs this at startup for every CSV in `data/` that lacks laps.
  `detect_and_write_laps` sweeps a fallback ladder of `(min_vel, gate_half_width)`
  pairs because the start gate is placed at the first high-speed GPS sample —
  too-low `min_vel` puts the gate on the paddock push-off, not on the real line.
- No utilizar en los calculos lap 0 y la ultima vuelta
- Signal reference: `.claude/Variables_CSV.pdf`
- Boolean columns are float64 — compare with `== 1.0`
- Full signal list and units in `Variables_CSV.pdf`

## Stack
Polars (not pandas), Plotly dark theme, Streamlit, NumPy, SciPy

## Project structure
- `data/` — CSV files
- `src/dashboard.py` — Streamlit entry point: `streamlit run src/dashboard.py`
- `src/tc.py` — TC metrics and figures
- `src/tv.py` — TV metrics and figures
- `src/dynamics.py` — dynamics metrics and figures
- `src/driver.py` — driver performance metrics and figures
- `src/powertrain.py` — powertrain metrics and figures
- `src/lapcount.py` — lap detection from GPS
- `utils.py` — colours, dark theme, shared data helpers

## Code conventions
- English code and comments, type hints, units in variable names
- One function = one metric or one figure
- `src/` modules return `go.Figure` — never render directly
- `src/dashboard.py` is the only file that calls `st.plotly_chart()`
- Colours and theme constants live in `utils.py`

## Dashboard
- Sidebar: CSV file selector, lap selector, active systems filter
- Tabs: Dynamics | Powertrain | TC | TV | RB | Driver
- Each tab groups all relevant signals and metrics for that system
- Plots share x-axis (distance [m]), stacked vertically within each tab

## Analysis categories and key signals
- **Powertrain**: motor torques, currents, temperatures, power per wheel, SoC
- **Dynamics**: vx, ax, ay, slip ratio, slip angle, tyre forces, GG diagram
- **TC**: slip ratio per wheel vs ±0.20 reference, TC torque limits, TC enable
- **TV**: yaw rate error, Mz tracking (desired vs actual), torque distribution
- **RB**: regenerative torque per wheel, brake balance, SR vs −0.20 reference
- **Driver**: lap times, delta, throttle, brake, steering
  - Throttle KPIs (`src/driver.py`): histogram, full-throttle time (TP > 95 %),
    throttle speed = median |dTP/dt| with TP < 100 % and brake released.
    Median is used because mean is dominated by 100 Hz transients.
  - All driver figures take a `dfs` dict and overlay multiple runs (driver-vs-driver).

## Visual constants
- Wheel colours: FL=#4DB3F2, FR=#F28C40, RL=#73D973, RR=#D973D9
- Lap colours: purple (fastest) → RdYlGn gradient → red (slowest)
- Reference lines: SR=+0.20 (TC), SR=−0.20 (RB)
- Dark theme constants in `utils.py`

## Behaviour
- Flag any function that renders directly instead of returning `go.Figure`
- Challenge physical correctness of metrics against signal definitions
- Flag anything overcomplicated for a Formula Student team
