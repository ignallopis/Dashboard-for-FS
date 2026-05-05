# Codex Agent - CAT17x

## Project Goal
Analyse telemetry from a 4WD electric Formula Student car and keep the
result as a professional Streamlit dashboard where KPIs and raw signals are
visualised together, organised by system.

## Project
- Formula Student 4WD electric telemetry analysis.
- One motor per wheel: FL, FR, RL, RR.
- No hydraulic braking, regenerative only.
- Active systems in scope: Torque Vectoring (TV), Traction Control (TC),
  Regenerative Braking (RB).
- Optimal slip ratio: `+0.20` in acceleration for TC, `-0.20` in braking
  for RB.

## Data Loading
- Entry point: `utils.load_data(path)`. It filters valid data and adds `dt_s`.
- `utils.load_data(path)` raises `ValueError` if the CSV has no valid laps.
- Lap detection:
  `src.lapcount.csv_needs_lap_detection(path)` and `detect_and_write_laps(path)`.
- `detect_and_write_laps(path)` overwrites the CSV with `laps` and `laptime`.
- The dashboard runs lap detection at startup for CSVs in `data/` that lack laps.
- Do not use lap 0 or the last lap in calculations.
- Boolean columns are `float64`; compare with `== 1.0`.
- Signal reference and units: `docs/context/Variables_CSV.pdf`.

## Stack
Polars, Plotly dark theme, Streamlit, NumPy, SciPy.

## Project Structure
- `data/`: CSV files.
- `src/dashboard.py`: Streamlit entry point, `streamlit run src/dashboard.py`.
- `src/tc.py`: TC metrics and figures.
- `src/tv.py`: TV metrics and figures.
- `src/rb.py`: RB metrics and figures.
- `src/dynamics.py`: dynamics metrics and figures.
- `src/driver.py`: driver performance metrics and figures.
- `src/powertrain.py`: powertrain metrics and figures.
- `src/lapcount.py`: lap detection from GPS.
- `utils.py`: colours, dark theme, shared data helpers.

## Code Conventions
- English code and comments, with type hints and units in variable names when useful.
- One function = one metric or one figure.
- `src/` modules return `go.Figure`; never render directly.
- `src/dashboard.py` is the only file that calls `st.plotly_chart()`.
- Colours and theme constants live in `utils.py`.

## Module API Pattern
- Dashboard-facing functions in `src/` may return `go.Figure`,
  `(go.Figure, kpis_dict)`, or `list[go.Figure]`.
- Standalone CLI helpers are acceptable inside `src/` modules if they print
  KPIs and call `.show()` only from a local `main()` path.
- All `st.*` calls belong exclusively in `src/dashboard.py`.

## Dashboard
- Sidebar: CSV selector, lap selector, active systems filter.
- Tabs: Dynamics | Powertrain | TC | TV | RB | Driver.
- Each tab groups signals and metrics for one system.
- Plots share x-axis as distance `[m]` and stack vertically.

## Analysis Scope
- Powertrain: motor torques, currents, temperatures, power per wheel, SoC.
- Dynamics: `vx`, `ax`, `ay`, slip ratio, slip angle, tyre forces, GG diagram.
- TC: slip ratio per wheel versus `+0.20`, TC torque limits, TC enable.
- TV: yaw rate error, Mz tracking, desired versus actual Mz, torque distribution.
- RB: regenerative torque per wheel, brake balance, slip ratio versus `-0.20`.
- Driver: lap times, delta, throttle, brake, steering.
- Driver figures take a `dfs` dict and overlay multiple runs.

## Visual Constants
- Wheel colours: `FL=#4DB3F2`, `FR=#F28C40`, `RL=#73D973`, `RR=#D973D9`.
- Lap colours: purple for fastest through `RdYlGn` to red for slowest.
- Reference lines: `SR=+0.20` for TC and `SR=-0.20` for RB.
- Dark theme constants live in `utils.py`.

## Vehicle Parameters
- Full reference: `docs/context/cat17x_parameters.md`.
- Key values commonly reused in analysis:
  - Wheelbase `1.53 m`
  - Front track `1.225 m`
  - Rear track `1.175 m`
  - Total mass `288 kg`
  - CoG height `0.278 m`
  - Front roll stiffness share `47.5 %`
  - Max motor torque `+27.5 N·m`
  - Max regen torque `-27.5 N·m`
  - Gear ratio `9.05`
  - Wheel radius `0.2032 m`
  - Max power `80 kW`

## Codex Project Memory
- Stable project rules live here in `AGENTS.md`.
- Non-obvious, discovered, or fast-changing facts live in `.codex/skills.md`.
- When a task depends on project quirks or recent discoveries, consult
  `.codex/skills.md` before editing.
- When you confirm a new quirk or implementation trap, add it to
  `.codex/skills.md`.

## Validation
- Prefer the project virtualenv for checks and local runs:
  `./.venv/bin/python` and `./.venv/bin/streamlit`.

## Behaviour
- Flag any function that renders directly instead of returning `go.Figure`.
- Challenge physical correctness of metrics against signal definitions.
- Flag anything overcomplicated for a Formula Student team.
