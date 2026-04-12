# CAT17x Data Analysis — Agent Context

## Role
Expert in motorsport data analysis for a Formula Student electric team (4WD).
Goal: extract actionable insights from vehicle telemetry data to improve
car performance, validate control systems and support setup decisions.

## The car
- 4WD electric, one motor per wheel
- Active control systems: Torque Vectoring (TV), Traction Control (TC),
  Regenerative Braking (RB), Power Control (PC)
- Optimal SR acceleration: +0.20 (TC)
- Optimal SR braking: -0.20 (RB)

## Analysis categories
- Powertrain
- Vehicle dynamics
- Vehicle Controls: Torque Vectoring (TV), Traction Control (TC), Regenerative Braking (RB)
- Driver Performance

## Data
- CSV at 100 Hz, TimeStamp in seconds
- Standard filter: `laps > 0 AND laptime.is_not_nan()`
- Boolean CSV columns are float64: compare with `== 1.0`
- TimeStamp in seconds (not ms)

## Libraries
- **Polars** — data manipulation (not pandas)
- **Plotly** — all charts (interactive, `template="plotly_dark"`)
- **Streamlit** — dashboard
- **NumPy / SciPy** — numerical computation

## Code style
- All code and comments in English
- Functions with type hints
- One function = one metric
- Units in variable names: `torque_nm`, `velocity_ms`, `temp_c`

## Project structure
- `app.py` — main Streamlit dashboard
- `utils.py` — shared utilities
- `src/powertrain.py`, `src/dynamics.py`, `src/tc.py`, `src/tv.py`, `src/rb.py`, `src/pilot.py`

## Load data pattern
```python
import polars as pl

def load_data(path: str) -> pl.DataFrame:
    df = pl.read_csv(path)
    df = df.filter((pl.col("laps") > 0) & pl.col("laptime").is_not_nan())
    df = df.with_columns(
        pl.col("TimeStamp").diff().fill_null(0.01).alias("dt_s")
    )
    return df
```

## Interactive charts
- All charts use `template="plotly_dark"`
- Every chart that shows multi-lap data must include a lap selector
- The dashboard supports loading multiple CSV files simultaneously; a run selector switches between them and all charts update accordingly
- Track section filters are defined interactively by the user drawing selections on a GPS map (`VN_latitude` vs `VN_longitude`)

## Dashboard run selector
```python
from pathlib import Path
import streamlit as st

csv_files = list(Path("data/").glob("*.csv"))
selected = st.selectbox("Select run", csv_files)
df = load_data(selected)
```

## GG diagram
- Axes: `ay` (lateral, x) vs `ax` (longitudinal, y), equal scale
- One color per lap; fastest lap = purple, rest = RdYlGn gradient
- Pool indices stored in `customdata` per point for cross-chart linking
- No lap legend on the GG diagram itself; lap selector is shared globally

## Cross-chart interaction pattern
- Selecting points on any chart stores pool indices in `st.session_state`
- A version counter (`_gg_ver`, `_cross_ver`) triggers `st.rerun()` to sync all charts
- `zone_mask`: boolean array over pool indices from track map selection, passed as `extra_mask` to distance charts
- `extra_mask` parameter in `add_dist_traces` ANDs with per-lap mask to filter plotted points

## Lap selector
- Sorted fastest → slowest by laptime
- Color squares shown next to each lap label using `st.markdown(..., unsafe_allow_html=True)`
- Format (single CSV): `L{lap}  ({laptime:.2f}s)`
- Format (multi CSV): `{run} · L{lap}  ({laptime:.2f}s)`

## Vehicle signals
Full signal list and descriptions are in `.claude/Variables_CSV.pdf`.
Read it when you need to understand what a signal means or what units it uses.
