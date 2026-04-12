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
- The dashboard supports loading multiple CSV files simultaneously
- Track section filters are defined interactively via GPS map selections (`VN_latitude` vs `VN_longitude`)
