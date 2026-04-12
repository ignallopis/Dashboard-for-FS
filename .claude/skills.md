# Skills — CAT17x

## Load data
```python
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

def load_data(path: str) -> pl.DataFrame:
    df = pl.read_csv(path)
    df = df.filter((pl.col("laps") > 0) & pl.col("laptime").is_not_nan())
    df = df.with_columns(
        pl.col("TimeStamp").diff().fill_null(0.01).alias("dt_s")
    )
    return df

# In dashboard: list all available runs
csv_files = list(Path("data/").glob("*.csv"))
selected = st.selectbox("Select run", csv_files)
df = load_data(selected)
```

## Libraries
- **Polars** — data manipulation (not pandas)
- **Plotly** — all charts (interactive)
- **Streamlit** — dashboard
- **NumPy / SciPy** — numerical computation

## Code style
- All code and comments in English
- Functions with type hints
- One function = one metric
- Units in variable names: `torque_nm`, `velocity_ms`, `temp_c`
- Boolean CSV columns are float64: compare with `== 1.0`
- TimeStamp in seconds (not ms)

## Project structure
Each analysis category has its own script:
- `src/powertrain.py`
- `src/dynamics.py`
- `src/controls.py`
- `src/tc.py`
- `src/tv.py`
- `src/rb.py`
- `src/pilot.py`

## Interactive charts
- All charts use `template="plotly_dark"`
- Every chart that shows multi-lap data must include a lap selector
- The dashboard supports loading multiple CSV files simultaneously.
  A run selector switches between them and all charts update accordingly
- Track section filters are defined interactively by the user drawing
  selections on a GPS map (`VN_latitude` vs `VN_longitude`)

## Vehicle signals
Full signal list and descriptions are in `.claude/Variables_CSV.pdf`.
Read it when you need to understand what a signal means or what units it uses.