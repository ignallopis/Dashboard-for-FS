# CAT17x Telemetry Dashboard

Interactive telemetry analysis dashboard for the CAT17x Formula Student electric car, built with Streamlit and Plotly.

## The car

CAT17x is a 4WD electric Formula Student car with one motor per wheel and the following active control systems:

- **TV** — Torque Vectoring
- **TC** — Traction Control
- **RB** — Regenerative Braking
- **PC** — Power Control

## Dashboard modules

| Module | Description |
|---|---|
| Powertrain | Motor torques, power, energy consumption per wheel |
| Dynamics | Longitudinal and lateral accelerations, load transfer |
| Cornering | GG diagram, grip factor analysis |
| Torque Vectoring | TV intervention, yaw rate tracking |
| Traction Control | Slip ratio, TC activation and corrections |
| Regenerative Braking | Braking slip ratio, RB activation |
| Driver | Throttle, brake, steering inputs |
| Lap Sectors | Sector time comparison across laps |
| Track Map | GPS-based track map with channel overlay |
| Video Analysis | Synchronized video with telemetry data |

## Setup

```bash
pip install -r requirements.txt
streamlit run src/dashboard.py
```

Data files (CSV at 100 Hz) go in the `data/` folder.
