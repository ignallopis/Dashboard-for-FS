# CAT17x Data Analysis

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
- Some CSVs require a prior script to calculate `laps` and `laptime`
- Standard filter: `laps > 0 AND laptime.is_not_nan()`
- Test CSV: `data/run4_2025-08-24.csv`

## How to work
- When something is ambiguous or unclear, ask before assuming
- At the start of every session, read `.claude/skills.md`

## Skills
Reusable knowledge is stored in `.claude/skills.md`.
Update it autonomously when you learn something worth keeping.