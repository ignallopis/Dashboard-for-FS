---
name: signal-lookup
description: "Use when you need the meaning, units, sign, or validity of a CAT17x telemetry signal or vehicle parameter, or before using an unfamiliar CSV column in a calculation. Triggers on 'qué es la señal X', 'what units is X', 'does column X exist', 'qué convención tiene X'."
---

# Signal Lookup

Resolve what a CAT17x signal/parameter means **before** using it — names lie, the data
doesn't.

## Where the answers live
- **Signals** (meaning + units): `docs/context/Variables_CSV.pdf`. Read it for any
  channel you're unsure about.
- **Vehicle parameters** (mass, geometry, limits, aero, brake, gear): `docs/context/cat17x_parameters.md`
  (source of truth = `Parameters.m`). Use these as constants — never hardcode numbers.
- **Tyre model**: `docs/context/Fx_Pacejka.md`, `docs/context/Fy_Pacejka.md`.
- **Discovered quirks**: `.claude/knowledge.md`.

## Conventions & traps (check before trusting a column)
- **`Steering`** = steering-potentiometer value in **radians**, NOT a road-wheel angle.
  Use it **directly** — **never ÷ `STEERING_RATIO` (3.15)** (mechanical column ratio,
  reference only). Right turns **negative**.
- **Booleans are `float64`** → compare with `== 1.0`, not truthiness.
- **Sampling**: 100 Hz, `TimeStamp` in **seconds** (not ms).
- **Standard filter**: `laps > 0 AND laptime.is_not_nan()`; never use lap 0 or the last lap.
- **Estimated vs measured**: `Est_FZ*` are *estimated* vertical loads → corner-weighting /
  cross-weight read ~50% by construction. Flag it; don't over-trust.
- **All-zero in human runs**: `steering_actualPosRad`, `AS_Steering`, and `delta`/
  `globalDelta` (those are DV lap-time deltas) — not usable as a road-wheel angle.
- **Control enable flags lie on the team's circuit logs**: `TV_Enable` / `RB_Enable` can
  stay 0 while the mode/torque channels are active → prefer mode/command/torque evidence.
- **Uncalibrated on circuit logs** (`Cerpa_FSG`, `Martinez_FSG`): no `Pot_Calibration_*`,
  `DAMPER_CALIBRATED=False` → roll/pitch are damper-derived & uncalibrated; setup
  deviation-vs-theory KPIs are hidden by design.

## How to answer
State the signal's **meaning, units, sign, and sampling**, and call out any trap above
that applies. If a column might not exist, check a CSV header (`data/Cerpa_FSG.csv`)
before relying on it.
