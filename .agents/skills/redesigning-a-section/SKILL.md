---
name: redesigning-a-section
description: "Use when rethinking or rebuilding a whole CAT17x dashboard sub-section — in any tab (Dynamics, Powertrain, Driver, TV/TC/RB, Setup, Grip…) — into a coherent SET of figures, deciding WHICH metrics it should have, before scaffolding individual ones. Triggers on 'rediseña/replantea la sección X', 'qué métricas pongo en X', 'analicemos el comportamiento/rendimiento en X', 'mira qué falta y qué sobra en esta sección', 'esta sección es un cajón de sastre'."
---

# Redesigning a Section

Design a whole dashboard sub-section as a **coherent set of car-behavior figures**, then
hand each one to `new-metric`. This is the zoom level **above** `new-metric`: that skill
builds ONE figure; this one decides WHICH figures the section should have and why they earn
their place together.

**Core principle:** each figure measures a *different axis of the physics* — not five views
of the same number. **Physical-and-simple beats sophisticated-and-opaque.**

## When to use
- Rebuilding/auditing an entire sub-section (Braking, Cornering, Acceleration, Setup…).
- "What should this section show about how the car behaves in X?"
- A section feels like a grab-bag, has duplicated metrics, or has dead/contrived plots.

**Not for:** adding a single metric to an existing, healthy section → go straight to
`new-metric`. Pure render/layout fixes → neither skill.

## The flow

### 1. Brainstorm the question (design before code)
Invoke `superpowers:brainstorming`. Understand, one question at a time, **what the user
wants this section to reveal** — the behavior, capability or quality it should expose
(however that reads for the area: a chassis phase, a powertrain limit, a control system's
quality, a driver trait). Don't jump to formulas. Skim a recently-redesigned section for
the house style and depth, but copy the *approach*, not its specific figures.

### 2. Inventory signals before formulas
For every candidate figure, resolve its signals with the **signal-lookup** skill against
`docs/context/variables_csv.pdf`: real column name, units, sign, sampling, traps.
(`Steering` is radians used **directly** — never ÷3.15 — right turns negative; booleans
`== 1.0`; 100 Hz.) Prefer the signal that **directly measures** what the figure claims over
a convenient proxy, and beware channels that read flat-zero or stale in these logs.
Validate a derived/estimated signal against an independent one and record the correlation.

### 3. Diversify the angles across figures (the heart of this skill)
A good section spans **distinct, non-overlapping angles** on its topic — each figure must
answer something the others can't. The right set of angles depends entirely on the area;
find them by asking "what are the genuinely different questions an engineer asks about
this?" Common angle types, to adapt (not a checklist):
- **capability / envelope** — how much can it do (a percentile vs a target/limit).
- **balance / distribution** — how the effort splits across wheels/axles/phases.
- **utilisation** — how close to the limit it runs (achieved ÷ available).
- **quality / fidelity** — for control systems, how well it tracks its own setpoint.
- **consistency** — spread/repeatability across laps, runs or drivers.

If two candidates compute the same underlying quantity, keep one. **Informative
disagreement is valuable** — when two honest figures point different ways because they
measure different things, both earn their place; document the cross-read.

### 4. Audit the SET with reviewing-metrics (design review)
**REQUIRED SUB-SKILL:** invoke **reviewing-metrics** with the Skill tool — one invocation
per candidate figure, plus dimension 6 (coherence) once over the whole set: no duplication,
the plot order tells a story, every plot earns its slot. Applying the 6 dimensions "from
memory" does NOT count — the skill must actually load. Put each metric in its right home —
don't mix *driver* with *car*, or *control-quality* with *vehicle-behavior*, in the same
section.

### 5. CHECKPOINT — show the review to the user before any plan or code
Report the per-figure verdict table (**Mantener / Simplificar / Eliminar**) and the
set-coherence verdict to the user, then STOP and wait for their call on which figures
survive. No spec, no plan, no implementation until they've seen and ruled on the review.

### 6. Spec → plan
Use `superpowers:brainstorming` to write the spec, then `superpowers:writing-plans`,
covering only the figures the user approved. **Never commit — leave the changes in the
working tree; the user makes the commits himself, when he decides.**

### 7. Execute — one new-metric invocation per figure
For each approved figure, invoke **new-metric** with the Skill tool — **one invocation per
figure**, also when *modifying* an existing figure, not only for brand-new ones. It
scaffolds + wires + verifies on real data and ends by re-auditing the *built* figure with
reviewing-metrics, reporting that verdict too. Never implement figures inline "because the
plan already describes them" — the plan fixes *what*; `new-metric` governs *how*.

## Red flags — stop and rethink
- Several figures that are the same quantity dressed differently → one angle, not many.
- A figure you can't explain in ≤2 sentences, or whose name doesn't match its formula.
- A proxy signal standing in for what the figure claims to measure (or one that reads stale/zero here).
- Reaching for a sophisticated composite KPI when a direct channel + a reference line says it plainer.
- Keeping a plot "because the team built it" / "because it already runs" — audit it the same.
- "Running" reviewing-metrics or new-metric *from memory* instead of invoking them with the Skill tool.
- Writing figure code straight from the plan without a `new-metric` invocation per figure.
- Any code written before the user has seen and ruled on the review verdicts (step 5).

## One worked example (illustrative — do not copy the figures)
Dynamics › Braking was rebuilt into five figures, each a *different angle*: decel envelope
(capability), F/R force balance (distribution), longitudinal load transfer (load), per-axle
grip utilisation (utilisation), per-axle slip (slip kinematics). Force said *front-limited*
while slip said *rear nears lock-up* — an informative disagreement that justified keeping
both. The point is the **spread of angles**, not these specific signals; a Powertrain,
Driver or control-system (TV/TC/RB) section will have entirely different angles.
