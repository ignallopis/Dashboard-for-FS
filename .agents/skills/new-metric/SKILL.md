---
name: new-metric
description: "Use when adding, modifying or wiring a metric/KPI/figure in the CAT17x Streamlit dashboard, including each figure of a section redesign. Triggers on 'añade una métrica', 'new KPI', 'add a chart/figure to the dashboard', 'monta este plot', 'cambia esta figura'."
---

# New Metric

Scaffold a CAT17x dashboard metric end-to-end with project conventions, then audit it.
**Physical-and-simple beats sophisticated-and-opaque** — if you can't explain it in two sentences, don't add it.

## 1. Before writing
- **Scope check:** if you're really rebuilding a whole sub-section (deciding *which*
  figures it should have, not adding one to a healthy section), stop and invoke the
  **redesigning-a-section** skill (Skill tool) first — it diversifies the figures and
  audits the set for coherence, then hands each approved figure back here. This skill
  builds ONE figure.
- Confirm the question it answers and the granularity (per-sample / per-lap /
  per-event / per-run). If unclear, ask.
- Check it isn't redundant with an existing figure (grep the target module).
- Verify every signal with the **signal-lookup** skill: `Steering` rad used **directly**
  (never ÷3.15), right turns negative, booleans `== 1.0`, 100 Hz, `TimeStamp` in s.

## 2. Write the figure (in the matching src/ module)
- One function = one metric. Module by area: `dynamics` / `powertrain` / `gripfactor` /
  `cornering` / `driver` / `tc` / `tv` / `rb`.
- Signature: `def <name>_fig(dfs: dict[str, pl.DataFrame]) -> tuple[go.Figure, dict]`
  (single-df variants are fine where the section already uses them). Return
  `(figure, kpis)`; **never call `st.*` here** — `dashboard.py` is the only renderer.
- Filter with `utils.load_data` output / `ensure_complete_laps_df` (standard filter
  `laps > 0 AND laptime.is_not_nan()`; drop lap 0 and the last lap).
- Styling: build via `utils.make_dark_figure` / `utils.apply_dark_layout`; per-run colour
  via `utils.driver_color` (**never** a per-module palette). Semantic palettes
  (`WHEEL_COLORS`, good/warn/bad) stay separate. Add a target/reference line where a
  target exists. Constants from `docs/context/cat17x_parameters.md`, not magic numbers.
- Robustness: report `samples`/`counts`; prefer P95 over raw `max`; guard NaN and
  div-by-~0 (`nanmean` / `nanstd`).

## 3. Wire into the dashboard
- Call it in the right tab/sub-section. 1 run → `st.metric` cards; ≥2 runs → comparison
  table via `_render_summary_df` / `_show_summary_table`.
- Add the metric's one-line explanation to `METRIC_HELP` (tooltip).

## 4. Verify
1. `./.venv/bin/python -c "import ast,pathlib; ast.parse(pathlib.Path('src/<mod>.py').read_text())"`
2. `./.venv/bin/python -c "from src.<mod> import <name>_fig"`
3. `fig, kpis = <name>_fig(dfs)` on `data/Cerpa_FSG.csv` + `data/Martinez_FSG.csv`;
   sanity-check KPI ranges and that nothing is NaN.
4. `./.venv/bin/streamlit run src/dashboard.py` → open the tab → check layout/legend.

## 5. Audit (required — not from memory)
**REQUIRED SUB-SKILL:** invoke **reviewing-metrics** with the Skill tool on the built
metric — one invocation per figure; never batch several figures into one review, never
apply its 6 dimensions from memory. **Report its table + verdict to the user** in your
reply. Not done until it returns **Mantener** (or you've applied its *Simplificar* and
re-audited). Record anything durable in `.claude/knowledge.md`.
