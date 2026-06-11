---
name: reviewing-metrics
description: "Use when asked to review, audit, sanity-check or critique a CAT17x dashboard metric, KPI, plot or a whole section — or before adding a new one. Triggers on phrases like 'revisa este KPI', 'mira si está bien', 'esto tiene sentido?', 'is this metric sound', or any doubt that a metric is contrived, hard to interpret, or physically dubious."
---

# Reviewing Metrics

## Overview
Audit a CAT17x dashboard metric against 6 dimensions and return a verdict.

**Core principle:** a metric must be physically correct, mean what its name says, be explainable in two sentences, and readable at a glance. **Physical-and-simple beats sophisticated-and-opaque** — if you can't explain it or read it off the plot, cut it.

## When to use
- User asks to review / audit / sanity-check / critique a KPI, plot, or section.
- Before merging or adding a new metric.
- A metric "feels" contrived, confusing, or far-fetched ("cogido por pinzas").

**Not for:** pure render/layout fixes or refactors that don't change a calculation.

## How to run a review
1. **Read the metric's actual code** (the `*_fig` function) and trace the formula end to end. Don't review from the name or caption.
2. **Verify every signal** against the data and project conventions: column names actually exist (check a CSV header), units, sign, sampling (100 Hz, `TimeStamp` in s). `Steering` is radians used **directly** — never ÷3.15; right turns negative. Yaw rate `VN_gz` is rad/s.
3. Walk the **6 dimensions** below; mark each ✅ PASA / ⚠️ DUDOSO / ❌ FALLA with a one-line reason. Always output the table — even when the answer seems obvious.
4. Give a **verdict**. If Simplificar/Eliminar, propose the minimal concrete change (don't just complain).
5. For a whole **section**, review each metric, then apply dimension 6 (coherence) once over the set.
6. **The review is a deliverable for the user, not an internal check.** Put the table and
   verdict in your reply. When the review runs inside another flow (redesigning-a-section,
   new-metric, an implementation plan), the user must see it **before** any implementation
   that depends on it starts — never fold the review silently into the work.

## The 6 dimensions

| # | Dimensión | Pasa si… |
|---|-----------|----------|
| 1 | **Cálculo correcto** | Unidades con sentido físico que casan con el nombre; convenciones de señal respetadas (Steering rad directo, right=neg, 100 Hz); filtro estándar `ensure_complete_laps_df()` aplicado; sin factores cosméticos (que se cancelan o sólo reescalan sin cambiar el ranking); si combina señales, **mismo** pre-proceso y ventana en todas; constantes desde `docs/context/cat17x_parameters.md`, no números mágicos. |
| 2 | **Validez física** | Mide lo que el nombre afirma (no un proxy con el sentido invertido); **no mezcla piloto con coche ni control con vehículo**; se calcula en el régimen donde la señal tiene información (buena relación señal-ruido); la dirección "más = mejor/peor" es la esperada, no al revés. |
| 3 | **Interpretabilidad** | Explicable al equipo en **≤2 frases** con unidades y dirección (¿le escribirías su entrada en `METRIC_HELP`?); no redundante con una métrica más simple ya existente; accionable (te dice qué tocar); honesta sobre qué **NO** captura. |
| 4 | **Legibilidad visual** | Estilo compartido (`make_dark_figure`, `driver_color`, `add_trend_line`/`add_zero_line`); línea de referencia/target donde haya objetivo; ejes con unidades; bins/rangos sensatos; distingue per-lap / per-evento / per-run; no saturado, el insight se lee de un vistazo. |
| 5 | **Robustez estadística** | Muestra suficiente (reporta `samples`/`counts`); resistente a outliers (percentil tipo P95 en vez de `max` crudo); NaN y división-por-~0 guardados (`nanmean`/`nanstd`); no depende de **una sola vuelta de oro**, reproducible run-a-run. |
| 6 | **Coherencia de sección** *(sólo al revisar una sección entera)* | Ninguna métrica **duplica** a otra; el orden de los plots cuenta una historia; cada plot se gana su sitio (si no aporta señal accionable, proponer quitarlo). |

## Verdict
- **Mantener** — todas ✅ (algún ⚠️ menor tolerable).
- **Simplificar** — la señal de fondo es real pero el cálculo o el visual están rebuscados; propón la versión mínima y física.
- **Eliminar** — falla la 1, 2 o 3 de forma que no se arregla sin rehacerla, o es redundante con algo más simple.

## Examples
- ✅ **Patrón bueno — `decel_envelope_fig`** ([src/dynamics.py](src/dynamics.py)): P95 de deceleración por bins de velocidad vs target 1.79 g. Filtro estándar, target trazable a parámetros, percentil robusto, estilo compartido, devuelve `(fig, kpi_dict)`. Imita esto.
- ❌ **Malo (un único caso)** — un KPI de "stability" ya eliminado que dividía el jitter del volante entre el del yaw: unidades de segundos, confundía la actividad del piloto con el comportamiento del coche, y subía cuando el piloto **serraba** el volante (invertido). Falló dims 1–3 → eliminado. Con un ejemplo basta; los criterios se sostienen solos.

## Common mistakes (rationalizations to cut)
| Excuse | Reality |
|--------|---------|
| "It's sophisticated, so it must be good" | Sophistication ≠ insight. Opaque = suspect. |
| "The team built it, so keep it" | Provenance doesn't make the math right. Audit it the same. |
| "More metrics = more insight" | Redundant/confusing metrics dilute the signal. Fewer, clearer wins. |
| "It's already implemented, don't touch it" | Sunk cost. If it fails dims 1–3, propose removal. |
| "It runs without error" | Running ≠ correct. Check units, confounders, direction, sample size. |
| "I'll just trust the name/caption" | Read the formula. Names lie; the code doesn't. |
| "I'll fold the review into the implementation" | The verdict table goes to the user BEFORE code that depends on it. |
