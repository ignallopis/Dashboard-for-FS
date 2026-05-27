# Motorsport Dashboard Review - CAT17x

Revision hecha desde la perspectiva de un ingeniero de pista que recibe este
dashboard para analizar datos de un Formula Student 4WD electrico con TV, TC y
regen-only braking. El foco no es "si hay muchas graficas", sino si el software
ayuda a decidir rapido: donde se pierde tiempo, si el coche esta limitado por
piloto/setup/control/energia, y si las conclusiones son fisicamente fiables.

Datos contrastados en esta revision:

- CSVs disponibles: `Cerpa_FSG.csv`, `Martinez_FSG.csv`.
- Ambos runs cargan 9 vueltas validas con `utils.load_data(..., complete_laps_only=False)`.
- Canales reales presentes: SR/SA/Fx/Fy/Fz por rueda, torques objetivo/actuales,
  TV/TC/RB/PC channels, VN GPS/IMU, dampers, SoC, Vbat/Vmin/Tmax/Tavg.
- Algunas funciones potentes existen en codigo pero no son accesibles desde la UI
  actual: `powertrain.energy_budget_per_lap_fig`, `powertrain.energy_budget_breakdown_fig`,
  `powertrain.pc_master_attribution_figs_kpis`, `tv.yaw_rate_triple_fig`,
  `_render_tc_control_impact`, `_render_tv_function_check`,
  `_render_tv_control_attribution`, `_render_driver_circuit_map_section`,
  `_render_driver_cornering_section`.

## P0 - Riesgos de conclusion incorrecta

### P0.1 La ultima vuelta entra por defecto en los calculos

Regla del proyecto: no usar vuelta 0 ni la ultima vuelta. La UI excluye vuelta 0,
pero el selector de vueltas carga por defecto todas las vueltas disponibles. Al
pasar por `select_laps_df`, se anade el marcador de seleccion explicita y los
modulos downstream ya no aplican su filtro automatico de "remove last lap".

Impacto motorsport:

- Si la ultima vuelta es in-lap, cooldown, lap incompleta o contiene artefactos
  de salida/meta, contamina energia por vuelta, tiempos medios, grip factors,
  estabilidad de frenada, evolucion termica y comparaciones de piloto.
- Un ingeniero podria atribuir una caida de rendimiento a bateria/neumatico/setup
  cuando realmente es seleccion de vuelta.

Accion recomendada:

- Cambiar el default del sidebar a `available_laps[:-1]`, salvo que solo haya
  una vuelta util.
- Mostrar una opcion explicita `Include last lap (advanced)` con warning.
- En tablas/figuras, marcar si una vuelta seleccionada es la ultima del CSV.

### P0.2 El editor manual de meta puede sobrescribir todos los CSV desde una pagina de analisis

En Lap Analysis existe un editor de finish line con botones `Apply To All CSVs`
y `Restore Auto`. Es una accion destructiva sobre todos los CSVs de `data/`,
no solo los runs seleccionados.

Impacto motorsport:

- Desde una pantalla de analisis de performance, un click puede reescribir
  `laps` y `laptime` de toda la base local.
- Si se cambia la meta para comparar una curva concreta, se puede invalidar
  silenciosamente la comparabilidad historica de runs anteriores.

Accion recomendada:

- Mover la herramienta a una seccion `Data / Lap Detection`.
- Requerir confirmacion y listar explicitamente los CSVs afectados.
- Por defecto aplicar solo a runs seleccionados, no a todo `data/`.
- Guardar backup o escribir version nueva del CSV antes de sobrescribir.

### P0.3 Temperaturas y voltajes usan extremos no filtrados y pueden ser fisicamente imposibles

Con los datos actuales, `Tmax` llega a cientos de grados (`Cerpa_FSG.csv` > 400,
`Martinez_FSG.csv` > 300) y `Tavg` llega aun mas alto. El dashboard presenta
`Peak battery Tmax` en grados C sin una capa de plausibilidad. Tambien `Vmin`
tiene outliers grandes y `Battery Status` usa minimos/peaks que pueden estar
dominados por glitches.

Impacto motorsport:

- Un pico falso de temperatura de bateria puede provocar una decision equivocada
  de parar el coche, bajar potencia o redisenar refrigeracion.
- Un minimo absoluto de tension puede parecer un problema de pack cuando es un
  dropout de sensor.

Accion recomendada:

- Introducir validacion de rango por canal antes de KPIs: por ejemplo bateria
  temp plausible, cell voltage plausible, pack voltage plausible.
- Usar P95/P99 o eventos sostenidos en vez de max/min absolutos para KPIs
  principales.
- Si se detectan outliers imposibles, mostrar banner: "channel not trusted",
  no mezclarlo con metricas fiables.

### P0.4 Hay analisis clave implementados pero no accesibles desde la UI

Ejemplos verificados:

- `_render_tc_control_impact()` existe, pero `_tab_tc()` solo llama a
  `_render_tc_function_check()`.
- `_render_tv_function_check()` y `_render_tv_control_attribution()` existen,
  pero `_tab_tv()` no los llama.
- `_render_driver_circuit_map_section()` esta definida, pero `_tab_driver()`
  no la llama.
- `_render_driver_cornering_section()` esta definida, pero no aparece en las
  opciones de `Driver section`.
- `energy_budget_per_lap_fig`, `energy_budget_breakdown_fig`,
  `pc_master_attribution_figs_kpis` y `yaw_rate_triple_fig` existen, pero no
  estan expuestas en dashboard.

Impacto motorsport:

- El usuario ve una version incompleta del software y puede pensar que no hay
  trazabilidad TC/TV/PC o presupuesto energetico.
- Se pierde diagnostico de alto valor: si TC corrige overslip, si TV realmente
  rota el coche, si PC/Master esta limitando aceleracion, si el presupuesto de
  energia cuadra con lo medido.

Accion recomendada:

- Anadir una seccion `Advanced diagnostics` por sistema o exponer esos bloques
  detras de toggles.
- Prioridad alta: `TC behaviour`, `TV observable balance`, `PC/Master attribution`,
  `Energy Budget`, `Circuit Map`.

## P1 - Faltan metricas clave para un ingeniero de pista

### P1.1 Falta una pantalla "Run Summary" orientada a decision

Ahora el dashboard empieza por `Driver -> Lap Analysis`, que es potente pero
requiere elegir dos vueltas. Un ingeniero normalmente necesita primero una vista
global: mejor vuelta, stint trend, energia, limitaciones, alertas y principal
perdida de tiempo.

Falta:

- Mejor vuelta por run y delta contra referencia.
- Ranking de runs por fastest, average, consistency, energy/lap, recovered/lap.
- Alertas de calidad de datos: ultima vuelta incluida, canales no plausibles,
  calibracion de pot no validada, enable flags no fiables.
- Top 3 oportunidades: curvas donde se pierde mas tiempo, braking zones con
  early/late brake, zonas con TC/RB/TV fuera de objetivo.

Accion recomendada:

- Crear `Overview` como primera seccion.
- Mostrar 6-8 KPIs maximos y un panel de alertas.
- Desde cada alerta, link mental o boton a la seccion relevante.

### P1.2 Falta "where is the lap time?" como workflow principal, no solo por corner

Lap Analysis ya tiene comparacion A/B, delta y fases por curva. Es lo mas cercano
a una herramienta de pista real. Pero falta convertirlo en flujo principal de
respuesta:

- Ranking automatico de sectores/curvas por delta total.
- Separacion clara de perdida por braking, entry, apex, exit, straight.
- Tabla por curva con `delta`, `min speed`, `apex speed`, `exit speed`,
  `brake point`, `throttle point`, `coast time`, `steering peak`.
- Identificacion de "driver lost" vs "car/control limited": por ejemplo throttle
  100 % + PC active, overslip + TC active, brake + lockup risk, TV understeer.

Accion recomendada:

- Hacer que `Where is the time?` sea el bloque principal debajo de los selectores
  de A/B.
- Mantener el detalle de curva como drill-down.

### P1.3 Falta performance por energia: tiempo por kWh y energia por fase

Powertrain tiene energia por vuelta y existe Energy Budget, pero no esta expuesto
ni conectado con rendimiento. En Formula Student, especialmente endurance, no
vale solo "mas rapido": importa eficiencia energetica.

Falta:

- `lap time vs net energy`, con cuadrantes: fast/efficient, fast/expensive,
  slow/efficient, slow/expensive.
- `Wh per lap`, `Wh per km`, `Wh per second`, `Wh recovered / Wh consumed`.
- Energia por fase: accel, braking regen, corner/coast.
- Energia por curva o sector para ver si el piloto gana tiempo gastando demasiado.
- Modelo vs medido: usar `energy_budget_per_lap_fig` y breakdown como pantalla.

Accion recomendada:

- Exponer `Energy Budget per Lap` y `Energy Budget Breakdown`.
- Cambiar `Powertrain -> Energy per Lap` para explicar si menor energia es bueno
  o simplemente vuelta lenta.

### P1.4 Falta trazabilidad PC/Master en Powertrain

El dashboard muestra funcion PC basica: si `P_bat` esta bajo 80 kW. Pero existe
`pc_master_attribution_figs_kpis`, que diagnostica fidelidad de pedal, cap de
potencia y perdida de aceleracion.

Impacto motorsport:

- Si el piloto pide full throttle pero el coche no acelera, hay que saber si
  es grip, power cap, torque master, TC o motor/inverter.
- En los datos actuales el peak Pbat esta lejos de 80 kW; esto es informacion
  critica para entender si el coche esta limitado por potencia disponible o por
  otra cosa.

Accion recomendada:

- Anadir bloque `Powertrain -> Power Limit / Master`.
- Mostrar: % WOT con PC active, peak Pbat, near-cap at WOT, ax loss when PC cut,
  pedal vs torque correlation, actual-master torque MAE.

### P1.5 TC necesita pasar de "esta cerca de SR target" a "cuanto tiempo cuesta o salva"

TC function check es bueno para validar SR respecto a +0.20. Pero para pista
faltan metricas accionables:

- Tiempo con TC interviniendo por vuelta/curva.
- Perdida de aceleracion durante intervencion.
- Comparacion exit speed y delta time en curvas con/sin TC.
- Eventos de overslip ordenados por severidad y posicion en pista.
- Diferencia por rueda: cual satura primero y si es setup, diferencial virtual
  o mapa de torque.

Accion recomendada:

- Exponer `_render_tc_control_impact()`.
- En TC, anadir track map de eventos: overslip, cut active, recovery time.
- Reducir plots agregados per-lap si no llevan a decision directa.

### P1.6 TV necesita explicar si mejora balance o solo sigue una referencia interna

La UI principal de TV muestra KPIs de yaw/Mz tracking y corner balance. Pero
los bloques de function check y attribution no se llaman. Falta tambien una
vista que compare yaw real, yaw desired y yaw steering-implied.

Impacto motorsport:

- Un bajo error contra referencia interna no garantiza que el coche gire mejor.
- Un ingeniero necesita ver si el piloto esta en understeer, si TV ayuda a
  rotar, y si genera oversteer o yaw disturbance.

Accion recomendada:

- Exponer `tv_control_attribution_figs_kpis()` y `yaw_rate_triple_fig()`.
- Mostrar por curva: median balance %, yaw gain, Mz actual/requested, entry
  rotation, speed delta.
- Renombrar "Function check - is TV adding yaw moment so the car turns?" a algo
  mas tecnico: "TV effectiveness - yaw gain and balance response".

### P1.7 RB mezcla demasiados KPIs y necesita jerarquia de frenada

RB muestra 19 KPIs en una sola cabecera. Hay buena informacion, pero no esta
jerarquizada. Para pista, el flujo debe ser:

- Frena lo suficiente: decel peak/p95, brake point, stopping distance.
- Es estable: lockup, yaw disturbance, beta, pitch.
- Recupera energia: Wh, efficiency, coverage.
- Sigue objetivo: SR target -0.20, current target, torque/Fz bias.

Accion recomendada:

- Dividir RB en cuatro grupos visuales: `Decel`, `Stability`, `Energy`, `Control`.
- Mantener tablas de eventos, pero anadir ranking de braking zones por perdida
  y por regen opportunity.
- Aclarar en captions que no hay hydraulic braking real; cualquier canal `HydTrq`
  debe tratarse como demanda/modelo si no hay sensor validado.

### P1.8 Dynamics Cornering es bueno pero demasiado fragmentado

Dynamics tiene las mejores captions y explicaciones tecnicas. Aun asi, la
seccion `Cornering` mezcla understeer angle, steering-vs-ay, LTD y legacy tyre
workload. Falta una lectura unificada: "el coche es subvirador/sobrevirador,
por que, y en que curvas".

Accion recomendada:

- Convertir `Understeer`, `Steering vs ay`, `LTD`, `SA balance`, `Body beta`,
  `Friction circle` en un mismo diagnostico de balance.
- Sacar `SA balance` y `Friction circle` de legacy si los canales Est_SA/FX/FY/FZ
  son fiables; son metricas utiles para setup.
- Mantener "Legacy" solo para metricas no calibradas o redundantes.

### P1.9 Driver tabs basicas no conectan con el mapa ni con la perdida de tiempo

Throttle, Brake y Steering tienen metricas por vuelta, pero muchas son agregadas.
En pista, un agregado de `median |dTP/dt|` o `mean brake aggressiveness` es
menos util que saber donde ocurre y si gana/pierde tiempo.

Accion recomendada:

- Para cada tab Driver, anadir top events por posicion:
  - Throttle: earliest throttle per corner, apex-to-throttle distance, exit
    throttle ramp and exit speed.
  - Brake: brake point, release point, brake duration, coasting before brake.
  - Steering: steering peak, steering rate, corrections after apex.
- Volver accesible `Circuit Map` o integrarlo arriba de Driver.

## P2 - Visualizacion, coherencia y distribucion

### P2.1 Captions incoherentes entre secciones

Dynamics tiene captions especificas para casi todo: braking model, decel envelope,
Rouelle stability, traction model, setup calibration, LTD. En cambio:

- Powertrain no explica debajo de Energy, Power Distribution, Battery Status ni
  Thermal Evolution que significan las metricas ni como leerlas.
- TC function check no explica el criterio de target band, condiciones de armado
  ni que significa "Under target".
- TV KPIs no explica unidades de yaw/Mz ni si el objetivo viene del controlador.
- RB tiene muchos KPIs sin una guia de lectura.
- Driver Throttle/Steering no tienen captions; Brake Application Point si.

Accion recomendada:

- Regla UI: cada `subheader` con figura debe tener una caption corta si el
  grafico no es autoexplicativo.
- Caption template: `What it is`, `How to read it`, `When not to trust it`.
- No poner captions largas de paper si no llevan a accion inmediata; mover
  teoria larga a expander `Method`.

### P2.2 Orden de secciones no sigue el flujo de un ingeniero de pista

Orden actual principal: Driver, Dynamics, Powertrain, TC, TV, RB. Dentro de
Driver empieza por Lap Analysis.

Problema:

- Antes de comparar curvas, el usuario necesita saber que runs/vueltas son
  validos, que vuelta es benchmark, y que sistemas estan activos.
- Sistemas de control estan separados de las zonas donde impactan el tiempo.

Accion recomendada:

- Orden propuesto: `Overview`, `Lap Analysis`, `Driver`, `Dynamics`, `Powertrain`,
  `Controls`.
- Alternativa minima: dejar secciones actuales, pero hacer Driver Overview
  accesible antes de Lap Analysis.

### P2.3 Demasiados KPIs planos en algunas pantallas

Powertrain Energy tiene 12 KPIs; RB tiene 19. Cuando todo tiene el mismo peso,
nada tiene peso.

Accion recomendada:

- Maximo 4 KPIs headline por bloque.
- El resto a tabla/expander.
- Agrupar visualmente por pregunta, no por disponibilidad de variable.

Ejemplo RB:

- Headline: `Mean decel`, `Recovered Wh`, `Lockup time`, `SR in target`.
- Advanced: gains, delays, oscillation, Fz bias, pitch, beta.

### P2.4 Mezcla de idiomas

El dashboard mezcla ingles y espanol:

- "Curvas detectadas..." aparece en espanol.
- La mayoria de headers/captions estan en ingles.
- Botones como `Apply To All CSVs` conviven con textos en espanol.

Impacto:

- No rompe el analisis, pero reduce sensacion profesional y consistencia.

Accion recomendada:

- Elegir un idioma unico para UI. Recomendaria ingles si el codigo y nombres
  de canales ya estan en ingles.
- Mantener comentarios internos en ingles como ya marca el proyecto.

### P2.5 Nombres de metricas poco accionables o ambiguos

Ejemplos:

- `Coefficient of Variation` en energia no dice si se refiere a net energy.
- `FB / FF ratio` y `FB share` necesitan contexto de TV.
- `Yaw gain` necesita definicion visual cerca del KPI.
- `Brake -> decel gain` y `Torque -> decel gain` son utiles pero deberian estar
  en grupo `Control model`, no headline principal.
- `Plausability` aparece escrito asi en tablas/video; deberia ser `Plausibility`.

Accion recomendada:

- Nombres mas especificos: `Net energy CV`, `TV feedback/feedforward ratio`,
  `Brake pedal to decel gain`, `Regen torque to decel gain`.
- Unificar spelling de `Plausibility`.

### P2.6 Multi-run overlays pueden saturar la lectura

El overlay multi-run es tecnicamente bueno, pero para muchas figuras por rueda
puede volverse dificil: 2 runs x 4 ruedas x varias trazas genera demasiada
leyenda y color/dash.

Accion recomendada:

- En multi-run, ofrecer modo `Overlay` vs `Small multiples`.
- Para wheel traces, permitir filtrar eje/rueda.
- En graficas de distribucion, priorizar summary bands por run sobre todas las
  muestras coloreadas.

### P2.7 No hay capa clara de calidad de dato

Hay warnings puntuales, pero falta un panel persistente:

- GPS/lap detection confidence.
- Canales con rango imposible.
- Calibracion de damper/pot.
- Enable flags no fiables.
- Distance source usada: `dist_km` o GPS/integracion.

Accion recomendada:

- Crear `Data Quality` en sidebar o top banner.
- Mostrar estado por run con semaforo.
- Cualquier KPI derivado de un canal no validado debe tener badge `untrusted`.

## P3 - Limpieza y simplificacion

### P3.1 Mover legacy de verdad a advanced

Los expanders `Legacy` son utiles para desarrollo, pero un usuario de pista no
deberia preguntarse si un grafico legacy es fiable o no.

Accion recomendada:

- Renombrar a `Advanced tyre diagnostics` si son validos.
- Si no son validados, mantenerlos en `Experimental`.

### P3.2 Reducir duplicacion entre Dynamics y Driver cornering

Hay analisis de curvas en Driver Lap Analysis, Driver Cornering Section
inaccesible, Dynamics Cornering y TV Corner Balance. Esto puede ser potente,
pero ahora no queda claro cual es la fuente de verdad para curvas.

Accion recomendada:

- Definir una unica geometria de curva compartida y mostrar su configuracion en
  todas las secciones que la usan.
- Unificar nomenclatura: usar `Turn` o `Curve`, no ambos mezclados.

### P3.3 Convertir figuras existentes no expuestas en backlog cerrado

Backlog directo:

- Exponer `Energy Budget per Lap`.
- Exponer `Energy Budget Breakdown` con selector run/lap.
- Exponer `PC/Master attribution`.
- Exponer `TC behaviour`.
- Exponer `TV function check`, `TV attribution`, `Yaw-rate triple`.
- Exponer `Circuit Map`.
- Revisar si `Driver Cornering Analysis` debe existir o fusionarse con `Lap Analysis`.

## Evaluacion por area

### Driver

Lo bueno:

- `Lap Analysis` es el centro correcto para comparar vueltas.
- El detalle de curva con delta, throttle, brake, steering, ay y GG es muy util.
- La vuelta potencial por sectores es una idea fuerte para pista.
- Video Analysis puede ser diferencial si hay onboard sincronizado.

Falta:

- Overview inicial de runs/vueltas.
- Circuit Map accesible.
- Top opportunities automaticas.
- Driver controls conectados a delta time, no solo agregados por vuelta.

Sobra o debe bajar prioridad:

- Histograma de throttle como grafico principal; util, pero menos accionable que
  throttle point por curva.
- Steering smoothness agregado sin mapa/curva; necesita contexto.

### Dynamics

Lo bueno:

- Es la seccion con mejor contexto y captions.
- Braking/Acceleration envelopes tienen referencias fisicas.
- Setup distingue calibracion validada/parcial/fallida.
- LTD y steering-vs-ay son relevantes para setup.

Falta:

- Integrar friction circle, SA balance y beta en una lectura de balance.
- Track/GG global interactivo visible como herramienta principal de chassis.
- Data-quality badges para pot/damper/Fz.

Sobra o debe moverse:

- Aero heave si no hay calibracion o parametros de heave; ahora puede confundir.
- Spring velocity cuando T1 calibration no esta validada.

### Powertrain

Lo bueno:

- Energia, potencia por rueda, bateria y termicas cubren la base.
- PC function check ya mira 80 kW cap.

Falta:

- Energy Budget visible.
- Eficiencia energetica contra tiempo.
- PC/Master attribution visible.
- Eventos de thermal derating, IxT/load, inverter/motor status, voltage sag
  sostenido.

Riesgo:

- Peaks de `Tmax/Tavg` y minimos de tension sin plausibilidad.

### TC

Lo bueno:

- La metrica contra SR target +0.20 es correcta para el objetivo del proyecto.
- La funcion detecta claramente cuando el coche esta under target.

Falta:

- Impacto en tiempo y aceleracion.
- Mapa de eventos por posicion.
- Exponer `_render_tc_control_impact()`.

Sobra o debe bajar:

- Varias graficas per-lap de MAE/bias son utiles para controls, pero para pista
  deben quedar detras de resumen/event map.

### TV

Lo bueno:

- Usa yaw/Mz y balance por curva, no solo torque split.
- La idea de corner balance conecta con comportamiento del coche.

Falta:

- Exponer function check y control attribution.
- Exponer yaw real vs desired vs steering-implied.
- Interpretacion por curva: TV mejora o empeora entrada/apex/salida.

Riesgo:

- Tracking de referencia interna puede parecer "bueno" aunque el coche siga
  subvirando.

### RB

Lo bueno:

- Mucho contenido relevante: SR, lockup, energia, balance, yaw/beta/pitch.
- Considera que `RB_Enable` puede no ser fiable y usa inferencia.

Falta:

- Jerarquia visual.
- Ranking por braking zone.
- Separar seguridad/estabilidad de eficiencia energetica.

Sobra o debe bajar:

- 19 KPIs al mismo nivel.
- Gains y delays como headline principal.

### Video Analysis

Lo bueno:

- Sincronizar video, mapa y telemetria es muy valioso para driver coaching.
- Comparar con otra vuelta es la direccion correcta.

Falta:

- Estado claro cuando no hay video y que run/lap se esta sincronizando.
- Boton o puente desde Lap Analysis: abrir video directamente en la curva o
  distancia seleccionada.
- Configurar senales por perfil: driver, controls, powertrain.

## Prioridad de implementacion

### Si solo hay 1 dia

- Excluir ultima vuelta por defecto.
- Anadir banner de calidad de dato para ultima vuelta, pot calibration y outliers
  de `Tmax/Tavg/Vmin`.
- Exponer `TC behaviour`, `TV attribution`, `PC/Master attribution` y `Circuit Map`
  si no requieren cambios grandes.
- Agrupar RB headline a 4 KPIs y mover resto a advanced.

### Si hay 1 semana

- Crear `Overview` con fastest/average/consistency/energy/alerts/top losses.
- Exponer `Energy Budget` y conectar energia con lap time.
- Reorganizar Driver para que `Where is the time?` sea el flujo principal.
- Anadir event maps para TC, RB y PC.
- Unificar captions con template corto por bloque.

### Si hay 1 mes

- Crear una capa comun de `Data Quality` por canal y run.
- Unificar geometria de curvas para Driver, Dynamics y TV.
- Convertir control diagnostics en analisis por curva: TC/RB/TV/PC contribution
  a delta time.
- Anadir small multiples para multi-run y filtros por rueda/sistema.
- Preparar export de reporte por session: seleccion de runs, vueltas, alerts,
  top findings y figuras clave.

## Conclusiones

El dashboard ya contiene mucho mas analisis del que se ve en la UI. La base es
buena: carga robusta, cache, dark theme, comparacion A/B, modelos fisicos y
funciones de control por sistema. El problema principal no es falta de codigo,
sino priorizacion y trazabilidad de decisiones.

Desde pista, el software deberia contestar en este orden:

1. Son fiables estos datos?
2. Que vuelta/run es referencia?
3. Donde se pierde o gana tiempo?
4. Es piloto, setup, energia o control?
5. Que cambio hago antes de la siguiente salida?

Hoy contesta partes de esas preguntas, pero obliga al usuario a navegar por
demasiadas secciones, algunos diagnosticos clave no estan expuestos, y algunas
metricas pueden ser fisicamente enganosas si no se filtran outliers o vueltas no
validas.
