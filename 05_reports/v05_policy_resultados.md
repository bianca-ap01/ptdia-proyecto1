# V05 - Politica de decision basada en riesgo

## Objetivo

Convertir el score probabilistico del baseline V01 en una decision operativa. Esta etapa no intenta entrenar un modelo nuevo; usa el mejor modelo defendible hasta ahora para responder una pregunta distinta: que hacer con cada transaccion segun su riesgo.

La politica evaluada tiene tres acciones:

| Accion | Regla | Interpretacion |
| --- | --- | --- |
| `approve` | score menor al umbral de revision | La transaccion se aprueba automaticamente. |
| `review` | score entre umbral de revision y umbral de escalamiento | La transaccion consume capacidad de revision manual. |
| `escalate` | score mayor o igual al umbral de escalamiento | La transaccion se bloquea o escala a un proceso de mayor severidad. |

## Por que esta etapa es necesaria

PR-AUC y ROC-AUC miden calidad de ranking, pero no dicen directamente que decision tomar. En fraude, los errores no tienen el mismo costo:

- Aprobar fraude es costoso porque implica perdida economica directa.
- Bloquear o escalar una transaccion legitima tambien cuesta, porque introduce friccion y puede afectar al cliente.
- Revisar manualmente cuesta menos que bloquear, pero esta limitado por capacidad operativa.

Por eso la seleccion final no debe depender solo de una metrica estadistica. Debe traducirse a una politica con costos y restricciones.

## Supuestos usados

Se definio un escenario base normalizado:

| Componente | Costo |
| --- | ---: |
| Fraude aprobado | 100 |
| Transaccion legitima escalada/bloqueada | 10 |
| Revision manual | 2 |
| Capacidad maxima de revision | 5% |

Estos valores no son dinero real. Son ponderaciones relativas para comparar politicas. La relacion elegida refleja el criterio de que dejar pasar fraude debe penalizarse mucho mas que revisar, y que bloquear una transaccion legitima debe ser mas costoso que revisarla.

## Metodo

1. Se tomaron las predicciones de V01 en validacion y holdout.
2. Se probaron pares de umbrales sobre validacion.
3. Se descartaron politicas con mas de 5% de revision en validacion.
4. Se eligio la politica con menor costo esperado por transaccion en validacion.
5. Se aplicaron esos mismos umbrales en holdout sin recalibrar.
6. Se comparo contra politicas simples: aprobar todo, revisar top 5% y escalar top 1%.

Esta forma evita elegir umbrales mirando el holdout. El holdout queda como simulacion temporal de datos futuros.

## Umbrales seleccionados

| Umbral | Valor |
| --- | ---: |
| Revision | 0.459783 |
| Escalamiento | 0.779122 |

Interpretacion: una transaccion con score menor a 0.459783 se aprueba; entre 0.459783 y 0.779122 se revisa; desde 0.779122 se escala.

## Resultado global

| Split | Costo por transaccion | Fraude detectado | Fraude no detectado | Revision | Escalamiento | Aprobacion automatica |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validacion | 1.1395 | 71.83% | 28.17% | 4.98% | 2.35% | 92.67% |
| Holdout | 1.3680 | 66.17% | 33.83% | 5.10% | 2.48% | 92.42% |

El deterioro de validacion a holdout confirma que hay deriva temporal. Aun asi, la politica mantiene una reduccion importante de costo frente a aprobar todo.

## Comparacion contra politicas simples

| Split | Politica | Costo por transaccion | Fraude detectado | Revision | Escalamiento |
| --- | --- | ---: | ---: | ---: | ---: |
| Validacion | Aprobar todo | 3.4341 | 0.00% | 0.00% | 0.00% |
| Validacion | Revisar top 5% | 1.3407 | 63.87% | 5.00% | 0.00% |
| Validacion | Escalar top 1% | 2.5445 | 26.20% | 0.00% | 1.00% |
| Validacion | Politica V05 | 1.1395 | 71.83% | 4.98% | 2.35% |
| Holdout | Aprobar todo | 3.4804 | 0.00% | 0.00% | 0.00% |
| Holdout | Revisar top 5% | 1.5247 | 59.20% | 5.23% | 0.00% |
| Holdout | Escalar top 1% | 2.5959 | 25.75% | 0.00% | 1.01% |
| Holdout | Politica V05 | 1.3680 | 66.17% | 5.10% | 2.48% |

La politica V05 supera a las tres politicas simples en costo por transaccion y deteccion de fraude en ambos splits. Su ventaja viene de separar dos acciones: revision para riesgo alto pero incierto, y escalamiento para riesgo extremo.

## Resultado por segmento UID

| Split | Segmento | Costo por transaccion | Fraude detectado | Revision | Escalamiento |
| --- | --- | ---: | ---: | ---: | ---: |
| Validacion | UID conocido | 0.7493 | 70.69% | 2.32% | 1.14% |
| Validacion | UID desconocido | 1.4732 | 72.34% | 7.26% | 3.39% |
| Holdout | UID conocido | 0.9163 | 55.50% | 1.83% | 0.84% |
| Holdout | UID desconocido | 1.7318 | 69.69% | 7.73% | 3.80% |

El segmento UID desconocido concentra mas riesgo y consume mas revision. Esto coincide con el EDA: los patrones de identidad y recurrencia son informativos, pero no son igual de estables para todas las transacciones.

El segmento UID conocido en holdout muestra una caida marcada de deteccion. Esto no significa que UID conocido sea irrelevante; significa que una politica global puede no estar calibrada de forma optima para segmentos con distribuciones distintas.

## Decision

Se acepta V05 como primera politica operativa documentada, no como solucion final. Es mejor que las politicas simples y convierte el score del modelo en acciones explicables.

La decision tecnica es mantener V01 como modelo base y usar V05 para reportar el impacto operacional.

## Riesgos y controles

| Riesgo | Evidencia | Control propuesto |
| --- | --- | --- |
| Drift temporal | El costo sube de 1.1395 en validacion a 1.3680 en holdout. | Recalibrar umbrales por ventana temporal. |
| Capacidad excedida en holdout | Revision pasa de 4.98% a 5.10%. | Si el limite es estricto, usar percentil diario/semanal en vez de umbral fijo. |
| Diferencia por segmento UID | UID desconocido consume 7.73% de revision en holdout. | Evaluar umbrales por segmento en una V06. |
| Costos asumidos | Los costos son ponderaciones, no valores reales del negocio. | Hacer sensibilidad de costos antes de una recomendacion final. |

## Artefactos

- `p1/04_scripts/v05_policy_decision.py`
- `p1/03_outputs/v05_policy_decision/policy_thresholds.csv`
- `p1/03_outputs/v05_policy_decision/policy_segment_results.csv`
- `p1/03_outputs/v05_policy_decision/policy_baselines.csv`
- `p1/03_outputs/v05_policy_decision/v05_policy_report.md`
- `p1/03_outputs/v05_policy_decision/valid_policy_decisions.csv`
- `p1/03_outputs/v05_policy_decision/holdout_policy_decisions.csv`

## Siguiente paso recomendado

Ejecutar V06 con analisis de sensibilidad y calibracion:

- sensibilidad de costos para demostrar que la recomendacion no depende de una sola combinacion arbitraria;
- calibracion de probabilidades para que el score sea mas interpretable como riesgo;
- politica por segmento UID si mejora costo sin romper la capacidad operacional.
