# V06 - Robustez, calibracion y segmentacion de politica

Kernel Kaggle publico:
https://www.kaggle.com/code/biancaaguinaga/p1-ieee-fraud-v06-policy-robustness

## Objetivo

Auditar si la politica V05 depende demasiado de una sola eleccion de costos o de una sola politica global. Esta etapa no busca reemplazar el modelo V01 ni la politica V05 fina; busca responder si la recomendacion es estable bajo escenarios razonables.

Se evaluaron tres preguntas:

1. Que pasa si cambian los costos relativos.
2. Si el score puede interpretarse como probabilidad o solo como ranking.
3. Si usar umbrales separados para UID conocido y UID desconocido mejora la decision.

## Alcance metodologico

V06 usa una grilla compacta de umbrales. Por eso sus numeros no deben leerse como una optimizacion mas fina que V05. La funcion de V06 es diagnostica: comparar estabilidad, trade-offs y riesgos.

## Escenarios de sensibilidad

| Escenario | Interpretacion |
| --- | --- |
| `base` | Mismos costos base de V05. |
| `friction_high` | Penaliza mas escalar/bloquear transacciones legitimas. |
| `fraud_high` | Penaliza mas dejar pasar fraude. |
| `capacity_tight` | Reduce la capacidad de revision a 3%. |
| `capacity_loose` | Aumenta la capacidad de revision a 10%. |

## Resultados globales en holdout

| Escenario | Costo por transaccion | Fraude detectado | Revision | Escalamiento | Aprobacion automatica |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 1.3809 | 68.80% | 5.07% | 3.78% | 91.15% |
| friction_high | 1.5068 | 63.12% | 4.52% | 1.96% | 93.53% |
| fraud_high | 2.3357 | 75.15% | 4.99% | 7.34% | 87.67% |
| capacity_tight | 1.5100 | 63.93% | 3.10% | 3.77% | 93.13% |
| capacity_loose | 1.1695 | 75.35% | 9.63% | 2.87% | 87.50% |

La respuesta es coherente:

- Si escalar legitimas se vuelve mas costoso, el modelo escala menos y detecta menos fraude.
- Si dejar pasar fraude se vuelve mas costoso, el modelo escala mas y detecta mas fraude.
- Si se permite mas revision, el costo baja y la deteccion sube.
- Si se restringe la revision, sube el costo y baja la deteccion.

Esto apoya que la politica esta actuando segun los incentivos definidos, no por un umbral arbitrario.

## Politica global vs politica segmentada por UID

En el escenario base sobre holdout:

| Politica | Costo por transaccion | Fraude detectado | Fraude no detectado | Revision | Escalamiento | Aprobacion automatica |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Global | 1.3809 | 68.80% | 31.20% | 5.07% | 3.78% | 91.15% |
| Segmentada UID | 1.3724 | 66.56% | 33.44% | 4.78% | 2.82% | 92.41% |

La politica segmentada reduce levemente el costo por transaccion, pero tambien detecta menos fraude. La mejora de costo viene principalmente de menor intervencion operativa, no de mejor captura de fraude.

Decision: no adoptar todavia umbrales segmentados como politica principal. Mantenerlos como candidato si el objetivo del negocio prioriza reducir friccion o capacidad por encima de capturar mas fraude.

## Calibracion del score

| Split | Score | Brier | Score promedio | Tasa real de fraude |
| --- | --- | ---: | ---: | ---: |
| Validacion | Bruto | 0.0440 | 14.03% | 3.43% |
| Validacion | Isotonico | 0.0200 | 3.43% | 3.43% |
| Holdout | Bruto | 0.0479 | 14.30% | 3.48% |
| Holdout | Isotonico | 0.0219 | 3.55% | 3.48% |

El score bruto ordena bien el riesgo, pero esta mal calibrado como probabilidad: en promedio predice alrededor de 14% de riesgo cuando la tasa real ronda 3.5%. Despues de calibracion isotonica entrenada en validacion, el score promedio queda mucho mas cerca de la tasa real y el Brier mejora en holdout.

Decision: para ranking y umbrales se puede seguir usando el score bruto; para comunicar riesgo como probabilidad o estimar perdida esperada, se debe usar calibracion.

## Decisiones tomadas

| Decision | Sustento |
| --- | --- |
| Mantener V01 como score base | Sigue siendo el mejor modelo general documentado. |
| Mantener V05 como primera politica operativa | V05 tiene busqueda de umbrales mas fina y supera politicas simples. |
| No promover aun la politica segmentada UID | Baja un poco el costo, pero captura menos fraude; requiere decidir si se prioriza menor friccion o mayor deteccion. |
| Incorporar calibracion para interpretacion probabilistica | El score bruto esta sobreestimado; la calibracion reduce Brier y alinea score promedio con tasa real. |
| Hacer sensibilidad de costos en la entrega | Demuestra que las decisiones cambian de forma coherente ante escenarios alternativos. |

## Riesgos y controles

| Riesgo | Control |
| --- | --- |
| Confundir score con probabilidad | Reportar calibracion antes de hablar de riesgo porcentual. |
| Elegir politica por un solo escenario | Presentar sensibilidad de costos. |
| Sobreoptimizar umbrales con holdout | Los umbrales se eligen en validacion y se reportan en holdout. |
| Adoptar segmentacion por aparente mejora pequena | Exigir que la mejora sea consistente con el objetivo del negocio. |

## Artefactos

- `p1/02_kaggle_kernels/v06_policy_robustness/main.py`
- `p1/04_scripts/v06_policy_robustness.py`
- `p1/03_outputs/v06_policy_robustness/global_policy_sensitivity.csv`
- `p1/03_outputs/v06_policy_robustness/segmented_uid_policy_sensitivity.csv`
- `p1/03_outputs/v06_policy_robustness/policy_robustness_combined.csv`
- `p1/03_outputs/v06_policy_robustness/calibration_summary.csv`
- `p1/03_outputs/v06_policy_robustness/calibration_bins.csv`
- `p1/03_outputs/v06_policy_robustness/v06_policy_robustness_report.md`
- `p1/03_outputs/v06_policy_robustness_kaggle/`

## Proxima etapa

Preparar la version consolidada de modelado para entrega: narrativa del pipeline, tabla comparativa V01-V06, decision final recomendada y limitaciones.
