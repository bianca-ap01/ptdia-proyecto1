# V01b - Full feature baseline temporal

Kernel Kaggle:

`biancaaguinaga/p1-ieee-fraud-v01b-full-baseline`

Estado final: `COMPLETE`

Outputs locales:

`p1/03_outputs/v01b_full_baseline/`

Bitacora general de decisiones:

`p1/01_planning/decision_log_modelado.md`

## Objetivo

Evaluar si el baseline temporal mejora al usar todas las features permitidas, sin reducir el bloque `V`.

Este experimento no se planteo como reemplazo automatico de V01. Su funcion fue responder una pregunta concreta:

> Cuanta senal se pierde al recortar el bloque `V` a 80 columnas en V01?

## Diseno

- Train temporal: primeras 70% filas por `TransactionDT`.
- Validacion: siguiente 15%.
- Holdout futuro: ultimo 15%.
- Modelo: LightGBM binario con early stopping.
- Features finales: 444.
- `TransactionDT`, `DT_day_index` y `DT_week_index` no se usaron como predictores.
- `DT_hour` se mantiene como unico derivado temporal candidato.

## Resultado V01b

| Segmento | Filas | Fraud rate | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Validacion global | 88,581 | 3.43% | 0.5899 | 0.9227 |
| Validacion UID conocido | 40,835 | 2.33% | 0.6078 | 0.9525 |
| Validacion UID desconocido | 47,746 | 4.38% | 0.6052 | 0.9014 |
| Holdout global | 88,581 | 3.48% | 0.5303 | 0.9004 |
| Holdout UID conocido | 39,512 | 1.93% | 0.4475 | 0.9005 |
| Holdout UID desconocido | 49,069 | 4.73% | 0.5636 | 0.8881 |

## Comparacion contra V01

| Segmento | V01 PR-AUC | V01b PR-AUC | Diferencia |
| --- | ---: | ---: | ---: |
| Validacion global | 0.5954 | 0.5899 | -0.0055 |
| Validacion UID conocido | 0.6485 | 0.6078 | -0.0407 |
| Validacion UID desconocido | 0.5942 | 0.6052 | +0.0110 |
| Holdout global | 0.5436 | 0.5303 | -0.0133 |
| Holdout UID conocido | 0.5159 | 0.4475 | -0.0685 |
| Holdout UID desconocido | 0.5632 | 0.5636 | +0.0004 |

## Lectura

El baseline con todas las features no mejora el resultado global ni el holdout futuro. La mejora en UID desconocido es marginal en holdout y no compensa la caida en UID conocido ni en el desempeno global.

La lectura metodologica es que agregar todas las columnas `V` no produce automaticamente una mejor generalizacion temporal. Es probable que algunas columnas adicionales agreguen ruido, redundancia o senal inestable. Esto refuerza la necesidad de una auditoria de features antes de adoptar el bloque completo.

## Decision

V01b queda como experimento de control, pero no reemplaza a V01 como baseline principal.

Se mantiene V01 como referencia inicial porque:

- Tiene mejor PR-AUC global en validacion y holdout.
- Tiene mejor desempeno en UID conocido.
- Usa menos features, por lo que es mas simple y menos costoso.
- La ganancia de V01b en UID desconocido es practicamente nula en holdout.

## Proximo paso

Continuar con V02: features UID historicas y ablations controladas sobre la base V01.

