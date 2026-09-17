# V04 - Comparacion de modelos y feature sets auditados

Kernel Kaggle:

`biancaaguinaga/p1-ieee-fraud-v04-model-comparison`

Estado final: `COMPLETE`

Outputs locales:

`p1/03_outputs/v04_model_comparison/`

Bitacora general de decisiones:

`p1/01_planning/decision_log_modelado.md`

## Objetivo

Comparar modelos y feature sets auditados para decidir si conviene pasar de V01 a una version mas reducida y estable.

Se probaron:

- `lgbm_keep`: LightGBM con 44 features `keep`.
- `lgbm_keep_monitor`: LightGBM con 44 `keep` + 9 `monitor`.
- `xgb_keep_monitor`: XGBoost con 53 features `keep + monitor`.
- `cat_keep_monitor`: CatBoost con 53 features `keep + monitor`.

## Resultados globales

| Modelo | Valid PR-AUC | Holdout PR-AUC | Holdout ROC-AUC |
| --- | ---: | ---: | ---: |
| V01 baseline | 0.5954 | 0.5436 | 0.9051 |
| lgbm_keep | 0.5474 | 0.5002 | 0.9001 |
| lgbm_keep_monitor | 0.5739 | 0.5131 | 0.8996 |
| xgb_keep_monitor | 0.5504 | 0.5128 | 0.9055 |
| cat_keep_monitor | 0.5383 | 0.4989 | 0.9020 |

## Resultados por UID en holdout

Nota: V04 usa la definicion conservadora de UID conocido respecto a la ventana de entrenamiento.

| Modelo | UID conocido PR-AUC | UID desconocido PR-AUC |
| --- | ---: | ---: |
| V01 baseline comparable | 0.5946 | 0.5458 |
| lgbm_keep | 0.4130 | 0.5278 |
| lgbm_keep_monitor | 0.5425 | 0.5181 |
| xgb_keep_monitor | 0.4923 | 0.5296 |
| cat_keep_monitor | 0.4148 | 0.5218 |

## Lectura

La reduccion agresiva a `keep` empeora el desempeno. Agregar `monitor` mejora frente a `keep`, especialmente en UID conocido, pero no alcanza al baseline V01.

Esto indica que algunas features clasificadas como `low_priority` o no incluidas en el nucleo auditado pueden aportar senal combinada, aunque individualmente no parezcan fuertes. En modelos de boosting, muchas senales pequenas pueden sumar por interacciones. Por eso V03 debe usarse para monitorear y entender riesgos, no para eliminar features de forma agresiva en este momento.

XGBoost y CatBoost no superan a LightGBM bajo este feature set auditado. XGBoost queda parecido a LightGBM en holdout global y algo mejor en UID desconocido que `lgbm_keep_monitor`, pero sigue por debajo de V01.

## Decision

No se reemplaza V01 por ningun modelo de V04.

V01 sigue siendo la referencia predictiva principal porque:

- Tiene mejor holdout PR-AUC global.
- Tiene mejor UID conocido y UID desconocido bajo la comparacion comparable.
- La poda de features reduce demasiado la senal.

V03 sigue siendo util, pero como herramienta de monitoreo:

- `monitor` debe observarse por drift.
- `review` no debe priorizarse sin ablation.
- No se recomienda podar automaticamente `low_priority`.

## Implicancia

El siguiente paso debe pasar de "mejorar score" a "convertir score en decision", usando V01 como modelo de referencia:

- definir umbrales de aprobar/revisar/escalar;
- medir costo esperado;
- respetar capacidad maxima de revision;
- evaluar fraude no detectado y friccion operacional.

Esto corresponde a V05: politica de decision.

