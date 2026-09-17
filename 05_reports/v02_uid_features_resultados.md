# V02 - Features UID historicas

Kernel Kaggle:

`biancaaguinaga/p1-ieee-fraud-v02-uid-features`

Estado final: `COMPLETE`

Outputs locales:

`p1/03_outputs/v02_uid_features/`

Bitacora general de decisiones:

`p1/01_planning/decision_log_modelado.md`

## Objetivo

Evaluar si agregar memoria historica de entidades tipo UID mejora el baseline temporal V01 sin usar informacion futura ni target encoding.

## Diseno

Se probaron tres ablations con el mismo split temporal:

- `base`: comparable a V01.
- `uid_freq`: base + conteos/frecuencias/flags de UID conocido.
- `uid_freq_amt`: uid_freq + estadisticas historicas de `TransactionAmt` por UID.

Las features UID se calcularon usando solo la ventana de entrenamiento. No se uso `isFraud` para construir UIDs, frecuencias ni agregaciones.

Nota metodologica: en V02, la marca de UID conocido para validacion y holdout se calcula contra la ventana de entrenamiento. Es una definicion conservadora, coherente con la regla anti-leakage usada para construir features.

## Resultados

| Modelo | Split | Segmento | PR-AUC | ROC-AUC |
| --- | --- | --- | ---: | ---: |
| base | valid | global | 0.5954 | 0.9257 |
| uid_freq | valid | global | 0.5938 | 0.9281 |
| uid_freq_amt | valid | global | 0.6071 | 0.9311 |
| base | holdout | global | 0.5436 | 0.9051 |
| uid_freq | holdout | global | 0.5381 | 0.9060 |
| uid_freq_amt | holdout | global | 0.5442 | 0.9080 |

## Resultado por UID

| Modelo | Split | UID conocido PR-AUC | UID desconocido PR-AUC |
| --- | --- | ---: | ---: |
| base | valid | 0.6485 | 0.5942 |
| uid_freq | valid | 0.6937 | 0.5815 |
| uid_freq_amt | valid | 0.7408 | 0.5756 |
| base | holdout | 0.5946 | 0.5458 |
| uid_freq | holdout | 0.6282 | 0.5346 |
| uid_freq_amt | holdout | 0.6831 | 0.5315 |

## Lectura

Las features UID si aportan senal para transacciones con UID conocido. La mejora es grande tanto en validacion como en holdout:

- Validacion UID conocido: 0.6485 -> 0.7408 con `uid_freq_amt`.
- Holdout UID conocido: 0.5946 -> 0.6831 con `uid_freq_amt`.

Pero el mismo conjunto de features perjudica UID desconocido:

- Validacion UID desconocido: 0.5942 -> 0.5756.
- Holdout UID desconocido: 0.5458 -> 0.5315.

La mejora global en holdout es muy pequena:

- Base: 0.5436.
- `uid_freq_amt`: 0.5442.
- Diferencia: +0.0005.

## Decision

No se adopta `uid_freq_amt` como reemplazo general del baseline en este momento.

La razon es que mejora claramente los UIDs conocidos, pero degrada los UIDs desconocidos. Como el proyecto busca un sistema robusto ante datos futuros y cambios temporales, no conviene aceptar una mejora global casi nula si viene con degradacion del segmento mas dificil.

## Implicancia para el plan

Las features UID no se descartan. Quedan como candidatas para una estrategia segmentada:

- usar features UID o un modelo especializado para UID conocido;
- mantener un modelo mas transaccional o regularizado para UID desconocido;
- evaluar una politica diferenciada por segmento.

Antes de adoptar esta idea, corresponde ejecutar V03: auditoria de features, estabilidad temporal y riesgo de sobreajuste.
