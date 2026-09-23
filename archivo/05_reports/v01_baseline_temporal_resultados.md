# V01 - Baseline temporal

Kernel Kaggle:

`biancaaguinaga/p1-ieee-fraud-v01-baseline-temporal`

Estado final: `COMPLETE`

Outputs locales:

`p1/03_outputs/v01_baseline_temporal/`

Bitacora general de decisiones:

`p1/01_planning/decision_log_modelado.md`

## Diseno

- Train temporal: primeras 70% filas por `TransactionDT`.
- Validacion: siguiente 15%.
- Holdout futuro: ultimo 15%.
- Modelo: LightGBM binario con early stopping.
- Features finales: 185.
- `TransactionDT`, `DT_day_index` y `DT_week_index` no se usaron como predictores.
- `DT_hour` se mantiene como unico derivado temporal candidato.

## Resultado principal

| Segmento | Filas | Fraud rate | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Validacion global | 88,581 | 3.43% | 0.5954 | 0.9257 |
| Validacion UID conocido | 40,835 | 2.33% | 0.6485 | 0.9568 |
| Validacion UID desconocido | 47,746 | 4.38% | 0.5942 | 0.9011 |
| Holdout global | 88,581 | 3.48% | 0.5436 | 0.9051 |
| Holdout UID conocido | 39,512 | 1.93% | 0.5159 | 0.9020 |
| Holdout UID desconocido | 49,069 | 4.73% | 0.5632 | 0.8933 |

## Lectura

El baseline ya supera ampliamente una referencia trivial de clase desbalanceada: la tasa base de fraude ronda 3.4%, mientras que PR-AUC llega a 0.5954 en validacion y 0.5436 en holdout futuro.

La caida de PR-AUC de validacion a holdout muestra degradacion temporal, que es justamente uno de los riesgos planteados en el proyecto. El resultado sostiene la necesidad de monitoreo por ventanas y de evaluar estrategias adaptativas mas adelante.

El desempeno por UID confirma que el problema no es uniforme. En validacion, UID conocido obtiene PR-AUC mayor que UID desconocido. En holdout, UID desconocido tiene mayor tasa de fraude y mejor PR-AUC que UID conocido, lo que sugiere que el comportamiento de entidades cambia entre ventanas y que no basta con una lectura global.

## Auditoria metodologica

En una version previa, `DT_day_index` aparecio con importancia alta. Se corrigio porque esa variable representa posicion temporal absoluta y podia contradecir la decision metodologica de no usar tiempo crudo como predictor. La version final excluye `TransactionDT`, `DT_day_index` y `DT_week_index` del set de entrenamiento.

## Proximo paso

Ejecutar V02 con features UID historicas y ablations:

- baseline sin UID;
- baseline + frequency encoding de UIDs;
- baseline + agregaciones historicas por UID;
- metricas separadas por UID conocido y desconocido.
