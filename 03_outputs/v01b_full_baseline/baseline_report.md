# V01b full feature baseline temporal - IEEE-CIS Fraud Detection

## Objetivo

Construir un baseline temporal de control usando todas las features permitidas, sin `TransactionDT` crudo ni derivados temporales absolutos como predictores.

## Diseno

- Train temporal: primeras 70% filas por `TransactionDT`.
- Validacion: siguiente 15%.
- Holdout futuro: ultimo 15%.
- Features usadas: 444.
- Modelo: LightGBM binario con early stopping.

Nota: `TransactionDT`, `DT_day_index` y `DT_week_index` no se usan como predictores; se reservan para split, ventanas y monitoreo. El unico derivado temporal usado como candidato es `DT_hour`.

## Resultados principales

- valid_global: PR-AUC=0.58987, ROC-AUC=0.92269, fraud_rate=0.0343, rows=88581.
- holdout_global: PR-AUC=0.53033, ROC-AUC=0.90035, fraud_rate=0.0348, rows=88581.

## Lectura metodologica

Este experimento no reemplaza automaticamente a V01. Su funcion es medir si la reduccion del bloque `V` dejo fuera senal util. Solo se adoptara como nueva referencia si mejora holdout y no empeora estabilidad temporal ni UID desconocido.

## Archivos generados

- `baseline_summary.json`
- `baseline_metrics.csv`
- `baseline_window_metrics.csv`
- `baseline_feature_importance.csv`
- `baseline_valid_predictions.csv`
- `baseline_holdout_predictions.csv`
