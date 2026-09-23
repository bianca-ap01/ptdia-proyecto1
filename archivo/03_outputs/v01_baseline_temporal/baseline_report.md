# V01 baseline temporal - IEEE-CIS Fraud Detection

## Objetivo

Construir un primer baseline temporal reproducible, sin `TransactionDT` crudo como feature y con metricas globales, por ventana y por UID conocido/desconocido.

## Diseno

- Train temporal: primeras 70% filas por `TransactionDT`.
- Validacion: siguiente 15%.
- Holdout futuro: ultimo 15%.
- Features usadas: 185.
- Modelo: LightGBM binario con early stopping.

Nota: `TransactionDT`, `DT_day_index` y `DT_week_index` no se usan como predictores; se reservan para split, ventanas y monitoreo. El unico derivado temporal usado como candidato es `DT_hour`.

## Resultados principales

- valid_global: PR-AUC=0.59541, ROC-AUC=0.92569, fraud_rate=0.0343, rows=88581.
- holdout_global: PR-AUC=0.54363, ROC-AUC=0.90507, fraud_rate=0.0348, rows=88581.

## Lectura metodologica

Este baseline no busca ser el modelo final. Su funcion es establecer una referencia temporal honesta para medir si las siguientes familias de features realmente aportan. Las transformaciones de monto, flags de faltantes y encodings se mantienen como candidatos verificables; su permanencia dependera de ablations posteriores.

## Archivos generados

- `baseline_summary.json`
- `baseline_metrics.csv`
- `baseline_window_metrics.csv`
- `baseline_feature_importance.csv`
- `baseline_valid_predictions.csv`
- `baseline_holdout_predictions.csv`
