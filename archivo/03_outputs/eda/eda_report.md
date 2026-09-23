# EDA propio - IEEE-CIS Fraud Detection
## Resumen ejecutivo
- Train tiene 590,540 filas y test tiene 506,691 filas.
- Tasa de fraude en train: 3.50%.
- La tasa semanal de fraude varia entre 1.85% y 5.06%.
- Candidate UID conocido en validacion tardia: 42.76% de filas.
- Adversarial validation train/test ROC-AUC: 1.000.
- Maxima correlacion absoluta numerica con isFraud: 0.383.

## Decisiones recomendadas
- **Usar PR-AUC como metrica principal**: Fraude en train = 3.50%; accuracy seria enganosa. Impacto: Reportar PR-AUC, recall, precision, F1 y FPR por ventana temporal.
- **No usar TransactionDT crudo como feature**: Es la variable que define el orden temporal; usarla cruda puede aprender posicion historica. Impacto: Derivar hora/dia/semana y usar TransactionDT para split y monitoreo.
- **Separar validacion temporal en UID conocido y UID desconocido**: 42.76% de filas tardias comparten candidate_uid con entrenamiento temprano. Impacto: Reportar desempeno por segmento y evitar conclusiones infladas.
- **Crear features UID, pero sin usar informacion futura**: card/D/addr forman clientes candidatos repetidos; top soluciones IEEE explotan esa senal. Impacto: Frequency/agregaciones calculadas solo con ventana de entrenamiento.
- **Conservar outliers de TransactionAmt**: Outliers IQR tienen tasa de fraude 5.07%. Impacto: Usar log1p, flags y winsorizar solo para modelos sensibles, no eliminar.
- **Tratar faltantes por bloque**: 74 features tienen mas de 80% faltante en train. Impacto: Agregar indicadores de ausencia, has_identity y missing-block features.
- **Auditar distribution shift antes de seleccionar features**: 4 features numericas tienen KS train/test > 0.20; adversarial AUC = 1.000. Impacto: Penalizar features inestables y medir robustez temporal.
- **No descartar columnas por correlacion con target solamente**: Max abs corr con isFraud = 0.383; no hay copia directa obvia del target. Impacto: Combinar correlacion, estabilidad temporal y permutation importance.
- **Modelar drift explicitamente**: La tasa semanal de fraude varia entre 1.85% y 5.06%. Impacto: Ventanas temporales, PSI/KS y comparacion static vs adaptive.

## Tablas generadas
- `structural_summary.csv`
- `missing_summary.csv`
- `target_correlations.csv`
- `temporal_summary.csv`
- `category_risk.csv`
- `numeric_shift_ks.csv`
- `uid_overlap_analysis.csv`
- `amount_analysis.csv`
- `redundancy_pairs.csv`
- `adversarial_metrics.csv`
- `adversarial_importance.csv`
- `decision_log.csv`

## Graficos generados
- `plots/01_target_balance.png`
- `plots/02_fraud_rate_by_week.png`
- `plots/03_missing_train_test.png`
- `plots/04_numeric_shift_ks.png`
- `plots/05_target_correlations.png`
- `plots/06_productcd_risk.png`
- `plots/07_amount_distribution.png`
