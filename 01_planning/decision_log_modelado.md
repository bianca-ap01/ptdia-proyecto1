# Bitacora de decisiones de modelado

Proyecto 1 - IEEE-CIS Fraud Detection

Este documento registra todas las decisiones metodologicas y tecnicas tomadas durante la etapa de modelado. La regla de trabajo es: ninguna decision queda justificada solo por costumbre, por intuicion o porque otra solucion la haya usado. Cada decision debe tener evidencia, criterio y una forma de verificacion.

## Criterio de documentacion

Cada decision se documentara con:

- **Fecha**: dia de la decision.
- **Etapa**: EDA, baseline, features, auditoria, comparacion de modelos, politica de decision, drift o adaptacion.
- **Decision**: que se hizo o que se descarto.
- **Motivo**: que problema resuelve.
- **Evidencia**: dato, metrica, grafico, log o archivo que la sustenta.
- **Riesgo controlado**: leakage, sobreajuste, metrica enganosa, costo computacional, drift, etc.
- **Verificacion**: como se comprobara si la decision fue correcta.
- **Artefactos**: archivos locales o kernel Kaggle asociados.

## Decisiones registradas

### 2026-09-17 - EDA propio en Kaggle

**Etapa:** EDA.

**Decision:** Ejecutar un EDA propio antes de construir modelos.

**Motivo:** El plan de modelado necesitaba evidencia propia sobre distribucion, drift, faltantes, UIDs, outliers y train/test shift. No era suficiente basarse en el EDA del companero ni en soluciones externas.

**Evidencia:** El kernel `biancaaguinaga/p1-ieee-fraud-eda` corrio correctamente y genero outputs locales.

**Riesgo controlado:** Construir modelos con supuestos no verificados.

**Verificacion:** Revision de `eda_summary.json`, `eda_report.md`, `decision_log.csv`, `numeric_shift_ks.csv`, `uid_overlap_analysis.csv`, `target_correlations.csv` y `temporal_summary.csv`.

**Artefactos:**

- `p1/02_kaggle_kernels/eda/main.py`
- `p1/03_outputs/eda/eda_report.md`
- `p1/03_outputs/eda/eda_summary.json`
- `p1/03_outputs/eda/decision_log.csv`

### 2026-09-17 - PR-AUC como metrica principal

**Etapa:** Diseno de evaluacion.

**Decision:** Usar PR-AUC como metrica principal y reportar Recall, Precision, F1, ROC-AUC y FPR como metricas complementarias.

**Motivo:** El fraude representa solo 3.50% del train. Accuracy puede ser alta aunque el modelo no detecte fraudes.

**Evidencia:** `p1/03_outputs/eda/eda_summary.json` reporta `train_fraud_rate_pct = 3.499`.

**Riesgo controlado:** Metrica enganosa por desbalance de clases.

**Verificacion:** Todos los notebooks de modelado deben exportar PR-AUC, Recall, Precision, F1, ROC-AUC y FPR.

**Artefactos:**

- `p1/01_planning/plan_modelado_eda.md`
- `p1/03_outputs/v01_baseline_temporal/baseline_metrics.csv`

### 2026-09-17 - Validacion temporal

**Etapa:** Diseno de evaluacion.

**Decision:** No usar split aleatorio. Usar train temporal, validacion temporal y holdout futuro.

**Motivo:** El proyecto exige evaluar sobre datos futuros y el EDA encontro variacion temporal real.

**Evidencia:** La tasa semanal de fraude varia entre 1.85% y 5.06%.

**Riesgo controlado:** Leakage temporal y estimacion optimista del desempeno.

**Verificacion:** Cada kernel debe reportar los cortes temporales usados y separar validacion/holdout.

**Artefactos:**

- `p1/03_outputs/eda/temporal_summary.csv`
- `p1/03_outputs/v01_baseline_temporal/baseline_summary.json`

### 2026-09-17 - No usar `TransactionDT` crudo

**Etapa:** Features y evaluacion temporal.

**Decision:** Excluir `TransactionDT` como predictor. Usarlo solo para ordenamiento, splits y monitoreo.

**Motivo:** `TransactionDT` define la posicion temporal de la transaccion. Usarlo crudo puede hacer que el modelo aprenda periodo historico en lugar de patrones de fraude.

**Evidencia:** El EDA propio obtuvo adversarial validation train/test con ROC-AUC = 1.000, explicado por variables temporales.

**Riesgo controlado:** Sobreajuste a posicion temporal y mala generalizacion futura.

**Verificacion:** Revisar que `TransactionDT` no aparezca en `baseline_feature_importance.csv` ni en la lista de features entrenadas.

**Artefactos:**

- `p1/03_outputs/eda/adversarial_metrics.csv`
- `p1/02_kaggle_kernels/v01_baseline_temporal/main.py`
- `p1/03_outputs/v01_baseline_temporal/baseline_feature_importance.csv`

### 2026-09-17 - Corregir exclusion de `DT_day_index` y `DT_week_index`

**Etapa:** Baseline V01.

**Decision:** Excluir `DT_day_index` y `DT_week_index` del entrenamiento final de V01. Mantenerlos solo para metricas por ventana y monitoreo.

**Motivo:** En una version inicial del baseline, `DT_day_index` aparecio con importancia alta. Aunque era derivado de `TransactionDT`, seguia codificando posicion temporal absoluta.

**Evidencia:** La importancia de features de la version previa mostro `DT_day_index` entre variables relevantes. La version final ya no lo incluye como predictor.

**Riesgo controlado:** Contradecir el criterio metodologico de no usar tiempo absoluto como predictor.

**Verificacion:** La version final de `baseline_feature_importance.csv` no contiene `TransactionDT`, `DT_day_index` ni `DT_week_index`.

**Artefactos:**

- `p1/02_kaggle_kernels/v01_baseline_temporal/main.py`
- `p1/03_outputs/v01_baseline_temporal/baseline_feature_importance.csv`
- `p1/v01_baseline_temporal_resultados.md`

### 2026-09-17 - Baseline temporal con LightGBM

**Etapa:** Baseline V01.

**Decision:** Construir el primer baseline con LightGBM, usando 70% train temporal, 15% validacion y 15% holdout futuro.

**Motivo:** El dataset es tabular, grande y mixto: variables numericas, categoricas codificadas, faltantes e interacciones no lineales. LightGBM permite obtener una primera referencia fuerte sin asumir linealidad.

**Evidencia:** El kernel `biancaaguinaga/p1-ieee-fraud-v01-baseline-temporal` completo correctamente. La version final usa 185 features y early stopping.

**Riesgo controlado:** Empezar con un modelo demasiado simple que subestime la senal disponible, o con un modelo demasiado complejo sin baseline.

**Verificacion:** Comparar V02, V03 y modelos posteriores contra este baseline.

**Artefactos:**

- `p1/02_kaggle_kernels/v01_baseline_temporal/main.py`
- `p1/03_outputs/v01_baseline_temporal/baseline_metrics.csv`
- `p1/03_outputs/v01_baseline_temporal/baseline_report.md`
- `p1/v01_baseline_temporal_resultados.md`

### 2026-09-17 - Separar resultados por UID conocido y desconocido

**Etapa:** Baseline V01 y validacion futura.

**Decision:** Reportar metricas separadas para UID conocido y UID desconocido.

**Motivo:** El EDA mostro que los UIDs conocidos y desconocidos tienen tasas de fraude distintas y representan niveles de dificultad distintos.

**Evidencia:** En el EDA, 42.76% de filas tardias comparten UID candidato con train temprano. En V01, validacion y holdout reportan segmentos separados.

**Riesgo controlado:** Ocultar degradacion o dependencia excesiva de historial bajo una metrica global.

**Verificacion:** Cada version de modelado debe exportar metricas por segmento.

**Artefactos:**

- `p1/03_outputs/eda/uid_overlap_analysis.csv`
- `p1/03_outputs/v01_baseline_temporal/baseline_metrics.csv`

### 2026-09-17 - Baseline V01 final

**Etapa:** Resultado V01.

**Decision:** Aceptar V01 como baseline temporal inicial corregido.

**Motivo:** Produce una referencia reproducible y metodologicamente consistente para comparar mejoras futuras.

**Evidencia:** Resultados principales:

| Segmento | PR-AUC | ROC-AUC |
| --- | ---: | ---: |
| Validacion global | 0.5954 | 0.9257 |
| Holdout global | 0.5436 | 0.9051 |
| Holdout UID conocido | 0.5159 | 0.9020 |
| Holdout UID desconocido | 0.5632 | 0.8933 |

**Riesgo controlado:** Avanzar a features mas complejas sin una referencia base.

**Verificacion:** Las siguientes versiones deben demostrar mejora contra estos resultados, especialmente en holdout y UID desconocido.

**Artefactos:**

- `p1/v01_baseline_temporal_resultados.md`
- `p1/03_outputs/v01_baseline_temporal/baseline_metrics.csv`

## Decisiones pendientes

### 2026-09-17 - Ejecutar V01b full feature baseline

**Etapa:** Experimento de control posterior a V01.

**Decision:** Ejecutar un baseline con todas las features permitidas, sin reducir el bloque `V`, para comparar contra V01 reducido.

**Motivo:** V01 redujo el bloque `V` a 80 columnas para obtener una referencia rapida y controlada. Antes de construir features UID mas complejas, conviene medir cuanta senal se pierde al recortar features.

**Evidencia que motiva la decision:** El EDA mostro que varias columnas `V` tienen correlacion relevante con `isFraud`, y V01 encontro `V258` y `V257` entre las features mas importantes.

**Riesgo controlado:** Avanzar con una base reducida sin saber si descarto demasiada senal util.

**Riesgo introducido:** Mayor costo de entrenamiento y posible sobreajuste a features redundantes o inestables.

**Verificacion:** Comparar V01b contra V01 en validacion, holdout, UID conocido, UID desconocido, metricas por ventana, tiempo de entrenamiento y feature importance.

**Criterio de adopcion:** V01b solo reemplaza a V01 si mejora holdout y no empeora estabilidad temporal ni UID desconocido. Si solo mejora validacion, se interpreta como posible sobreajuste.

**Artefactos esperados:**

- `p1/02_kaggle_kernels/v01b_full_baseline/main.py`
- `p1/03_outputs/v01b_full_baseline/baseline_metrics.csv`
- `p1/v01b_full_baseline_temporal_resultados.md`

### 2026-09-17 - Resultado V01b full feature baseline

**Etapa:** Experimento de control posterior a V01.

**Decision:** No reemplazar V01 por V01b como baseline principal.

**Motivo:** V01b usa todas las features permitidas, pero no mejora la generalizacion global ni el holdout futuro frente a V01. La mejora en UID desconocido es marginal y no compensa la caida en UID conocido.

**Evidencia:**

| Segmento | V01 PR-AUC | V01b PR-AUC | Diferencia |
| --- | ---: | ---: | ---: |
| Validacion global | 0.5954 | 0.5899 | -0.0055 |
| Validacion UID conocido | 0.6485 | 0.6078 | -0.0407 |
| Validacion UID desconocido | 0.5942 | 0.6052 | +0.0110 |
| Holdout global | 0.5436 | 0.5303 | -0.0133 |
| Holdout UID conocido | 0.5159 | 0.4475 | -0.0685 |
| Holdout UID desconocido | 0.5632 | 0.5636 | +0.0004 |

**Riesgo controlado:** Adoptar mas features solo por parecer mas completo.

**Interpretacion:** El bloque completo agrega senal para algunas transacciones, pero tambien ruido, redundancia o inestabilidad temporal. La evidencia favorece mantener el baseline reducido y auditar features antes de expandir.

**Verificacion:** V01b completo correctamente en Kaggle y genero outputs comparables.

**Artefactos:**

- `p1/02_kaggle_kernels/v01b_full_baseline/main.py`
- `p1/03_outputs/v01b_full_baseline/baseline_metrics.csv`
- `p1/v01b_full_baseline_temporal_resultados.md`

### V02 - Features UID historicas

### 2026-09-17 - Ejecutar V02 features UID historicas

**Etapa:** Features UID y ablations.

**Decision:** Ejecutar V02 con features historicas de UID sobre la base V01 reducida, comparando familias de features por ablation.

**Motivo:** El EDA mostro que una parte importante de las transacciones tardias comparte UID candidato con el entrenamiento temprano, y que UID conocido/desconocido tienen distinta tasa de fraude. Esto sugiere que el historial de entidad puede aportar senal, pero debe verificarse sin leakage.

**Evidencia que motiva la decision:**

- `uid_overlap_analysis.csv`: 42.76% de filas tardias tienen UID conocido.
- V01: el desempeno cambia entre UID conocido y desconocido.
- V01b: agregar todas las features sin criterio no mejoro holdout; por eso V02 agrega una familia de features concreta y verificable.

**Riesgo controlado:** Quedarnos con un modelo transaccional que ignora memoria historica de tarjeta/cliente.

**Riesgo introducido:** Leakage temporal si las frecuencias o agregaciones usan informacion de validacion/holdout para construir features.

**Regla anti-leakage:** Las frecuencias y agregaciones de UID para validacion y holdout se calcularan usando solo la ventana de entrenamiento. No se usara `isFraud` para construir estas features.

**Ablations a ejecutar:**

- `base`: misma logica de V01.
- `uid_freq`: base + frecuencias/conteos de UIDs calculados en train.
- `uid_freq_amt`: uid_freq + agregaciones historicas de `TransactionAmt` por UID.

**Criterio de adopcion:** Las features UID se adoptan si mejoran PR-AUC en holdout global o en UID desconocido sin degradar de forma importante UID conocido ni aumentar senales de sobreajuste.

**Artefactos esperados:**

- `p1/02_kaggle_kernels/v02_uid_features/main.py`
- `p1/03_outputs/v02_uid_features/v02_model_comparison.csv`
- `p1/v02_uid_features_resultados.md`

### 2026-09-17 - Resultado V02 features UID historicas

**Etapa:** Features UID y ablations.

**Decision:** No adoptar todavia las features UID como reemplazo general del baseline V01.

**Motivo:** Las features UID mejoran mucho el desempeno en UID conocido, pero degradan UID desconocido. La mejora global en holdout es demasiado pequena para justificar el cambio como modelo unico.

**Evidencia:**

| Modelo | Split | Segmento | PR-AUC |
| --- | --- | --- | ---: |
| base | holdout | global | 0.5436 |
| uid_freq | holdout | global | 0.5381 |
| uid_freq_amt | holdout | global | 0.5442 |
| base | holdout | UID conocido | 0.5946 |
| uid_freq_amt | holdout | UID conocido | 0.6831 |
| base | holdout | UID desconocido | 0.5458 |
| uid_freq_amt | holdout | UID desconocido | 0.5315 |

**Riesgo controlado:** Aceptar una mejora global minima que oculta degradacion en el segmento mas dificil.

**Interpretacion:** El historial de UID contiene senal real para entidades conocidas, pero puede hacer que el modelo dependa de memoria historica y generalice peor a entidades desconocidas.

**Verificacion:** V02 corrio tres ablations en Kaggle con el mismo split temporal.

**Artefactos:**

- `p1/02_kaggle_kernels/v02_uid_features/main.py`
- `p1/03_outputs/v02_uid_features/v02_model_comparison.csv`
- `p1/03_outputs/v02_uid_features/v02_report.md`
- `p1/v02_uid_features_resultados.md`

**Proxima decision tecnica:** Evaluar si las UID features deben usarse solo en una estrategia segmentada o si deben regularizarse/auditarse antes de entrar al modelo final.

### V03 - Auditoria de features

### 2026-09-17 - Ejecutar V03 auditoria de features

**Etapa:** Auditoria de features y estabilidad temporal.

**Decision:** Ejecutar una auditoria de features sobre la base V01 para combinar importancia predictiva, shift temporal y estabilidad.

**Motivo:** V01b mostro que agregar todas las features no mejora por si solo, y V02 mostro que agregar features UID mejora un segmento pero degrada otro. Antes de seguir complejizando el modelo, corresponde auditar que variables son predictivas y cuales son inestables.

**Evidencia que motiva la decision:**

- V01b full features empeoro holdout global frente a V01.
- V02 UID features mejoro UID conocido pero empeoro UID desconocido.
- El EDA encontro shift en variables temporales, `V` y `D15`.

**Riesgo controlado:** Adoptar features por importancia aparente sin evaluar si generalizan temporalmente.

**Metodo:**

- Reentrenar el baseline V01.
- Calcular importancia por ganancia de LightGBM.
- Calcular permutation importance en validacion para las features mas importantes.
- Medir KS train-valid y train-holdout por feature.
- Medir cambios de missingness por feature.
- Clasificar features como `keep`, `monitor`, `low_priority` o `review`.

**Criterio de uso posterior:**

- `keep`: feature predictiva y sin shift alto.
- `monitor`: feature predictiva pero con shift o cambio de missingness.
- `low_priority`: baja importancia y baja evidencia predictiva.
- `review`: feature con shift alto o comportamiento sospechoso que requiere revision antes de entrar al modelo final.

**Artefactos esperados:**
- `p1/02_kaggle_kernels/v03_feature_audit/main.py`
- `p1/03_outputs/v03_feature_audit/feature_audit.csv`
- `p1/v03_feature_audit_resultados.md`

### 2026-09-17 - Resultado V03 auditoria de features

**Etapa:** Auditoria de features y estabilidad temporal.

**Decision:** Usar la clasificacion de V03 para guiar los siguientes modelos: `keep` como nucleo confiable, `monitor` con seguimiento, `review` solo bajo prueba explicita y `low_priority` como secundarias.

**Motivo:** La auditoria mostro que no todas las features aportan igual ni tienen la misma estabilidad. Algunas variables son predictivas pero cambian fuerte en el tiempo, por lo que no conviene tratarlas igual que features estables.

**Evidencia:**

| Recomendacion | Features |
| --- | ---: |
| keep | 44 |
| monitor | 9 |
| review | 20 |
| low_priority | 112 |

Top `monitor`: `D1n`, `D2n`, `D15n`, `D10n`, `C9`, `id_02`, `id_20`, `D11`, `id_01`.

**Riesgo controlado:** Adoptar features por importancia aparente sin considerar drift temporal.

**Interpretacion:** Las variables `D*n` son utiles pero riesgosas por shift; variables de tarjeta, monto y varias `C` parecen mas confiables. Los flags agregados de missingness no deben asumirse utiles sin ablation.

**Verificacion:** V03 reprodujo el baseline V01 y genero `feature_audit.csv`, `permutation_importance.csv` y `feature_distribution_audit.csv`.

**Artefactos:**

- `p1/02_kaggle_kernels/v03_feature_audit/main.py`
- `p1/03_outputs/v03_feature_audit/feature_audit.csv`
- `p1/03_outputs/v03_feature_audit/v03_feature_audit_report.md`
- `p1/v03_feature_audit_resultados.md`

**Proxima decision tecnica:** Ejecutar V04 con comparacion de modelos o feature sets usando `keep` y `monitor` como base auditada.

### 2026-09-17 - Ejecutar V04 comparacion con feature sets auditados

**Etapa:** Comparacion de modelos y feature sets.

**Decision:** Ejecutar V04 comparando:

- LightGBM con `keep`.
- LightGBM con `keep + monitor`.
- XGBoost con `keep + monitor`, si esta disponible.
- CatBoost con `keep + monitor`, si esta disponible.

**Motivo:** V03 separo features confiables (`keep`) de features predictivas pero inestables (`monitor`). Ahora hay que medir si conviene usar solo el nucleo estable o tambien las features monitoreadas. Ademas, se verifica si la conclusion depende de LightGBM o se mantiene con otro algoritmo.

**Evidencia que motiva la decision:**

- V03 encontro 44 features `keep` y 9 features `monitor`.
- Las features `monitor` incluyen variables `D*n` con mucha senal pero shift alto.
- V02 mostro que agregar senal historica puede mejorar un segmento y empeorar otro, por lo que la comparacion debe hacerse por holdout y segmentos.

**Riesgo controlado:** Elegir features solo por importancia sin validar estabilidad, o elegir un modelo por un resultado aislado.

**Criterio de adopcion:** Un candidato se considera mejor que V01 solo si mejora holdout global o UID desconocido sin deteriorar de forma importante UID conocido, y si mantiene resultados razonables por ventana temporal.

**Artefactos esperados:**

- `p1/02_kaggle_kernels/v04_model_comparison/main.py`
- `p1/03_outputs/v04_model_comparison/v04_model_comparison.csv`
- `p1/v04_model_comparison_resultados.md`

### 2026-09-17 - Resultado V04 comparacion de modelos

**Etapa:** Comparacion de modelos y feature sets.

**Decision:** No reemplazar V01 por modelos basados solo en `keep` o `keep + monitor`.

**Motivo:** Todos los candidatos de V04 quedan por debajo de V01 en holdout global. La poda de features reduce demasiado la senal, aunque mejore la interpretabilidad.

**Evidencia:**

| Modelo | Holdout PR-AUC | Holdout ROC-AUC |
| --- | ---: | ---: |
| V01 baseline | 0.5436 | 0.9051 |
| lgbm_keep | 0.5002 | 0.9001 |
| lgbm_keep_monitor | 0.5131 | 0.8996 |
| xgb_keep_monitor | 0.5128 | 0.9055 |
| cat_keep_monitor | 0.4989 | 0.9020 |

Por UID en holdout, V01 comparable tambien supera a los candidatos auditados:

| Modelo | UID conocido PR-AUC | UID desconocido PR-AUC |
| --- | ---: | ---: |
| V01 baseline comparable | 0.5946 | 0.5458 |
| lgbm_keep_monitor | 0.5425 | 0.5181 |
| xgb_keep_monitor | 0.4923 | 0.5296 |

**Riesgo controlado:** Podar features de forma demasiado agresiva solo porque algunas parecen de baja prioridad individual.

**Interpretacion:** V03 sirve para monitoreo y auditoria, no para eliminar automaticamente features. En modelos de boosting, features de baja importancia individual pueden aportar senal combinada.

**Verificacion:** V04 corrio en Kaggle con LightGBM, XGBoost y CatBoost sobre el mismo split temporal.

**Artefactos:**

- `p1/02_kaggle_kernels/v04_model_comparison/main.py`
- `p1/03_outputs/v04_model_comparison/v04_model_comparison.csv`
- `p1/03_outputs/v04_model_comparison/v04_report.md`
- `p1/v04_model_comparison_resultados.md`

**Proxima decision tecnica:** Avanzar a V05 politica de decision usando V01 como modelo de referencia, porque ya tenemos un score base defendible.

### 2026-09-17 - Ejecutar V05 politica de decision

**Etapa:** Politica de decision basada en riesgo.

**Decision:** Construir una politica de tres acciones usando los scores de V01:

- `approve`: aprobar automaticamente.
- `review`: enviar a revision manual.
- `escalate`: escalar/bloquear por riesgo alto.

**Motivo:** El proyecto no pide solo clasificar fraude; pide transformar el riesgo estimado en una decision operacional con capacidad limitada de revision y costos asimetricos.

**Supuestos iniciales de costo:** Se usara un escenario base normalizado:

- Fraude aprobado: 100 unidades de costo.
- Transaccion legitima escalada/bloqueada: 10 unidades de costo.
- Revision manual: 2 unidades de costo por transaccion revisada.

**Justificacion de costos:** No representan dinero real, sino ponderaciones relativas. El fraude no detectado debe costar mucho mas que una revision; una transaccion legitima bloqueada debe costar mas que una revision porque genera friccion fuerte para el cliente.

**Restriccion operacional:** La revision manual no debe superar 5% de las transacciones.

**Metodo:** Seleccionar umbrales sobre validacion, aplicar la politica sin recalibrar en holdout y comparar contra politicas baseline.

**Riesgo controlado:** Elegir umbrales arbitrarios o evaluar el modelo solo por PR-AUC sin traducirlo a accion.

**Artefactos esperados:**

- `p1/02_kaggle_kernels/v05_policy_decision/main.py`
- `p1/04_scripts/v05_policy_decision.py`
- `p1/03_outputs/v05_policy_decision/policy_thresholds.csv`
- `p1/03_outputs/v05_policy_decision_kaggle/`
- `p1/05_reports/v05_policy_resultados.md`

**Resultado ejecutado:**

Umbrales seleccionados en validacion:

| Umbral | Valor |
| --- | ---: |
| Revision | 0.459783 |
| Escalamiento | 0.779122 |

Resultados globales:

| Split | Costo por transaccion | Fraude detectado | Fraude no detectado | Revision | Escalamiento | Aprobacion automatica |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validacion | 1.1395 | 71.83% | 28.17% | 4.98% | 2.35% | 92.67% |
| Holdout | 1.3680 | 66.17% | 33.83% | 5.10% | 2.48% | 92.42% |

Comparacion en holdout:

| Politica | Costo por transaccion | Fraude detectado | Revision | Escalamiento |
| --- | ---: | ---: | ---: | ---: |
| Aprobar todo | 3.4804 | 0.00% | 0.00% | 0.00% |
| Revisar top 5% | 1.5247 | 59.20% | 5.23% | 0.00% |
| Escalar top 1% | 2.5959 | 25.75% | 0.00% | 1.01% |
| Politica V05 | 1.3680 | 66.17% | 5.10% | 2.48% |

**Decision posterior:** Aceptar V05 como primera politica operativa documentada. No reemplaza el modelo V01; lo usa para transformar scores en acciones. La politica supera a las reglas simples comparadas, pero queda marcada como version inicial porque el holdout excede levemente la capacidad objetivo de revision (5.10% vs 5.00%) y porque hay diferencias fuertes por segmento UID.

**Riesgo controlado:** Evitar concluir solo con PR-AUC. El score sirve para ordenar riesgo, pero una solucion de fraude necesita traducir ese ordenamiento en decisiones con costos y capacidad.

**Proxima decision tecnica:** Avanzar a V06 con sensibilidad de costos, calibracion y/o umbrales por segmento UID para verificar si la politica se mantiene estable bajo supuestos alternativos.

### 2026-09-17 - Ejecutar V06 robustez de politica

**Etapa:** Robustez de la politica de decision.

**Decision:** Evaluar tres dimensiones antes de recomendar la politica final:

- sensibilidad a costos, porque los costos usados en V05 son supuestos relativos;
- calibracion, porque un score con buen ranking no necesariamente es una probabilidad confiable;
- umbrales por segmento UID, porque V05 mostro diferencias fuertes entre UID conocido y desconocido.

**Motivo:** V05 es prometedora, pero una decision operacional no debe depender de un unico escenario de costos ni de una unica politica global si hay evidencia de heterogeneidad por segmento.

**Riesgo controlado:** Sobreajustar la recomendacion al escenario base de V05.

**Artefactos esperados:**

- `p1/02_kaggle_kernels/v06_policy_robustness/main.py`
- `p1/04_scripts/v06_policy_robustness.py`
- `p1/03_outputs/v06_policy_robustness/global_policy_sensitivity.csv`
- `p1/03_outputs/v06_policy_robustness/segmented_uid_policy_sensitivity.csv`
- `p1/03_outputs/v06_policy_robustness/calibration_summary.csv`
- `p1/03_outputs/v06_policy_robustness_kaggle/`
- `p1/05_reports/v06_policy_robustness_resultados.md`

**Resultado ejecutado:**

Sensibilidad global en holdout:

| Escenario | Costo por transaccion | Fraude detectado | Revision | Escalamiento | Aprobacion automatica |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 1.3809 | 68.80% | 5.07% | 3.78% | 91.15% |
| fraud_high | 2.3357 | 75.15% | 4.99% | 7.34% | 87.67% |
| capacity_tight | 1.5100 | 63.93% | 3.10% | 3.77% | 93.13% |
| capacity_loose | 1.1695 | 75.35% | 9.63% | 2.87% | 87.50% |

Comparacion de politica base en holdout:

| Politica | Costo por transaccion | Fraude detectado | Fraude no detectado | Revision | Escalamiento |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global | 1.3809 | 68.80% | 31.20% | 5.07% | 3.78% |
| Segmentada UID | 1.3724 | 66.56% | 33.44% | 4.78% | 2.82% |

Calibracion:

| Split | Score | Brier | Score promedio | Tasa real de fraude |
| --- | --- | ---: | ---: | ---: |
| Validacion | Bruto | 0.0440 | 14.03% | 3.43% |
| Validacion | Isotonico | 0.0200 | 3.43% | 3.43% |
| Holdout | Bruto | 0.0479 | 14.30% | 3.48% |
| Holdout | Isotonico | 0.0219 | 3.55% | 3.48% |

**Decision posterior:** Mantener V05 como politica principal y agregar V06 como auditoria de robustez. No adoptar todavia umbrales segmentados UID: reducen levemente costo por transaccion, pero tambien reducen deteccion de fraude. Para comunicar probabilidades o perdida esperada, usar calibracion isotonica; para ranking y umbrales, el score bruto sigue siendo valido.

**Riesgo controlado:** Evitar que el proyecto recomiende un unico umbral sin mostrar sensibilidad a costos ni verificar calibracion.

**Proxima decision tecnica:** Preparar la version consolidada de modelado para entrega, con comparacion V01-V06, decision final y limitaciones.
