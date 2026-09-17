# Documento consolidado de modelado

Proyecto 1 - IEEE-CIS Fraud Detection

## 1. Objetivo del modelado

El objetivo de esta etapa fue construir y justificar la parte de creacion del modelo para un sistema adaptativo de deteccion de fraude. La meta no fue solamente entrenar un clasificador, sino construir una solucion que:

1. Detecte transacciones fraudulentas en un conjunto altamente desbalanceado.
2. Evaluar el desempeno como problema temporal, no como particion aleatoria.
3. Transforma el score de riesgo en decisiones operativas: aprobar, revisar o escalar.
4. Documenta riesgos de drift, calibracion y estabilidad por segmento.

La decision metodologica central fue tratar el fraude como un problema de ranking y decision bajo costo, no como un problema de accuracy.

## 2. Evidencia principal del EDA

El EDA propio se ejecuto en Kaggle y sus resultados estan en `p1/03_outputs/eda/`.

### 2.1 Desbalance de clases

El conjunto de entrenamiento contiene 590,540 transacciones y una tasa de fraude de 3.50%. Esto implica que accuracy no es una metrica adecuada: un modelo que prediga siempre "no fraude" tendria accuracy alta, pero utilidad nula para detectar fraude.

Decision tomada:

- Usar PR-AUC como metrica principal.
- Reportar ROC-AUC como metrica complementaria.
- Evaluar tambien resultados por segmento UID y por ventana temporal.

Sustento: en problemas desbalanceados, Precision-Recall describe mejor la calidad de deteccion de la clase minoritaria que accuracy o, muchas veces, ROC-AUC aislado.

### 2.2 Drift temporal

La tasa semanal de fraude varia aproximadamente entre 1.85% y 5.06%. Esto indica que la distribucion no es estacionaria. Por eso, una particion aleatoria mezclaria pasado y futuro y podria sobreestimar el desempeno.

Decision tomada:

- Ordenar por `TransactionDT`.
- Usar train temporal, validacion temporal y holdout futuro.
- No seleccionar modelos mirando el holdout.

### 2.3 Riesgo de usar tiempo absoluto

El EDA encontro que train y test pueden separarse casi perfectamente mediante validacion adversarial. La variable temporal es muy fuerte para distinguir periodos, pero eso no significa que sea una regla general de fraude.

Decision tomada:

- No usar `TransactionDT` crudo como predictor.
- Excluir tambien `DT_day_index` y `DT_week_index`.
- Usar esas variables solo para ordenar, partir y monitorear.

### 2.4 UIDs conocidos y desconocidos

El EDA mostro que una parte importante de las transacciones tardias comparte un UID candidato con ventanas anteriores, y que UID conocido y UID desconocido tienen tasas de fraude distintas.

Decision tomada:

- Reportar desempeno separado por UID conocido y UID desconocido.
- Probar features historicas de UID solo si se calculan sin mirar validacion/holdout.
- No aceptar mejoras globales pequenas si degradan el segmento UID desconocido.

### 2.5 Faltantes y variables anonimizadas

Muchas columnas tienen faltantes estructurales, especialmente variables de identidad y bloques `V`. La ausencia de informacion puede representar un canal, producto o flujo distinto, por lo que no debe tratarse automaticamente como error.

Decision tomada:

- Mantener modelos de arboles que toleran faltantes.
- No imputar todo mecanicamente con media.
- Auditar missingness y shift antes de podar features.

## 3. Diseno experimental

Todas las versiones de modelado usaron la misma filosofia de evaluacion:

- entrenamiento en el pasado;
- validacion en una ventana posterior;
- holdout como simulacion de futuro;
- metrica principal PR-AUC;
- comparacion por UID conocido/desconocido;
- documentacion de cada decision tecnica.

El baseline V01 uso:

- 70% train temporal;
- 15% validacion;
- 15% holdout futuro;
- LightGBM binario con early stopping;
- 185 features finales;
- exclusion de `TransactionDT`, `DT_day_index` y `DT_week_index`.

## 4. Resumen de versiones

| Version | Objetivo | Resultado principal | Decision |
| --- | --- | --- | --- |
| EDA | Entender distribucion, drift, faltantes, UIDs y shift train/test. | Fraude 3.50%, drift temporal y fuerte diferencia por UID. | Usar validacion temporal, PR-AUC y segmentos UID. |
| V01 | Baseline temporal LightGBM. | Holdout PR-AUC 0.5436. | Aceptar como baseline principal. |
| V01b | Probar todas las features permitidas. | Holdout PR-AUC 0.5303. | No reemplaza V01. |
| V02 | Agregar features UID historicas. | UID conocido mejora, UID desconocido empeora. | No adoptar como modelo unico. |
| V03 | Auditar importancia, shift y missingness. | 44 `keep`, 9 `monitor`, 20 `review`, 112 `low_priority`. | Usar para monitoreo, no para poda automatica. |
| V04 | Comparar modelos y feature sets auditados. | Ningun candidato supera V01. | Mantener V01. |
| V05 | Convertir score en politica de decision. | Holdout: costo 1.3680, fraude detectado 66.17%. | Aceptar como primera politica operativa. |
| V06 | Evaluar robustez, calibracion y segmentacion. | Calibracion mejora Brier; segmentacion UID reduce costo pero detecta menos fraude. | Mantener V05 y recomendar calibracion para probabilidades. |

## 5. Resultados predictivos

### 5.1 Baseline principal V01

| Segmento | Filas | Fraud rate | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Validacion global | 88,581 | 3.43% | 0.5954 | 0.9257 |
| Validacion UID conocido | 40,835 | 2.33% | 0.6485 | 0.9568 |
| Validacion UID desconocido | 47,746 | 4.38% | 0.5942 | 0.9011 |
| Holdout global | 88,581 | 3.48% | 0.5436 | 0.9051 |
| Holdout UID conocido | 39,512 | 1.93% | 0.5159 | 0.9020 |
| Holdout UID desconocido | 49,069 | 4.73% | 0.5632 | 0.8933 |

Lectura: V01 da una referencia fuerte y reproducible. La caida de validacion a holdout confirma drift temporal, pero el modelo mantiene capacidad de ranking en datos futuros.

### 5.2 Comparacion contra V01b

| Segmento | V01 PR-AUC | V01b PR-AUC | Diferencia |
| --- | ---: | ---: | ---: |
| Validacion global | 0.5954 | 0.5899 | -0.0055 |
| Validacion UID conocido | 0.6485 | 0.6078 | -0.0407 |
| Validacion UID desconocido | 0.5942 | 0.6052 | +0.0110 |
| Holdout global | 0.5436 | 0.5303 | -0.0133 |
| Holdout UID conocido | 0.5159 | 0.4475 | -0.0685 |
| Holdout UID desconocido | 0.5632 | 0.5636 | +0.0004 |

Decision: no usar todas las features solo por completitud. V01b agrega costo y complejidad sin mejorar generalizacion global.

### 5.3 Features UID historicas

| Modelo | Split | Global PR-AUC | UID conocido PR-AUC | UID desconocido PR-AUC |
| --- | --- | ---: | ---: | ---: |
| base | valid | 0.5954 | 0.6485 | 0.5942 |
| uid_freq_amt | valid | 0.6071 | 0.7408 | 0.5756 |
| base | holdout | 0.5436 | 0.5946 | 0.5458 |
| uid_freq_amt | holdout | 0.5442 | 0.6831 | 0.5315 |

Decision: las features UID tienen senal real para clientes conocidos, pero degradan clientes desconocidos. No se adoptan como modelo unico.

### 5.4 Auditoria y comparacion de modelos

V03 clasifico las features en:

| Recomendacion | Cantidad | Uso |
| --- | ---: | --- |
| `keep` | 44 | Senal util y relativamente estable. |
| `monitor` | 9 | Senal util, pero con shift o missingness cambiante. |
| `review` | 20 | Requieren prueba especifica antes de priorizarse. |
| `low_priority` | 112 | Baja evidencia individual en esta etapa. |

V04 probo modelos con el set auditado:

| Modelo | Valid PR-AUC | Holdout PR-AUC | Holdout ROC-AUC |
| --- | ---: | ---: | ---: |
| V01 baseline | 0.5954 | 0.5436 | 0.9051 |
| lgbm_keep | 0.5474 | 0.5002 | 0.9001 |
| lgbm_keep_monitor | 0.5739 | 0.5131 | 0.8996 |
| xgb_keep_monitor | 0.5504 | 0.5128 | 0.9055 |
| cat_keep_monitor | 0.5383 | 0.4989 | 0.9020 |

Decision: no podar agresivamente. La auditoria sirve para monitorear y entender riesgos, no para eliminar features automaticamente.

## 6. Politica de decision

El proyecto no termina con un score. Un sistema de fraude debe decidir que hacer con cada transaccion. Por eso V05 tradujo el score de V01 en tres acciones:

| Accion | Criterio | Interpretacion |
| --- | --- | --- |
| `approve` | score bajo | aprobar automaticamente |
| `review` | score intermedio | enviar a revision manual |
| `escalate` | score alto | bloquear o escalar |

### 6.1 Supuestos de costo

| Componente | Costo normalizado |
| --- | ---: |
| Fraude aprobado | 100 |
| Transaccion legitima escalada/bloqueada | 10 |
| Revision manual | 2 |
| Capacidad maxima de revision | 5% |

Estos valores son ponderaciones relativas, no dinero real. Sirven para comparar politicas bajo el supuesto de que dejar pasar fraude es mucho mas grave que revisar, y que bloquear erroneamente es mas costoso que revisar.

### 6.2 Umbrales seleccionados

| Umbral | Valor |
| --- | ---: |
| Revision | 0.459783 |
| Escalamiento | 0.779122 |

Los umbrales se eligieron en validacion y se aplicaron a holdout sin recalibrar.

### 6.3 Resultado de la politica V05

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

Decision: V05 se acepta como primera politica operativa. Supera a reglas simples y convierte el score en acciones explicables.

## 7. Robustez y calibracion

V06 verifico si la politica cambia de manera razonable ante costos distintos.

| Escenario | Costo por transaccion | Fraude detectado | Revision | Escalamiento | Aprobacion automatica |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 1.3809 | 68.80% | 5.07% | 3.78% | 91.15% |
| friction_high | 1.5068 | 63.12% | 4.52% | 1.96% | 93.53% |
| fraud_high | 2.3357 | 75.15% | 4.99% | 7.34% | 87.67% |
| capacity_tight | 1.5100 | 63.93% | 3.10% | 3.77% | 93.13% |
| capacity_loose | 1.1695 | 75.35% | 9.63% | 2.87% | 87.50% |

Lectura:

- Si escalar legitimas cuesta mas, la politica escala menos.
- Si dejar pasar fraude cuesta mas, la politica detecta mas fraude.
- Si hay mas capacidad de revision, aumenta la deteccion y baja el costo.
- Si la capacidad se reduce, aumenta el costo y cae la deteccion.

Esto muestra que la politica responde a los incentivos definidos.

### 7.1 Calibracion

| Split | Score | Brier | Score promedio | Tasa real de fraude |
| --- | --- | ---: | ---: | ---: |
| Validacion | Bruto | 0.0440 | 14.03% | 3.43% |
| Validacion | Isotonico | 0.0200 | 3.43% | 3.43% |
| Holdout | Bruto | 0.0479 | 14.30% | 3.48% |
| Holdout | Isotonico | 0.0219 | 3.55% | 3.48% |

Decision: el score bruto sirve para ranking y umbrales, pero no debe comunicarse como probabilidad sin calibracion. Para hablar de riesgo porcentual o perdida esperada, se recomienda calibracion isotonica.

### 7.2 Segmentacion por UID

| Politica | Costo por transaccion | Fraude detectado | Fraude no detectado | Revision | Escalamiento |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global | 1.3809 | 68.80% | 31.20% | 5.07% | 3.78% |
| Segmentada UID | 1.3724 | 66.56% | 33.44% | 4.78% | 2.82% |

Decision: no adoptar todavia la segmentacion UID. Baja levemente el costo, pero tambien reduce la deteccion de fraude. Puede ser candidata si el negocio prioriza reducir friccion por encima de maximizar deteccion.

## 8. Decision final recomendada

La recomendacion actual es:

1. Mantener V01 como modelo base principal.
2. Usar V05 como politica operativa principal.
3. Usar V06 como soporte de robustez y calibracion.
4. No adoptar V01b como reemplazo.
5. No adoptar features UID como modelo unico.
6. No podar features agresivamente solo por la auditoria V03.
7. Calibrar el score si se va a reportar como probabilidad.

Esta decision es conservadora pero defendible. El criterio no fue escoger el experimento con el mayor numero aislado, sino el que mantiene mejor balance entre desempeno futuro, estabilidad, interpretabilidad operacional y riesgo de leakage.

## 9. Limitaciones

| Limitacion | Impacto | Control propuesto |
| --- | --- | --- |
| Costos normalizados | No representan dinero real del negocio. | Repetir sensibilidad con costos reales si estan disponibles. |
| Drift temporal | El desempeno cae de validacion a holdout. | Monitoreo por ventana y recalibracion periodica. |
| Capacidad de revision fija | V05 supera levemente 5% en holdout. | Usar umbrales por percentil diario/semanal si la capacidad es estricta. |
| Datos anonimizados | Limita interpretacion causal de features. | Interpretar variables como senales predictivas, no causas. |
| UID features no adoptadas | Se deja senal util sin explotar para UID conocido. | Evaluar modelo segmentado si se permite mayor complejidad. |
| Score bruto mal calibrado | Riesgo de interpretar score como probabilidad real. | Usar calibracion isotonica para comunicacion probabilistica. |

## 10. Trabajo futuro

Las siguientes mejoras son razonables, pero no necesarias para cerrar esta primera entrega:

- integrar calibracion directamente en el pipeline final;
- hacer sensibilidad de costos con valores definidos por negocio;
- probar politica de capacidad por percentil temporal en vez de umbral fijo;
- evaluar un modelo especializado para UID conocido y otro para UID desconocido;
- implementar monitoreo automatico de PR-AUC, tasa de fraude, missingness y drift;
- explorar reentrenamiento por ventana temporal para simular adaptacion.

## 11. Referencias metodologicas

Las decisiones del pipeline se apoyan en evidencia experimental propia y en criterios metodologicos aceptados para fraude, clases desbalanceadas, drift y calibracion:

| Tema | Referencia | Uso en este proyecto |
| --- | --- | --- |
| Clases desbalanceadas | Krawczyk, 2016, *Learning from imbalanced data: open challenges and future directions*. | Justifica no usar accuracy como metrica principal y cuidar la clase minoritaria. |
| Precision-Recall | Saito y Rehmsmeier, 2015, *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. | Sustenta PR-AUC como metrica principal en fraude. |
| Concept drift | Gama et al., 2014, *A Survey on Concept Drift Adaptation*. | Sustenta validacion temporal, monitoreo y necesidad de adaptacion. |
| Validacion temporal | Documentacion de scikit-learn sobre `TimeSeriesSplit` y evaluacion temporal. | Sustenta no mezclar pasado y futuro con splits aleatorios. |
| Calibracion | Niculescu-Mizil y Caruana, 2005; Zadrozny y Elkan, 2002. | Sustenta calibrar scores antes de interpretarlos como probabilidades. |

## 12. Artefactos principales

| Tipo | Ruta |
| --- | --- |
| Planteo | `p1/00_brief/Project 1.pdf` |
| Plan metodologico | `p1/01_planning/plan_modelado_eda.md` |
| Bitacora de decisiones | `p1/01_planning/decision_log_modelado.md` |
| Kernels Kaggle | `p1/02_kaggle_kernels/` |
| Outputs | `p1/03_outputs/` |
| Scripts locales | `p1/04_scripts/` |
| Reportes por version | `p1/05_reports/` |

## 13. Conclusion

El modelo recomendado para esta etapa es V01, un LightGBM temporal con 185 features y validacion futura. No fue reemplazado por modelos mas complejos porque los experimentos posteriores no demostraron mejora robusta en holdout.

La politica recomendada es V05: aprobar transacciones de bajo riesgo, revisar riesgo intermedio y escalar riesgo alto. Esta politica reduce el costo frente a reglas simples y detecta una proporcion importante del fraude en holdout.

La entrega debe presentar V06 como evidencia adicional: la decision es sensible a costos de forma coherente, el score necesita calibracion para interpretarse como probabilidad, y la segmentacion UID es prometedora pero no suficientemente clara para adoptarse como politica principal en esta primera version.
