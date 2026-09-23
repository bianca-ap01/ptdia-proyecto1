# Entrega final del código — IEEE-CIS

Este paquete implementa una secuencia reproducible y auditable. Los cuatro notebooks son el entregable; `core.py` y `stages.py` son la fuente compartida de su código. `build_notebooks.py` los sincroniza y, sólo con `--owner`, prepara kernels privados de Kaggle. `final/` es la bitácora anterior y no se usa para elegir el modelo nuevo. El estado de ejecución se encuentra en `run_manifest.json`.

| Orden | Notebook | Salida principal |
| --- | --- | --- |
| 1 | `01_eda.ipynb` | EDA por periodos relativos de 30 días y auditoría de fuga. |
| 2 | `02_experimentacion.ipynb` | Cuatro familias, dos folds históricos, hiperparámetros y predicciones de validación. |
| 3 | `03_costos_escenarios.ipynb` | Cuatro escenarios, umbrales, ranking por menor arrepentimiento máximo y elección congelada. |
| 4 | `04_modelo_final.ipynb` | Evaluación retrospectiva, costo, calibración, capacidad y control de calidad con gate de promoción. |

El EDA guarda `monthly_missingness.csv` como muestra de cuatro variables y `monthly_missingness_all.csv` con la tasa de faltantes de las 431 variables originales por periodo de 30 días. `plot.ipynb` resume el perfil completo y muestra las columnas con mayor cambio temporal.
También guarda `feature_associations.csv` (Pearson para variables numéricas y V de Cramér para categóricas), `feature_importance_gain.csv` y `feature_importance_permutation.csv`. Estos diagnósticos se calculan con entrenamiento histórico y una muestra de validación; no sustituyen la selección de modelos y costos de las etapas siguientes. `plot.ipynb` grafica los rankings y distingue exclusiones explícitas, columnas usadas solo en el EDA, `DT_hour` como predictor actual y variables temporales de un EDA anterior que ya no se generan.

## Contrato temporal

El corte cronológico es 70/15/15 por `TransactionDT`. Los dos folds internos evalúan los tramos 50–65 % y 75–90 % de la duración temporal del train inicial, de modo que ambos caben dentro de ese periodo en IEEE-CIS. Un ajuste usa etiquetas con al menos siete días de antigüedad. El último tramo de siete días de validación no puede informar la elección al inicio del holdout. No se usa la etiqueta del test sin etiqueta de la competencia. Las estadísticas de preprocesamiento y la selección de variables `V` se ajustan en cada ventana de entrenamiento. Cinco atributos por UID usan sólo eventos anteriores.

La implementación congela modelo, hiperparámetros y umbrales antes de acceder a las etiquetas del holdout. La evaluación final se informa como histórica; una prueba prospectiva requiere nuevas transacciones etiquetadas.

## Costos y escenarios

Por transacción, la simulación suma `0.65 × revisión + 4 × escalación directa + 4 × handoff tras revisión + 5.75 × monto de fraude no detenido`. Una revisión que identifica fraude conduce a confirmación/escalación; si falla cualquiera de los dos pasos, queda pérdida residual. Las tasas revisión/escalación son 100/100, 90/98, 80/95 y 60/90 %. La cuenta de handoffs y la pérdida residual son valores **esperados** bajo cada supuesto, no etiquetas disponibles al modelo cuando decide.

El costo evitado se mide frente a aprobar todas las transacciones: `ahorro = costo_aprobar_todo − costo_política`. Por eso los fraudes detenidos aparecen como pérdida evitada, no como ingreso por `TransactionAmt`. [Forrester TEI encargado por Mastercard](https://tei.forrester.com/go/mastercard/decisionintelligenceandrulesservices/) aporta USD 0.65 por revisión y USD 4 para seguimiento de **algunas falsas alertas**. Aplicar USD 4 a una escalación de fraude confirmado es una hipótesis de nuestro flujo, no un precio demostrado por Forrester. [LexisNexis 2025](https://risk.lexisnexis.com/insights-resources/research/US-CA-true-cost-of-fraud-study) informa el multiplicador total 5.75 para servicios financieros de EE. UU.; la divisa real de `TransactionAmt` no está documentada y puede existir solapamiento con costos operativos. Las cifras son escenarios ilustrativos, no ahorro observado.

La capacidad primaria es 5 % de la mediana diaria de transacciones del train disponible. Limita **revisiones iniciales**. Escalaciones directas y posteriores se contabilizan por separado; no se afirma que haya capacidad real suficiente para esa cola. Se informa sensibilidad a 3 % y 10 %.
`04_modelo_final.ipynb` guarda `daily_workload.csv` con revisiones, escalaciones directas, handoffs esperados y demanda total por día y escenario.

La elección minimiza el peor arrepentimiento porcentual frente al menor costo de validación de cada escenario. Los desempates son menor arrepentimiento medio, mayor PR-AUC y menor complejidad: regresión logística, árbol, LightGBM, XGBoost; una actualización periódica suma carga operativa y las ventanas más cortas implican más ajustes. Los umbrales y esta regla quedan congelados antes del holdout.

## Reproducción

1. Descargar los CSV de la competencia IEEE-CIS en `../ieee-fraud-detection/`, o adjuntar `ieee-fraud-detection` a cada kernel privado.
2. Ejecutar `python build_notebooks.py` tras cambiar `core.py` o `stages.py`. Los notebooks generados contienen las funciones y corren de arriba abajo sin importar archivos auxiliares.
3. Ejecutar los cuatro notebooks en el orden de la tabla. En local: `python run.py eda`, luego `experimentacion`, `costos` y `modelo_final`. Para preparar kernels de Kaggle, usar `python build_notebooks.py --owner CUENTA_AUTORIZADA` tras verificar la cuenta activa. Subir cada `kaggle/<etapa>/input/` después de que la etapa anterior termine, descargar los archivos en `outputs/<etapa>/` y comprobar `run_summary.json`.
4. Ejecutar `python verify.py` tras descargar todos los outputs. `python verify.py --notebooks-only` valida sólo la estructura de notebooks.

Los kernels se configuran como privados y CPU. Cada etapa guarda un manifiesto de estado. El kernel de costos depende de los outputs de experimentación; el final depende de ambos. Si una ejecución falla, `run_summary.json` identifica el error; no se reutilizan outputs de corridas anteriores como si fueran nuevos. El usuario autorizó `badexample`: las cuatro etapas terminaron y sus outputs IEEE-CIS se descargaron. `python verify.py` pasó sobre ellos. La prueba sintética local no es evidencia de desempeño. El estado preciso de cada etapa está en `run_manifest.json`.

La validación eligió XGBoost periódico de 30 días con umbrales 0.3872/0.6521 y cupo de 159 revisiones iniciales por día. En el holdout histórico obtuvo PR-AUC 0.548 y, bajo eficacia 80/95, costo simulado 12.312 por transacción. El informe de siete páginas, la presentación y la infografía A3 están en `../07_latex/` y se basan en estos outputs nuevos. `../07_latex/make_figures.py` reconstruye los gráficos compartidos desde los CSV guardados.

## Política de calidad

La cota de PR-AUC semanal es el percentil 10 de las semanas de los folds internos del modelo elegido. Dos bloques semanalmente observados **con etiquetas maduras** por debajo de esa cota disparan un candidato de ventana reciente de 60 días. El candidato se ajusta antes de una semana de calibración, usa otra semana posterior como gate y se promueve sólo si su peor costo de escenario y PR-AUC en ese gate mejoran frente al vigente. PSI/KS son diagnósticos de shift, no disparadores aislados. La bitácora registra tanto mantener como rechazar/promover, y no inventa una mejora si no se observa un disparo.
`quality_ledger.csv` conserva cada control semanal; `monthly_quality_ledger.csv` resume métrica madura, umbral, acción y PR-AUC observado después de la primera acción del mes.
