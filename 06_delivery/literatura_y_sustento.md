# Literatura y sustento metodologico

Este documento resume las fuentes usadas para sostener las decisiones faltantes del proyecto.

## Clases desbalanceadas y PR-AUC

El dataset tiene 3.50% de fraude en train. En este contexto, accuracy puede ser enganosa porque predecir todo como legitimo produce un valor alto sin detectar fraude.

Saito y Rehmsmeier (2015) sostienen que Precision-Recall es mas informativa que ROC cuando se evalua clasificacion binaria sobre datos desbalanceados. Por eso el proyecto usa PR-AUC como metrica principal y ROC-AUC como complemento.

Uso en el proyecto:

- V01 se selecciona por holdout PR-AUC.
- V04 compara modelos con PR-AUC.
- V05 traduce ranking a decision operativa y fue publicado como kernel reproducible.

## Concept drift y evaluacion temporal

Gama et al. (2014) definen concept drift como cambios temporales en la relacion entre entradas y variable objetivo. Esto aplica al fraude: los atacantes cambian comportamiento, aparecen nuevos patrones y la tasa de fraude varia por semana.

Uso en el proyecto:

- split temporal 70/15/15;
- no usar split aleatorio;
- medir validacion vs holdout;
- monitorear degradacion temporal.

## Ventanas deslizantes y olvido

Bifet y Gavalda (2007) proponen Adaptive Windowing para datos que cambian con el tiempo. La idea central es no tratar todo el historial como igualmente valido: cuando hay cambio, conviene olvidar datos antiguos o reducir la ventana efectiva.

Uso en el proyecto:

- propuesta de reentrenamiento con ventana fija deslizante;
- monitoreo por ventana corta/media/historica;
- gatillos de drift para recalibracion o reentrenamiento.

## Calibracion de probabilidades

Niculescu-Mizil y Caruana (2005) muestran que modelos como boosting pueden rankear bien pero producir probabilidades mal calibradas, y que Platt scaling o isotonic regression pueden corregir sesgos. La documentacion de scikit-learn tambien advierte que un clasificador bien calibrado permite interpretar `predict_proba` como frecuencia esperada.

Uso en el proyecto:

- V06 encontro que el score bruto promedio en holdout era 14.30% mientras la tasa real era 3.48%.
- La calibracion isotonica redujo Brier de 0.0479 a 0.0219 y alinio el score promedio con la tasa real.
- Decision: score bruto para ranking, score calibrado para probabilidad/comunicacion.
- La corrida publica V06 deja trazables la calibracion, la sensibilidad de costos y la comparacion global vs segmentada.

## Despliegue en Cloud Run

El taller de despliegue y la documentacion oficial de Google Cloud sostienen un flujo reproducible:

1. API;
2. Docker;
3. Artifact Registry;
4. Cloud Build;
5. Cloud Run;
6. Cloud Logging/Monitoring.

Uso en el proyecto:

- API FastAPI `/predict`;
- contenedor con modelo y politica;
- build automatico desde GitHub;
- imagen versionada en Artifact Registry;
- servicio serverless en Cloud Run;
- logs para auditoria y drift.

## Fuentes

- Saito, T. y Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*.
- Gama, J., Zliobaite, I., Bifet, A., Pechenizkiy, M. y Bouchachia, A. (2014). *A Survey on Concept Drift Adaptation*.
- Bifet, A. y Gavalda, R. (2007). *Learning from Time-Changing Data with Adaptive Windowing*.
- Niculescu-Mizil, A. y Caruana, R. (2005). *Predicting Good Probabilities with Supervised Learning*.
- Zadrozny, B. y Elkan, C. (2002). *Transforming Classifier Scores into Accurate Probability Estimates*.
- scikit-learn documentation: probability calibration and time series validation.
- Google Cloud documentation: Cloud Run, Cloud Build, Artifact Registry.
