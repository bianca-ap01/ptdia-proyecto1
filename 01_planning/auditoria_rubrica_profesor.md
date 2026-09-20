# Auditoria contra planteo y rubrica del profesor

Fuente revisada: `p1/00_brief/UTEC_2026_1__Planificación_y_Toma_de_Decisiones_en_IA (1).pdf`

## 1. Lectura general

El proyecto exige construir un sistema inteligente adaptativo con datos no estacionarios. Para nuestro caso, el dataset IEEE-CIS Fraud Detection encaja bien porque tiene transacciones ordenadas temporalmente, drift en la tasa de fraude y cambios de distribucion entre ventanas.

La rubrica no premia solo "tener un buen modelo". Evalua seis dimensiones:

| Criterio | Puntos | Lectura para nuestro proyecto |
| --- | ---: | --- |
| Definicion del problema, objetivos, restricciones y metricas | 4 | Debemos declarar claramente que el sistema detecta fraude, bajo restricciones de capacidad, costos, friccion y temporalidad. |
| Analisis y preparacion de datos | 3 | El EDA debe mostrar evolucion temporal, drift, anomalías, faltantes y decisiones de preprocesamiento. |
| Modelado y evaluacion temporal | 3 | Hay que comparar modelos y analizar degradacion en el tiempo. |
| Diseno del sistema y estrategia de adaptacion | 4 | Debe haber arquitectura, flujo de datos, modulo predictivo, toma de decisiones, incertidumbre, accion y adaptacion al drift. |
| Calidad del producto entregado | 4 | Codigo claro, reproducible, documento tecnico, diagramas, tablas, protocolos e infografia. |
| Presentacion del avance | 2 | El equipo debe poder explicar hacia donde va el proyecto y defender decisiones. |

## 2. Cobertura actual del proyecto

| Exigencia del PDF | Estado actual | Evidencia actual | Brecha |
| --- | --- | --- | --- |
| Caso de uso concreto y relevante | Cubierto | Fraude financiero IEEE-CIS esta definido en el plan, consolidado e informe tecnico. | Mantener el alcance: apoyo a decision, no bloqueo irreversible sin control humano. |
| Objetivos del sistema | Cubierto | `06_delivery/objetivos_restricciones_metricas.md` separa objetivos tecnicos, temporales, operativos, adaptativos y sociales. | Sin brecha relevante. |
| Restricciones | Cubierto | `06_delivery/objetivos_restricciones_metricas.md` formaliza temporalidad, capacidad humana, costos, privacidad, drift y autonomia limitada. | Sin brecha relevante. |
| Metricas tecnicas | Cubierto | PR-AUC, ROC-AUC y resultados por segmento. | Agregar F1/recall/precision si estan disponibles o justificar por que PR-AUC domina. |
| Metricas de decision | Cubierto | V05 costo esperado, fraude detectado, revision, escalamiento; kernel publico V05 completo. | Sin brecha relevante. |
| Metricas sociales | Cubierto | Se formalizan friccion, legitimas escaladas, UID conocido/desconocido y revision humana. | Mantener limitacion: no hay atributos demograficos para equidad directa. |
| Arquitectura del sistema | Cubierto | `06_delivery/arquitectura_sistema_mermaid.md` e informe tecnico incluyen diagrama end-to-end. | Sin brecha relevante. |
| Adquisicion e integracion de datos | Cubierto | Kernels cargan IEEE-CIS, unen transaction/identity y el despliegue describe flujo real. | Sin brecha relevante. |
| Modulo predictivo | Cubierto | V01 LightGBM como modelo base. | Documentar entradas/salidas del modulo. |
| Modulo de toma de decisiones | Cubierto | V05 approve/review/escalate. | Incluir en arquitectura y explicar responsable humano vs automatizado. |
| Manejo de incertidumbre | Cubierto | V06 publico reporta calibracion isotonica y Brier; delivery explica score bruto vs probabilidad. | Sin brecha relevante. |
| Componente de accion | Cubierto | V05 define approve/review/escalate; arquitectura y despliegue describen cola y escalamiento. | Sin brecha relevante. |
| EDA temporal | Cubierto | EDA propio, validacion adversarial, tasa semanal, UID, missingness. | Seleccionar visuales concretos para el documento final. |
| Preprocesamiento temporal | Cubierto | Split temporal, exclusion de variables temporales crudas y protocolo de ventanas deslizantes. | Sin brecha relevante. |
| Division temporal train/valid/test | Cubierto | V01-V06 usan 70/15/15 temporal; V05/V06 autocontenidos reentrenan V01 antes de politica/robustez. | Diagrama incluido en delivery y LaTeX. |
| Al menos dos enfoques tradicionales y uno avanzado | Parcial | LightGBM, XGBoost, CatBoost. | El profesor dice "tradicionales" y "avanzado"; conviene etiquetar: Logistic Regression/Random Forest como tradicionales, boosting como avanzado. Si no se entrenan, justificar que LightGBM/XGBoost/CatBoost son comparativos de boosting, pero podria bajar puntos. |
| Comparacion de modelos | Cubierto parcialmente | V04 compara LightGBM/XGBoost/CatBoost sobre features auditadas y el consolidado incluye tabla final. | Si el docente exige literalmente modelos tradicionales, agregar Logistic Regression y Random Forest. |
| Degradacion temporal | Cubierto | Validacion vs holdout, metricas por ventana. | Incluir grafico o tabla de degradacion temporal. |
| Propuesta de despliegue | Cubierto | `06_delivery/propuesta_despliegue_gcp.md` adapta el taller GCP a FastAPI, Docker, Artifact Registry, Cloud Build y Cloud Run. | Sin brecha relevante. |
| Deteccion de drift | Cubierto | `06_delivery/protocolo_drift_adaptacion.md` define KS/PSI, score drift, missingness, performance y alertas. | Sin brecha relevante. |
| Adaptacion al drift | Cubierto | Protocolo propone recalibracion, ajuste de umbrales y reentrenamiento con ventanas deslizantes. | Simular ventanas queda como trabajo futuro deseable. |
| Documento tecnico max. 8 paginas | Cubierto | `06_delivery/informe_tecnico_8_paginas.md` y `07_latex/informe.pdf`. | Verificar extension final tras compilacion. |
| Infografia final | Cubierto | `06_delivery/infografia_final_mermaid.md` y `07_latex/infografia.pdf`. | Sin brecha relevante. |
| Presentacion 15-20 min | Cubierto | `06_delivery/guion_presentacion.md` y `07_latex/presentacion.pdf`. | Completar nombres del equipo. |

## 3. Riesgo principal de nota

La parte de modelado esta fuerte y ya se completo la mayor parte de sistema/adaptacion/producto final. La brecha principal remanente es interpretar literalmente la exigencia de modelos "tradicionales":

- falta posiblemente un baseline tradicional adicional si el docente interpreta estrictamente "dos enfoques tradicionales y uno avanzado".

## 4. Acciones recomendadas antes de entregar

### Prioridad 1 - cerrar rubrica tecnica

1. Verificar que el PDF final compile correctamente.
2. Mantener los enlaces publicos de Kaggle y GitHub visibles.
3. Preparar defensa oral de por que V05/V06 son politica y auditoria, no modelos nuevos.
4. Si hay tiempo, entrenar Logistic Regression y Random Forest para cerrar la lectura literal de la rubrica.

### Prioridad 2 - fortalecer modelado segun enunciado

1. Evaluar si conviene entrenar un baseline tradicional adicional:
   - Logistic Regression con imputacion/encoding simple;
   - Random Forest o HistGradientBoosting como tradicional/intermedio;
   - mantener LightGBM/XGBoost/CatBoost como avanzados.
2. Si el tiempo no alcanza, justificar que V01-V04 comparan modelos de boosting y que V05-V06 cubren decision, calibracion y robustez operacional.

### Prioridad 3 - producto final

1. Crear infografia del ciclo de vida:
   - datos transaccionales;
   - EDA temporal;
   - entrenamiento;
   - score de fraude;
   - politica approve/review/escalate;
   - monitoreo de drift;
   - recalibracion/reentrenamiento.
2. Preparar presentacion de 15-20 minutos.
3. Preparar respuestas a preguntas sobre:
   - por que PR-AUC;
   - por que no split aleatorio;
   - por que no usar `TransactionDT`;
   - por que V01 y no V01b/V02/V04;
   - como se detecta drift;
   - que pasa si cambia el costo de revisar o bloquear.

## 5. Decision sobre lo que ya tenemos

Lo construido hasta ahora cubre las secciones de:

- analisis de datos;
- modelado temporal;
- comparacion de modelos;
- politica de decision;
- calibracion;
- sensibilidad de costos;
- arquitectura, despliegue, drift/adaptacion, informe, infografia y presentacion.

Para maximizar nota, la unica mejora tecnica de bajo riesgo pendiente es agregar modelos tradicionales simples si el profesor exige esa categoria de forma estricta.

## 6. Plan inmediato sugerido

1. Recompilar PDFs en `07_latex/`.
2. Confirmar que README y reportes enlazan los kernels publicos.
3. Evaluar si ejecutamos un baseline tradicional rapido para cubrir explicitamente la frase "dos enfoques tradicionales y uno avanzado".
