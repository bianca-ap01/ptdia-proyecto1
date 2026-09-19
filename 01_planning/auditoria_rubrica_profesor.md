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
| Caso de uso concreto y relevante | Cubierto parcialmente | Fraude financiero IEEE-CIS esta definido en el plan y consolidado. | Falta explicitar mejor contexto real, usuarios del sistema y alcance de decisiones. |
| Objetivos del sistema | Cubierto parcialmente | `modelado_consolidado.md` define deteccion, score y decisiones. | Falta separar objetivos tecnicos, operativos y sociales en una tabla final. |
| Restricciones | Parcial | Se menciona capacidad de revision 5%, drift, costos normalizados y no usar tiempo crudo. | Falta lista formal de restricciones: datos anonimizados, latencia, capacidad humana, no automatizar bloqueo total sin revision, privacidad. |
| Metricas tecnicas | Cubierto | PR-AUC, ROC-AUC y resultados por segmento. | Agregar F1/recall/precision si estan disponibles o justificar por que PR-AUC domina. |
| Metricas de decision | Cubierto | V05 costo esperado, fraude detectado, revision, escalamiento. | Mejorar tabla de costo y umbrales en el informe de 8 paginas. |
| Metricas sociales | Debil | Se menciona friccion de cliente y UID conocido/desconocido. | Falta formalizar cobertura, friccion, falsos positivos y riesgo de trato desigual por segmento. |
| Arquitectura del sistema | Falta como visual | La logica existe en texto, no en diagrama final. | Crear diagrama de arquitectura end-to-end. |
| Adquisicion e integracion de datos | Parcial | Kernels cargan IEEE-CIS y unen transaction/identity. | Falta describir flujo de datos como sistema real. |
| Modulo predictivo | Cubierto | V01 LightGBM como modelo base. | Documentar entradas/salidas del modulo. |
| Modulo de toma de decisiones | Cubierto | V05 approve/review/escalate. | Incluir en arquitectura y explicar responsable humano vs automatizado. |
| Manejo de incertidumbre | Parcial | V06 calibracion. | Explicar que score bruto no es probabilidad y donde entra calibracion. |
| Componente de accion | Parcial | V05 define approve/review/escalate. | Describir alertas, cola de revision y escalamiento operativo. |
| EDA temporal | Cubierto | EDA propio, validacion adversarial, tasa semanal, UID, missingness. | Seleccionar visuales concretos para el documento final. |
| Preprocesamiento temporal | Parcial | Split temporal, exclusion de variables temporales crudas. | Falta explicar ventanas deslizantes/rezagos como propuesta de adaptacion. |
| Division temporal train/valid/test | Cubierto | V01-V04 usan 70/15/15 temporal. | Incluir diagrama simple de particion temporal. |
| Al menos dos enfoques tradicionales y uno avanzado | Parcial | LightGBM, XGBoost, CatBoost. | El profesor dice "tradicionales" y "avanzado"; conviene etiquetar: Logistic Regression/Random Forest como tradicionales, boosting como avanzado. Si no se entrenan, justificar que LightGBM/XGBoost/CatBoost son comparativos de boosting, pero podria bajar puntos. |
| Comparacion de modelos | Cubierto parcialmente | V04 compara LightGBM/XGBoost/CatBoost sobre features auditadas. | Falta una tabla final clara y quizas un baseline tradicional simple. |
| Degradacion temporal | Cubierto | Validacion vs holdout, metricas por ventana. | Incluir grafico o tabla de degradacion temporal. |
| Propuesta de despliegue | Debil | V05/V06 sugieren decision operativa, pero no despliegue completo. | Crear seccion: batch/stream, frecuencia, monitoreo, responsables, escalabilidad. |
| Deteccion de drift | Parcial | KS train-valid/holdout en V03 y adversarial validation en EDA. | Falta presentarlo como protocolo de drift operativo. |
| Adaptacion al drift | Debil | V06 recomienda recalibracion y ventanas. | Falta simular o proponer explicitamente ventanas deslizantes y reentrenamiento periodico. |
| Documento tecnico max. 8 paginas | Pendiente | Existe consolidado largo. | Hay que condensarlo a 8 paginas. |
| Infografia final | Pendiente | No existe todavia. | Crear infografia de ciclo de vida IA. |
| Presentacion 15-20 min | Pendiente | No existe deck final. | Preparar guion o slides. |

## 3. Riesgo principal de nota

La parte de modelado esta fuerte, pero la rubrica asigna 4 puntos a diseno del sistema y adaptacion, y 4 puntos a calidad del producto. Ahi todavia hay brechas:

- falta arquitectura visual del sistema;
- falta propuesta de despliegue;
- falta protocolo de drift/adaptacion;
- falta infografia final;
- falta condensar el documento a maximo 8 paginas;
- falta posiblemente un baseline tradicional adicional si el docente interpreta estrictamente "dos enfoques tradicionales y uno avanzado".

## 4. Acciones recomendadas antes de entregar

### Prioridad 1 - cerrar rubrica tecnica

1. Crear un documento tecnico de maximo 8 paginas.
2. Agregar diagrama de arquitectura del sistema.
3. Agregar protocolo de drift y adaptacion con ventanas deslizantes.
4. Definir objetivos, restricciones y metricas en tabla formal.
5. Agregar seccion de despliegue: flujo de datos, autonomia, frecuencia de actualizacion, costos y escalabilidad.

### Prioridad 2 - fortalecer modelado segun enunciado

1. Evaluar si conviene entrenar un baseline tradicional adicional:
   - Logistic Regression con imputacion/encoding simple;
   - Random Forest o HistGradientBoosting como tradicional/intermedio;
   - mantener LightGBM/XGBoost/CatBoost como avanzados.
2. Si el tiempo no alcanza, justificar que V01-V04 comparan modelos de boosting y que la prioridad fue evaluacion temporal robusta.

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

Lo construido hasta ahora es una base muy buena para las secciones de:

- analisis de datos;
- modelado temporal;
- comparacion de modelos;
- politica de decision;
- calibracion;
- sensibilidad de costos.

Pero todavia no basta para cubrir "excelente" en toda la rubrica. Para maximizar nota, el siguiente trabajo debe moverse de experimentos de modelo hacia producto final: arquitectura, despliegue, drift/adaptacion, documento de 8 paginas e infografia.

## 6. Plan inmediato sugerido

1. Crear `informe_tecnico_8_paginas.md` como version condensada para entrega.
2. Crear diagramas:
   - arquitectura del sistema;
   - split temporal;
   - ciclo de adaptacion ante drift.
3. Crear `protocolo_drift_adaptacion.md` con ventanas fijas/deslizantes, metricas de monitoreo y acciones.
4. Evaluar si ejecutamos un baseline tradicional rapido para cubrir explicitamente la frase "dos enfoques tradicionales y uno avanzado".
