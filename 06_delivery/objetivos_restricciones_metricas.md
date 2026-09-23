# Objetivos, restricciones y metricas del sistema

> **Borrador histórico V05/V06.** Las métricas de costo y acción de esta versión preceden a la política final con cupo diario. Consulte `final/REPORTE.md` para cifras actuales.

## Caso de uso

El sistema propuesto es un sistema inteligente adaptativo para deteccion de fraude en transacciones electronicas. Cada transaccion recibe un score de riesgo y una decision operativa: aprobar, enviar a revision manual o escalar/bloquear.

Las metricas y decisiones reportadas provienen de kernels publicos de Kaggle para EDA y V01-V07, con outputs descargados al repositorio para trazabilidad.

El caso es relevante porque el fraude financiero combina tres condiciones exigidas por el proyecto:

- los datos llegan en el tiempo;
- el comportamiento de clientes y atacantes cambia;
- los errores tienen costos asimetricos.

## Objetivos

| Tipo | Objetivo | Como se mide |
| --- | --- | --- |
| Tecnico | Detectar transacciones fraudulentas en un dataset desbalanceado. | PR-AUC, ROC-AUC, recall de fraude y degradacion validacion-holdout. |
| Temporal | Mantener desempeno razonable en ventanas futuras. | Comparacion validacion vs holdout y metricas por ventana temporal. |
| Operativo | Convertir el score en acciones de negocio. | Costo esperado, tasa de revision, tasa de escalamiento y fraude detectado. |
| Adaptativo | Detectar cambios de distribucion y gatillar recalibracion o reentrenamiento. | KS/PSI, cambios de tasa de fraude, drift de score, caida de PR-AUC. |
| Social | Reducir friccion injustificada y evitar automatizacion excesiva. | Tasa de legitimas escaladas, cobertura por segmento UID conocido/desconocido y revision humana en casos ambiguos. |

## Restricciones

| Restriccion | Implicancia para el diseno |
| --- | --- |
| Datos temporales | No se permite split aleatorio para evaluar el modelo. |
| Desbalance de clase | Accuracy no es metrica principal; se prioriza PR-AUC y deteccion de la clase minoritaria. |
| Datos anonimizados | Las variables se interpretan como senales predictivas, no como causas de fraude. |
| Riesgo de leakage temporal | `TransactionDT`, `DT_day_index` y `DT_week_index` no se usan como predictores. |
| Capacidad humana limitada | La cola de revision se limita alrededor de 5% de transacciones. |
| Costo de falsos positivos | Escalar una transaccion legitima genera friccion y posible perdida de confianza. |
| Autonomia limitada | El sistema recomienda y automatiza aprobaciones de bajo riesgo, pero deja revision humana para casos intermedios. |
| Drift | Los umbrales y el modelo deben monitorearse; no se asume estacionariedad. |
| Privacidad y cumplimiento | En despliegue real se deben aplicar controles de acceso, minimizacion de datos y logs sin informacion sensible innecesaria. |

## Metricas tecnicas

| Metrica | Uso | Justificacion |
| --- | --- | --- |
| PR-AUC | Metrica principal de modelo. | Es mas informativa con clases desbalanceadas porque se enfoca en precision y recall de la clase positiva. |
| ROC-AUC | Metrica complementaria. | Mide capacidad de ranking global, pero puede verse optimista con fuerte desbalance. |
| Recall de fraude | Sensibilidad al fraude real. | Mide que proporcion del fraude es capturada por revision o escalamiento. |
| Precision | Calidad de alertas. | Importa para no saturar al equipo de revision. |
| Brier score | Calidad probabilistica del score calibrado. | Se usa cuando el score se comunica como probabilidad. |
| PR-AUC por ventana | Monitoreo temporal. | Detecta degradacion que una metrica global puede ocultar. |

## Metricas de decision

| Metrica | Definicion | Resultado base |
| --- | --- | --- |
| Costo por transaccion | Costo normalizado promedio bajo una politica. | V05 holdout: 1.3680. |
| Fraude detectado | Fraude enviado a revision o escalamiento. | V05 holdout: 66.17%. |
| Fraude no detectado | Fraude aprobado automaticamente. | V05 holdout: 33.83%. |
| Tasa de revision | Transacciones enviadas a revision manual. | V05 holdout: 5.10%. |
| Tasa de escalamiento | Transacciones de riesgo extremo. | V05 holdout: 2.48%. |
| Aprobacion automatica | Transacciones aprobadas sin intervencion. | V05 holdout: 92.42%. |

## Metricas sociales y de riesgo

| Metrica | Motivo |
| --- | --- |
| Tasa de legitimas escaladas | Aproxima friccion injustificada para usuarios legitimos. |
| Desempeno UID conocido vs desconocido | Evita ocultar degradacion en clientes nuevos o menos observados. |
| Cobertura de decision automatica | Mide cuanto del flujo queda sin revision humana. |
| Porcentaje de casos ambiguos revisados | Controla riesgo de automatizar decisiones inciertas. |
| Estabilidad de umbrales | Detecta si una politica deja de respetar capacidad operativa por drift. |

## Decision final vinculada a metricas

La metrica de seleccion del modelo es PR-AUC en holdout temporal, complementada por desempeno por segmento UID. La metrica de seleccion de politica es costo por transaccion bajo restricciones de capacidad.

Con esa regla:

- V01 queda como modelo base porque tiene mejor holdout PR-AUC global entre las variantes probadas.
- V05 queda como politica operativa porque reduce costo frente a aprobar todo, revisar top 5% y escalar top 1%.
- V06 queda como auditoria porque demuestra sensibilidad coherente a costos y necesidad de calibracion.
- V07 queda como benchmark adicional porque el MLP no supera a LightGBM en holdout.
