# Guion de presentacion 15-20 minutos

> **Guion histórico V01–V07.** Las cifras de V05 aquí citadas no son las de la evaluación final. Para la exposición, use el informe y la presentación de `07_latex/` contrastados con `final/REPORTE.md`.

## Distribucion sugerida

| Tiempo | Seccion | Responsable sugerido |
| ---: | --- | --- |
| 2 min | Problema, alcance, objetivos y restricciones | Integrante 1 |
| 4 min | EDA temporal, EDA de referencia y preparacion | Integrante 2 |
| 5 min | Diseno experimental V01-V07 y seleccion final | Integrante 3 |
| 4 min | Politica V05, incertidumbre y calibracion V06 | Integrante 4 |
| 3 min | Arquitectura, despliegue GCP y drift/adaptacion | Integrante 5 |
| 2 min | Rubrica, limitaciones y cierre | Equipo |

## Mensaje central

No construimos solo un clasificador. Construimos el diseno de un sistema adaptativo para fraude: estima riesgo, toma decisiones bajo capacidad limitada, monitorea drift y define cuando recalibrar o reentrenar. La defensa debe enfatizar el por que de cada decision y la evidencia que la sostiene.

## Puntos clave por seccion

### 1. Problema

- Dataset IEEE-CIS Fraud Detection.
- Fraude financiero con comportamiento cambiante.
- Clase positiva minoritaria: 3.50%.
- El objetivo no es accuracy, sino detectar fraude y tomar decisiones.

### 2. EDA

- Tasa semanal de fraude varia entre 1.85% y 5.06%.
- Train/test tienen fuerte separacion temporal.
- Hay diferencias entre UID conocido y desconocido.
- Muchos faltantes son estructurales, no ruido aleatorio.
- El EDA de referencia complemento la lectura con bloques V, faltantes de identity, cola pesada del monto, montos multi-decimales y comparacion con MLP.
- No se copiaron sus metricas porque usa otro split y otra politica de costos.

### 3. Modelado

- Split temporal 70/15/15.
- V01 LightGBM fue el baseline principal.
- V01b con todas las features no mejoro.
- V02 UID mejoro UID conocido pero empeoro UID desconocido.
- V04 XGBoost/CatBoost/feature sets auditados no superaron V01.
- V07 MLP no supero LightGBM: holdout PR-AUC 0.2218 vs 0.5313.
- EDA y V01-V07 estan publicados como kernels publicos en Kaggle.
- La pregunta de cada version debe decirse explicitamente: V01 base, V01b todas las features, V02 memoria UID, V03 auditoria, V04 modelos alternativos, V05 decision, V06 incertidumbre y V07 red neuronal/baseline tradicional.

### 4. Decision operativa

- V05 define approve/review/escalate.
- Costos: fraude aprobado 100, legitima escalada 10, revision 2.
- V05 en holdout: costo 1.3680, fraude detectado 66.17%, revision 5.10%.
- Supera reglas simples.
- V06 confirma que el score bruto no debe comunicarse como probabilidad sin calibracion.

### 5. Drift y despliegue

- Propuesta GCP: FastAPI, Docker, Artifact Registry, Cloud Build, Cloud Run.
- Monitoreo: PR-AUC, KS/PSI, score drift, capacidad de revision.
- Adaptacion: recalibrar, ajustar umbrales o reentrenar con ventana deslizante.

### 6. Rubrica y cierre

- Problema, datos, modelado, sistema adaptativo, producto y presentacion quedan cubiertos.
- La infografia final mapea cada criterio de la rubrica a evidencia concreta.
- La unica brecha formal posible es si el docente exige literalmente dos modelos tradicionales distintos; en ese caso conviene agregar Random Forest o HistGradientBoosting simple.

## Preguntas probables y respuestas

| Pregunta | Respuesta breve |
| --- | --- |
| Por que PR-AUC y no accuracy? | Porque el fraude es 3.50%; accuracy puede ser alta prediciendo todo como legitimo. |
| Por que no split aleatorio? | Porque el sistema operaria sobre futuro; mezclar tiempos genera leakage temporal. |
| Por que no usar `TransactionDT` como predictor? | Porque codifica posicion temporal absoluta y podria sobreajustar al periodo. |
| Por que V01 y no V01b/MLP? | V01b usa mas features pero empeora holdout; V07 MLP queda por debajo de LightGBM. |
| Por que no adoptar UID features? | Mejoran UID conocido, pero empeoran UID desconocido y la mejora global es minima. |
| Que pasa si cambia el costo de bloquear? | V06 muestra que la politica escala menos y reduce friccion cuando ese costo sube. |
| Como detectan drift? | Con KS/PSI, cambios de score, missingness, PR-AUC por ventana y costo operativo. |
| Como se adapta el sistema? | Recalibrando probabilidades/umbrales o reentrenando con ventana deslizante. |

## Cierre recomendado

La solucion final recomendada es V01 + V05 + monitoreo/calibracion V06, con V07 como benchmark adicional. Es una decision conservadora y defendible: prioriza generalizacion temporal, trazabilidad publica en Kaggle, costos operativos y capacidad de adaptacion.
