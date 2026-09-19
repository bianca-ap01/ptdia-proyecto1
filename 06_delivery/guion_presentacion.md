# Guion de presentacion 15-20 minutos

## Distribucion sugerida

| Tiempo | Seccion | Responsable sugerido |
| ---: | --- | --- |
| 2 min | Problema, dataset y objetivos | Integrante 1 |
| 3 min | EDA temporal y decisiones de preparacion | Integrante 2 |
| 4 min | Modelado V01-V04 y seleccion del modelo | Integrante 3 |
| 4 min | Politica de decision V05 y robustez V06 | Integrante 4 |
| 3 min | Arquitectura, despliegue y drift/adaptacion | Integrante 5 |
| 2 min | Limitaciones, trabajo futuro y cierre | Equipo |

## Mensaje central

No construimos solo un clasificador. Construimos el diseno de un sistema adaptativo para fraude: estima riesgo, toma decisiones bajo capacidad limitada, monitorea drift y define cuando recalibrar o reentrenar.

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

### 3. Modelado

- Split temporal 70/15/15.
- V01 LightGBM fue el baseline principal.
- V01b con todas las features no mejoro.
- V02 UID mejoro UID conocido pero empeoro UID desconocido.
- V04 XGBoost/CatBoost/feature sets auditados no superaron V01.

### 4. Decision operativa

- V05 define approve/review/escalate.
- Costos: fraude aprobado 100, legitima escalada 10, revision 2.
- V05 en holdout: costo 1.3680, fraude detectado 66.17%, revision 5.10%.
- Supera reglas simples.

### 5. Drift y despliegue

- Propuesta GCP: FastAPI, Docker, Artifact Registry, Cloud Build, Cloud Run.
- Monitoreo: PR-AUC, KS/PSI, score drift, capacidad de revision.
- Adaptacion: recalibrar, ajustar umbrales o reentrenar con ventana deslizante.

## Preguntas probables y respuestas

| Pregunta | Respuesta breve |
| --- | --- |
| Por que PR-AUC y no accuracy? | Porque el fraude es 3.50%; accuracy puede ser alta prediciendo todo como legitimo. |
| Por que no split aleatorio? | Porque el sistema operaria sobre futuro; mezclar tiempos genera leakage temporal. |
| Por que no usar `TransactionDT` como predictor? | Porque codifica posicion temporal absoluta y podria sobreajustar al periodo. |
| Por que V01 y no V01b? | V01b usa mas features, pero empeora holdout global. |
| Por que no adoptar UID features? | Mejoran UID conocido, pero empeoran UID desconocido y la mejora global es minima. |
| Que pasa si cambia el costo de bloquear? | V06 muestra que la politica escala menos y reduce friccion cuando ese costo sube. |
| Como detectan drift? | Con KS/PSI, cambios de score, missingness, PR-AUC por ventana y costo operativo. |
| Como se adapta el sistema? | Recalibrando probabilidades/umbrales o reentrenando con ventana deslizante. |

## Cierre recomendado

La solucion final recomendada es V01 + V05 + monitoreo V06. Es una decision conservadora, pero defendible: prioriza generalizacion temporal, trazabilidad, costos operativos y capacidad de adaptacion.
