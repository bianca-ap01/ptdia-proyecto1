# API de decisión antifraude — FastAPI en Cloud Run

Sirve el modelo del proyecto como servicio HTTP: score, probabilidad calibrada
y acción de la política en una sola respuesta.

## Endpoints

| Método | Ruta | Devuelve |
| --- | --- | --- |
| `GET` | `/health` | Estado, versión del modelo y umbrales vigentes |
| `POST` | `/predict` | `score`, `calibrated_probability`, `action`, `latency_ms` |
| `GET` | `/metrics` | PSI del score contra validación y reparto de acciones |
| `GET` | `/docs` | OpenAPI interactivo (lo genera FastAPI) |

Ejemplo:

```bash
curl -X POST "$URL/predict" -H 'Content-Type: application/json' -d '{
  "TransactionAmt": 315.0,
  "TransactionDT": 13500000,
  "features": {"ProductCD": "W", "card1": 7919, "DeviceType": "desktop", "C13": 5, "D1": 100}
}'
```

```json
{
  "score": 0.663505,
  "calibrated_probability": 0.139803,
  "action": "escalate",
  "thresholds": {"review": 0.326006, "escalate": 0.624796},
  "model_version": "lightgbm_v01_static",
  "latency_ms": 1.502
}
```

Solo `TransactionAmt` es obligatorio. Las demás variables van en `features`;
las que falten quedan como nulo, que es lo que el modelo vio en entrenamiento
—LightGBM trata nulos de forma nativa y el dataset original tiene faltantes
masivos—. Cuantas más se envíen, mejor informada la decisión.

## Qué se congela y por qué

`train_serving_model.py` serializa tres artefactos que solo tienen sentido
juntos:

- `model.txt` — LightGBM entrenado con la ventana histórica (PR-AUC holdout **0.51910**)
- `calibrator.joblib` — isotónica ajustada **solo con validación**
- `serving_config.json` — umbrales, lista de 185 variables y estadísticos del encoder

Los cuartiles del monto, los mapas y las frecuencias categóricas salen del
entrenamiento y no se recalculan en línea: si se recalcularan con el tráfico
del día, la misma transacción recibiría decisiones distintas según quién más
operara esa hora. Los umbrales son los elegidos en validación y congelados,
los mismos con los que se evaluó el holdout del informe.

## Diferencia esperada con el informe

Sobre 300 transacciones reales de holdout, la API reparte **78.7 % aprobar /
15.3 % revisar / 6.0 % escalar**, mientras el informe reporta 90.2 / 5.5 / 4.3.
No es una discrepancia del modelo: son dos regímenes distintos.

El informe aplica un **cupo de 158 revisiones diarias**, que trunca la cola de
revisión y devuelve el excedente a aprobación. La API decide transacción a
transacción, sin conocer el volumen del día ni la capacidad restante. Para
reproducir el comportamiento del informe hace falta un componente de cola que
lleve el conteo diario; servirlo sin cupo equivale al escenario de capacidad
infinita, que el propio estudio marca como límite optimista.

## Servicio desplegado

**https://fraude-api-agz3gnkzgq-uc.a.run.app** — proyecto `ptdia-fraude-9491`,
región `us-central1`.

```bash
curl https://fraude-api-agz3gnkzgq-uc.a.run.app/health
```

Latencia medida desde Lima sobre doce solicitudes: **mediana 404 ms**
(rango 364–617), contra **2.8 ms** de cómputo interno. Cerca del 99 % del
tiempo es red, no modelo.

## Desplegar desde cero

Requiere una cuenta de facturación **activa**; Cloud Run y Cloud Build no
operan sin ella.

```bash
PROJECT=ptdia-fraude
REGION=us-central1

gcloud projects create $PROJECT
gcloud config set project $PROJECT
gcloud billing projects link $PROJECT --billing-account=<ID_ABIERTO>

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
                       artifactregistry.googleapis.com

gcloud artifacts repositories create ptdia \
  --repository-format=docker --location=$REGION

cd serving
gcloud builds submit --config cloudbuild.yaml
```

`cloudbuild.yaml` construye la imagen, la publica en Artifact Registry y
despliega en Cloud Run en un solo paso. Al terminar:

```bash
URL=$(gcloud run services describe fraude-api --region=$REGION --format='value(status.url)')
curl "$URL/health"
```

## Probar en local

```bash
cd serving
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
# http://localhost:8080/docs
```

O con el contenedor, que es lo que corre en Cloud Run:

```bash
docker build -t fraude-api .
docker run -p 8080:8080 fraude-api
```

## Límites

El servicio está abierto (`--allow-unauthenticated`) para facilitar la
demostración. Antes de exponer algo equivalente en operación haría falta
autenticación, límite de tasa, registro de auditoría de cada decisión y
cuotas por cliente.

`/metrics` mantiene una ventana en memoria de las últimas 5 000 solicitudes.
Cloud Run recicla instancias, así que es telemetría de conveniencia, no
registro durable: lo duradero va al log estructurado que consume Cloud
Logging. El PSI que expone es la misma señal que, en el estudio, **no**
detectó la caída de PR-AUC de 0.732 a 0.473; se publica para vigilancia, no
como alarma única.

Los costos y umbrales provienen de una simulación con supuestos declarados.
Ninguna cifra procede de operación real.
