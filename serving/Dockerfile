# LightGBM necesita libgomp (OpenMP), que la imagen slim no trae.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY artifacts ./artifacts

# Cloud Run inyecta PORT en tiempo de ejecucion; 8080 es el valor del contrato.
ENV PORT=8080
EXPOSE 8080

# Un worker: el modelo ocupa memoria y Cloud Run escala por instancias,
# no por procesos dentro del contenedor.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
