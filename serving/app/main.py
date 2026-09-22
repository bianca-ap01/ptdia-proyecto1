"""API de decision antifraude.

Sirve las tres piezas que el informe describe como un solo acto de decision:
el score del modelo, la probabilidad calibrada y la accion de la politica.
Separarlas seria enganoso: un score sin umbral no es una decision, y un umbral
sin calibracion no se puede comunicar como probabilidad.

Endpoints:
  GET  /health   estado y version del modelo
  POST /predict  score, probabilidad calibrada y accion
  GET  /metrics  PSI del score contra validacion y reparto de acciones
"""

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import lightgbm as lgb
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
DAY = 86400
# Ventana de observacion en memoria. Cloud Run puede reciclar la instancia, asi
# que esto es telemetria de conveniencia, no un registro de auditoria: lo
# duradero va al log estructurado que consume Cloud Logging.
WINDOW = 5000

config: dict[str, Any] = json.loads((ARTIFACTS / "serving_config.json").read_text())
booster = lgb.Booster(model_file=str(ARTIFACTS / "model.txt"))
calibrator = joblib.load(ARTIFACTS / "calibrator.joblib")

FEATURES: list[str] = config["features"]
ENCODER: dict[str, Any] = config["encoder"]
REVIEW_THR: float = config["thresholds"]["review"]
ESCALATE_THR: float = config["thresholds"]["escalate"]
REFERENCE_DECILES: list[float] = config["reference_score_deciles"]

recent_scores: deque = deque(maxlen=WINDOW)
recent_actions: deque = deque(maxlen=WINDOW)

app = FastAPI(
    title="Sistema adaptativo de decision antifraude",
    description=(
        "IEEE-CIS Fraud Detection. Los costos y umbrales provienen de una "
        "simulacion con supuestos declarados, no de operacion real."
    ),
    version=str(config.get("model", "unknown")),
)


class Transaction(BaseModel):
    """Una transaccion. Solo TransactionAmt es obligatorio.

    El resto de las 185 variables se aceptan en `features`; las ausentes
    quedan como NaN, que es exactamente lo que el modelo vio en
    entrenamiento: LightGBM trata los nulos de forma nativa y el dataset
    original tiene faltantes masivos.
    """

    TransactionAmt: float = Field(..., gt=0, description="Monto de la operacion")
    TransactionDT: Optional[float] = Field(
        None, description="Contador temporal relativo en segundos"
    )
    features: dict[str, Union[float, str, None]] = Field(
        default_factory=dict, description="Resto de variables disponibles"
    )


class Decision(BaseModel):
    score: float
    calibrated_probability: float
    action: str
    thresholds: dict[str, float]
    model_version: str
    latency_ms: float


def encode(payload: Transaction) -> np.ndarray:
    """Reaplica la transformacion con los estadisticos congelados.

    Los cuartiles, mapas y frecuencias salen del entrenamiento; recalcularlos
    en linea con el trafico del dia haria que el mismo caso recibiera
    decisiones distintas segun quien mas transaccionara esa hora.
    """
    raw: dict[str, Any] = dict(payload.features)
    raw["TransactionAmt"] = payload.TransactionAmt
    if payload.TransactionDT is not None:
        raw["TransactionDT"] = payload.TransactionDT

    amount = payload.TransactionAmt
    derived: dict[str, float] = {
        "TransactionAmt_log1p": float(np.log1p(amount)),
        "TransactionAmt_outlier_iqr": float(
            amount < ENCODER["amt_low"] or amount > ENCODER["amt_high"]
        ),
        "has_identity": float(raw.get("DeviceType") is not None),
    }
    if payload.TransactionDT is not None:
        derived["DT_hour"] = float((payload.TransactionDT // 3600) % 24)
        day = payload.TransactionDT // DAY
        for col in ("D1", "D2", "D10", "D15"):
            value = raw.get(col)
            if value is not None:
                try:
                    derived[f"{col}n"] = float(value) - day
                except (TypeError, ValueError):
                    pass

    row = np.full(len(FEATURES), np.nan, dtype="float32")
    for i, name in enumerate(FEATURES):
        if name in derived:
            row[i] = derived[name]
        elif name.endswith("_label") and name[:-6] in ENCODER["maps"]:
            key = str(raw.get(name[:-6], "__MISSING__"))
            row[i] = ENCODER["maps"][name[:-6]].get(key, -1)
        elif name.endswith("_freq") and name[:-5] in ENCODER["freqs"]:
            key = str(raw.get(name[:-5], "__MISSING__"))
            row[i] = ENCODER["freqs"][name[:-5]].get(key, 0.0)
        else:
            value = raw.get(name)
            if value is not None:
                try:
                    row[i] = float(value)
                except (TypeError, ValueError):
                    row[i] = np.nan
    return row.reshape(1, -1)


def decide(score: float) -> str:
    """Tres acciones. Escalar es retencion con confirmacion humana,
    nunca bloqueo irreversible automatico."""
    if score >= ESCALATE_THR:
        return "escalate"
    if score >= REVIEW_THR:
        return "review"
    return "approve"


def psi(observed: list, deciles: list) -> float:
    """PSI del score contra los deciles de validacion.

    Vale recordar lo que el trabajo midio: esta senal no detecto la caida de
    PR-AUC de 0.732 a 0.473, porque el drift era de concepto y no de
    covariables. Se expone para vigilancia, no como unica alarma.
    """
    if len(observed) < 100:
        return float("nan")
    values = np.asarray(observed)
    total = 0.0
    for lo, hi in zip(deciles[:-1], deciles[1:]):
        actual = float(((values >= lo) & (values < hi)).mean())
        expected = 0.1
        actual = max(actual, 1e-6)
        total += (actual - expected) * np.log(actual / expected)
    return round(float(total), 5)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": config.get("model"),
        "features": len(FEATURES),
        "thresholds": {"review": REVIEW_THR, "escalate": ESCALATE_THR},
        "pr_auc_holdout": config.get("metrics", {}).get("pr_auc_holdout"),
    }


@app.post("/predict", response_model=Decision)
def predict(payload: Transaction) -> Decision:
    started = time.perf_counter()
    score = float(booster.predict(encode(payload))[0])
    action = decide(score)

    recent_scores.append(score)
    recent_actions.append(action)

    return Decision(
        score=round(score, 6),
        calibrated_probability=round(float(calibrator.predict([score])[0]), 6),
        action=action,
        thresholds={"review": REVIEW_THR, "escalate": ESCALATE_THR},
        model_version=str(config.get("model", "unknown")),
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )


@app.get("/metrics")
def metrics() -> dict:
    """Cierra el lazo de monitoreo: sin etiquetas solo se puede vigilar la
    distribucion del score y el reparto de acciones. La degradacion real
    necesita etiquetas maduras, que llegan con retraso."""
    observed = list(recent_scores)
    actions = list(recent_actions)
    total = len(actions)
    counts = {a: actions.count(a) for a in ("approve", "review", "escalate")}
    return {
        "observed_requests": total,
        "window": WINDOW,
        "score_psi_vs_validation": psi(observed, REFERENCE_DECILES),
        "psi_alert_threshold": 0.25,
        "action_share": {
            k: round(v / total, 4) if total else None for k, v in counts.items()
        },
        "score_mean": round(float(np.mean(observed)), 5) if observed else None,
        "note": (
            "El PSI del score no detecto la degradacion observada en el "
            "estudio (PR-AUC 0.732 a 0.473 con PSI <= 0.011). La alerta "
            "operativa exige ademas metricas con etiquetas maduras."
        ),
    }
