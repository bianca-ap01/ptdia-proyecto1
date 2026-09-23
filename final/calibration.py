"""Calibracion de probabilidades y banda de abstencion.

El score del modelo ordena bien pero no es una probabilidad: su media esta
muy por encima de la prevalencia real. Mientras la politica solo compare el
score contra un umbral, eso da igual. Deja de dar igual en cuanto alguien
lea el score como "probabilidad de fraude" para explicar una decision al
cliente, o para sumar riesgo esperado en pesos.

Ajusta isotonica sobre validacion y la aplica a holdout, sin tocar el
modelo. Reporta Brier antes y despues, la curva de confiabilidad por decil
y la cobertura de la banda de revision leida como region de abstencion.

Salidas en final/kaggle/sistema_final/outputs/:
  calibration_summary.csv  Brier y score medio vs prevalencia
  calibration_bins.csv     curva de confiabilidad, 10 bins
  abstention_summary.csv   riesgo dentro y fuera de la banda de revision
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "kaggle" / "sistema_final" / "outputs"
DECISIONS = OUTDIR / "v01_policy_decisions.csv.gz"
VALID_PREDS = (
    HERE / "kaggle" / "experimentacion" / "inputs" / "v01_reference"
    / "baseline_valid_predictions.csv"
)

N_BINS = 10


def load_decisions():
    """Decisiones de V01 en holdout, con score, etiqueta y accion tomada."""
    cols = ["isFraud", "score", "split", "action", "uid_known"]
    return pd.read_csv(DECISIONS, usecols=cols)


def load_valid():
    """Scores de V01 sobre validacion, que es donde se ajusta la isotonica."""
    valid = pd.read_csv(VALID_PREDS, usecols=["isFraud", "pred"])
    return valid.rename(columns={"pred": "score"})


def fit_isotonic(valid):
    """Isotonica ajustada SOLO con validacion.

    Ajustarla con holdout filtraria informacion del periodo reservado hacia
    la calibracion, que es justo el error que el protocolo temporal evita.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(valid.score.to_numpy(), valid.isFraud.to_numpy())
    return iso


def reliability_bins(y_true, y_score, n_bins=N_BINS):
    """Curva de confiabilidad por cuantiles del score."""
    edges = np.quantile(y_score, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(y_score, edges[1:-1])
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "rows": int(mask.sum()),
                "mean_score": round(float(y_score[mask].mean()), 5),
                "observed_rate": round(float(y_true[mask].mean()), 5),
                "gap": round(float(y_score[mask].mean() - y_true[mask].mean()), 5),
            }
        )
    return pd.DataFrame(rows)


def abstention_band(holdout):
    """La cola de revision funciona como region de abstencion.

    El sistema no decide solo: deriva esa banda a un analista. Cuantificar
    su cobertura y el riesgo dentro y fuera es lo que permite discutir si la
    banda esta bien puesta.
    """
    rows = []
    for action, group in holdout.groupby("action"):
        rows.append(
            {
                "action": action,
                "rows": len(group),
                "share": round(len(group) / len(holdout), 4),
                "fraud_rate": round(float(group.isFraud.mean()), 5),
                "frauds": int(group.isFraud.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("fraud_rate")


def main():
    decisions = load_decisions()
    holdout = decisions[decisions.split == "holdout"].copy()

    iso = fit_isotonic(load_valid())
    holdout["score_calibrated"] = iso.predict(holdout.score.to_numpy())

    y = holdout.isFraud.to_numpy()
    raw = holdout.score.to_numpy()
    cal = holdout.score_calibrated.to_numpy()
    prevalence = float(y.mean())

    summary = pd.DataFrame(
        [
            {
                "variant": "raw",
                "brier": round(brier_score_loss(y, raw), 5),
                "mean_score": round(float(raw.mean()), 5),
                "observed_prevalence": round(prevalence, 5),
                "overestimation_factor": round(float(raw.mean()) / prevalence, 2),
            },
            {
                "variant": "isotonic",
                "brier": round(brier_score_loss(y, cal), 5),
                "mean_score": round(float(cal.mean()), 5),
                "observed_prevalence": round(prevalence, 5),
                "overestimation_factor": round(float(cal.mean()) / prevalence, 2),
            },
        ]
    )
    summary.to_csv(OUTDIR / "calibration_summary.csv", index=False)

    bins_raw = reliability_bins(y, raw)
    bins_raw["variant"] = "raw"
    bins_cal = reliability_bins(y, cal)
    bins_cal["variant"] = "isotonic"
    pd.concat([bins_raw, bins_cal]).to_csv(
        OUTDIR / "calibration_bins.csv", index=False
    )

    bands = abstention_band(holdout)
    bands.to_csv(OUTDIR / "abstention_summary.csv", index=False)

    print("=== Calibracion (holdout, isotonica ajustada en validacion) ===")
    print(summary.to_string(index=False))
    reduction = 1 - summary.brier.iloc[1] / summary.brier.iloc[0]
    print(f"\nReduccion de Brier: {reduction:.1%}")

    print("\n=== Banda de abstencion: riesgo por accion ===")
    print(bands.to_string(index=False))

    approve = bands[bands.action == "approve"]
    if not approve.empty:
        print(
            f"\nDecidido automaticamente: {float(approve['share'].iloc[0]):.1%} "
            f"de las transacciones, con prevalencia residual "
            f"{float(approve['fraud_rate'].iloc[0]):.4f}."
        )
    print(f"\nEscrito en {OUTDIR.relative_to(HERE.parent)}/")


if __name__ == "__main__":
    main()
