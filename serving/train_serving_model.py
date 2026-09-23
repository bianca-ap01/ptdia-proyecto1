"""Entrena y serializa los artefactos que sirve la API.

Toma la decision congelada en
`08_entrega_final_codigo/outputs/costos/frozen_choice.json` —XGBoost periodico
de 30 dias, con sus umbrales y su cupo— y reutiliza el codigo del paquete de
entrega. Reimplementar el preprocesamiento aqui arriesgaria servir un modelo
parecido pero no identico al que documenta el informe.

El pipeline de entrega guarda predicciones y metricas, no un modelo. Para
servir hay que reentrenar con la configuracion congelada y guardar juntas las
tres piezas que solo significan algo en conjunto:

  model.json           el clasificador entrenado con la ventana historica
  calibrator.joblib    isotonica ajustada solo con validacion
  serving_config.json  umbrales, cupo, variables y estadisticos del encoder

Los umbrales provienen de la seleccion hecha en validacion y se sirven
congelados: recalcularlos con trafico nuevo daria decisiones distintas a las
que el informe evaluo.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACKAGE = ROOT / "08_entrega_final_codigo"
ARTIFACTS = HERE / "artifacts"

sys.path.insert(0, str(PACKAGE))
from core import cuts, fit_snapshot, load_train, score_snapshot  # noqa: E402

DAY = 86400
WEEK = 7 * DAY


def frozen_choice():
    """La decision que el informe declara congelada en validacion."""
    return json.loads(
        (PACKAGE / "outputs" / "costos" / "frozen_choice.json").read_text()
    )


def selected_params(model_name):
    """Hiperparametros elegidos por validacion interna, no por holdout."""
    selection = json.loads(
        (PACKAGE / "outputs" / "experimentacion" / "selected_hyperparameters.json")
        .read_text()
    )
    return selection["selected"][model_name]["params"]


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    choice = frozen_choice()
    model_name = choice["model"]
    window_days = int(choice["window_days"])
    params = selected_params(model_name)

    print(f"Decision congelada: {model_name} periodico {window_days} d")
    print(f"  umbrales: revisar {choice['review_threshold']:.6f} "
          f"escalar {choice['escalate_threshold']:.6f}")
    print(f"  cupo diario: {choice['review_capacity_per_day']}")
    print(f"  hiperparametros: {params}")

    print("\nCargando datos...")
    # load_train busca los CSV en `root.parent`, asi que el root que espera es
    # el paquete de entrega, no la raiz del repositorio.
    df = load_train(PACKAGE)
    train_cut, valid_cut = cuts(df)

    # El corte de seleccion es el que `frozen_choice` declara: hasta ahi se
    # eligio la politica, y despues quedan semanas de validacion que el modelo
    # no vio. Entrenar con la ventana previa a ese corte y calibrar con lo que
    # sigue reproduce la separacion del pipeline; usar la ventana pegada al fin
    # de validacion no dejaria ninguna fila limpia para calibrar.
    cutoff = float(choice["selection_period_end"])
    floor = cutoff - window_days * DAY
    train_mask = (
        (df.TransactionDT >= floor) & (df.TransactionDT < cutoff)
    ).to_numpy()
    print(f"Entrenando con {int(train_mask.sum())} filas de la ventana de "
          f"{window_days} dias previa al corte...")

    snapshot = fit_snapshot(df, train_mask, model_name, params, cutoff)
    encoder = snapshot["encoder"]

    # La calibracion y los deciles de referencia se ajustan con las semanas de
    # validacion ANTERIORES a la ventana de entrenamiento. Usar toda la
    # validacion incluiria las semanas con las que se acaba de entrenar y
    # produciria un PR-AUC inflado que no describe el comportamiento en
    # produccion. La metrica de referencia del modelo es la del informe, que
    # se obtiene por bloques con reajuste; aqui solo se mide sobre datos no
    # vistos para verificar que el ajuste es sano.
    holdout_of_valid = df.loc[
        (df.TransactionDT >= cutoff) & (df.TransactionDT <= valid_cut)
    ]
    valid_scores = score_snapshot(snapshot, holdout_of_valid)
    pr_valid = average_precision_score(holdout_of_valid.isFraud, valid_scores)

    print(f"Calibrando con {len(holdout_of_valid)} filas de validacion "
          f"posteriores a la ventana de entrenamiento...")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(valid_scores, holdout_of_valid.isFraud.to_numpy())

    snapshot["model"].save_model(str(ARTIFACTS / "model.json"))
    joblib.dump(calibrator, ARTIFACTS / "calibrator.joblib")

    config = {
        "model": f"{model_name}_periodic_{window_days}d",
        "frozen_choice_digest": choice["digest"],
        "features": list(snapshot["features"]),
        "encoder": {
            "amt_low": float(encoder.low),
            "amt_high": float(encoder.high),
            "maps": {k: {str(a): int(b) for a, b in v.items()}
                     for k, v in encoder.maps.items()},
            "freqs": {k: {str(a): float(b) for a, b in v.items()}
                      for k, v in encoder.freqs.items()},
        },
        "thresholds": {
            "review": float(choice["review_threshold"]),
            "escalate": float(choice["escalate_threshold"]),
        },
        "review_capacity_per_day": int(choice["review_capacity_per_day"]),
        "metrics": {"pr_auc_valid_out_of_window": round(float(pr_valid), 5),
                    "pr_auc_validation_report": 0.52874,
                    "pr_auc_holdout_report": 0.548},
        "reference_score_deciles": [
            float(q) for q in np.quantile(valid_scores, np.linspace(0, 1, 11))
        ],
        "training_rows": int(snapshot["train_rows"]),
        "training_window_days": window_days,
        "cutoffs": {"train": float(train_cut), "valid": float(valid_cut)},
        "cost_provenance": choice.get("cost_provenance", {}),
        "notes": (
            "Umbrales y cupo provienen de la seleccion en validacion y se "
            "sirven congelados. Los costos del informe son ilustrativos."
        ),
    }
    (ARTIFACTS / "serving_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2)
    )

    print(f"\nPR-AUC fuera de ventana: {pr_valid:.5f}  (informe: 0.529 valid / 0.548 holdout)")
    print(f"Variables: {len(config['features'])}")
    print(f"Artefactos en {ARTIFACTS.relative_to(ROOT)}/")
    for item in sorted(ARTIFACTS.iterdir()):
        print(f"  {item.name}  {item.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
