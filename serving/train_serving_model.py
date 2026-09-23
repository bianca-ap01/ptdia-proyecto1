"""Entrena y serializa los artefactos que sirve la API.

El pipeline de experimentacion nunca guardo un modelo: cada corrida entrenaba,
media y descartaba. Para servir hace falta congelar tres cosas juntas, porque
una sin las otras no produce una decision:

  model.txt            LightGBM entrenado con la ventana historica
  calibrator.joblib    isotonica ajustada en validacion
  serving_config.json  umbrales de la politica, lista de features y metadatos

Los umbrales se toman de policy_thresholds_valid.csv, elegidos en validacion y
congelados, igual que en el holdout del informe. Servir con umbrales
recalculados sobre otro periodo daria decisiones distintas a las evaluadas.
"""

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "ieee-fraud-detection"
FINAL = ROOT / "final"
ARTIFACTS = HERE / "artifacts"

SEED = 42
DAY = 86400
CUT_VALID = 10437998.1
CUT_HOLDOUT = 13151846.0

CAT_COLS = [
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType",
    "DeviceInfo", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]


def load():
    tx = pd.read_csv(DATA / "train_transaction.csv")
    ident = pd.read_csv(DATA / "train_identity.csv")
    ident.columns = [c.replace("-", "_") for c in ident.columns]
    df = tx.merge(ident, on="TransactionID", how="left")
    return df.sort_values("TransactionDT").reset_index(drop=True)


def fit_encoder(train):
    """Estadisticos de la ventana historica, que la API reaplica en linea."""
    q1, q3 = train.TransactionAmt.quantile([0.25, 0.75])
    span = q3 - q1
    maps, freqs = {}, {}
    for col in CAT_COLS:
        if col not in train:
            continue
        series = train[col].astype("string").fillna("__MISSING__")
        maps[col] = {str(v): i for i, v in enumerate(series.unique())}
        freqs[col] = {
            str(k): float(v)
            for k, v in (series.value_counts(dropna=False) / len(series)).items()
        }
    return {
        "amt_low": float(q1 - 1.5 * span),
        "amt_high": float(q3 + 1.5 * span),
        "maps": maps,
        "freqs": freqs,
    }


def transform(df, encoder, feature_names):
    """Misma transformacion que usara la API, para que no diverjan."""
    day = df.TransactionDT // DAY
    device = df["DeviceType"] if "DeviceType" in df else pd.Series(
        np.nan, index=df.index
    )
    derived = {
        "DT_hour": (df.TransactionDT // 3600) % 24,
        "has_identity": device.notna().astype("float32"),
        "TransactionAmt_log1p": np.log1p(df.TransactionAmt),
        "TransactionAmt_outlier_iqr": (
            (df.TransactionAmt < encoder["amt_low"])
            | (df.TransactionAmt > encoder["amt_high"])
        ).astype("float32"),
    }
    for col in ("D1", "D2", "D10", "D15"):
        if col in df:
            derived[f"{col}n"] = df[col] - day

    out = {}
    for name in feature_names:
        if name in derived:
            out[name] = derived[name].astype("float32")
        elif name.endswith("_label") and name[:-6] in encoder["maps"]:
            col = name[:-6]
            out[name] = (
                df[col].astype("string").fillna("__MISSING__")
                .map(encoder["maps"][col]).fillna(-1).astype("float32")
            )
        elif name.endswith("_freq") and name[:-5] in encoder["freqs"]:
            col = name[:-5]
            out[name] = (
                df[col].astype("string").fillna("__MISSING__")
                .map(encoder["freqs"][col]).fillna(0).astype("float32")
            )
        elif name in df:
            out[name] = pd.to_numeric(df[name], errors="coerce").astype("float32")
        else:
            out[name] = pd.Series(np.nan, index=df.index, dtype="float32")
    return pd.DataFrame(out, index=df.index).replace([np.inf, -np.inf], np.nan)


def load_thresholds():
    """Umbrales congelados en validacion para el LightGBM estatico."""
    path = (
        FINAL / "kaggle" / "sistema_final" / "outputs" / "policy_thresholds_valid.csv"
    )
    table = pd.read_csv(path)
    row = table[(table.model == "lightgbm") & (table.strategy == "static")].iloc[0]
    review = next(
        c for c in table.columns if "review" in c.lower() and "thr" in c.lower()
    )
    escalate = next(
        c for c in table.columns if "escal" in c.lower() and "thr" in c.lower()
    )
    return float(row[review]), float(row[escalate])


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    import lightgbm as lgb

    print("Cargando datos...")
    df = load()
    feature_names = json.loads((FINAL / "v01_features.json").read_text())

    train_ix = df.index[df.TransactionDT <= CUT_VALID]
    valid_ix = df.index[
        (df.TransactionDT > CUT_VALID) & (df.TransactionDT <= CUT_HOLDOUT)
    ]
    holdout_ix = df.index[df.TransactionDT > CUT_HOLDOUT]

    encoder = fit_encoder(df.loc[train_ix])
    features = transform(df, encoder, feature_names)

    print("Entrenando LightGBM...")
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=220, learning_rate=0.05, num_leaves=48,
        min_child_samples=120, subsample=0.85, colsample_bytree=0.75,
        reg_alpha=0.05, reg_lambda=0.20, class_weight="balanced",
        random_state=SEED, n_jobs=4, verbosity=-1,
    )
    model.fit(features.loc[train_ix], df.loc[train_ix, "isFraud"])

    valid_scores = model.predict_proba(features.loc[valid_ix])[:, 1]
    holdout_scores = model.predict_proba(features.loc[holdout_ix])[:, 1]

    print("Ajustando calibrador isotonico sobre validacion...")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(valid_scores, df.loc[valid_ix, "isFraud"])

    pr_valid = average_precision_score(df.loc[valid_ix, "isFraud"], valid_scores)
    pr_holdout = average_precision_score(
        df.loc[holdout_ix, "isFraud"], holdout_scores
    )
    review_thr, escalate_thr = load_thresholds()

    model.booster_.save_model(str(ARTIFACTS / "model.txt"))
    joblib.dump(calibrator, ARTIFACTS / "calibrator.joblib")

    config = {
        "model": "lightgbm_v01_static",
        "features": feature_names,
        "encoder": encoder,
        "thresholds": {"review": review_thr, "escalate": escalate_thr},
        "metrics": {
            "pr_auc_valid": round(float(pr_valid), 5),
            "pr_auc_holdout": round(float(pr_holdout), 5),
        },
        "reference_score_deciles": [
            float(q) for q in np.quantile(valid_scores, np.linspace(0, 1, 11))
        ],
        "training_rows": int(len(train_ix)),
        "cutoffs": {"valid": CUT_VALID, "holdout": CUT_HOLDOUT},
        "notes": (
            "Umbrales elegidos en validacion y congelados. Los costos del "
            "informe son supuestos ilustrativos."
        ),
    }
    (ARTIFACTS / "serving_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2)
    )
    shutil.copy2(FINAL / "v01_features.json", ARTIFACTS / "v01_features.json")

    print(f"\nPR-AUC valid={pr_valid:.5f} holdout={pr_holdout:.5f}")
    print(f"Umbrales: revisar>={review_thr:.4f} escalar>={escalate_thr:.4f}")
    print(f"Artefactos en {ARTIFACTS.relative_to(ROOT)}/")
    for item in sorted(ARTIFACTS.iterdir()):
        print(f"  {item.name}  {item.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
