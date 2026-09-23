"""Evalua los modelos tradicionales con el MISMO protocolo temporal que los boosters.

`traditional_baselines()` entrena regresion logistica y arbol de decision una
sola vez sobre el train inicial y reporta PR-AUC agregada en valid y holdout:
cuatro filas, sin dimension temporal. Los boosters, en cambio, tienen 243 filas
por bloque en `block_metrics.csv`.

Esa asimetria importa porque el criterio evaluado es "modelado y evaluacion
TEMPORAL": tal como estaba, los dos enfoques tradicionales nunca se median en
el eje donde el trabajo hace su aporte, y la comparacion no era homologa.

Este script los corre por bloques de 7 dias, con la misma latencia de etiqueta
de 7 dias y las mismas metricas al cupo del 5 %, en dos estrategias: estatica
(un ajuste, como referencia congelada) y periodica con ventana de 45 dias, que
fue la mejor para los boosters.

Salida: final/kaggle/experimentacion/outputs/traditional_block_metrics.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
DATA = HERE / "ieee-fraud-detection"
OUT = HERE / "kaggle" / "experimentacion" / "outputs" / "traditional_block_metrics.csv"

SEED = 42
DAY = 86400
WEEK = 7 * DAY
CUT_VALID = 10437998.1
CUT_HOLDOUT = 13151846.0
WINDOW_DAYS = 45
CAPACITY = 0.05

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


def encode(df, feature_names, fit_ix):
    """Codifica con estadisticos ajustados SOLO sobre `fit_ix`.

    Reproduce WindowEncoder: cuartiles del monto, mapas y frecuencias
    categoricas salen de la ventana historica, nunca del bloque evaluado.
    """
    fit = df.loc[fit_ix]
    q1, q3 = fit.TransactionAmt.quantile([0.25, 0.75])
    span = q3 - q1
    low, high = float(q1 - 1.5 * span), float(q3 + 1.5 * span)

    day = df.TransactionDT // DAY
    derived = {
        "DT_hour": (df.TransactionDT // 3600) % 24,
        "has_identity": df.DeviceType.notna().astype("float32"),
        "TransactionAmt_log1p": np.log1p(df.TransactionAmt),
        "TransactionAmt_outlier_iqr": (
            (df.TransactionAmt < low) | (df.TransactionAmt > high)
        ).astype("float32"),
    }
    for col in ("D1", "D2", "D10", "D15"):
        if col in df:
            derived[f"{col}n"] = df[col] - day

    maps, freqs = {}, {}
    for col in CAT_COLS:
        if col not in df:
            continue
        series = fit[col].astype("string").fillna("__MISSING__")
        maps[col] = {v: i for i, v in enumerate(series.unique())}
        freqs[col] = (series.value_counts(dropna=False) / len(series)).to_dict()

    out = {}
    for name in feature_names:
        if name in derived:
            out[name] = derived[name].astype("float32")
        elif name.endswith("_label") and name[:-6] in maps:
            col = name[:-6]
            out[name] = (
                df[col].astype("string").fillna("__MISSING__")
                .map(maps[col]).fillna(-1).astype("float32")
            )
        elif name.endswith("_freq") and name[:-5] in freqs:
            col = name[:-5]
            out[name] = (
                df[col].astype("string").fillna("__MISSING__")
                .map(freqs[col]).fillna(0).astype("float32")
            )
        elif name in df:
            out[name] = pd.to_numeric(df[name], errors="coerce").astype("float32")
        else:
            out[name] = pd.Series(np.nan, index=df.index, dtype="float32")

    return pd.DataFrame(out, index=df.index).replace([np.inf, -np.inf], np.nan)


def make_estimator(name):
    """Mismos hiperparametros que traditional_baselines(), para comparabilidad."""
    if name == "regresion_logistica":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(
                max_iter=120, class_weight="balanced", solver="saga",
                random_state=SEED, n_jobs=4,
            ),
        )
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=120, class_weight="balanced",
            random_state=SEED,
        ),
    )


def blocks_for(df, lo, hi):
    """Bloques consecutivos de 7 dias dentro de (lo, hi]."""
    out, start = [], lo
    while start < hi:
        end = min(start + WEEK, hi)
        ix = df.index[(df.TransactionDT > start) & (df.TransactionDT <= end)]
        if len(ix):
            out.append((start, end, ix))
        start = end
    return out


def metrics_at_capacity(y_true, scores):
    """PR-AUC y recall/precision al cupo del 5 %, como en block_metrics.csv."""
    total = int(y_true.sum())
    if not total:
        return np.nan, np.nan, np.nan
    pr_auc = average_precision_score(y_true, scores)
    k = max(1, int(round(CAPACITY * len(scores))))
    top = np.argsort(scores)[::-1][:k]
    caught = int(y_true.to_numpy()[top].sum())
    return pr_auc, caught / total, caught / k


def main():
    print("Cargando datos...")
    df = load()
    feature_names = json.loads((HERE / "v01_features.json").read_text())
    initial_train = df.index[df.TransactionDT <= CUT_VALID]

    rows = []
    for model_name in ("regresion_logistica", "arbol_decision"):
        print(f"\n{model_name}")
        static_features = encode(df, feature_names, initial_train)
        static_model = make_estimator(model_name)
        static_model.fit(
            static_features.loc[initial_train],
            df.loc[initial_train, "isFraud"].to_numpy(dtype=int),
        )

        for split, lo, hi in (
            ("valid", CUT_VALID, CUT_HOLDOUT),
            ("holdout", CUT_HOLDOUT, float(df.TransactionDT.max())),
        ):
            for number, (start, _end, ix) in enumerate(blocks_for(df, lo, hi)):
                y = df.loc[ix, "isFraud"]

                pred = static_model.predict_proba(static_features.loc[ix])[:, 1]
                pr_auc, recall, precision = metrics_at_capacity(y, pred)
                rows.append({
                    "model": model_name, "strategy": "static",
                    "window_days": "initial_70pct", "split": split,
                    "block": number, "rows": len(ix),
                    "fraud_rate": round(float(y.mean()), 5),
                    "pr_auc": round(float(pr_auc), 5),
                    "recall_at_5pct": round(float(recall), 5),
                    "precision_at_5pct": round(float(precision), 5),
                })

                cutoff = start - WEEK
                floor = cutoff - WINDOW_DAYS * DAY
                fit_ix = df.index[
                    (df.TransactionDT >= floor) & (df.TransactionDT < cutoff)
                ]
                if len(fit_ix) < 1000 or df.loc[fit_ix, "isFraud"].sum() < 20:
                    continue
                feats = encode(df, feature_names, fit_ix)
                model = make_estimator(model_name)
                model.fit(
                    feats.loc[fit_ix],
                    df.loc[fit_ix, "isFraud"].to_numpy(dtype=int),
                )
                pred = model.predict_proba(feats.loc[ix])[:, 1]
                pr_auc, recall, precision = metrics_at_capacity(y, pred)
                rows.append({
                    "model": model_name, "strategy": "periodic",
                    "window_days": str(WINDOW_DAYS), "split": split,
                    "block": number, "rows": len(ix),
                    "fraud_rate": round(float(y.mean()), 5),
                    "pr_auc": round(float(pr_auc), 5),
                    "recall_at_5pct": round(float(recall), 5),
                    "precision_at_5pct": round(float(precision), 5),
                })
            print(f"  {split}: listo")

    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)

    holdout = result[result.split == "holdout"]
    print("\n=== PR-AUC media en holdout, por bloque ===")
    print(
        holdout.groupby(["model", "strategy"])
        .pr_auc.agg(["mean", "std", "count"])
        .round(4).to_string()
    )
    print(f"\nEscrito: {OUT.relative_to(HERE)} ({len(result)} filas)")


if __name__ == "__main__":
    main()
