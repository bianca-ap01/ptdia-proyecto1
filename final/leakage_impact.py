"""Cuantifica la fuga de V01: cuartiles del monto ajustados sobre todas las filas.

V01 calculo los cortes IQR de `TransactionAmt` sobre las 590 540 filas, no solo
sobre su ventana de entrenamiento. Esos cortes definen
`TransactionAmt_outlier_iqr`, una de las 185 variables. La version corregida
(`WindowEncoder.fit`) los ajusta solo con historico.

Eso vuelve asimetrica la comparacion central del trabajo: V01 se evalua con una
ventaja informativa que las estrategias adaptativas no tienen. El reporte
declara la fuga pero no mide su tamano, asi que la conclusion "adaptar no
confirmo mejora" queda sin respaldo cuantitativo.

Este script entrena el MISMO modelo dos veces, identico en semilla,
hiperparametros y features, cambiando solo el origen de los cuartiles, y
reporta la diferencia de PR-AUC.

Salida: final/kaggle/experimentacion/outputs/leakage_impact.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "ieee-fraud-detection"
OUT = HERE / "kaggle" / "experimentacion" / "outputs" / "leakage_impact.csv"

SEED = 42
CUT_VALID = 10437998.1
CUT_HOLDOUT = 13151846.0
CAT_COLS = [
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType",
    "DeviceInfo", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]


def load():
    """Transacciones e identidad unidas, ordenadas temporalmente."""
    tx = pd.read_csv(DATA / "train_transaction.csv")
    ident = pd.read_csv(DATA / "train_identity.csv")
    ident.columns = [c.replace("-", "_") for c in ident.columns]
    df = tx.merge(ident, on="TransactionID", how="left")
    return df.sort_values("TransactionDT").reset_index(drop=True)


def iqr_bounds(amounts):
    """Cortes IQR de Tukey."""
    q1, q3 = amounts.quantile([0.25, 0.75])
    span = q3 - q1
    return float(q1 - 1.5 * span), float(q3 + 1.5 * span)


def build_features(df, feature_names, bounds, train_mask):
    """Matriz de 185 variables.

    `bounds` es lo unico que cambia entre las dos variantes. Los mapas
    categoricos se ajustan siempre solo con train, para que la comparacion
    aisle el efecto de los cuartiles.
    """
    low, high = bounds
    train = df.loc[train_mask]
    out = {}

    day = df.TransactionDT // 86400
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
        series = train[col].astype("string").fillna("__MISSING__")
        maps[col] = {v: i for i, v in enumerate(series.unique())}
        freqs[col] = (series.value_counts(dropna=False) / len(series)).to_dict()

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


def train_and_score(features, y, train_mask, valid_mask, holdout_mask):
    """LightGBM con los hiperparametros de V01."""
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=220, learning_rate=0.05, num_leaves=48,
        min_child_samples=120, subsample=0.85, colsample_bytree=0.75,
        reg_alpha=0.05, reg_lambda=0.20, class_weight="balanced",
        random_state=SEED, n_jobs=4, verbosity=-1,
    )
    model.fit(features.loc[train_mask], y.loc[train_mask])
    return (
        average_precision_score(
            y.loc[valid_mask], model.predict_proba(features.loc[valid_mask])[:, 1]
        ),
        average_precision_score(
            y.loc[holdout_mask], model.predict_proba(features.loc[holdout_mask])[:, 1]
        ),
    )


def main():
    print("Cargando datos...")
    df = load()
    y = df.isFraud
    feature_names = json.loads((HERE / "v01_features.json").read_text())

    train_mask = df.TransactionDT <= CUT_VALID
    valid_mask = (df.TransactionDT > CUT_VALID) & (df.TransactionDT <= CUT_HOLDOUT)
    holdout_mask = df.TransactionDT > CUT_HOLDOUT

    leaked = iqr_bounds(df.TransactionAmt)                    # como lo hizo V01
    clean = iqr_bounds(df.loc[train_mask, "TransactionAmt"])  # solo historico

    flag_leaked = (df.TransactionAmt < leaked[0]) | (df.TransactionAmt > leaked[1])
    flag_clean = (df.TransactionAmt < clean[0]) | (df.TransactionAmt > clean[1])
    changed = int((flag_leaked != flag_clean).loc[holdout_mask].sum())

    print(f"Cuartiles con fuga : low={leaked[0]:.4f} high={leaked[1]:.4f}")
    print(f"Cuartiles limpios  : low={clean[0]:.4f} high={clean[1]:.4f}")
    print(
        f"Filas de holdout que cambian de valor: {changed} "
        f"de {int(holdout_mask.sum())}"
    )

    rows = []
    for label, bounds in (("leaked_v01", leaked), ("clean_temporal", clean)):
        print(f"\nEntrenando variante '{label}'...")
        features = build_features(df, feature_names, bounds, train_mask)
        pr_valid, pr_holdout = train_and_score(
            features, y, train_mask, valid_mask, holdout_mask
        )
        print(f"  PR-AUC valid={pr_valid:.5f} holdout={pr_holdout:.5f}")
        rows.append(
            {
                "variant": label,
                "pr_auc_valid": round(pr_valid, 5),
                "pr_auc_holdout": round(pr_holdout, 5),
                "quantile_low": round(bounds[0], 4),
                "quantile_high": round(bounds[1], 4),
                "rows_flag_changed_holdout": changed,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT, index=False)

    delta = summary.pr_auc_holdout.iloc[0] - summary.pr_auc_holdout.iloc[1]
    print("\n" + summary.to_string(index=False))
    print(f"\nVentaja de la fuga en holdout: {delta:+.5f} PR-AUC")
    print(f"Escrito: {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
