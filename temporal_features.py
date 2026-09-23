"""Prueba features con memoria por entidad: lags y agregaciones por periodo.

El enunciado pide explicitamente generar atributos que incorporen la dimension
temporal —ventanas deslizantes, variables rezagadas, agregaciones por periodo—.
El pipeline actual no tiene ninguno: `row_features` solo usa informacion de la
propia fila ("Only own-row information"), y la temporalidad vive unicamente en
el protocolo de evaluacion y reentrenamiento.

Este script agrega cinco variables con memoria, todas calculadas hacia atras
sobre el UID candidato, y mide si aportan sobre la misma superficie de 185:

  uid_txn_count_prev      cuantas operaciones previas tiene esa entidad
  uid_amt_mean_prev       monto medio historico de la entidad (expanding)
  uid_amt_delta           desvio del monto actual frente a esa media
  uid_seconds_since_prev  tiempo desde la operacion anterior de la entidad
  uid_amt_lag1            monto de la operacion anterior de la entidad

Todas usan shift(1) dentro del UID ordenado por tiempo, de modo que una fila
nunca ve su propio valor ni el futuro. Es la unica forma de que un lag sea
honesto bajo el protocolo temporal del trabajo.

Salida: final/kaggle/experimentacion/outputs/temporal_features_impact.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
DATA = HERE / "ieee-fraud-detection"
OUT = (
    HERE / "kaggle" / "experimentacion" / "outputs" / "temporal_features_impact.csv"
)

SEED = 42
CUT_VALID = 10437998.1
CUT_HOLDOUT = 13151846.0
DAY = 86400

CAT_COLS = [
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType",
    "DeviceInfo", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]

TEMPORAL = [
    "uid_txn_count_prev",
    "uid_amt_mean_prev",
    "uid_amt_delta",
    "uid_seconds_since_prev",
    "uid_amt_lag1",
]


def load():
    tx = pd.read_csv(DATA / "train_transaction.csv")
    ident = pd.read_csv(DATA / "train_identity.csv")
    ident.columns = [c.replace("-", "_") for c in ident.columns]
    df = tx.merge(ident, on="TransactionID", how="left")
    return df.sort_values("TransactionDT").reset_index(drop=True)


def candidate_uid(df):
    """Identidad aproximada, con atributos de la propia operacion."""
    parts = [
        df[col].astype("string").fillna("__NA__")
        for col in ("card1", "addr1", "P_emaildomain")
    ]
    return parts[0].str.cat(parts[1:], sep="|")


def temporal_features(df, uid):
    """Lags y agregaciones expanding por entidad, siempre hacia atras.

    `shift(1)` dentro de cada UID garantiza que la fila n solo vea las
    operaciones 1..n-1 de esa entidad. No hace falta cortar por ventana: la
    causalidad ya la impone el orden temporal del DataFrame.
    """
    work = pd.DataFrame(
        {"uid": uid, "dt": df.TransactionDT, "amt": df.TransactionAmt},
        index=df.index,
    )
    grouped = work.groupby("uid", sort=False)

    out = pd.DataFrame(index=df.index)
    out["uid_txn_count_prev"] = grouped.cumcount().astype("float32")
    out["uid_amt_mean_prev"] = (
        grouped.amt.transform(lambda s: s.shift(1).expanding().mean())
        .astype("float32")
    )
    out["uid_amt_lag1"] = grouped.amt.shift(1).astype("float32")
    out["uid_seconds_since_prev"] = (work.dt - grouped.dt.shift(1)).astype("float32")
    out["uid_amt_delta"] = (work.amt - out.uid_amt_mean_prev).astype("float32")
    return out


def encode_base(df, feature_names, fit_ix):
    """Las 185 variables originales, con estadisticos de la ventana historica."""
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


def train_and_score(features, y, train_ix, valid_ix, holdout_ix):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=220, learning_rate=0.05, num_leaves=48,
        min_child_samples=120, subsample=0.85, colsample_bytree=0.75,
        reg_alpha=0.05, reg_lambda=0.20, class_weight="balanced",
        random_state=SEED, n_jobs=4, verbosity=-1,
    )
    model.fit(features.loc[train_ix], y.loc[train_ix])
    scores = {
        label: average_precision_score(
            y.loc[ix], model.predict_proba(features.loc[ix])[:, 1]
        )
        for label, ix in (("valid", valid_ix), ("holdout", holdout_ix))
    }
    importance = pd.Series(
        model.feature_importances_, index=features.columns
    ).sort_values(ascending=False)
    return scores, importance


def main():
    print("Cargando datos...")
    df = load()
    y = df.isFraud
    feature_names = json.loads((HERE / "v01_features.json").read_text())

    train_ix = df.index[df.TransactionDT <= CUT_VALID]
    valid_ix = df.index[
        (df.TransactionDT > CUT_VALID) & (df.TransactionDT <= CUT_HOLDOUT)
    ]
    holdout_ix = df.index[df.TransactionDT > CUT_HOLDOUT]

    base = encode_base(df, feature_names, train_ix)

    print("Construyendo features temporales por UID...")
    extra = temporal_features(df, candidate_uid(df))
    enriched = pd.concat([base, extra], axis=1)

    rows = []
    for label, features in (("base_185", base), ("base_mas_temporales", enriched)):
        print(f"\nEntrenando '{label}' ({features.shape[1]} variables)...")
        scores, importance = train_and_score(
            features, y, train_ix, valid_ix, holdout_ix
        )
        print(f"  PR-AUC valid={scores['valid']:.5f} holdout={scores['holdout']:.5f}")
        if label != "base_185":
            print("  posicion de las nuevas variables por importancia:")
            ranks = {
                name: int(importance.index.get_loc(name)) + 1 for name in TEMPORAL
            }
            for name, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
                print(f"    {rank:3d}/{features.shape[1]}  {name}")
        rows.append({
            "variant": label,
            "n_features": features.shape[1],
            "pr_auc_valid": round(float(scores["valid"]), 5),
            "pr_auc_holdout": round(float(scores["holdout"]), 5),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT, index=False)
    delta = summary.pr_auc_holdout.iloc[1] - summary.pr_auc_holdout.iloc[0]
    print("\n" + summary.to_string(index=False))
    print(f"\nAporte de las features temporales en holdout: {delta:+.5f} PR-AUC")
    print(f"Escrito: {OUT.relative_to(HERE)}")


if __name__ == "__main__":
    main()
