import gc
import json
import os
import random
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except Exception as exc:
    raise RuntimeError("LightGBM is required for the V07 reference model") from exc

try:
    import tensorflow as tf
except Exception as exc:
    raise RuntimeError("TensorFlow is required for the V07 MLP benchmark") from exc


DATA_DIR_CANDIDATES = [
    Path("/kaggle/input/ieee-fraud-detection"),
    Path("/kaggle/input/competitions/ieee-fraud-detection"),
]
DATA_DIR = next((path for path in DATA_DIR_CANDIDATES if path.exists()), DATA_DIR_CANDIDATES[0])
OUT = Path("/kaggle/working")
PLOTS = OUT / "plots"
PLOTS.mkdir(exist_ok=True)
RANDOM_STATE = 42


def seed_everything(seed=RANDOM_STATE):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def reduce_memory(df):
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif df[col].dtype == "object":
            df[col] = df[col].astype("category")
    return df


def load_train():
    txn = reduce_memory(pd.read_csv(DATA_DIR / "train_transaction.csv"))
    identity = reduce_memory(pd.read_csv(DATA_DIR / "train_identity.csv"))
    df = txn.merge(identity, on="TransactionID", how="left")
    del txn, identity
    gc.collect()
    return df.sort_values("TransactionDT").reset_index(drop=True)


def temporal_split(df, train_q=0.70, valid_q=0.85):
    q_train = df["TransactionDT"].quantile(train_q)
    q_valid = df["TransactionDT"].quantile(valid_q)
    train_idx = df.index[df["TransactionDT"] <= q_train]
    valid_idx = df.index[(df["TransactionDT"] > q_train) & (df["TransactionDT"] <= q_valid)]
    holdout_idx = df.index[df["TransactionDT"] > q_valid]
    return train_idx, valid_idx, holdout_idx, {"train_cutoff": float(q_train), "valid_cutoff": float(q_valid)}


def add_base_features(df, train_stats=None):
    df = df.copy()
    day = (df["TransactionDT"] // (24 * 3600)).astype("int16")
    hour = ((df["TransactionDT"] // 3600) % 24).astype("int8")

    df["DT_hour"] = hour
    df["DT_day_index"] = day
    df["DT_week_index"] = (day // 7).astype("int16")

    identity_cols = [c for c in df.columns if c.startswith("id_") or c in ["DeviceType", "DeviceInfo"]]
    df["has_identity"] = 0
    if identity_cols:
        df["has_identity"] = df[identity_cols[0]].notna().astype("int8")

    df["TransactionAmt_log1p"] = np.log1p(df["TransactionAmt"].astype("float64")).astype("float32")
    df["TransactionAmt_multi_decimal"] = (
        ((df["TransactionAmt"].astype("float64") * 100).round(4) % 1).abs() > 1e-6
    ).astype("int8")

    if train_stats is None:
        q1, q3 = df["TransactionAmt"].quantile([0.25, 0.75])
        iqr = q3 - q1
        train_stats = {
            "amt_iqr_low": float(q1 - 1.5 * iqr),
            "amt_iqr_high": float(q3 + 1.5 * iqr),
        }

    df["TransactionAmt_outlier_iqr"] = (
        (df["TransactionAmt"] < train_stats["amt_iqr_low"])
        | (df["TransactionAmt"] > train_stats["amt_iqr_high"])
    ).astype("int8")

    v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    d_cols = [c for c in df.columns if c.startswith("D") and c[1:].isdigit()]
    id_cols = [c for c in df.columns if c.startswith("id_")]

    df["missing_V_count"] = df[v_cols].isna().sum(axis=1).astype("int16") if v_cols else 0
    df["missing_D_count"] = df[d_cols].isna().sum(axis=1).astype("int8") if d_cols else 0
    df["missing_id_count"] = df[id_cols].isna().sum(axis=1).astype("int8") if id_cols else 0
    df["missing_core_count"] = df[["card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2"]].isna().sum(axis=1).astype("int8")

    for col in ["D1", "D2", "D10", "D15"]:
        if col in df.columns:
            df[f"{col}n"] = (df[col].astype("float32") - day.astype("float32")).astype("float32")

    return df, train_stats


def candidate_uid(df):
    day = (df["TransactionDT"] // (24 * 3600)).astype("float32")
    parts = pd.DataFrame(index=df.index)
    for col in ["card1", "card2", "card3", "card5", "addr1", "addr2"]:
        parts[col] = df[col].astype("string").fillna("NA")
    parts["D1_anchor"] = (df["D1"].astype("float32") - day).round(0).astype("string").fillna("NA")
    return pd.util.hash_pandas_object(parts, index=False).astype("uint64")


def build_encoding_maps(train_df, cat_cols):
    maps, freqs = {}, {}
    n = len(train_df)
    for col in cat_cols:
        s = train_df[col].astype("string").fillna("__MISSING__")
        uniques = pd.Series(s.unique())
        maps[col] = {v: i for i, v in enumerate(uniques)}
        freqs[col] = (s.value_counts(dropna=False) / n).to_dict()
    return maps, freqs


def apply_encodings(df, cat_cols, maps, freqs):
    df = df.copy()
    for col in cat_cols:
        s = df[col].astype("string").fillna("__MISSING__")
        df[f"{col}_label"] = s.map(maps[col]).fillna(-1).astype("int32")
        df[f"{col}_freq"] = s.map(freqs[col]).fillna(0).astype("float32")
    return df


def prepare_matrix(df, train_idx):
    drop_cols = {"TransactionID", "isFraud", "TransactionDT", "DT_day_index", "DT_week_index"}
    object_cols = [c for c in df.columns if str(df[c].dtype) in ["object", "category", "string"]]
    feature_cols = [c for c in df.columns if c not in drop_cols and c not in object_cols]

    X = df[feature_cols].copy()
    for col in feature_cols:
        if X[col].dtype == "bool":
            X[col] = X[col].astype("int8")
        elif not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    all_missing_train = X.loc[train_idx].isna().all()
    feature_cols = [c for c in feature_cols if not all_missing_train.get(c, False)]

    v_cols = [c for c in feature_cols if c.startswith("V") and c[1:].isdigit()]
    if len(v_cols) > 80:
        y_train = df.loc[train_idx, "isFraud"].astype("float32")
        v_corr = {}
        for col in v_cols:
            s = X.loc[train_idx, col]
            if s.notna().sum() > 100 and s.nunique(dropna=True) > 1:
                v_corr[col] = abs(s.astype("float32").corr(y_train))
        keep_v = set(pd.Series(v_corr).sort_values(ascending=False).head(80).index)
        feature_cols = [c for c in feature_cols if not (c.startswith("V") and c[1:].isdigit()) or c in keep_v]

    return X[feature_cols], feature_cols


def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype("int8")
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    return {
        "threshold": float(threshold),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fpr,
        "flagged_rate": float(y_pred.mean()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_segment(model_name, split, segment, y_true, y_prob):
    if len(y_true) == 0 or y_true.nunique() < 2:
        return {
            "model": model_name,
            "split": split,
            "segment": segment,
            "rows": int(len(y_true)),
            "fraud_rate": float(y_true.mean()) if len(y_true) else np.nan,
            "pr_auc": np.nan,
            "roc_auc": np.nan,
        }
    base = {
        "model": model_name,
        "split": split,
        "segment": segment,
        "rows": int(len(y_true)),
        "fraud_rate": float(y_true.mean()),
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
    out = {}
    for prefix, vals in [
        ("thr05", metrics_at_threshold(y_true, y_prob, 0.5)),
        ("top5", metrics_at_threshold(y_true, y_prob, np.quantile(y_prob, 0.95))),
    ]:
        for k, v in vals.items():
            out[f"{prefix}_{k}"] = v
    return {**base, **out}


def evaluate_all(model_name, y, valid_idx, holdout_idx, valid_prob, holdout_prob, known_valid, known_holdout):
    return pd.DataFrame([
        evaluate_segment(model_name, "valid", "global", y.loc[valid_idx], valid_prob),
        evaluate_segment(model_name, "valid", "uid_known", y.loc[valid_idx][known_valid.values], valid_prob[known_valid.values]),
        evaluate_segment(model_name, "valid", "uid_unknown", y.loc[valid_idx][~known_valid.values], valid_prob[~known_valid.values]),
        evaluate_segment(model_name, "holdout", "global", y.loc[holdout_idx], holdout_prob),
        evaluate_segment(model_name, "holdout", "uid_known", y.loc[holdout_idx][known_holdout.values], holdout_prob[known_holdout.values]),
        evaluate_segment(model_name, "holdout", "uid_unknown", y.loc[holdout_idx][~known_holdout.values], holdout_prob[~known_holdout.values]),
    ])


def fit_lightgbm(X_train, y_train, X_valid, y_valid, X_holdout):
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=550,
        learning_rate=0.05,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=120,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=0.20,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(50)],
    )
    return model.predict_proba(X_valid)[:, 1], model.predict_proba(X_holdout)[:, 1], {
        "best_iteration": int(model.best_iteration_ or model.n_estimators)
    }


def fit_logistic_sgd(X_train, y_train, X_valid, X_holdout):
    clf = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=1e-5,
        l1_ratio=0.05,
        class_weight="balanced",
        max_iter=35,
        tol=1e-3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_valid)[:, 1], clf.predict_proba(X_holdout)[:, 1], {
        "n_iter": int(clf.n_iter_)
    }


def fit_mlp_keras(X_train, y_train, X_valid, y_valid, X_holdout, scale_pos_weight):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.18),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.10),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
        ],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=18,
        batch_size=4096,
        class_weight={0: 1.0, 1: float(scale_pos_weight)},
        callbacks=callbacks,
        verbose=2,
    )
    valid_prob = model.predict(X_valid, batch_size=8192, verbose=0).ravel()
    holdout_prob = model.predict(X_holdout, batch_size=8192, verbose=0).ravel()
    best_epoch = int(np.argmax(history.history["val_pr_auc"]) + 1)
    return valid_prob, holdout_prob, {
        "best_epoch": best_epoch,
        "epochs_run": len(history.history["loss"]),
        "best_val_pr_auc_keras": float(np.max(history.history["val_pr_auc"])),
    }


def plot_pr_curves(predictions, y_valid, y_holdout):
    for split, y_true, key in [("valid", y_valid, "valid_prob"), ("holdout", y_holdout, "holdout_prob")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, pred in predictions.items():
            precision, recall, _ = precision_recall_curve(y_true, pred[key])
            ax.plot(recall, precision, label=name)
        ax.axhline(float(y_true.mean()), color="black", linestyle=":", linewidth=1, label="fraud rate")
        ax.set_title(f"V07 Precision-Recall curves - {split}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(PLOTS / f"v07_pr_curves_{split}.png", dpi=140)
        plt.close()


def write_report(metrics, summaries, feature_cols):
    lines = []
    lines.append("# V07 MLP benchmark y complemento experimental\n\n")
    lines.append("## Objetivo\n\n")
    lines.append("Agregar una comparacion reproducible con MLP y un baseline tradicional sobre el mismo split temporal 70/15/15 usado por V01.\n\n")
    lines.append("## Relacion con el EDA de referencia\n\n")
    lines.append("El kernel de Fabryzzio aporta cuatro ideas que complementan el EDA: faltantes por bloques V, ausencia estructural de identity, montos con mas de dos decimales y riesgo de clientes/UID recurrentes. V07 incorpora `TransactionAmt_multi_decimal` y mantiene la evaluacion por UID conocido/desconocido; no cambia el split principal para conservar comparabilidad con V01-V06.\n\n")
    lines.append(f"Features usadas: {len(feature_cols)}.\n\n")
    lines.append("## Modelos\n\n")
    lines.append(pd.DataFrame(summaries).to_markdown(index=False))
    lines.append("\n\n## Resultados globales\n\n")
    cols = ["model", "split", "segment", "rows", "fraud_rate", "pr_auc", "roc_auc"]
    lines.append(metrics[metrics["segment"] == "global"][cols].to_markdown(index=False))
    lines.append("\n\n## Holdout por UID\n\n")
    lines.append(metrics[(metrics["split"] == "holdout") & (metrics["segment"] != "global")][cols].to_markdown(index=False))
    lines.append("\n\n## Decision\n\n")
    lines.append("El MLP se acepta como benchmark adicional de deep learning. Solo reemplazaria a V01 si mejora PR-AUC holdout y no degrada UID desconocido. Si queda por debajo, su valor es cerrar la comparacion experimental y reforzar que los modelos tabulares de boosting son mas adecuados para esta primera version.\n")
    (OUT / "v07_mlp_report.md").write_text("".join(lines), encoding="utf-8")


def main():
    seed_everything()
    print("Loading data...")
    df = load_train()
    train_idx, valid_idx, holdout_idx, cutoffs = temporal_split(df)
    df, train_stats = add_base_features(df, train_stats=None)

    cat_cols = [
        "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
        "DeviceType", "DeviceInfo",
    ] + [f"M{i}" for i in range(1, 10) if f"M{i}" in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]
    maps, freqs = build_encoding_maps(df.loc[train_idx], cat_cols)
    df = apply_encodings(df, cat_cols, maps, freqs)

    uid = candidate_uid(df)
    known_valid = uid.loc[valid_idx].isin(set(uid.loc[train_idx]))
    known_holdout = uid.loc[holdout_idx].isin(set(uid.loc[df.index[df.index <= valid_idx.max()]]))

    X, feature_cols = prepare_matrix(df, train_idx)
    y = df["isFraud"].astype("int8")
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]
    X_holdout, y_holdout = X.loc[holdout_idx], y.loc[holdout_idx]
    scale_pos_weight = float((y_train == 0).sum() / max(int((y_train == 1).sum()), 1))

    print(f"Features: {len(feature_cols)}. Scale pos weight: {scale_pos_weight:.2f}")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train)).astype("float32")
    X_valid_scaled = scaler.transform(imputer.transform(X_valid)).astype("float32")
    X_holdout_scaled = scaler.transform(imputer.transform(X_holdout)).astype("float32")

    model_specs = []
    predictions = {}
    metrics_frames = []

    for name, fitter, args in [
        ("lightgbm_v01_feature_surface", fit_lightgbm, (X_train, y_train, X_valid, y_valid, X_holdout)),
        ("sgd_logistic_traditional", fit_logistic_sgd, (X_train_scaled, y_train, X_valid_scaled, X_holdout_scaled)),
        ("mlp_keras_weighted", fit_mlp_keras, (X_train_scaled, y_train.to_numpy(), X_valid_scaled, y_valid.to_numpy(), X_holdout_scaled, scale_pos_weight)),
    ]:
        print(f"Training {name}...")
        start = time.time()
        valid_prob, holdout_prob, extra = fitter(*args)
        elapsed = time.time() - start
        predictions[name] = {"valid_prob": valid_prob, "holdout_prob": holdout_prob}
        metrics_frames.append(evaluate_all(name, y, valid_idx, holdout_idx, valid_prob, holdout_prob, known_valid, known_holdout))
        model_specs.append({
            "model": name,
            "elapsed_seconds": float(elapsed),
            "features": len(feature_cols),
            **extra,
            **cutoffs,
        })
        print(f"{name} complete in {elapsed:.1f}s")

    metrics = pd.concat(metrics_frames, ignore_index=True)
    summary = pd.DataFrame(model_specs)
    metrics.to_csv(OUT / "v07_mlp_benchmark_metrics.csv", index=False)
    summary.to_csv(OUT / "v07_mlp_benchmark_summary.csv", index=False)
    (OUT / "v07_mlp_benchmark_summary.json").write_text(json.dumps(model_specs, indent=2), encoding="utf-8")

    pred_rows = []
    for name, pred in predictions.items():
        tmp_valid = pd.DataFrame({
            "model": name,
            "split": "valid",
            "TransactionID": df.loc[valid_idx, "TransactionID"].to_numpy(),
            "isFraud": y_valid.to_numpy(),
            "pred": pred["valid_prob"],
            "uid_known": known_valid.to_numpy(),
        })
        tmp_holdout = pd.DataFrame({
            "model": name,
            "split": "holdout",
            "TransactionID": df.loc[holdout_idx, "TransactionID"].to_numpy(),
            "isFraud": y_holdout.to_numpy(),
            "pred": pred["holdout_prob"],
            "uid_known": known_holdout.to_numpy(),
        })
        pred_rows.extend([tmp_valid, tmp_holdout])
    pd.concat(pred_rows, ignore_index=True).to_csv(OUT / "v07_predictions.csv", index=False)

    plot_pr_curves(predictions, y_valid, y_holdout)
    write_report(metrics, model_specs, feature_cols)

    print("V07 metrics:")
    print(metrics[["model", "split", "segment", "rows", "fraud_rate", "pr_auc", "roc_auc"]].to_string(index=False))
    print("V07 summary:")
    print(summary.to_string(index=False))
    print("V07 complete.")


if __name__ == "__main__":
    main()
