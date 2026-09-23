import gc
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except Exception as exc:
    raise RuntimeError("LightGBM is required for V04") from exc

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


DATA_DIR_CANDIDATES = [
    Path("/kaggle/input/ieee-fraud-detection"),
    Path("/kaggle/input/competitions/ieee-fraud-detection"),
]
DATA_DIR = next((path for path in DATA_DIR_CANDIDATES if path.exists()), DATA_DIR_CANDIDATES[0])
OUT = Path("/kaggle/working")
PLOTS = OUT / "plots"
PLOTS.mkdir(exist_ok=True)
RANDOM_STATE = 42

KEEP_FEATURES = ['V258', 'C5', 'D3', 'C13', 'C1', 'C14', 'card1', 'card2', 'TransactionAmt', 'addr1', 'D2', 'C11', 'card6_label', 'P_emaildomain_label', 'C6', 'card5', 'D4', 'M4_freq', 'V257', 'D15', 'C2', 'C8', 'dist1', 'card3', 'card6_freq', 'P_emaildomain_freq', 'D8', 'M4_label', 'M5_label', 'D1', 'D10', 'C10', 'M5_freq', 'C4', 'ProductCD_label', 'R_emaildomain_label', 'V45', 'D5', 'M6_freq', 'TransactionAmt_log1p', 'C12', 'DeviceInfo_freq', 'V78', 'id_19']
MONITOR_FEATURES = ['D1n', 'D2n', 'D15n', 'D10n', 'C9', 'id_02', 'id_20', 'D11', 'id_01']


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


def add_base_features(df):
    df = df.copy()
    day = (df["TransactionDT"] // (24 * 3600)).astype("int16")
    df["DT_hour"] = ((df["TransactionDT"] // 3600) % 24).astype("int8")
    df["DT_day_index"] = day
    df["DT_week_index"] = (day // 7).astype("int16")
    identity_cols = [c for c in df.columns if c.startswith("id_") or c in ["DeviceType", "DeviceInfo"]]
    df["has_identity"] = df[identity_cols[0]].notna().astype("int8") if identity_cols else 0
    df["TransactionAmt_log1p"] = np.log1p(df["TransactionAmt"].astype("float64")).astype("float32")
    q1, q3 = df["TransactionAmt"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df["TransactionAmt_outlier_iqr"] = ((df["TransactionAmt"] < q1 - 1.5 * iqr) | (df["TransactionAmt"] > q3 + 1.5 * iqr)).astype("int8")
    for col in ["D1", "D2", "D10", "D15"]:
        if col in df.columns:
            df[f"{col}n"] = (df[col].astype("float32") - day.astype("float32")).astype("float32")
    return df


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


def candidate_uid_for_segments(df):
    day = (df["TransactionDT"] // (24 * 3600)).astype("float32")
    parts = pd.DataFrame(index=df.index)
    for col in ["card1", "card2", "card3", "card5", "addr1", "addr2"]:
        parts[col] = df[col].astype("string").fillna("NA")
    parts["D1_anchor"] = (df["D1"].astype("float32") - day).round(0).astype("string").fillna("NA")
    return pd.util.hash_pandas_object(parts, index=False).astype("uint64")


def feature_matrix(df, features):
    existing = [f for f in features if f in df.columns]
    X = df[existing].copy()
    for col in existing:
        if X[col].dtype == "bool":
            X[col] = X[col].astype("int8")
        elif not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X, existing


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


def evaluate(model_name, split_name, segment, y_true, y_prob):
    if len(y_true) == 0 or y_true.nunique() < 2:
        return {"model": model_name, "split": split_name, "segment": segment, "rows": int(len(y_true)), "fraud_rate": float(y_true.mean()) if len(y_true) else np.nan, "pr_auc": np.nan, "roc_auc": np.nan}
    out = {
        "model": model_name,
        "split": split_name,
        "segment": segment,
        "rows": int(len(y_true)),
        "fraud_rate": float(y_true.mean()),
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
    for prefix, vals in [("thr05", metrics_at_threshold(y_true, y_prob, 0.5)), ("top5", metrics_at_threshold(y_true, y_prob, np.quantile(y_prob, 0.95)))]:
        for k, v in vals.items():
            out[f"{prefix}_{k}"] = v
    return out


def window_metrics(model_name, split_name, df_eval, y_prob):
    tmp = df_eval[["isFraud", "DT_week_index"]].copy()
    tmp["prob"] = y_prob
    rows = []
    for window, g in tmp.groupby("DT_week_index"):
        if g["isFraud"].nunique() < 2:
            continue
        rows.append({
            "model": model_name,
            "split": split_name,
            "window": int(window),
            "rows": int(len(g)),
            "fraud_rate": float(g["isFraud"].mean()),
            "pr_auc": average_precision_score(g["isFraud"], g["prob"]),
            "roc_auc": roc_auc_score(g["isFraud"], g["prob"]),
        })
    return pd.DataFrame(rows)


def make_model(kind, scale_pos_weight):
    if kind == "lgbm":
        return lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=650,
            learning_rate=0.04,
            num_leaves=48,
            min_child_samples=120,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.08,
            reg_lambda=0.30,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
    if kind == "xgb" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.04,
            max_depth=6,
            min_child_weight=20,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.05,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
        )
    if kind == "cat" and CatBoostClassifier is not None:
        return CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=8,
            loss_function="Logloss",
            eval_metric="PRAUC",
            random_seed=RANDOM_STATE,
            verbose=100,
            auto_class_weights="Balanced",
            allow_writing_files=False,
        )
    return None


def fit_predict(kind, model, X_train, y_train, X_valid, y_valid, X_holdout):
    if kind == "lgbm":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)],
        )
        return model.predict_proba(X_valid)[:, 1], model.predict_proba(X_holdout)[:, 1], int(model.best_iteration_ or model.n_estimators)
    if kind == "xgb":
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        return model.predict_proba(X_valid)[:, 1], model.predict_proba(X_holdout)[:, 1], int(getattr(model, "best_iteration", 0) or model.n_estimators)
    if kind == "cat":
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
        return model.predict_proba(X_valid)[:, 1], model.predict_proba(X_holdout)[:, 1], int(model.get_best_iteration() or model.get_param("iterations"))
    raise ValueError(kind)


def run_candidate(name, kind, features, df, y, uid, train_idx, valid_idx, holdout_idx):
    X, used_features = feature_matrix(df, features)
    scale_pos_weight = float((y.loc[train_idx] == 0).sum() / max((y.loc[train_idx] == 1).sum(), 1))
    model = make_model(kind, scale_pos_weight)
    if model is None:
        return None, None, {"model": name, "status": "skipped_missing_dependency", "features": len(used_features)}
    start = time.time()
    valid_prob, holdout_prob, best_iter = fit_predict(kind, model, X.loc[train_idx], y.loc[train_idx], X.loc[valid_idx], y.loc[valid_idx], X.loc[holdout_idx])
    elapsed = time.time() - start
    known_valid = uid.loc[valid_idx].isin(set(uid.loc[train_idx]))
    known_holdout = uid.loc[holdout_idx].isin(set(uid.loc[train_idx]))
    metrics = [
        evaluate(name, "valid", "global", y.loc[valid_idx], valid_prob),
        evaluate(name, "valid", "uid_known", y.loc[valid_idx][known_valid.values], valid_prob[known_valid.values]),
        evaluate(name, "valid", "uid_unknown", y.loc[valid_idx][~known_valid.values], valid_prob[~known_valid.values]),
        evaluate(name, "holdout", "global", y.loc[holdout_idx], holdout_prob),
        evaluate(name, "holdout", "uid_known", y.loc[holdout_idx][known_holdout.values], holdout_prob[known_holdout.values]),
        evaluate(name, "holdout", "uid_unknown", y.loc[holdout_idx][~known_holdout.values], holdout_prob[~known_holdout.values]),
    ]
    windows = pd.concat([
        window_metrics(name, "valid", df.loc[valid_idx], valid_prob),
        window_metrics(name, "holdout", df.loc[holdout_idx], holdout_prob),
    ], ignore_index=True)
    summary = {
        "model": name,
        "kind": kind,
        "status": "complete",
        "features": len(used_features),
        "best_iteration": best_iter,
        "elapsed_seconds": elapsed,
    }
    del X, model
    gc.collect()
    return pd.DataFrame(metrics), windows, summary


def plot_results(metrics):
    global_rows = metrics[(metrics["segment"] == "global")].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    for split, g in global_rows.groupby("split"):
        ax.plot(g["model"], g["pr_auc"], marker="o", label=split)
    ax.set_title("V04 PR-AUC global")
    ax.set_ylabel("PR-AUC")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "v04_global_pr_auc.png", dpi=140)
    plt.close()


def write_report(metrics, summaries):
    lines = []
    lines.append("# V04 model comparison - IEEE-CIS Fraud Detection\n\n")
    lines.append("## Objetivo\n\n")
    lines.append("Comparar feature sets auditados y modelos sobre el mismo split temporal.\n\n")
    lines.append("## Candidatos\n\n")
    lines.append(pd.DataFrame(summaries).to_markdown(index=False))
    lines.append("\n\n")
    lines.append("## Resultados globales\n\n")
    cols = ["model", "split", "segment", "rows", "fraud_rate", "pr_auc", "roc_auc"]
    lines.append(metrics[metrics["segment"] == "global"][cols].to_markdown(index=False))
    lines.append("\n\n")
    lines.append("## Resultados por UID en holdout\n\n")
    lines.append(metrics[(metrics["split"] == "holdout") & (metrics["segment"] != "global")][cols].to_markdown(index=False))
    lines.append("\n")
    (OUT / "v04_report.md").write_text("".join(lines), encoding="utf-8")


def main():
    print("Loading data...")
    df = load_train()
    train_idx, valid_idx, holdout_idx, cutoffs = temporal_split(df)
    df = add_base_features(df)
    cat_cols = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]
    cat_cols += [f"M{i}" for i in range(1, 10) if f"M{i}" in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]
    maps, freqs = build_encoding_maps(df.loc[train_idx], cat_cols)
    df = apply_encodings(df, cat_cols, maps, freqs)
    y = df["isFraud"].astype("int8")
    uid = candidate_uid_for_segments(df)
    keep_monitor = KEEP_FEATURES + [f for f in MONITOR_FEATURES if f not in KEEP_FEATURES]
    candidates = [
        ("lgbm_keep", "lgbm", KEEP_FEATURES),
        ("lgbm_keep_monitor", "lgbm", keep_monitor),
        ("xgb_keep_monitor", "xgb", keep_monitor),
        ("cat_keep_monitor", "cat", keep_monitor),
    ]
    all_metrics, all_windows, summaries = [], [], []
    for name, kind, features in candidates:
        print(f"Running candidate: {name}")
        metrics, windows, summary = run_candidate(name, kind, features, df, y, uid, train_idx, valid_idx, holdout_idx)
        summaries.append({**summary, **cutoffs})
        if metrics is not None:
            all_metrics.append(metrics)
            all_windows.append(windows)
    metrics = pd.concat(all_metrics, ignore_index=True)
    windows = pd.concat(all_windows, ignore_index=True)
    metrics.to_csv(OUT / "v04_model_comparison.csv", index=False)
    windows.to_csv(OUT / "v04_window_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "v04_summary.csv", index=False)
    (OUT / "v04_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    plot_results(metrics)
    write_report(metrics, summaries)
    print("V04 metrics:")
    print(metrics[["model", "split", "segment", "rows", "fraud_rate", "pr_auc", "roc_auc"]].to_string(index=False))
    print("V04 summary:")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("V04 complete.")


if __name__ == "__main__":
    main()
