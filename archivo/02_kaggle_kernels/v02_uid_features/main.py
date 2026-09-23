import gc
import json
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
    raise RuntimeError("LightGBM is required for this kernel") from exc


DATA_DIR_CANDIDATES = [
    Path("/kaggle/input/ieee-fraud-detection"),
    Path("/kaggle/input/competitions/ieee-fraud-detection"),
]
DATA_DIR = next((path for path in DATA_DIR_CANDIDATES if path.exists()), DATA_DIR_CANDIDATES[0])
OUT = Path("/kaggle/working")
PLOTS = OUT / "plots"
PLOTS.mkdir(exist_ok=True)
RANDOM_STATE = 42


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
    df["DT_hour"] = ((df["TransactionDT"] // 3600) % 24).astype("int8")
    df["DT_day_index"] = day
    df["DT_week_index"] = (day // 7).astype("int16")

    identity_cols = [c for c in df.columns if c.startswith("id_") or c in ["DeviceType", "DeviceInfo"]]
    df["has_identity"] = df[identity_cols[0]].notna().astype("int8") if identity_cols else 0

    df["TransactionAmt_log1p"] = np.log1p(df["TransactionAmt"].astype("float64")).astype("float32")

    if train_stats is None:
        q1, q3 = df["TransactionAmt"].quantile([0.25, 0.75])
        iqr = q3 - q1
        train_stats = {"amt_iqr_low": float(q1 - 1.5 * iqr), "amt_iqr_high": float(q3 + 1.5 * iqr)}

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


def make_uid_strings(df):
    day = (df["TransactionDT"] // (24 * 3600)).astype("float32")
    d1n = (df["D1"].astype("float32") - day).round(0).astype("string").fillna("NA")
    d2n = (df["D2"].astype("float32") - day).round(0).astype("string").fillna("NA")

    def s(col):
        return df[col].astype("string").fillna("NA") if col in df.columns else pd.Series("NA", index=df.index, dtype="string")

    uids = pd.DataFrame(index=df.index)
    uids["uid_card_addr_d1"] = s("card1") + "_" + s("card2") + "_" + s("card3") + "_" + s("card5") + "_" + s("addr1") + "_" + s("addr2") + "_" + d1n
    uids["uid_card_addr"] = s("card1") + "_" + s("addr1")
    uids["uid_card_email"] = s("card1") + "_" + s("card2") + "_" + s("P_emaildomain")
    uids["uid_card_product"] = s("card1") + "_" + s("ProductCD") + "_" + s("addr1")
    uids["uid_card_d2"] = s("card1") + "_" + s("card2") + "_" + d2n
    return uids


def candidate_uid_for_segments(df):
    return pd.util.hash_pandas_object(make_uid_strings(df)[["uid_card_addr_d1"]], index=False).astype("uint64")


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


def add_uid_features(df, train_idx, mode):
    if mode == "base":
        return df, []

    df = df.copy()
    uid_strings = make_uid_strings(df)
    train_uids = uid_strings.loc[train_idx]
    added = []

    for col in uid_strings.columns:
        train_s = train_uids[col].astype("string").fillna("NA")
        all_s = uid_strings[col].astype("string").fillna("NA")
        counts = train_s.value_counts(dropna=False)
        freqs = counts / len(train_s)
        df[f"{col}_count_train"] = all_s.map(counts).fillna(0).astype("float32")
        df[f"{col}_freq_train"] = all_s.map(freqs).fillna(0).astype("float32")
        df[f"{col}_known_train"] = (df[f"{col}_count_train"] > 0).astype("int8")
        added += [f"{col}_count_train", f"{col}_freq_train", f"{col}_known_train"]

    if mode != "uid_freq_amt":
        return df, added

    global_amt_mean = float(df.loc[train_idx, "TransactionAmt"].mean())
    global_amt_std = float(df.loc[train_idx, "TransactionAmt"].std())
    for col in uid_strings.columns:
        train_s = train_uids[col].astype("string").fillna("NA")
        stats_df = pd.DataFrame({
            "uid": train_s,
            "amt": df.loc[train_idx, "TransactionAmt"].astype("float32"),
        }).groupby("uid")["amt"].agg(["mean", "std", "median", "max", "min"])
        all_s = uid_strings[col].astype("string").fillna("NA")
        for stat in ["mean", "std", "median", "max", "min"]:
            name = f"{col}_amt_{stat}_train"
            fill_value = global_amt_std if stat == "std" else global_amt_mean
            df[name] = all_s.map(stats_df[stat]).fillna(fill_value).astype("float32")
            added.append(name)
        mean_col = f"{col}_amt_mean_train"
        std_col = f"{col}_amt_std_train"
        diff_name = f"{col}_amt_diff_mean_train"
        z_name = f"{col}_amt_z_train"
        df[diff_name] = (df["TransactionAmt"].astype("float32") - df[mean_col]).astype("float32")
        df[z_name] = (df[diff_name] / (df[std_col].replace(0, np.nan))).replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
        added += [diff_name, z_name]
    return df, added


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
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "flagged_rate": float(y_pred.mean()),
    }


def evaluate_segment(model_name, split_name, segment, y_true, y_prob):
    if len(y_true) == 0 or y_true.nunique() < 2:
        return {"model": model_name, "split": split_name, "segment": segment, "rows": len(y_true), "fraud_rate": float(y_true.mean()) if len(y_true) else np.nan, "pr_auc": np.nan, "roc_auc": np.nan}
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


def train_one(df_base, train_idx, valid_idx, holdout_idx, uid_segment, mode):
    work = df_base.copy()
    work, uid_added = add_uid_features(work, train_idx, mode)
    X, feature_cols = prepare_matrix(work, train_idx)
    y = work["isFraud"].astype("int8")

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
        X.loc[train_idx],
        y.loc[train_idx],
        eval_set=[(X.loc[valid_idx], y.loc[valid_idx])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(100)],
    )

    valid_prob = model.predict_proba(X.loc[valid_idx])[:, 1]
    holdout_prob = model.predict_proba(X.loc[holdout_idx])[:, 1]
    known_valid = uid_segment.loc[valid_idx].isin(set(uid_segment.loc[train_idx]))
    known_holdout = uid_segment.loc[holdout_idx].isin(set(uid_segment.loc[train_idx]))

    metrics = [
        evaluate_segment(mode, "valid", "global", y.loc[valid_idx], valid_prob),
        evaluate_segment(mode, "valid", "uid_known", y.loc[valid_idx][known_valid.values], valid_prob[known_valid.values]),
        evaluate_segment(mode, "valid", "uid_unknown", y.loc[valid_idx][~known_valid.values], valid_prob[~known_valid.values]),
        evaluate_segment(mode, "holdout", "global", y.loc[holdout_idx], holdout_prob),
        evaluate_segment(mode, "holdout", "uid_known", y.loc[holdout_idx][known_holdout.values], holdout_prob[known_holdout.values]),
        evaluate_segment(mode, "holdout", "uid_unknown", y.loc[holdout_idx][~known_holdout.values], holdout_prob[~known_holdout.values]),
    ]
    windows = pd.concat([
        window_metrics(mode, "valid", work.loc[valid_idx], valid_prob),
        window_metrics(mode, "holdout", work.loc[holdout_idx], holdout_prob),
    ], ignore_index=True)
    importance = pd.DataFrame({
        "model": mode,
        "feature": feature_cols,
        "importance": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values(["model", "importance"], ascending=[True, False])
    preds_valid = pd.DataFrame({
        "model": mode,
        "TransactionID": work.loc[valid_idx, "TransactionID"].to_numpy(),
        "isFraud": y.loc[valid_idx].to_numpy(),
        "pred": valid_prob,
        "uid_known": known_valid.to_numpy(),
    })
    preds_holdout = pd.DataFrame({
        "model": mode,
        "TransactionID": work.loc[holdout_idx, "TransactionID"].to_numpy(),
        "isFraud": y.loc[holdout_idx].to_numpy(),
        "pred": holdout_prob,
        "uid_known": known_holdout.to_numpy(),
    })
    summary = {
        "model": mode,
        "features": int(len(feature_cols)),
        "uid_features_added": int(len(uid_added)),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
    }
    del X, work, model
    gc.collect()
    return pd.DataFrame(metrics), windows, importance, preds_valid, preds_holdout, summary


def plot_comparison(metrics):
    subset = metrics[(metrics["segment"] == "global")].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, g in subset.groupby("split"):
        ax.plot(g["model"], g["pr_auc"], marker="o", label=split)
    ax.set_title("V02 ablations - PR-AUC global")
    ax.set_ylabel("PR-AUC")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "v02_global_pr_auc.png", dpi=140)
    plt.close()


def write_report(summary, metrics):
    lines = []
    lines.append("# V02 UID features - IEEE-CIS Fraud Detection\n\n")
    lines.append("## Objetivo\n\n")
    lines.append("Evaluar si features historicas por UID mejoran el baseline temporal V01 sin usar informacion futura ni target encoding.\n\n")
    lines.append("## Ablations\n\n")
    lines.append("- `base`: baseline comparable a V01.\n")
    lines.append("- `uid_freq`: base + conteos/frecuencias/conocido por UID.\n")
    lines.append("- `uid_freq_amt`: uid_freq + estadisticas historicas de `TransactionAmt` por UID.\n\n")
    lines.append("## Regla anti-leakage\n\n")
    lines.append("Las features UID para validacion y holdout se calculan usando solo la ventana de entrenamiento. No se usa `isFraud` para construirlas.\n\n")
    lines.append("## Resultados globales\n\n")
    global_rows = metrics[metrics["segment"] == "global"][["model", "split", "rows", "fraud_rate", "pr_auc", "roc_auc"]]
    lines.append(global_rows.to_markdown(index=False))
    lines.append("\n\n")
    lines.append("## Resumen de features\n\n")
    lines.append(pd.DataFrame(summary).to_markdown(index=False))
    lines.append("\n")
    (OUT / "v02_report.md").write_text("".join(lines), encoding="utf-8")


def main():
    print("Loading data...")
    df = load_train()
    train_idx, valid_idx, holdout_idx, cutoffs = temporal_split(df)
    print("Split sizes:", len(train_idx), len(valid_idx), len(holdout_idx))
    df, _ = add_base_features(df)

    cat_cols = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]
    cat_cols += [f"M{i}" for i in range(1, 10) if f"M{i}" in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]
    maps, freqs = build_encoding_maps(df.loc[train_idx], cat_cols)
    df = apply_encodings(df, cat_cols, maps, freqs)
    uid_segment = candidate_uid_for_segments(df)

    all_metrics, all_windows, all_importance, all_valid_preds, all_holdout_preds, summaries = [], [], [], [], [], []
    for mode in ["base", "uid_freq", "uid_freq_amt"]:
        print(f"Training ablation: {mode}")
        metrics, windows, importance, valid_preds, holdout_preds, summary = train_one(df, train_idx, valid_idx, holdout_idx, uid_segment, mode)
        all_metrics.append(metrics)
        all_windows.append(windows)
        all_importance.append(importance)
        all_valid_preds.append(valid_preds)
        all_holdout_preds.append(holdout_preds)
        summaries.append({**summary, **cutoffs})

    metrics = pd.concat(all_metrics, ignore_index=True)
    windows = pd.concat(all_windows, ignore_index=True)
    importance = pd.concat(all_importance, ignore_index=True)
    valid_preds = pd.concat(all_valid_preds, ignore_index=True)
    holdout_preds = pd.concat(all_holdout_preds, ignore_index=True)

    metrics.to_csv(OUT / "v02_model_comparison.csv", index=False)
    windows.to_csv(OUT / "v02_window_metrics.csv", index=False)
    importance.to_csv(OUT / "v02_feature_importance.csv", index=False)
    valid_preds.to_csv(OUT / "v02_valid_predictions.csv", index=False)
    holdout_preds.to_csv(OUT / "v02_holdout_predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "v02_summary.csv", index=False)
    (OUT / "v02_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    plot_comparison(metrics)
    write_report(summaries, metrics)

    print("V02 metrics:")
    print(metrics[["model", "split", "segment", "rows", "fraud_rate", "pr_auc", "roc_auc"]].to_string(index=False))
    print("V02 complete.")


if __name__ == "__main__":
    main()
