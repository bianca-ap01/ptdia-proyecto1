import gc
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

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


def train_baseline(X, y, train_idx, valid_idx):
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
    return model


def ks_stat(a, b, sample_n=120000):
    a = a.dropna()
    b = b.dropna()
    if len(a) < 100 or len(b) < 100:
        return np.nan
    rng = np.random.default_rng(RANDOM_STATE)
    if len(a) > sample_n:
        a = a.iloc[rng.choice(len(a), sample_n, replace=False)]
    if len(b) > sample_n:
        b = b.iloc[rng.choice(len(b), sample_n, replace=False)]
    try:
        return float(stats.ks_2samp(a.astype("float64"), b.astype("float64")).statistic)
    except Exception:
        return np.nan


def feature_distribution_audit(X, train_idx, valid_idx, holdout_idx):
    rows = []
    for col in X.columns:
        tr = X.loc[train_idx, col]
        va = X.loc[valid_idx, col]
        ho = X.loc[holdout_idx, col]
        rows.append({
            "feature": col,
            "train_missing_pct": float(100 * tr.isna().mean()),
            "valid_missing_pct": float(100 * va.isna().mean()),
            "holdout_missing_pct": float(100 * ho.isna().mean()),
            "missing_delta_train_valid": float(abs(100 * tr.isna().mean() - 100 * va.isna().mean())),
            "missing_delta_train_holdout": float(abs(100 * tr.isna().mean() - 100 * ho.isna().mean())),
            "ks_train_valid": ks_stat(tr, va),
            "ks_train_holdout": ks_stat(tr, ho),
            "train_nunique": int(tr.nunique(dropna=True)),
            "valid_nunique": int(va.nunique(dropna=True)),
            "holdout_nunique": int(ho.nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


def permutation_importance(model, X_valid, y_valid, top_features, base_score):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    X_small = X_valid.copy()
    if len(X_small) > 60000:
        idx = X_small.sample(60000, random_state=RANDOM_STATE).index
        X_small = X_small.loc[idx]
        y_small = y_valid.loc[idx]
        base = average_precision_score(y_small, model.predict_proba(X_small)[:, 1])
    else:
        y_small = y_valid
        base = base_score
    for col in top_features:
        Xp = X_small.copy()
        Xp[col] = rng.permutation(Xp[col].to_numpy())
        pred = model.predict_proba(Xp)[:, 1]
        score = average_precision_score(y_small, pred)
        rows.append({
            "feature": col,
            "perm_pr_auc": float(score),
            "perm_pr_auc_drop": float(base - score),
        })
    return pd.DataFrame(rows).sort_values("perm_pr_auc_drop", ascending=False)


def classify_features(audit):
    audit = audit.copy()
    audit["max_ks"] = audit[["ks_train_valid", "ks_train_holdout"]].max(axis=1)
    audit["max_missing_delta"] = audit[["missing_delta_train_valid", "missing_delta_train_holdout"]].max(axis=1)
    audit["gain_rank"] = audit["gain_importance"].rank(ascending=False, method="min")
    audit["perm_rank"] = audit["perm_pr_auc_drop"].rank(ascending=False, method="min")
    audit["is_predictive"] = (audit["gain_rank"] <= 40) | (audit["perm_pr_auc_drop"].fillna(0) > 0.001)
    audit["is_shifted"] = (audit["max_ks"].fillna(0) >= 0.12) | (audit["max_missing_delta"] >= 10)
    audit["recommendation"] = "low_priority"
    audit.loc[audit["is_predictive"] & ~audit["is_shifted"], "recommendation"] = "keep"
    audit.loc[audit["is_predictive"] & audit["is_shifted"], "recommendation"] = "monitor"
    audit.loc[~audit["is_predictive"] & audit["is_shifted"], "recommendation"] = "review"
    return audit.sort_values(["recommendation", "gain_importance"], ascending=[True, False])


def plot_outputs(audit):
    rec_counts = audit["recommendation"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    rec_counts.plot(kind="bar", ax=ax)
    ax.set_title("Feature audit recommendations")
    ax.set_ylabel("features")
    plt.tight_layout()
    plt.savefig(PLOTS / "feature_recommendations.png", dpi=140)
    plt.close()

    top = audit.sort_values("gain_importance", ascending=False).head(25).sort_values("gain_importance")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["feature"], top["gain_importance"])
    ax.set_title("Top gain importance")
    plt.tight_layout()
    plt.savefig(PLOTS / "top_gain_importance.png", dpi=140)
    plt.close()


def write_report(summary, audit, model_metrics):
    lines = []
    lines.append("# V03 feature audit - IEEE-CIS Fraud Detection\n\n")
    lines.append("## Objetivo\n\n")
    lines.append("Auditar features del baseline V01 combinando importancia predictiva, permutation importance, shift temporal y cambios de missingness.\n\n")
    lines.append("## Metricas del modelo auditado\n\n")
    lines.append(pd.DataFrame([model_metrics]).to_markdown(index=False))
    lines.append("\n\n")
    lines.append("## Recomendaciones\n\n")
    rec = audit["recommendation"].value_counts().rename_axis("recommendation").reset_index(name="features")
    lines.append(rec.to_markdown(index=False))
    lines.append("\n\n")
    lines.append("## Top features a monitorear\n\n")
    monitor = audit[audit["recommendation"] == "monitor"].sort_values("gain_importance", ascending=False).head(20)
    cols = ["feature", "gain_importance", "perm_pr_auc_drop", "max_ks", "max_missing_delta"]
    lines.append(monitor[cols].to_markdown(index=False))
    lines.append("\n\n")
    lines.append("## Criterio\n\n")
    lines.append("- `keep`: predictiva y sin shift alto.\n")
    lines.append("- `monitor`: predictiva, pero con shift o cambio de missingness.\n")
    lines.append("- `review`: poco predictiva, pero con shift alto.\n")
    lines.append("- `low_priority`: baja evidencia predictiva y sin alerta fuerte.\n")
    (OUT / "v03_feature_audit_report.md").write_text("".join(lines), encoding="utf-8")
    (OUT / "v03_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


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
    X, feature_cols = prepare_matrix(df, train_idx)
    y = df["isFraud"].astype("int8")

    print("Training audit baseline...")
    model = train_baseline(X, y, train_idx, valid_idx)
    valid_pred = model.predict_proba(X.loc[valid_idx])[:, 1]
    holdout_pred = model.predict_proba(X.loc[holdout_idx])[:, 1]
    model_metrics = {
        "valid_pr_auc": float(average_precision_score(y.loc[valid_idx], valid_pred)),
        "valid_roc_auc": float(roc_auc_score(y.loc[valid_idx], valid_pred)),
        "holdout_pr_auc": float(average_precision_score(y.loc[holdout_idx], holdout_pred)),
        "holdout_roc_auc": float(roc_auc_score(y.loc[holdout_idx], holdout_pred)),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "features": int(len(feature_cols)),
    }

    print("Computing distribution audit...")
    dist = feature_distribution_audit(X, train_idx, valid_idx, holdout_idx)
    gain = pd.DataFrame({
        "feature": feature_cols,
        "gain_importance": model.booster_.feature_importance(importance_type="gain"),
        "split_importance": model.booster_.feature_importance(importance_type="split"),
    })
    top_features = gain.sort_values("gain_importance", ascending=False).head(60)["feature"].tolist()
    print("Computing permutation importance for top features...")
    perm = permutation_importance(model, X.loc[valid_idx], y.loc[valid_idx], top_features, model_metrics["valid_pr_auc"])
    audit = dist.merge(gain, on="feature", how="left").merge(perm, on="feature", how="left")
    audit["perm_pr_auc_drop"] = audit["perm_pr_auc_drop"].fillna(0)
    audit = classify_features(audit)

    summary = {
        **cutoffs,
        **model_metrics,
        "recommendations": audit["recommendation"].value_counts().to_dict(),
        "top_monitor_features": audit[audit["recommendation"] == "monitor"].sort_values("gain_importance", ascending=False).head(20)["feature"].tolist(),
    }

    audit.to_csv(OUT / "feature_audit.csv", index=False)
    dist.to_csv(OUT / "feature_distribution_audit.csv", index=False)
    gain.sort_values("gain_importance", ascending=False).to_csv(OUT / "gain_importance.csv", index=False)
    perm.to_csv(OUT / "permutation_importance.csv", index=False)
    pd.DataFrame([model_metrics]).to_csv(OUT / "v03_model_metrics.csv", index=False)
    plot_outputs(audit)
    write_report(summary, audit, model_metrics)

    print("V03 model metrics:")
    print(json.dumps(model_metrics, indent=2))
    print("Recommendations:")
    print(audit["recommendation"].value_counts().to_string())
    print("Top monitor features:")
    print(audit[audit["recommendation"] == "monitor"].sort_values("gain_importance", ascending=False).head(20)[["feature", "gain_importance", "perm_pr_auc_drop", "max_ks", "max_missing_delta"]].to_string(index=False))
    print("V03 complete.")


if __name__ == "__main__":
    main()
