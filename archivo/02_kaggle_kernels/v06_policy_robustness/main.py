import gc
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except Exception as exc:
    raise RuntimeError("LightGBM is required for this baseline kernel") from exc


DATA_DIR_CANDIDATES = [
    Path("/kaggle/input/ieee-fraud-detection"),
    Path("/kaggle/input/competitions/ieee-fraud-detection"),
]
DATA_DIR = next((path for path in DATA_DIR_CANDIDATES if path.exists()), DATA_DIR_CANDIDATES[0])
OUT = Path("/kaggle/working")
PLOTS = OUT / "plots"
PLOTS.mkdir(exist_ok=True)

RANDOM_STATE = 42
SCENARIOS = [
    {
        "scenario": "base",
        "cost_fraud_approved": 100.0,
        "cost_legit_escalated": 10.0,
        "cost_review": 2.0,
        "max_review_rate": 0.05,
    },
    {
        "scenario": "friction_high",
        "cost_fraud_approved": 100.0,
        "cost_legit_escalated": 25.0,
        "cost_review": 2.0,
        "max_review_rate": 0.05,
    },
    {
        "scenario": "fraud_high",
        "cost_fraud_approved": 200.0,
        "cost_legit_escalated": 10.0,
        "cost_review": 2.0,
        "max_review_rate": 0.05,
    },
    {
        "scenario": "capacity_tight",
        "cost_fraud_approved": 100.0,
        "cost_legit_escalated": 10.0,
        "cost_review": 2.0,
        "max_review_rate": 0.03,
    },
    {
        "scenario": "capacity_loose",
        "cost_fraud_approved": 100.0,
        "cost_legit_escalated": 10.0,
        "cost_review": 2.0,
        "max_review_rate": 0.10,
    },
]


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
    return df


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
    c_cols = [c for c in df.columns if c.startswith("C") and c[1:].isdigit()]
    id_cols = [c for c in df.columns if c.startswith("id_")]

    df["missing_V_count"] = df[v_cols].isna().sum(axis=1).astype("int16") if v_cols else 0
    df["missing_D_count"] = df[d_cols].isna().sum(axis=1).astype("int8") if d_cols else 0
    df["missing_id_count"] = df[id_cols].isna().sum(axis=1).astype("int8") if id_cols else 0
    df["missing_core_count"] = df[["card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2"]].isna().sum(axis=1).astype("int8")

    # Normalized date anchors used only as non-target historical anchors. They do
    # not use the label and are derived from information available in the row.
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


def temporal_split(df, train_q=0.70, valid_q=0.85):
    q_train = df["TransactionDT"].quantile(train_q)
    q_valid = df["TransactionDT"].quantile(valid_q)
    train_idx = df.index[df["TransactionDT"] <= q_train]
    valid_idx = df.index[(df["TransactionDT"] > q_train) & (df["TransactionDT"] <= q_valid)]
    holdout_idx = df.index[df["TransactionDT"] > q_valid]
    return train_idx, valid_idx, holdout_idx, {"train_cutoff": float(q_train), "valid_cutoff": float(q_valid)}


def build_encoding_maps(train_df, cat_cols):
    maps = {}
    freqs = {}
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

    # Drop columns that are entirely missing in train.
    all_missing_train = X.loc[train_idx].isna().all()
    feature_cols = [c for c in feature_cols if not all_missing_train.get(c, False)]
    # V columns are numerous. For the first executable baseline, keep the most
    # target-correlated V columns measured only on the training window. Later
    # notebooks audit the full V block and compare against this lighter base.
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
    X = X[feature_cols]
    return X, feature_cols


def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype("int8")
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    return {
        "threshold": threshold,
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


def evaluate_segment(name, y_true, y_prob):
    if len(y_true) == 0 or y_true.nunique() < 2:
        empty_metrics = metrics_at_threshold(y_true, y_prob, 0.5) if len(y_true) else {}
        return {
            "segment": name,
            "rows": len(y_true),
            "fraud_rate": float(y_true.mean()) if len(y_true) else np.nan,
            "pr_auc": np.nan,
            "roc_auc": np.nan,
            **empty_metrics,
        }
    base = {
        "segment": name,
        "rows": int(len(y_true)),
        "fraud_rate": float(y_true.mean()),
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
    # For a first model, report two neutral thresholds: 0.5 and top 5% review.
    m05 = metrics_at_threshold(y_true, y_prob, 0.5)
    cutoff_top5 = np.quantile(y_prob, 0.95)
    mtop = metrics_at_threshold(y_true, y_prob, cutoff_top5)
    out = {}
    for k, v in m05.items():
        out[f"thr05_{k}"] = v
    for k, v in mtop.items():
        out[f"top5_{k}"] = v
    return {**base, **out}


def window_metrics(df_eval, y_prob, window_col="DT_week_index"):
    tmp = df_eval[["isFraud", window_col]].copy()
    tmp["prob"] = y_prob
    rows = []
    for window, g in tmp.groupby(window_col):
        if g["isFraud"].nunique() < 2:
            continue
        rows.append({
            "window": int(window),
            "rows": int(len(g)),
            "fraud_rate": float(g["isFraud"].mean()),
            "pr_auc": average_precision_score(g["isFraud"], g["prob"]),
            "roc_auc": roc_auc_score(g["isFraud"], g["prob"]),
        })
    return pd.DataFrame(rows)


def df_to_md(df):
    if df.empty:
        return "_Sin filas._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def apply_policy(df, review_threshold, escalate_threshold, score_col="pred"):
    out = df.copy()
    out["decision"] = "approve"
    out.loc[out[score_col] >= review_threshold, "decision"] = "review"
    out.loc[out[score_col] >= escalate_threshold, "decision"] = "escalate"
    return out


def evaluate_policy_decision(df, scenario):
    y = df["isFraud"].astype(int)
    decision = df["decision"]
    approved = decision == "approve"
    reviewed = decision == "review"
    escalated = decision == "escalate"
    fraud_approved = int(((y == 1) & approved).sum())
    legit_escalated = int(((y == 0) & escalated).sum())
    review_count = int(reviewed.sum())
    fraud_total = int((y == 1).sum())
    legit_total = int((y == 0).sum())
    caught_fraud = int(((y == 1) & (reviewed | escalated)).sum())
    cost = (
        fraud_approved * scenario["cost_fraud_approved"]
        + legit_escalated * scenario["cost_legit_escalated"]
        + review_count * scenario["cost_review"]
    )
    return {
        "rows": int(len(df)),
        "fraud_total": fraud_total,
        "fraud_rate": float(y.mean()),
        "cost_total": float(cost),
        "cost_per_txn": float(cost / len(df)),
        "fraud_detection_rate": float(caught_fraud / fraud_total) if fraud_total else np.nan,
        "fraud_miss_rate": float(fraud_approved / fraud_total) if fraud_total else np.nan,
        "legit_escalation_rate": float(legit_escalated / legit_total) if legit_total else np.nan,
        "review_rate": float(reviewed.mean()),
        "escalate_rate": float(escalated.mean()),
        "approve_rate": float(approved.mean()),
    }


def evaluate_thresholds(y, pred, review_threshold, escalate_threshold, scenario):
    approved = pred < review_threshold
    reviewed = (pred >= review_threshold) & (pred < escalate_threshold)
    escalated = pred >= escalate_threshold
    fraud = y == 1
    legit = y == 0

    fraud_approved = int((fraud & approved).sum())
    legit_escalated = int((legit & escalated).sum())
    review_count = int(reviewed.sum())
    fraud_total = int(fraud.sum())
    legit_total = int(legit.sum())
    caught_fraud = int((fraud & (reviewed | escalated)).sum())
    cost = (
        fraud_approved * scenario["cost_fraud_approved"]
        + legit_escalated * scenario["cost_legit_escalated"]
        + review_count * scenario["cost_review"]
    )
    return {
        "rows": int(len(y)),
        "fraud_total": fraud_total,
        "fraud_rate": float(fraud.mean()),
        "cost_total": float(cost),
        "cost_per_txn": float(cost / len(y)),
        "fraud_detection_rate": float(caught_fraud / fraud_total) if fraud_total else np.nan,
        "fraud_miss_rate": float(fraud_approved / fraud_total) if fraud_total else np.nan,
        "legit_escalation_rate": float(legit_escalated / legit_total) if legit_total else np.nan,
        "review_rate": float(reviewed.mean()),
        "escalate_rate": float(escalated.mean()),
        "approve_rate": float(approved.mean()),
    }


def robustness_threshold_grid(valid, max_review_rate):
    low = max(0.50, 1.0 - (max_review_rate * 4.0))
    qs_review = np.linspace(low, 0.995, 18)
    qs_escalate = np.linspace(max(low + 0.02, 0.85), 0.999, 18)
    review_thresholds = np.unique(np.quantile(valid["pred"], qs_review))
    escalate_thresholds = np.unique(np.quantile(valid["pred"], qs_escalate))
    for rt in review_thresholds:
        for et in escalate_thresholds:
            if et <= rt:
                continue
            yield float(rt), float(et)


def optimize_global(valid, scenario):
    y = valid["isFraud"].astype(int).to_numpy()
    pred = valid["pred"].to_numpy()
    rows = []
    for rt, et in robustness_threshold_grid(valid, scenario["max_review_rate"]):
        metrics = evaluate_thresholds(y, pred, rt, et, scenario)
        if metrics["review_rate"] <= scenario["max_review_rate"]:
            rows.append({"review_threshold": rt, "escalate_threshold": et, **metrics})
    grid = pd.DataFrame(rows)
    return grid.sort_values(["cost_per_txn", "fraud_miss_rate", "legit_escalation_rate"], ascending=True)


def optimize_segment(valid, scenario):
    known = valid[valid["uid_known"].astype(bool)]
    unknown = valid[~valid["uid_known"].astype(bool)]
    known_grid = optimize_global(known, scenario)
    unknown_grid = optimize_global(unknown, scenario)
    if known_grid.empty or unknown_grid.empty:
        return pd.DataFrame()

    known_best = known_grid.iloc[0]
    unknown_best = unknown_grid.iloc[0]
    scored = pd.concat([
        apply_policy(known, known_best["review_threshold"], known_best["escalate_threshold"]),
        apply_policy(unknown, unknown_best["review_threshold"], unknown_best["escalate_threshold"]),
    ], axis=0)
    metrics = evaluate_policy_decision(scored, scenario)
    return pd.DataFrame([{
        "known_review_threshold": float(known_best["review_threshold"]),
        "known_escalate_threshold": float(known_best["escalate_threshold"]),
        "unknown_review_threshold": float(unknown_best["review_threshold"]),
        "unknown_escalate_threshold": float(unknown_best["escalate_threshold"]),
        **metrics,
    }])


def apply_segment_policy(df, row):
    known = df[df["uid_known"].astype(bool)]
    unknown = df[~df["uid_known"].astype(bool)]
    scored_known = apply_policy(known, row["known_review_threshold"], row["known_escalate_threshold"])
    scored_unknown = apply_policy(unknown, row["unknown_review_threshold"], row["unknown_escalate_threshold"])
    return pd.concat([scored_known, scored_unknown], axis=0).sort_index()


def calibration_table(valid, holdout):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(valid["pred"], valid["isFraud"])
    valid = valid.copy()
    holdout = holdout.copy()
    valid["pred_iso"] = iso.predict(valid["pred"])
    holdout["pred_iso"] = iso.predict(holdout["pred"])

    rows = []
    for split, df in [("valid", valid), ("holdout", holdout)]:
        for score_col in ["pred", "pred_iso"]:
            rows.append({
                "split": split,
                "score": score_col,
                "brier": brier_score_loss(df["isFraud"], df[score_col]),
                "mean_score": float(df[score_col].mean()),
                "fraud_rate": float(df["isFraud"].mean()),
            })

    bin_rows = []
    for split, df in [("valid", valid), ("holdout", holdout)]:
        tmp = df.copy()
        tmp["bin"] = pd.qcut(tmp["pred"], q=10, duplicates="drop")
        grouped = tmp.groupby("bin", observed=True).agg(
            rows=("isFraud", "size"),
            mean_pred=("pred", "mean"),
            mean_pred_iso=("pred_iso", "mean"),
            fraud_rate=("isFraud", "mean"),
        )
        grouped.insert(0, "split", split)
        bin_rows.append(grouped.reset_index(drop=True))
    return pd.DataFrame(rows), pd.concat(bin_rows, ignore_index=True)


def run_policy_robustness(valid_pred, holdout_pred):
    policy_rows = []
    segment_rows = []

    for scenario in SCENARIOS:
        global_grid = optimize_global(valid_pred, scenario)
        if global_grid.empty:
            raise RuntimeError(f"No global policy found for scenario {scenario['scenario']}")
        global_best = global_grid.iloc[0]
        for split, df in [("valid", valid_pred), ("holdout", holdout_pred)]:
            scored = apply_policy(df, global_best["review_threshold"], global_best["escalate_threshold"])
            policy_rows.append({
                "scenario": scenario["scenario"],
                "policy_type": "global",
                "split": split,
                "review_threshold": float(global_best["review_threshold"]),
                "escalate_threshold": float(global_best["escalate_threshold"]),
                **evaluate_policy_decision(scored, scenario),
            })

        segment_grid = optimize_segment(valid_pred, scenario)
        if not segment_grid.empty:
            segment_best = segment_grid.iloc[0]
            for split, df in [("valid", valid_pred), ("holdout", holdout_pred)]:
                scored = apply_segment_policy(df, segment_best)
                segment_rows.append({
                    "scenario": scenario["scenario"],
                    "policy_type": "segmented_uid",
                    "split": split,
                    "known_review_threshold": float(segment_best["known_review_threshold"]),
                    "known_escalate_threshold": float(segment_best["known_escalate_threshold"]),
                    "unknown_review_threshold": float(segment_best["unknown_review_threshold"]),
                    "unknown_escalate_threshold": float(segment_best["unknown_escalate_threshold"]),
                    **evaluate_policy_decision(scored, scenario),
                })

    policy_results = pd.DataFrame(policy_rows)
    segment_results = pd.DataFrame(segment_rows)
    combined = pd.concat([policy_results, segment_results], ignore_index=True, sort=False)
    calibration_summary, calibration_bins = calibration_table(valid_pred, holdout_pred)

    policy_results.to_csv(OUT / "global_policy_sensitivity.csv", index=False)
    segment_results.to_csv(OUT / "segmented_uid_policy_sensitivity.csv", index=False)
    combined.to_csv(OUT / "policy_robustness_combined.csv", index=False)
    calibration_summary.to_csv(OUT / "calibration_summary.csv", index=False)
    calibration_bins.to_csv(OUT / "calibration_bins.csv", index=False)

    base_compare = combined[(combined["scenario"] == "base") & (combined["split"] == "holdout")].copy()
    base_compare = base_compare[[
        "policy_type",
        "cost_per_txn",
        "fraud_detection_rate",
        "fraud_miss_rate",
        "review_rate",
        "escalate_rate",
        "approve_rate",
    ]]

    holdout_global = policy_results[policy_results["split"] == "holdout"][[
        "scenario",
        "cost_per_txn",
        "fraud_detection_rate",
        "review_rate",
        "escalate_rate",
        "approve_rate",
    ]]

    report = []
    report.append("# V06 robustez de politica\n\n")
    report.append("## Comparacion base en holdout\n\n")
    report.append(df_to_md(base_compare))
    report.append("\n\n## Sensibilidad de escenarios globales en holdout\n\n")
    report.append(df_to_md(holdout_global))
    report.append("\n\n## Calibracion\n\n")
    report.append(df_to_md(calibration_summary))
    report.append("\n")
    (OUT / "v06_policy_robustness_report.md").write_text("".join(report), encoding="utf-8")

    summary = {
        "base_holdout": base_compare.to_dict(orient="records"),
        "calibration": calibration_summary.to_dict(orient="records"),
    }
    (OUT / "v06_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_outputs(valid_df, valid_prob, holdout_df, holdout_prob, importance):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(valid_prob, bins=80, alpha=0.7, label="valid")
    ax.hist(holdout_prob, bins=80, alpha=0.7, label="holdout")
    ax.set_title("Distribucion de scores")
    ax.set_xlabel("probabilidad estimada")
    ax.set_ylabel("filas")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "score_distribution.png", dpi=140)
    plt.close()

    top = importance.head(30).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top["feature"], top["importance"])
    ax.set_title("Top 30 feature importance - baseline")
    plt.tight_layout()
    plt.savefig(PLOTS / "feature_importance_top30.png", dpi=140)
    plt.close()


def save_report(summary, metrics, feature_cols):
    lines = []
    lines.append("# V01 baseline temporal - IEEE-CIS Fraud Detection\n\n")
    lines.append("## Objetivo\n\n")
    lines.append("Construir un primer baseline temporal reproducible, sin `TransactionDT` crudo como feature y con metricas globales, por ventana y por UID conocido/desconocido.\n\n")
    lines.append("## Diseno\n\n")
    lines.append(f"- Train temporal: primeras 70% filas por `TransactionDT`.\n")
    lines.append(f"- Validacion: siguiente 15%.\n")
    lines.append(f"- Holdout futuro: ultimo 15%.\n")
    lines.append(f"- Features usadas: {len(feature_cols)}.\n")
    lines.append("- Modelo: LightGBM binario con early stopping.\n\n")
    lines.append("Nota: `TransactionDT`, `DT_day_index` y `DT_week_index` no se usan como predictores; se reservan para split, ventanas y monitoreo. El unico derivado temporal usado como candidato es `DT_hour`.\n\n")
    lines.append("## Resultados principales\n\n")
    for _, row in metrics.iterrows():
        if row["segment"] in ["valid_global", "holdout_global"]:
            lines.append(f"- {row['segment']}: PR-AUC={row['pr_auc']:.5f}, ROC-AUC={row['roc_auc']:.5f}, fraud_rate={row['fraud_rate']:.4f}, rows={int(row['rows'])}.\n")
    lines.append("\n## Lectura metodologica\n\n")
    lines.append("Este baseline no busca ser el modelo final. Su funcion es establecer una referencia temporal honesta para medir si las siguientes familias de features realmente aportan. Las transformaciones de monto, flags de faltantes y encodings se mantienen como candidatos verificables; su permanencia dependera de ablations posteriores.\n\n")
    lines.append("## Archivos generados\n\n")
    for name in [
        "baseline_summary.json",
        "baseline_metrics.csv",
        "baseline_window_metrics.csv",
        "baseline_feature_importance.csv",
        "baseline_valid_predictions.csv",
        "baseline_holdout_predictions.csv",
    ]:
        lines.append(f"- `{name}`\n")
    (OUT / "baseline_report.md").write_text("".join(lines), encoding="utf-8")


def main():
    print("Loading data...")
    df = load_train()
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    print("Shape after merge:", df.shape)

    train_idx, valid_idx, holdout_idx, cutoffs = temporal_split(df)
    print("Temporal split sizes:", len(train_idx), len(valid_idx), len(holdout_idx))

    df, train_stats = add_base_features(df.loc[:, :], train_stats=None)

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

    print("Training LightGBM baseline...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(50)],
    )

    valid_prob = model.predict_proba(X_valid)[:, 1]
    holdout_prob = model.predict_proba(X_holdout)[:, 1]

    metrics = []
    metrics.append(evaluate_segment("valid_global", y_valid, valid_prob))
    metrics.append(evaluate_segment("valid_uid_known", y_valid.loc[known_valid.values], valid_prob[known_valid.values]))
    metrics.append(evaluate_segment("valid_uid_unknown", y_valid.loc[~known_valid.values], valid_prob[~known_valid.values]))
    metrics.append(evaluate_segment("holdout_global", y_holdout, holdout_prob))
    metrics.append(evaluate_segment("holdout_uid_known", y_holdout.loc[known_holdout.values], holdout_prob[known_holdout.values]))
    metrics.append(evaluate_segment("holdout_uid_unknown", y_holdout.loc[~known_holdout.values], holdout_prob[~known_holdout.values]))
    metrics = pd.DataFrame(metrics)

    valid_windows = window_metrics(df.loc[valid_idx], valid_prob)
    valid_windows.insert(0, "split", "valid")
    holdout_windows = window_metrics(df.loc[holdout_idx], holdout_prob)
    holdout_windows.insert(0, "split", "holdout")
    window_out = pd.concat([valid_windows, holdout_windows], ignore_index=True)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    valid_pred = pd.DataFrame({
        "TransactionID": df.loc[valid_idx, "TransactionID"].to_numpy(),
        "TransactionDT": df.loc[valid_idx, "TransactionDT"].to_numpy(),
        "isFraud": y_valid.to_numpy(),
        "pred": valid_prob,
        "uid_known": known_valid.to_numpy(),
    })
    holdout_pred = pd.DataFrame({
        "TransactionID": df.loc[holdout_idx, "TransactionID"].to_numpy(),
        "TransactionDT": df.loc[holdout_idx, "TransactionDT"].to_numpy(),
        "isFraud": y_holdout.to_numpy(),
        "pred": holdout_prob,
        "uid_known": known_holdout.to_numpy(),
    })

    summary = {
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
        "holdout_rows": int(len(holdout_idx)),
        "features": int(len(feature_cols)),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "train_fraud_rate": float(y_train.mean()),
        "valid_fraud_rate": float(y_valid.mean()),
        "holdout_fraud_rate": float(y_holdout.mean()),
        "valid_uid_known_pct": float(known_valid.mean()),
        "holdout_uid_known_pct": float(known_holdout.mean()),
        **cutoffs,
    }

    metrics.to_csv(OUT / "baseline_metrics.csv", index=False)
    window_out.to_csv(OUT / "baseline_window_metrics.csv", index=False)
    importance.to_csv(OUT / "baseline_feature_importance.csv", index=False)
    valid_pred.to_csv(OUT / "baseline_valid_predictions.csv", index=False)
    holdout_pred.to_csv(OUT / "baseline_holdout_predictions.csv", index=False)
    (OUT / "baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_outputs(df.loc[valid_idx], valid_prob, df.loc[holdout_idx], holdout_prob, importance)
    save_report(summary, metrics, feature_cols)
    robustness_summary = run_policy_robustness(valid_pred, holdout_pred)

    print("Summary:")
    print(json.dumps(summary, indent=2))
    print("Metrics:")
    print(metrics[["segment", "rows", "fraud_rate", "pr_auc", "roc_auc"]].to_string(index=False))
    print("Robustness summary:")
    print(json.dumps(robustness_summary, indent=2))
    print("V06 policy robustness complete.")


if __name__ == "__main__":
    main()
