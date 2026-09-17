import gc
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

DATA_DIR_CANDIDATES = [
    Path("/kaggle/input/ieee-fraud-detection"),
    Path("/kaggle/input/competitions/ieee-fraud-detection"),
]
DATA_DIR = next((path for path in DATA_DIR_CANDIDATES if path.exists()), DATA_DIR_CANDIDATES[0])
OUT = Path("/kaggle/working")
PLOTS = OUT / "plots"
PLOTS.mkdir(exist_ok=True)


def save_csv(df, name):
    path = OUT / name
    df.to_csv(path, index=False)
    return path


def save_plot(name):
    path = PLOTS / name
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    return path


def reduce_memory(df):
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif df[col].dtype == "object":
            nunique = df[col].nunique(dropna=True)
            if nunique < max(1000, 0.5 * len(df)):
                df[col] = df[col].astype("category")
    return df


def load_data():
    train_txn = reduce_memory(pd.read_csv(DATA_DIR / "train_transaction.csv"))
    test_txn = reduce_memory(pd.read_csv(DATA_DIR / "test_transaction.csv"))
    train_id = reduce_memory(pd.read_csv(DATA_DIR / "train_identity.csv"))
    test_id = reduce_memory(pd.read_csv(DATA_DIR / "test_identity.csv"))

    train = train_txn.merge(train_id, on="TransactionID", how="left")
    test = test_txn.merge(test_id, on="TransactionID", how="left")
    del train_txn, test_txn, train_id, test_id
    gc.collect()

    train["dataset"] = "train"
    test["dataset"] = "test"
    test["isFraud"] = np.nan
    return train, test


def add_time_features(df):
    df = df.copy()
    df["DT_day"] = (df["TransactionDT"] // (24 * 3600)).astype("int16")
    df["DT_week"] = (df["DT_day"] // 7).astype("int16")
    df["DT_hour"] = ((df["TransactionDT"] // 3600) % 24).astype("int8")
    df["DT_month_proxy"] = (df["DT_day"] // 30).astype("int8")
    return df


def candidate_uid(df):
    day = (df["TransactionDT"] // (24 * 3600)).astype("float32")
    parts = pd.DataFrame(index=df.index)
    for col in ["card1", "card2", "card3", "card5", "addr1", "addr2"]:
        parts[col] = df[col].astype("string").fillna("NA")
    parts["D1_anchor"] = (df["D1"].astype("float32") - day).round(0).astype("string").fillna("NA")
    return pd.util.hash_pandas_object(parts, index=False).astype("uint64")


def structural_summary(train, test):
    rows = []
    for name, df in [("train", train), ("test", test)]:
        identity_cols = [c for c in df.columns if c.startswith("id_") or c in ["DeviceType", "DeviceInfo"]]
        rows.append({
            "dataset": name,
            "rows": len(df),
            "columns": df.shape[1],
            "transaction_id_unique": bool(df["TransactionID"].is_unique),
            "time_days": float((df["TransactionDT"].max() - df["TransactionDT"].min()) / (3600 * 24)),
            "identity_coverage_pct": float(100 * df[identity_cols[0]].notna().mean()) if identity_cols else np.nan,
            "fraud_rate_pct": float(100 * df["isFraud"].mean()) if name == "train" else np.nan,
        })
    return pd.DataFrame(rows)


def missing_summary(train, test):
    rows = []
    for col in train.columns:
        if col in ["dataset", "isFraud"]:
            continue
        rows.append({
            "feature": col,
            "train_missing_pct": 100 * train[col].isna().mean(),
            "test_missing_pct": 100 * test[col].isna().mean() if col in test.columns else np.nan,
            "missing_delta_abs": abs(100 * train[col].isna().mean() - 100 * test[col].isna().mean()) if col in test.columns else np.nan,
            "train_nunique": train[col].nunique(dropna=True),
            "test_nunique": test[col].nunique(dropna=True) if col in test.columns else np.nan,
            "dtype": str(train[col].dtype),
        })
    return pd.DataFrame(rows).sort_values(["missing_delta_abs", "train_missing_pct"], ascending=False)


def target_correlations(train):
    num_cols = train.select_dtypes(include=[np.number]).columns.difference(["isFraud", "TransactionID"])
    corrs = train[num_cols].corrwith(train["isFraud"]).dropna()
    out = pd.DataFrame({
        "feature": corrs.index,
        "corr_isFraud": corrs.values,
        "abs_corr": corrs.abs().values,
    }).sort_values("abs_corr", ascending=False)
    return out


def temporal_summary(train):
    rows = []
    train = add_time_features(train)
    for key in ["DT_week", "DT_month_proxy"]:
        g = train.groupby(key, observed=True).agg(
            rows=("TransactionID", "size"),
            fraud_rate=("isFraud", "mean"),
            pr_base_rate=("isFraud", "mean"),
            amt_mean=("TransactionAmt", "mean"),
            amt_median=("TransactionAmt", "median"),
            identity_coverage=("id_01", lambda s: s.notna().mean()) if "id_01" in train.columns else ("TransactionID", "size"),
        ).reset_index().rename(columns={key: "window"})
        g.insert(0, "window_type", key)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def category_risk(train):
    cat_cols = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType"]
    cat_cols += [f"M{i}" for i in range(1, 10) if f"M{i}" in train.columns]
    rows = []
    for col in cat_cols:
        if col not in train.columns:
            continue
        tmp = train.groupby(col, observed=True)["isFraud"].agg(["size", "mean"]).reset_index()
        tmp.columns = ["value", "rows", "fraud_rate"]
        tmp["feature"] = col
        tmp["global_fraud_rate"] = train["isFraud"].mean()
        tmp["lift"] = tmp["fraud_rate"] / tmp["global_fraud_rate"]
        rows.append(tmp.sort_values("rows", ascending=False).head(25))
    return pd.concat(rows, ignore_index=True).sort_values(["lift", "rows"], ascending=[False, False])


def numeric_distribution_shift(train, test, sample_n=250000):
    numeric = train.select_dtypes(include=[np.number]).columns.intersection(test.select_dtypes(include=[np.number]).columns)
    numeric = [c for c in numeric if c not in ["isFraud", "TransactionID"]]
    rng = np.random.default_rng(42)
    rows = []
    for col in numeric:
        a = train[col].dropna()
        b = test[col].dropna()
        if len(a) < 100 or len(b) < 100:
            continue
        if len(a) > sample_n:
            a = a.iloc[rng.choice(len(a), sample_n, replace=False)]
        if len(b) > sample_n:
            b = b.iloc[rng.choice(len(b), sample_n, replace=False)]
        try:
            ks = stats.ks_2samp(a.astype("float64"), b.astype("float64")).statistic
        except Exception:
            ks = np.nan
        rows.append({
            "feature": col,
            "ks_train_test": ks,
            "train_mean": float(a.mean()),
            "test_mean": float(b.mean()),
            "mean_delta": float(b.mean() - a.mean()),
            "train_missing_pct": 100 * train[col].isna().mean(),
            "test_missing_pct": 100 * test[col].isna().mean(),
        })
    return pd.DataFrame(rows).sort_values("ks_train_test", ascending=False)


def uid_overlap_analysis(train):
    train = train.copy()
    train["candidate_uid"] = candidate_uid(train)
    cutoff = train["TransactionDT"].quantile(0.80)
    train["split"] = np.where(train["TransactionDT"] <= cutoff, "train_early", "valid_late")
    uid_sides = train.groupby("candidate_uid")["split"].nunique()
    crossing_uids = set(uid_sides[uid_sides > 1].index)
    train["uid_crosses_temporal_cut"] = train["candidate_uid"].isin(crossing_uids)

    late = train[train["split"] == "valid_late"]
    known_late = late["candidate_uid"].isin(set(train.loc[train["split"] == "train_early", "candidate_uid"]))
    rows = [{
        "cutoff_quantile": 0.80,
        "cutoff_transactiondt": float(cutoff),
        "candidate_uid_count": int(train["candidate_uid"].nunique()),
        "crossing_uid_count": int(len(crossing_uids)),
        "all_rows_in_crossing_uid_pct": float(100 * train["uid_crosses_temporal_cut"].mean()),
        "late_rows_known_uid_pct": float(100 * known_late.mean()),
        "late_known_uid_fraud_rate": float(late.loc[known_late, "isFraud"].mean()),
        "late_unknown_uid_fraud_rate": float(late.loc[~known_late, "isFraud"].mean()),
        "late_known_rows": int(known_late.sum()),
        "late_unknown_rows": int((~known_late).sum()),
    }]
    return pd.DataFrame(rows)


def amount_analysis(train):
    amt = train["TransactionAmt"].astype("float64")
    frac = (amt * 100).round(4) % 1
    multi_dec = frac.abs() > 1e-6
    q1, q3 = amt.quantile([0.25, 0.75])
    iqr = q3 - q1
    out_iqr = (amt < q1 - 1.5 * iqr) | (amt > q3 + 1.5 * iqr)
    rows = [{
        "transaction_amt_skew": float(stats.skew(amt.dropna())),
        "multi_decimal_pct": float(100 * multi_dec.mean()),
        "multi_decimal_fraud_rate": float(train.loc[multi_dec, "isFraud"].mean()),
        "non_multi_decimal_fraud_rate": float(train.loc[~multi_dec, "isFraud"].mean()),
        "iqr_outlier_pct": float(100 * out_iqr.mean()),
        "iqr_outlier_fraud_rate": float(train.loc[out_iqr, "isFraud"].mean()),
        "non_outlier_fraud_rate": float(train.loc[~out_iqr, "isFraud"].mean()),
        "amt_p50": float(amt.quantile(0.50)),
        "amt_p95": float(amt.quantile(0.95)),
        "amt_p99": float(amt.quantile(0.99)),
    }]
    return pd.DataFrame(rows)


def redundancy_summary(train):
    rows = []
    for prefix in ["C", "D", "V"]:
        cols = [c for c in train.columns if c.startswith(prefix) and c[1:].isdigit()]
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(train[c])]
        if len(cols) < 2:
            continue
        sample = train[cols].sample(min(len(train), 120000), random_state=42)
        corr = sample.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = upper.stack().reset_index()
        pairs.columns = ["feature_a", "feature_b", "abs_corr"]
        top = pairs.sort_values("abs_corr", ascending=False).head(50)
        top.insert(0, "group", prefix)
        rows.append(top)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def adversarial_validation(train, test):
    feature_candidates = [
        "TransactionAmt", "DT_day", "DT_hour", "DT_month_proxy",
        "card1", "card2", "card3", "card5", "addr1", "addr2",
        "dist1", "dist2", "C1", "C2", "C5", "C13", "D1", "D2", "D10", "D15",
    ]
    feature_candidates += [f"V{i}" for i in [1, 3, 4, 12, 34, 53, 62, 70, 95, 127, 170, 201, 258, 307]]
    feature_candidates = [c for c in feature_candidates if c in train.columns and c in test.columns]

    base_cols = list(dict.fromkeys(feature_candidates + ["TransactionID", "TransactionDT"]))
    tr = add_time_features(train[base_cols].copy())
    te = add_time_features(test[base_cols].copy())
    tr["target_is_test"] = 0
    te["target_is_test"] = 1
    both = pd.concat([tr, te], ignore_index=True)
    y = both.pop("target_is_test").astype("int8")
    both = both.drop(columns=["TransactionID"], errors="ignore")

    for col in both.columns:
        if not pd.api.types.is_numeric_dtype(both[col]):
            codes, _ = pd.factorize(both[col].astype("string"), sort=True)
            both[col] = codes
        both[col] = both[col].astype("float32")
        both[col] = both[col].fillna(-999)

    if len(both) > 300000:
        idx = both.sample(300000, random_state=42).index
        X = both.loc[idx]
        yy = y.loc[idx]
    else:
        X = both
        yy = y

    X_train, X_val, y_train, y_val = train_test_split(X, yy, test_size=0.30, random_state=42, stratify=yy)
    model = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.08, max_leaf_nodes=31, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pred)
    ap = average_precision_score(y_val, pred)

    # Cheap permutation importance on validation.
    base = auc
    rng = np.random.default_rng(42)
    imps = []
    for col in X_val.columns:
        Xp = X_val.copy()
        Xp[col] = rng.permutation(Xp[col].to_numpy())
        auc_p = roc_auc_score(y_val, model.predict_proba(Xp)[:, 1])
        imps.append({"feature": col, "adversarial_auc_drop": base - auc_p})
    imp = pd.DataFrame(imps).sort_values("adversarial_auc_drop", ascending=False)
    metrics = pd.DataFrame([{"adversarial_roc_auc": auc, "adversarial_pr_auc": ap, "features_used": len(feature_candidates)}])
    return metrics, imp


def plot_key_figures(train, test, temporal, missing, shift, corr, category):
    fig, ax = plt.subplots(figsize=(6, 4))
    train["isFraud"].value_counts(normalize=True).sort_index().mul(100).plot(kind="bar", ax=ax, color=["#55A868", "#C44E52"])
    ax.set_title("Distribucion de isFraud")
    ax.set_xlabel("isFraud")
    ax.set_ylabel("%")
    save_plot("01_target_balance.png")

    weeks = temporal[temporal["window_type"] == "DT_week"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(weeks["window"], 100 * weeks["fraud_rate"], marker="o", color="#C44E52")
    ax.set_title("Tasa de fraude por semana proxy")
    ax.set_xlabel("semana desde inicio")
    ax.set_ylabel("% fraude")
    save_plot("02_fraud_rate_by_week.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    missing.head(30).sort_values("train_missing_pct").plot.barh(x="feature", y=["train_missing_pct", "test_missing_pct"], ax=ax)
    ax.set_title("Top faltantes train/test")
    ax.set_xlabel("% faltante")
    save_plot("03_missing_train_test.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    shift.head(30).sort_values("ks_train_test").plot.barh(x="feature", y="ks_train_test", ax=ax, color="#8172B2")
    ax.set_title("Mayor distribution shift numerico train/test (KS)")
    ax.set_xlabel("KS statistic")
    save_plot("04_numeric_shift_ks.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    corr.head(25).sort_values("abs_corr").plot.barh(x="feature", y="corr_isFraud", ax=ax, color="#4C72B0")
    ax.set_title("Variables numericas mas correlacionadas con fraude")
    ax.set_xlabel("correlacion con isFraud")
    save_plot("05_target_correlations.png")

    prod = category[category["feature"] == "ProductCD"].sort_values("fraud_rate", ascending=False)
    if len(prod):
        fig, ax = plt.subplots(figsize=(6, 4))
        prod.plot.bar(x="value", y="fraud_rate", ax=ax, color="#DD8452")
        ax.set_title("Tasa de fraude por ProductCD")
        ax.set_ylabel("fraud_rate")
        save_plot("06_productcd_risk.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    train["TransactionAmt"].clip(upper=train["TransactionAmt"].quantile(0.99)).hist(bins=80, ax=axes[0], color="#4C72B0")
    axes[0].set_title("TransactionAmt cap p99")
    np.log1p(train["TransactionAmt"]).hist(bins=80, ax=axes[1], color="#55A868")
    axes[1].set_title("log1p(TransactionAmt)")
    save_plot("07_amount_distribution.png")


def make_decision_log(structural, missing, corr, temporal, uid, amount, shift, adv_metrics):
    fraud_rate = structural.loc[structural["dataset"] == "train", "fraud_rate_pct"].iloc[0]
    max_corr = corr["abs_corr"].max()
    weekly = temporal[temporal["window_type"] == "DT_week"]
    min_week = 100 * weekly["fraud_rate"].min()
    max_week = 100 * weekly["fraud_rate"].max()
    known_late = uid["late_rows_known_uid_pct"].iloc[0]
    adv_auc = adv_metrics["adversarial_roc_auc"].iloc[0]
    high_missing = int((missing["train_missing_pct"] > 80).sum())
    high_shift = int((shift["ks_train_test"] > 0.20).sum())

    rows = [
        {
            "decision": "Usar PR-AUC como metrica principal",
            "evidence": f"Fraude en train = {fraud_rate:.2f}%; accuracy seria enganosa.",
            "modeling_impact": "Reportar PR-AUC, recall, precision, F1 y FPR por ventana temporal.",
        },
        {
            "decision": "No usar TransactionDT crudo como feature",
            "evidence": "Es la variable que define el orden temporal; usarla cruda puede aprender posicion historica.",
            "modeling_impact": "Derivar hora/dia/semana y usar TransactionDT para split y monitoreo.",
        },
        {
            "decision": "Separar validacion temporal en UID conocido y UID desconocido",
            "evidence": f"{known_late:.2f}% de filas tardias comparten candidate_uid con entrenamiento temprano.",
            "modeling_impact": "Reportar desempeno por segmento y evitar conclusiones infladas.",
        },
        {
            "decision": "Crear features UID, pero sin usar informacion futura",
            "evidence": "card/D/addr forman clientes candidatos repetidos; top soluciones IEEE explotan esa senal.",
            "modeling_impact": "Frequency/agregaciones calculadas solo con ventana de entrenamiento.",
        },
        {
            "decision": "Conservar outliers de TransactionAmt",
            "evidence": f"Outliers IQR tienen tasa de fraude {100 * amount['iqr_outlier_fraud_rate'].iloc[0]:.2f}%.",
            "modeling_impact": "Usar log1p, flags y winsorizar solo para modelos sensibles, no eliminar.",
        },
        {
            "decision": "Tratar faltantes por bloque",
            "evidence": f"{high_missing} features tienen mas de 80% faltante en train.",
            "modeling_impact": "Agregar indicadores de ausencia, has_identity y missing-block features.",
        },
        {
            "decision": "Auditar distribution shift antes de seleccionar features",
            "evidence": f"{high_shift} features numericas tienen KS train/test > 0.20; adversarial AUC = {adv_auc:.3f}.",
            "modeling_impact": "Penalizar features inestables y medir robustez temporal.",
        },
        {
            "decision": "No descartar columnas por correlacion con target solamente",
            "evidence": f"Max abs corr con isFraud = {max_corr:.3f}; no hay copia directa obvia del target.",
            "modeling_impact": "Combinar correlacion, estabilidad temporal y permutation importance.",
        },
        {
            "decision": "Modelar drift explicitamente",
            "evidence": f"La tasa semanal de fraude varia entre {min_week:.2f}% y {max_week:.2f}%.",
            "modeling_impact": "Ventanas temporales, PSI/KS y comparacion static vs adaptive.",
        },
    ]
    return pd.DataFrame(rows)


def write_report(tables):
    structural = tables["structural"]
    missing = tables["missing"]
    corr = tables["corr"]
    temporal = tables["temporal"]
    uid = tables["uid"]
    amount = tables["amount"]
    shift = tables["shift"]
    adv_metrics = tables["adv_metrics"]
    decisions = tables["decisions"]

    fraud_rate = structural.loc[structural["dataset"] == "train", "fraud_rate_pct"].iloc[0]
    train_rows = int(structural.loc[structural["dataset"] == "train", "rows"].iloc[0])
    test_rows = int(structural.loc[structural["dataset"] == "test", "rows"].iloc[0])
    weekly = temporal[temporal["window_type"] == "DT_week"]

    lines = []
    lines.append("# EDA propio - IEEE-CIS Fraud Detection\n")
    lines.append("## Resumen ejecutivo\n")
    lines.append(f"- Train tiene {train_rows:,} filas y test tiene {test_rows:,} filas.\n")
    lines.append(f"- Tasa de fraude en train: {fraud_rate:.2f}%.\n")
    lines.append(f"- La tasa semanal de fraude varia entre {100 * weekly['fraud_rate'].min():.2f}% y {100 * weekly['fraud_rate'].max():.2f}%.\n")
    lines.append(f"- Candidate UID conocido en validacion tardia: {uid['late_rows_known_uid_pct'].iloc[0]:.2f}% de filas.\n")
    lines.append(f"- Adversarial validation train/test ROC-AUC: {adv_metrics['adversarial_roc_auc'].iloc[0]:.3f}.\n")
    lines.append(f"- Maxima correlacion absoluta numerica con isFraud: {corr['abs_corr'].max():.3f}.\n")
    lines.append("\n## Decisiones recomendadas\n")
    for _, row in decisions.iterrows():
        lines.append(f"- **{row['decision']}**: {row['evidence']} Impacto: {row['modeling_impact']}\n")
    lines.append("\n## Tablas generadas\n")
    for name in [
        "structural_summary.csv", "missing_summary.csv", "target_correlations.csv",
        "temporal_summary.csv", "category_risk.csv", "numeric_shift_ks.csv",
        "uid_overlap_analysis.csv", "amount_analysis.csv", "redundancy_pairs.csv",
        "adversarial_metrics.csv", "adversarial_importance.csv", "decision_log.csv",
    ]:
        lines.append(f"- `{name}`\n")
    lines.append("\n## Graficos generados\n")
    for path in sorted(PLOTS.glob("*.png")):
        lines.append(f"- `plots/{path.name}`\n")
    (OUT / "eda_report.md").write_text("".join(lines), encoding="utf-8")


def main():
    print("Loading IEEE-CIS data...")
    train, test = load_data()
    train = add_time_features(train)
    test = add_time_features(test)
    print("Shapes:", train.shape, test.shape)

    print("Computing structural summary...")
    structural = structural_summary(train, test)
    missing = missing_summary(train, test)
    corr = target_correlations(train)
    temporal = temporal_summary(train)
    category = category_risk(train)
    shift = numeric_distribution_shift(train, test)
    uid = uid_overlap_analysis(train)
    amount = amount_analysis(train)
    redundancy = redundancy_summary(train)
    adv_metrics, adv_importance = adversarial_validation(train, test)
    decisions = make_decision_log(structural, missing, corr, temporal, uid, amount, shift, adv_metrics)

    print("Saving outputs...")
    save_csv(structural, "structural_summary.csv")
    save_csv(missing, "missing_summary.csv")
    save_csv(corr, "target_correlations.csv")
    save_csv(temporal, "temporal_summary.csv")
    save_csv(category, "category_risk.csv")
    save_csv(shift, "numeric_shift_ks.csv")
    save_csv(uid, "uid_overlap_analysis.csv")
    save_csv(amount, "amount_analysis.csv")
    save_csv(redundancy, "redundancy_pairs.csv")
    save_csv(adv_metrics, "adversarial_metrics.csv")
    save_csv(adv_importance, "adversarial_importance.csv")
    save_csv(decisions, "decision_log.csv")

    plot_key_figures(train, test, temporal, missing, shift, corr, category)
    tables = {
        "structural": structural,
        "missing": missing,
        "corr": corr,
        "temporal": temporal,
        "category": category,
        "shift": shift,
        "uid": uid,
        "amount": amount,
        "redundancy": redundancy,
        "adv_metrics": adv_metrics,
        "adv_importance": adv_importance,
        "decisions": decisions,
    }
    write_report(tables)

    summary = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_fraud_rate_pct": float(100 * train["isFraud"].mean()),
        "weekly_fraud_rate_min_pct": float(100 * temporal.loc[temporal["window_type"] == "DT_week", "fraud_rate"].min()),
        "weekly_fraud_rate_max_pct": float(100 * temporal.loc[temporal["window_type"] == "DT_week", "fraud_rate"].max()),
        "late_rows_known_uid_pct": float(uid["late_rows_known_uid_pct"].iloc[0]),
        "adversarial_roc_auc": float(adv_metrics["adversarial_roc_auc"].iloc[0]),
        "max_abs_corr_isFraud": float(corr["abs_corr"].max()),
        "high_missing_features_gt80pct": int((missing["train_missing_pct"] > 80).sum()),
        "high_numeric_shift_ks_gt020": int((shift["ks_train_test"] > 0.20).sum()),
    }
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("EDA complete.")


if __name__ == "__main__":
    main()
