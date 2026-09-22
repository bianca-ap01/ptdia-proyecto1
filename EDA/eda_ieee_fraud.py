# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # EDA — IEEE-CIS Fraud Detection
#
# Análisis exploratorio completo del dataset de la competición
# [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data).
#
# ## Descripción del dataset
#
# El dataset proviene de transacciones de e-commerce reales proporcionadas por Vesta Corporation.
# Se divide en dos tablas unidas por `TransactionID`:
#
# **Transaction table** (~590K train, ~506K test):
# - `TransactionDT` — timedelta desde una fecha de referencia (no es timestamp real)
# - `TransactionAmt` — monto de la transacción (USD)
# - `ProductCD` — código de producto (W, H, C, S, R)
# - `card1`–`card6` — información de la tarjeta de pago (tipo, categoría, banco emisor)
# - `addr1`, `addr2` — dirección de facturación (zip code, país/región)
# - `dist1`, `dist2` — distancias entre ubicaciones de transacción
# - `P_emaildomain`, `R_emaildomain` — dominio de email del comprador/receptor
# - `C1`–`C14` — features de conteo (ej: direcciones asociadas a la tarjeta)
# - `D1`–`D15` — timedeltas (ej: días desde transacciones previas)
# - `M1`–`M9` — features de match (ej: nombre en tarjeta vs dirección)
# - `V1`–`V339` — features numéricas engineered por Vesta (muchas sparse)
#
# **Identity table** (~144K train, ~141K test):
# - `id_01`–`id_38` — información de identidad (red, firma digital)
# - `DeviceType` — tipo de dispositivo (mobile, desktop)
# - `DeviceInfo` — info del dispositivo (navegador, OS, modelo)
#
# **Target**: `isFraud` — 1 si la transacción es fraudulenta, 0 si no.
#
# ## Estructura del notebook
#
# 1. **Fase 1**: Familiarización con los datos
# 2. **Fase 2**: Descubrimiento de patrones (temporal, segmentación, relaciones)
# 3. **Fase 3**: Investigación de anomalías
# 4. **Fase 4**: Generación de insights
# 5. **Fase 5**: Formulación de preguntas

# %% [markdown]
# ---
# ## Setup e Imports

# %%
import gc
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 100)
pd.set_option("display.float_format", "{:.4f}".format)

# Ruta local a los datos
DATA_DIR = Path("../ieee-fraud-detection")
assert DATA_DIR.exists(), f"Data dir not found: {DATA_DIR.resolve()}"

print(f"Data dir: {DATA_DIR.resolve()}")
print("Files:", [f.name for f in DATA_DIR.iterdir()])

# %% [markdown]
# ---
# ## Funciones utilitarias
#
# Adaptadas de `02_kaggle_kernels/eda/main.py`.

# %%
def reduce_memory(df):
    """Reduce memory usage by downcasting numeric types and converting low-cardinality strings to category."""
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


def add_time_features(df):
    """Derive day, week, hour, month proxy from TransactionDT."""
    df = df.copy()
    df["DT_day"] = (df["TransactionDT"] // (24 * 3600)).astype("int16")
    df["DT_week"] = (df["DT_day"] // 7).astype("int16")
    df["DT_hour"] = ((df["TransactionDT"] // 3600) % 24).astype("int8")
    df["DT_month_proxy"] = (df["DT_day"] // 30).astype("int8")
    return df


def candidate_uid(df):
    """Build candidate user ID from card + address + D1 anchor."""
    day = (df["TransactionDT"] // (24 * 3600)).astype("float32")
    parts = pd.DataFrame(index=df.index)
    for col in ["card1", "card2", "card3", "card5", "addr1", "addr2"]:
        parts[col] = df[col].astype("string").fillna("NA")
    parts["D1_anchor"] = (df["D1"].astype("float32") - day).round(0).astype("string").fillna("NA")
    return pd.util.hash_pandas_object(parts, index=False).astype("uint64")

# %% [markdown]
# ---
# ## Carga de datos

# %%
print("Cargando train_transaction.csv ...")
train_txn = reduce_memory(pd.read_csv(DATA_DIR / "train_transaction.csv"))
print(f"  shape: {train_txn.shape}")

print("Cargando test_transaction.csv ...")
test_txn = reduce_memory(pd.read_csv(DATA_DIR / "test_transaction.csv"))
print(f"  shape: {test_txn.shape}")

print("Cargando train_identity.csv ...")
train_id = reduce_memory(pd.read_csv(DATA_DIR / "train_identity.csv"))
print(f"  shape: {train_id.shape}")

print("Cargando test_identity.csv ...")
test_id = reduce_memory(pd.read_csv(DATA_DIR / "test_identity.csv"))
print(f"  shape: {test_id.shape}")

# %%
# Merge transaction + identity
train = train_txn.merge(train_id, on="TransactionID", how="left")
test = test_txn.merge(test_id, on="TransactionID", how="left")
del train_txn, test_txn, train_id, test_id
gc.collect()

# Marcar dataset y agregar features temporales
train["dataset"] = "train"
test["dataset"] = "test"
test["isFraud"] = np.nan

train = add_time_features(train)
test = add_time_features(test)

print(f"Train: {train.shape}")
print(f"Test:  {test.shape}")

# %% [markdown]
# ---
# # Fase 1: Familiarización con los datos
#
# Objetivo: catalogar tablas, documentar esquema, evaluar calidad, identificar
# cobertura temporal y granularidad.

# %% [markdown]
# ### 1.1 Resumen estructural

# %%
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

structural = structural_summary(train, test)
structural

# %% [markdown]
# ### 1.2 Tipos de datos y categorías de features

# %%
# Clasificar columnas por grupo semántico
feature_groups = {
    "target": ["isFraud"],
    "id_key": ["TransactionID"],
    "time": ["TransactionDT", "DT_day", "DT_week", "DT_hour", "DT_month_proxy"],
    "amount": ["TransactionAmt"],
    "product": ["ProductCD"],
    "card": [c for c in train.columns if c.startswith("card")],
    "address": ["addr1", "addr2"],
    "distance": ["dist1", "dist2"],
    "email": ["P_emaildomain", "R_emaildomain"],
    "C_count": [c for c in train.columns if c.startswith("C") and c[1:].isdigit()],
    "D_timedelta": [c for c in train.columns if c.startswith("D") and c[1:].isdigit()],
    "M_match": [c for c in train.columns if c.startswith("M") and c[1:].isdigit()],
    "V_vesta": [c for c in train.columns if c.startswith("V") and c[1:].isdigit()],
    "id_identity": [c for c in train.columns if c.startswith("id_")],
    "device": ["DeviceType", "DeviceInfo"],
}

print("Feature groups:")
for group, cols in feature_groups.items():
    existing = [c for c in cols if c in train.columns]
    print(f"  {group:15s}: {len(existing):3d} columns")

# %%
# Dtypes overview
dtype_counts = train.dtypes.value_counts()
print("Dtypes en train:")
print(dtype_counts)

# %% [markdown]
# ### 1.3 Resumen de valores faltantes

# %%
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

missing = missing_summary(train, test)

# %%
# Top 30 features con más faltantes en train
print("Top 30 features con más % faltante en train:")
missing.sort_values("train_missing_pct", ascending=False).head(30)[
    ["feature", "train_missing_pct", "test_missing_pct", "missing_delta_abs", "train_nunique"]
]

# %%
# Features con mayor diferencia de faltantes entre train/test
print("Top 20 features con mayor diferencia de faltantes train vs test:")
missing.sort_values("missing_delta_abs", ascending=False).head(20)[
    ["feature", "train_missing_pct", "test_missing_pct", "missing_delta_abs"]
]

# %%
# Distribución de % faltante
fig, ax = plt.subplots(figsize=(10, 4))
missing["train_missing_pct"].hist(bins=50, ax=ax, color="#4C72B0", alpha=0.7, label="train")
missing["test_missing_pct"].dropna().hist(bins=50, ax=ax, color="#DD8452", alpha=0.7, label="test")
ax.set_xlabel("% Missing")
ax.set_ylabel("Número de features")
ax.set_title("Distribución de % de valores faltantes por feature")
ax.legend()
plt.tight_layout()
plt.show()

# %%
# Bloques de faltantes: cuántas features tienen >80%, >50%, >20% faltantes
thresholds = [80, 50, 20, 5, 0]
for t in thresholds:
    n = (missing["train_missing_pct"] > t).sum()
    print(f"  Features con >{t}% faltante en train: {n}")

# %% [markdown]
# ### 1.4 Cobertura temporal

# %%
print(f"TransactionDT range (train): {train['TransactionDT'].min()} — {train['TransactionDT'].max()}")
print(f"TransactionDT range (test):  {test['TransactionDT'].min()} — {test['TransactionDT'].max()}")
print(f"Train days: {train['DT_day'].nunique()} unique days, range {train['DT_day'].min()}-{train['DT_day'].max()}")
print(f"Test days:  {test['DT_day'].nunique()} unique days, range {test['DT_day'].min()}-{test['DT_day'].max()}")
print(f"Train weeks: {train['DT_week'].nunique()}")
print(f"Test weeks:  {test['DT_week'].nunique()}")

# %%
# Volumen diario train vs test
fig, ax = plt.subplots(figsize=(14, 4))
train.groupby("DT_day").size().plot(ax=ax, label="train", color="#4C72B0")
test.groupby("DT_day").size().plot(ax=ax, label="test", color="#DD8452")
ax.set_xlabel("Día (desde referencia)")
ax.set_ylabel("Transacciones")
ax.set_title("Volumen diario de transacciones — train vs test")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.5 Distribución del target

# %%
fraud_counts = train["isFraud"].value_counts()
fraud_pct = train["isFraud"].value_counts(normalize=True) * 100
print("Distribución del target isFraud:")
print(pd.DataFrame({"count": fraud_counts, "pct": fraud_pct}))

fig, ax = plt.subplots(figsize=(5, 4))
fraud_pct.sort_index().plot(kind="bar", ax=ax, color=["#55A868", "#C44E52"])
ax.set_title("Distribución de isFraud")
ax.set_xlabel("isFraud")
ax.set_ylabel("%")
for i, (v, p) in enumerate(zip(fraud_counts.sort_index(), fraud_pct.sort_index())):
    ax.text(i, p + 0.5, f"{v:,}\n({p:.2f}%)", ha="center", fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# # Fase 2: Descubrimiento de patrones
#
# Exploración sistemática en 3 vectores:
# 1. Patrones temporales
# 2. Patrones de segmentación
# 3. Patrones de relación

# %% [markdown]
# ## 2.1 Patrones temporales

# %%
def temporal_summary(train):
    rows = []
    for key in ["DT_week", "DT_month_proxy"]:
        g = train.groupby(key, observed=True).agg(
            rows=("TransactionID", "size"),
            fraud_rate=("isFraud", "mean"),
            amt_mean=("TransactionAmt", "mean"),
            amt_median=("TransactionAmt", "median"),
        ).reset_index().rename(columns={key: "window"})
        g.insert(0, "window_type", key)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)

temporal = temporal_summary(train)

# %%
# Tasa de fraude por semana
weeks = temporal[temporal["window_type"] == "DT_week"]

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].bar(weeks["window"], weeks["rows"], color="#4C72B0", alpha=0.8)
axes[0].set_ylabel("Transacciones")
axes[0].set_title("Volumen y tasa de fraude por semana")

ax2 = axes[0].twinx()
ax2.plot(weeks["window"], 100 * weeks["fraud_rate"], marker="o", color="#C44E52", linewidth=2)
ax2.set_ylabel("% Fraude", color="#C44E52")

axes[1].plot(weeks["window"], weeks["amt_mean"], marker="s", color="#55A868", label="Mean")
axes[1].plot(weeks["window"], weeks["amt_median"], marker="^", color="#8172B2", label="Median")
axes[1].set_xlabel("Semana desde inicio")
axes[1].set_ylabel("TransactionAmt")
axes[1].set_title("Monto promedio y mediano por semana")
axes[1].legend()

plt.tight_layout()
plt.show()

# %%
# Tasa de fraude por hora del día
hourly = train.groupby("DT_hour").agg(
    rows=("TransactionID", "size"),
    fraud_rate=("isFraud", "mean"),
    amt_mean=("TransactionAmt", "mean"),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].bar(hourly["DT_hour"], hourly["rows"], color="#4C72B0", alpha=0.8)
axes[0].set_xlabel("Hora del día")
axes[0].set_ylabel("Transacciones")
axes[0].set_title("Volumen por hora")

axes[1].bar(hourly["DT_hour"], 100 * hourly["fraud_rate"], color="#C44E52", alpha=0.8)
axes[1].set_xlabel("Hora del día")
axes[1].set_ylabel("% Fraude")
axes[1].set_title("Tasa de fraude por hora")

plt.tight_layout()
plt.show()

# %%
# Tasa de fraude por día de la semana (proxy: DT_day % 7)
train["DT_dow"] = train["DT_day"] % 7
dow = train.groupby("DT_dow").agg(
    rows=("TransactionID", "size"),
    fraud_rate=("isFraud", "mean"),
).reset_index()

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(dow["DT_dow"], 100 * dow["fraud_rate"], color="#DD8452", alpha=0.8)
ax.set_xlabel("Día de la semana (proxy)")
ax.set_ylabel("% Fraude")
ax.set_title("Tasa de fraude por día de la semana (proxy)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2.2 Patrones de segmentación

# %%
# Tasa de fraude por categoría
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

category = category_risk(train)

# %%
# ProductCD
prod = category[category["feature"] == "ProductCD"].sort_values("fraud_rate", ascending=False)
print("Fraud rate by ProductCD:")
print(prod[["value", "rows", "fraud_rate", "lift"]].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
prod_sorted = prod.sort_values("rows", ascending=False)
axes[0].bar(prod_sorted["value"].astype(str), prod_sorted["rows"], color="#4C72B0")
axes[0].set_title("Volumen por ProductCD")
axes[0].set_ylabel("Transacciones")

axes[1].bar(prod_sorted["value"].astype(str), 100 * prod_sorted["fraud_rate"], color="#C44E52")
axes[1].set_title("Tasa de fraude por ProductCD")
axes[1].set_ylabel("% Fraude")

plt.tight_layout()
plt.show()

# %%
# card4 (red de tarjeta) y card6 (tipo debit/credit)
for feat in ["card4", "card6"]:
    sub = category[category["feature"] == feat].sort_values("rows", ascending=False)
    print(f"\nFraud rate by {feat}:")
    print(sub[["value", "rows", "fraud_rate", "lift"]].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, feat, color in zip(axes, ["card4", "card6"], ["#55A868", "#8172B2"]):
    sub = category[category["feature"] == feat].sort_values("rows", ascending=False)
    ax.barh(sub["value"].astype(str), 100 * sub["fraud_rate"], color=color)
    ax.set_xlabel("% Fraude")
    ax.set_title(f"Tasa de fraude por {feat}")
plt.tight_layout()
plt.show()

# %%
# DeviceType
sub = category[category["feature"] == "DeviceType"].sort_values("rows", ascending=False)
print("Fraud rate by DeviceType:")
print(sub[["value", "rows", "fraud_rate", "lift"]].to_string(index=False))

# %%
# Top email domains por fraude
for feat in ["P_emaildomain", "R_emaildomain"]:
    sub = category[category["feature"] == feat].sort_values("fraud_rate", ascending=False).head(15)
    print(f"\nTop fraud rate by {feat}:")
    print(sub[["value", "rows", "fraud_rate", "lift"]].to_string(index=False))

# %%
# M features (match flags)
m_feats = [f for f in category["feature"].unique() if f.startswith("M")]
if m_feats:
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    for i, feat in enumerate(sorted(m_feats)):
        sub = category[category["feature"] == feat].sort_values("rows", ascending=False)
        if i < len(axes):
            axes[i].barh(sub["value"].astype(str), 100 * sub["fraud_rate"], color="#DD8452")
            axes[i].set_title(feat)
            axes[i].set_xlabel("% Fraude")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Tasa de fraude por M features (match flags)", fontsize=14)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 2.3 Patrones de relación

# %% [markdown]
# ### 2.3.1 Correlaciones con target

# %%
def target_correlations(train):
    num_cols = train.select_dtypes(include=[np.number]).columns.difference(["isFraud", "TransactionID"])
    corrs = train[num_cols].corrwith(train["isFraud"]).dropna()
    out = pd.DataFrame({
        "feature": corrs.index,
        "corr_isFraud": corrs.values,
        "abs_corr": corrs.abs().values,
    }).sort_values("abs_corr", ascending=False)
    return out

corr = target_correlations(train)

# %%
# Top 30 features más correlacionadas con isFraud
print("Top 30 features correlacionadas con isFraud:")
corr.head(30)

# %%
fig, ax = plt.subplots(figsize=(9, 8))
top30 = corr.head(30).sort_values("abs_corr")
colors = ["#C44E52" if v < 0 else "#4C72B0" for v in top30["corr_isFraud"]]
ax.barh(top30["feature"], top30["corr_isFraud"], color=colors)
ax.set_xlabel("Correlación con isFraud")
ax.set_title("Top 30 features — correlación con isFraud")
ax.axvline(0, color="black", linewidth=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.3.2 TransactionAmt — análisis profundo

# %%
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

amount = amount_analysis(train)
amount.T

# %%
# Distribución de TransactionAmt
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

train["TransactionAmt"].clip(upper=train["TransactionAmt"].quantile(0.99)).hist(bins=80, ax=axes[0], color="#4C72B0")
axes[0].set_title("TransactionAmt (cap p99)")
axes[0].set_xlabel("USD")

np.log1p(train["TransactionAmt"]).hist(bins=80, ax=axes[1], color="#55A868")
axes[1].set_title("log1p(TransactionAmt)")
axes[1].set_xlabel("log1p(USD)")

# Fraude vs no-fraude
for label, color, val in [(0, "#55A868", "No fraude"), (1, "#C44E52", "Fraude")]:
    sub = train[train["isFraud"] == label]["TransactionAmt"]
    np.log1p(sub).hist(bins=80, ax=axes[2], color=color, alpha=0.5, label=val, density=True)
axes[2].set_title("log1p(TransactionAmt) por clase")
axes[2].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.3.3 Distribution shift train vs test (KS)

# %%
def numeric_distribution_shift(train, test, sample_n=250000):
    numeric = train.select_dtypes(include=[np.number]).columns.intersection(
        test.select_dtypes(include=[np.number]).columns
    )
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

shift = numeric_distribution_shift(train, test)

# %%
# Top 30 features con mayor KS statistic
print("Top 30 features con mayor distribution shift (KS) train vs test:")
shift.head(30)

# %%
fig, ax = plt.subplots(figsize=(9, 8))
top_shift = shift.head(30).sort_values("ks_train_test")
ax.barh(top_shift["feature"], top_shift["ks_train_test"], color="#8172B2")
ax.set_xlabel("KS statistic")
ax.set_title("Top 30 features — distribution shift train vs test (KS)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.3.4 Redundancia entre features V/C/D

# %%
def redundancy_summary(train):
    rows = []
    for prefix in ["C", "D", "V"]:
        cols = [c for c in train.columns if c.startswith(prefix) and c[1:].isdigit()]
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(train[c])]
        if len(cols) < 2:
            continue
        sample = train[cols].sample(min(len(train), 120000), random_state=42)
        corr_mat = sample.corr().abs()
        upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
        pairs = upper.stack().reset_index()
        pairs.columns = ["feature_a", "feature_b", "abs_corr"]
        top = pairs.sort_values("abs_corr", ascending=False).head(50)
        top.insert(0, "group", prefix)
        rows.append(top)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

redundancy = redundancy_summary(train)

# %%
# Pares más correlacionados por grupo
print("Top pares redundantes (|corr| > 0.95):")
high_red = redundancy[redundancy["abs_corr"] > 0.95]
print(f"  Pares con |corr| > 0.95: {len(high_red)}")
for group in ["C", "D", "V"]:
    n = len(high_red[high_red["group"] == group])
    print(f"    Grupo {group}: {n} pares")

print("\nTop 20 pares más correlacionados:")
redundancy.head(20)

# %% [markdown]
# ### 2.3.5 Análisis de UID candidato y leakage temporal

# %%
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

uid = uid_overlap_analysis(train)
uid.T

# %% [markdown]
# ---
# # Fase 3: Investigación de anomalías
#
# Identificar 3-5 anomalías específicas de las fases anteriores,
# investigar cada una y determinar si son problemas de calidad o fenómenos reales.

# %% [markdown]
# ### 3.1 Anomalía: TransactionAmt con decimales multi-nivel

# %%
amt = train["TransactionAmt"].astype("float64")
frac = (amt * 100).round(4) % 1
multi_dec = frac.abs() > 1e-6

print(f"Transacciones con decimales más allá de centavos: {multi_dec.sum():,} ({100*multi_dec.mean():.2f}%)")
print(f"  Fraud rate (multi-decimal): {train.loc[multi_dec, 'isFraud'].mean():.4f}")
print(f"  Fraud rate (standard):      {train.loc[~multi_dec, 'isFraud'].mean():.4f}")

# Posible causa: conversión de moneda extranjera
print(f"\n  Amt stats (multi-dec):  mean={amt[multi_dec].mean():.2f}, median={amt[multi_dec].median():.2f}")
print(f"  Amt stats (standard):  mean={amt[~multi_dec].mean():.2f}, median={amt[~multi_dec].median():.2f}")

# Distribución por ProductCD
print("\n  Multi-decimal by ProductCD:")
cross = pd.crosstab(train["ProductCD"], multi_dec, normalize="index") * 100
cross.columns = ["standard_pct", "multi_dec_pct"]
print(cross)

# %% [markdown]
# ### 3.2 Anomalía: Features con >80% faltante — ¿bloque de identity?

# %%
high_missing = missing[missing["train_missing_pct"] > 80].sort_values("train_missing_pct", ascending=False)
print(f"Features con >80% faltante en train: {len(high_missing)}")

# ¿Son todas del grupo identity?
id_cols = set(c for c in train.columns if c.startswith("id_") or c in ["DeviceType", "DeviceInfo"])
high_missing_names = set(high_missing["feature"])
from_identity = high_missing_names & id_cols
from_other = high_missing_names - id_cols

print(f"  De identity: {len(from_identity)}")
print(f"  De transaction: {len(from_other)}")
if from_other:
    print(f"  Transaction features >80% missing: {sorted(from_other)}")

# ¿El patrón de faltantes es igual en train/test?
print(f"\n  Max missing_delta_abs para features >80% missing: {high_missing['missing_delta_abs'].max():.2f}%")

# %% [markdown]
# ### 3.3 Anomalía: Distribution shift extremo en features específicas

# %%
extreme_shift = shift[shift["ks_train_test"] > 0.20]
print(f"Features con KS > 0.20 (distribution shift severo): {len(extreme_shift)}")
print(extreme_shift[["feature", "ks_train_test", "train_mean", "test_mean", "mean_delta"]].head(15))

# Verificar si son temporales
temporal_features = ["DT_day", "DT_week", "DT_month_proxy", "TransactionDT"]
temp_shift = extreme_shift[extreme_shift["feature"].isin(temporal_features)]
non_temp_shift = extreme_shift[~extreme_shift["feature"].isin(temporal_features)]
print(f"\n  Temporal features con alto shift: {len(temp_shift)}")
print(f"  Non-temporal features con alto shift: {len(non_temp_shift)}")
if len(non_temp_shift):
    print("  Non-temporal high-shift features:")
    print(non_temp_shift[["feature", "ks_train_test", "mean_delta"]].to_string(index=False))

# %% [markdown]
# ### 3.4 Anomalía: Variación semanal de tasa de fraude

# %%
weeks = temporal[temporal["window_type"] == "DT_week"]
fraud_min = 100 * weeks["fraud_rate"].min()
fraud_max = 100 * weeks["fraud_rate"].max()
fraud_std = 100 * weeks["fraud_rate"].std()
print(f"Tasa de fraude semanal: min={fraud_min:.2f}%, max={fraud_max:.2f}%, std={fraud_std:.2f}%")
print(f"Ratio max/min: {fraud_max/fraud_min:.2f}x")

# Semanas con tasa anómala (>2 std del promedio)
fraud_mean = weeks["fraud_rate"].mean()
fraud_2std = fraud_mean + 2 * weeks["fraud_rate"].std()
anomalous_weeks = weeks[weeks["fraud_rate"] > fraud_2std]
print(f"\nSemanas con tasa >2σ del promedio ({100*fraud_2std:.2f}%):")
if len(anomalous_weeks):
    print(anomalous_weeks[["window", "rows", "fraud_rate"]].to_string(index=False))
else:
    print("  Ninguna")

# %% [markdown]
# ### 3.5 Anomalía: UID conocido vs desconocido — diferencia de fraud rate

# %%
print("Análisis UID — segmento late (validación temporal):")
print(uid.T.to_string())
print(f"\nDiferencia de fraud rate known vs unknown UID:")
diff = uid["late_known_uid_fraud_rate"].iloc[0] - uid["late_unknown_uid_fraud_rate"].iloc[0]
print(f"  {uid['late_known_uid_fraud_rate'].iloc[0]:.4f} vs {uid['late_unknown_uid_fraud_rate'].iloc[0]:.4f} = delta {diff:.4f}")
print(f"  Esto indica {'posible leakage' if abs(diff) > 0.01 else 'diferencia menor'} al usar UID features en validación.")

# %% [markdown]
# ---
# # Fase 4: Generación de insights
#
# Criterios: cada insight debe ser **accionable**, **sorprendente** o **significativo**.

# %% [markdown]
# ### Insight 1: Desbalance extremo del target requiere métricas especiales

# %%
fraud_rate = structural.loc[structural["dataset"] == "train", "fraud_rate_pct"].iloc[0]
print(f"Fraud rate: {fraud_rate:.2f}%")
print("→ Accuracy sería engañosa (un clasificador dummy logra ~96.5%).")
print("→ Usar PR-AUC como métrica principal. Complementar con recall, precision, F1, ROC-AUC.")

# %% [markdown]
# ### Insight 2: TransactionAmt con decimales multi-nivel es señal de fraude

# %%
print(f"Multi-decimal fraud rate:     {amount['multi_decimal_fraud_rate'].iloc[0]:.4f}")
print(f"Non-multi-decimal fraud rate: {amount['non_multi_decimal_fraud_rate'].iloc[0]:.4f}")
lift_dec = amount['multi_decimal_fraud_rate'].iloc[0] / amount['non_multi_decimal_fraud_rate'].iloc[0]
print(f"Lift: {lift_dec:.2f}x")
print("→ Montos con decimales extraños (conversión de moneda) correlacionan con fraude.")
print("→ Feature: is_multi_decimal flag.")

# %% [markdown]
# ### Insight 3: Distribution shift temporal significativo limita generalización

# %%
n_high_shift = (shift["ks_train_test"] > 0.20).sum()
print(f"Features con KS > 0.20: {n_high_shift}")
print("→ Features temporales y derivadas del tiempo dominan el shift.")
print("→ Necesario: adversarial validation, PSI monitoring, temporal cross-validation.")

# %% [markdown]
# ### Insight 4: UIDs candidatos cruzan corte temporal — riesgo de leakage

# %%
print(f"% filas en UIDs que cruzan corte temporal: {uid['all_rows_in_crossing_uid_pct'].iloc[0]:.2f}%")
print(f"% filas tardías con UID conocido: {uid['late_rows_known_uid_pct'].iloc[0]:.2f}%")
print("→ Muchos 'usuarios' aparecen en ambos lados del split temporal.")
print("→ Features UID son poderosas pero deben calcularse sin información futura.")
print("→ Reportar desempeño separado: UID conocido vs desconocido.")

# %% [markdown]
# ### Insight 5: Bloques de V-features son altamente redundantes

# %%
high_red_v = redundancy[(redundancy["group"] == "V") & (redundancy["abs_corr"] > 0.95)]
print(f"Pares V con |corr| > 0.95: {len(high_red_v)}")
print("→ Reducción agresiva de V features necesaria.")
print("→ Estrategia: PCA por bloque o selección por permutation importance.")

# %% [markdown]
# ---
# # Fase 5: Formulación de preguntas
#
# Preguntas específicas y accionables para investigación más profunda.

# %% [markdown]
# ### Adversarial validation train vs test

# %%
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


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

    # Permutation importance on validation
    base_auc = auc
    rng = np.random.default_rng(42)
    imps = []
    for col in X_val.columns:
        Xp = X_val.copy()
        Xp[col] = rng.permutation(Xp[col].to_numpy())
        auc_p = roc_auc_score(y_val, model.predict_proba(Xp)[:, 1])
        imps.append({"feature": col, "adversarial_auc_drop": base_auc - auc_p})
    imp = pd.DataFrame(imps).sort_values("adversarial_auc_drop", ascending=False)
    metrics = pd.DataFrame([{"adversarial_roc_auc": auc, "adversarial_pr_auc": ap, "features_used": len(feature_candidates)}])
    return metrics, imp


print("Running adversarial validation...")
adv_metrics, adv_importance = adversarial_validation(train, test)

# %%
print("Adversarial validation metrics:")
print(adv_metrics.to_string(index=False))

print("\nTop 15 features que más distinguen train de test:")
print(adv_importance.head(15).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(9, 6))
top_imp = adv_importance.head(20).sort_values("adversarial_auc_drop")
ax.barh(top_imp["feature"], top_imp["adversarial_auc_drop"], color="#C44E52")
ax.set_xlabel("AUC drop (permutation importance)")
ax.set_title("Adversarial Validation — Top features que distinguen train vs test")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Decision log — Resumen de decisiones de modelado

# %%
def make_decision_log(structural, missing, corr, temporal, uid, amount, shift, adv_metrics):
    fraud_rate = structural.loc[structural["dataset"] == "train", "fraud_rate_pct"].iloc[0]
    weekly = temporal[temporal["window_type"] == "DT_week"]
    min_week = 100 * weekly["fraud_rate"].min()
    max_week = 100 * weekly["fraud_rate"].max()
    known_late = uid["late_rows_known_uid_pct"].iloc[0]
    adv_auc = adv_metrics["adversarial_roc_auc"].iloc[0]
    high_missing = int((missing["train_missing_pct"] > 80).sum())
    high_shift = int((shift["ks_train_test"] > 0.20).sum())
    max_corr = corr["abs_corr"].max()

    rows = [
        {
            "decision": "Usar PR-AUC como métrica principal",
            "evidence": f"Fraude en train = {fraud_rate:.2f}%; accuracy sería engañosa.",
            "modeling_impact": "Reportar PR-AUC, recall, precision, F1 y FPR por ventana temporal.",
        },
        {
            "decision": "No usar TransactionDT crudo como feature",
            "evidence": "Es la variable que define el orden temporal; usarla cruda puede aprender posición histórica.",
            "modeling_impact": "Derivar hora/día/semana y usar TransactionDT para split y monitoreo.",
        },
        {
            "decision": "Separar validación temporal en UID conocido y UID desconocido",
            "evidence": f"{known_late:.2f}% de filas tardías comparten candidate_uid con entrenamiento temprano.",
            "modeling_impact": "Reportar desempeño por segmento y evitar conclusiones infladas.",
        },
        {
            "decision": "Crear features UID, pero sin usar información futura",
            "evidence": "card/D/addr forman clientes candidatos repetidos; top soluciones IEEE explotan esa señal.",
            "modeling_impact": "Frequency/agregaciones calculadas solo con ventana de entrenamiento.",
        },
        {
            "decision": "Conservar outliers de TransactionAmt",
            "evidence": f"Outliers IQR tienen tasa de fraude {100 * amount['iqr_outlier_fraud_rate'].iloc[0]:.2f}%.",
            "modeling_impact": "Usar log1p, flags y winsorizar solo para modelos sensibles, no eliminar.",
        },
        {
            "decision": "Tratar faltantes por bloque",
            "evidence": f"{high_missing} features tienen más de 80% faltante en train.",
            "modeling_impact": "Agregar indicadores de ausencia, has_identity y missing-block features.",
        },
        {
            "decision": "Auditar distribution shift antes de seleccionar features",
            "evidence": f"{high_shift} features numéricas tienen KS train/test > 0.20; adversarial AUC = {adv_auc:.3f}.",
            "modeling_impact": "Penalizar features inestables y medir robustez temporal.",
        },
        {
            "decision": "No descartar columnas por correlación con target solamente",
            "evidence": f"Max abs corr con isFraud = {max_corr:.3f}; no hay copia directa obvia del target.",
            "modeling_impact": "Combinar correlación, estabilidad temporal y permutation importance.",
        },
        {
            "decision": "Modelar drift explícitamente",
            "evidence": f"La tasa semanal de fraude varía entre {min_week:.2f}% y {max_week:.2f}%.",
            "modeling_impact": "Ventanas temporales, PSI/KS y comparación static vs adaptive.",
        },
    ]
    return pd.DataFrame(rows)

decisions = make_decision_log(structural, missing, corr, temporal, uid, amount, shift, adv_metrics)
decisions

# %% [markdown]
# ---
# ## Preguntas para investigación más profunda
#
# | # | Pregunta | Proceso | Datos necesarios | Por qué importa |
# |---|---------|---------|------------------|-----------------|
# | 1 | ¿Cómo varía el desempeño del modelo entre UID conocido vs desconocido en validación temporal? | Entrenamiento con temporal split | Features UID, split 80/20 | Evitar sobreestimación; UID conocido puede inflar métricas |
# | 2 | ¿Cuáles V-features sobreviven feature selection basada en estabilidad temporal + importancia? | Feature selection | V1-V339, KS scores, permutation importance | 339 features son muchas; reducir redundancia sin perder señal |
# | 3 | ¿Los montos multi-decimal (conversión de moneda) predicen fraude independientemente de otras señales? | Ablation study | TransactionAmt, is_multi_decimal flag | Verificar si la señal es genuina o correlacionada con ProductCD |
# | 4 | ¿Cuál es el impacto de distribution shift en las D-features sobre predicciones en test? | Adversarial analysis por grupo | D1-D15, temporal splits | D-features tienen alto shift; podrían degradar generalización |
# | 5 | ¿Un ensemble de modelos por ProductCD supera a un modelo global? | Comparativa de modelos | Todas las features, ProductCD como segmento | ProductCD tiene fraud rates muy diferentes; modelos especializados podrían capturar mejor |

# %% [markdown]
# ---
# ## Resumen ejecutivo

# %%
import json

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

print(json.dumps(summary, indent=2))

# %% [markdown]
# ---
# **Fin del EDA. Próximos pasos: feature engineering y modelado baseline.**
