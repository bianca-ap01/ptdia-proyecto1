"""Shared, deterministic IEEE-CIS helpers for the four final notebooks.

The notebook entry points call these functions. Nothing is fitted on future
rows: a preprocessing instance is created for each historical training window.
"""

from __future__ import annotations

import gc
import gzip
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, precision_score, recall_score

DAY = 86400
WEEK = 7 * DAY
SEED = 42
CAT_COLS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"] + [f"M{i}" for i in range(1, 10)]
MONITOR_COLS = ["TransactionAmt", "D1n", "D2n", "D10n", "D15n", "C9", "id_02", "id_20", "D11", "id_01"]


def paths(stage: str):
    here = Path(__file__).resolve().parent
    candidates = [
        Path("/kaggle/input/ieee-fraud-detection"),
        Path("/kaggle/input/competitions/ieee-fraud-detection"),
        here.parent / "ieee-fraud-detection",
    ]
    data = next((p for p in candidates if (p / "train_transaction.csv").exists()), None)
    if data is None:
        raise FileNotFoundError("IEEE-CIS competition input unavailable")
    output = Path("/kaggle/working") if os.name == "posix" and Path("/kaggle/working").exists() else here / "kaggle" / stage / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    return data, output


def manifest(output: Path, **values):
    target = output / "run_summary.json"
    current = json.loads(target.read_text()) if target.exists() else {}
    current.update(values)
    target.write_text(json.dumps(current, indent=2, default=float), encoding="utf-8")


def save_plot(output: Path, name: str):
    plt.tight_layout()
    plt.savefig(output / "plots" / name, dpi=140, bbox_inches="tight")
    plt.close()


def load_data(data: Path, include_test: bool = False):
    def one(prefix: str):
        txn = pd.read_csv(data / f"{prefix}_transaction.csv", low_memory=False)
        identity = pd.read_csv(data / f"{prefix}_identity.csv", low_memory=False)
        # IEEE-CIS test identity uses id-01 style names, while train uses id_01.
        identity.rename(columns=lambda c: c.replace("id-", "id_") if c.startswith("id-") else c, inplace=True)
        if txn.TransactionID.duplicated().any() or identity.TransactionID.duplicated().any():
            raise ValueError("TransactionID must be unique in each source")
        out = txn.merge(identity, on="TransactionID", how="left", validate="one_to_one", indicator="_identity_join")
        del txn, identity
        gc.collect()
        out["has_identity_join"] = out["_identity_join"].eq("both").astype("int8")
        out.drop(columns="_identity_join", inplace=True)
        out.sort_values(["TransactionDT", "TransactionID"], inplace=True)
        out.reset_index(drop=True, inplace=True)
        return out

    train = one("train")
    test = one("test") if include_test else None
    return train, test


def split_bounds(df: pd.DataFrame):
    return float(df.TransactionDT.quantile(0.70)), float(df.TransactionDT.quantile(0.85))


def split_name(dt: pd.Series, bounds: tuple[float, float]):
    return np.where(dt <= bounds[0], "train", np.where(dt <= bounds[1], "valid", "holdout"))


def row_features(df: pd.DataFrame):
    """Only own-row information. Fitted statistics are added by WindowEncoder."""
    out = df.copy()
    day = (out.TransactionDT // DAY).astype("int16")
    out["DT_hour"] = ((out.TransactionDT // 3600) % 24).astype("int8")
    out["DT_day_index"] = day
    out["DT_week_index"] = (day // 7).astype("int16")
    out["has_identity"] = out["has_identity_join"].astype("int8")
    out["TransactionAmt_log1p"] = np.log1p(out.TransactionAmt.clip(lower=0)).astype("float32")
    for family in ("V", "D", "id_"):
        cols = [c for c in out if c.startswith(family) and (family == "id_" or c[len(family):].isdigit())]
        out[f"missing_{family.replace('_', '')}_count"] = out[cols].isna().sum(axis=1).astype("int16") if cols else 0
    core = [c for c in ["card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2"] if c in out]
    out["missing_core_count"] = out[core].isna().sum(axis=1).astype("int8")
    for col in ["D1", "D2", "D10", "D15"]:
        out[f"{col}n"] = (out[col] - day).astype("float32")
    return out


def candidate_uid(df: pd.DataFrame):
    parts = df[["card1", "card2", "card3", "card5", "addr1", "addr2"]].astype("string").fillna("NA")
    parts = parts.copy()
    parts["D1_anchor"] = (df.D1 - df.TransactionDT // DAY).round(0).astype("string").fillna("NA")
    return pd.util.hash_pandas_object(parts, index=False).astype("uint64")


def feature_list():
    here = Path(__file__).resolve().parent
    for path in (here / "v01_features.json", here / "input" / "v01_features.json"):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    if "V01_FEATURES" in globals():
        return list(globals()["V01_FEATURES"])
    raise FileNotFoundError("v01_features.json is required")


@dataclass
class WindowEncoder:
    features: list[str]
    q_low: float = 0.0
    q_high: float = 0.0
    maps: dict | None = None
    freqs: dict | None = None

    def fit(self, train: pd.DataFrame):
        q1, q3 = train.TransactionAmt.quantile([.25, .75])
        self.q_low, self.q_high = float(q1 - 1.5 * (q3 - q1)), float(q3 + 1.5 * (q3 - q1))
        self.maps, self.freqs = {}, {}
        for col in CAT_COLS:
            if col not in train:
                continue
            s = train[col].astype("string").fillna("__MISSING__")
            self.maps[col] = {v: i for i, v in enumerate(s.unique())}
            self.freqs[col] = (s.value_counts(dropna=False) / len(s)).to_dict()
        return self

    def transform(self, rows: pd.DataFrame):
        cols = {}
        for name in self.features:
            if name == "TransactionAmt_outlier_iqr":
                cols[name] = ((rows.TransactionAmt < self.q_low) | (rows.TransactionAmt > self.q_high)).astype("float32")
            elif name.endswith("_label") and name[:-6] in self.maps:
                col = name[:-6]
                cols[name] = rows[col].astype("string").fillna("__MISSING__").map(self.maps[col]).fillna(-1).astype("float32")
            elif name.endswith("_freq") and name[:-5] in self.freqs:
                col = name[:-5]
                cols[name] = rows[col].astype("string").fillna("__MISSING__").map(self.freqs[col]).fillna(0).astype("float32")
            elif name in rows:
                cols[name] = pd.to_numeric(rows[name], errors="coerce").astype("float32")
            else:
                cols[name] = pd.Series(np.nan, index=rows.index, dtype="float32")
        return pd.DataFrame(cols, index=rows.index).replace([np.inf, -np.inf], np.nan)


def psi(reference, current, bins=10):
    a = np.asarray(reference, dtype=float)
    b = np.asarray(current, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 100 or len(b) < 100:
        return np.nan
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0 if np.isclose(np.mean(a), np.mean(b)) else np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.clip(np.histogram(a, edges)[0] / len(a), 1e-6, 1)
    pb = np.clip(np.histogram(b, edges)[0] / len(b), 1e-6, 1)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def safe_ap(y, pred):
    return float(average_precision_score(y, pred)) if len(y) and len(np.unique(y)) == 2 else np.nan


def evaluation_blocks(df: pd.DataFrame, bounds: tuple[float, float]):
    """Seven-day blocks clipped to each split; first valid week is label warm-up."""
    blocks = []
    for split, start, end in (("valid", bounds[0], bounds[1]), ("holdout", bounds[1], float(df.TransactionDT.max()) + 1)):
        index = 0
        t = start
        while t < end:
            u = min(t + WEEK, end)
            # First validation week warms up the seven-day label delay.
            if not (split == "valid" and index == 0):
                mask = (df.TransactionDT > t) & (df.TransactionDT <= u) if split == "valid" else (df.TransactionDT > t) & (df.TransactionDT < u)
                ix = df.index[mask].to_numpy()
                if len(ix):
                    blocks.append({"split": split, "block": index, "start": t, "end": u, "partial": u - t < WEEK, "indices": ix})
            t = u
            index += 1
    return blocks


def prediction_rows(df: pd.DataFrame, ix, uid_known, pred, model, strategy, window_days, block):
    return pd.DataFrame({
        "TransactionID": df.loc[ix, "TransactionID"].to_numpy(),
        "TransactionDT": df.loc[ix, "TransactionDT"].to_numpy(),
        "TransactionAmt": df.loc[ix, "TransactionAmt"].to_numpy(),
        "isFraud": df.loc[ix, "isFraud"].to_numpy(),
        "uid_known": uid_known,
        "score": np.asarray(pred),
        "model": model,
        "strategy": strategy,
        "window_days": window_days,
        "split": block["split"],
        "block": block["block"],
    })


def append_predictions(output: Path, rows: pd.DataFrame):
    path = output / "predictions.csv.gz"
    rows.to_csv(path, mode="at" if path.exists() else "wt", index=False, header=not path.exists(), compression="gzip")


def quick_cost(y, amount, score, review_fraction=.05, review_effectiveness=.8):
    """Offline same-capacity ranking proxy; operational queue is in policy notebook."""
    n = len(score)
    k = max(1, int(n * review_fraction))
    review = np.zeros(n, dtype=bool)
    review[np.argsort(score)[-k:]] = True
    loss = 4.41 * np.asarray(amount) * np.asarray(y)
    total = float(np.sum(np.where(review, 2 + (1 - review_effectiveness) * loss, loss)))
    return total / max(n, 1), float(recall_score(y, review, zero_division=0)), float(precision_score(y, review, zero_division=0))


def block_metrics(rows: pd.DataFrame, seconds=0.0, updated=False, device="none", psi_score=np.nan):
    y = rows.isFraud.to_numpy(dtype=int)
    p = rows.score.to_numpy(dtype=float)
    cost, recall, precision = quick_cost(y, rows.TransactionAmt, p)
    return {
        "model": rows.model.iloc[0], "strategy": rows.strategy.iloc[0], "window_days": rows.window_days.iloc[0],
        "split": rows.split.iloc[0], "block": int(rows.block.iloc[0]), "rows": len(rows),
        "fraud_rate": float(np.mean(y)), "pr_auc": safe_ap(y, p), "recall_at_5pct": recall,
        "precision_at_5pct": precision, "cost_per_txn_proxy": cost,
        "uid_known_pr_auc": safe_ap(y[rows.uid_known.to_numpy(dtype=bool)], p[rows.uid_known.to_numpy(dtype=bool)]),
        "uid_unknown_pr_auc": safe_ap(y[~rows.uid_known.to_numpy(dtype=bool)], p[~rows.uid_known.to_numpy(dtype=bool)]),
        "train_seconds": seconds, "updated": int(updated), "device": device, "psi_score": psi_score,
    }


def run_eda():
    data, out = paths("eda")
    manifest(out, status="running", stage="load", source=str(data))
    train, test = load_data(data, include_test=True)
    bounds = split_bounds(train)
    train["week"] = (train.TransactionDT // WEEK).astype(int)
    prevalence = train.groupby("week", observed=True).isFraud.agg(["size", "sum", "mean"]).reset_index()
    prevalence.to_csv(out / "fraud_by_week.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = train.isFraud.value_counts().reindex([0, 1])
    ax.bar(["Legitima", "Fraude"], counts.values, color=["#4c78a8", "#e45756"])
    ax.set(ylabel="Transacciones", title="Desbalance de clases en train etiquetado")
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom")
    save_plot(out, "01_clases.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(prevalence.week, 100 * prevalence["mean"], marker="o", color="#e45756")
    ax.axvline(bounds[0] / WEEK, color="#f2a541", linestyle="--", label="70 %")
    ax.axvline(bounds[1] / WEEK, color="#8338ec", linestyle="--", label="85 %")
    ax.set(xlabel="Semana relativa", ylabel="Fraude (%)", title="Prevalencia semanal y cortes cronologicos")
    ax.legend()
    save_plot(out, "02_prevalencia_semanal.png")

    miss = pd.DataFrame({"train_missing": train.isna().mean(), "test_missing": test.isna().mean()})
    miss["delta"] = miss.test_missing - miss.train_missing
    miss.sort_values("train_missing", ascending=False).to_csv(out / "missingness.csv")
    chosen = miss.sort_values("train_missing", ascending=False).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(chosen))
    ax.barh(y - .2, chosen.train_missing * 100, height=.4, label="train")
    ax.barh(y + .2, chosen.test_missing * 100, height=.4, label="test")
    ax.set_yticks(y, chosen.index)
    ax.set(xlabel="Datos faltantes (%)", title="20 columnas con mas faltantes en train")
    ax.legend()
    save_plot(out, "03_faltantes.png")

    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, frame, color in (("train", train, "#4c78a8"), ("test", test, "#e45756")):
        values = frame.TransactionAmt.dropna().to_numpy()
        values = rng.choice(values, min(80000, len(values)), replace=False)
        ax.hist(np.log1p(values.clip(min=0)), bins=65, density=True, alpha=.5, label=label, color=color)
    ax.set(xlabel="log(1 + monto)", ylabel="Densidad", title="Distribucion de montos; muestra aleatoria fija")
    ax.legend()
    save_plot(out, "04_montos.png")

    # Three numeric features ranked highly in the existing V01 importance.
    # Plot with fixed sampling and shared within-feature limits for readability.
    feature_stats = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, col in zip(axes, ("V258", "C5", "D1")):
        bounds_col = train[col].dropna().quantile([.01, .99]).to_numpy()
        for value, label, color in ((0, "Legitima", "#4c78a8"), (1, "Fraude", "#e45756")):
            part = train.loc[train.isFraud == value, col]
            feature_stats.append({"feature": col, "class": label, "rows": len(part),
                                  "missing_rate": float(part.isna().mean()), "median": float(part.median())})
            values = part.dropna().to_numpy()
            if len(values) > 40000:
                values = rng.choice(values, 40000, replace=False)
            ax.hist(np.clip(values, *bounds_col), bins=45, density=True, alpha=.45,
                    label=label, color=color)
        ax.set(xlabel=col, ylabel="Densidad", title=f"{col}: clases en train")
    axes[0].legend()
    save_plot(out, "06_variables_v01.png")
    pd.DataFrame(feature_stats).to_csv(out / "important_feature_stats.csv", index=False)

    for col in ("ProductCD", "DeviceType"):
        group = train.groupby(col, observed=True, dropna=False).isFraud.agg(["size", "mean"]).sort_values("size", ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(group.index.astype(str), group["mean"] * 100, color="#72b7b2")
        ax.axhline(train.isFraud.mean() * 100, color="black", linestyle="--", label="Tasa global")
        ax.set(xlabel=col, ylabel="Fraude (%)", title=f"Tasa descriptiva por {col}; train etiquetado")
        ax.legend()
        save_plot(out, f"05_riesgo_{col}.png")
        group.to_csv(out / f"risk_{col}.csv")

    # Row availability and preprocessing audits: do not assert that opaque V/C/D
    # columns can be proven leakage-free from public metadata.
    audit = [
        ("TransactionID", "excluir", "Llave unica y proxy de orden; no predictor."),
        ("TransactionDT / DT_day_index / DT_week_index", "excluir como predictor", "Solo orden, cortes y monitoreo."),
        ("TransactionAmt_outlier_iqr", "corregido", "Cuartiles ajustados en cada ventana; V01 los ajustaba sobre todas las filas."),
        ("Categorias y frecuencias", "controlado", "Mapas construidos solo con filas del entrenamiento permitido."),
        ("UID e historicos", "controlado con limite", "UID por atributos propios; agregados etiquetados solo del pasado, sin fila futura."),
        ("V/C/D anonimizadas", "no demostrable completamente", "Auditar disponibilidad en inferencia; el origen de cada campo no es publico."),
        ("Validacion / early stopping", "separacion", "La validacion selecciona decisiones; holdout final reservado para evaluacion."),
    ]
    pd.DataFrame(audit, columns=["feature_or_step", "status", "reason"]).to_csv(out / "leakage_audit.csv", index=False)
    summary = {
        "train_rows": len(train), "test_rows": len(test), "train_fraud_rate": float(train.isFraud.mean()),
        "train_day_min": float(train.TransactionDT.min() / DAY), "train_day_max": float(train.TransactionDT.max() / DAY),
        "test_day_min": float(test.TransactionDT.min() / DAY), "test_day_max": float(test.TransactionDT.max() / DAY),
        "test_has_target": "isFraud" in test, "train_cutoff": bounds[0], "valid_cutoff": bounds[1],
        "weekly_fraud_rate_min": float(prevalence["mean"].min()), "weekly_fraud_rate_max": float(prevalence["mean"].max()),
    }
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest(out, status="complete", stage="done", **summary)
    return out, summary


def _v01_predictions():
    here = Path(__file__).resolve().parent
    candidates = [here / "v01_reference", here.parent / "03_outputs" / "v01_baseline_temporal"]
    candidates += list(Path("/kaggle/input").glob("**/baseline_valid_predictions.csv")) if Path("/kaggle/input").exists() else []
    for path in candidates:
        folder = path if path.is_dir() else path.parent
        a, b = folder / "baseline_valid_predictions.csv", folder / "baseline_holdout_predictions.csv"
        if a.exists() and b.exists():
            return pd.concat([pd.read_csv(a), pd.read_csv(b)], ignore_index=True)
    return None


def run_drift():
    data, out = paths("monitoreo")
    manifest(out, status="running", stage="load")
    train, test = load_data(data, include_test=True)
    bounds = split_bounds(train)
    train = row_features(train)
    test = row_features(test)
    train["origin"], test["origin"] = "train", "test"
    # Calendar-aligned one-week snapshots, with moving 7/30-day means.
    cols = ["TransactionDT", "origin", "has_identity", "TransactionAmt"] + [c for c in MONITOR_COLS if c not in ("TransactionAmt",) and c in train]
    both = pd.concat([train[cols], test[cols]], ignore_index=True)
    both["day"] = (both.TransactionDT // DAY).astype(int)
    daily = both.groupby(["origin", "day"], observed=True).agg(
        rows=("TransactionAmt", "size"), amt_median=("TransactionAmt", "median"),
        identity_rate=("has_identity", "mean"), C9_median=("C9", "median")
    ).reset_index().sort_values("day")
    for name in ("amt_median", "identity_rate", "C9_median"):
        for days in (7, 30):
            # Compute each origin separately to preserve the unobserved gap.
            daily[f"{name}_rolling_{days}d"] = np.nan
            for origin, part in daily.groupby("origin", sort=False):
                lookup = part.set_index("day")[name].sort_index()
                daily.loc[part.index, f"{name}_rolling_{days}d"] = [
                    lookup.loc[max(lookup.index.min(), d-days+1):d].mean() for d in part.day
                ]
    daily.to_csv(out / "sliding_daily.csv", index=False)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ax, col, ylabel in zip(axes, ("amt_median", "identity_rate", "C9_median"), ("Monto mediano", "Cobertura identidad", "C9 mediana")):
        for days, style in ((7, "-"), (30, "--")):
            for origin, part in daily.groupby("origin", sort=False):
                ax.plot(part.day, part[f"{col}_rolling_{days}d"], style,
                        label=f"{days} dias" if origin == "train" else None)
        ax.axvline(train.TransactionDT.max() / DAY, color="gray", linestyle=":")
        ax.axvline(test.TransactionDT.min() / DAY, color="gray", linestyle=":")
        ax.set(ylabel=ylabel, title=f"{ylabel}: ventanas moviles")
        ax.legend()
    axes[-1].set_xlabel("Dia relativo; el espacio vacio es el hueco train/test")
    save_plot(out, "01_ventanas_7_30_dias.png")

    base = train.loc[train.TransactionDT <= bounds[0]]
    compare = []
    for origin, frame in (("valid", train.loc[(train.TransactionDT > bounds[0]) & (train.TransactionDT <= bounds[1])]),
                          ("holdout", train.loc[train.TransactionDT > bounds[1]]), ("test_unlabeled", test)):
        for col in MONITOR_COLS + ["has_identity"]:
            a = pd.to_numeric(base[col], errors="coerce").dropna().to_numpy()
            b = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy()
            if len(a) > 50000:
                a = np.random.default_rng(SEED).choice(a, 50000, replace=False)
            if len(b) > 50000:
                b = np.random.default_rng(SEED).choice(b, 50000, replace=False)
            compare.append({"period": origin, "feature": col, "ks": float(ks_2samp(a, b).statistic) if len(a) and len(b) else np.nan,
                            "psi": psi(a, b), "missing_reference": float(base[col].isna().mean()),
                            "missing_current": float(frame[col].isna().mean())})
    shift = pd.DataFrame(compare)
    shift.to_csv(out / "feature_shift.csv", index=False)
    heat = shift.pivot(index="feature", columns="period", values="ks")[["valid", "holdout", "test_unlabeled"]]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(heat.to_numpy(), cmap="YlOrRd", vmin=0, vmax=max(.2, np.nanmax(heat.to_numpy())))
    ax.set_xticks(range(len(heat.columns)), ["valid", "holdout", "test sin etiqueta"], rotation=15, ha="right")
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set(title="KS frente al train inicial; se excluye tiempo absoluto")
    fig.colorbar(im, ax=ax, label="KS")
    save_plot(out, "02_ks_train_valid_holdout_test.png")

    v01 = _v01_predictions()
    if v01 is not None:
        v01["week"] = (v01.TransactionDT // WEEK).astype(int)
        scores = []
        first_week = int(v01.week.min())
        ref = v01.loc[v01.week == first_week, "pred"].to_numpy()
        for week, group in v01.groupby("week"):
            scores.append({"week": int(week), "rows": len(group), "psi_score": psi(ref, group.pred),
                           "pr_auc_matured": safe_ap(group.isFraud, group.pred), "fraud_rate": float(group.isFraud.mean())})
        scores = pd.DataFrame(scores)
        scores.to_csv(out / "v01_score_windows.csv", index=False)
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axes[0].plot(scores.week, scores.psi_score, marker="o")
        axes[0].axhline(.25, color="#e45756", linestyle="--", label="Alerta ilustrativa 0.25")
        axes[0].set(ylabel="PSI del score", title="Cambio de scores V01 frente a primera semana evaluada")
        axes[0].legend()
        axes[1].plot(scores.week, scores.pr_auc_matured, marker="o", label="PR-AUC tras llegar etiquetas")
        axes[1].plot(scores.week, scores.fraud_rate, marker="s", label="Prevalencia")
        axes[1].set(xlabel="Semana relativa", ylabel="Proporcion / PR-AUC")
        axes[1].legend()
        save_plot(out, "03_score_y_desempeno_v01.png")
    summary = {"shift_rows": len(shift), "score_windows": 0 if v01 is None else int(v01.week.nunique()),
               "test_has_target": "isFraud" in test, "score_source": "V01 historical predictions" if v01 is not None else "unavailable"}
    manifest(out, status="complete", stage="done", **summary)
    return out, summary


def make_model(name: str, device: str, positive_weight: float):
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            objective="binary", n_estimators=220, learning_rate=.05, num_leaves=48,
            min_child_samples=120, subsample=.85, colsample_bytree=.75,
            reg_alpha=.05, reg_lambda=.20, class_weight="balanced",
            random_state=SEED, n_jobs=4, verbosity=-1,
            device_type="gpu" if device == "gpu" else "cpu",
        )
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=160, learning_rate=.05, max_depth=6, min_child_weight=20,
            subsample=.85, colsample_bytree=.8, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
            device="cuda" if device == "gpu" else "cpu", n_jobs=4,
            scale_pos_weight=positive_weight, random_state=SEED,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=160, learning_rate=.05, depth=7, l2_leaf_reg=8,
            loss_function="Logloss", auto_class_weights="Balanced",
            task_type="GPU" if device == "gpu" else "CPU", devices="0" if device == "gpu" else None,
            random_seed=SEED, verbose=False, allow_writing_files=False,
        )
    raise ValueError(name)


def fit_model(name: str, x_train, y_train):
    weight = float((y_train == 0).sum() / max(int((y_train == 1).sum()), 1))
    start = time.perf_counter()
    error = None
    devices = ("gpu", "cpu") if (os.name == "nt" or Path("/dev/nvidia0").exists()) else ("cpu",)
    for device in devices:
        try:
            model = make_model(name, device, weight)
            model.fit(x_train, y_train)
            return model, device, time.perf_counter() - start, error
        except Exception as exc:
            if device == "cpu":
                raise
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
            gc.collect()
    raise RuntimeError("unreachable")


def model_snapshot(df: pd.DataFrame, train_ix, features: list[str], name: str):
    encoder = WindowEncoder(features).fit(df.loc[train_ix])
    x_train = encoder.transform(df.loc[train_ix])
    y_train = df.loc[train_ix, "isFraud"].to_numpy(dtype=np.int8)
    model, device, seconds, fallback = fit_model(name, x_train, y_train)
    del x_train
    gc.collect()
    return {"encoder": encoder, "model": model, "device": device, "seconds": seconds, "fallback": fallback,
            "rows": len(train_ix), "first_dt": float(df.loc[train_ix, "TransactionDT"].min()),
            "last_dt": float(df.loc[train_ix, "TransactionDT"].max())}


def predict_snapshot(snapshot, df: pd.DataFrame, ix):
    x = snapshot["encoder"].transform(df.loc[ix])
    p = snapshot["model"].predict_proba(x)[:, 1]
    del x
    return np.asarray(p, dtype="float32")


def traditional_baselines(df: pd.DataFrame, bounds, features, out):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    train_ix = df.index[df.TransactionDT <= bounds[0]]
    valid_ix = df.index[(df.TransactionDT > bounds[0]) & (df.TransactionDT <= bounds[1])]
    hold_ix = df.index[df.TransactionDT > bounds[1]]
    encoder = WindowEncoder(features).fit(df.loc[train_ix])
    xtr = encoder.transform(df.loc[train_ix])
    xva = encoder.transform(df.loc[valid_ix])
    xho = encoder.transform(df.loc[hold_ix])
    ytr = df.loc[train_ix, "isFraud"].to_numpy(dtype=int)
    rows = []
    for name, estimator in (
        ("regresion_logistica", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                              StandardScaler(), LogisticRegression(max_iter=120, class_weight="balanced", solver="saga", random_state=SEED, n_jobs=4))),
        ("arbol_decision", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                         DecisionTreeClassifier(max_depth=8, min_samples_leaf=120, class_weight="balanced", random_state=SEED))),
    ):
        start = time.perf_counter()
        estimator.fit(xtr, ytr)
        seconds = time.perf_counter() - start
        for split, ix, x in (("valid", valid_ix, xva), ("holdout", hold_ix, xho)):
            pred = estimator.predict_proba(x)[:, 1]
            rows.append({"model": name, "split": split, "rows": len(ix),
                         "pr_auc": safe_ap(df.loc[ix, "isFraud"], pred), "train_seconds": seconds})
    result = pd.DataFrame(rows)
    result.to_csv(out / "traditional_baselines.csv", index=False)
    return result


def _known_before(uid: pd.Series, dt: pd.Series, start: float, ix):
    historical = set(uid.loc[dt < start].to_numpy())
    return uid.loc[ix].isin(historical).to_numpy(dtype=bool)


def _eligible_indices(df: pd.DataFrame, block, window_days):
    cutoff = block["start"] - WEEK
    floor = -np.inf if window_days == "expanding" else cutoff - int(window_days) * DAY
    ix = df.index[(df.TransactionDT < cutoff) & (df.TransactionDT >= floor)].to_numpy()
    if len(ix) < 3000 or df.loc[ix, "isFraud"].sum() < 20:
        raise ValueError(f"Insufficient historical labels for {window_days} days at {block['start']}")
    if float(df.loc[ix, "TransactionDT"].max()) >= cutoff:
        raise AssertionError("Label latency violated")
    return ix


def _triggered(history: list[dict], current_start: float, reference_ap: float, last_update_number: int, block_number: int):
    if block_number - last_update_number < 2 or not history:
        return False, "cooldown_or_first_block"
    # PSI needs no labels, but uses only the PREVIOUS completed block.
    # This branch never fired in our run: the score PSI peaked at 0.0106
    # (v01_score_windows.csv) while PR-AUC fell from 0.732 to 0.473, so the
    # degradation was concept drift with no visible covariate shift. The .25
    # threshold is the conventional value and is kept so the run stays
    # comparable, but on this signal it sits 23x above anything observed and
    # the labelled rule below is what actually protects the system. Lowering
    # it without measuring the false-alarm rate would trade a detector that
    # never fires for one that fires on noise.
    if np.isfinite(history[-1]["psi_score"]) and history[-1]["psi_score"] > .25:
        return True, "previous_score_psi_gt_025"
    matured = [h for h in history if h["end"] <= current_start - WEEK and np.isfinite(h["pr_auc"])]
    if len(matured) >= 2 and np.isfinite(reference_ap):
        if all(h["pr_auc"] < .9 * reference_ap for h in matured[-2:]):
            return True, "two_matured_pr_auc_drops_gt_10pct"
    return False, "no_alert"


def run_models():
    data, out = paths("experimentacion")
    manifest(out, status="running", stage="load", label_delay_days=7)
    df, _ = load_data(data)
    if os.getenv("PTDIA_SMOKE") == "1":
        df = df.iloc[:100000].copy().reset_index(drop=True)
    bounds = split_bounds(df)
    df = row_features(df)
    uid = candidate_uid(df)
    features = feature_list()
    if any(x in features for x in ["TransactionDT", "TransactionID", "DT_day_index", "DT_week_index", "isFraud"]):
        raise AssertionError("Forbidden predictors in V01 feature list")
    models = ["lightgbm", "xgboost", "catboost"] if os.getenv("PTDIA_SMOKE") != "1" else ["lightgbm"]
    windows = [14, 30, 45, "expanding"] if os.getenv("PTDIA_SMOKE") != "1" else [14]
    blocks = evaluation_blocks(df, bounds)
    if os.getenv("PTDIA_SMOKE") == "1":
        blocks = blocks[:2]
    traditional_baselines(df, bounds, features, out)
    manifest(out, stage="models", split_cutoffs=list(bounds), block_count=len(blocks), features=len(features))
    all_metrics = []
    fits = []
    reference = None if os.getenv("PTDIA_SMOKE") == "1" else _v01_predictions()
    static_map = None if reference is None else pd.Series(reference.pred.to_numpy(), index=reference.TransactionID).to_dict()
    train_ix = df.index[df.TransactionDT <= bounds[0]].to_numpy()
    for model_name in models:
        if model_name == "lightgbm" and static_map is not None:
            static = None
            static_device = "V01_saved_scores"
            static_seconds = 0.0
        else:
            static = model_snapshot(df, train_ix, features, model_name)
            static_device = static["device"]
            static_seconds = static["seconds"]
            fits.append({"model": model_name, "strategy": "static", "window_days": "initial_70pct",
                         "split": "initial", "block": -1, "train_rows": static["rows"], "train_seconds": static_seconds,
                         "device": static_device, "fallback": static["fallback"]})
        static_history = []
        reference_scores = None
        for block in blocks:
            ix = block["indices"]
            infer_start = time.perf_counter()
            if static_map is not None and model_name == "lightgbm":
                pred = df.loc[ix, "TransactionID"].map(static_map).to_numpy(dtype="float32")
                if np.isnan(pred).any():
                    raise AssertionError("V01 reference predictions missing rows")
            else:
                pred = predict_snapshot(static, df, ix)
            infer_seconds = time.perf_counter() - infer_start
            known = _known_before(uid, df.TransactionDT, block["start"], ix)
            rows = prediction_rows(df, ix, known, pred, model_name, "static", "initial_70pct", block)
            append_predictions(out, rows)
            psi_score = 0.0 if reference_scores is None else psi(reference_scores, pred)
            reference_scores = pred if reference_scores is None else reference_scores
            metric = block_metrics(rows, seconds=static_seconds if not static_history else 0,
                                   updated=False, device=static_device, psi_score=psi_score)
            metric["inference_ms_per_txn"] = 1000 * infer_seconds / len(ix) if static is not None else np.nan
            all_metrics.append(metric)
            static_history.append({"end": block["end"], "pr_auc": safe_ap(rows.isFraud, pred), "psi_score": psi_score})
        del static
        gc.collect()

        for window_days in windows:
            for strategy in ("periodic", "alert"):
                active = None
                history = []
                initial_ap = np.nan
                reference_scores = None
                last_update_number = -99
                for block_number, block in enumerate(blocks):
                    if strategy == "periodic":
                        update, reason = True, "weekly"
                    elif active is None:
                        update, reason = True, "initial_window_model"
                    else:
                        update, reason = _triggered(history, block["start"], initial_ap, last_update_number, block_number)
                    seconds = 0.0
                    if update:
                        eligible = _eligible_indices(df, block, window_days)
                        active = model_snapshot(df, eligible, features, model_name)
                        seconds = active["seconds"]
                        last_update_number = block_number
                        fits.append({"model": model_name, "strategy": strategy, "window_days": window_days,
                                     "split": block["split"], "block": block["block"], "train_rows": active["rows"],
                                     "train_first_dt": active["first_dt"], "train_last_dt": active["last_dt"],
                                     "available_before": block["start"] - WEEK, "train_seconds": seconds,
                                     "device": active["device"], "fallback": active["fallback"], "reason": reason})
                    ix = block["indices"]
                    infer_start = time.perf_counter()
                    pred = predict_snapshot(active, df, ix)
                    infer_seconds = time.perf_counter() - infer_start
                    known = _known_before(uid, df.TransactionDT, block["start"], ix)
                    rows = prediction_rows(df, ix, known, pred, model_name, strategy, window_days, block)
                    append_predictions(out, rows)
                    psi_score = 0.0 if reference_scores is None else psi(reference_scores, pred)
                    reference_scores = pred if reference_scores is None else reference_scores
                    ap = safe_ap(rows.isFraud, pred)
                    if not np.isfinite(initial_ap) and np.isfinite(ap):
                        initial_ap = ap
                    metric = block_metrics(rows, seconds=seconds, updated=update,
                                           device=active["device"], psi_score=psi_score)
                    metric["inference_ms_per_txn"] = 1000 * infer_seconds / len(ix)
                    all_metrics.append(metric)
                    history.append({"end": block["end"], "pr_auc": ap, "psi_score": psi_score})
                    if len(all_metrics) % 4 == 0:
                        pd.DataFrame(all_metrics).to_csv(out / "block_metrics.csv", index=False)
                        pd.DataFrame(fits).to_csv(out / "fit_log.csv", index=False)
                        manifest(out, stage="models", completed_rows=len(all_metrics), last_model=model_name,
                                 last_strategy=strategy, last_window=str(window_days))
                del active
                gc.collect()
        print(f"Finished {model_name}")

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(out / "block_metrics.csv", index=False)
    fit_log = pd.DataFrame(fits)
    fit_log.to_csv(out / "fit_log.csv", index=False)
    # Paired lines preserve week order and make window-size effects visible.
    for model_name in models:
        selected = metrics[(metrics.model == model_name) & (metrics.split == "holdout")]
        fig, ax = plt.subplots(figsize=(10, 5))
        for (strategy, window), part in selected.groupby(["strategy", "window_days"], dropna=False):
            ax.plot(part.block, part.pr_auc, marker="o", linewidth=1.2, label=f"{strategy} {window}")
        ax.set(xlabel="Bloque futuro de siete dias", ylabel="PR-AUC", title=f"{model_name}: modelos y ventanas en holdout")
        ax.legend(ncol=2, fontsize=8)
        save_plot(out, f"01_pr_auc_{model_name}.png")
    hold = metrics[metrics.split == "holdout"]
    summary = hold.groupby(["model", "strategy", "window_days"], dropna=False).agg(
        mean_pr_auc=("pr_auc", "mean"), mean_cost_proxy=("cost_per_txn_proxy", "mean"),
        updates=("updated", "sum"), train_seconds=("train_seconds", "sum"), blocks=("block", "size")
    ).reset_index()
    summary.to_csv(out / "holdout_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy, part in summary.groupby("strategy"):
        ax.scatter(part.updates, part.mean_cost_proxy, label=strategy, s=50)
    ax.set(xlabel="Actualizaciones en holdout", ylabel="Costo ilustrativo por transaccion", title="Costo y frecuencia de actualizacion")
    ax.legend()
    save_plot(out, "02_costo_vs_actualizaciones.png")
    result = {"status": "complete", "stage": "done", "models": models, "windows": windows,
              "rows": len(df), "blocks": len(blocks), "fit_count": len(fit_log),
              "prediction_rows": int(metrics.rows.sum()), "source_v01_exact": static_map is not None}
    manifest(out, **result)
    return out, result


def _model_outputs():
    here = Path(__file__).resolve().parent
    candidates = [here / "kaggle" / "experimentacion" / "outputs" / "predictions.csv.gz"]
    candidates += list(Path("/kaggle/input").glob("**/predictions.csv.gz")) if Path("/kaggle/input").exists() else []
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Model predictions missing; run 03_modelos_adaptacion first")


def _policy_actions(frame, low, high, daily_capacity):
    """Decisions in timestamp order, using only a fixed prior score threshold and a daily counter."""
    ordered = frame.sort_values(["TransactionDT", "TransactionID"]).copy()
    score = ordered.score.to_numpy(dtype=float)
    candidates = (score >= low) & (score < high)
    days = (ordered.TransactionDT // DAY).astype(int)
    order = pd.Series(candidates, index=ordered.index).groupby(days).cumsum().to_numpy()
    action = np.full(len(ordered), "approve", dtype="<U8")
    action[candidates & (order <= daily_capacity)] = "review"
    action[score >= high] = "escalate"
    ordered["action"] = action
    return ordered


def _policy_metrics(decisions, review_effectiveness=.8, update_cost=0.0, updates=0,
                    review_cost=2.0, false_escalation_cost=10.0):
    y = decisions.isFraud.to_numpy(dtype=bool)
    amount = decisions.TransactionAmt.to_numpy(dtype=float)
    action = decisions.action.to_numpy()
    loss = 4.41 * amount
    approved = action == "approve"
    reviewed = action == "review"
    escalated = action == "escalate"
    # Escalation means a temporary hold and manual confirmation. Assumed
    # prevention effectiveness is 95%, not a guaranteed outcome.
    direct = np.where(y & approved, loss, 0.0)
    direct += np.where(reviewed, review_cost + np.where(y, (1-review_effectiveness)*loss, 0.0), 0.0)
    direct += np.where(escalated, np.where(y, .05*loss, false_escalation_cost), 0.0)
    total = float(direct.sum() + update_cost*updates)
    legit = ~y
    return {
        "rows": len(y), "fraud_count": int(y.sum()), "cost_total": total,
        "cost_per_txn": total / max(len(y), 1), "reward_total": -total,
        "review_count": int(reviewed.sum()), "review_rate": float(reviewed.mean()),
        "escalate_count": int(escalated.sum()), "approve_count": int(approved.sum()),
        "fraud_approved": int((y & approved).sum()),
        "fraud_referred": int((y & (reviewed | escalated)).sum()),
        "fraud_referred_rate": float((y & (reviewed | escalated)).sum() / y.sum()) if y.any() else np.nan,
        "false_escalations_per_1000_legit": 1000*float((legit & escalated).sum() / legit.sum()) if legit.any() else np.nan,
        "updates": int(updates), "update_cost_total": float(update_cost*updates),
    }


def _optimize_thresholds(valid, daily_capacity):
    scores = valid.score.to_numpy(dtype=float)
    low_options = np.unique(np.quantile(scores, [.65, .75, .82, .88, .93]))
    high_options = np.unique(np.quantile(scores, [.96, .98, .99, .995]))
    best = None
    for low in low_options:
        for high in high_options:
            if low >= high:
                continue
            decisions = _policy_actions(valid, low, high, daily_capacity)
            metric = _policy_metrics(decisions)
            row = {"review_threshold": float(low), "escalate_threshold": float(high), **metric}
            if best is None or row["cost_per_txn"] < best["cost_per_txn"]:
                best = row
    if best is None:
        raise ValueError("No feasible policy thresholds")
    return best


def run_policy():
    data, out = paths("sistema_final")
    manifest(out, status="running", stage="load")
    path = _model_outputs()
    predictions = pd.read_csv(path, compression="gzip", low_memory=False)
    predictions["window_days"] = predictions.window_days.astype(str)
    predictions["uid_known"] = predictions.uid_known.astype(str).str.lower().eq("true")
    source = pd.read_csv(data / "train_transaction.csv", usecols=["TransactionDT"])
    train_cut = float(source.TransactionDT.quantile(.70))
    holdout_cut = float(source.TransactionDT.quantile(.85))
    day_counts = source.loc[source.TransactionDT <= train_cut].groupby(source.TransactionDT // DAY).size()
    daily_capacity = max(1, int(np.floor(.05 * day_counts.median())))
    del source
    gc.collect()

    fit_path = path.with_name("fit_log.csv")
    fit_log = pd.read_csv(fit_path) if fit_path.exists() else pd.DataFrame()
    groups = predictions.groupby(["model", "strategy", "window_days"], sort=False)
    results = []
    thresholds = []
    decisions_selected = []
    for (model, strategy, window_days), frame in groups:
        valid = frame.loc[frame.split == "valid"].copy()
        hold = frame.loc[frame.split == "holdout"].copy()
        if valid.empty or hold.empty:
            continue
        choice = _optimize_thresholds(valid, daily_capacity)
        thresholds.append({"model": model, "strategy": strategy, "window_days": window_days,
                           "daily_capacity": daily_capacity, "review_threshold": choice["review_threshold"],
                           "escalate_threshold": choice["escalate_threshold"],
                           "valid_cost_per_txn": choice["cost_per_txn"]})
        decided = _policy_actions(hold, choice["review_threshold"], choice["escalate_threshold"], daily_capacity)
        updates = 0
        if not fit_log.empty and strategy != "static":
            mask = ((fit_log.model == model) & (fit_log.strategy == strategy) &
                    (fit_log.window_days.astype(str) == window_days) & (fit_log.split == "holdout"))
            updates = int(mask.sum())
        for q in (.6, .8, 1.0):
            for update_cost in (0.0, 100.0, 500.0):
                for fp_cost in (5.0, 10.0, 25.0):
                    metric = _policy_metrics(decided, review_effectiveness=q, update_cost=update_cost,
                                             updates=updates, false_escalation_cost=fp_cost)
                    results.append({"model": model, "strategy": strategy, "window_days": window_days,
                                    "scenario": "model", "review_effectiveness": q,
                                    "update_cost": update_cost, "false_escalation_cost": fp_cost, **metric})
        if model == "lightgbm" and strategy == "static":
            decisions_selected.append(decided)
        # Release the many scenario columns before the next configuration.
        del valid, hold, decided
        gc.collect()

    if not decisions_selected:
        raise ValueError("V01 static decisions unavailable")
    static_hold = decisions_selected[0]
    no_model = static_hold.copy()
    no_model["action"] = "approve"
    baseline = _policy_metrics(no_model, review_effectiveness=.8)
    results.append({"model": "none", "strategy": "approve_all", "window_days": "none", "scenario": "no_model",
                    "review_effectiveness": .8, "update_cost": 0.0, "false_escalation_cost": 10.0, **baseline})

    # The recalibration action leaves the V01 model fixed. Thresholds are
    # reselected only from already mature prior out-of-sample predictions.
    static_all = predictions[(predictions.model == "lightgbm") & (predictions.strategy == "static")].copy()
    recalibrated_parts = []
    recalibration_log = []
    for block, current in static_all.loc[static_all.split == "holdout"].groupby("block", sort=True):
        start = holdout_cut + int(block) * WEEK
        available = static_all.loc[(static_all.TransactionDT < start - WEEK) &
                                   (static_all.TransactionDT >= start - 37*DAY)]
        if len(available) >= 3000 and available.isFraud.sum() >= 20:
            chosen = _optimize_thresholds(available, daily_capacity)
            low, high, reason = chosen["review_threshold"], chosen["escalate_threshold"], "matured_last_30d"
        else:
            row = next(t for t in thresholds if t["model"] == "lightgbm" and t["strategy"] == "static")
            low, high, reason = row["review_threshold"], row["escalate_threshold"], "initial_validation"
        recalibrated_parts.append(_policy_actions(current, low, high, daily_capacity))
        recalibration_log.append({"block": block, "known_before": start-WEEK, "reason": reason,
                                  "review_threshold": low, "escalate_threshold": high})
    recalibrated = pd.concat(recalibrated_parts, ignore_index=True)
    # Adjacent weekly blocks may split the same calendar day. Enforce the
    # shared daily review budget once across their combined decisions.
    recalibrated.sort_values(["TransactionDT", "TransactionID"], inplace=True)
    review_so_far = recalibrated.action.eq("review").groupby(recalibrated.TransactionDT // DAY).cumsum()
    recalibrated.loc[recalibrated.action.eq("review") & (review_so_far > daily_capacity), "action"] = "approve"
    metric = _policy_metrics(recalibrated, review_effectiveness=.8)
    results.append({"model": "lightgbm", "strategy": "recalibrate_thresholds", "window_days": "none",
                    "scenario": "model", "review_effectiveness": .8, "update_cost": 0.0,
                    "false_escalation_cost": 10.0, **metric})
    pd.DataFrame(recalibration_log).to_csv(out / "recalibration_log.csv", index=False)

    result = pd.DataFrame(results)
    result.to_csv(out / "policy_scenarios.csv", index=False)
    pd.DataFrame(thresholds).to_csv(out / "policy_thresholds_valid.csv", index=False)
    static_hold.to_csv(out / "v01_policy_decisions.csv.gz", index=False, compression="gzip")
    # Actual human review burden and false escalation by available UID segment.
    social = []
    for name, part in (("known", static_hold.loc[static_hold.uid_known]),
                       ("unknown", static_hold.loc[~static_hold.uid_known])):
        social.append({"uid_segment": name, **_policy_metrics(part, review_effectiveness=.8)})
    pd.DataFrame(social).to_csv(out / "social_uid_proxy.csv", index=False)

    central = result[(result.review_effectiveness == .8) & (result.update_cost == 100) &
                     (result.false_escalation_cost == 10)].copy()
    central = pd.concat([central, result[result.strategy.isin(["approve_all", "recalibrate_thresholds"])]], ignore_index=True)
    central.sort_values("cost_per_txn", inplace=True)
    central.to_csv(out / "central_comparison.csv", index=False)
    anchors = central.loc[central.strategy.isin(["approve_all", "static", "recalibrate_thresholds"])]
    anchors = anchors.loc[(anchors.strategy != "static") | (anchors.model == "lightgbm")]
    best = pd.concat([central.head(9), anchors]).drop_duplicates(
        ["model", "strategy", "window_days"]
    ).sort_values("cost_per_txn", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = [f"{r.model}/{r.strategy}/{r.window_days}" for r in best.itertuples()]
    ax.barh(labels, best.cost_per_txn, color="#4c78a8")
    ax.set(xlabel="USD ilustrativos por transaccion", title="Sin modelo frente a politicas: escenario central")
    save_plot(out, "01_costo_escenarios.png")

    selected = result[(result.model == "lightgbm") & (result.strategy.isin(["static", "periodic", "alert"])) &
                      (result.window_days.isin(["initial_70pct", "30"])) &
                      (result.false_escalation_cost == 10)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for (strategy, window), part in selected.groupby(["strategy", "window_days"]):
        pivot = part.pivot_table(index="update_cost", columns="review_effectiveness", values="cost_per_txn")
        for q in pivot.columns:
            ax.plot(pivot.index, pivot[q], marker="o", label=f"{strategy}/{window}, revision {q:.0%}")
    ax.set(xlabel="Costo por actualizacion (USD supuestos)", ylabel="Costo por transaccion (USD)",
           title="Sensibilidad a revision y reentrenamiento")
    ax.legend(fontsize=8)
    save_plot(out, "02_sensibilidad.png")

    summary = {"status": "complete", "stage": "done", "daily_review_capacity": daily_capacity,
               "configurations": len(thresholds), "scenarios": len(result),
               "central_no_model_cost_per_txn": baseline["cost_per_txn"],
               "central_v01_cost_per_txn": float(central.loc[(central.model == "lightgbm") &
                                                           (central.strategy == "static"), "cost_per_txn"].iloc[0]),
               "costs_are": "illustrative_USD", "review_effectiveness": [0.6, 0.8, 1.0]}
    manifest(out, **summary)
    return out, summary
