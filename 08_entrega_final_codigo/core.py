"""Single source of truth for temporal features, models and policy accounting.

All fitting functions receive an explicit historical training frame. Selection
stages never load holdout labels. The final stage evaluates a frozen policy
on a later historical period.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DAY = 86_400
WEEK = 7 * DAY
SEED = 42
CAT_COLS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
            "DeviceType", "DeviceInfo"] + [f"M{i}" for i in range(1, 10)]
LAG_COLS = ["uid_prior_count", "uid_prior_amount", "uid_prior_mean",
            "uid_seconds_since_prior", "uid_amount_vs_prior_mean"]
SCENARIOS = {
    "optimista_100_100": (1.00, 1.00),
    "favorable_90_98": (0.90, 0.98),
    "adverso_80_95": (0.80, 0.95),
    "severo_60_90": (0.60, 0.90),
}
LOW_Q = (.65, .75, .82, .88, .93)
HIGH_Q = (.96, .98, .99, .995)
GRIDS = {
    "logistic": [{"C": c} for c in (.1, 1.0)],
    "tree": [{"max_depth": d, "min_samples_leaf": n}
             for d in (6, 10) for n in (100, 300)],
    "lightgbm": [{"num_leaves": leaves, "min_child_samples": n}
                 for leaves in (31, 63) for n in (100, 200)],
    "xgboost": [{"max_depth": d, "min_child_weight": n}
                for d in (4, 6) for n in (10, 30)],
}


def json_dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def data_dir(root: Path) -> Path:
    choices = [Path("/kaggle/input/ieee-fraud-detection"),
               Path("/kaggle/input/competitions/ieee-fraud-detection"),
               root.parent / "ieee-fraud-detection"]
    for path in choices:
        if (path / "train_transaction.csv").exists():
            return path
    raise FileNotFoundError("IEEE-CIS CSVs unavailable; attach the Kaggle competition source")


def output_dir(root: Path, stage: str) -> Path:
    path = Path("/kaggle/working") if Path("/kaggle/working").exists() else root / "outputs" / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_train(root: Path) -> pd.DataFrame:
    data = data_dir(root)
    txn = pd.read_csv(data / "train_transaction.csv", low_memory=False)
    ident = pd.read_csv(data / "train_identity.csv", low_memory=False)
    ident.rename(columns=lambda c: c.replace("id-", "id_") if c.startswith("id-") else c, inplace=True)
    if txn.TransactionID.duplicated().any() or ident.TransactionID.duplicated().any():
        raise ValueError("TransactionID must be unique")
    df = txn.merge(ident, on="TransactionID", how="left", validate="one_to_one", indicator=True)
    df["has_identity"] = df["_merge"].eq("both").astype("int8")
    df.drop(columns="_merge", inplace=True)
    return df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)


def cuts(df: pd.DataFrame) -> tuple[float, float]:
    return tuple(float(x) for x in df.TransactionDT.quantile([.70, .85]).to_numpy())


def relative_day(df: pd.DataFrame) -> pd.Series:
    return ((df.TransactionDT - df.TransactionDT.min()) // DAY).astype(int)


def own_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build row-local and prior-event features without future labels or rows."""
    out = df.copy()
    day = (out.TransactionDT // DAY).astype(int)
    out["DT_hour"] = ((out.TransactionDT // 3600) % 24).astype("int8")
    out["TransactionAmt_log1p"] = np.log1p(out.TransactionAmt.clip(lower=0)).astype("float32")
    for family in ("V", "D", "id_"):
        cols = [c for c in out if c.startswith(family) and
                (family == "id_" or c[len(family):].isdigit())]
        out[f"missing_{family.replace('_', '')}_count"] = out[cols].isna().sum(axis=1).astype("int16")
    core = [c for c in ("card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2") if c in out]
    out["missing_core_count"] = out[core].isna().sum(axis=1).astype("int8")
    for col in ("D1", "D2", "D10", "D15"):
        if col in out:
            out[f"{col}n"] = (out[col] - day).astype("float32")
    uid_parts = out[["card1", "card2", "card3", "card5", "addr1", "addr2"]].astype("string").fillna("NA")
    uid_parts = uid_parts.copy()
    uid_parts["D1_anchor"] = (out.D1 - day).round().astype("string").fillna("NA")
    out["uid"] = pd.util.hash_pandas_object(uid_parts, index=False).astype("uint64")
    grouped = out.groupby("uid", sort=False)
    prior_count = grouped.cumcount().astype("float32")
    prior_sum = grouped.TransactionAmt.cumsum() - out.TransactionAmt
    prior_mean = prior_sum.div(prior_count.replace(0, np.nan))
    prior_amount = grouped.TransactionAmt.shift(1)
    prior_time = grouped.TransactionDT.shift(1)
    out["uid_prior_count"] = prior_count
    out["uid_prior_amount"] = prior_amount.astype("float32")
    out["uid_prior_mean"] = prior_mean.astype("float32")
    out["uid_seconds_since_prior"] = (out.TransactionDT - prior_time).astype("float32")
    out["uid_amount_vs_prior_mean"] = (out.TransactionAmt - prior_mean).astype("float32")
    return out


def select_features(train: pd.DataFrame) -> list[str]:
    """Select numeric columns and top 80 V columns using training rows only."""
    excluded = {"TransactionID", "TransactionDT", "isFraud", "uid"}
    numeric = [c for c in train.select_dtypes(include="number").columns if c not in excluded]
    numeric = [c for c in numeric if train[c].notna().sum() >= 100 and train[c].nunique(dropna=True) > 1]
    vcols = [c for c in numeric if c.startswith("V") and c[1:].isdigit()]
    if len(vcols) > 80:
        corr = train[vcols].corrwith(train.isFraud.astype(float)).abs().fillna(0)
        keep = set(corr.sort_values(ascending=False, kind="stable").head(80).index)
        numeric = [c for c in numeric if c not in vcols or c in keep]
    encoded = [f"{c}_{suffix}" for c in CAT_COLS if c in train for suffix in ("label", "freq")]
    return numeric + encoded + ["TransactionAmt_outlier_iqr"]


class WindowEncoder:
    def fit(self, train: pd.DataFrame, features: list[str]):
        self.features = features
        q1, q3 = train.TransactionAmt.quantile([.25, .75])
        self.low = float(q1 - 1.5 * (q3 - q1))
        self.high = float(q3 + 1.5 * (q3 - q1))
        self.maps = {}
        self.freqs = {}
        for col in CAT_COLS:
            if col in train:
                s = train[col].astype("string").fillna("__MISSING__")
                self.maps[col] = {v: i for i, v in enumerate(s.unique())}
                self.freqs[col] = (s.value_counts() / len(s)).to_dict()
        self.fit_max_dt = float(train.TransactionDT.max())
        return self

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        values = {}
        for name in self.features:
            if name == "TransactionAmt_outlier_iqr":
                values[name] = ((rows.TransactionAmt < self.low) | (rows.TransactionAmt > self.high)).astype("float32")
            elif name.endswith("_label") and name[:-6] in self.maps:
                col = name[:-6]
                values[name] = rows[col].astype("string").fillna("__MISSING__").map(self.maps[col]).fillna(-1).astype("float32")
            elif name.endswith("_freq") and name[:-5] in self.freqs:
                col = name[:-5]
                values[name] = rows[col].astype("string").fillna("__MISSING__").map(self.freqs[col]).fillna(0).astype("float32")
            else:
                values[name] = pd.to_numeric(rows[name], errors="coerce").astype("float32")
        return pd.DataFrame(values, index=rows.index).replace([np.inf, -np.inf], np.nan)


def make_model(name: str, params: dict, y: np.ndarray):
    if name == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=params["C"], class_weight="balanced",
                                                max_iter=300, solver="lbfgs", random_state=SEED))
    if name == "tree":
        return make_pipeline(SimpleImputer(strategy="median"),
                             DecisionTreeClassifier(**params, class_weight="balanced", random_state=SEED))
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(objective="binary", n_estimators=200, learning_rate=.05,
                                  subsample=.85, colsample_bytree=.8, class_weight="balanced",
                                  random_state=SEED, n_jobs=2, verbosity=-1, **params)
    if name == "xgboost":
        from xgboost import XGBClassifier
        weight = float((y == 0).sum() / max((y == 1).sum(), 1))
        return XGBClassifier(n_estimators=200, learning_rate=.05, subsample=.85,
                             colsample_bytree=.8, objective="binary:logistic",
                             eval_metric="aucpr", tree_method="hist", n_jobs=2,
                             scale_pos_weight=weight, random_state=SEED, **params)
    raise ValueError(name)


def fit_snapshot(df: pd.DataFrame, train_mask: np.ndarray, name: str, params: dict,
                 available_before: float):
    train = df.loc[train_mask]
    if len(train) < 3_000 or train.isFraud.sum() < 20:
        raise ValueError(f"Too few historical rows for {name}: {len(train)}")
    if not float(train.TransactionDT.max()) < available_before:
        raise AssertionError("Training includes labels unavailable at prediction time")
    features = select_features(train)
    encoder = WindowEncoder().fit(train, features)
    x = encoder.transform(train)
    y = train.isFraud.to_numpy(dtype=np.int8)
    model = make_model(name, params, y)
    model.fit(x, y)
    return {"model": model, "encoder": encoder, "features": features,
            "fit_max_dt": encoder.fit_max_dt, "train_rows": len(train)}


def score_snapshot(snapshot, rows: pd.DataFrame) -> np.ndarray:
    return snapshot["model"].predict_proba(snapshot["encoder"].transform(rows))[:, 1].astype("float32")


def safe_ap(y, score) -> float:
    return float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def metrics(y, score, threshold=.5) -> dict:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    pred = score >= threshold
    return {"rows": len(y), "fraud": int(y.sum()), "pr_auc": safe_ap(y, score),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "brier": float(brier_score_loss(y, np.clip(score, 0, 1)))}


def daily_review_cap(train: pd.DataFrame, fraction=.05) -> int:
    counts = train.groupby((train.TransactionDT // DAY).astype(int)).size()
    return max(1, int(np.floor(fraction * counts.median())))


def policy_actions(frame: pd.DataFrame, low: float, high: float, cap: int) -> np.ndarray:
    """Sequential online actions: 0 approve, 1 review, 2 direct escalation."""
    score = frame.score.to_numpy(dtype=float)
    day = (frame.TransactionDT.to_numpy() // DAY).astype(int)
    candidate = (score >= low) & (score < high)
    running = np.cumsum(candidate, dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1]
    stops = np.r_[starts[1:], len(day)]
    before = np.r_[0, running[stops[:-1] - 1]]
    rank = running - np.repeat(before, stops - starts)
    action = np.zeros(len(score), dtype=np.int8)
    action[candidate & (rank <= cap)] = 1
    action[score >= high] = 2
    return action


def policy_cost(frame: pd.DataFrame, action: np.ndarray, review_rate: float,
                escalation_rate: float) -> dict:
    """Expected cost; labels are used only after actions are assigned."""
    fraud = frame.isFraud.to_numpy(dtype=bool)
    amount = frame.TransactionAmt.to_numpy(dtype=float)
    review = action == 1
    direct = action == 2
    approve = action == 0
    reviewed_fraud = review & fraud
    direct_fraud = direct & fraud
    approved_fraud = approve & fraud
    review_charge = .65 * int(review.sum())
    direct_charge = 4.0 * int(direct.sum())
    handoffs = review_rate * int(reviewed_fraud.sum())
    handoff_charge = 4.0 * handoffs
    unstopped_amount = (amount[approved_fraud].sum()
                        + (1 - review_rate * escalation_rate) * amount[reviewed_fraud].sum()
                        + (1 - escalation_rate) * amount[direct_fraud].sum())
    loss = 5.75 * unstopped_amount
    total = float(review_charge + direct_charge + handoff_charge + loss)
    no_model = float(5.75 * amount[fraud].sum())
    avoided = float(no_model - total)
    assert np.isclose(total - no_model, -avoided)
    return {"cost_total": total, "cost_per_txn": total / len(frame),
            "approve_all_total": no_model, "avoided_loss_total": avoided,
            "review_count": int(review.sum()), "direct_escalations": int(direct.sum()),
            "expected_review_handoffs": handoffs, "approved_fraud": int(approved_fraud.sum()),
            "fraud_not_stopped_amount": float(unstopped_amount)}


def candidate_grid(frame: pd.DataFrame, cap: int):
    score = frame.score.to_numpy(dtype=float)
    for low in np.unique(np.quantile(score, LOW_Q)):
        for high in np.unique(np.quantile(score, HIGH_Q)):
            if low < high:
                yield float(low), float(high), policy_actions(frame, low, high, cap)


def config_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
