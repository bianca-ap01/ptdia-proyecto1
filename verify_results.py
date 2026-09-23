"""Checks for the four downloaded Kaggle stages; run after all stages complete."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "kaggle"

for name in ("01_eda", "02_monitoreo_drift", "03_modelos_adaptacion", "04_politica_juego"):
    nbformat.validate(nbformat.read(ROOT / f"{name}.ipynb", as_version=4))

stages = ("eda", "monitoreo", "experimentacion", "sistema_final")
summaries = {}
for stage in stages:
    summary = json.loads((OUT / stage / "outputs" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete", (stage, summary)
    summaries[stage] = summary

eda = summaries["eda"]
assert eda["train_rows"] == 590540 and eda["test_rows"] == 506691
assert eda["test_has_target"] is False
shift = pd.read_csv(OUT / "monitoreo" / "outputs" / "feature_shift.csv")
assert "TransactionDT" not in set(shift.feature)
assert set(shift.period) == {"valid", "holdout", "test_unlabeled"}

model_out = OUT / "experimentacion" / "outputs"
fits = pd.read_csv(model_out / "fit_log.csv")
adapt = fits.loc[fits.strategy != "static"]
assert (adapt.train_last_dt < adapt.available_before).all(), "A fit used unavailable labels"
assert set(fits.model) == {"lightgbm", "xgboost", "catboost"}
assert set(adapt.window_days.astype(str)) == {"14", "30", "45", "expanding"}
assert set(adapt.strategy) == {"periodic", "alert"}
metrics = pd.read_csv(model_out / "block_metrics.csv")
assert int(metrics.rows.sum()) == summaries["experimentacion"]["prediction_rows"]
assert (metrics.rows > 0).all() and metrics.pr_auc.between(0, 1).all()
assert (model_out / "predictions.csv.gz").stat().st_size > 0

policy_out = OUT / "sistema_final" / "outputs"
choices = pd.read_csv(policy_out / "policy_thresholds_valid.csv")
assert len(choices) == 27
decisions = pd.read_csv(policy_out / "v01_policy_decisions.csv.gz")
assert set(decisions.action) <= {"approve", "review", "escalate"}
capacity = summaries["sistema_final"]["daily_review_capacity"]
daily_reviews = decisions.action.eq("review").groupby(decisions.TransactionDT // 86400).sum()
assert (daily_reviews <= capacity).all(), "Daily review capacity exceeded"

central = pd.read_csv(policy_out / "central_comparison.csv")
v01 = central.loc[(central.model == "lightgbm") & (central.strategy == "static")].iloc[0]
assert int(v01.rows) == len(decisions)
fraud = decisions.isFraud.to_numpy(dtype=bool)
loss = 4.41 * decisions.TransactionAmt.to_numpy(dtype=float)
action = decisions.action.to_numpy()
cost = np.where(fraud & (action == "approve"), loss, 0.0)
cost += np.where(action == "review", 2 + np.where(fraud, .2 * loss, 0.0), 0.0)
cost += np.where(action == "escalate", np.where(fraud, .05 * loss, 10.0), 0.0)
assert np.isclose(cost.sum() / len(decisions), v01.cost_per_txn), "V01 cost mismatch"
assert "approve_all" in set(central.strategy)

print("PASS: 4 notebooks, 4 complete stages, temporal cutoffs, model coverage, capacity and cost")
