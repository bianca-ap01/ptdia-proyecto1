"""Reprice saved validation predictions without consulting holdout labels.

The external dollar figures are scenario assumptions, not measured IEEE-CIS costs.
Run with: python final/cost_sensitivity.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "kaggle/experimentacion/outputs/predictions.csv.gz"
OUTPUT = ROOT / "cost_sensitivity_valid.csv"
DAY = 86_400
_capacity_values = pd.read_csv(
    ROOT / "kaggle/sistema_final/outputs/policy_thresholds_valid.csv",
    usecols=["daily_capacity"],
).daily_capacity.unique()
if len(_capacity_values) != 1:
    raise ValueError("Saved policy run must have one training-derived daily capacity")
CAPACITY = int(_capacity_values[0])

# Forrester TEI: https://tei.forrester.com/go/mastercard/decisionintelligenceandrulesservices/
# USD 0.65/alert reviewed; USD 4 follow-up applies to a subset of false alerts.
# LexisNexis 2025: https://risk.lexisnexis.com/insights-resources/research/US-CA-true-cost-of-fraud-study
# USD 5.75 total cost per USD 1 direct fraud loss in US financial services.
# The latter may already include operating costs, so these are sensitivity
# specifications, not additive estimates of actual business expenditure.
SCENARIOS = {
    "literal_proposal": (0.65, 4.0, "all", 1.0, 1.0),
    "fp_followup_residual80": (0.65, 4.0, "legitimate", 0.8, 0.95),
    "all_followup_residual60": (0.65, 4.0, "all", 0.6, 0.95),
}


def actions(score, day, low, high):
    """Return action codes 0=approve, 1=review, 2=escalate in time order."""
    candidate = (score >= low) & (score < high)
    running = np.cumsum(candidate, dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1]
    stops = np.r_[starts[1:], len(day)]
    before_day = np.r_[0, running[stops[:-1] - 1]]
    rank = running - np.repeat(before_day, stops - starts)
    action = np.zeros(len(score), dtype=np.uint8)
    action[candidate & (rank <= CAPACITY)] = 1
    action[score >= high] = 2
    return action


def evaluate(action, fraud, amount, spec):
    review_cost, followup_cost, followup_basis, review_eff, escalation_eff = spec
    reviewed = action == 1
    escalated = action == 2
    approved_fraud = fraud & (action == 0)
    fraud_reviewed = fraud & reviewed
    fraud_escalated = fraud & escalated
    followups = escalated if followup_basis == "all" else escalated & ~fraud
    loss_amount = (amount[approved_fraud].sum()
                   + (1 - review_eff) * amount[fraud_reviewed].sum()
                   + (1 - escalation_eff) * amount[fraud_escalated].sum())
    total = review_cost * reviewed.sum() + followup_cost * followups.sum() + 5.75 * loss_amount
    return float(total / len(action)), int(reviewed.sum()), int(escalated.sum()), int(approved_fraud.sum())


def main():
    usecols = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud",
               "score", "model", "strategy", "window_days", "split"]
    data = pd.read_csv(PREDICTIONS, compression="gzip", usecols=usecols,
                       dtype={"window_days": str}, low_memory=False)
    valid = data.loc[data.split == "valid"].drop(columns="split")
    if valid.empty or valid[["TransactionAmt", "score"]].isna().any().any():
        raise ValueError("Validation predictions are absent or incomplete")
    rows = []
    for (model, strategy, window), frame in valid.groupby(
            ["model", "strategy", "window_days"], sort=True):
        frame = frame.sort_values(["TransactionDT", "TransactionID"], kind="mergesort")
        score = frame.score.to_numpy(dtype=float)
        day = (frame.TransactionDT.to_numpy() // DAY).astype(int)
        fraud = frame.isFraud.to_numpy(dtype=bool)
        amount = frame.TransactionAmt.to_numpy(dtype=float)
        lows = np.unique(np.quantile(score, [.65, .75, .82, .88, .93]))
        highs = np.unique(np.quantile(score, [.96, .98, .99, .995]))
        best = {name: None for name in SCENARIOS}
        for low in lows:
            for high in highs:
                if low >= high:
                    continue
                action = actions(score, day, low, high)
                for name, spec in SCENARIOS.items():
                    cost, reviews, escalations, fn = evaluate(action, fraud, amount, spec)
                    candidate = (cost, float(low), float(high), reviews, escalations, fn)
                    if best[name] is None or cost < best[name][0]:
                        best[name] = candidate
        for name, chosen in best.items():
            if chosen is None:
                raise ValueError(f"No candidate thresholds for {model}/{strategy}/{window}")
            cost, low, high, reviews, escalations, fn = chosen
            rows.append({"scenario": name, "model": model, "strategy": strategy,
                         "window_days": window, "rows": len(frame),
                         "capacity_per_day": CAPACITY, "review_threshold": low,
                         "escalate_threshold": high, "valid_cost_per_txn": cost,
                         "reviews": reviews, "escalations": escalations,
                         "approved_fraud": fn})
    result = pd.DataFrame(rows).sort_values(
        ["scenario", "valid_cost_per_txn", "model", "strategy"])
    result.to_csv(OUTPUT, index=False, float_format="%.10f")
    print(result.groupby("scenario", sort=True).head(1).to_string(index=False))
    print(f"Wrote {len(result)} validation-only rows to {OUTPUT}")


if __name__ == "__main__":
    main()
