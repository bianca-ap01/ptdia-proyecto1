import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


BASE = Path("p1/03_outputs/v01_baseline_temporal")
OUT = Path("p1/03_outputs/v06_policy_robustness")
OUT.mkdir(exist_ok=True)


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


def load_predictions():
    valid = pd.read_csv(BASE / "baseline_valid_predictions.csv")
    holdout = pd.read_csv(BASE / "baseline_holdout_predictions.csv")
    valid["split"] = "valid"
    holdout["split"] = "holdout"
    return valid, holdout


def apply_policy(df, review_threshold, escalate_threshold, score_col="pred"):
    out = df.copy()
    out["decision"] = "approve"
    out.loc[out[score_col] >= review_threshold, "decision"] = "review"
    out.loc[out[score_col] >= escalate_threshold, "decision"] = "escalate"
    return out


def evaluate_policy(df, scenario):
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


def threshold_grid(valid, max_review_rate):
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
    for rt, et in threshold_grid(valid, scenario["max_review_rate"]):
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
    metrics = evaluate_policy(scored, scenario)
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


def main():
    valid, holdout = load_predictions()
    policy_rows = []
    segment_rows = []

    for scenario in SCENARIOS:
        global_grid = optimize_global(valid, scenario)
        if global_grid.empty:
            raise RuntimeError(f"No global policy found for scenario {scenario['scenario']}")
        global_best = global_grid.iloc[0]
        for split, df in [("valid", valid), ("holdout", holdout)]:
            scored = apply_policy(df, global_best["review_threshold"], global_best["escalate_threshold"])
            policy_rows.append({
                "scenario": scenario["scenario"],
                "policy_type": "global",
                "split": split,
                "review_threshold": float(global_best["review_threshold"]),
                "escalate_threshold": float(global_best["escalate_threshold"]),
                **evaluate_policy(scored, scenario),
            })

        segment_grid = optimize_segment(valid, scenario)
        if not segment_grid.empty:
            segment_best = segment_grid.iloc[0]
            for split, df in [("valid", valid), ("holdout", holdout)]:
                scored = apply_segment_policy(df, segment_best)
                segment_rows.append({
                    "scenario": scenario["scenario"],
                    "policy_type": "segmented_uid",
                    "split": split,
                    "known_review_threshold": float(segment_best["known_review_threshold"]),
                    "known_escalate_threshold": float(segment_best["known_escalate_threshold"]),
                    "unknown_review_threshold": float(segment_best["unknown_review_threshold"]),
                    "unknown_escalate_threshold": float(segment_best["unknown_escalate_threshold"]),
                    **evaluate_policy(scored, scenario),
                })

    policy_results = pd.DataFrame(policy_rows)
    segment_results = pd.DataFrame(segment_rows)
    combined = pd.concat([policy_results, segment_results], ignore_index=True, sort=False)
    calibration_summary, calibration_bins = calibration_table(valid, holdout)

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
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
