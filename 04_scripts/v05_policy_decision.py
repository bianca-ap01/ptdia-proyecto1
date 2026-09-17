import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("p1/03_outputs/v01_baseline_temporal")
OUT = Path("p1/03_outputs/v05_policy_decision")
OUT.mkdir(exist_ok=True)

COST_FRAUD_APPROVED = 100.0
COST_LEGIT_ESCALATED = 10.0
COST_REVIEW = 2.0
MAX_REVIEW_RATE = 0.05


def df_to_md(df):
    if df.empty:
        return "_Sin filas._"
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
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


def apply_policy(df, review_threshold, escalate_threshold):
    out = df.copy()
    out["decision"] = "approve"
    out.loc[out["pred"] >= review_threshold, "decision"] = "review"
    out.loc[out["pred"] >= escalate_threshold, "decision"] = "escalate"
    return out


def evaluate_policy(df):
    y = df["isFraud"].astype(int)
    decision = df["decision"]
    approved = decision == "approve"
    reviewed = decision == "review"
    escalated = decision == "escalate"

    fraud_approved = int(((y == 1) & approved).sum())
    legit_escalated = int(((y == 0) & escalated).sum())
    review_count = int(reviewed.sum())

    cost = (
        fraud_approved * COST_FRAUD_APPROVED
        + legit_escalated * COST_LEGIT_ESCALATED
        + review_count * COST_REVIEW
    )
    fraud_total = int((y == 1).sum())
    legit_total = int((y == 0).sum())
    caught_fraud = int(((y == 1) & (reviewed | escalated)).sum())
    auto_approved = int(approved.sum())

    return {
        "rows": int(len(df)),
        "fraud_total": fraud_total,
        "fraud_rate": float(y.mean()),
        "cost_total": float(cost),
        "cost_per_txn": float(cost / len(df)),
        "fraud_approved": fraud_approved,
        "fraud_caught": caught_fraud,
        "fraud_detection_rate": float(caught_fraud / fraud_total) if fraud_total else np.nan,
        "fraud_miss_rate": float(fraud_approved / fraud_total) if fraud_total else np.nan,
        "legit_escalated": legit_escalated,
        "legit_escalation_rate": float(legit_escalated / legit_total) if legit_total else np.nan,
        "review_count": review_count,
        "review_rate": float(reviewed.mean()),
        "escalate_count": int(escalated.sum()),
        "escalate_rate": float(escalated.mean()),
        "approve_count": auto_approved,
        "approve_rate": float(approved.mean()),
        "automatic_coverage": float((approved | escalated).mean()),
    }


def threshold_grid(valid):
    # Use score quantiles so the grid focuses on operationally plausible regions.
    qs_review = np.linspace(0.80, 0.99, 40)
    qs_escalate = np.linspace(0.90, 0.999, 45)
    review_thresholds = np.unique(np.quantile(valid["pred"], qs_review))
    escalate_thresholds = np.unique(np.quantile(valid["pred"], qs_escalate))
    for rt in review_thresholds:
        for et in escalate_thresholds:
            if et <= rt:
                continue
            yield float(rt), float(et)


def optimize_policy(valid):
    rows = []
    for rt, et in threshold_grid(valid):
        scored = apply_policy(valid, rt, et)
        metrics = evaluate_policy(scored)
        if metrics["review_rate"] <= MAX_REVIEW_RATE:
            rows.append({
                "review_threshold": rt,
                "escalate_threshold": et,
                **metrics,
            })
    grid = pd.DataFrame(rows)
    grid = grid.sort_values(["cost_per_txn", "fraud_miss_rate", "legit_escalation_rate"], ascending=True)
    return grid


def baseline_policies(valid):
    policies = []

    approve_all = valid.copy()
    approve_all["decision"] = "approve"
    policies.append({"policy": "approve_all", **evaluate_policy(approve_all)})

    top5 = valid.copy()
    thr = float(np.quantile(valid["pred"], 0.95))
    top5["decision"] = np.where(top5["pred"] >= thr, "review", "approve")
    policies.append({"policy": "review_top5_no_escalate", "review_threshold": thr, "escalate_threshold": np.nan, **evaluate_policy(top5)})

    top1_escalate = valid.copy()
    thr_es = float(np.quantile(valid["pred"], 0.99))
    top1_escalate["decision"] = np.where(top1_escalate["pred"] >= thr_es, "escalate", "approve")
    policies.append({"policy": "escalate_top1_only", "review_threshold": np.nan, "escalate_threshold": thr_es, **evaluate_policy(top1_escalate)})

    return pd.DataFrame(policies)


def segment_eval(df, review_threshold, escalate_threshold):
    scored = apply_policy(df, review_threshold, escalate_threshold)
    rows = []
    for segment, mask in {
        "global": np.ones(len(scored), dtype=bool),
        "uid_known": scored["uid_known"].astype(bool).to_numpy(),
        "uid_unknown": (~scored["uid_known"].astype(bool)).to_numpy(),
    }.items():
        rows.append({"segment": segment, **evaluate_policy(scored.loc[mask])})
    return pd.DataFrame(rows), scored


def main():
    valid, holdout = load_predictions()
    grid = optimize_policy(valid)
    if grid.empty:
        raise RuntimeError("No feasible threshold pair found under review capacity.")
    best = grid.iloc[0].to_dict()
    review_threshold = float(best["review_threshold"])
    escalate_threshold = float(best["escalate_threshold"])

    valid_segments, valid_scored = segment_eval(valid, review_threshold, escalate_threshold)
    holdout_segments, holdout_scored = segment_eval(holdout, review_threshold, escalate_threshold)
    valid_segments.insert(0, "split", "valid")
    holdout_segments.insert(0, "split", "holdout")
    segment_results = pd.concat([valid_segments, holdout_segments], ignore_index=True)

    base_valid = baseline_policies(valid)
    base_holdout_rows = []
    for _, row in base_valid.iterrows():
        policy = row["policy"]
        if policy == "approve_all":
            tmp = holdout.copy()
            tmp["decision"] = "approve"
        elif policy == "review_top5_no_escalate":
            tmp = holdout.copy()
            tmp["decision"] = np.where(tmp["pred"] >= row["review_threshold"], "review", "approve")
        else:
            tmp = holdout.copy()
            tmp["decision"] = np.where(tmp["pred"] >= row["escalate_threshold"], "escalate", "approve")
        base_holdout_rows.append({"policy": policy, **evaluate_policy(tmp)})
    base_valid.insert(0, "split", "valid")
    base_holdout = pd.DataFrame(base_holdout_rows)
    base_holdout.insert(0, "split", "holdout")
    baseline_results = pd.concat([base_valid, base_holdout], ignore_index=True)

    policy_thresholds = pd.DataFrame([{
        "review_threshold": review_threshold,
        "escalate_threshold": escalate_threshold,
        "cost_fraud_approved": COST_FRAUD_APPROVED,
        "cost_legit_escalated": COST_LEGIT_ESCALATED,
        "cost_review": COST_REVIEW,
        "max_review_rate": MAX_REVIEW_RATE,
    }])

    grid.to_csv(OUT / "threshold_grid_valid.csv", index=False)
    policy_thresholds.to_csv(OUT / "policy_thresholds.csv", index=False)
    segment_results.to_csv(OUT / "policy_segment_results.csv", index=False)
    baseline_results.to_csv(OUT / "policy_baselines.csv", index=False)
    valid_scored.to_csv(OUT / "valid_policy_decisions.csv", index=False)
    holdout_scored.to_csv(OUT / "holdout_policy_decisions.csv", index=False)

    summary = {
        "selected_policy": policy_thresholds.iloc[0].to_dict(),
        "valid_global": segment_results[(segment_results.split == "valid") & (segment_results.segment == "global")].iloc[0].to_dict(),
        "holdout_global": segment_results[(segment_results.split == "holdout") & (segment_results.segment == "global")].iloc[0].to_dict(),
    }
    (OUT / "policy_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = []
    report.append("# V05 politica de decision\n\n")
    report.append("## Supuestos de costo\n\n")
    report.append(f"- Fraude aprobado: {COST_FRAUD_APPROVED}.\n")
    report.append(f"- Legitima escalada/bloqueada: {COST_LEGIT_ESCALATED}.\n")
    report.append(f"- Revision manual: {COST_REVIEW}.\n")
    report.append(f"- Capacidad maxima de revision: {MAX_REVIEW_RATE:.0%}.\n\n")
    report.append("## Umbrales seleccionados en validacion\n\n")
    report.append(df_to_md(policy_thresholds))
    report.append("\n\n## Resultados por segmento\n\n")
    report.append(df_to_md(segment_results[["split", "segment", "cost_per_txn", "fraud_detection_rate", "fraud_miss_rate", "review_rate", "escalate_rate", "approve_rate", "legit_escalation_rate"]]))
    report.append("\n\n## Baselines de politica\n\n")
    report.append(df_to_md(baseline_results[["split", "policy", "cost_per_txn", "fraud_detection_rate", "review_rate", "escalate_rate", "approve_rate"]]))
    report.append("\n")
    (OUT / "v05_policy_report.md").write_text("".join(report), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
