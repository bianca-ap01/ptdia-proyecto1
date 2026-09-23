"""Audit source notebooks and, when present, the downloaded four-stage run."""

import argparse
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from core import SCENARIOS, config_digest, policy_actions, policy_cost

SOURCE_ROOT = Path(__file__).resolve().parent
ROOT = SOURCE_ROOT
NOTEBOOKS = ("01_eda", "02_experimentacion", "03_costos_escenarios", "04_modelo_final")


def verify_notebooks():
    for name in NOTEBOOKS:
        path = SOURCE_ROOT / f"{name}.ipynb"
        nbformat.validate(nbformat.read(path, as_version=4))
    print("PASS: four valid standalone notebooks")


def verify_outputs():
    stages = ("eda", "experimentacion", "costos", "modelo_final")
    summaries = {}
    for stage in stages:
        path = ROOT / "outputs" / stage / "run_summary.json"
        summaries[stage] = json.loads(path.read_text())
        assert summaries[stage]["status"] == "complete", (stage, summaries[stage])
    assert summaries["eda"]["holdout_labels_used"] is False
    assert summaries["experimentacion"]["holdout_labels_used"] is False
    assert summaries["costos"]["holdout_labels_used"] is False
    exp = ROOT / "outputs" / "experimentacion"
    cv = pd.read_csv(exp / "inner_cv.csv")
    assert set(cv.model) == {"logistic", "tree", "lightgbm", "xgboost"}
    assert set(cv.fold) == {1, 2}
    assert (cv.train_last_dt < cv.available_before).all()
    fits = pd.read_csv(exp / "fit_log.csv")
    assert (fits.train_last_dt < fits.available_before).all()
    config = json.loads((exp / "selected_hyperparameters.json").read_text())
    digest = config.pop("digest")
    assert digest == config_digest(config)
    config["digest"] = digest
    choice = json.loads((ROOT / "outputs" / "costos" / "frozen_choice.json").read_text())
    choice_digest = choice.pop("digest")
    assert choice_digest == config_digest(choice)
    choice["digest"] = choice_digest
    assert choice["training_config_digest"] == config["digest"]
    assert choice["holdout_labels_used"] is False
    assert choice["selection_period_end"] <= config["valid_cut"] - 7 * 86_400
    ranking = pd.read_csv(ROOT / "outputs" / "costos" / "candidate_ranking.csv")
    winner = ranking.iloc[0]
    assert winner.model == choice["model"] and winner.strategy == choice["strategy"]
    assert str(winner.window_days) == choice["window_days"]
    assert np.isclose(winner.review_threshold, choice["review_threshold"])
    assert np.isclose(winner.escalate_threshold, choice["escalate_threshold"])
    final = ROOT / "outputs" / "modelo_final"
    final_fits = pd.read_csv(final / "fit_log.csv")
    assert (final_fits.train_last_dt < final_fits.available_before).all()
    if choice["strategy"] == "static":
        assert (final_fits.available_before == config["train_cut"] - 7 * 86_400).all()
    decisions = pd.read_csv(final / "holdout_decisions.csv.gz")
    assert set(decisions.action) <= {"approve", "review", "escalate"}
    daily = decisions.action.eq("review").groupby(decisions.TransactionDT // 86_400).sum()
    assert (daily <= choice["review_capacity_per_day"]).all()
    expected = policy_actions(decisions, choice["review_threshold"],
                              choice["escalate_threshold"], choice["review_capacity_per_day"])
    observed = decisions.action.map({"approve": 0, "review": 1, "escalate": 2}).to_numpy()
    assert np.array_equal(expected, observed), "Action used labels or unfrozen thresholds"
    workload = pd.read_csv(final / "daily_workload.csv")
    assert (workload.review_count <= workload.review_capacity).all()
    assert np.allclose(workload.expected_total_escalations,
                          workload.direct_escalations + workload.expected_review_handoffs)
    for scenario, (review, _) in SCENARIOS.items():
        part = workload[workload.scenario == scenario]
        assert part.review_count.sum() == int((observed == 1).sum())
        assert part.direct_escalations.sum() == int((observed == 2).sum())
        assert np.isclose(part.expected_review_handoffs.sum(),
                          review * int(((observed == 1) & decisions.isFraud.to_numpy(dtype=bool)).sum()))
    saved = pd.read_csv(final / "holdout_cost_scenarios.csv").set_index("scenario")
    for scenario, (review, escalate) in SCENARIOS.items():
        computed = policy_cost(decisions, observed, review, escalate)
        for field in ("cost_total", "approve_all_total", "avoided_loss_total", "cost_per_txn"):
            assert np.isclose(computed[field], saved.loc[scenario, field]), (scenario, field)
        assert np.isclose(computed["cost_total"] - computed["approve_all_total"],
                          -computed["avoided_loss_total"])
    ledger = pd.read_csv(final / "quality_ledger.csv")
    used = ledger.dropna(subset=["candidate_train_last_dt"])
    assert (used.candidate_train_last_dt < used.calibration_end - 7 * 86_400).all()
    assert (used.gate_end <= used.start - 7 * 86_400).all()
    monthly_quality = pd.read_csv(final / "monthly_quality_ledger.csv")
    assert set(monthly_quality.relative_month) == set(ledger.relative_month)
    assert (monthly_quality.result_rows_after_first_action >= 0).all()
    assert summaries["modelo_final"]["retrospective_holdout"] is True
    print("PASS: splits, fitting cutoffs, frozen choice, review capacity, cost and adaptation gates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebooks-only", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Package output root; useful for synthetic smoke verification")
    args = parser.parse_args()
    ROOT = args.root
    verify_notebooks()
    if not args.notebooks_only:
        verify_outputs()
