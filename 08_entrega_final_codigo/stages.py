"""Four executable stages for the final code deliverable."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.isotonic import IsotonicRegression

from core import (DAY, WEEK, GRIDS, SCENARIOS, candidate_grid, config_digest, cuts,
                  daily_review_cap, data_dir, fit_snapshot, json_dump, load_train,
                  metrics, output_dir, own_row_features, policy_actions, policy_cost,
                  relative_day, safe_ap, score_snapshot)


def package_root() -> Path:
    return Path(__file__).resolve().parent


def progress(out: Path, stage: str, status="running", **extra):
    json_dump(out / "run_summary.json", {"stage": stage, "status": status, **extra})


def artifact(root: Path, stage: str, name: str) -> Path:
    local = root / "outputs" / stage / name
    if local.exists():
        return local
    candidates = sorted(Path("/kaggle/input").glob(f"**/{name}")) if Path("/kaggle/input").exists() else []
    candidates = [p for p in candidates if "ieee-fraud-detection" not in str(p)]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Expected one {name} from {stage}; found {len(candidates)}")


def prepared(root: Path) -> tuple[pd.DataFrame, tuple[float, float]]:
    df = load_train(root)
    bounds = cuts(df)
    return own_row_features(df), bounds


def run_eda(root: Path | None = None) -> Path:
    root = root or package_root()
    out = output_dir(root, "eda")
    progress(out, "eda")
    df, (train_end, valid_end) = prepared(root)
    day = relative_day(df)
    df["month_relative"] = day // 30 + 1
    period = np.where(df.TransactionDT <= train_end, "train",
                      np.where(df.TransactionDT <= valid_end, "valid", "holdout_hidden"))
    df["period"] = period
    visible = df.loc[df.period != "holdout_hidden"].copy()
    visible["day_relative"] = day.loc[visible.index]
    monthly = visible.groupby("month_relative", sort=True).agg(
        rows=("isFraud", "size"), fraud=("isFraud", "sum"),
        fraud_rate=("isFraud", "mean"), median_amount=("TransactionAmt", "median"),
        identity_missing=("has_identity", lambda x: 1 - x.mean()),
    ).reset_index()
    monthly.to_csv(out / "monthly_eda.csv", index=False)
    temporal = []
    for span in (1, 7, 30):
        grouped = visible.groupby(visible.day_relative // span, sort=True).agg(
            rows=("isFraud", "size"), fraud_rate=("isFraud", "mean"),
            identity_missing=("has_identity", lambda x: 1 - x.mean()),
            first_day=("day_relative", "min"), last_day=("day_relative", "max"),
        )
        grouped.index.name = "period_index"
        grouped = grouped.reset_index()
        grouped.insert(0, "window_days", span)
        grouped["mid_day"] = (grouped.first_day + grouped.last_day) / 2
        temporal.append(grouped)
    pd.concat(temporal, ignore_index=True).to_csv(out / "temporal_eda.csv", index=False)
    missing = visible.groupby("month_relative")[["TransactionAmt", "D1", "C9", "id_02"]].apply(
        lambda x: x.isna().mean()).reset_index()
    missing.to_csv(out / "monthly_missingness.csv", index=False)
    source = data_dir(root)
    raw_columns = list(pd.read_csv(source / "train_transaction.csv", nrows=0).columns)
    raw_columns += [c.replace("id-", "id_") if c.startswith("id-") else c
                    for c in pd.read_csv(source / "train_identity.csv", nrows=0).columns]
    raw_features = [c for c in dict.fromkeys(raw_columns)
                    if c not in {"TransactionID", "TransactionDT", "isFraud"}]
    all_missing = visible[raw_features].isna().groupby(visible.month_relative).mean()
    all_missing.index.name = "month_relative"
    all_missing = all_missing.reset_index().melt(
        id_vars="month_relative", var_name="feature", value_name="missing_rate")
    all_missing["rows"] = all_missing.month_relative.map(visible.groupby("month_relative").size())
    all_missing.to_csv(out / "monthly_missingness_all.csv", index=False)
    # Diagnostic associations use only labels available before validation.
    model_df = df.drop(columns=["month_relative", "period"])
    train_mask = (df.TransactionDT < train_end - WEEK).to_numpy()
    train = model_df.loc[train_mask]
    target = train.isFraud.astype(float)
    associations = []
    for name in raw_features:
        values = train[name]
        if pd.api.types.is_numeric_dtype(values):
            signed = values.corr(target)
            measure = "pearson_r"
            strength = abs(signed)
        else:
            table = pd.crosstab(values.astype("string").fillna("__MISSING__"), target)
            chi2 = chi2_contingency(table, correction=False).statistic if min(table.shape) > 1 else np.nan
            signed = np.nan
            measure = "cramers_v"
            strength = np.sqrt(chi2 / len(train))
        associations.append({"feature": name, "measure": measure, "association": strength,
                             "signed_correlation": signed, "non_null_rows": int(values.notna().sum()),
                             "missing_rate": float(values.isna().mean())})
    pd.DataFrame(associations).to_csv(out / "feature_associations.csv", index=False)

    # A separate LightGBM diagnostic follows the current window-specific feature selection.
    snapshot = fit_snapshot(model_df, train_mask, "lightgbm", GRIDS["lightgbm"][0], train_end - WEEK)
    model = snapshot["model"]
    gain = pd.DataFrame({
        "feature": snapshot["features"],
        "gain": model.booster_.feature_importance(importance_type="gain"),
        "splits": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    gain.to_csv(out / "feature_importance_gain.csv", index=False)
    valid = model_df.loc[(df.TransactionDT > train_end) & (df.TransactionDT <= valid_end)]
    sample = valid.sample(n=min(10_000, len(valid)), random_state=42)
    x_valid = snapshot["encoder"].transform(sample)
    y_valid = sample.isFraud.to_numpy()
    baseline_ap = safe_ap(y_valid, model.predict_proba(x_valid)[:, 1])
    rng = np.random.default_rng(42)
    permutation = []
    for name in gain.head(30).feature:
        original = x_valid[name].to_numpy(copy=True)
        x_valid[name] = rng.permutation(original)
        permuted_ap = safe_ap(y_valid, model.predict_proba(x_valid)[:, 1])
        x_valid[name] = original
        permutation.append({"feature": name, "pr_auc_drop": baseline_ap - permuted_ap,
                            "baseline_pr_auc": baseline_ap, "permuted_pr_auc": permuted_ap,
                            "validation_rows": len(sample)})
    pd.DataFrame(permutation).to_csv(out / "feature_importance_permutation.csv", index=False)
    json_dump(out / "feature_diagnostics.json", {
        "association_rows": len(train), "association_period": "train before validation minus 7 days",
        "raw_features": len(raw_features), "baseline": "LightGBM first grid configuration",
        "model_features": len(snapshot["features"]), "permutation_rows": len(sample),
        "permutation_features": len(permutation), "seed": 42,
        "note": "Exploratory diagnostic only; not the model or policy chosen in later stages.",
    })
    first = visible.loc[visible.month_relative == visible.month_relative.min(), "TransactionAmt"].dropna()
    shift = []
    for month, part in visible.groupby("month_relative"):
        values = part.TransactionAmt.dropna()
        shift.append({"month_relative": int(month), "amount_ks_vs_month1":
                      float(ks_2samp(first, values).statistic) if len(values) else float("nan")})
    pd.DataFrame(shift).to_csv(out / "monthly_shift.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(monthly.month_relative, monthly.fraud_rate, marker="o")
    axes[0].set(ylabel="Fraud prevalence", title="Relative 30-day periods; holdout labels hidden")
    axes[1].plot(monthly.month_relative, monthly.identity_missing, marker="o")
    axes[1].set(xlabel="Relative month", ylabel="Missing identity share")
    fig.tight_layout()
    fig.savefig(out / "monthly_eda.png", dpi=140)
    plt.close(fig)
    json_dump(out / "leakage_audit.json", {
        "excluded_from_predictors": ["TransactionID", "TransactionDT", "isFraud", "uid"],
        "fitted_only_on_history": ["V-column ranking", "category maps and frequencies", "amount IQR", "imputation"],
        "causal_prior_event_features": ["uid_prior_count", "uid_prior_amount", "uid_prior_mean",
                                        "uid_seconds_since_prior", "uid_amount_vs_prior_mean"],
        "remaining_uncertainty": "Anonymized IEEE-CIS columns have no operational availability contract.",
        "holdout_labels_used": False,
    })
    progress(out, "eda", "complete", rows=len(df), train_cut=train_end, valid_cut=valid_end,
             observed_months=int(monthly.month_relative.nunique()), holdout_labels_used=False)
    return out


def _inner_folds(df: pd.DataFrame, train_end: float):
    origin = float(df.TransactionDT.min())
    span = train_end - origin
    if span < 60 * DAY:
        raise ValueError("Initial training period is too short for two embargoed folds")
    for number, (start_fraction, stop_fraction) in enumerate(((.50, .65), (.75, .90)), 1):
        start, stop = origin + start_fraction * span, origin + stop_fraction * span
        train_mask = (df.TransactionDT < start - WEEK).to_numpy()
        valid_mask = ((df.TransactionDT >= start) & (df.TransactionDT < stop)).to_numpy()
        yield number, start, stop, train_mask, valid_mask


def _weekly_blocks(df: pd.DataFrame, start: float, stop: float):
    t, number = start, 0
    while t < stop:
        end = min(t + WEEK, stop)
        mask = (df.TransactionDT >= t) & (df.TransactionDT < end)
        if mask.any():
            yield number, t, end, df.loc[mask]
        t, number = end, number + 1


def _predict_candidate(df: pd.DataFrame, start: float, stop: float, name: str,
                       params: dict, strategy: str, window_days: int | None,
                       quality_floor: float, static_cutoff: float | None = None):
    parts, fits = [], []
    active = None
    history = []
    for block, t, end, rows in _weekly_blocks(df, start, stop):
        matured = [h for h in history if h["end"] <= t - WEEK]
        failed = len(matured) >= 2 and all(h["pr_auc"] < quality_floor for h in matured[-2:])
        update = active is None or strategy == "periodic" or (strategy == "quality" and failed)
        if update:
            cutoff = static_cutoff if strategy == "static" and static_cutoff is not None else t - WEEK
            if cutoff > t - WEEK:
                raise AssertionError("Static model includes labels unavailable at evaluation time")
            floor = -np.inf if strategy == "static" else cutoff - int(window_days) * DAY
            mask = ((df.TransactionDT < cutoff) & (df.TransactionDT >= floor)).to_numpy()
            active = fit_snapshot(df, mask, name, params, cutoff)
            fits.append({"model": name, "strategy": strategy, "window_days": window_days,
                         "block": block, "eval_start": t, "available_before": cutoff,
                         "train_last_dt": active["fit_max_dt"], "train_rows": active["train_rows"],
                         "trigger": bool(failed)})
        score = score_snapshot(active, rows)
        item = rows[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "uid"]].copy()
        item["score"] = score
        item["model"] = name
        item["strategy"] = strategy
        item["window_days"] = "static" if window_days is None else str(window_days)
        item["block"] = block
        parts.append(item)
        history.append({"end": end, "pr_auc": safe_ap(rows.isFraud, score)})
    return pd.concat(parts, ignore_index=True), fits


def run_experiment(root: Path | None = None) -> Path:
    root = root or package_root()
    out = output_dir(root, "experimentacion")
    progress(out, "experimentacion")
    df, (train_end, valid_end) = prepared(root)
    folds = list(_inner_folds(df, train_end))
    cv, weekly = [], []
    for name, grid in GRIDS.items():
        candidates = grid[:1] if os.getenv("PTDIA_SMOKE") == "1" else grid
        for param_index, params in enumerate(candidates):
            for fold, start, stop, train_mask, valid_mask in folds:
                snapshot = fit_snapshot(df, train_mask, name, params, start - WEEK)
                part = df.loc[valid_mask]
                score = score_snapshot(snapshot, part)
                cv.append({"model": name, "param_index": param_index, "params": json.dumps(params, sort_keys=True),
                           "fold": fold, "eval_start": start, "eval_end": stop,
                           "pr_auc": safe_ap(part.isFraud, score),
                           "train_last_dt": snapshot["fit_max_dt"], "available_before": start - WEEK,
                           "train_rows": snapshot["train_rows"], "eval_rows": len(part),
                           "feature_count": len(snapshot["features"])})
                for block, _, _, rows in _weekly_blocks(part, start, stop):
                    mask = part.index.isin(rows.index)
                    weekly.append({"model": name, "param_index": param_index, "fold": fold,
                                   "week": block, "pr_auc": safe_ap(rows.isFraud, score[mask])})
            pd.DataFrame(cv).to_csv(out / "inner_cv.csv", index=False)
            progress(out, "experimentacion", completed_configs=len({(r['model'], r['param_index']) for r in cv}))
    cv_frame = pd.DataFrame(cv)
    week_frame = pd.DataFrame(weekly)
    week_frame.to_csv(out / "inner_weekly.csv", index=False)
    selected = {}
    for name, part in cv_frame.groupby("model"):
        rank = part.groupby("param_index").pr_auc.agg(["mean", "min"]).sort_values(
            ["mean", "min"], ascending=False)
        index = int(rank.index[0])
        aps = week_frame[(week_frame.model == name) & (week_frame.param_index == index)].pr_auc.dropna()
        selected[name] = {"params": GRIDS[name][index], "param_index": index,
                          "inner_mean_pr_auc": float(rank.iloc[0]["mean"]),
                          "quality_floor": float(aps.quantile(.10))}
    config = {"selected": selected, "train_cut": train_end, "valid_cut": valid_end,
              "label_delay_days": 7, "fold_train_span_fractions": [[.50, .65], [.75, .90]],
              "models": list(GRIDS), "adaptive_windows_days": [30, 60, 90]}
    config["digest"] = config_digest(config)
    json_dump(out / "selected_hyperparameters.json", config)
    all_parts, fits = [], []
    for name in GRIDS:
        options = [("static", None)]
        if name in ("lightgbm", "xgboost"):
            options += [("periodic", d) for d in (30, 60, 90)] + [("quality", 60)]
        for strategy, window in options:
            pred, fit_rows = _predict_candidate(df, train_end, valid_end, name,
                                                selected[name]["params"], strategy, window,
                                                selected[name]["quality_floor"])
            all_parts.append(pred)
            fits += fit_rows
            pd.DataFrame(fits).to_csv(out / "fit_log.csv", index=False)
            progress(out, "experimentacion", completed_candidates=len(all_parts))
    predictions = pd.concat(all_parts, ignore_index=True)
    predictions.to_csv(out / "validation_predictions.csv.gz", index=False, compression="gzip")
    if not all(row["train_last_dt"] < row["available_before"] for row in fits):
        raise AssertionError("A fitted model used unavailable labels")
    progress(out, "experimentacion", "complete", candidate_count=len(all_parts),
             prediction_rows=len(predictions), config_digest=config["digest"],
             holdout_labels_used=False)
    return out


def _selection_slice(frame: pd.DataFrame, train_end: float, valid_end: float):
    # First week of validation is reserved for calibration; final seven days
    # have labels that would not be mature when holdout begins.
    return frame[(frame.TransactionDT >= train_end + WEEK) &
                 (frame.TransactionDT < valid_end - WEEK)].sort_values(
                     ["TransactionDT", "TransactionID"]).reset_index(drop=True)


def _robust_table(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    minima = table.groupby("scenario").cost_per_txn.min().to_dict()
    table["regret_pct"] = table.apply(
        lambda r: 100 * (r.cost_per_txn / minima[r.scenario] - 1), axis=1)
    return table


def run_cost(root: Path | None = None) -> Path:
    root = root or package_root()
    out = output_dir(root, "costos")
    progress(out, "costos")
    config = json.loads(artifact(root, "experimentacion", "selected_hyperparameters.json").read_text())
    preds = pd.read_csv(artifact(root, "experimentacion", "validation_predictions.csv.gz"),
                        dtype={"window_days": str}, low_memory=False)
    train = pd.read_csv(data_dir(root) / "train_transaction.csv", usecols=["TransactionDT"])
    available = train[train.TransactionDT < config["train_cut"] - WEEK]
    cap = daily_review_cap(available)
    table = []
    for (name, strategy, window), group in preds.groupby(["model", "strategy", "window_days"], sort=True):
        # The quality-triggered replay is a technical comparator. Its
        # ungated training updates are not a deployable policy candidate;
        # the final notebook uses a separate promotion gate.
        if strategy == "quality":
            continue
        frame = _selection_slice(group, config["train_cut"], config["valid_cut"])
        if len(frame) < 1_000 or frame.isFraud.sum() < 20:
            raise ValueError("Policy validation slice too small")
        ap = safe_ap(frame.isFraud, frame.score)
        for low, high, action in candidate_grid(frame, cap):
            for scenario, (review, escalate) in SCENARIOS.items():
                cost = policy_cost(frame, action, review, escalate)
                table.append({"model": name, "strategy": strategy, "window_days": window,
                              "review_threshold": low, "escalate_threshold": high,
                              "scenario": scenario, "validation_pr_auc": ap, **cost})
    scored = _robust_table(table)
    scored.to_csv(out / "validation_scenarios.csv", index=False, float_format="%.10f")
    keys = ["model", "strategy", "window_days", "review_threshold", "escalate_threshold"]
    rank = scored.groupby(keys, dropna=False).agg(
        worst_regret_pct=("regret_pct", "max"), mean_regret_pct=("regret_pct", "mean"),
        mean_cost=("cost_per_txn", "mean"), validation_pr_auc=("validation_pr_auc", "first")
    ).reset_index()
    model_complexity = {"logistic": 0, "tree": 1, "lightgbm": 2, "xgboost": 3}
    window_overhead = {"static": 0, "90": 1, "60": 2, "30": 3}
    rank["complexity_rank"] = (rank.model.map(model_complexity) +
                               rank.window_days.map(window_overhead).fillna(9) +
                               rank.strategy.map({"static": 0, "periodic": 4}))
    rank = rank.sort_values(
        ["worst_regret_pct", "mean_regret_pct", "validation_pr_auc", "complexity_rank"],
        ascending=[True, True, False, True])
    rank.to_csv(out / "candidate_ranking.csv", index=False)
    winner = rank.iloc[0]
    choice = {"model": winner.model, "strategy": winner.strategy,
              "window_days": winner.window_days, "review_threshold": float(winner.review_threshold),
              "escalate_threshold": float(winner.escalate_threshold),
              "worst_regret_pct": float(winner.worst_regret_pct),
              "review_capacity_per_day": cap, "training_config_digest": config["digest"],
              "selection_period_end": config["valid_cut"] - WEEK,
              "holdout_labels_used": False,
              "cost_provenance": {
                  "review_usd": "Forrester TEI: 0.75 minutes at USD 52/hour",
                  "escalation_usd": "USD 4 per confirmed-fraud escalation is a workflow assumption; Forrester USD 4 concerns some false-alert follow-up",
                  "fraud_loss_multiplier": "LexisNexis 2025 US financial services benchmark, not an IEEE-CIS tariff",
              }}
    choice["digest"] = config_digest(choice)
    json_dump(out / "frozen_choice.json", choice)
    selected = preds[(preds.model == choice["model"]) & (preds.strategy == choice["strategy"]) &
                     (preds.window_days == choice["window_days"])]
    selected = _selection_slice(selected, config["train_cut"], config["valid_cut"])
    sensitivity = []
    for fraction in (.03, .05, .10):
        limit = daily_review_cap(available, fraction)
        action = policy_actions(selected, choice["review_threshold"], choice["escalate_threshold"], limit)
        for scenario, (review, escalation) in SCENARIOS.items():
            sensitivity.append({"capacity_fraction": fraction, "capacity_per_day": limit,
                                "scenario": scenario, **policy_cost(selected, action, review, escalation)})
    pd.DataFrame(sensitivity).to_csv(out / "capacity_sensitivity.csv", index=False)
    progress(out, "costos", "complete", candidates=len(rank), scenario_rows=len(scored),
             chosen=choice, holdout_labels_used=False)
    return out


def _weekly_quality(frame: pd.DataFrame, floor: float, label_delay=WEEK):
    weeks = []
    start, stop = float(frame.TransactionDT.min()), float(frame.TransactionDT.max()) + 1
    for number, t, end, rows in _weekly_blocks(frame, start, stop):
        weeks.append({"week": number, "start": t, "end": end,
                      "pr_auc": safe_ap(rows.isFraud, rows.score),
                      "quality_floor": floor, "passes": bool(safe_ap(rows.isFraud, rows.score) >= floor),
                      "labels_mature_at": end + label_delay})
    return pd.DataFrame(weeks)


def _best_thresholds_one_frame(frame: pd.DataFrame, cap: int):
    """Choose one policy on a past calibration week using minimax regret."""
    rows = []
    for low, high, action in candidate_grid(frame, cap):
        for scenario, (review, escalation) in SCENARIOS.items():
            rows.append({"low": low, "high": high, "scenario": scenario,
                         "cost_per_txn": policy_cost(frame, action, review, escalation)["cost_per_txn"]})
    scored = _robust_table(rows)
    ranked = scored.groupby(["low", "high"]).agg(
        worst=("regret_pct", "max"), mean=("regret_pct", "mean")
    ).reset_index().sort_values(["worst", "mean", "low", "high"])
    return float(ranked.iloc[0].low), float(ranked.iloc[0].high)


def _gate_cost(frame: pd.DataFrame, score: np.ndarray, low: float, high: float,
               cap: int) -> tuple[float, float]:
    scored = frame[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]].copy()
    scored["score"] = score
    scored = scored.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    action = policy_actions(scored, low, high, cap)
    values = [policy_cost(scored, action, *rates)["cost_per_txn"] for rates in SCENARIOS.values()]
    return float(max(values)), safe_ap(scored.isFraud, scored.score)


def _quality_replay(df: pd.DataFrame, start: float, stop: float, name: str,
                    params: dict, floor: float, low: float, high: float, cap: int):
    """Prequential shadow controller with separate calibration and promotion gates."""
    base_mask = (df.TransactionDT < start - WEEK).to_numpy()
    active = fit_snapshot(df, base_mask, name, params, start - WEEK)
    history, parts, ledger = [], [], []
    for block, t, end, rows in _weekly_blocks(df, start, stop):
        matured = [h for h in history if h["end"] <= t - WEEK]
        breach = len(matured) >= 2 and all(h["pr_auc"] < floor for h in matured[-2:])
        entry = {"week": block, "start": t, "end": end, "quality_floor": floor,
                 "matured_blocks": len(matured), "last_two_breaches": int(sum(
                     h["pr_auc"] < floor for h in matured[-2:])),
                 "last_matured_pr_auc": matured[-1]["pr_auc"] if matured else np.nan,
                 "previous_matured_pr_auc": matured[-2]["pr_auc"] if len(matured) > 1 else np.nan,
                 "decision": "keep", "candidate_promoted": False,
                 "candidate_train_last_dt": np.nan, "calibration_end": np.nan,
                 "gate_end": np.nan, "incumbent_gate_pr_auc": np.nan,
                 "candidate_gate_pr_auc": np.nan, "incumbent_gate_worst_cost": np.nan,
                 "candidate_gate_worst_cost": np.nan}
        if breach:
            train_end = t - 3 * WEEK
            calibration_end = t - 2 * WEEK
            gate_end = t - WEEK
            train_mask = ((df.TransactionDT >= train_end - 60 * DAY) &
                          (df.TransactionDT < train_end)).to_numpy()
            calibration = df[(df.TransactionDT >= train_end) &
                             (df.TransactionDT < calibration_end)]
            gate = df[(df.TransactionDT >= calibration_end) &
                      (df.TransactionDT < gate_end)]
            entry.update({"decision": "reject_candidate", "calibration_end": calibration_end,
                          "gate_end": gate_end})
            if len(calibration) >= 1000 and len(gate) >= 1000 and gate.isFraud.sum() >= 20:
                candidate = fit_snapshot(df, train_mask, name, params, train_end)
                entry["candidate_train_last_dt"] = candidate["fit_max_dt"]
                cal_score = score_snapshot(candidate, calibration)
                cal_frame = calibration[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]].copy()
                cal_frame["score"] = cal_score
                cal_frame = cal_frame.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
                candidate_low, candidate_high = _best_thresholds_one_frame(cal_frame, cap)
                incumbent_gate_cost, incumbent_ap = _gate_cost(
                    gate, score_snapshot(active, gate), low, high, cap)
                candidate_gate_cost, candidate_ap = _gate_cost(
                    gate, score_snapshot(candidate, gate), candidate_low, candidate_high, cap)
                entry.update({"incumbent_gate_pr_auc": incumbent_ap,
                              "candidate_gate_pr_auc": candidate_ap,
                              "incumbent_gate_worst_cost": incumbent_gate_cost,
                              "candidate_gate_worst_cost": candidate_gate_cost})
                if candidate_gate_cost < incumbent_gate_cost and candidate_ap >= incumbent_ap:
                    active, low, high = candidate, candidate_low, candidate_high
                    entry.update({"decision": "promote_candidate", "candidate_promoted": True})
        score = score_snapshot(active, rows)
        item = rows[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "uid"]].copy()
        item["score"] = score
        item["block"] = block
        item["review_threshold"] = low
        item["escalate_threshold"] = high
        parts.append(item)
        entry["observed_pr_auc_after_action"] = safe_ap(rows.isFraud, score)
        history.append({"end": end, "pr_auc": entry["observed_pr_auc_after_action"]})
        ledger.append(entry)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(ledger)


def run_final(root: Path | None = None) -> Path:
    root = root or package_root()
    out = output_dir(root, "modelo_final")
    progress(out, "modelo_final")
    config = json.loads(artifact(root, "experimentacion", "selected_hyperparameters.json").read_text())
    choice = json.loads(artifact(root, "costos", "frozen_choice.json").read_text())
    if choice["training_config_digest"] != config["digest"] or choice["holdout_labels_used"]:
        raise AssertionError("Choice was not frozen from validation")
    df, bounds = prepared(root)
    if not np.allclose(bounds, [config["train_cut"], config["valid_cut"]]):
        raise AssertionError("Data split differs from the frozen choice")
    name = choice["model"]
    params = config["selected"][name]["params"]
    window = None if choice["window_days"] == "static" else int(choice["window_days"])
    start = config["valid_cut"]
    stop = float(df.TransactionDT.max()) + 1
    holdout, fits = _predict_candidate(df, start, stop, name, params, choice["strategy"], window,
                                       config["selected"][name]["quality_floor"],
                                       static_cutoff=config["train_cut"] - WEEK)
    holdout = holdout.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    holdout.to_csv(out / "holdout_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(fits).to_csv(out / "fit_log.csv", index=False)
    action = policy_actions(holdout, choice["review_threshold"], choice["escalate_threshold"],
                            choice["review_capacity_per_day"])
    holdout["action"] = np.array(["approve", "review", "escalate"], dtype=object)[action]
    holdout.to_csv(out / "holdout_decisions.csv.gz", index=False, compression="gzip")
    scenarios = []
    for scenario, (review, escalation) in SCENARIOS.items():
        scenarios.append({"scenario": scenario, **policy_cost(holdout, action, review, escalation)})
    pd.DataFrame(scenarios).to_csv(out / "holdout_cost_scenarios.csv", index=False)
    workload = []
    holdout_day = (holdout.TransactionDT // DAY).astype(int).to_numpy()
    for day in np.unique(holdout_day):
        mask = holdout_day == day
        day_actions = action[mask]
        day_fraud = holdout.isFraud.to_numpy(dtype=bool)[mask]
        reviews = int((day_actions == 1).sum())
        direct = int((day_actions == 2).sum())
        reviewed_fraud = int(((day_actions == 1) & day_fraud).sum())
        for scenario, (review, _) in SCENARIOS.items():
            handoffs = review * reviewed_fraud
            workload.append({"relative_day": int(day - df.TransactionDT.min() // DAY),
                             "scenario": scenario, "review_count": reviews,
                             "review_capacity": choice["review_capacity_per_day"],
                             "direct_escalations": direct,
                             "expected_review_handoffs": handoffs,
                             "expected_total_escalations": direct + handoffs})
    pd.DataFrame(workload).to_csv(out / "daily_workload.csv", index=False)
    # Calibration is fitted on the first validation week only. It is used for
    # probability reporting, never for policy choice or holdout fitting.
    valid = pd.read_csv(artifact(root, "experimentacion", "validation_predictions.csv.gz"),
                        dtype={"window_days": str}, low_memory=False)
    cal = valid[(valid.model == name) & (valid.strategy == choice["strategy"]) &
                (valid.window_days == choice["window_days"]) &
                (valid.TransactionDT < config["train_cut"] + WEEK)]
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(cal.score, cal.isFraud)
    calibrated = isotonic.predict(holdout.score)
    overall = metrics(holdout.isFraud, holdout.score, choice["escalate_threshold"])
    overall["brier_calibrated"] = metrics(holdout.isFraud, calibrated)["brier"]
    overall["retrospective_holdout"] = True
    json_dump(out / "final_metrics.json", overall)
    month_by_id = pd.Series(relative_day(df).to_numpy() // 30 + 1,
                            index=df.TransactionID.to_numpy())
    holdout["relative_month"] = holdout.TransactionID.map(month_by_id).to_numpy()
    monthly = []
    for month, part in holdout.groupby("relative_month"):
        ix = part.index.to_numpy()
        row = {"relative_month": int(month), **metrics(part.isFraud, part.score,
                                                       choice["escalate_threshold"])}
        row.update(policy_cost(part, action[ix], *SCENARIOS["adverso_80_95"]))
        monthly.append(row)
    pd.DataFrame(monthly).to_csv(out / "monthly_metrics.csv", index=False)
    shadow, ledger = _quality_replay(
        df, start, stop, name, params, config["selected"][name]["quality_floor"],
        choice["review_threshold"], choice["escalate_threshold"],
        choice["review_capacity_per_day"])
    shadow.to_csv(out / "quality_replay_predictions.csv.gz", index=False, compression="gzip")
    ledger["relative_month"] = ((ledger.start - float(df.TransactionDT.min())) //
                                (30 * DAY)).astype(int) + 1
    ledger.to_csv(out / "quality_ledger.csv", index=False)
    shadow["relative_month"] = ((shadow.TransactionDT - float(df.TransactionDT.min())) //
                                (30 * DAY)).astype(int) + 1
    monthly_quality = []
    origin = float(df.TransactionDT.min())
    for month, month_rows in shadow.groupby("relative_month"):
        month_start = origin + (int(month) - 1) * 30 * DAY
        month_end = month_start + 30 * DAY
        checks = ledger[(ledger.start < month_end) & (ledger.end > month_start)]
        actions = checks.loc[checks.decision != "keep", "decision"]
        first_action = checks.loc[checks.decision != "keep", "start"]
        month_scores = month_rows
        if len(first_action):
            month_scores = month_scores[month_scores.TransactionDT >= first_action.min()]
        monthly_quality.append({
            "relative_month": int(month),
            "last_matured_pr_auc": float(checks.last_matured_pr_auc.iloc[-1]),
            "quality_floor": float(checks.quality_floor.iloc[-1]),
            "breach_checks": int((checks.last_two_breaches == 2).sum()),
            "actions": ", ".join(actions) if len(actions) else "keep",
            "result_rows_after_first_action": len(month_scores),
            "observed_pr_auc_after_first_action": safe_ap(month_scores.isFraud, month_scores.score),
        })
    pd.DataFrame(monthly_quality).to_csv(out / "monthly_quality_ledger.csv", index=False)
    uid_rows = []
    seen = set(df.loc[df.TransactionDT < start, "uid"])
    for known, part in holdout.groupby(holdout.uid.isin(seen)):
        uid_rows.append({"uid_known_before_holdout": bool(known), "rows": len(part),
                         "pr_auc": safe_ap(part.isFraud, part.score),
                         "direct_escalations": int((part.action == "escalate").sum())})
    pd.DataFrame(uid_rows).to_csv(out / "uid_coverage.csv", index=False)
    progress(out, "modelo_final", "complete", rows=len(holdout), selected_choice_digest=choice["digest"],
             retrospective_holdout=True, holdout_evaluation_count=1)
    return out


RUNNERS = {"eda": run_eda, "experimentacion": run_experiment,
           "costos": run_cost, "modelo_final": run_final}


def run(stage: str, root: Path | None = None):
    if stage not in RUNNERS:
        raise ValueError(stage)
    root = root or package_root()
    out = output_dir(root, stage)
    try:
        return RUNNERS[stage](root)
    except Exception as exc:
        progress(out, stage, "failed", error=repr(exc), traceback=traceback.format_exc())
        raise
