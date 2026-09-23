"""Rebuild the figures in the report and slides from the saved IEEE-CIS outputs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
DATA = ROOT.parent / "08_entrega_final_codigo" / "outputs"
OUT.mkdir(exist_ok=True)

NAVY, TEAL, CORAL, GOLD = "#14324c", "#008a89", "#d46a4e", "#b38331"
COLORS = {"logistic": "#71839a", "tree": GOLD, "lightgbm": TEAL, "xgboost": NAVY}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.labelcolor": NAVY, "text.color": NAVY,
    "axes.titleweight": "bold", "figure.facecolor": "white", "savefig.facecolor": "white",
})


def save(name):
    plt.savefig(OUT / name, bbox_inches="tight", pad_inches=.12)
    plt.close()


eda = pd.read_csv(DATA / "eda/monthly_eda.csv")
shift = pd.read_csv(DATA / "eda/monthly_shift.csv")
missing = pd.read_csv(DATA / "eda/monthly_missingness_all.csv")
spread = (missing.groupby("feature").missing_rate.max()
          - missing.groupby("feature").missing_rate.min()).sort_values(ascending=False)
fig, axes = plt.subplots(2, 2, figsize=(11.4, 5.6), constrained_layout=True)
x = eda.month_relative
axes[0, 0].plot(x, eda.fraud_rate * 100, "o-", color=CORAL, lw=2)
axes[0, 0].set(title="Prevalencia de fraude", ylabel="Fraude (%)", xlabel="Periodo de 30 días")
axes[0, 1].plot(x, eda.identity_missing * 100, "o-", color=TEAL, lw=2)
axes[0, 1].set(title="Identidad ausente", ylabel="Filas sin identidad (%)", xlabel="Periodo de 30 días")
axes[1, 0].bar(x, shift.amount_ks_vs_month1, color=NAVY)
axes[1, 0].set(title="Cambio de monto frente al periodo 1", ylabel="KS (diagnóstico)", xlabel="Periodo de 30 días")
top = spread.head(8).sort_values()
axes[1, 1].barh(top.index, top.values * 100, color=GOLD)
axes[1, 1].set(title="Mayor cambio de faltantes", xlabel="Rango entre periodos (puntos porcentuales)")
for ax in axes.flat:
    ax.grid(axis="y", alpha=.18)
fig.savefig(OUT / "eda_evidence.pdf", bbox_inches="tight", pad_inches=.12)
plt.close(fig)

cv = pd.read_csv(DATA / "experimentacion/inner_cv.csv")
weekly = pd.read_csv(DATA / "experimentacion/inner_weekly.csv")
best = (cv.groupby(["model", "param_index"]).pr_auc.agg(["mean", "min"])
        .reset_index().sort_values(["model", "mean", "min"], ascending=[True, False, False])
        .drop_duplicates("model"))
fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.2), constrained_layout=True)
for _, row in best.iterrows():
    s = weekly[(weekly.model == row.model) & (weekly.param_index == row.param_index)]
    for fold, part in s.groupby("fold"):
        axes[0].plot(part.week + (fold - 1) * 4.2, part.pr_auc, "o-", ms=3,
                     color=COLORS[row.model], alpha=.72, label=row.model if fold == 1 else None)
axes[0].axvline(3.6, color="#aab4c1", ls="--", lw=1)
axes[0].set(title="PR-AUC semanal en dos folds históricos", ylabel="PR-AUC", xlabel="Semana dentro del fold (separados)")
axes[0].legend(ncol=2, fontsize=8)
rank = pd.read_csv(DATA / "costos/candidate_ranking.csv")
rank = rank[(rank.strategy == "periodic") & (rank.window_days.isin(["30", "60", "90"]))]
rank = rank.sort_values("validation_pr_auc", ascending=False).drop_duplicates(["model", "window_days"])
for model, offset in [("lightgbm", -.17), ("xgboost", .17)]:
    part = rank[rank.model == model].set_index("window_days")
    axes[1].bar(np.arange(3) + offset, [part.loc[str(d), "validation_pr_auc"] for d in (30, 60, 90)],
                width=.34, color=COLORS[model], label=model)
axes[1].set(xticks=range(3), xticklabels=["30", "60", "90"], ylim=(0, .65),
            title="PR-AUC en validación por ventana", ylabel="PR-AUC", xlabel="Días de entrenamiento reciente")
axes[1].legend(fontsize=8)
for ax in axes: ax.grid(axis="y", alpha=.18)
save("model_time.pdf")

hold = pd.read_csv(DATA / "modelo_final/holdout_cost_scenarios.csv")
cap = pd.read_csv(DATA / "costos/capacity_sensitivity.csv")
scenarios = ["optimista_100_100", "favorable_90_98", "adverso_80_95", "severo_60_90"]
labels = ["100/100", "90/98", "80/95", "60/90"]
hold = hold.set_index("scenario").loc[scenarios]
fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.1), constrained_layout=True)
bars = axes[0].bar(labels, hold.cost_per_txn, color=[TEAL, TEAL, NAVY, CORAL])
axes[0].axhline(hold.approve_all_total.iloc[0] / 88581, color=GOLD, ls="--", label="Aprobar todo")
axes[0].set(title="Costo retrospectivo por transacción", ylabel="Unidades monetarias ilustrativas", xlabel="Eficacia revisión/escalación")
axes[0].legend(fontsize=8)
for bar, val in zip(bars, hold.cost_per_txn):
    axes[0].text(bar.get_x()+bar.get_width()/2, val+.35, f"{val:.1f}", ha="center", fontsize=8)
subset = cap[cap.scenario == "adverso_80_95"].sort_values("capacity_fraction")
axes[1].plot(subset.capacity_fraction * 100, subset.cost_per_txn, "o-", color=NAVY, lw=2)
axes[1].set(xticks=[3, 5, 10], title="Sensibilidad del cupo en validación (80/95)",
            xlabel="Cupo de revisión inicial (%)", ylabel="Costo por transacción")
for ax in axes: ax.grid(axis="y", alpha=.18)
save("cost_scenarios.pdf")

q = pd.read_csv(DATA / "modelo_final/quality_ledger.csv")
fig, ax = plt.subplots(figsize=(11.4, 2.7), constrained_layout=True)
ax.plot(q.week + 1, q.observed_pr_auc_after_action, "o-", color=TEAL, lw=2,
        label="PR-AUC observada en bloque posterior")
ax.plot(q.week + 1, q.last_matured_pr_auc, "s--", color=NAVY, lw=1.5,
        label="Última PR-AUC con etiqueta madura")
ax.axhline(q.quality_floor.iloc[0], color=CORAL, ls="--", label="Cota 0.4897")
ax.set(xticks=q.week + 1, xlabel="Semana del holdout", ylabel="PR-AUC",
       title="Una alerta madura; la siguiente medición se recuperó")
ax.legend(ncol=3, fontsize=8, loc="lower right")
ax.grid(axis="y", alpha=.18)
save("quality_controller.pdf")

work = pd.read_csv(DATA / "modelo_final/daily_workload.csv")
work = work[work.scenario == "adverso_80_95"].sort_values("relative_day")
fig, ax = plt.subplots(figsize=(11.4, 2.6), constrained_layout=True)
ax.plot(work.relative_day, work.expected_total_escalations, color=CORAL, lw=1.8,
        label="Escalaciones esperadas (directas + derivadas)")
ax.axhline(work.review_capacity.iloc[0], color=NAVY, ls="--",
           label="Cupo diario de revisiones iniciales: 159")
ax.set(xlabel="Día relativo", ylabel="Casos por día",
       title="La segunda cola necesita una capacidad propia")
ax.legend(fontsize=8, loc="upper left")
ax.grid(axis="y", alpha=.18)
save("daily_workload.pdf")
