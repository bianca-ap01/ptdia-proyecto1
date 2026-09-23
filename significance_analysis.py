"""Incertidumbre estadistica del ranking de estrategias de adaptacion.

Lee block_metrics.csv y responde dos preguntas que el reporte dejaba abiertas:

1. La ventaja de adaptar sobre no adaptar, es distinguible del ruido entre
   bloques? Se usa t pareado porque las estrategias comparten los mismos
   bloques de holdout.
2. La ventana de 45 dias es realmente mejor que la de 30, o la diferencia
   cabe dentro de la variacion entre bloques?

Salida: final/kaggle/experimentacion/outputs/significance_summary.csv
"""

from pathlib import Path

import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
BLOCKS = HERE / "kaggle" / "experimentacion" / "outputs" / "block_metrics.csv"
OUT = HERE / "kaggle" / "experimentacion" / "outputs" / "significance_summary.csv"

MODELS = ["lightgbm", "xgboost", "catboost"]
STATIC = ("static", "initial_70pct")


def pr_auc_by_block(df, model, strategy, window):
    """PR-AUC ordenada por bloque para una configuracion."""
    sel = df[
        (df.model == model) & (df.strategy == strategy) & (df.window_days == window)
    ]
    return sel.sort_values("block").pr_auc.to_numpy()


def paired_test(treatment, control, label, model):
    """t pareado + IC95 de la diferencia.

    Los bloques son los mismos en ambas configuraciones, asi que el pareo
    elimina la varianza entre bloques, que es la que domina con n=5.
    """
    delta = treatment - control
    t_stat, p_value = stats.ttest_rel(treatment, control)
    ci_low, ci_high = stats.t.interval(
        0.95, len(delta) - 1, loc=delta.mean(), scale=stats.sem(delta)
    )
    return {
        "model": model,
        "comparison": label,
        "mean_treatment": round(float(treatment.mean()), 4),
        "mean_control": round(float(control.mean()), 4),
        "mean_delta": round(float(delta.mean()), 4),
        "blocks_favorable": f"{int((delta > 0).sum())}/{len(delta)}",
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_value), 4),
        "ci95_low": round(float(ci_low), 4),
        "ci95_high": round(float(ci_high), 4),
        "significant_at_05": bool(p_value < 0.05),
    }


def main():
    blocks = pd.read_csv(BLOCKS)
    holdout = blocks[blocks.split == "holdout"].copy()
    holdout["window_days"] = holdout.window_days.astype(str)

    rows = []
    for model in MODELS:
        static = pr_auc_by_block(holdout, model, *STATIC)
        periodic_45 = pr_auc_by_block(holdout, model, "periodic", "45")
        periodic_30 = pr_auc_by_block(holdout, model, "periodic", "30")

        rows.append(paired_test(periodic_45, static, "periodic45_vs_static", model))
        rows.append(
            paired_test(periodic_45, periodic_30, "periodic45_vs_periodic30", model)
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT, index=False)

    print(summary.to_string(index=False))
    print(f"\nEscrito: {OUT.relative_to(HERE)}")

    adapt = summary[summary.comparison == "periodic45_vs_static"]
    window = summary[summary.comparison == "periodic45_vs_periodic30"]
    print(
        f"\nAdaptar vs no adaptar: significativo en "
        f"{int(adapt.significant_at_05.sum())}/{len(adapt)} familias."
    )
    print(
        f"Ventana 45 vs 30 dias: significativo en "
        f"{int(window.significant_at_05.sum())}/{len(window)} familias."
    )


if __name__ == "__main__":
    main()
