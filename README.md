# Project 1 - IEEE-CIS Fraud Detection

Estructura del trabajo de modelado.

| Carpeta | Contenido | Uso |
| --- | --- | --- |
| `00_brief/` | Planteo original del proyecto. | Fuente de requisitos. |
| `01_planning/` | Plan metodologico y log de decisiones. | Trazabilidad de por que se tomo cada decision. |
| `02_kaggle_kernels/` | Codigo de notebooks/kernels ejecutados en Kaggle. | Reproducibilidad de EDA y modelos V01-V04. |
| `03_outputs/` | Resultados descargados o generados por version. | Metricas, predicciones, reportes, logs y graficos. |
| `04_scripts/` | Scripts locales posteriores a los kernels. | Politica de decision V05 y robustez V06. |
| `05_reports/` | Reportes narrativos por version. | Material base para la entrega final. |

## Versiones

| Version | Tema | Codigo | Outputs | Reporte |
| --- | --- | --- | --- | --- |
| EDA | Analisis exploratorio propio | `02_kaggle_kernels/eda/` | `03_outputs/eda/` | `03_outputs/eda/eda_report.md` |
| V01 | Baseline temporal LightGBM | `02_kaggle_kernels/v01_baseline_temporal/` | `03_outputs/v01_baseline_temporal/` | `05_reports/v01_baseline_temporal_resultados.md` |
| V01b | Baseline con todas las features | `02_kaggle_kernels/v01b_full_baseline/` | `03_outputs/v01b_full_baseline/` | `05_reports/v01b_full_baseline_temporal_resultados.md` |
| V02 | Features UID | `02_kaggle_kernels/v02_uid_features/` | `03_outputs/v02_uid_features/` | `05_reports/v02_uid_features_resultados.md` |
| V03 | Auditoria de features | `02_kaggle_kernels/v03_feature_audit/` | `03_outputs/v03_feature_audit/` | `05_reports/v03_feature_audit_resultados.md` |
| V04 | Comparacion de modelos | `02_kaggle_kernels/v04_model_comparison/` | `03_outputs/v04_model_comparison/` | `05_reports/v04_model_comparison_resultados.md` |
| V05 | Politica de decision | `04_scripts/v05_policy_decision.py` | `03_outputs/v05_policy_decision/` | `05_reports/v05_policy_resultados.md` |
| V06 | Robustez y calibracion | `04_scripts/v06_policy_robustness.py` | `03_outputs/v06_policy_robustness/` | `05_reports/v06_policy_robustness_resultados.md` |

## Documentos principales

- `00_brief/Project 1.pdf`
- `00_brief/UTEC_2026_1__Planificación_y_Toma_de_Decisiones_en_IA (1).pdf`
- `01_planning/plan_modelado_eda.md`
- `01_planning/decision_log_modelado.md`
- `01_planning/auditoria_rubrica_profesor.md`
- `05_reports/modelado_consolidado.md`

## Decision actual

Mantener V01 como modelo base documentado, usar V05 como primera politica operativa, y usar V06 como auditoria de robustez/calibracion para la entrega consolidada.
