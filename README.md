# Proyecto 1 — IEEE-CIS Fraud Detection

Sistema inteligente adaptativo sobre datos no estacionarios.

## Nueva entrega de código

La carpeta **[`08_entrega_final_codigo/`](08_entrega_final_codigo/README.md)** contiene los cuatro notebooks, el código compartido, el protocolo de validación temporal, outputs IEEE-CIS descargados y el verificador. Las cuatro etapas terminaron en kernels privados de la cuenta Kaggle autorizada (`badexample`) y la verificación integral pasó. El [informe](07_latex/informe.pdf), la [presentación](07_latex/presentacion.pdf) y la [infografía](07_latex/infografia.pdf) se actualizaron con esos resultados. [`08_entrega_final_codigo/run_manifest.json`](08_entrega_final_codigo/run_manifest.json) registra el estado por etapa y la limitación retrospectiva del holdout.

`final/` conserva la entrega histórica y sus resultados anteriores; las cifras de sus documentos no deben atribuirse a los notebooks nuevos.

## Entrega histórica

| Qué | Dónde |
| --- | --- |
| Informe técnico | [`final/REPORTE.md`](final/REPORTE.md) |
| Auditoría actual de validación, costos y rúbrica | [`final/AUDITORIA_RUBRICA.md`](final/AUDITORIA_RUBRICA.md) |
| Sensibilidad de costos con selección en validación | [`final/cost_sensitivity_valid.csv`](final/cost_sensitivity_valid.csv) · [`final/cost_sensitivity.py`](final/cost_sensitivity.py) |
| 1 · Análisis exploratorio temporal | [`final/01_eda.ipynb`](final/01_eda.ipynb) |
| 2 · Monitoreo y detección de drift | [`final/02_monitoreo_drift.ipynb`](final/02_monitoreo_drift.ipynb) |
| 3 · Modelos y estrategias de adaptación | [`final/03_modelos_adaptacion.ipynb`](final/03_modelos_adaptacion.ipynb) |
| 4 · Política de decisión y costo | [`final/04_politica_juego.ipynb`](final/04_politica_juego.ipynb) |
| Verificación automática de resultados | [`final/verify_results.py`](final/verify_results.py) |
| Salidas de cada etapa | `final/kaggle/<etapa>/outputs/` |
| API de decisión (FastAPI + Cloud Run) | [`serving/`](serving/README.md) |

Reproducir sin reentrenar:

```bash
pip install -e .
python final/verify_results.py   # comprueba cortes, cupo y reconciliación de costo
```

Los datos crudos (~1.3 GB) no se versionan: se descargan de la competencia
`ieee-fraud-detection` de Kaggle a `ieee-fraud-detection/`.

## Bitácora experimental (histórico)

Las carpetas `01_planning/` … `06_delivery/` documentan las versiones V01–V07 que
precedieron a la nueva entrega. `07_latex/` contiene ahora los PDFs actualizados.

| Carpeta | Contenido | Uso |
| --- | --- | --- |
| `00_brief/` | Planteo original del proyecto. | Fuente de requisitos. |
| `01_planning/` | Plan metodologico y log de decisiones. | Trazabilidad de por que se tomo cada decision. |
| `02_kaggle_kernels/` | Codigo de notebooks/kernels ejecutados en Kaggle. | Reproducibilidad de EDA y versiones V01-V07. |
| `03_outputs/` | Resultados descargados o generados por version. | Metricas, predicciones, reportes, logs y graficos. |
| `04_scripts/` | Scripts locales de respaldo. | Politica de decision V05 y robustez V06 antes de publicarlas en Kaggle. |
| `05_reports/` | Reportes narrativos por version. | Material base para la entrega final. |
| `06_delivery/` | Borradores previos alineados a la rubrica. | Conservan cifras de V05/V06 que no corresponden a la evaluación de `final/`; no usarlos como resultados actuales. |
| `07_latex/` | Informe, presentación e infografía compilables en LaTeX. | Actualizados desde los outputs de `08_entrega_final_codigo/`; no usar cifras de `final/REPORTE.md` como resultados nuevos. |

## Versiones

| Version | Tema | Codigo | Outputs | Reporte |
| --- | --- | --- | --- | --- |
| EDA | Analisis exploratorio propio | `02_kaggle_kernels/eda/` | `03_outputs/eda/` | `03_outputs/eda/eda_report.md` |
| V01 | Baseline temporal LightGBM | `02_kaggle_kernels/v01_baseline_temporal/` | `03_outputs/v01_baseline_temporal/` | `05_reports/v01_baseline_temporal_resultados.md` |
| V01b | Baseline con todas las features | `02_kaggle_kernels/v01b_full_baseline/` | `03_outputs/v01b_full_baseline/` | `05_reports/v01b_full_baseline_temporal_resultados.md` |
| V02 | Features UID | `02_kaggle_kernels/v02_uid_features/` | `03_outputs/v02_uid_features/` | `05_reports/v02_uid_features_resultados.md` |
| V03 | Auditoria de features | `02_kaggle_kernels/v03_feature_audit/` | `03_outputs/v03_feature_audit/` | `05_reports/v03_feature_audit_resultados.md` |
| V04 | Comparacion de modelos | `02_kaggle_kernels/v04_model_comparison/` | `03_outputs/v04_model_comparison/` | `05_reports/v04_model_comparison_resultados.md` |
| V05 | Politica de decision | `02_kaggle_kernels/v05_policy_decision/` y `04_scripts/v05_policy_decision.py` | `03_outputs/v05_policy_decision/` y `03_outputs/v05_policy_decision_kaggle/` | `05_reports/v05_policy_resultados.md` |
| V06 | Robustez y calibracion | `02_kaggle_kernels/v06_policy_robustness/` y `04_scripts/v06_policy_robustness.py` | `03_outputs/v06_policy_robustness/` y `03_outputs/v06_policy_robustness_kaggle/` | `05_reports/v06_policy_robustness_resultados.md` |
| V07 | Benchmark MLP y EDA complementario | `02_kaggle_kernels/v07_mlp_benchmark/` | `03_outputs/v07_mlp_benchmark_kaggle/` | `05_reports/v07_mlp_benchmark_resultados.md` |

## Kernels publicos en Kaggle

- EDA: https://www.kaggle.com/code/biancaaguinaga/p1-ieee-fraud-eda
- V01 baseline temporal: https://www.kaggle.com/code/biancaaguinaga/p1-ieee-fraud-v01-baseline-temporal
- V01b baseline con todas las features: https://www.kaggle.com/code/biancaaguinaga/p1-ieee-fraud-v01b-full-baseline
- V02 features UID: https://www.kaggle.com/code/biancaaguinaga/p1-ieee-fraud-v02-uid-features
- V03 auditoria de features: https://www.kaggle.com/code/biancaaguinaga/p1-ieee-fraud-v03-feature-audit
- V04 comparacion de modelos: https://www.kaggle.com/code/biancaaguinaga/p1-ieee-fraud-v04-model-comparison
- V05 politica de decision: https://www.kaggle.com/code/biancaaguinaga/p1-ieee-fraud-v05-policy-decision
- V06 robustez y calibracion: https://www.kaggle.com/code/biancaaguinaga/p1-ieee-fraud-v06-policy-robustness
- V07 benchmark MLP: https://www.kaggle.com/code/biancaaguinaga/p1-ieee-fraud-v07-mlp-benchmark

## Documentos principales

- `00_brief/Project 1.pdf`
- `00_brief/UTEC_2026_1__Planificación_y_Toma_de_Decisiones_en_IA (1).pdf`
- `00_brief/Taller_Despliegue_GCP.pptx.pdf`
- `01_planning/plan_modelado_eda.md`
- `01_planning/decision_log_modelado.md`
- `01_planning/auditoria_rubrica_profesor.md`
- `05_reports/modelado_consolidado.md`
- `06_delivery/informe_tecnico_8_paginas.md`
- `06_delivery/arquitectura_sistema_mermaid.md`
- `06_delivery/propuesta_despliegue_gcp.md`
- `06_delivery/protocolo_drift_adaptacion.md`
- `06_delivery/infografia_final_mermaid.md`
- `06_delivery/guion_presentacion.md`
- `07_latex/informe.pdf`
- `07_latex/presentacion.pdf`
- `07_latex/infografia.pdf`

## Decisión histórica

La evaluación de `final/` mantuvo V01 estático como referencia operativa. La selección por costo en su validación favoreció LightGBM periódico de 45 días, pero esa ventaja no se confirmó en su holdout. Esos resultados son anteriores al protocolo de `08_entrega_final_codigo/` y no constituyen la evaluación de los notebooks nuevos.
