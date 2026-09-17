# V03 feature audit - IEEE-CIS Fraud Detection

## Objetivo

Auditar features del baseline V01 combinando importancia predictiva, permutation importance, shift temporal y cambios de missingness.

## Metricas del modelo auditado

|   valid_pr_auc |   valid_roc_auc |   holdout_pr_auc |   holdout_roc_auc |   best_iteration |   features |
|---------------:|----------------:|-----------------:|------------------:|-----------------:|-----------:|
|       0.595406 |        0.925686 |         0.543631 |          0.905074 |              547 |        185 |

## Recomendaciones

| recommendation   |   features |
|:-----------------|-----------:|
| low_priority     |        112 |
| keep             |         44 |
| review           |         20 |
| monitor          |          9 |

## Top features a monitorear

| feature   |   gain_importance |   perm_pr_auc_drop |    max_ks |   max_missing_delta |
|:----------|------------------:|-------------------:|----------:|--------------------:|
| D1n       |          189260   |         0.0337797  | 0.604382  |             1.01207 |
| D2n       |          139602   |         0.0236394  | 0.400203  |             8.07558 |
| D15n      |          113391   |         0.016189   | 0.453441  |             6.76677 |
| D10n      |           93624.5 |         0.0245998  | 0.525621  |             6.60332 |
| C9        |           36282.4 |         0.0129949  | 0.139912  |             0       |
| id_02     |           25738.6 |         0.00365629 | 0.126568  |             8.78058 |
| id_20     |           20042.2 |         0.00361889 | 0.140107  |             8.77437 |
| D11       |           19174.1 |         0.00764298 | 0.0983796 |            26.2685  |
| id_01     |           16158.5 |         0.00115571 | 0.122537  |             8.91654 |

## Criterio

- `keep`: predictiva y sin shift alto.
- `monitor`: predictiva, pero con shift o cambio de missingness.
- `review`: poco predictiva, pero con shift alto.
- `low_priority`: baja evidencia predictiva y sin alerta fuerte.
