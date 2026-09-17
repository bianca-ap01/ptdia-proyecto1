# V04 model comparison - IEEE-CIS Fraud Detection

## Objetivo

Comparar feature sets auditados y modelos sobre el mismo split temporal.

## Candidatos

| model             | kind   | status   |   features |   best_iteration |   elapsed_seconds |   train_cutoff |   valid_cutoff |
|:------------------|:-------|:---------|-----------:|-----------------:|------------------:|---------------:|---------------:|
| lgbm_keep         | lgbm   | complete |         44 |              649 |           29.7351 |     1.0438e+07 |    1.31518e+07 |
| lgbm_keep_monitor | lgbm   | complete |         53 |              649 |           30.9528 |     1.0438e+07 |    1.31518e+07 |
| xgb_keep_monitor  | xgb    | complete |         53 |              500 |           33.5543 |     1.0438e+07 |    1.31518e+07 |
| cat_keep_monitor  | cat    | complete |         53 |              499 |           63.1816 |     1.0438e+07 |    1.31518e+07 |

## Resultados globales

| model             | split   | segment   |   rows |   fraud_rate |   pr_auc |   roc_auc |
|:------------------|:--------|:----------|-------:|-------------:|---------:|----------:|
| lgbm_keep         | valid   | global    |  88581 |    0.0343415 | 0.547415 |  0.920925 |
| lgbm_keep         | holdout | global    |  88581 |    0.0348043 | 0.500231 |  0.900067 |
| lgbm_keep_monitor | valid   | global    |  88581 |    0.0343415 | 0.573939 |  0.923283 |
| lgbm_keep_monitor | holdout | global    |  88581 |    0.0348043 | 0.513071 |  0.899636 |
| xgb_keep_monitor  | valid   | global    |  88581 |    0.0343415 | 0.550403 |  0.920139 |
| xgb_keep_monitor  | holdout | global    |  88581 |    0.0348043 | 0.512772 |  0.90554  |
| cat_keep_monitor  | valid   | global    |  88581 |    0.0343415 | 0.538279 |  0.917568 |
| cat_keep_monitor  | holdout | global    |  88581 |    0.0348043 | 0.498874 |  0.901987 |

## Resultados por UID en holdout

| model             | split   | segment     |   rows |   fraud_rate |   pr_auc |   roc_auc |
|:------------------|:--------|:------------|-------:|-------------:|---------:|----------:|
| lgbm_keep         | holdout | uid_known   |  29240 |    0.0169973 | 0.413008 |  0.900606 |
| lgbm_keep         | holdout | uid_unknown |  59341 |    0.0435786 | 0.527829 |  0.893782 |
| lgbm_keep_monitor | holdout | uid_known   |  29240 |    0.0169973 | 0.542524 |  0.915845 |
| lgbm_keep_monitor | holdout | uid_unknown |  59341 |    0.0435786 | 0.518081 |  0.884721 |
| xgb_keep_monitor  | holdout | uid_known   |  29240 |    0.0169973 | 0.492277 |  0.920137 |
| xgb_keep_monitor  | holdout | uid_unknown |  59341 |    0.0435786 | 0.529582 |  0.894177 |
| cat_keep_monitor  | holdout | uid_known   |  29240 |    0.0169973 | 0.414835 |  0.90952  |
| cat_keep_monitor  | holdout | uid_unknown |  59341 |    0.0435786 | 0.521776 |  0.893247 |
