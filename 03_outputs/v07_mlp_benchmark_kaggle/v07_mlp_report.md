# V07 MLP benchmark y complemento experimental

## Objetivo

Agregar una comparacion reproducible con MLP y un baseline tradicional sobre el mismo split temporal 70/15/15 usado por V01.

## Relacion con el EDA de referencia

El kernel de Fabryzzio aporta cuatro ideas que complementan el EDA: faltantes por bloques V, ausencia estructural de identity, montos con mas de dos decimales y riesgo de clientes/UID recurrentes. V07 incorpora `TransactionAmt_multi_decimal` y mantiene la evaluacion por UID conocido/desconocido; no cambia el split principal para conservar comparabilidad con V01-V06.

Features usadas: 186.

## Modelos

| model                        |   elapsed_seconds |   features |   best_iteration |   train_cutoff |   valid_cutoff |   n_iter |   best_epoch |   epochs_run |   best_val_pr_auc_keras |
|:-----------------------------|------------------:|-----------:|-----------------:|---------------:|---------------:|---------:|-------------:|-------------:|------------------------:|
| lightgbm_v01_feature_surface |           58.9538 |        186 |              549 |     1.0438e+07 |    1.31518e+07 |      nan |          nan |          nan |              nan        |
| sgd_logistic_traditional     |           29.6014 |        186 |              nan |     1.0438e+07 |    1.31518e+07 |       35 |          nan |          nan |              nan        |
| mlp_keras_weighted           |           27.6586 |        186 |              nan |     1.0438e+07 |    1.31518e+07 |      nan |            9 |           12 |                0.438859 |

## Resultados globales

| model                        | split   | segment   |   rows |   fraud_rate |   pr_auc |   roc_auc |
|:-----------------------------|:--------|:----------|-------:|-------------:|---------:|----------:|
| lightgbm_v01_feature_surface | valid   | global    |  88581 |    0.0343415 | 0.588452 |  0.924965 |
| lightgbm_v01_feature_surface | holdout | global    |  88581 |    0.0348043 | 0.531283 |  0.905519 |
| sgd_logistic_traditional     | valid   | global    |  88581 |    0.0343415 | 0.297394 |  0.796393 |
| sgd_logistic_traditional     | holdout | global    |  88581 |    0.0348043 | 0.163835 |  0.769616 |
| mlp_keras_weighted           | valid   | global    |  88581 |    0.0343415 | 0.447644 |  0.874728 |
| mlp_keras_weighted           | holdout | global    |  88581 |    0.0348043 | 0.221833 |  0.849569 |

## Holdout por UID

| model                        | split   | segment     |   rows |   fraud_rate |    pr_auc |   roc_auc |
|:-----------------------------|:--------|:------------|-------:|-------------:|----------:|----------:|
| lightgbm_v01_feature_surface | holdout | uid_known   |  39512 |    0.0193359 | 0.481359  |  0.90414  |
| lightgbm_v01_feature_surface | holdout | uid_unknown |  49069 |    0.04726   | 0.556717  |  0.893754 |
| sgd_logistic_traditional     | holdout | uid_known   |  39512 |    0.0193359 | 0.057035  |  0.739215 |
| sgd_logistic_traditional     | holdout | uid_unknown |  49069 |    0.04726   | 0.306814  |  0.773562 |
| mlp_keras_weighted           | holdout | uid_known   |  39512 |    0.0193359 | 0.0688141 |  0.81288  |
| mlp_keras_weighted           | holdout | uid_unknown |  49069 |    0.04726   | 0.487474  |  0.860605 |

## Decision

El MLP se acepta como benchmark adicional de deep learning. Solo reemplazaria a V01 si mejora PR-AUC holdout y no degrada UID desconocido. Si queda por debajo, su valor es cerrar la comparacion experimental y reforzar que los modelos tabulares de boosting son mas adecuados para esta primera version.
