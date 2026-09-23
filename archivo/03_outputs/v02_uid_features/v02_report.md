# V02 UID features - IEEE-CIS Fraud Detection

## Objetivo

Evaluar si features historicas por UID mejoran el baseline temporal V01 sin usar informacion futura ni target encoding.

## Ablations

- `base`: baseline comparable a V01.
- `uid_freq`: base + conteos/frecuencias/conocido por UID.
- `uid_freq_amt`: uid_freq + estadisticas historicas de `TransactionAmt` por UID.

## Regla anti-leakage

Las features UID para validacion y holdout se calculan usando solo la ventana de entrenamiento. No se usa `isFraud` para construirlas.

## Resultados globales

| model        | split   |   rows |   fraud_rate |   pr_auc |   roc_auc |
|:-------------|:--------|-------:|-------------:|---------:|----------:|
| base         | valid   |  88581 |    0.0343415 | 0.595406 |  0.925686 |
| base         | holdout |  88581 |    0.0348043 | 0.543631 |  0.905074 |
| uid_freq     | valid   |  88581 |    0.0343415 | 0.593764 |  0.928145 |
| uid_freq     | holdout |  88581 |    0.0348043 | 0.538118 |  0.906028 |
| uid_freq_amt | valid   |  88581 |    0.0343415 | 0.607145 |  0.93105  |
| uid_freq_amt | holdout |  88581 |    0.0348043 | 0.54415  |  0.907965 |

## Resumen de features

| model        |   features |   uid_features_added |   best_iteration |   train_cutoff |   valid_cutoff |
|:-------------|-----------:|---------------------:|-----------------:|---------------:|---------------:|
| base         |        185 |                    0 |              547 |     1.0438e+07 |    1.31518e+07 |
| uid_freq     |        200 |                   15 |              549 |     1.0438e+07 |    1.31518e+07 |
| uid_freq_amt |        235 |                   50 |              548 |     1.0438e+07 |    1.31518e+07 |
