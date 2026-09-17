# V05 politica de decision

## Supuestos de costo

- Fraude aprobado: 100.0.
- Legitima escalada/bloqueada: 10.0.
- Revision manual: 2.0.
- Capacidad maxima de revision: 5%.

## Umbrales seleccionados en validacion

| review_threshold | escalate_threshold | cost_fraud_approved | cost_legit_escalated | cost_review | max_review_rate |
| --- | --- | --- | --- | --- | --- |
| 0.459783 | 0.779122 | 100 | 10 | 2 | 0.05 |

## Resultados por segmento

| split | segment | cost_per_txn | fraud_detection_rate | fraud_miss_rate | review_rate | escalate_rate | approve_rate | legit_escalation_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | global | 1.13948 | 0.718277 | 0.281723 | 0.0498188 | 0.0235152 | 0.926666 | 0.00749366 |
| valid | uid_known | 0.749308 | 0.706933 | 0.293067 | 0.0232399 | 0.0113873 | 0.965373 | 0.00200587 |
| valid | uid_unknown | 1.47317 | 0.723445 | 0.276555 | 0.0725506 | 0.0338877 | 0.893562 | 0.0122875 |
| holdout | global | 1.36804 | 0.661693 | 0.338307 | 0.0509816 | 0.0247796 | 0.924239 | 0.0091815 |
| holdout | uid_known | 0.916329 | 0.554974 | 0.445026 | 0.0182982 | 0.00840251 | 0.973299 | 0.00196139 |
| holdout | uid_unknown | 1.73177 | 0.696852 | 0.303148 | 0.0772993 | 0.0379669 | 0.884734 | 0.0151658 |

## Baselines de politica

| split | policy | cost_per_txn | fraud_detection_rate | review_rate | escalate_rate | approve_rate |
| --- | --- | --- | --- | --- | --- | --- |
| valid | approve_all | 3.43415 | 0 | 0 | 0 | 1 |
| valid | review_top5_no_escalate | 1.34069 | 0.638725 | 0.0500107 | 0 | 0.949989 |
| valid | escalate_top1_only | 2.54445 | 0.261999 | 0 | 0.0100021 | 0.989998 |
| holdout | approve_all | 3.48043 | 0 | 0 | 0 | 1 |
| holdout | review_top5_no_escalate | 1.52471 | 0.591956 | 0.0522685 | 0 | 0.947731 |
| holdout | escalate_top1_only | 2.59593 | 0.257541 | 0 | 0.0101489 | 0.989851 |
