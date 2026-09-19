# V06 robustez de politica

## Comparacion base en holdout

| policy_type | cost_per_txn | fraud_detection_rate | fraud_miss_rate | review_rate | escalate_rate | approve_rate |
| --- | --- | --- | --- | --- | --- | --- |
| global | 1.38093 | 0.687966 | 0.312034 | 0.0506542 | 0.0377959 | 0.91155 |
| segmented_uid | 1.37244 | 0.665585 | 0.334415 | 0.0477642 | 0.028155 | 0.924081 |

## Sensibilidad de escenarios globales en holdout

| scenario | cost_per_txn | fraud_detection_rate | review_rate | escalate_rate | approve_rate |
| --- | --- | --- | --- | --- | --- |
| base | 1.38093 | 0.687966 | 0.0506542 | 0.0377959 | 0.91155 |
| friction_high | 1.50681 | 0.631203 | 0.0451564 | 0.0195866 | 0.935257 |
| fraud_high | 2.33574 | 0.751541 | 0.0498527 | 0.0734469 | 0.8767 |
| capacity_tight | 1.50996 | 0.639312 | 0.0309547 | 0.0377056 | 0.93134 |
| capacity_loose | 1.16948 | 0.753487 | 0.0963186 | 0.0287195 | 0.874962 |

## Calibracion

| split | score | brier | mean_score | fraud_rate |
| --- | --- | --- | --- | --- |
| valid | pred | 0.0440376 | 0.140342 | 0.0343415 |
| valid | pred_iso | 0.0199836 | 0.0343415 | 0.0343415 |
| holdout | pred | 0.047877 | 0.143039 | 0.0348043 |
| holdout | pred_iso | 0.0219158 | 0.0354783 | 0.0348043 |
