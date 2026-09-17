# V03 - Auditoria de features

Kernel Kaggle:

`biancaaguinaga/p1-ieee-fraud-v03-feature-audit`

Estado final: `COMPLETE`

Outputs locales:

`p1/03_outputs/v03_feature_audit/`

Bitacora general de decisiones:

`p1/01_planning/decision_log_modelado.md`

## Objetivo

Auditar las features del baseline V01 combinando:

- importancia por ganancia de LightGBM;
- permutation importance en validacion;
- shift temporal medido con KS;
- cambios de missingness entre train, validacion y holdout.

La pregunta no fue "que modelo gana?", sino:

> En que features podemos confiar para seguir modelando?

## Resultado del modelo auditado

| Metrica | Valor |
| --- | ---: |
| Valid PR-AUC | 0.5954 |
| Valid ROC-AUC | 0.9257 |
| Holdout PR-AUC | 0.5436 |
| Holdout ROC-AUC | 0.9051 |
| Features auditadas | 185 |

## Clasificacion de features

| Recomendacion | Features | Interpretacion |
| --- | ---: | --- |
| `keep` | 44 | Predictivas y sin shift alto. |
| `monitor` | 9 | Predictivas, pero con shift o cambio de missingness. |
| `review` | 20 | Poco predictivas o sospechosas, con shift alto. |
| `low_priority` | 112 | Baja evidencia predictiva para esta etapa. |

## Features `keep` principales

Estas features tienen evidencia predictiva y no muestran shift alto bajo los criterios usados:

| Feature | Lectura |
| --- | --- |
| `V258`, `V257` | Variables anonimizadas con senal fuerte. |
| `C1`, `C2`, `C5`, `C6`, `C11`, `C13`, `C14` | Conteos/agregados anonimizados con alta importancia. |
| `card1`, `card2`, `card3`, `card5`, `card6` | Variables de tarjeta con senal estable. |
| `TransactionAmt`, `TransactionAmt_log1p` | El monto original y transformado aportan informacion. |
| `addr1`, `dist1` | Ubicacion/distancia anonimizada con senal. |
| `P_emaildomain`, `R_emaildomain` | Dominios de email aportan segmentacion de riesgo. |

## Features `monitor`

Estas features son predictivas, pero muestran shift temporal o cambios relevantes de missingness:

| Feature | Motivo |
| --- | --- |
| `D1n` | Muy predictiva, pero KS alto: 0.6044. |
| `D2n` | Predictiva, KS alto: 0.4002. |
| `D15n` | Predictiva, KS alto: 0.4534. |
| `D10n` | Predictiva, KS alto: 0.5256. |
| `C9` | Predictiva, KS sobre umbral: 0.1399. |
| `id_02` | Predictiva, shift y cambio de missingness. |
| `id_20` | Predictiva, shift y cambio de missingness. |
| `D11` | Predictiva, cambio fuerte de missingness: 26.27 puntos. |
| `id_01` | Predictiva, shift y cambio de missingness. |

Decision: no eliminarlas ahora, porque aportan senal. Pero deben monitorearse por ventana y no deben ser la unica base de decision.

## Features `review`

Estas features tienen shift alto y baja evidencia predictiva en esta auditoria:

`id_13`, `missing_D_count`, `M3_label`, `missing_V_count`, `id_17`, `D13`, `D6`, `D14`, `M3_freq`, `M2_label`, `M9_label`, `M7_label`, `M8_label`, `M9_freq`, `M8_freq`, `M2_freq`, `M7_freq`, `id_07`, `M1_freq`, `M1_label`.

Decision: no se eliminan del repositorio ni del analisis, pero no se deben priorizar para modelos siguientes sin una prueba especifica. Si aparecen en modelos futuros, deben justificarse por mejora en holdout y estabilidad.

## Hallazgos importantes

### 1. Las variables `D*n` son utiles pero riesgosas

`D1n`, `D2n`, `D10n` y `D15n` aparecen como predictivas, pero con alto shift. Esto confirma que las fechas relativas/anclas historicas capturan informacion de comportamiento, pero tambien cambian entre ventanas.

Implicancia: se pueden usar, pero deben estar en monitoreo de drift.

### 2. Las variables de tarjeta y monto son mas estables

`card1`, `card2`, `TransactionAmt`, `addr1` y varias `C` muestran importancia alta con menor shift. Estas son buenas candidatas para mantenerse como nucleo del modelo.

### 3. No todas las features de missingness ayudan

`missing_D_count` y `missing_V_count` fueron clasificadas como `review`: tienen shift alto y baja evidencia predictiva directa. Esto no invalida la idea de tratar faltantes como informacion, pero muestra que no todo flag agregado automaticamente es util.

### 4. `DT_hour` quedo en baja prioridad

`DT_hour` no aparece como feature fuerte. Esto es sano metodologicamente: no estamos dependiendo de una variable temporal simple para sostener el modelo.

## Decision

Para el siguiente paso:

- Usar `keep` como nucleo confiable.
- Mantener `monitor` con seguimiento de drift.
- No priorizar `review` salvo que una ablation futura demuestre mejora real.
- Mantener `low_priority` como candidatas secundarias, no como base de decision.

## Proximo paso

V04 deberia comparar modelos usando un feature set auditado:

- LightGBM con features `keep + monitor`.
- LightGBM con solo `keep`.
- CatBoost/XGBoost si el costo de ejecucion es razonable.
- Reporte por holdout, UID conocido, UID desconocido y ventanas temporales.

