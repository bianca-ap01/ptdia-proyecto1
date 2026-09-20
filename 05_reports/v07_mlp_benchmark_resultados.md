# V07 - Benchmark MLP y EDA complementario

Kernel Kaggle publico:
https://www.kaggle.com/code/biancaaguinaga/p1-ieee-fraud-v07-mlp-benchmark

## Objetivo

Incorporar dos aportes del notebook de referencia de Fabryzzio:

1. Revisar si su EDA complementa el EDA propio del proyecto.
2. Agregar una comparacion con MLP bajo el mismo split temporal 70/15/15 usado por V01-V06.

El MLP no se incorpora como reemplazo automatico del modelo vigente. Se incorpora como benchmark de deep learning para responder a la rubrica y verificar empiricamente si una red densa generica aporta frente a modelos tabulares de boosting.

## Notebook de referencia revisado

Fuente:
https://www.kaggle.com/code/fabryzziomezatorres/ieee-fraud-eda-riesgos

El notebook fue descargado a:

- `p1/00_brief/reference_kaggle/fabryzziomezatorres_ieee_fraud_eda_riesgos/`

## Aportes de EDA que complementan el proyecto

| Hallazgo del notebook de referencia | Decision en nuestro proyecto |
| --- | --- |
| El faltante de `id_*` y `Device*` es estructural por el left join con identity. | Mantener `has_identity` y no imputar como si fuera error aleatorio. |
| Las columnas `V` faltan por bloques, no de forma independiente. | Mantener conteos de faltantes y usar V03 para monitorear bloques con missingness cambiante. |
| `TransactionAmt` tiene cola pesada y algunos montos con mas de dos decimales. | No eliminar outliers de monto; en V07 se agrega `TransactionAmt_multi_decimal`. |
| Hay riesgo de que clientes/UID recurrentes crucen cortes temporales. | Mantener evaluacion separada UID conocido/desconocido; no adoptar features UID como modelo unico si degradan UID desconocido. |
| El MLP necesita imputacion y escalado; no maneja `NaN` ni escalas dispares como LightGBM. | Ejecutar MLP como benchmark separado, no como sustituto directo de V01. |

## Diferencias metodologicas importantes

No se adopta literalmente todo el pipeline del notebook de referencia porque usa un split 60/gap/valid/test y una politica de costo distinta. Para conservar comparabilidad con V01-V06, V07 usa:

- split temporal 70/15/15;
- misma logica de features base de V01;
- comparacion por PR-AUC y ROC-AUC;
- evaluacion global y por UID conocido/desconocido;
- LightGBM como referencia, regresion logistica SGD como baseline tradicional y MLP Keras ponderado como benchmark deep learning.

## Resultados globales

| Modelo | Valid PR-AUC | Valid ROC-AUC | Holdout PR-AUC | Holdout ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| LightGBM referencia V01 surface | 0.5885 | 0.9250 | 0.5313 | 0.9055 |
| SGD logistic tradicional | 0.2974 | 0.7964 | 0.1638 | 0.7696 |
| MLP Keras ponderado | 0.4476 | 0.8747 | 0.2218 | 0.8496 |

## Resultados holdout por UID

| Modelo | UID conocido PR-AUC | UID desconocido PR-AUC |
| --- | ---: | ---: |
| LightGBM referencia V01 surface | 0.4814 | 0.5567 |
| SGD logistic tradicional | 0.0570 | 0.3068 |
| MLP Keras ponderado | 0.0688 | 0.4875 |

## Lectura

El MLP mejora al baseline lineal tradicional en validacion, pero cae fuerte en holdout y queda lejos de LightGBM. Esto es coherente con la literatura practica de datos tabulares: un MLP denso necesita escalado, imputacion y ajuste cuidadoso, mientras que boosting sobre arboles aprovecha mejor faltantes, interacciones no lineales y variables heterogeneas.

El resultado por UID tambien confirma que el MLP no debe reemplazar a V01: en UID conocido cae a PR-AUC 0.0688, y aunque en UID desconocido llega a 0.4875, sigue por debajo de la referencia LightGBM.

## Decision esperada

V07 no reemplaza a V01. Se acepta como benchmark adicional de deep learning y como evidencia de que, para esta primera version, LightGBM sigue siendo el modelo base mas defendible. La regresion logistica SGD queda como baseline tradicional simple y el MLP como comparacion de red neuronal.

## Artefactos

- `p1/02_kaggle_kernels/v07_mlp_benchmark/main.py`
- `p1/00_brief/reference_kaggle/fabryzziomezatorres_ieee_fraud_eda_riesgos/`
- `p1/03_outputs/v07_mlp_benchmark_kaggle/v07_mlp_benchmark_metrics.csv`
- `p1/03_outputs/v07_mlp_benchmark_kaggle/v07_mlp_benchmark_summary.json`
- `p1/03_outputs/v07_mlp_benchmark_kaggle/v07_mlp_report.md`
- `p1/03_outputs/v07_mlp_benchmark_kaggle/plots/`
