# Proyecto 1 — IEEE-CIS Fraud Detection

Sistema inteligente adaptativo sobre datos no estacionarios: un clasificador de
fraude, una política que traduce el score en acción bajo capacidad limitada, y
un lazo de monitoreo que decide cuándo actualizar el modelo.

## Informe

**[`REPORTE.md`](REPORTE.md)** — documento técnico completo.

**[`documentos/informe.pdf`](documentos/informe.pdf)** — versión de 8 páginas para entrega ·
**[`presentacion.pdf`](documentos/presentacion.pdf)** ·
**[`infografia.pdf`](documentos/infografia.pdf)**

## Notebooks

Se ejecutan en orden; cada uno guarda sus salidas en `kaggle/<etapa>/outputs/`.

| | Notebook | Qué responde |
| --- | --- | --- |
| 1 | [`01_eda.ipynb`](01_eda.ipynb) | Estructura, desbalance y evolución temporal del fraude |
| 2 | [`02_monitoreo_drift.ipynb`](02_monitoreo_drift.ipynb) | Qué señales detectan el cambio de distribución, y cuáles no |
| 3 | [`03_modelos_adaptacion.ipynb`](03_modelos_adaptacion.ipynb) | 27 configuraciones de modelo × estrategia de actualización |
| 4 | [`04_politica_juego.ipynb`](04_politica_juego.ipynb) | Decisión, costo y sensibilidad a los supuestos |

## Análisis complementarios

| Script | Pregunta que responde |
| --- | --- |
| [`significance_analysis.py`](significance_analysis.py) | ¿La ventaja de adaptar sobrevive un test pareado? |
| [`calibration.py`](calibration.py) | ¿El score se puede comunicar como probabilidad? |
| [`leakage_impact.py`](leakage_impact.py) | ¿Cuánto vale la fuga de cuartiles del monto? |
| [`temporal_features.py`](temporal_features.py) | ¿Aportan los lags y agregaciones por entidad? |
| [`traditional_temporal.py`](traditional_temporal.py) | ¿Responden los modelos tradicionales a la adaptación? |

## API de decisión

[`serving/`](serving/README.md) — FastAPI sobre Cloud Run. Devuelve score,
probabilidad calibrada y acción en una sola respuesta.

```bash
curl https://fraude-api-agz3gnkzgq-uc.a.run.app/health
```

## Reproducir

```bash
pip install -e .
python verify_results.py   # cortes temporales, cupo y reconciliación de costo
```

`verify_results.py` recomprueba los resultados contra los CSV sin reentrenar
nada. Para la corrida completa hacen falta los datos crudos (~1.3 GB), que se
descargan de la competencia `ieee-fraud-detection` de Kaggle a
`ieee-fraud-detection/`; `prepare_kaggle.py` arma los cuatro kernels.

## Bitácora experimental

[`archivo/`](archivo/) conserva el enunciado del curso y las versiones V01–V07
que precedieron a esta entrega. Se mantiene por trazabilidad de decisiones; las
cifras vigentes son las de `REPORTE.md`.
