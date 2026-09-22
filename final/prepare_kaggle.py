"""Stage each notebook as its own private Kaggle kernel, without secrets."""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "eda": "01_eda.ipynb",
    "monitoreo": "02_monitoreo_drift.ipynb",
    "experimentacion": "03_modelos_adaptacion.ipynb",
    "sistema_final": "04_politica_juego.ipynb",
}
SLUGS = {stage: f"jeffreyamc/ptdia-final-{stage.replace('_', '-')}" for stage in SOURCES}

for stage, notebook in SOURCES.items():
    folder = ROOT / "kaggle" / stage / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    (ROOT / "kaggle" / stage / "outputs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / notebook, folder / "main.ipynb")
    shutil.copy2(ROOT / "common.py", folder / "common.py")
    shutil.copy2(ROOT / "v01_features.json", folder / "v01_features.json")
    if stage in ("monitoreo", "experimentacion"):
        source = ROOT.parent / "03_outputs" / "v01_baseline_temporal"
        target = folder / "v01_reference"
        target.mkdir(exist_ok=True)
        for name in ("baseline_valid_predictions.csv", "baseline_holdout_predictions.csv"):
            shutil.copy2(source / name, target / name)
    metadata = {
        "id": SLUGS[stage], "title": f"ptdia-final-{stage.replace('_', '-')}", "code_file": "main.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": stage == "monitoreo", "enable_internet": False,
        "competition_sources": ["ieee-fraud-detection"],
        "dataset_sources": [],
        "kernel_sources": ([SLUGS["experimentacion"]] if stage == "sistema_final" else
                           ["biancaaguinaga/p1-ieee-fraud-v01-baseline-temporal"] if stage in ("monitoreo", "experimentacion") else []),
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(stage, folder)
