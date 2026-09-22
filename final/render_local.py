"""Embed downloaded Kaggle figures/results in the four local notebooks.

Run from the repository root after downloading all four output folders.
The notebooks detect Windows and display saved files without fitting models.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

for name in ("01_eda", "02_monitoreo_drift", "03_modelos_adaptacion", "04_politica_juego"):
    path = ROOT / f"{name}.ipynb"
    book = nbformat.read(path, as_version=4)
    NotebookClient(book, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(REPO)}}).execute()
    nbformat.write(book, path)
    print("Embedded results in", path.name)
