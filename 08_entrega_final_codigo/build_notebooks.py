"""Generate four standalone Kaggle notebooks from the reviewed shared source."""

import ast
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("eda", "01_eda.ipynb", "EDA temporal y auditoría de fuga",
     "Explora periodos relativos de 30 días, muestra prevalencia y faltantes antes del holdout, y fija el contrato de disponibilidad de datos."),
    ("experimentacion", "02_experimentacion.ipynb", "Comparación y selección por validación temporal",
     "Ajusta dos modelos tradicionales y dos avanzados con folds pasados; compara ventanas deslizantes y guarda sólo predicciones de validación."),
    ("costos", "03_costos_escenarios.ipynb", "Simulación de cuatro escenarios de costo",
     "Selecciona una política por menor arrepentimiento máximo en validación; nunca lee las etiquetas de holdout."),
    ("modelo_final", "04_modelo_final.ipynb", "Modelo final y control de calidad",
     "Aplica la elección congelada, evalúa el holdout retrospectivo y reproduce el ciclo de monitoreo, reentrenamiento y promoción."),
]

def code_sections(path: Path, omit_core_import=False):
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    nodes = ast.parse(source).body
    sections = []
    current, line_count = [], 0
    for node in nodes:
        if omit_core_import and isinstance(node, ast.ImportFrom) and node.module == "core":
            continue
        chunk = "".join(lines[node.lineno - 1:node.end_lineno])
        if current and line_count + chunk.count("\n") > 110:
            sections.append("\n\n".join(current))
            current, line_count = [], 0
        current.append(chunk)
        line_count += chunk.count("\n")
    if current:
        sections.append("\n\n".join(current))
    return sections


def md(value, ident):
    return {"cell_type": "markdown", "id": ident, "metadata": {}, "source": value.splitlines(keepends=True)}


def code(value, ident):
    return {"cell_type": "code", "execution_count": None, "id": ident,
            "metadata": {}, "outputs": [], "source": value.splitlines(keepends=True)}


def notebook(stage, title, purpose):
    cells = [md(f"# {title}\n\n{purpose}\n\n**Protocolo:** IEEE-CIS, orden temporal 70/15/15, siete días de demora de etiquetas, semilla 42. El holdout fue inspeccionado en versiones anteriores: toda evaluación allí es retrospectiva.\n", "intro"),
             md("## Entorno y rutas\n\nEl mismo notebook corre en Kaggle con la competencia adjunta o localmente tras descargar los CSV.\n", "environment"),
             code("from pathlib import Path\nROOT = Path.cwd()\nif (ROOT / '08_entrega_final_codigo').exists():\n    ROOT = ROOT / '08_entrega_final_codigo'\n__file__ = str(ROOT / 'stages.py')\nprint('Paquete:', ROOT)", "setup")]
    for filename, label, omit in (("core.py", "datos, modelos y costos", False),
                                   ("stages.py", "etapas y verificaciones", True)):
        for number, section in enumerate(code_sections(ROOT / filename, omit), 1):
            cells.append(md(f"## Implementación compartida: {label} ({number})\n\nCódigo sincronizado desde `{filename}` por `build_notebooks.py`.\n", f"{filename[:4]}-note-{number}"))
            cells.append(code(section, f"{filename[:4]}-code-{number}"))
    cells.append(md("## Ejecutar y revisar resultados\n\nSe guardan CSV/JSON y un `run_summary.json` con estado final. Las decisiones de modelo y costo se congelan antes de evaluar el holdout.\n", "execute-note"))
    cells.append(code(f"result = run('{stage}', ROOT)\nprint('Etapa completa:', result)\nprint((result / 'run_summary.json').read_text()[:1000])", "execute"))
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                           "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", help="Explicitly authorized Kaggle account for private kernel staging")
    args = parser.parse_args()
    if args.owner and not args.owner.replace("-", "").replace("_", "").isalnum():
        parser.error("--owner must be a Kaggle username")
    slugs = {name: f"{args.owner}/ptdia-entrega-final-{name.replace('_', '-')}"
             for name, *_ in STAGES} if args.owner else {}
    for stage, name, title, purpose in STAGES:
        book = notebook(stage, title, purpose)
        (ROOT / name).write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        if not args.owner:
            print(name, "cells", len(book["cells"]), "Kaggle staging skipped: --owner required")
            continue
        folder = ROOT / "kaggle" / stage / "input"
        folder.mkdir(parents=True, exist_ok=True)
        (ROOT / "kaggle" / stage / "outputs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, folder / "main.ipynb")
        sources = []
        if stage in ("costos", "modelo_final"):
            sources.append(slugs["experimentacion"])
        if stage == "modelo_final":
            sources.append(slugs["costos"])
        metadata = {"id": slugs[stage], "title": f"ptdia-entrega-final-{stage.replace('_', '-')}",
                    "code_file": "main.ipynb", "language": "python", "kernel_type": "notebook",
                    "is_private": True, "enable_gpu": False, "enable_internet": False,
                    "competition_sources": ["ieee-fraud-detection"], "dataset_sources": [],
                    "kernel_sources": sources, "model_sources": []}
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(name, "cells", len(book["cells"]), "slug", slugs[stage])


if __name__ == "__main__":
    main()
