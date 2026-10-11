"""Tests de la lógica pura de harness/empaquetar_entrega.py (sin ejecutar notebooks)."""

import json
import zipfile

import empaquetar_entrega as emp


def _tocar(raiz, rel, texto="x"):
    ruta = raiz / rel
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")


def test_archivos_paquete_incluye_y_excluye(tmp_path):
    for rel in (
        "agent.py", "README.md", "requirements.txt", "src/a.py", "tests/t.py",
        "notebooks/py/00.py", "notebooks/00.ipynb", "data/raw/x.csv",
        "reports/figures/f.png", "reports/metrics/eda.json", "docs/d.md",
    ):
        _tocar(tmp_path, rel)
    for rel in (
        ".git/config", ".venv/lib/x.py", ".cache/huggingface/m.bin", "progress/PROGRESS.md",
        ".claude/agents/a.md", "reports/informe/figs/g.png", "src/__pycache__/a.pyc",
        "notebooks/.ipynb_checkpoints/c.ipynb", "entrega/z.zip", ".pytest_cache/v",
        "otro/suelto.txt", "reports/informe/build_informe.py",
    ):
        _tocar(tmp_path, rel)
    incluidos = {p.as_posix() for p in emp.archivos_paquete(tmp_path)}
    assert {"agent.py", "src/a.py", "data/raw/x.csv", "reports/metrics/eda.json",
            "notebooks/00.ipynb", "notebooks/py/00.py"} <= incluidos
    assert not any(
        p.startswith((".git/", ".venv/", ".cache/", "progress/", ".claude/", "entrega/",
                      "reports/informe", "otro/", ".pytest_cache/"))
        or "__pycache__" in p or ".ipynb_checkpoints" in p
        for p in incluidos
    )


def test_errores_notebook():
    sin_salidas = {"cells": [{"cell_type": "code", "outputs": []}]}
    con_error = {"cells": [{"cell_type": "code",
                            "outputs": [{"output_type": "error", "ename": "ValueError"}]}]}
    bueno = {"cells": [{"cell_type": "markdown"},
                       {"cell_type": "code", "outputs": [{"output_type": "stream"}]}]}
    assert any("sin salidas" in e for e in emp.errores_notebook(sin_salidas, "n"))
    assert any("error" in e for e in emp.errores_notebook(con_error, "n"))
    assert emp.errores_notebook(bueno, "n") == []


def test_errores_modo(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"modo": "full"}))
    (tmp_path / "b.json").write_text(json.dumps({"modo": "fast"}))
    errores = emp.errores_modo(tmp_path, ("a", "b", "c"))
    assert len(errores) == 2
    assert any("b.json" in e and "fast" in e for e in errores)
    assert any("c.json" in e for e in errores)


def test_zip_con_raiz_pdf_y_nombres_nfd(tmp_path):
    nfd = "Reséas.csv"  # "n" con tilde en NFD
    _tocar(tmp_path, "agent.py")
    _tocar(tmp_path, f"data/raw/{nfd}", "a,b")
    pdf = tmp_path / "informe.pdf"
    pdf.write_bytes(b"%PDF")
    destino = tmp_path / "salida" / "p.zip"
    n = emp.escribir_zip(tmp_path, destino, pdf)
    assert n == 3
    with zipfile.ZipFile(destino) as z:
        nombres = z.namelist()
    assert f"{emp.NOMBRE}/data/raw/{nfd}" in nombres
    assert f"{emp.NOMBRE}/{emp.PDF_NOMBRE}" in nombres


def test_sin_pdf_el_zip_se_crea(tmp_path):
    _tocar(tmp_path, "agent.py")
    assert emp.escribir_zip(tmp_path, tmp_path / "e" / "p.zip", None) == 1


def test_comprobaciones_previas_detecta_todo(tmp_path):
    errores = emp.comprobaciones_previas(tmp_path)
    assert any("00_EDA_y_preparacion.ipynb: no existe" in e for e in errores)
    assert any("eda.json: no existe" in e for e in errores)
    assert any("agent.py" in e for e in errores)
