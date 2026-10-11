"""Empaquetado de la entrega por Classroom: zip del código + verificación en carpeta limpia.

Genera ``entrega/NilePrime_AliGuerrero_codigo.zip`` (carpeta raíz ``NilePrime_AliGuerrero_codigo/``)
con los notebooks ejecutados (.ipynb con salidas, modo full), sus fuentes, ``src/``, ``tests/``,
``data/raw/``, ``agent.py``, ``requirements.txt``, ``README.md``, ``docs/``, ``harness/`` y las
salidas ``reports/figures`` y ``reports/metrics``. Excluye ``.git``, ``.venv``, ``.cache``,
``progress/``, ``.claude/``, ``reports/informe/figs`` y las cachés de Python.

Antes de empaquetar comprueba que el trabajo está en modo full (notebooks con salidas y sin
errores, JSON de métricas con ``"modo": "full"``, ``agent.py`` intacto). Después, por defecto,
descomprime el zip en un directorio temporal FUERA del repo y ejecuta allí los 5 notebooks de
principio a fin (copia con salidas limpiadas, sin ``NP_FAST``) con nbclient y el kernel de
``.venv``; el registro queda en ``entrega/verificacion.json``.

Uso:
    python harness/empaquetar_entrega.py                  # empaqueta y verifica
    python harness/empaquetar_entrega.py --sin-verificar  # solo empaqueta
    python harness/empaquetar_entrega.py --conservar      # no borra el directorio temporal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

from common import ROOT, configurar_salida, python_venv, reejecutar_en_venv

NOMBRE = "NilePrime_AliGuerrero_codigo"
ENTREGA = ROOT / "entrega"
ZIP_SALIDA = ENTREGA / f"{NOMBRE}.zip"
PDF_ORIGEN = ROOT / "reports" / "informe_nile_prime.pdf"
PDF_NOMBRE = "informe_nile_prime.pdf"
VERIFICACION = ENTREGA / "verificacion.json"
SHA256_AGENT = "ff102044ff4126604f44cfe03ea11eeb9b91f5768d4477add8847b24ec3fb376"

NOTEBOOKS = (
    "00_EDA_y_preparacion",
    "01_sentimiento",
    "02_recomendador",
    "03_abandono_random_forest",
    "04_aprendizaje_por_refuerzo",
)
METRICAS = ("eda", "sentimiento", "recomendador", "abandono", "rl")

# Entradas de la raíz del repo que viajan en el zip (lista blanca) ...
INCLUIR = (
    "agent.py",
    "requirements.txt",
    "README.md",
    "pyproject.toml",
    "notebooks",
    "src",
    "tests",
    "docs",
    "harness",
    "data/raw",
    "reports/figures",
    "reports/metrics",
)
# ... y nombres/rutas que se excluyen siempre, aunque cuelguen de una entrada incluida.
EXCLUIR_NOMBRES = {
    ".git", ".venv", ".cache", ".claude", "progress", "entrega", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints",
}
EXCLUIR_RUTAS = ("reports/informe/figs",)
EXCLUIR_SUFIJOS = (".pyc", ".pyo")


# --------------------------------------------------------------------------- lógica pura

def _excluido(rel: Path) -> bool:
    texto = rel.as_posix()
    if any(parte in EXCLUIR_NOMBRES for parte in rel.parts):
        return True
    if rel.suffix in EXCLUIR_SUFIJOS:
        return True
    return any(texto == r or texto.startswith(r + "/") for r in EXCLUIR_RUTAS)


def archivos_paquete(raiz: Path) -> list[Path]:
    """Rutas relativas (a ``raiz``) de los archivos que entran en el zip, ordenadas."""
    encontrados: set[Path] = set()
    for entrada in INCLUIR:
        ruta = raiz / entrada
        if ruta.is_file():
            candidatos = [ruta]
        elif ruta.is_dir():
            candidatos = [p for p in ruta.rglob("*") if p.is_file()]
        else:
            continue
        for p in candidatos:
            rel = p.relative_to(raiz)
            if not _excluido(rel):
                encontrados.add(rel)
    return sorted(encontrados)


def errores_notebook(nb: dict, nombre: str) -> list[str]:
    """Problemas de un .ipynb (dict JSON): sin salidas en celdas de código o con salida de error."""
    errores = []
    codigo = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    if not codigo:
        return [f"{nombre}: sin celdas de código"]
    if not any(c.get("outputs") for c in codigo):
        errores.append(f"{nombre}: sin salidas (no está ejecutado)")
    for i, celda in enumerate(nb["cells"]):
        for salida in celda.get("outputs", []) if celda.get("cell_type") == "code" else []:
            if salida.get("output_type") == "error":
                errores.append(f"{nombre}: celda {i} con salida de error ({salida.get('ename')})")
    return errores


def errores_modo(carpeta_metricas: Path, retos: tuple[str, ...] = METRICAS) -> list[str]:
    """JSON de métricas ausentes o que no están en ``"modo": "full"``."""
    errores = []
    for reto in retos:
        ruta = carpeta_metricas / f"{reto}.json"
        if not ruta.exists():
            errores.append(f"{reto}.json: no existe")
            continue
        try:
            modo = json.loads(ruta.read_text(encoding="utf-8")).get("modo")
        except (json.JSONDecodeError, AttributeError):
            errores.append(f"{reto}.json: JSON inválido")
            continue
        if modo != "full":
            errores.append(f'{reto}.json: modo "{modo}" (debe ser "full")')
    return errores


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def comprobaciones_previas(raiz: Path = ROOT) -> list[str]:
    """Todos los errores que impiden empaquetar. Lista vacía = listo."""
    errores: list[str] = []
    for nb in NOTEBOOKS:
        ruta = raiz / "notebooks" / f"{nb}.ipynb"
        if not ruta.exists():
            errores.append(f"{nb}.ipynb: no existe")
            continue
        try:
            errores += errores_notebook(json.loads(ruta.read_text(encoding="utf-8")), f"{nb}.ipynb")
        except json.JSONDecodeError:
            errores.append(f"{nb}.ipynb: JSON inválido")
    errores += errores_modo(raiz / "reports" / "metrics")
    agent = raiz / "agent.py"
    if not agent.exists():
        errores.append("agent.py: no existe")
    elif sha256(agent) != SHA256_AGENT:
        errores.append("agent.py: sha256 distinto del original (no se debe modificar)")
    return errores


def escribir_zip(raiz: Path, destino: Path, pdf: Path | None) -> int:
    """Crea el zip (carpeta raíz ``NOMBRE/``). Devuelve el nº de archivos."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    archivos = archivos_paquete(raiz)
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:  # nombres no ASCII -> UTF-8
        for rel in archivos:
            z.write(raiz / rel, f"{NOMBRE}/{rel.as_posix()}")
        if pdf is not None and pdf.exists():
            z.write(pdf, f"{NOMBRE}/{PDF_NOMBRE}")
    return len(archivos) + (1 if pdf is not None and pdf.exists() else 0)


# --------------------------------------------------------------------------- verificación

def _celda_fallida(nb) -> str:
    for i, celda in enumerate(nb.cells):
        for salida in celda.get("outputs", []) if celda.cell_type == "code" else []:
            if salida.get("output_type") == "error":
                return f"celda {i}: {salida.get('ename')}: {str(salida.get('evalue'))[:200]}"
    return "celda desconocida"


def verificar_csv(paquete: Path) -> str | None:
    """Comprueba (en un subproceso aislado) que ``src.data.find_csv`` encuentra los 3 CSV."""
    codigo = (
        "from src import data\n"
        "for p in ('Rese*.csv','Metadatos*.csv','interacciones*.csv'):\n"
        "    data.find_csv(p)\n"
    )
    r = subprocess.run([str(python_venv()), "-c", codigo], cwd=paquete, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    return None if r.returncode == 0 else (r.stderr.strip().splitlines() or ["fallo"])[-1]


def verificar_paquete(zip_ruta: Path, conservar: bool) -> tuple[list[dict], dict]:
    """Descomprime el zip fuera del repo y ejecuta los 5 notebooks desde cero."""
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError, CellTimeoutError

    temporal = Path(tempfile.mkdtemp(prefix="np_verif_"))
    if ROOT in temporal.resolve().parents:
        raise RuntimeError("el directorio temporal debe estar fuera del repositorio")
    entorno = {"HF_HOME": str(ROOT / ".cache" / "huggingface")}
    previo = {k: os.environ.get(k) for k in (*entorno, "NP_FAST")}
    os.environ.update(entorno)
    os.environ.pop("NP_FAST", None)
    registro: list[dict] = []
    info: dict = {}
    try:
        with zipfile.ZipFile(zip_ruta) as z:
            z.extractall(temporal)
        paquete = temporal / NOMBRE
        info["csv"] = verificar_csv(paquete) or "ok"
        print(f"  zip extraído (find_csv: {info['csv']}); HF_HOME = caché del repo (.cache)")
        for nb_nombre in NOTEBOOKS:
            nb = nbformat.read(paquete / "notebooks" / f"{nb_nombre}.ipynb", as_version=4)
            for celda in nb.cells:  # copia en memoria con las salidas limpiadas
                if celda.cell_type == "code":
                    celda.outputs, celda.execution_count = [], None
            n_codigo = sum(c.cell_type == "code" for c in nb.cells)
            cliente = NotebookClient(
                nb, timeout=1800, kernel_name="python3",
                resources={"metadata": {"path": str(paquete / "notebooks")}},
            )
            t0, fallo = time.time(), None
            try:
                cliente.execute()
            except CellTimeoutError:
                fallo = "una celda superó el timeout de 1800 s"
            except CellExecutionError:
                fallo = _celda_fallida(nb)
            except Exception as e:  # noqa: BLE001 - cualquier fallo del kernel es un FAIL
                fallo = f"{type(e).__name__}: {str(e)[:200]}"
            registro.append({
                "notebook": nb_nombre, "resultado": "FAIL" if fallo else "OK",
                "segundos": round(time.time() - t0, 1), "celdas_codigo": n_codigo,
                "error": fallo,
            })
            r = registro[-1]
            print(f"  {nb_nombre:<30} {r['resultado']:<4} {r['segundos']:>7.1f}s"
                  + (f"  -> {fallo}" if fallo else ""))
    finally:
        for k, v in previo.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if conservar:
            print(f"  directorio temporal conservado: {temporal}")
        else:
            shutil.rmtree(temporal, ignore_errors=True)
    info["hf_home"] = ".cache/huggingface del repositorio (solo durante la verificación)"
    return registro, info


# --------------------------------------------------------------------------- CLI

def main() -> int:
    configurar_salida()
    ap = argparse.ArgumentParser(description="Empaqueta la entrega (zip) y la verifica.")
    ap.add_argument("--sin-verificar", action="store_true", help="no ejecuta los notebooks del zip")
    ap.add_argument("--conservar", action="store_true", help="conserva el directorio temporal")
    args = ap.parse_args()

    print("== Empaquetado de la entrega ==")
    errores = comprobaciones_previas()
    if errores:
        print("FAIL comprobaciones previas:")
        for e in errores:
            print(f"  - {e}")
        if any("modo" in e or "sin salidas" in e for e in errores):
            print("  ejecuta antes `python harness/validate.py --full`")
        return 1
    print("OK   comprobaciones previas (notebooks con salidas, métricas en full, agent.py intacto)")

    pdf = PDF_ORIGEN if PDF_ORIGEN.exists() else None
    n = escribir_zip(ROOT, ZIP_SALIDA, pdf)
    print(f"OK   {ZIP_SALIDA.relative_to(ROOT)}: {n} archivos, "
          f"{ZIP_SALIDA.stat().st_size / 1e6:.1f} MB")
    if pdf is not None:
        shutil.copy2(pdf, ENTREGA / PDF_NOMBRE)
        print(f"OK   PDF copiado a entrega/{PDF_NOMBRE}")
    else:
        print("AVISO PDF pendiente (T09): el zip no lleva informe; regenerar el paquete en T10")

    if args.sin_verificar:
        print("AVISO verificación omitida (--sin-verificar)")
        return 0

    print("== Verificación en carpeta temporal (Restart & Run All, modo full) ==")
    t0 = time.time()
    registro, info = verificar_paquete(ZIP_SALIDA, args.conservar)
    total = round(time.time() - t0, 1)
    ok = all(r["resultado"] == "OK" for r in registro) and info.get("csv") == "ok"
    VERIFICACION.write_text(json.dumps({
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "zip_bytes": ZIP_SALIDA.stat().st_size,
        "pdf_incluido": pdf is not None,
        "hf_home": info["hf_home"],
        "find_csv": info["csv"],
        "segundos_total": total,
        "resultado": "OK" if ok else "FAIL",
        "notebooks": registro,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'OK  ' if ok else 'FAIL'} verificación en {total:.0f} s -> entrega/verificacion.json")
    return 0 if ok else 1


if __name__ == "__main__":
    reejecutar_en_venv()
    sys.exit(main())
