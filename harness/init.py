"""Arranque de sesión: verifica el entorno y resume el estado del proyecto.

Uso:  python harness/init.py [--install] [--update-manifest]
  --install  crea el entorno virtual .venv (si no existe) e instala requirements.txt en él.
Si existe .venv, el script se relanza solo dentro de él (da igual con qué Python se invoque).
Salida compacta (unas 30 líneas como máximo). Código de salida != 0 si algo falla.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from importlib import metadata

from common import (
    ESTADOS,
    ROOT,
    VENV,
    cargar_tareas,
    configurar_salida,
    en_venv,
    errores_tareas,
    python_venv,
    reejecutar_en_venv,
    siguiente_tarea,
)

MIN_PYTHON = (3, 11)
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "harness" / "data_manifest.json"
REQUIREMENTS = ROOT / "requirements.txt"
PROGRESS = ROOT / "progress" / "PROGRESS.md"
N_CSV = 3
AGENT = ROOT / "agent.py"
AGENT_SHA256 = "ff102044ff4126604f44cfe03ea11eeb9b91f5768d4477add8847b24ec3fb376"


def nfc(texto: str) -> str:
    """Normaliza Unicode: los CSV traen tildes descompuestas (NFD) en el nombre."""
    return unicodedata.normalize("NFC", texto)


def sha256(ruta) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def lanzador() -> str:
    """Nombre del ejecutable de Python disponible en el PATH (python, py o python3)."""
    for candidato in ("python", "py", "python3"):
        if shutil.which(candidato):
            return candidato
    return "python"


def revisar_python() -> tuple[bool, list[str]]:
    v = sys.version_info
    ok = (v.major, v.minor) >= MIN_PYTHON
    minimo = ".".join(map(str, MIN_PYTHON))
    estado = "OK" if ok else f"FALLA (se requiere >= {minimo})"
    entorno = ".venv" if en_venv() else "global (sin .venv)"
    version = f"{v.major}.{v.minor}.{v.micro}"
    return ok, [f"Python {version} {estado} · lanzador: {lanzador()} · entorno: {entorno}"]


def revisar_datos(actualizar: bool) -> tuple[bool, list[str]]:
    """Compara data/raw/*.csv con el manifiesto (tamaño y sha256)."""
    reales = {nfc(p.name): p for p in sorted(RAW.glob("*.csv"))}
    if not actualizar and not MANIFEST.exists():
        return False, ["Datos: FALTA harness/data_manifest.json (--update-manifest lo regenera)"]
    if actualizar:
        archivos = [
            {"nombre": n, "bytes": p.stat().st_size, "sha256": sha256(p)}
            for n, p in reales.items()
        ]
        MANIFEST.write_text(
            json.dumps({"archivos": archivos}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        ok = len(archivos) == N_CSV
        return ok, [f"Datos: manifiesto escrito con {len(archivos)} CSV (se esperan {N_CSV})"]

    esperados = json.loads(MANIFEST.read_text(encoding="utf-8"))["archivos"]
    ok, lineas = len(esperados) == N_CSV, []
    for e in esperados:
        p = reales.get(nfc(e["nombre"]))
        if p is None:
            estado = "FALTA"
        elif p.stat().st_size != e["bytes"]:
            estado = "TAMAÑO DISTINTO"
        elif sha256(p) != e["sha256"]:
            estado = "HASH DISTINTO"
        else:
            estado = "OK"
        ok &= estado == "OK"
        lineas.append(f"  {estado:<5} {e['nombre']} ({e['bytes'] / 1e6:.1f} MB, {e['sha256'][:8]})")
    return ok, [f"Datos (data/raw): {'OK' if ok else 'FALLA'}", *lineas]


def revisar_agent() -> tuple[bool, list[str]]:
    """agent.py es el entorno original del profesor: no debe cambiar."""
    ok = AGENT.is_file() and sha256(AGENT) == AGENT_SHA256
    return ok, [f"agent.py: {'OK (sin modificar)' if ok else 'FALLA: falta o fue modificado'}"]


def revisar_dependencias() -> tuple[bool, list[str]]:
    """Comprueba que cada paquete fijado en requirements.txt está instalado."""
    faltan, distintas, total = [], [], 0
    for linea in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", linea.strip())
        if not m:
            continue
        total += 1
        nombre, version = m.groups()
        try:
            instalada = metadata.version(nombre)
        except metadata.PackageNotFoundError:
            faltan.append(nombre)
            continue
        if instalada != version:
            distintas.append(f"{nombre} {instalada} (fijada {version})")
    lineas = [f"Dependencias: {total - len(faltan)}/{total} instaladas"]
    if faltan:
        lineas.append(f"  FALTAN: {', '.join(faltan)} -> python harness/init.py --install")
    if distintas:
        lineas.append(f"  AVISO versión distinta: {', '.join(distintas)}")
    return not faltan, lineas


def revisar_kernel() -> tuple[bool, list[str]]:
    """Prueba ZeroMQ en un subproceso: si está roto, ningún notebook puede ejecutarse."""
    prueba = (
        "import zmq; s = zmq.Context().socket(zmq.REP); "
        "s.bind_to_random_port('tcp://127.0.0.1'); import ipykernel"
    )
    try:
        p = subprocess.run([sys.executable, "-c", prueba], capture_output=True, timeout=60)
        ok = p.returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    if ok:
        return True, ["Kernel de Jupyter (ZeroMQ): OK"]
    return False, [
        "Kernel de Jupyter (ZeroMQ): FALLA en este intérprete",
        "  -> python harness/init.py --install  (crea .venv con dependencias limpias)",
    ]


def instalar() -> bool:
    """Crea .venv si no existe e instala requirements.txt dentro."""
    if not python_venv().exists():
        print("Creando el entorno virtual .venv ...")
        if subprocess.run([sys.executable, "-m", "venv", str(VENV)]).returncode != 0:
            return False
    pip = [str(python_venv()), "-m", "pip", "install", "-q", "--disable-pip-version-check"]
    print("Instalando requirements.txt en .venv ...")
    return subprocess.run([*pip, "-r", str(REQUIREMENTS)], cwd=ROOT).returncode == 0


def ultima_sesion(max_lineas: int = 5) -> list[str]:
    """Líneas de la sección 'Última sesión' de progress/PROGRESS.md."""
    if not PROGRESS.exists():
        return ["  (sin progress/PROGRESS.md)"]
    lineas, dentro = [], False
    for linea in PROGRESS.read_text(encoding="utf-8").splitlines():
        if linea.startswith("## "):
            dentro = "ltima sesi" in linea
            continue
        if dentro and linea.strip():
            lineas.append("  " + linea.strip())
    return lineas[:max_lineas] or ["  (sin entradas)"]


def resumen_tareas() -> tuple[bool, list[str]]:
    tareas = cargar_tareas()
    errores = errores_tareas(tareas)
    conteo = Counter(t["estado"] for t in tareas)
    lineas = ["Tareas: " + " · ".join(f"{e} {conteo.get(e, 0)}" for e in ESTADOS)]
    lineas += [f"  ERROR tasks.json: {e}" for e in errores[:5]]
    en_curso = [t for t in tareas if t["estado"] == "in_progress"]
    if en_curso:
        lineas.append(f"EN CURSO: {en_curso[0]['id']} — {en_curso[0]['titulo']}")
    sig = siguiente_tarea(tareas)
    if sig:
        lineas.append(f"SIGUIENTE: {sig['id']} — {sig['titulo']}")
        lineas.append(f"  validación: {sig['validacion']}")
    else:
        lineas.append("SIGUIENTE: ninguna tarea pending con dependencias en done")
    return not errores, lineas


def main() -> int:
    configurar_salida()
    parser = argparse.ArgumentParser(description="Arranque de sesión del arnés.")
    parser.add_argument("--install", action="store_true", help="instala requirements.txt")
    parser.add_argument(
        "--update-manifest", action="store_true", help="regenera harness/data_manifest.json"
    )
    args = parser.parse_args()

    if args.install and not instalar():
        print("FALLA: no se pudo crear .venv o instalar requirements.txt")
        return 1
    reejecutar_en_venv([a for a in sys.argv if a != "--install"])

    print("== init.py · Nile Prime (Módulo 10) ==")
    todo_ok = True
    for revisar in (
        revisar_python,
        lambda: revisar_datos(args.update_manifest),
        revisar_agent,
        revisar_dependencias,
        revisar_kernel,
    ):
        ok, lineas = revisar()
        todo_ok &= ok
        print("\n".join(lineas))
    print("Última sesión (progress/PROGRESS.md):")
    print("\n".join(ultima_sesion()))
    ok, lineas = resumen_tareas()
    todo_ok &= ok
    print("\n".join(lineas))
    print(f"ENTORNO: {'OK' if todo_ok else 'CON FALLOS'}")
    return 0 if todo_ok else 1


if __name__ == "__main__":
    sys.exit(main())
