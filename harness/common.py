"""Utilidades compartidas del arnés: rutas, tareas y salida por consola."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS_PATH = ROOT / "harness" / "tasks.json"
VENV = ROOT / ".venv"

FASES = ("entrega", "escalado")
ESTADOS = ("pending", "in_progress", "done", "blocked", "backlog")
CAMPOS = (
    "id",
    "titulo",
    "fase",
    "estado",
    "depende_de",
    "criterios_aceptacion",
    "artifacts",
    "validacion",
)


def configurar_salida() -> None:
    """Fuerza UTF-8 en la consola (en Windows el valor por defecto rompe tildes)."""
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def python_venv() -> Path:
    """Ruta del intérprete del entorno virtual del proyecto (.venv)."""
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def en_venv() -> bool:
    return Path(sys.prefix).resolve() == VENV.resolve()


def reejecutar_en_venv(argv: list[str] | None = None) -> None:
    """Si existe .venv y no es el intérprete actual, relanza el script dentro de él."""
    destino = python_venv()
    if destino.exists() and not en_venv():
        sys.exit(subprocess.call([str(destino), *(argv or sys.argv)]))


def cargar_tareas(ruta: Path = TASKS_PATH) -> list[dict]:
    """Devuelve la lista de tareas de tasks.json."""
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)["tareas"]


def errores_tareas(tareas: list[dict]) -> list[str]:
    """Valida el esquema y las reglas de tasks.json. Lista vacía = correcto."""
    errores: list[str] = []
    ids = [t.get("id") for t in tareas]
    for duplicado in sorted({i for i in ids if ids.count(i) > 1}):
        errores.append(f"id duplicado: {duplicado}")
    for t in tareas:
        tid = t.get("id", "?")
        for campo in CAMPOS:
            if campo not in t:
                errores.append(f"{tid}: falta el campo '{campo}'")
        if t.get("fase") not in FASES:
            errores.append(f"{tid}: fase inválida '{t.get('fase')}'")
        if t.get("estado") not in ESTADOS:
            errores.append(f"{tid}: estado inválido '{t.get('estado')}'")
        for campo in ("depende_de", "criterios_aceptacion", "artifacts"):
            if not isinstance(t.get(campo, []), list):
                errores.append(f"{tid}: '{campo}' debe ser una lista")
        for dep in t.get("depende_de", []):
            if dep not in ids:
                errores.append(f"{tid}: depende de una tarea inexistente '{dep}'")
        if t.get("estado") != "backlog" and not t.get("criterios_aceptacion"):
            errores.append(f"{tid}: sin criterios de aceptación")
    errores += [f"ciclo de dependencias: {' -> '.join(c)}" for c in ciclos_dependencias(tareas)]
    en_curso = [t["id"] for t in tareas if t.get("estado") == "in_progress"]
    if len(en_curso) > 1:
        errores.append(f"más de una tarea in_progress: {', '.join(en_curso)}")
    return errores


def ciclos_dependencias(tareas: list[dict]) -> list[list[str]]:
    """Ciclos en depende_de (búsqueda en profundidad). Cada ciclo se devuelve cerrado: A, B, A."""
    grafo = {
        t.get("id"): [d for d in t.get("depende_de", []) if isinstance(d, str)]
        for t in tareas
        if isinstance(t.get("depende_de", []), list)
    }
    ciclos, cerrados = [], set()

    def visitar(nodo: str, camino: list[str]) -> None:
        if nodo in camino:
            ciclos.append([*camino[camino.index(nodo) :], nodo])
            return
        if nodo in cerrados or nodo not in grafo:
            return
        for dep in grafo[nodo]:
            visitar(dep, [*camino, nodo])
        cerrados.add(nodo)

    for nodo in grafo:
        visitar(nodo, [])
    return ciclos


def buscar_tarea(tareas: list[dict], tid: str) -> dict | None:
    """Busca una tarea por id (sin distinguir mayúsculas)."""
    for t in tareas:
        if t["id"].lower() == tid.lower():
            return t
    return None


def siguiente_tarea(tareas: list[dict]) -> dict | None:
    """Primera tarea pending cuyas dependencias están todas en done."""
    hechas = {t["id"] for t in tareas if t["estado"] == "done"}
    for t in tareas:
        if t["estado"] == "pending" and set(t["depende_de"]) <= hechas:
            return t
    return None
