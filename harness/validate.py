"""Puerta de calidad: lint, tests, notebooks y artifacts.

Uso:  python harness/validate.py [--task ID] [--quick | --full] [--timeout SEG]
  --quick  (por defecto) define NP_FAST=1: los notebooks submuestrean y NO se
           reescribe el .ipynb.
  --full   usa todos los datos, sincroniza .py -> .ipynb y guarda el notebook
           ejecutado. Obligatorio antes de marcar done una tarea de notebook.
Código de salida != 0 si alguna comprobación falla.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

from common import (
    ROOT,
    buscar_tarea,
    cargar_tareas,
    configurar_salida,
    errores_tareas,
    reejecutar_en_venv,
)

NB_DIR = ROOT / "notebooks"
NB_SRC = NB_DIR / "py"
METRICS_DIR = ROOT / "reports" / "metrics"
LINT_DIRS = ("src", "tests", "harness")
# Esquema de reports/metrics/<reto>.json (docs/ESTANDARES.md, sección 9)
RETOS = ("sentimiento", "recomendador", "abandono", "rl")
RETOS_CON_UMBRAL = ("sentimiento", "abandono")
CLAVES_RETO = ("config", "baseline", "metricas")
RUTA_ABS = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:(?:\\{1,2}|/(?!/))[A-Za-z_]|/Users/|/home/")
MAX_DETALLE = 12
TIMEOUT = {"quick": 300, "full": 1800}

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
Resultado = tuple[str, str, str, list[str]]  # (estado, nombre, resumen, detalle)


def rel(ruta) -> str:
    return ruta.relative_to(ROOT).as_posix()


def ejecutar(args: list[str]) -> tuple[int, str]:
    """Ejecuta un módulo de Python con el intérprete actual y devuelve (código, salida)."""
    p = subprocess.run(
        [sys.executable, "-m", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return p.returncode, (p.stdout + p.stderr).strip()


def chk_tareas(tareas: list[dict]) -> Resultado:
    errores = errores_tareas(tareas)
    en_curso = sum(t["estado"] == "in_progress" for t in tareas)
    resumen = f"{len(tareas)} tareas, {en_curso} in_progress"
    return (FAIL if errores else PASS, "tasks.json", resumen, errores)


def chk_ruff() -> Resultado:
    dirs = [d for d in LINT_DIRS if (ROOT / d).is_dir()]
    codigo, salida = ejecutar(["ruff", "check", "--quiet", "--output-format", "concise", *dirs])
    if codigo == 0:
        return (PASS, "ruff", " ".join(dirs), [])
    return (FAIL, "ruff", f"código {codigo}", salida.splitlines())


def chk_pytest() -> Resultado:
    codigo, salida = ejecutar(["pytest", "-q", "--no-header", "-p", "no:cacheprovider"])
    lineas = salida.splitlines()
    ultima = lineas[-1].strip("= ") if lineas else ""
    if codigo == 0:
        return (PASS, "pytest", ultima, [])
    if codigo == 5:
        return (WARN, "pytest", "no hay tests todavía", [])
    return (FAIL, "pytest", ultima, lineas[-MAX_DETALLE:])


def rutas_absolutas(texto: str) -> list[str]:
    """Fragmentos del texto que parecen rutas absolutas (C:\\, /Users, /home)."""
    return [m.group(0) for m in RUTA_ABS.finditer(texto)]


def chk_rutas() -> Resultado:
    """Rutas absolutas en el código fuente (src, tests y fuentes de notebooks)."""
    hallazgos, n = [], 0
    for carpeta in (ROOT / "src", ROOT / "tests", NB_SRC):
        for py in sorted(carpeta.rglob("*.py")):
            n += 1
            lineas = py.read_text(encoding="utf-8").splitlines()
            for i, linea in enumerate(lineas, 1):
                if rutas_absolutas(linea):
                    hallazgos.append(f"{rel(py)}:{i}: ruta absoluta")
    if hallazgos:
        return (FAIL, "rutas", f"{len(hallazgos)} rutas absolutas", hallazgos)
    return (PASS, "rutas", f"{n} archivos .py sin rutas absolutas", [])


def errores_estructura(nb) -> list[str]:
    """La 1.ª celda es markdown con título y objetivo; la 1.ª de código prepara el entorno."""
    errores = []
    celdas = nb.cells
    if not celdas or celdas[0].cell_type != "markdown":
        return ["la primera celda debe ser markdown (título y objetivo)"]
    primera = celdas[0].source
    if not primera.lstrip().startswith("# "):
        errores.append("la primera celda debe empezar con un título '# ...'")
    if "objetivo" not in primera.lower():
        errores.append("la primera celda debe declarar el objetivo de negocio")
    codigo = [c for c in celdas if c.cell_type == "code"]
    if not codigo or "google.colab" not in codigo[0].source:
        errores.append("la primera celda de código debe ser la de entorno (detección de Colab)")
    return errores


def errores_salidas(nb) -> list[str]:
    """Salidas de error y rutas absolutas en un notebook ya ejecutado."""
    errores = []
    for i, celda in enumerate(nb.cells):
        if celda.cell_type != "code":
            continue
        for salida in celda.get("outputs", []):
            if salida.get("output_type") == "error":
                errores.append(f"celda {i}: salida de error {salida.get('ename')}")
            plano = salida.get("data", {}).get("text/plain", "")
            texto = salida.get("text", "") or json.dumps(plano)
            if rutas_absolutas(texto):
                errores.append(f"celda {i}: ruta absoluta en la salida")
    return errores


def seleccionar_notebooks(tarea: dict | None) -> list | None:
    """Fuentes jupytext a ejecutar. None = la tarea no declara notebooks."""
    fuentes = sorted(NB_SRC.glob("*.py"))
    if tarea is None:
        return fuentes
    declarados = {a.split("/")[-1][:-6] for a in tarea["artifacts"] if a.endswith(".ipynb")}
    if not declarados:
        return None
    return [f for f in fuentes if f.stem in declarados]


def chk_notebooks(tarea: dict | None, modo: str, timeout: int) -> Resultado:
    nombre = "notebooks"
    fuentes = seleccionar_notebooks(tarea)
    if fuentes is None:
        return (SKIP, nombre, "la tarea no declara notebooks", [])
    huerfanos = [
        f"{rel(nb)}: sin fuente en notebooks/py/ (no se edita el .ipynb a mano)"
        for nb in sorted(NB_DIR.glob("*.ipynb"))
        if not (NB_SRC / f"{nb.stem}.py").exists()
    ]
    if not fuentes and not huerfanos:
        return (WARN, nombre, "no hay notebooks todavía", [])
    try:
        import jupytext
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError, CellTimeoutError
    except ImportError as e:
        return (FAIL, nombre, f"falta una dependencia: {e.name}", [])

    errores = list(huerfanos)
    for fuente in fuentes:
        nb = jupytext.read(fuente)
        propios = errores_estructura(nb)
        if not propios:
            cliente = NotebookClient(
                nb,
                timeout=timeout,
                kernel_name="python3",
                resources={"metadata": {"path": str(NB_DIR)}},
            )
            try:
                cliente.execute()
            except CellTimeoutError:
                propios.append(f"una celda superó el timeout de {timeout}s")
            except CellExecutionError as e:
                propios.append(f"error de ejecución: {e.ename}: {str(e.evalue)[:200]}")
            except Exception as e:  # noqa: BLE001 - cualquier fallo del kernel es un FAIL
                propios.append(f"fallo al ejecutar: {type(e).__name__}: {str(e)[:200]}")
            propios += errores_salidas(nb)
        if not propios and modo == "full":
            jupytext.write(nb, NB_DIR / f"{fuente.stem}.ipynb")
        errores += [f"{rel(fuente)}: {e}" for e in propios]
    resumen = f"{len(fuentes)} ejecutados ({modo}, timeout {timeout}s/celda)"
    return (FAIL if errores else PASS, nombre, resumen, errores)


def errores_metricas(datos, reto: str, modo: str) -> list[str]:
    """Esquema de un JSON de métricas: 'modo' siempre; en los de reto, además config,
    baseline, metricas y (si aplica) umbral con valor y justificación."""
    if not isinstance(datos, dict):
        return ["debe ser un objeto JSON"]
    errores = []
    valor = datos.get("modo")
    if valor not in ("fast", "full"):
        errores.append("sin la clave modo (fast | full)")
    elif modo == "full" and valor != "full":
        errores.append('no tiene "modo": "full"')
    if reto not in RETOS:
        return errores
    for clave in CLAVES_RETO:
        if not isinstance(datos.get(clave), dict) or not datos[clave]:
            errores.append(f"sin la clave {clave} (objeto no vacío)")
    if reto in RETOS_CON_UMBRAL and "umbral" not in datos:
        errores.append("sin la clave umbral (obligatoria en este reto)")
    if "umbral" in datos:
        umbral = datos["umbral"]
        if not isinstance(umbral, dict) or "valor" not in umbral or not umbral.get("justificacion"):
            errores.append("umbral debe tener valor y justificacion")
    return errores


def chk_artifacts(tareas: list[dict], tarea: dict | None, modo: str) -> Resultado:
    """Existencia de los artifacts declarados (de --task o de todas las tareas done)."""
    objetivo = [tarea] if tarea else [t for t in tareas if t["estado"] == "done"]
    errores, total = [], 0
    for t in objetivo:
        for artifact in t["artifacts"]:
            total += 1
            ruta = ROOT / artifact
            if modo == "quick" and ruta.suffix == ".ipynb" and ruta.parent == NB_DIR:
                ruta = NB_SRC / f"{ruta.stem}.py"  # el .ipynb solo lo escribe --full
            if not ruta.is_file() or ruta.stat().st_size == 0:
                errores.append(f"{t['id']}: falta o está vacío {artifact}")
                continue
            if ruta.suffix == ".json" and METRICS_DIR in ruta.parents:
                try:
                    datos = json.loads(ruta.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    errores.append(f"{t['id']}: JSON inválido {artifact}")
                    continue
                errores += [
                    f"{t['id']}: {artifact} {e}" for e in errores_metricas(datos, ruta.stem, modo)
                ]
    ids = ", ".join(t["id"] for t in objetivo) or "ninguna tarea"
    resumen = f"{total - len(errores)}/{total} presentes ({ids})"
    return (FAIL if errores else PASS, "artifacts", resumen, errores)


def main() -> int:
    configurar_salida()
    parser = argparse.ArgumentParser(description="Puerta de calidad del arnés.")
    parser.add_argument("--task", metavar="ID", help="valida solo esa tarea (p. ej. T02)")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--quick", action="store_true", help="NP_FAST=1 (por defecto)")
    grupo.add_argument("--full", action="store_true", help="todos los datos")
    parser.add_argument("--timeout", type=int, help="segundos por celda de notebook")
    args = parser.parse_args()
    reejecutar_en_venv()

    modo = "full" if args.full else "quick"
    if modo == "quick":
        os.environ["NP_FAST"] = "1"
    else:
        os.environ.pop("NP_FAST", None)

    tareas = cargar_tareas()
    tarea = None
    if args.task:
        tarea = buscar_tarea(tareas, args.task)
        if tarea is None:
            print(f"FAIL: no existe la tarea '{args.task}' en harness/tasks.json")
            return 2

    resultados = [
        chk_tareas(tareas),
        chk_ruff(),
        chk_pytest(),
        chk_rutas(),
        chk_notebooks(tarea, modo, args.timeout or TIMEOUT[modo]),
        chk_artifacts(tareas, tarea, modo),
    ]

    print(f"== validate.py · modo {modo} · tarea {tarea['id'] if tarea else 'todas'} ==")
    for estado, nombre, resumen, detalle in resultados:
        print(f"{estado:<4}  {nombre:<10} {resumen}")
        for linea in detalle[:MAX_DETALLE]:
            print(f"        {linea}")
        if len(detalle) > MAX_DETALLE:
            print(f"        ... y {len(detalle) - MAX_DETALLE} más")
    fallos = sum(r[0] == FAIL for r in resultados)
    avisos = sum(r[0] == WARN for r in resultados)
    veredicto = "FAIL" if fallos else "PASS"
    print(f"RESULTADO: {veredicto} ({fallos} fallos, {avisos} avisos)")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
