# Arnés de ingeniería — referencia

Detalle de lo que `CLAUDE.md` resume. El caso y los criterios de cada reto están en
[CONTEXTO_CASO.md](CONTEXTO_CASO.md).

## Scripts (Python, multiplataforma)
Usa `python`; si no existe en el PATH, `py`. `init.py` imprime cuál detecta.
Si existe `.venv/`, ambos scripts **se relanzan solos dentro de él**, se invoquen con el
Python que sea: no hace falta activar nada.

| Comando | Qué hace |
|---|---|
| `python harness/init.py` | Verifica Python, los 3 CSV contra `data_manifest.json`, que `agent.py` no cambió, las dependencias y que ZeroMQ pueda arrancar un kernel; resume progreso y SIGUIENTE tarea. |
| `python harness/init.py --install` | Crea `.venv` (si no existe), instala `requirements.txt` en él y verifica. |
| `python harness/init.py --update-manifest` | Regenera el manifiesto (solo si los datos cambian a propósito). |
| `python harness/validate.py --task T02 --quick` | Puerta de calidad rápida (`NP_FAST=1`). |
| `python harness/validate.py --task T02 --full` | Puerta de calidad completa; escribe el `.ipynb` ejecutado. |
| `python harness/validate.py --full` | Todo el repo, todos los notebooks. |

## Qué comprueba `validate.py`
1. **tasks.json**: esquema, dependencias existentes y sin ciclos, una sola tarea `in_progress`
   (`init.py` hace la misma comprobación en cada arranque).
2. **ruff** sobre `src/`, `tests/` y `harness/` (config en `pyproject.toml`).
3. **pytest** (sin tests todavía = aviso).
4. **rutas**: ninguna ruta absoluta (`C:\`, `/Users`, `/home`) en `src/`, `tests/` ni `notebooks/py/`.
5. **notebooks**: convierte `notebooks/py/*.py` con jupytext, ejecuta con nbclient (directorio de
   trabajo `notebooks/`, timeout por celda 300 s en quick y 1800 s en full, ajustable con
   `--timeout`), y falla si hay salidas de error, rutas absolutas en las salidas, falta la celda
   de título/objetivo o la de entorno, o existe un `.ipynb` sin fuente en `notebooks/py/`.
   Con `--task`, solo los notebooks declarados en sus artifacts. Sin notebooks = aviso.
6. **artifacts**: existen y no están vacíos (los de `--task`, o los de todas las tareas `done`).
   Los JSON de `reports/metrics/` deben ser válidos y llevar `"modo"` (`fast` o `full`); en
   `--full` debe ser `full`. Los JSON de reto deben tener además `config`, `baseline` y
   `metricas` (objetos no vacíos) y, en `sentimiento` y `abandono`, `umbral` con `valor` y
   `justificacion`. En `--quick`, de un artifact `.ipynb` se exige su fuente `.py`.

`--quick` ejecuta los notebooks en memoria y **no** reescribe el `.ipynb`; `--full` lo
sincroniza y lo guarda ejecutado. Tras un `--quick`, las figuras y métricas de `reports/`
son de submuestra: solo las de `--full` se entregan y se confirman en git.

## Contrato de un notebook
- Fuente: `notebooks/py/NN_nombre.py` en formato *percent* de jupytext. El `.ipynb` de
  `notebooks/` lo genera `validate.py --full`; **nunca se edita a mano**.
- Celda 1 (markdown): `# Título` y el **objetivo** de negocio (debe aparecer la palabra "objetivo").
- Primera celda de código = **entorno**: detecta Colab (`'google.colab' in sys.modules`),
  prepara el proyecto, resuelve la raíz sin rutas absolutas, instala lo que falte (ver
  "Dependencias por entorno"), imports y `SEED = 42`.
- Lee `FAST = os.environ.get("NP_FAST") == "1"` y, si está activo, submuestrea datos y reduce
  iteraciones para terminar en pocos minutos.
- Cada JSON de métricas incluye `"modo": "fast"` o `"modo": "full"`. Los de reto
  (`sentimiento`, `recomendador`, `abandono`, `rl`) siguen además el esquema de
  [ESTANDARES.md](ESTANDARES.md): `config`, `baseline`, `metricas` y `umbral` si aplica.
- Los notebooks de reto cumplen [ESTANDARES.md](ESTANDARES.md) y terminan con la ficha del modelo.
- Independiente (Restart & Run All), sin errores sin capturar, en español, con el QUÉ y el
  POR QUÉ antes de cada bloque.
- **Audiencia técnica** (profesor, AI Engineer, data scientist). El markdown es técnico,
  preciso y conciso: objetivo, supuestos, decisiones de diseño y trade-offs, métricas con su
  interpretación, riesgos (fuga, desbalance, sesgo) y limitaciones. Terminología correcta: no
  se simplifican los conceptos. Cada sección cierra con una **Conclusión técnica** y una línea
  **Implicación de negocio** (una frase, que alimenta el informe). El lenguaje no técnico es
  exclusivo del informe PDF (T09).
- Figuras en `reports/figures/NN_nombre.png` (dpi=150); métricas en `reports/metrics/<reto>.json`.

## Dependencias por entorno
- **Local**: el `.venv` con `requirements.txt` (versiones fijadas). La celda de entorno **no
  instala nada** en local.
- **Colab**: la celda de entorno **NO reinstala** `numpy`, `pandas`, `scikit-learn`,
  `matplotlib`, `seaborn` ni `torch`. Colab ya los trae y cambiarlos obliga a reiniciar el
  entorno, lo que rompe *Run All*. Solo instala con `%pip install -q paquete==versión` las
  dependencias que **falten** (comprobándolo antes, p. ej. con `importlib.util.find_spec`),
  siempre con versión fijada y dentro de `if EN_COLAB:`.
- El código no debe depender de detalles de una versión concreta de las librerías que Colab
  trae (API estable de pandas, scikit-learn y matplotlib).

| Entorno | Python | Librerías base | Estado |
|---|---|---|---|
| Local (Windows 11, `.venv`) | 3.13.9 | pandas 2.3.3, numpy 2.3.5, scikit-learn 1.7.2, matplotlib 3.10.6, seaborn 0.13.2, joblib 1.5.2, jupytext 1.19.4, nbclient 0.10.2 | Probado el 2026-10-10 (arnés y ejecución de notebooks) |
| Google Colab | las que traiga el runtime | las preinstaladas por Colab (sin fijar) + las que falten, fijadas | **Pendiente**: las versiones reales se registran aquí al hacer la prueba de T07 |

## Ciclo de una tarea
`pending` → `in_progress` (solo una) → lector (contexto mínimo) → implementador
(`validate.py --quick`) → orquestadora (`validate.py --full` si hay notebook) → revisor →
`done` + nota de sesión + `PROGRESS.md` + commit. Si el revisor da FAIL, vuelve al
implementador con los hallazgos (máximo 2 ciclos; después se pregunta a Ali).

## progress/
- `PROGRESS.md`: estado actual, menos de 40 líneas, se sobrescribe. `init.py` lee su sección
  `## Última sesión`.
- `sessions/AAAA-MM-DD_HHMM_<tarea>.md`: qué se hizo, resultados, problemas, siguiente paso
  (máximo 25 líneas).
- `decisions.md`: fecha, contexto, decisión y alternativas descartadas.
