"""Genera el informe PDF de Nile Prime para público NO técnico.

Flujo (un solo comando, desde la raíz del proyecto):

    python reports/informe/build_informe.py

1. Lee las métricas de ``reports/metrics/*.json`` que producen los notebooks (modo ``full``).
2. Consolida las cifras del informe en ``reports/metrics/resumen.json`` (ninguna cifra se copia
   a mano: todas salen de los JSON de los notebooks).
3. Dibuja gráficos simples para negocio en ``reports/informe/figs/`` (SVG).
4. Rellena ``reports/informe/plantilla.html.j2`` (Jinja2) y guarda ``reports/informe/informe.html``.
5. Imprime el HTML a PDF con un navegador Chromium sin interfaz (Edge o Chrome en Windows,
   Chromium en Linux/macOS, o Playwright si está instalado) en ``reports/informe_nile_prime.pdf``.

Variables de entorno opcionales:
    NP_CHROME   ruta explícita al ejecutable de Chrome/Edge/Chromium.
    NP_SOLO_HTML=1  no genera el PDF (útil para revisar el HTML).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
from jinja2 import Environment, FileSystemLoader, StrictUndefined  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
METRICAS = RAIZ / "reports" / "metrics"
FIGS = AQUI / "figs"
PDF_SALIDA = RAIZ / "reports" / "informe_nile_prime.pdf"
HTML_SALIDA = AQUI / "informe.html"

# Paleta del informe (coherente con la portada y el CSS de la plantilla).
TINTA = "#1B2430"
NILO = "#0F766E"  # color de "nuestra solución"
ARENA = "#C08A3E"  # color de contraste / referencia destacada
GRIS = "#A3ACB9"
GRIS_OSCURO = "#5B6573"
ALERTA = "#B4533A"


# --------------------------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------------------------
def cargar(nombre: str) -> dict:
    """Carga ``reports/metrics/<nombre>.json`` y exige que venga de una ejecución completa."""
    ruta = METRICAS / f"{nombre}.json"
    with open(ruta, encoding="utf-8") as fh:
        datos = json.load(fh)
    if datos.get("modo") != "full":
        raise ValueError(
            f"{ruta.name} tiene modo={datos.get('modo')!r}: ejecuta el notebook en modo full "
            "(validate.py --full) antes de generar el informe."
        )
    return datos


def g(datos: dict, *claves, defecto=None):
    """Acceso anidado tolerante: ``g(d, 'a', 'b')`` → ``d['a']['b']`` o ``defecto``."""
    actual = datos
    for c in claves:
        if not isinstance(actual, dict) or c not in actual:
            return defecto
        actual = actual[c]
    return actual


def es_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def fmt(x: float, dec: int = 1) -> str:
    """Número con formato español (coma decimal, punto de miles)."""
    texto = f"{x:,.{dec}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x: float | None, dec: int = 1) -> str:
    return "—" if x is None else f"{fmt(100 * x, dec)} %"


def estilo_ejes(ax) -> None:
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(GRIS)
    ax.tick_params(colors=GRIS_OSCURO, length=0, labelsize=10)
    ax.xaxis.label.set_color(GRIS_OSCURO)
    ax.yaxis.label.set_color(GRIS_OSCURO)


def guardar(fig, nombre: str) -> str:
    FIGS.mkdir(parents=True, exist_ok=True)
    ruta = FIGS / f"{nombre}.svg"
    fig.savefig(ruta, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return f"figs/{nombre}.svg"


plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "Inter", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "svg.fonttype": "path",
    }
)


# --------------------------------------------------------------------------------------------
# Consolidación de cifras (→ resumen.json)
# --------------------------------------------------------------------------------------------
EDADES = {"youth": "Jóvenes", "adults": "Adultos", "seniors": "Mayores"}
GENEROS = {"Drama": "drama", "Comedy": "comedia", "Horror": "terror", "Romance": "romance",
           "Action": "acción", "Sci-Fi": "ciencia ficción", "Crime": "crimen", "Thriller": "suspense"}


PLURALES = {"Drama": "dramas puros", "Comedy": "comedias puras", "Horror": "películas de terror puras",
            "Romance": "romances puros"}


def segmentos_riesgo(aba: dict) -> list[dict] | None:
    """Segmentos edad × género exacto (T05b), en formato listo para el gráfico de negocio."""
    filas = g(aba, "techo_de_senal", "segmentos_riesgo", "edad_x_genero_exacto")
    if not filas:
        return None
    segs = [
        {
            "etiqueta": f"{EDADES.get(f['edad'], f['edad'])} · {GENEROS.get(f['genero'], f['genero'])} (único género)",
            "tasa": f["tasa_train"],
            "ic95": f["ic95_train"],
            "n": f["n_train"],
            "riesgo": bool(f.get("riesgo")),
            "corta": f"{EDADES.get(f['edad'], f['edad']).lower()} con "
                     f"{PLURALES.get(f['genero'], f['genero'])}",
        }
        for f in filas
    ]
    global_ = g(aba, "techo_de_senal", "segmentos_riesgo", "tasa_global_train")
    if global_ is not None:
        segs.append({"etiqueta": "Media de todas las visualizaciones", "tasa": global_, "ic95": None,
                     "n": None, "riesgo": False, "referencia": True})
    return segs


def consolidar() -> dict:
    eda = cargar("eda")
    sen = cargar("sentimiento")
    rec = cargar("recomendador")
    aba = cargar("abandono")
    rl = cargar("rl")

    # ---- Datos -----------------------------------------------------------------------------
    datos = {
        "resenas": g(eda, "filas", "resenas"),
        "peliculas": g(eda, "filas", "metadatos"),
        "interacciones": g(eda, "filas", "interacciones"),
        "usuarios": g(eda, "usuarios"),
        "abandono_global": g(eda, "abandono_global", "tasa"),
        "abandono_global_ic": g(eda, "abandono_global", "ic95"),
        "duplicados_resenas": g(eda, "duplicados", "resenas_por_texto"),
        "tiene_tiempo_visto": g(eda, "tiene_columna_tiempo_visto"),
    }

    # ---- Reto 1: sentimiento ---------------------------------------------------------------
    mt = g(sen, "modelos_test", defecto={})
    nombres_sen = {
        "Baseline (Dummy estratificado)": "Azar (referencia)",
        "Naive Bayes": "Naive Bayes",
        "Regresión Logística": "Regresión logística",
        "LinearSVC": "LinearSVC",
    }
    modelos_sen = []
    for clave, etiqueta in nombres_sen.items():
        if clave in mt:
            modelos_sen.append(
                {
                    "nombre": etiqueta,
                    "accuracy": g(mt, clave, "accuracy", "valor"),
                    "ic": g(mt, clave, "accuracy", "ic95"),
                    "final": False,
                }
            )
    if g(sen, "distilbert", "accuracy", "valor") is not None:
        modelos_sen.append(
            {
                "nombre": "DistilBERT (IA preentrenada)",
                "accuracy": g(sen, "distilbert", "accuracy", "valor"),
                "ic": g(sen, "distilbert", "accuracy", "ic95"),
                "final": False,
            }
        )
    clave_final = next((k for k in mt if "modelo final" in k), None)
    modelos_sen.append(
        {
            "nombre": "Modelo elegido",
            "accuracy": g(sen, "metricas", "accuracy", "valor"),
            "ic": g(sen, "metricas", "accuracy", "ic95"),
            "final": True,
        }
    )
    lat_final = g(sen, "costo_operativo", clave_final, "latencia_lote1_ms") if clave_final else None
    sentimiento = {
        "accuracy": g(sen, "metricas", "accuracy", "valor"),
        "accuracy_ic": g(sen, "metricas", "accuracy", "ic95"),
        "f1": g(sen, "metricas", "f1", "valor"),
        "umbral": g(sen, "umbral", "valor"),
        "costo_fp": g(sen, "umbral", "costos", "C_FP"),
        "costo_fn": g(sen, "umbral", "costos", "C_FN"),
        "distilbert_accuracy": g(sen, "distilbert", "accuracy", "valor"),
        "distilbert_latencia_ms": g(sen, "costo_operativo", "DistilBERT-SST2", "latencia_lote_ms"),
        "latencia_final_ms": lat_final,
        "resenas_usadas": g(sen, "config", "particion", "reseñas_totales"),
        "n_test": g(sen, "config", "particion", "test"),
        "modelos": modelos_sen,
    }

    # ---- Reto 2: recomendador --------------------------------------------------------------
    rt = g(rec, "recomendadores_test", defecto={})
    dosis = "0.6"
    sem = g(rec, "senal_sembrada", dosis, defecto={})

    def hit(bloque, nombre):
        return g(bloque, nombre, "hitrate@10", "valor")

    recomendador = {
        "real": {
            "Al azar": hit(rt, "Aleatorio"),
            "Lo más popular": hit(rt, "Popularidad global"),
            "Híbrido (nuestro)": hit(rt, "Híbrido"),
        },
        "sembrada": {
            "Al azar": hit(g(sem, "recomendadores", defecto={}), "Aleatorio"),
            "Lo más popular": hit(g(sem, "recomendadores", defecto={}), "Popularidad global"),
            "Híbrido (nuestro)": hit(g(sem, "recomendadores", defecto={}), "Híbrido"),
        },
        "dosis": float(dosis),
        "p_sembrada_vs_popularidad": g(sem, "hibrido_menos", "Popularidad global", "hit", "p_mcnemar"),
        "azar_teorico": g(rec, "baseline", "azar_teorico", "hit"),
        "cobertura_hibrido": g(rec, "cobertura_catalogo", "Híbrido"),
        "cobertura_popularidad": g(rec, "cobertura_catalogo", "Popularidad global"),
        "puesto_frio_hibrido": g(
            rec, "arranque_en_frio", "item_sin_interacciones_puesto_de_The_Godfather", "Híbrido"
        ),
        "puesto_frio_colaborativo": g(
            rec, "arranque_en_frio", "item_sin_interacciones_puesto_de_The_Godfather", "Colaborativo"
        ),
    }
    hs, ps = recomendador["sembrada"]["Híbrido (nuestro)"], recomendador["sembrada"]["Lo más popular"]
    recomendador["multiplo_sembrada"] = hs / ps if es_num(hs) and es_num(ps) and ps else None

    # ---- Reto 3: abandono ------------------------------------------------------------------
    deciles = g(aba, "tasa_abandono_por_decil_test", defecto={})
    abandono = {
        "auc": g(aba, "metricas", "roc_auc", "valor"),
        "auc_ic": g(aba, "metricas", "roc_auc", "ic95"),
        "p_permutacion": g(aba, "prueba_permutacion", "p_valor"),
        "auc_techo": g(aba, "techo_de_senal", "oraculo_test", "Oráculo (binario)", "valor"),
        "razon_equilibrio_segmento": g(aba, "techo_de_senal", "costo_segmento", "razon_equilibrio_fn_fp"),
        "umbral": g(aba, "umbral", "valor"),
        "umbral_degenera": g(aba, "umbral", "degenera"),
        "razon_equilibrio": g(aba, "umbral", "razon_equilibrio_marcar_a_todos"),
        "costo_fn": g(aba, "umbral", "costos", "C_FN"),
        "costo_fp": g(aba, "umbral", "costos", "C_FP"),
        "deciles": {int(k): v for k, v in deciles.items()},
        "segmentos": segmentos_riesgo(aba),
        "tasa_global_train": g(aba, "techo_de_senal", "segmentos_riesgo", "tasa_global_train"),
    }
    riesgo = [x for x in (abandono["segmentos"] or []) if x.get("riesgo")]
    abandono["segmentos_riesgo"] = riesgo
    if riesgo and abandono["tasa_global_train"]:
        abandono["multiplo_segmento"] = min(x["tasa"] for x in riesgo) / abandono["tasa_global_train"]
    if abandono["deciles"]:
        resto = [v for k, v in abandono["deciles"].items() if k != 10]
        abandono["decil_top"] = abandono["deciles"].get(10)
        abandono["decil_resto_media"] = sum(resto) / len(resto) if resto else None

    # ---- Reto 4: RL ------------------------------------------------------------------------
    base = g(rl, "baseline", "metricas", defecto={})

    def bloque_rl(modo: str) -> dict:
        return {
            "hibrido": g(base, modo, "hibrido_top1"),
            "azar": g(base, modo, "aleatoria"),
            "optimo": g(base, modo, "optimo"),
            "rl": g(rl, "metricas", f"qlearning_recompensa_final_{modo}", "valor"),
            "mejora_vs_hibrido": g(rl, "metricas", f"mejora_qlearning_vs_hibrido_top1_{modo}", "valor"),
            "regret_rl": g(rl, "metricas", f"qlearning_regret_acumulado_{modo}", "valor"),
        }

    desc = g(rl, "por_modo", "original", "descubrimiento_interacciones_del_grupo", defecto={})
    agente_rapido = g(desc, "Q-learning (película)", defecto={})
    refuerzo = {
        "original": bloque_rl("original"),
        "contiene": bloque_rl("contiene"),
        "calibrado": bloque_rl("calibrado"),
        "evitar_youth": g(agente_rapido, "youth", "evitar", "media"),
        "evitar_seniors": g(agente_rapido, "seniors", "evitar", "media"),
        "n_semillas": g(rl, "metricas", "qlearning_recompensa_final_original", "n"),
    }

    return {
        # `cargar` ya exige modo full en todas las fuentes; el arnés pide esta clave en el resumen.
        "modo": "full",
        "generado": dt.datetime.now().isoformat(timespec="seconds"),
        "fuente": "reports/metrics/*.json (modo full)",
        "datos": datos,
        "sentimiento": sentimiento,
        "recomendador": recomendador,
        "abandono": abandono,
        "refuerzo": refuerzo,
    }


# --------------------------------------------------------------------------------------------
# Gráficos para negocio
# --------------------------------------------------------------------------------------------
def fig_sentimiento(s: dict) -> str:
    modelos = sorted(s["modelos"], key=lambda m: m["accuracy"])
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    y = range(len(modelos))
    colores = [NILO if m["final"] else (GRIS if "Azar" in m["nombre"] else "#C9D0D8") for m in modelos]
    vals = [100 * m["accuracy"] for m in modelos]
    errs = [
        [100 * (m["accuracy"] - m["ic"][0]) for m in modelos],
        [100 * (m["ic"][1] - m["accuracy"]) for m in modelos],
    ]
    ax.barh(y, vals, color=colores, height=0.62, xerr=errs, ecolor=GRIS_OSCURO, capsize=2.5,
            error_kw={"lw": 0.8})
    for i, (v, m) in enumerate(zip(vals, modelos)):
        ax.text(v + 1.6, i, f"{fmt(v)} %", va="center", fontsize=10,
                color=NILO if m["final"] else GRIS_OSCURO, fontweight="bold" if m["final"] else None)
    ax.set_yticks(list(y), [m["nombre"] for m in modelos])
    for lbl, m in zip(ax.get_yticklabels(), modelos):
        if m["final"]:
            lbl.set_color(NILO)
            lbl.set_fontweight("bold")
    ax.set_xlim(0, 108)
    ax.set_xlabel("Reseñas clasificadas correctamente (%)")
    estilo_ejes(ax)
    ax.grid(axis="x", color="#E6E9EE", lw=0.6)
    ax.set_axisbelow(True)
    return guardar(fig, "r1_sentimiento")


def fig_recomendador(r: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.45), sharey=True)
    titulos = {
        "real": "Datos reales de Nile Prime",
        "sembrada": "Prueba con gustos simulados",
    }
    for ax, clave in zip(axes, ("real", "sembrada")):
        nombres = list(r[clave].keys())
        vals = [100 * (r[clave][n] or 0) for n in nombres]
        colores = [GRIS, ARENA, NILO]
        ax.bar(range(len(nombres)), vals, color=colores, width=0.62)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.12, fmt(v), ha="center", fontsize=10, color=TINTA,
                    fontweight="bold" if i == 2 else None)
        ax.set_xticks(range(len(nombres)), [n.replace(" (nuestro)", "\n(nuestro)") for n in nombres])
        ax.set_title(titulos[clave], fontsize=11, color=TINTA, loc="left", pad=8)
        estilo_ejes(ax)
    axes[0].set_ylabel("Aciertos por cada 100 usuarios")
    axes[0].set_ylim(0, max(6.0, 1.25 * max(100 * v for v in r["sembrada"].values() if v)))
    fig.tight_layout(w_pad=2.5)
    return guardar(fig, "r2_recomendador")


def fig_deciles(a: dict) -> str | None:
    if not a["deciles"]:
        return None
    ks = sorted(a["deciles"])
    vals = [100 * a["deciles"][k] for k in ks]
    fig, ax = plt.subplots(figsize=(7.2, 1.95))
    colores = [ALERTA if k == 10 else "#C9D0D8" for k in ks]
    ax.bar(ks, vals, color=colores, width=0.7)
    media = 100 * a["decil_resto_media"]
    ax.axhline(media, color=GRIS_OSCURO, lw=0.9, ls="--")
    ax.text(0.55, media + 0.35, f"media del resto: {fmt(media)} %", fontsize=9, color=GRIS_OSCURO)
    ax.text(10, vals[-1] + 0.35, f"{fmt(vals[-1])} %", ha="center", fontsize=10, color=ALERTA,
            fontweight="bold")
    ax.set_xticks(ks, [f"{k}" for k in ks])
    ax.set_xlabel("Grupos de visualizaciones ordenados por riesgo según el modelo (1 = menor, 10 = mayor)")
    ax.set_ylabel("Abandono real (%)")
    ax.set_ylim(0, max(vals) * 1.25)
    estilo_ejes(ax)
    return guardar(fig, "r3_deciles")


def fig_segmentos(a: dict) -> str | None:
    segs = a.get("segmentos")
    if not segs:
        return None
    segs = sorted(segs, key=lambda s: s["tasa"])
    fig, ax = plt.subplots(figsize=(7.2, 0.25 * len(segs) + 0.45))
    vals = [100 * s["tasa"] for s in segs]
    colores = [ALERTA if s.get("riesgo") else (GRIS_OSCURO if s.get("referencia") else "#C9D0D8")
               for s in segs]
    errs = [[100 * (s["tasa"] - s["ic95"][0]) if s.get("ic95") else 0 for s in segs],
            [100 * (s["ic95"][1] - s["tasa"]) if s.get("ic95") else 0 for s in segs]]
    ax.barh(range(len(segs)), vals, color=colores, height=0.68, xerr=errs, ecolor=GRIS_OSCURO,
            capsize=2.5, error_kw={"lw": 0.8})
    for i, (v, sg) in enumerate(zip(vals, segs)):
        x = 100 * sg["ic95"][1] if sg.get("ic95") else v
        ax.text(x + 0.6, i, f"{fmt(v)} %", va="center", fontsize=9.5, color=TINTA,
                fontweight="bold" if sg.get("riesgo") else None)
    ax.set_yticks(range(len(segs)), [s["etiqueta"] for s in segs])
    ax.set_xlabel("Abandono (%)")
    ax.set_xlim(0, max(vals) * 1.3)
    estilo_ejes(ax)
    return guardar(fig, "r3_segmentos")


def fig_refuerzo(rf: dict) -> str:
    o = rf["original"]
    etiquetas = ["Recomendador\nestático (híbrido)", "Recomendar\nal azar", "Agente que\naprende (RL)"]
    vals = [1000 * (1 - o["hibrido"]), 1000 * (1 - o["azar"]), 1000 * (1 - o["rl"])]
    fig, ax = plt.subplots(figsize=(7.2, 2.25))
    ax.bar(range(3), vals, color=[ARENA, GRIS, NILO], width=0.58)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, fmt(v, 1 if v >= 1 else 2), ha="center", fontsize=11, color=TINTA,
                fontweight="bold" if i == 2 else None)
    ax.set_xticks(range(3), etiquetas)
    ax.set_ylabel("Abandonos por cada\n1.000 recomendaciones")
    ax.set_ylim(0, max(vals) * 1.25)
    estilo_ejes(ax)
    return guardar(fig, "r4_refuerzo")


# --------------------------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------------------------
def buscar_navegador() -> str | None:
    candidatos = [os.environ.get("NP_CHROME")]
    for nombre in ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser"):
        candidatos.append(shutil.which(nombre))
    pf = [os.environ.get(v) for v in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
    for base in filter(None, pf):
        candidatos += [
            str(Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
            str(Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe"),
        ]
    candidatos += [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ]
    pw = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", Path.home() / ".cache" / "ms-playwright"))
    candidatos += [str(p) for p in sorted(pw.glob("chromium-*/chrome-linux/chrome"))]
    candidatos += [str(p) for p in sorted(pw.glob("chromium-*/chrome-win/chrome.exe"))]
    return next((c for c in candidatos if c and Path(c).exists()), None)


def imprimir_pdf(html: Path, pdf: Path) -> None:
    navegador = buscar_navegador()
    if navegador:
        cmd = [
            navegador, "--headless=new", "--disable-gpu", "--no-sandbox",
            "--no-pdf-header-footer", "--allow-file-access-from-files",
            f"--print-to-pdf={pdf}", html.as_uri(),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
        if pdf.exists() and pdf.stat().st_size > 0:
            return
    try:  # Alternativa: Playwright (pip install playwright && playwright install chromium)
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            b = p.chromium.launch()
            pagina = b.new_page()
            pagina.goto(html.as_uri(), wait_until="networkidle")
            pagina.pdf(path=str(pdf), prefer_css_page_size=True, print_background=True)
            b.close()
        return
    except ImportError as exc:
        raise RuntimeError(
            "No se encontró Edge/Chrome/Chromium ni Playwright. Instala uno de ellos o define "
            "NP_CHROME con la ruta del navegador."
        ) from exc


# --------------------------------------------------------------------------------------------
def main() -> None:
    resumen = consolidar()
    with open(METRICAS / "resumen.json", "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, ensure_ascii=False, indent=2)

    figuras = {
        "sentimiento": fig_sentimiento(resumen["sentimiento"]),
        "recomendador": fig_recomendador(resumen["recomendador"]),
        "segmentos": fig_segmentos(resumen["abandono"]),
        "refuerzo": fig_refuerzo(resumen["refuerzo"]),
    }

    entorno = Environment(loader=FileSystemLoader(AQUI), undefined=StrictUndefined, autoescape=True)
    entorno.filters.update({"pct": pct, "fmt": fmt})
    html = entorno.get_template("plantilla.html.j2").render(
        r=resumen, fig=figuras, fecha="11 de octubre de 2026"
    )
    HTML_SALIDA.write_text(html, encoding="utf-8")
    print(f"HTML: {HTML_SALIDA.relative_to(RAIZ)}")

    if os.environ.get("NP_SOLO_HTML") == "1":
        return
    imprimir_pdf(HTML_SALIDA, PDF_SALIDA)
    print(f"PDF:  {PDF_SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    sys.exit(main())
