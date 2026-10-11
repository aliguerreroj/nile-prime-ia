"""Estilo y guardado de figuras del proyecto Nile Prime.

Define la paleta común, un estilo de matplotlib consistente y ``save_fig``, que guarda en
``reports/figures/`` a 150 dpi sin rutas absolutas. Usa un backend no interactivo si no hay
pantalla disponible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from src.data import project_root

# Sin pantalla matplotlib cae solo a un backend no interactivo (Agg); las figuras se guardan
# siempre con save_fig, no se dependen de plt.show().

DPI = 150

# Paleta única del proyecto: ninguna figura debe usar colores literales.
PRINCIPAL = "#1f4e79"  # azul oscuro: color principal (barras, histogramas)
REFERENCIA = "#c0392b"  # rojo: línea de referencia / global / valor observado
ENFASIS = "#e07b39"  # naranja: elemento a destacar
NEUTRO = "#8a8f98"  # gris: contexto, distribuciones nulas o secundarias
GRUPOS_EDAD = {  # colores categóricos fijos por grupo de edad
    "youth": "#2a9d8f",
    "adults": "#1f4e79",
    "seniors": "#e07b39",
}
# Un color por recomendador del reto 2 (seis series distinguibles en una misma figura).
RECOMENDADORES = {
    "Aleatorio": "#8a8f98",
    "Popularidad global": "#6a4c93",
    "Popularidad por edad": "#2a9d8f",
    "Contenido": "#1f4e79",
    "Colaborativo": "#b8962e",
    "Híbrido": "#e07b39",
}
PALETA = {
    "principal": PRINCIPAL,
    "referencia": REFERENCIA,
    "enfasis": ENFASIS,
    "neutro": NEUTRO,
    **{f"edad_{k}": v for k, v in GRUPOS_EDAD.items()},
    **{f"rec_{k}": v for k, v in RECOMENDADORES.items()},
}
COLORES = [PRINCIPAL, ENFASIS, GRUPOS_EDAD["youth"], REFERENCIA, NEUTRO]


def set_style() -> None:
    """Aplica el estilo de matplotlib común a todas las figuras del proyecto."""
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "axes.prop_cycle": matplotlib.cycler(color=COLORES),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
        }
    )


def figures_dir() -> Path:
    """Devuelve ``reports/figures`` bajo la raíz del proyecto, creándola si falta."""
    carpeta = project_root() / "reports" / "figures"
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def save_fig(fig: matplotlib.figure.Figure, nombre: str) -> Path:
    """Guarda ``fig`` como ``reports/figures/<nombre>.png`` a 150 dpi y devuelve la ruta.

    Args:
        fig: figura de matplotlib.
        nombre: nombre del archivo, con o sin la extensión ``.png`` (p. ej. ``00_nulos``).
    """
    if not nombre.endswith(".png"):
        nombre += ".png"
    ruta = figures_dir() / nombre
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    return ruta
