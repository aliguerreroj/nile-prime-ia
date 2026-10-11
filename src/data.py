"""Carga y limpieza de los datos de Nile Prime (reseñas, metadatos e interacciones).

Todas las rutas se resuelven desde la raíz del proyecto (sin rutas absolutas) y los CSV se
localizan con ``glob``, porque sus nombres llevan tildes en Unicode NFD.

Modo rápido: si la variable de entorno ``NP_FAST=1`` está activa, las cargas devuelven una
submuestra reproducible (``SEED = 42``) para iterar rápido; sin ella, todos los datos.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

# Tamaños del modo rápido (los metadatos siempre se cargan completos).
FAST_N_RESENAS = 5_000  # reseñas, muestra estratificada por sentimiento
FAST_FRAC_USUARIOS = 0.10  # fracción de usuarios completos en interacciones

# Filas esperadas de cada CSV original.
N_RESENAS = 50_000
N_METADATOS = 1_000
N_INTERACCIONES = 100_624
N_RESENAS_UNICAS = 49_578  # tras limpiar HTML y quitar duplicados (418 exactos en bruto + 4)

# Año correcto de Apollo 13 (en el CSV viene "PG" por un error de origen).
APOLLO13_ANIO = 1995


def project_root() -> Path:
    """Devuelve la raíz del proyecto (la carpeta que contiene ``data/raw``).

    Sube desde el directorio de trabajo y desde la ubicación de este archivo hasta encontrar
    ``data/raw``; funciona en local, en notebooks (cwd = ``notebooks``) y en Colab.
    """
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        for carpeta in (base, *base.parents):
            if (carpeta / "data" / "raw").is_dir():
                return carpeta
    raise FileNotFoundError("No se encontró la carpeta data/raw desde el directorio actual.")


def find_csv(patron: str) -> Path:
    """Localiza un CSV de ``data/raw`` con un patrón glob (p. ej. ``'Rese*.csv'``).

    Evita comparar nombres con tildes: los archivos originales están en Unicode NFD.
    """
    coincidencias = sorted(glob.glob(str(project_root() / "data" / "raw" / patron)))
    if len(coincidencias) != 1:
        raise FileNotFoundError(f"Se esperaba 1 archivo para '{patron}', hay {len(coincidencias)}.")
    return Path(coincidencias[0])


def is_fast() -> bool:
    """Indica si el modo rápido está activo (``NP_FAST=1``)."""
    return os.environ.get("NP_FAST", "0") == "1"


def _resolver_rapido(rapido: bool | None) -> bool:
    return is_fast() if rapido is None else rapido


def clean_html(texto: pd.Series) -> pd.Series:
    """Elimina las etiquetas HTML (``<br />``, etc.) de una serie de textos.

    Cada etiqueta se sustituye por un espacio y se colapsan los espacios repetidos, para no
    pegar palabras contiguas.
    """
    return (
        texto.str.replace(r"<[^>]+>", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def load_reviews(sin_duplicados: bool = False, rapido: bool | None = None) -> pd.DataFrame:
    """Carga las reseñas (columnas ``review`` y ``sentiment``) con el HTML eliminado.

    La etiqueta ``sentiment`` no se modifica. Hay 50.000 filas (25.000 por clase).

    Args:
        sin_duplicados: si es ``True`` elimina las reseñas duplicadas comparando SOLO el
            texto (no la etiqueta) y conserva la primera aparición. Así ningún texto puede
            quedar a la vez en train y en test, ni siquiera si un mismo texto tuviera etiquetas
            distintas (hoy no ocurre: ninguna reseña tiene etiquetas contradictorias). La
            deduplicación se hace DESPUÉS de limpiar el HTML, porque textos que solo difieren
            en las etiquetas ``<br />`` son la misma reseña. Número de filas resultante: ver
            ``N_RESENAS_UNICAS``.
        rapido: fuerza o desactiva el modo rápido; por defecto lo decide ``NP_FAST``. En modo
            rápido devuelve ``FAST_N_RESENAS`` filas con muestra estratificada por sentimiento
            (semilla ``SEED``).
    """
    df = pd.read_csv(find_csv("Rese*.csv"), encoding="utf-8")
    df["review"] = clean_html(df["review"])
    if sin_duplicados:
        df = df.drop_duplicates(subset=["review"], keep="first")
    if _resolver_rapido(rapido):
        n_clase = FAST_N_RESENAS // df["sentiment"].nunique()
        df = df.groupby("sentiment", group_keys=False).sample(n=n_clase, random_state=SEED)
        df = df.sort_index()
    return df.reset_index(drop=True)


def load_metadata(titulos_unicos: bool = False) -> pd.DataFrame:
    """Carga y limpia los metadatos de películas (siempre completos, también en modo rápido).

    Limpieza:
        - ``movie_id``: identificador entero estable (posición en el CSV original, 0-999).
        - ``Runtime``: de texto ("142 min") a entero en minutos.
        - ``Gross``: de texto con comas ("28,341,469") a float.
        - ``Released_Year``: a entero. Apollo 13 trae "PG" por un error de origen y se
          corrige a 1995.
        - ``Meta_score`` y ``Gross`` nulos se dejan como NaN (float), SIN imputar: la
          imputación debe ajustarse dentro del pipeline con datos de entrenamiento.
        - ``Certificate`` nulo pasa a la categoría explícita ``"Desconocido"``.

    Duplicados: "Drishyam" aparece dos veces porque son dos películas distintas (2013 y
    2015) con el mismo título; por eso existe ``movie_id``.

    Args:
        titulos_unicos: si es ``True`` conserva, por ``Series_Title``, la fila con más
            ``No_of_Votes`` (999 filas); es lo que se usa al unir con interacciones, que solo
            traen el título. Con ``False`` quedan las 1.000 filas.
    """
    df = pd.read_csv(find_csv("Metadatos*.csv"), encoding="utf-8")
    df.insert(0, "movie_id", np.arange(len(df), dtype="int64"))

    df.loc[df["Series_Title"] == "Apollo 13", "Released_Year"] = str(APOLLO13_ANIO)
    df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors="raise").astype("int64")
    df["Runtime"] = df["Runtime"].str.extract(r"(\d+)", expand=False).astype("int64")
    df["Gross"] = pd.to_numeric(df["Gross"].str.replace(",", "", regex=False), errors="raise")
    df["Gross"] = df["Gross"].astype("float64")
    df["Meta_score"] = df["Meta_score"].astype("float64")
    df["Certificate"] = df["Certificate"].fillna("Desconocido")

    if titulos_unicos:
        df = (
            df.sort_values("No_of_Votes", ascending=False, kind="stable")
            .drop_duplicates(subset="Series_Title", keep="first")
            .sort_values("movie_id")
        )
    return df.reset_index(drop=True)


def load_interactions(sin_duplicados: bool = False, rapido: bool | None = None) -> pd.DataFrame:
    """Carga las interacciones usuario-película (100.624 filas, 10.000 usuarios).

    Columnas: ``User_ID``, ``Series_Title``, ``Watched_to_End``, ``Age_Group``.

    Args:
        sin_duplicados: si es ``True`` elimina filas duplicadas exactas (queda 100.623). La
            única es el usuario 6537 viendo las dos "Drishyam" (2013 y 2015): como la
            interacción solo trae el título, al trabajar con títulos únicos es un duplicado
            real.
        rapido: fuerza o desactiva el modo rápido; por defecto lo decide ``NP_FAST``. En modo
            rápido se eligen ``FAST_FRAC_USUARIOS`` de los usuarios (semilla ``SEED``) y se
            devuelven TODAS sus filas.
    """
    df = pd.read_csv(find_csv("interacciones*.csv"), encoding="utf-8")
    if _resolver_rapido(rapido):
        usuarios = np.sort(df["User_ID"].unique())
        rng = np.random.default_rng(SEED)
        n = int(round(len(usuarios) * FAST_FRAC_USUARIOS))
        elegidos = rng.choice(usuarios, size=n, replace=False)
        df = df[df["User_ID"].isin(elegidos)]
    if sin_duplicados:
        df = df.drop_duplicates()
    return df.reset_index(drop=True)
