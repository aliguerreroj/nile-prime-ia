"""Generadores de datos sintéticos con señal sembrada (validación de métodos).

Los datos reales de Nile Prime no muestran estructura de preferencias (ver notebook 00,
sección 9). Para demostrar que un método SÍ recupera señal cuando existe, este módulo fabrica
datos donde la señal es conocida y su intensidad es configurable.

Contenido actual (reto 2):
    ``generar_interacciones_sembradas``: interacciones usuario-título con preferencia por
    género de intensidad ``p_pref``.

Contenido actual (reto 3):
    ``generar_abandono_sembrado``: sustituye la etiqueta de abandono de la tabla real de
    features por una generada con probabilidad conocida (efecto de la duración y de
    combinaciones edad x género), con intensidad configurable y semilla fija.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
COLUMNAS = ["User_ID", "Series_Title", "Watched_to_End", "Age_Group"]


def _mascaras_por_genero(catalogo: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Devuelve (lista de géneros, matriz booleana ítems × géneros)."""
    listas = catalogo["Genre"].fillna("").str.split(", ")
    generos = sorted({g for gs in listas for g in gs if g})
    pos = {g: j for j, g in enumerate(generos)}
    m = np.zeros((len(catalogo), len(generos)), dtype=bool)
    for i, gs in enumerate(listas):
        for g in gs:
            if g:
                m[i, pos[g]] = True
    return generos, m


def generar_interacciones_sembradas(
    catalogo: pd.DataFrame,
    referencia: pd.DataFrame,
    p_pref: float,
    n_generos_pref: int = 2,
    seed: int = SEED,
) -> pd.DataFrame:
    """Genera interacciones usuario-título con preferencia por género de intensidad configurable.

    Modelo generativo (por usuario ``u`` con ``n_u`` interacciones, igual que en ``referencia``):

    1. ``u`` tiene ``n_generos_pref`` géneros preferidos, elegidos al azar y de forma uniforme
       entre los géneros del catálogo.
    2. El número de interacciones "preferidas" es ``Binomial(n_u, p_pref)`` (acotado por el
       tamaño del conjunto preferido): cada una es un título distinto de uniforme entre los
       que tienen al menos un género preferido.
    3. El resto son títulos distintos elegidos de forma uniforme entre los demás del catálogo.
    4. ``Watched_to_End`` ~ Bernoulli(tasa de la referencia), independiente de la preferencia;
       ``Age_Group`` se copia de la referencia (no hay señal por edad en este generador).

    Con ``p_pref = 0`` las elecciones son uniformes: no hay estructura (réplica del azar de los
    datos reales). Con ``p_pref`` alto, el histórico de cada usuario concentra sus géneros
    preferidos. La señal es por género, justo lo que codifica el modelo de contenido (caso
    favorable al método, no un escenario neutral).

    Args:
        catalogo: metadatos con ``Series_Title`` (únicos) y ``Genre`` (separador ", ").
        referencia: interacciones reales con ``User_ID``, ``Series_Title``, ``Watched_to_End`` y
            ``Age_Group``; fijan el nº de usuarios, el nº de interacciones por usuario, el
            grupo de edad y la tasa de visionado completo.
        p_pref: probabilidad (0-1) de que cada interacción provenga de los géneros preferidos.
        n_generos_pref: número de géneros preferidos por usuario.
        seed: semilla del generador (misma semilla → mismos datos).

    Returns:
        DataFrame con las columnas ``User_ID``, ``Series_Title``, ``Watched_to_End``,
        ``Age_Group``. ``df.attrs["generos_preferidos"]`` guarda ``{User_ID: [géneros]}``.
    """
    if not 0.0 <= p_pref <= 1.0:
        raise ValueError("p_pref debe estar en [0, 1]")
    rng = np.random.default_rng(seed)
    titulos = catalogo["Series_Title"].to_numpy()
    n_items = len(titulos)
    generos, mat = _mascaras_por_genero(catalogo)
    if not 1 <= n_generos_pref <= len(generos):
        raise ValueError("n_generos_pref fuera de rango")

    por_usuario = referencia.groupby("User_ID", sort=True).agg(
        n=("Series_Title", "size"), edad=("Age_Group", "first")
    )
    p_completa = float(referencia["Watched_to_End"].mean())

    ids, items, edades, prefs = [], [], [], {}
    filas_u = zip(por_usuario.index, por_usuario["n"], por_usuario["edad"], strict=True)
    for uid, n_u, edad in filas_u:
        n_u = min(int(n_u), n_items)
        gsel = rng.choice(len(generos), size=n_generos_pref, replace=False)
        prefs[uid] = [generos[j] for j in sorted(gsel)]
        pref_idx = np.flatnonzero(mat[:, gsel].any(axis=1))
        n_pref = min(int(rng.binomial(n_u, p_pref)), len(pref_idx))
        elegidos = rng.choice(pref_idx, size=n_pref, replace=False)
        libres = np.ones(n_items, dtype=bool)
        libres[elegidos] = False
        resto = rng.choice(np.flatnonzero(libres), size=n_u - n_pref, replace=False)
        sel = np.concatenate([elegidos, resto])
        items.append(sel)
        ids.append(np.full(n_u, uid))
        edades.append(np.full(n_u, edad, dtype=object))

    items = np.concatenate(items)
    df = pd.DataFrame(
        {
            "User_ID": np.concatenate(ids),
            "Series_Title": titulos[items],
            "Watched_to_End": rng.random(len(items)) < p_completa,
            "Age_Group": np.concatenate(edades),
        }
    )[COLUMNAS]
    df.attrs["generos_preferidos"] = prefs
    return df


# ---------------------------------------------------------------------------------------
# Reto 3: abandono con señal sembrada
# ---------------------------------------------------------------------------------------

#: Combinaciones edad x género sembradas por defecto: (grupo de edad, género, signo del efecto).
PARES_SEMBRADOS = (("youth", "Action", 1.0), ("seniors", "Comedy", -1.0))


def generar_abandono_sembrado(
    tabla: pd.DataFrame,
    intensidad: float = 1.0,
    prevalencia: float = 0.10,
    pares: tuple = PARES_SEMBRADOS,
    efecto_runtime: float = 0.8,
    efecto_par: float = 0.9,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera una etiqueta de abandono con probabilidad conocida sobre la tabla real de features.

    Sustituye la etiqueta real por una sintética con el modelo generativo:

        logit(p_i) = b0 + intensidad * ( efecto_runtime * z_i
                                         + efecto_par * sum_k signo_k * marca_ik )
        y_i ~ Bernoulli(p_i)

    donde ``marca_ik`` = 1[edad_i = e_k y género_k en el título i], ``z_i`` es la duración
    (``Runtime``) estandarizada, ``(e_k, género_k, signo_k)`` son los
    ``pares`` edad x género sembrados y ``b0`` se calibra por bisección para que la prevalencia
    media sea ``prevalencia``. Con ``intensidad = 0`` todas las probabilidades valen
    ``prevalencia`` (sin señal: debe reproducir el AUC ~ 0,5 de los datos reales). Es un caso
    favorable al método: la señal usa variables que el modelo ya recibe.

    Args:
        tabla: salida de ``features.construir_tabla`` (necesita ``Runtime``, ``Age_Group`` y las
            columnas ``g_<género>``).
        intensidad: multiplicador de los efectos (0 = sin señal).
        prevalencia: tasa media de abandono objetivo (0-1).
        pares: combinaciones edad x género con su signo.
        efecto_runtime: efecto en logit por desviación típica de la duración (a intensidad 1).
        efecto_par: efecto en logit de cada combinación (a intensidad 1).
        seed: semilla (misma semilla → mismas etiquetas).

    Returns:
        ``(y, p_verdadera)``: etiquetas 0/1 (``int8``) y probabilidad verdadera por fila.
    """
    if not 0.0 < prevalencia < 1.0:
        raise ValueError("prevalencia debe estar en (0, 1)")
    if intensidad < 0:
        raise ValueError("intensidad debe ser >= 0")
    rng = np.random.default_rng(seed)
    runtime = tabla["Runtime"].to_numpy(dtype=float)
    z = (runtime - runtime.mean()) / runtime.std()
    lineal = efecto_runtime * z
    for edad, genero, signo in pares:
        marca = (tabla["Age_Group"].to_numpy() == edad) & (tabla[f"g_{genero}"].to_numpy() == 1)
        lineal = lineal + efecto_par * signo * marca
    lineal = intensidad * lineal

    lo, hi = -20.0, 20.0
    for _ in range(80):  # bisección: la prevalencia media crece con b0
        b0 = (lo + hi) / 2
        media = (1.0 / (1.0 + np.exp(-(b0 + lineal)))).mean()
        lo, hi = (b0, hi) if media < prevalencia else (lo, b0)
    p = 1.0 / (1.0 + np.exp(-(b0 + lineal)))
    y = (rng.random(len(p)) < p).astype(np.int8)
    return y, p
