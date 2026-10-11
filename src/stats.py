"""Utilidades estadísticas del EDA: IC de Wilson, chi-cuadrado y V de Cramér."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def wilson_ci(exitos: np.ndarray | float, n: np.ndarray | float, z: float = 1.96):
    """Intervalo de Wilson para una proporción (95 % por defecto).

    Args:
        exitos: número de éxitos (escalar o array).
        n: tamaño de muestra (escalar o array).

    Returns:
        Tupla ``(inferior, superior)`` con la misma forma que la entrada.
    """
    exitos = np.asarray(exitos, dtype=float)
    n = np.asarray(n, dtype=float)
    p = exitos / n
    denom = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / denom
    margen = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centro - margen, centro + margen


def rate_table(df: pd.DataFrame, grupo: str, evento: str) -> pd.DataFrame:
    """Tabla de ``n``, eventos, tasa e IC de Wilson 95 % por categoría de ``grupo``.

    ``evento`` debe ser una columna booleana (``True`` = evento, p. ej. abandono).
    """
    t = df.groupby(grupo)[evento].agg(n="size", eventos="sum").astype({"eventos": int})
    t["tasa"] = t["eventos"] / t["n"]
    t["ic_inf"], t["ic_sup"] = wilson_ci(t["eventos"], t["n"])
    return t


def chi2_cramer(df: pd.DataFrame, a: str, b: str) -> dict:
    """Chi-cuadrado de independencia entre dos columnas categóricas y V de Cramér.

    Returns:
        dict con ``chi2``, ``p``, ``gl``, ``n`` y ``cramer_v`` (0 = sin asociación).
    """
    tabla = pd.crosstab(df[a], df[b])
    chi2, p, gl, _ = chi2_contingency(tabla, correction=False)
    n = tabla.to_numpy().sum()
    k = min(tabla.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0
    return {"chi2": float(chi2), "p": float(p), "gl": int(gl), "n": int(n), "cramer_v": v}


def cramer_v_corregida(chi2: float, n: int, filas: int, columnas: int) -> float:
    """V de Cramér con corrección de sesgo de Bergsma (2013).

    En tablas muy dispersas (pocas observaciones por celda) el chi-cuadrado bajo H0 vale en
    promedio ``gl`` y la V bruta queda sesgada al alza. Bergsma corrige phi² y las
    dimensiones de la tabla::

        phi2c = max(0, chi2/n - (r-1)(c-1)/(n-1))
        r_c = r - (r-1)²/(n-1);  c_c = c - (c-1)²/(n-1)
        V_c = sqrt(phi2c / min(r_c-1, c_c-1))

    Args:
        chi2: estadístico chi-cuadrado (sin corrección de continuidad).
        n: total de observaciones de la tabla.
        filas, columnas: dimensiones de la tabla de contingencia.

    Returns:
        V de Cramér corregida en [0, 1]; ≈ 0 si la asociación es la esperada por azar.
    """
    phi2 = max(0.0, chi2 / n - (filas - 1) * (columnas - 1) / (n - 1))
    r_c = filas - (filas - 1) ** 2 / (n - 1)
    c_c = columnas - (columnas - 1) ** 2 / (n - 1)
    denom = min(r_c - 1, c_c - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def cramer_v_esperada_h0(gl: int, n: int, k: int) -> float:
    """V de Cramér esperada bajo independencia: ``sqrt(gl / (n·k))`` con ``k = min(r,c) - 1``.

    Bajo H0, E[chi²] ≈ gl; sirve de referencia para juzgar si una V bruta indica efecto.
    """
    return float(np.sqrt(gl / (n * k)))


def pares_por_usuario(usuarios: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Posiciones de todos los pares (i < j) de interacciones del mismo usuario (vectorizado).

    Args:
        usuarios: array de ids de usuario, uno por interacción.

    Returns:
        ``(ia, ib)``: índices (sobre el array original) de cada par. Cada par aparece una vez.
    """
    usuarios = np.asarray(usuarios)
    orden = np.argsort(usuarios, kind="stable")
    u = usuarios[orden]
    _, cuentas = np.unique(u, return_counts=True)
    max_d = int(cuentas.max()) - 1 if len(cuentas) else 0
    ia, ib = [], []
    for d in range(1, max_d + 1):
        mismo = u[:-d] == u[d:]
        idx = np.flatnonzero(mismo)
        ia.append(orden[idx])
        ib.append(orden[idx + d])
    if not ia:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    return np.concatenate(ia), np.concatenate(ib)


def coherencia_permutacion(
    titulos: np.ndarray,
    usuarios: np.ndarray,
    similitud: np.ndarray,
    n_perm: int = 200,
    seed: int = 42,
) -> dict:
    """Coherencia intra-usuario frente a una prueba de permutación.

    Estadístico: proporción de pares de títulos del mismo usuario con ``similitud[i, j]``
    verdadera (p. ej. comparten género o director). Baseline nulo: se baraja la columna de
    títulos entre todas las interacciones (conserva el nº de interacciones por usuario y la
    popularidad global), se recalcula el estadístico y se repite ``n_perm`` veces. Los pares
    barajados con el mismo título dos veces se excluyen del estadístico barajado.

    Args:
        titulos: código entero de título (0..T-1) por interacción.
        usuarios: id de usuario por interacción.
        similitud: matriz booleana T×T (simétrica) de similitud entre títulos.
        n_perm: número de réplicas de permutación.
        seed: semilla del generador.

    Returns:
        dict con ``observado``, ``nulo_media``, ``nulo_ic95`` (percentiles 2,5-97,5),
        ``nulo_sd``, ``p_mayor`` (cola superior, con +1), ``p_dos_colas``, ``n_pares``,
        ``n_perm`` y ``nulo`` (array de réplicas).
    """
    titulos = np.asarray(titulos)
    ia, ib = pares_por_usuario(usuarios)
    obs = float(similitud[titulos[ia], titulos[ib]].mean())
    rng = np.random.default_rng(seed)
    nulo = np.empty(n_perm)
    for r in range(n_perm):
        perm = rng.permutation(titulos)
        ta, tb = perm[ia], perm[ib]
        ok = ta != tb
        nulo[r] = similitud[ta[ok], tb[ok]].mean()
    p_mayor = (1 + int((nulo >= obs).sum())) / (n_perm + 1)
    p_menor = (1 + int((nulo <= obs).sum())) / (n_perm + 1)
    return {
        "observado": obs,
        "nulo_media": float(nulo.mean()),
        "nulo_sd": float(nulo.std(ddof=1)) if n_perm > 1 else 0.0,
        "nulo_ic95": [float(np.percentile(nulo, 2.5)), float(np.percentile(nulo, 97.5))],
        "p_mayor": float(p_mayor),
        "p_dos_colas": float(min(1.0, 2 * min(p_mayor, p_menor))),
        "n_pares": int(len(ia)),
        "n_perm": int(n_perm),
        "nulo": nulo,
    }
