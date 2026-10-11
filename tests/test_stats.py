"""Tests de src/stats.py."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src import stats


def test_wilson_contiene_proporcion():
    lo, hi = stats.wilson_ci(10, 100)
    assert lo < 0.10 < hi and 0 <= lo and hi <= 1
    lo0, hi0 = stats.wilson_ci(0, 50)
    assert lo0 == 0 or lo0 < 1e-12 and hi0 > 0


def test_chi2_independencia_y_dependencia():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": rng.integers(0, 3, 5000), "y": rng.random(5000) < 0.1})
    assert stats.chi2_cramer(df, "g", "y")["cramer_v"] < 0.05
    df["y"] = df["g"] == 0
    assert stats.chi2_cramer(df, "g", "y")["cramer_v"] > 0.9


def test_rate_table():
    df = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 10, "y": [True] * 5 + [False] * 15})
    t = stats.rate_table(df, "g", "y")
    assert t.loc["a", "tasa"] == 0.5 and t.loc["b", "tasa"] == 0


def test_chi2_contra_valor_conocido():
    # Tabla fija 2x2: [[30, 70], [50, 50]]; chi2 a mano = 8.333..., V = sqrt(chi2/n)
    df = pd.DataFrame({
        "g": ["a"] * 100 + ["b"] * 100,
        "y": [True] * 30 + [False] * 70 + [True] * 50 + [False] * 50,
    })
    r = stats.chi2_cramer(df, "g", "y")
    esperado = chi2_contingency(np.array([[30, 70], [50, 50]]), correction=False)
    assert abs(r["chi2"] - esperado[0]) < 1e-9 and abs(r["p"] - esperado[1]) < 1e-12
    assert abs(r["chi2"] - 8.3333333) < 1e-6  # valor calculado a mano
    assert r["gl"] == 1 and r["n"] == 200
    assert abs(r["cramer_v"] - np.sqrt(8.3333333 / 200)) < 1e-6


def test_cramer_corregida_bajo_h0_y_bajo_dependencia():
    rng = np.random.default_rng(1)
    # Tabla dispersa 3 x 999 con ~34 obs. por celda, independiente: V bruta ~0,1, corregida ~0
    n = 100_000
    df = pd.DataFrame({"a": rng.integers(0, 3, n), "t": rng.integers(0, 999, n)})
    r = stats.chi2_cramer(df, "a", "t")
    esperada = stats.cramer_v_esperada_h0(r["gl"], r["n"], 2)
    assert 0.08 < r["cramer_v"] < 0.12 and abs(r["cramer_v"] - esperada) < 0.01
    vc = stats.cramer_v_corregida(r["chi2"], r["n"], 3, 999)
    assert vc < 0.03
    # Asociación perfecta: la corrección apenas la altera
    df["t"] = df["a"]
    r2 = stats.chi2_cramer(df, "a", "t")
    assert stats.cramer_v_corregida(r2["chi2"], r2["n"], 3, 3) > 0.99


def test_cramer_corregida_valor_a_mano():
    # chi2=8.3333, n=200, 2x2: phi2c = 8.3333/200 - 1/199 = 0.036641; r_c=c_c=2-1/199
    v = stats.cramer_v_corregida(8.3333333, 200, 2, 2)
    esperado = np.sqrt((8.3333333 / 200 - 1 / 199) / (1 - 1 / 199))
    assert abs(v - esperado) < 1e-9


def test_pares_por_usuario():
    ia, ib = stats.pares_por_usuario(np.array([1, 2, 1, 1, 2, 3]))
    pares = {tuple(sorted(p)) for p in zip(ia.tolist(), ib.tolist(), strict=True)}
    assert pares == {(0, 2), (0, 3), (2, 3), (1, 4)}


def test_coherencia_permutacion_detecta_estructura_y_azar():
    rng = np.random.default_rng(0)
    n_u, m, n_t = 300, 8, 40
    sim = (np.arange(n_t)[:, None] // 10) == (np.arange(n_t)[None, :] // 10)  # 4 "géneros"
    usuarios = np.repeat(np.arange(n_u), m)
    # Azar: títulos independientes del usuario
    # (títulos distintos dentro de cada usuario, como en los datos tras quitar duplicados)
    aleatorio = np.concatenate([rng.choice(n_t, m, replace=False) for _ in range(n_u)])
    r0 = stats.coherencia_permutacion(aleatorio, usuarios, sim, n_perm=100, seed=3)
    assert r0["p_dos_colas"] > 0.05 and abs(r0["observado"] - r0["nulo_media"]) < 0.01
    assert r0["n_pares"] == n_u * m * (m - 1) // 2 and len(r0["nulo"]) == 100
    # Estructura: cada usuario elige dentro de un solo bloque
    bloque = rng.integers(0, 4, n_u)
    con = np.concatenate([b * 10 + rng.choice(10, m, replace=False) for b in bloque])
    r1 = stats.coherencia_permutacion(con, usuarios, sim, n_perm=100, seed=3)
    assert r1["observado"] > 0.99 and r1["p_mayor"] < 0.05
    # Reproducible
    r2 = stats.coherencia_permutacion(con, usuarios, sim, n_perm=100, seed=3)
    assert np.allclose(r1["nulo"], r2["nulo"])
