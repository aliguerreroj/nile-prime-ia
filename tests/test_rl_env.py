"""Tests de src/rl_env.py: importación de agent.py, entorno, P, política óptima y Q-learning."""

import sys

import numpy as np
import pandas as pd
import pytest

from src import data
from src import rl_env as R


@pytest.fixture(scope="module")
def catalogo():
    return data.load_metadata()


@pytest.fixture(scope="module")
def interacciones():
    return data.load_interactions(sin_duplicados=True, rapido=True)


@pytest.fixture(scope="module")
def agent():
    return R.cargar_entorno_original()


# ---------------------------------------------------------------- importación de agent.py
def test_agent_no_modificado():
    assert R.sha256_agent() == R.SHA256_AGENT


def test_importa_agent_desde_otro_directorio(tmp_path, monkeypatch):
    previo = sys.modules.pop("agent")
    lector = pd.read_csv
    try:
        monkeypatch.chdir(tmp_path)  # sin CSV en el directorio actual
        modulo = R.cargar_entorno_original()
        assert hasattr(modulo, "SimulationEnvironment")
        assert len(modulo.movies_df) == 1000
        assert pd.read_csv is lector  # restaurado, sin efectos secundarios
    finally:
        sys.modules["agent"] = previo


# ---------------------------------------------------------------- bugs reales del original
def test_entorno_original_tiene_los_dos_bugs(agent, catalogo, interacciones):
    env = agent.SimulationEnvironment(catalogo, interacciones)
    assert env.reset() is None
    with pytest.raises(AttributeError, match="_get_state"):
        env.step(0)


def test_nileprime_reset_y_step(catalogo, interacciones):
    env = R.NilePrimeEnv(catalogo, interacciones, seed=0)
    estado = env.reset()
    assert estado in (0, 1, 2) and estado == R.GRUPOS.index(env.current_user["Age_Group"])
    s, r, done = env.step(0)
    assert s == estado and r in (0, 1) and done is True
    assert env.reset() in (0, 1, 2) and env.done is False


def test_reproducible_con_semilla(catalogo, interacciones):
    def traza(seed):
        env = R.NilePrimeEnv(catalogo, interacciones, modo="contiene", seed=seed)
        return [(env.reset(), env.step(a)[1]) for a in range(60)]

    assert traza(3) == traza(3)
    assert traza(3) != traza(4)


# ---------------------------------------------------------------- P y regla del profesor
def _p_profesor(agent, env_prof, genero, grupo, monkeypatch):
    """Recupera por bisección la probabilidad de ver completa del código del profesor."""
    env_prof.current_user = {"Age_Group": grupo}
    peli = {"Genre": genero}
    lo, hi = 0.0, 1.0
    for _ in range(30):
        u = (lo + hi) / 2
        monkeypatch.setattr(np.random, "rand", lambda u=u: u)
        if agent.SimulationEnvironment._simulate_user_watch(env_prof, peli):
            lo = u  # vio con u < p  →  p > u
        else:
            hi = u
    return (lo + hi) / 2


def test_matriz_original_coincide_con_el_profesor(agent, catalogo, interacciones, monkeypatch):
    env_prof = agent.SimulationEnvironment(catalogo, interacciones)
    P = R.matriz_probabilidades("original", catalogo["Genre"])
    unicos = catalogo.drop_duplicates("Genre").index
    for j in unicos:
        for i, g in enumerate(R.GRUPOS):
            p = _p_profesor(agent, env_prof, catalogo.loc[j, "Genre"], g, monkeypatch)
            assert abs(p - P[i, j]) < 1e-6


def test_mismas_decisiones_que_el_profesor_con_la_misma_semilla(agent, catalogo, interacciones):
    env_prof = agent.SimulationEnvironment(catalogo, interacciones)
    env = R.NilePrimeEnv(catalogo, interacciones, seed=11)
    np.random.seed(11)
    for k in range(300):
        grupo = R.GRUPOS[k % 3]
        peli = catalogo.iloc[k]
        env_prof.current_user = env.current_user = {"Age_Group": grupo}
        assert bool(env_prof._simulate_user_watch(peli)) == bool(env._simulate_user_watch(peli))


def test_conteo_de_peliculas_penalizadas(catalogo):
    por_modo = {}
    for modo in R.MODOS:
        P = R.matriz_probabilidades(modo, catalogo["Genre"])
        por_modo[modo] = [int(x) for x in (P < P.max(axis=1, keepdims=True) - 1e-12).sum(axis=1)]
    assert por_modo["original"] == [87, 0, 13]
    assert por_modo["calibrado"] == [87, 0, 13]
    assert por_modo["contiene"] == [741, 227, 310]


def test_calibrado_usa_las_tasas_de_los_datos(catalogo):
    P = R.matriz_probabilidades("calibrado", catalogo["Genre"])
    assert np.isclose(P[1].min(), 1 - R.CALIBRADO_DEFECTO["base"])
    assert np.isclose(P[0].min(), 1 - R.CALIBRADO_DEFECTO["segmento"]["youth"])


def test_valores_aleatorio_y_optimo_originales(catalogo):
    po = R.politica_optima(R.matriz_probabilidades("original", catalogo["Genre"]))
    assert np.allclose(po["valor_optimo"], 1.0)
    assert np.allclose(po["valor_azar"], [1 - 0.3 * 87 / 1000, 1.0, 1 - 0.25 * 13 / 1000])
    assert [len(x) for x in po["evitar"]] == [87, 0, 13]


def test_politica_optima_pequena():
    P = np.array([[0.7, 1.0, 1.0], [0.5, 0.9, 0.9]])
    po = R.politica_optima(P, np.array([0.25, 0.75]))
    assert list(po["evitar"][0]) == [0] and list(po["optimas"][1]) == [1, 2]
    assert np.isclose(po["valor_optimo_medio"], 0.25 * 1.0 + 0.75 * 0.9)


def test_monte_carlo_entorno_vs_matriz(catalogo, interacciones):
    """El simulador vectorizado (basado en P) es equivalente al entorno real."""
    gen = np.random.default_rng(0)
    for modo in R.MODOS:
        env = R.NilePrimeEnv(catalogo, interacciones, modo=modo, seed=1)
        P = R.matriz_probabilidades(modo, catalogo["Genre"])
        n = 4000
        acc, esperado = [], []
        for a in gen.integers(0, 1000, n):
            s = env.reset()
            acc.append(env.step(int(a))[1])
            esperado.append(P[s, a])
        esperado = np.array(esperado)
        z = (np.mean(acc) - esperado.mean()) / np.sqrt((esperado * (1 - esperado)).sum()) * n
        assert abs(z) < 4


# ---------------------------------------------------------------- agentes
def _problema_pequeno():
    P = np.array([[0.2, 0.9, 0.5, 0.4], [0.8, 0.3, 0.1, 0.1]])
    grupo = np.array([0, 1, 1, 0, 0, 1])
    return P, grupo


def test_q_learning_converge_en_problema_pequeno():
    P, grupo = _problema_pequeno()
    T = 6000
    res = R.entrenar_bandit(P, grupo, [0, 1, 2], T, R.programa_epsilon("exponencial", T),
                            alpha=None, checkpoints=R.pasos_checkpoint(T, 20))
    voraz = res["Q"].argmax(axis=2)
    assert (voraz[:, 0] == 1).all() and (voraz[:, 1] == 0).all()
    assert res["valor_voraz"][:, -1].min() > P.max(axis=1).mean() - 0.05
    reg = R.regret_por_paso(res["esperada"], res["estados"], P)
    assert reg[:, -1000:].mean() < reg[:, :1000].mean()


def test_ucb_y_reranking_funcionan():
    P, grupo = _problema_pequeno()
    T = 3000
    eps = R.programa_epsilon("exponencial", T)
    u = R.entrenar_bandit(P, grupo, [0, 1], T, eps, alpha=None, metodo="ucb")
    topk = np.array([[0, 2], [1, 3], [2, 3], [0, 1], [3, 2], [1, 0]])
    k = R.entrenar_bandit(P, grupo, [0, 1], T, eps, alpha=0.1, topk_fila=topk)
    assert u["esperada"].shape == k["esperada"].shape == (2, T)
    assert k["esperada"][:, -500:].mean() > k["esperada"][:, :500].mean() - 0.02


def test_entrenamiento_reproducible_e_independiente_del_lote():
    P, grupo = _problema_pequeno()
    eps = R.programa_epsilon("inverso", 500)
    a = R.entrenar_bandit(P, grupo, [5, 6], 500, eps)
    b = R.entrenar_bandit(P, grupo, [5, 6], 500, eps)
    c = R.entrenar_bandit(P, grupo, [6], 500, eps)
    assert np.array_equal(a["Q"], b["Q"])
    assert np.array_equal(a["estados"][1], c["estados"][0])


def test_programas_de_epsilon():
    for tipo in ("constante", "exponencial", "inverso"):
        e = R.programa_epsilon(tipo, 1000, 0.3, 0.01)
        assert len(e) == 1000 and e.min() >= 0.01 - 1e-12 and e[0] <= 0.3 + 1e-12
    assert np.isclose(R.programa_epsilon("exponencial", 1000)[-1], 0.01)


def test_tiempos_y_ic():
    r = np.zeros((2, 5000))
    r[:, :1000] = 1.0
    t = R.tiempo_convergencia(r, regret_azar=1.0, fraccion=0.1, ventana=100)
    assert (t > 1000).all() and (t < 1200).all()
    ic = R.ic_t([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ic["valor"] == 3.0 and ic["ic95"][0] < 3.0 < ic["ic95"][1]
    assert R.ic_t([2.0, 2.0])["ic95"] == [2.0, 2.0]


def test_agrupar_por_genero_y_reduccion(catalogo):
    ids, unicos = R.agrupar_por_genero(catalogo["Genre"])
    assert len(unicos) == catalogo["Genre"].nunique() == ids.max() + 1
    assert (np.asarray(unicos)[ids] == catalogo["Genre"].to_numpy()).all()
    for modo in R.MODOS:
        P = R.matriz_probabilidades(modo, catalogo["Genre"])
        assert R.matriz_por_genero(P, ids).shape == (3, len(unicos))


def test_auc_separacion():
    Q = np.array([[[0.2, 0.3, 0.9, 0.8]]])
    N = np.ones_like(Q, dtype=int)
    riesgo = np.array([[True, True, False, False]])
    assert R.auc_separacion(Q, N, riesgo)[0, 0] == 1.0
    assert R.auc_separacion(Q[:, :, ::-1], N, riesgo)[0, 0] == 0.0
    N0 = N.copy()
    N0[0, 0, 2:] = 0  # sin películas seguras probadas
    assert np.isnan(R.auc_separacion(Q, N0, riesgo)[0, 0])
