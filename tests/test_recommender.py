"""Tests del recomendador híbrido: métricas, partición sin fuga, API y arranque en frío."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import recommender as rec
from src import synthetic as syn
from tests.conftest import catalogo_mini, referencia_mini


@pytest.fixture(scope="module")
def cat():
    return catalogo_mini()


@pytest.fixture(scope="module")
def sim(cat):
    return rec.similitud_contenido(cat)


@pytest.fixture(scope="module")
def datos_aleatorios(cat):
    return syn.generar_interacciones_sembradas(cat, referencia_mini(250, 12), 0.0, seed=5)


@pytest.fixture(scope="module")
def datos_senal(cat):
    return syn.generar_interacciones_sembradas(cat, referencia_mini(600, 12), 0.9, seed=5)


# -- métricas contra casos calculados a mano ------------------------------------------------
def test_metricas_ranking_a_mano():
    # 2 usuarios, 6 ítems, k=3. U0: relevantes {1,4}, top3 = [4, 0, 1].
    # U1: relevante {5}, top3 = [0, 1, 2].
    topk = np.array([[4, 0, 1], [0, 1, 2]])
    rel = np.zeros((2, 6), bool)
    rel[0, [1, 4]] = True
    rel[1, 5] = True
    m = rec.metricas_ranking(topk, rel, 3)
    dcg0 = 1 / np.log2(2) + 1 / np.log2(4)
    idcg0 = 1 / np.log2(2) + 1 / np.log2(3)
    np.testing.assert_allclose(m["precision"], [2 / 3, 0])
    np.testing.assert_allclose(m["recall"], [1.0, 0])
    np.testing.assert_allclose(m["hit"], [1, 0])
    np.testing.assert_allclose(m["ndcg"], [dcg0 / idcg0, 0])


def test_metricas_un_relevante_recall_es_hit():
    topk = np.array([[3, 2, 1, 0]])
    rel = np.zeros((1, 5), bool)
    rel[0, 1] = True
    m = rec.metricas_ranking(topk, rel, 4)
    assert m["recall"][0] == m["hit"][0] == 1.0
    assert m["precision"][0] == pytest.approx(0.25)
    assert m["ndcg"][0] == pytest.approx(1 / np.log2(4))  # rango 3


def test_rangos_coinciden_con_metricas_y_con_rango_objetivo():
    rng = np.random.default_rng(0)
    sc = rng.random((50, 30))
    vistos = rng.random((50, 30)) < 0.2
    obj = np.array([np.flatnonzero(~vistos[i])[0] for i in range(50)])
    jit = rec.jitter_para(sc.shape)
    r = rec.evaluar_scores(sc, vistos, obj, 5, jit)
    r2 = rec.rango_objetivo(sc, vistos, obj, jit)
    np.testing.assert_array_equal(r["rango_objetivo"], r2)
    m2 = rec.metricas_desde_rango(r2, 5)
    for k_ in ("hit", "ndcg", "precision", "recall"):
        np.testing.assert_allclose(r["metricas"][k_], m2[k_])


def test_azar_teorico_y_cobertura_diversidad():
    assert rec.azar_teorico(np.array([100, 100]), 10)["hit"] == pytest.approx(0.1)
    assert rec.cobertura_catalogo(np.array([[0, 1], [1, 2]]), 10) == pytest.approx(0.3)
    sim = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]])
    d = rec.diversidad_intralista(np.array([[0, 1], [0, 2]]), sim)
    np.testing.assert_allclose(d, [0.0, 1.0])


def test_bootstrap_ic_contiene_media():
    x = np.random.default_rng(1).random(500)
    r = rec.bootstrap_ic({"x": x}, n_boot=200)["x"]
    assert r["ic95"][0] <= r["valor"] <= r["ic95"][1]


# -- partición sin fuga ----------------------------------------------------------------------
def test_particion_sin_fuga(datos_aleatorios):
    p = rec.particionar_loo(datos_aleatorios, seed=1)
    rec.comprobar_sin_fuga(p)
    assert len(p["val"]) == len(p["test"]) == len(p["elegibles"])
    assert len(p["train"]) == len(datos_aleatorios) - 2 * len(p["elegibles"])
    assert p["val"]["Watched_to_End"].all() and p["test"]["Watched_to_End"].all()
    # un retenido en train haría fallar la comprobación
    malo = dict(p, train=pd.concat([p["train"], p["test"].head(1)]))
    with pytest.raises(AssertionError):
        rec.comprobar_sin_fuga(malo)


# -- API del recomendador -------------------------------------------------------------------
@pytest.fixture(scope="module")
def modelo(cat, sim, datos_aleatorios):
    return rec.HybridRecommender(alpha=0.5).fit(cat, datos_aleatorios, sim_contenido=sim)


def test_recommend_excluye_vistos_y_devuelve_k(modelo, datos_aleatorios):
    uid = datos_aleatorios["User_ID"].iloc[0]
    vistos = set(datos_aleatorios.loc[datos_aleatorios["User_ID"] == uid, "Series_Title"])
    r = modelo.recommend(uid, k=10)
    assert len(r) == 10 and r["Series_Title"].is_unique
    assert not (set(r["Series_Title"]) & vistos)
    assert set(r["Series_Title"]) <= set(modelo.titulos_)
    assert list(r["score"]) == sorted(r["score"], reverse=True)


def test_arranque_en_frio_usuario(modelo, cat, datos_aleatorios):
    r = modelo.recommend("usuario_nuevo", k=5, grupo_edad="seniors")
    esperado = modelo.pop_edad_[modelo.grupos_.index("seniors")]
    assert len(r) == 5
    assert esperado[r["movie_id"].to_numpy()].min() >= np.sort(esperado)[-8]
    g = modelo.recommend("usuario_nuevo", k=5)  # sin grupo → global
    assert len(g) == 5
    h = modelo.recommend_historial([], k=5, grupo_edad="youth")
    assert len(h) == 5


def test_arranque_en_frio_item_sin_interacciones(cat, sim, datos_aleatorios):
    quitar = cat["Series_Title"].iloc[0]
    df = datos_aleatorios[datos_aleatorios["Series_Title"] != quitar]
    hib = rec.HybridRecommender(alpha=0.5).fit(cat, df, sim_contenido=sim)
    cf = rec.CollaborativeRecommender().fit(cat, df, sim_contenido=sim)
    assert hib.item_frio_[0] and hib.item_frio_.sum() == 1
    # historial: el vecino de contenido más cercano del ítem frío
    vecino = cat["Series_Title"].iloc[int(np.argmax(sim[0]))]
    r_h = hib.recommend_historial([vecino], k=10)
    r_cf = cf.recommend_historial([vecino], k=len(cat) - 1)
    assert quitar in set(r_h["Series_Title"])          # el híbrido lo puede recomendar
    assert r_cf["Series_Title"].iloc[-1] == quitar     # el colaborativo lo deja al final


def test_alpha_0_y_1_reproducen_contenido_y_colaborativo(cat, sim, datos_aleatorios):
    m = rec.HybridRecommender(alpha=0.3).fit(cat, datos_aleatorios, sim_contenido=sim)
    assert not m.item_frio_.any()
    uids = m.usuarios_[:40]
    fr = m._filas(uids)
    vistos = m.vistos_[fr].toarray() > 0
    for a, S in ((0.0, m.sim_contenido_), (1.0, m.sim_cf_)):
        sc, _ = m.puntuar_lote(uids, alpha=a)
        crudo = np.asarray(m.pos_[fr] @ S)
        for i in range(len(uids)):
            c = ~vistos[i]
            z = (crudo[i, c] - crudo[i, c].mean()) / crudo[i, c].std()
            np.testing.assert_allclose(sc[i, c], z, atol=1e-9)
    assert rec.ContentRecommender().alpha == 0.0
    assert rec.CollaborativeRecommender().alpha == 1.0


def test_contenido_vecinos_mismo_genero(cat, sim):
    v = rec.vecinos_contenido(sim, cat, "T0", n=3)
    assert len(v) == 3 and v["similitud"].is_monotonic_decreasing
    assert "T0" not in set(v["Series_Title"])


# -- el híbrido recupera señal sembrada y no la inventa -------------------------------------
def test_hibrido_supera_a_popularidad_con_senal(cat, sim, datos_senal):
    r = rec.ejecutar_pipeline(cat, datos_senal, k=10, n_boot=200, sim_contenido=sim)
    ic = r["ic"]
    assert ic["Híbrido"]["hit"]["valor"] > 1.3 * ic["Popularidad global"]["hit"]["valor"]
    assert r["pareados"]["Popularidad global"]["hit"]["a_mejor"]
    assert r["pareados"]["Aleatorio"]["ndcg"]["a_mejor"]


def test_sin_senal_no_hay_mejora(cat, sim, datos_aleatorios):
    r = rec.ejecutar_pipeline(cat, datos_aleatorios, k=10, n_boot=200, sim_contenido=sim)
    assert not r["pareados"]["Aleatorio"]["hit"]["a_mejor"]


def test_prueba_permutacion():
    n, ni = 300, 40
    rng = np.random.default_rng(0)
    # ranking perfecto: el retenido de cada usuario es su ítem de rango 1 (señal total)
    objetivo = rng.integers(0, ni, n)
    perfecto = np.tile(np.arange(2, ni + 2, dtype=np.int16), (n, 1))
    perfecto[np.arange(n), objetivo] = 1
    r = rec.prueba_permutacion(perfecto, objetivo, k=5, n_perm=100)
    assert r["hit_obs"] == 1.0 and r["p_hit"] == pytest.approx(1 / 101)
    # ranking aleatorio: sin señal → p no pequeño en la mayoría de semillas
    rangos = np.argsort(np.argsort(rng.random((n, ni)), axis=1), axis=1).astype(np.int16) + 1
    r0 = rec.prueba_permutacion(rangos, objetivo, k=5, n_perm=200)
    assert r0["p_hit"] > 0.01 and abs(r0["hit_nulo_media"] - 5 / ni) < 0.03


# -- colisión de títulos (popularidad) y agregación entre semillas ---------------------------
def test_pesos_por_colision_detecta_titulos_repetidos(cat):
    crudo = pd.concat([cat, cat.iloc[[3]]], ignore_index=True)  # T3 aparece dos veces
    w = rec.pesos_por_colision(cat, crudo)
    assert w[3] == 0.5 and np.delete(w, 3).tolist() == [1.0] * (len(cat) - 1)


def test_pesos_popularidad_reparte_solo_la_popularidad(cat, sim):
    inter = referencia_mini(120, 8, seed=1)
    w = np.ones(len(cat))
    w[5] = 0.5
    base = rec.PopularityRecommender().fit(cat, inter, sim_contenido=sim)
    ajust = rec.PopularityRecommender(pesos_popularidad=w).fit(cat, inter, sim_contenido=sim)
    assert ajust.pop_global_[5] == pytest.approx(base.pop_global_[5] / 2)
    assert np.allclose(np.delete(ajust.pop_global_, 5), np.delete(base.pop_global_, 5))
    assert ajust.pop_edad_[:, 5].sum() == pytest.approx(base.pop_edad_[:, 5].sum() / 2)
    # la matriz de interacciones (CF) no cambia
    assert (ajust.pos_ != base.pos_).nnz == 0
    with pytest.raises(ValueError):
        rec.PopularityRecommender(pesos_popularidad=np.ones(3)).fit(cat, inter)


def test_colision_ya_no_encabeza_arranque_en_frio(cat, sim):
    inter = referencia_mini(150, 8, seed=2)
    extra = pd.DataFrame({"User_ID": range(1000, 1060), "Series_Title": "T7",
                          "Watched_to_End": True, "Age_Group": "adults"})
    inter = pd.concat([inter, extra], ignore_index=True)  # T7 inflado artificialmente
    w = np.ones(len(cat))
    w[7] = 0.25
    sin = rec.HybridRecommender().fit(cat, inter, sim_contenido=sim)
    con = rec.HybridRecommender(pesos_popularidad=w).fit(cat, inter, sim_contenido=sim)
    assert sin.recommend("nuevo", k=1)["Series_Title"].iloc[0] == "T7"
    assert con.recommend("nuevo", k=1)["Series_Title"].iloc[0] != "T7"


def test_resumen_semillas():
    r = rec.resumen_semillas([1.0, 2.0, 3.0])
    assert r["valor"] == 2.0 and r["sd"] == pytest.approx(1.0) and r["n_semillas"] == 3
    assert r["ic95"][0] < 2.0 < r["ic95"][1]
    assert r["ic95"][1] - 2.0 == pytest.approx(4.302653 / np.sqrt(3), rel=1e-4)
    uno = rec.resumen_semillas([0.5])
    assert uno["sd"] == 0.0 and uno["ic95"] == [0.5, 0.5]


def test_agregar_pipelines_y_curva(cat, sim):
    res = []
    for s in (1, 2):
        d = syn.generar_interacciones_sembradas(cat, referencia_mini(250, 12), 0.9, seed=s)
        res.append(rec.ejecutar_pipeline(cat, d, k=10, n_boot=50, seed=s, sim_contenido=sim,
                                         rejilla=[0.0, 0.5, 1.0]))
    ag = rec.agregar_pipelines(res)
    assert ag["ic"]["Híbrido"]["hit"]["n_semillas"] == 2
    assert set(ag["ic"]["Híbrido"]["hit"]) >= {"valor", "ic95", "sd"}
    p = ag["pareados"]["Popularidad global"]["hit"]
    assert {"diferencia", "ic95", "p_mcnemar", "p_semillas", "p_max"} <= set(p)
    assert len(p["p_semillas"]) == 2
    cur = rec.curva_validacion_alpha(res)
    assert cur["cualquiera"]["alpha"] == [0.0, 0.5, 1.0]
    assert len(cur["cualquiera"]["ndcg_media"]) == 3
    assert len(cur["completa"]["p_vs_0_5_por_semilla"]) == 2
