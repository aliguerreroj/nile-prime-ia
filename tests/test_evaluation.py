"""Tests de src/evaluation.py (sin modelos ni descargas)."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src import evaluation as ev


def _datos(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    pred = np.where(rng.random(n) < 0.85, y, 1 - y)
    return y, pred


def test_metricas_coinciden_con_sklearn():
    y, p = _datos()
    m = ev.metricas_binarias(y, p)
    assert np.isclose(m["accuracy"], accuracy_score(y, p))
    assert np.isclose(m["f1"], f1_score(y, p))


def test_bootstrap_ci_contiene_valor_y_es_reproducible():
    y, p = _datos()
    a = ev.bootstrap_ci(y, p, n_boot=500, seed=1)
    b = ev.bootstrap_ci(y, p, n_boot=500, seed=1)
    assert a == b
    for m in ("accuracy", "f1"):
        lo, hi = a[m]["ic95"]
        assert lo <= a[m]["valor"] <= hi
        assert hi - lo < 0.15


def test_bootstrap_pareado_modelos_iguales_da_cero():
    y, p = _datos()
    r = ev.bootstrap_pareado(y, p, p, n_boot=200)
    assert r["accuracy"]["diferencia"] == 0 and r["accuracy"]["ic95"] == [0.0, 0.0]


def test_bootstrap_pareado_detecta_diferencia_clara():
    y, p = _datos(2000)
    malo = 1 - y  # siempre falla
    r = ev.bootstrap_pareado(y, p, malo, n_boot=200)
    assert r["accuracy"]["ic95"][0] > 0


def test_mcnemar_exacto():
    y = np.array([1] * 20 + [0] * 20)
    a = y.copy()
    b = y.copy()
    b[:10] = 1 - b[:10]  # B falla en 10 donde A acierta
    r = ev.mcnemar_exacto(y, a, b)
    assert (r["a_acierta_b_falla"], r["a_falla_b_acierta"]) == (10, 0)
    assert r["p_valor"] < 0.01
    assert ev.mcnemar_exacto(y, a, a)["p_valor"] == 1.0


def test_umbral_por_costo_se_desplaza_con_costos_asimetricos():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 5000)
    proba = np.clip(0.5 + (y - 0.5) * 0.5 + rng.normal(0, 0.2, 5000), 0, 1)
    sim = ev.umbral_por_costo(y, proba, 1, 1)
    # Un FN diez veces más caro empuja el umbral hacia abajo (más positivos).
    fn_caro = ev.umbral_por_costo(y, proba, 1, 10)
    assert fn_caro["umbral"] < sim["umbral"]
    assert sim["costo"] <= sim["costo_en_0_5"]


def test_bootstrap_media():
    v = np.random.default_rng(0).random(500)
    r = ev.bootstrap_media(v, n_boot=300)
    assert r["ic95"][0] <= r["valor"] <= r["ic95"][1]
    assert np.isclose(r["valor"], v.mean())


# ---------------------------------------------------------------------------------------
# Métricas con puntuaciones, bootstrap por grupos, calibración y permutación (reto 3)
# ---------------------------------------------------------------------------------------


def _datos_puntuacion(n=3000, seed=0, empates=False):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    s = rng.random(n) * 0.6 + 0.3 * y
    if empates:
        s = np.round(s, 1)
    return y, s


@pytest.mark.parametrize("empates", [False, True])
def test_metricas_puntuacion_igual_que_sklearn(empates):
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        f1_score,
        recall_score,
        roc_auc_score,
    )

    y, s = _datos_puntuacion(empates=empates)
    m = ev.metricas_puntuacion(y, s, umbral=0.5)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, s))
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, s))
    assert m["brier"] == pytest.approx(brier_score_loss(y, s))
    assert m["f1"] == pytest.approx(f1_score(y, s >= 0.5))
    assert m["recall"] == pytest.approx(recall_score(y, s >= 0.5))


def test_metricas_puntuacion_constante_es_azar():
    y, _ = _datos_puntuacion()
    m = ev.metricas_puntuacion(y, np.full(len(y), 0.15), umbral=0.3)
    assert m["roc_auc"] == pytest.approx(0.5)
    assert m["pr_auc"] == pytest.approx(y.mean())  # referencia de azar = prevalencia
    assert m["f1"] == 0.0 and m["recall"] == 0.0


def test_ponderado_equivale_a_repetir_filas():
    from sklearn.metrics import average_precision_score, roc_auc_score

    y, s = _datos_puntuacion(n=400)
    w = np.random.default_rng(1).integers(0, 4, size=len(y))
    auc, ap = ev._auc_ap_ponderados(y.astype(float), w.astype(float), ev._Orden(s))
    yr, sr = np.repeat(y, w), np.repeat(s, w)
    assert auc == pytest.approx(roc_auc_score(yr, sr))
    assert ap == pytest.approx(average_precision_score(yr, sr))


def test_bootstrap_por_grupos_ic_y_pares():
    y, s = _datos_puntuacion(n=4000)
    g = np.repeat(np.arange(1000), 4)
    azar = np.random.default_rng(5).random(len(y))
    r = ev.bootstrap_por_grupos(y, {"modelo": s, "azar": azar}, g, umbrales={"modelo": 0.5},
                                referencia="azar", n_boot=300, seed=1)
    m = r["modelos"]["modelo"]
    assert m["roc_auc"]["ic95"][0] <= m["roc_auc"]["valor"] <= m["roc_auc"]["ic95"][1]
    assert {"f1", "recall"} <= set(m)
    assert "f1" not in r["modelos"]["azar"]
    assert r["diferencias"]["modelo"]["roc_auc"]["ic95"][0] > 0  # supera claramente al azar
    az = r["modelos"]["azar"]["roc_auc"]["ic95"]
    assert az[0] < 0.5 < az[1]
    r2 = ev.bootstrap_por_grupos(y, {"modelo": s}, g, n_boot=300, seed=1)
    assert r2["modelos"]["modelo"]["roc_auc"] == r["modelos"]["modelo"]["roc_auc"]


def test_bootstrap_por_grupos_mas_ancho_con_correlacion_intragrupo():
    # Etiqueta constante dentro de cada grupo: el IC por grupos debe ser más ancho que por fila.
    rng = np.random.default_rng(0)
    n_g, k = 300, 10
    g = np.repeat(np.arange(n_g), k)
    y_g = (rng.random(n_g) < 0.3).astype(int)
    y = np.repeat(y_g, k)
    s = np.repeat(rng.random(n_g) * 0.5 + 0.2 * y_g, k) + rng.random(n_g * k) * 0.01
    ic_g = ev.bootstrap_por_grupos(y, {"m": s}, g, n_boot=300)["modelos"]["m"]["roc_auc"]["ic95"]
    ic_f = ev.bootstrap_por_grupos(y, {"m": s}, np.arange(len(y)), n_boot=300)["modelos"]["m"][
        "roc_auc"
    ]["ic95"]
    assert (ic_g[1] - ic_g[0]) >= (ic_f[1] - ic_f[0])


def test_p_valor_permutacion():
    assert ev.p_valor_permutacion(0.9, np.full(99, 0.5)) == pytest.approx(1 / 100)
    assert ev.p_valor_permutacion(0.4, np.full(99, 0.5)) == pytest.approx(1.0)
    assert 0 < ev.p_valor_permutacion(0.5, np.array([0.5, 0.4])) <= 1


def test_calibradores_corrigen_probabilidades_infladas():
    rng = np.random.default_rng(0)
    n = 6000
    y = (rng.random(n) < 0.1).astype(int)
    p_inflada = np.clip(0.5 + 0.1 * (rng.random(n) - 0.5) + 0.05 * y, 0, 1)  # media ~0.5
    g = np.arange(n) // 3
    b0 = ((p_inflada - y) ** 2).mean()
    for met in ("sigmoide", "isotonica"):
        pc = ev.calibrar_oof_por_grupos(met, p_inflada, y, g)
        assert (pc >= 0).all() and (pc <= 1).all()
        assert ((pc - y) ** 2).mean() < b0
        assert pc.mean() == pytest.approx(y.mean(), abs=0.02)
    with pytest.raises(ValueError):
        ev.ajustar_calibrador("otro", p_inflada, y)


def test_umbral_por_costo_desempate_con_referencia():
    y = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    p = np.full(len(y), 0.1)  # sin señal: marcar a nadie es lo óptimo con 3:1
    r = ev.umbral_por_costo(y, p, 1.0, 3.0, np.linspace(0.01, 0.99, 99), referencia=0.25)
    assert r["umbral"] == pytest.approx(0.25)
    assert r["costo"] == pytest.approx(3 * 0.2)


def test_calibrador_sigmoide_no_invierte_el_ranking():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.1).astype(int)
    ruido = 0.05 * rng.standard_normal(4000)
    p = np.clip(0.5 - 0.1 * y + ruido, 0.01, 0.99)  # puntuación invertida
    f = ev.ajustar_calibrador("sigmoide", p, y)
    q = f(np.array([0.2, 0.5, 0.8]))
    assert np.allclose(q, y.mean())  # constante: el ranking no se invierte
    iso = ev.ajustar_calibrador("isotonica", p, y)(np.array([0.2, 0.5, 0.8]))
    assert (np.diff(iso) >= 0).all()
