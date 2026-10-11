"""Utilidades de evaluación reutilizables: IC bootstrap, comparación pareada y umbral por costo.

Todas las funciones trabajan sobre predicciones ya calculadas (arrays de numpy), de modo que
no dependen de ningún modelo concreto. Convención: clasificación binaria con clase positiva = 1.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

METRICAS = ("accuracy", "f1")


def _contadores(y: np.ndarray, pred: np.ndarray, idx: np.ndarray | None = None):
    """Devuelve (tp, fp, fn, tn) sobre ``y``/``pred`` (o sus remuestreos ``idx`` por filas)."""
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    if idx is not None:
        y, pred = y[idx], pred[idx]
    eje = -1
    tp = (y & pred).sum(axis=eje)
    fp = (~y & pred).sum(axis=eje)
    fn = (y & ~pred).sum(axis=eje)
    tn = (~y & ~pred).sum(axis=eje)
    return tp, fp, fn, tn


def metricas_binarias(y: np.ndarray, pred: np.ndarray, idx: np.ndarray | None = None) -> dict:
    """Accuracy y F1 de la clase positiva; vectorizado si ``idx`` es una matriz de remuestreos.

    Si no hay predicciones ni positivos reales (F1 indefinido), F1 = 0.
    """
    tp, fp, fn, tn = _contadores(y, pred, idx)
    n = tp + fp + fn + tn
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros(np.shape(den), dtype=float), where=den > 0)
    return {"accuracy": (tp + tn) / n, "f1": f1}


def _indices_bootstrap(n: int, n_boot: int, seed: int, bloque: int = 250):
    """Genera matrices de índices de remuestreo por bloques para acotar la memoria."""
    rng = np.random.default_rng(seed)
    hechos = 0
    while hechos < n_boot:
        k = min(bloque, n_boot - hechos)
        yield rng.integers(0, n, size=(k, n))
        hechos += k


def bootstrap_ci(
    y: np.ndarray, pred: np.ndarray, n_boot: int = 1000, seed: int = 42, nivel: float = 0.95
) -> dict:
    """IC bootstrap percentil de accuracy y F1 (clase positiva) a partir de predicciones.

    Remuestrea las UNIDADES de evaluación con reemplazo ``n_boot`` veces (semilla fija) y
    devuelve para cada métrica ``{"valor": estimación puntual, "ic95": [inf, sup]}``.
    """
    y = np.asarray(y)
    pred = np.asarray(pred)
    muestras = {m: [] for m in METRICAS}
    for idx in _indices_bootstrap(len(y), n_boot, seed):
        r = metricas_binarias(y, pred, idx)
        for m in METRICAS:
            muestras[m].append(r[m])
    puntual = metricas_binarias(y, pred)
    a = (1 - nivel) / 2
    salida = {}
    for m in METRICAS:
        v = np.concatenate(muestras[m])
        lo, hi = np.quantile(v, [a, 1 - a])
        salida[m] = {"valor": float(puntual[m]), "ic95": [float(lo), float(hi)]}
    return salida


def bootstrap_pareado(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    nivel: float = 0.95,
) -> dict:
    """Bootstrap pareado de la diferencia (A - B) de accuracy y F1 sobre las MISMAS unidades.

    Devuelve por métrica ``{"diferencia", "ic95"}``. Con IC que contiene 0 no hay diferencia
    significativa al nivel indicado.
    """
    y = np.asarray(y)
    difs = {m: [] for m in METRICAS}
    for idx in _indices_bootstrap(len(y), n_boot, seed):
        ra = metricas_binarias(y, pred_a, idx)
        rb = metricas_binarias(y, pred_b, idx)
        for m in METRICAS:
            difs[m].append(ra[m] - rb[m])
    pa, pb = metricas_binarias(y, pred_a), metricas_binarias(y, pred_b)
    a = (1 - nivel) / 2
    salida = {}
    for m in METRICAS:
        lo, hi = np.quantile(np.concatenate(difs[m]), [a, 1 - a])
        salida[m] = {"diferencia": float(pa[m] - pb[m]), "ic95": [float(lo), float(hi)]}
    return salida


def mcnemar_exacto(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Prueba de McNemar exacta (binomial bilateral) sobre los desacuerdos de A y B.

    ``b`` = A acierta y B falla; ``c`` = A falla y B acierta. Bajo H0, b ~ Bin(b + c, 0,5).
    """
    y = np.asarray(y).astype(bool)
    ok_a = np.asarray(pred_a).astype(bool) == y
    ok_b = np.asarray(pred_b).astype(bool) == y
    b = int((ok_a & ~ok_b).sum())
    c = int((~ok_a & ok_b).sum())
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"a_acierta_b_falla": b, "a_falla_b_acierta": c, "p_valor": p}


def costo_esperado(
    y: np.ndarray, proba: np.ndarray, umbral: float, c_fp: float = 1.0, c_fn: float = 1.0
) -> float:
    """Costo medio por unidad al decidir ``positivo`` si ``proba >= umbral``."""
    y = np.asarray(y).astype(bool)
    pred = np.asarray(proba) >= umbral
    fp = (~y & pred).sum()
    fn = (y & ~pred).sum()
    return float((c_fp * fp + c_fn * fn) / len(y))


def curva_costo_umbral(
    y: np.ndarray,
    proba: np.ndarray,
    c_fp: float = 1.0,
    c_fn: float = 1.0,
    umbrales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Costo esperado para una rejilla de umbrales; devuelve ``(umbrales, costos)``."""
    if umbrales is None:
        umbrales = np.linspace(0.05, 0.95, 91)
    costos = np.array([costo_esperado(y, proba, t, c_fp, c_fn) for t in umbrales])
    return np.asarray(umbrales), costos


def umbral_por_costo(
    y: np.ndarray,
    proba: np.ndarray,
    c_fp: float = 1.0,
    c_fn: float = 1.0,
    umbrales: np.ndarray | None = None,
    referencia: float = 0.5,
) -> dict:
    """Umbral que minimiza el costo esperado FP/FN en validación.

    En empates elige el más cercano a ``referencia`` (0,5 por defecto; para probabilidades
    calibradas un desempate natural es el umbral de Bayes ``c_fp / (c_fp + c_fn)``). Devuelve
    el umbral, su costo y el costo con 0,5 como referencia.
    """
    t, c = curva_costo_umbral(y, proba, c_fp, c_fn, umbrales)
    mejores = np.flatnonzero(np.isclose(c, c.min()))
    k = mejores[np.argmin(np.abs(t[mejores] - referencia))]
    return {
        "umbral": float(t[k]),
        "costo": float(c[k]),
        "costo_en_0_5": costo_esperado(y, proba, 0.5, c_fp, c_fn),
    }


def bootstrap_media(
    valores: np.ndarray, n_boot: int = 1000, seed: int = 42, nivel: float = 0.95
) -> dict:
    """IC bootstrap percentil de la media de ``valores`` (p. ej. errores cuadráticos → Brier).

    Devuelve ``{"valor": media, "ic95": [inf, sup]}``.
    """
    valores = np.asarray(valores, dtype=float)
    medias = [valores[idx].mean(axis=1) for idx in _indices_bootstrap(len(valores), n_boot, seed)]
    a = (1 - nivel) / 2
    lo, hi = np.quantile(np.concatenate(medias), [a, 1 - a])
    return {"valor": float(valores.mean()), "ic95": [float(lo), float(hi)]}


# ---------------------------------------------------------------------------------------
# Métricas con puntuaciones y bootstrap por grupos (reto 3: las filas de un usuario no son
# independientes, así que se remuestrean USUARIOS completos).
# ---------------------------------------------------------------------------------------

METRICAS_PUNTUACION = ("roc_auc", "pr_auc", "brier")
METRICAS_UMBRAL = ("f1", "recall", "precision")


class _Orden:
    """Orden descendente de una puntuación con índice de grupo de empates (precalculado)."""

    def __init__(self, score: np.ndarray):
        score = np.asarray(score, dtype=float)
        self.orden = np.argsort(-score, kind="stable")
        s = score[self.orden]
        nuevo = np.empty(len(s), dtype=bool)
        nuevo[0] = True
        nuevo[1:] = s[1:] != s[:-1]
        self.gid = np.cumsum(nuevo) - 1
        self.n_gid = int(self.gid[-1]) + 1


def _auc_ap_ponderados(y: np.ndarray, w: np.ndarray, o: _Orden) -> tuple[float, float]:
    """ROC-AUC y average precision con pesos de fila ``w``; los empates se tratan en bloque."""
    yo, wo = y[o.orden], w[o.orden]
    wp = np.bincount(o.gid, weights=wo * yo, minlength=o.n_gid)
    wn = np.bincount(o.gid, weights=wo * (1 - yo), minlength=o.n_gid)
    tot_p, tot_n = wp.sum(), wn.sum()
    if tot_p == 0 or tot_n == 0:
        return float("nan"), float("nan")
    cum_p, cum_n = np.cumsum(wp), np.cumsum(wn)
    neg_debajo = tot_n - cum_n
    auc = float((wp * (neg_debajo + 0.5 * wn)).sum() / (tot_p * tot_n))
    prec = cum_p / np.maximum(cum_p + cum_n, 1e-300)
    ap = float((wp / tot_p * prec).sum())
    return auc, ap


def _metricas_ponderadas(
    y: np.ndarray, score: np.ndarray, w: np.ndarray, o: _Orden, umbral: float | None
) -> dict:
    auc, ap = _auc_ap_ponderados(y, w, o)
    sw = w.sum()
    out = {"roc_auc": auc, "pr_auc": ap, "brier": float((w * (score - y) ** 2).sum() / sw)}
    if umbral is not None:
        pred = score >= umbral
        tp = (w * (pred & (y == 1))).sum()
        fp = (w * (pred & (y == 0))).sum()
        fn = (w * (~pred & (y == 1))).sum()
        out["f1"] = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        out["recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        out["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    return out


def metricas_puntuacion(y: np.ndarray, score: np.ndarray, umbral: float | None = None) -> dict:
    """ROC-AUC, PR-AUC (average precision), Brier y, con ``umbral``, F1/recall/precisión.

    Equivalente a ``sklearn.metrics`` (empates tratados igual), pero admite el bootstrap por
    grupos de ``bootstrap_por_grupos``. ``score`` debe estar en [0, 1] para que el Brier tenga
    sentido. La clase positiva es ``y == 1``; el umbral decide ``score >= umbral``.
    """
    y = np.asarray(y).astype(float)
    score = np.asarray(score, dtype=float)
    return _metricas_ponderadas(y, score, np.ones(len(y)), _Orden(score), umbral)


def p_valor_permutacion(observado: float, nulos: np.ndarray) -> float:
    """P-valor empírico unilateral con corrección +1: ``(1 + #{nulo >= obs}) / (1 + B)``."""
    nulos = np.asarray(nulos, dtype=float)
    return float((1 + np.sum(nulos >= observado)) / (1 + len(nulos)))


def bootstrap_por_grupos(
    y: np.ndarray,
    puntuaciones: dict[str, np.ndarray],
    grupos: np.ndarray,
    umbrales: dict[str, float] | None = None,
    referencia: str | None = None,
    n_boot: int = 1000,
    seed: int = 42,
    nivel: float = 0.95,
) -> dict:
    """IC bootstrap percentil remuestreando GRUPOS (usuarios) completos, con pares opcionales.

    En cada remuestreo se eligen ``n_grupos`` grupos con reemplazo; las filas de un grupo
    elegido ``k`` veces pesan ``k``. Todos los modelos usan los MISMOS remuestreos, de modo que
    las diferencias frente a ``referencia`` son pareadas. Es mucho más eficiente que
    reconstruir los arrays (O(n) por remuestreo con ``bincount``).

    Args:
        y: etiquetas 0/1 por fila.
        puntuaciones: ``{modelo: score}`` (probabilidades en [0, 1]).
        grupos: identificador de grupo por fila (p. ej. ``User_ID``).
        umbrales: ``{modelo: umbral}`` para añadir F1, recall y precisión a esos modelos.
        referencia: modelo contra el que se calculan diferencias (A - referencia) pareadas.
        n_boot, seed, nivel: remuestreos, semilla e intervalo.

    Returns:
        ``{"modelos": {m: {métrica: {"valor", "ic95"}}}, "diferencias": {m: {métrica:
        {"diferencia", "ic95"}}}}``. Las diferencias solo incluyen métricas sin umbral.
    """
    y = np.asarray(y).astype(float)
    umbrales = umbrales or {}
    _, inv = np.unique(np.asarray(grupos), return_inverse=True)
    n_g = int(inv.max()) + 1
    ordenes = {m: _Orden(s) for m, s in puntuaciones.items()}
    scores = {m: np.asarray(s, dtype=float) for m, s in puntuaciones.items()}
    uno = np.ones(len(y))
    puntual = {
        m: _metricas_ponderadas(y, scores[m], uno, ordenes[m], umbrales.get(m)) for m in scores
    }
    muestras = {m: {k: [] for k in puntual[m]} for m in scores}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        cuenta = np.bincount(rng.integers(0, n_g, size=n_g), minlength=n_g)
        w = cuenta[inv].astype(float)
        for m in scores:
            r = _metricas_ponderadas(y, scores[m], w, ordenes[m], umbrales.get(m))
            for k, v in r.items():
                muestras[m][k].append(v)
    a = (1 - nivel) / 2
    modelos = {}
    for m in scores:
        modelos[m] = {}
        for k, v in muestras[m].items():
            lo, hi = np.nanquantile(np.array(v), [a, 1 - a])
            modelos[m][k] = {"valor": float(puntual[m][k]), "ic95": [float(lo), float(hi)]}
    diferencias = {}
    if referencia is not None:
        for m in scores:
            if m == referencia:
                continue
            diferencias[m] = {}
            for k in METRICAS_PUNTUACION:
                d = np.array(muestras[m][k]) - np.array(muestras[referencia][k])
                lo, hi = np.nanquantile(d, [a, 1 - a])
                diferencias[m][k] = {
                    "diferencia": float(puntual[m][k] - puntual[referencia][k]),
                    "ic95": [float(lo), float(hi)],
                }
    return {"modelos": modelos, "diferencias": diferencias}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def ajustar_calibrador(metodo: str, p: np.ndarray, y: np.ndarray):
    """Ajusta un calibrador monótono sobre probabilidades ``p`` y devuelve una función ``f(p)``.

    ``"sigmoide"``: regresión logística (Platt) sobre ``logit(p)``; ``"isotonica"``: regresión
    isotónica (creciente) con recorte a [0, 1]. El calibrador solo aprende de los ``(p, y)``
    recibidos. Ambos son monótonos NO decrecientes: si la pendiente de Platt saliera negativa
    (la puntuación está invertida respecto a la etiqueta en esos datos, típico cuando no hay
    señal), se devuelve la prevalencia constante en lugar de invertir el ranking.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    p = np.asarray(p, dtype=float)
    if metodo == "sigmoide":
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p), y)
        if lr.coef_[0, 0] <= 0:
            prev = float(np.mean(y))
            return lambda q: np.full(np.shape(q)[0], prev)
        return lambda q: lr.predict_proba(_logit(q))[:, 1]
    if metodo == "isotonica":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return lambda q: iso.predict(np.asarray(q, dtype=float))
    raise ValueError("metodo debe ser 'sigmoide' o 'isotonica'")


def calibrar_oof_por_grupos(
    metodo: str, p: np.ndarray, y: np.ndarray, grupos: np.ndarray, n_splits: int = 5
) -> np.ndarray:
    """Probabilidades calibradas out-of-fold: cada pliegue (agrupado) usa un calibrador ajustado
    con los demás. Sirve para elegir método y umbral en validación sin evaluar sobre lo ajustado.
    """
    from sklearn.model_selection import GroupKFold

    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    salida = np.empty(len(p))
    for tr, te in GroupKFold(n_splits=n_splits).split(p, y, groups=grupos):
        salida[te] = ajustar_calibrador(metodo, p[tr], y[tr])(p[te])
    return salida
