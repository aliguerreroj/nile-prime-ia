"""Recomendador híbrido de Nile Prime: contenido + filtrado colaborativo ítem-ítem + baselines.

Componentes
-----------
* **Contenido**: TF-IDF por campos (Overview, Genre, Director, Stars) con pesos fijados a priori
  y similitud coseno ítem-ítem.
* **Colaborativo**: kNN ítem-ítem por coseno sobre la matriz usuario × ítem implícita (binaria,
  dispersa). Se elige kNN ítem-ítem y no factorización (ALS/SVD) porque el catálogo es pequeño
  (999 ítems: la matriz de similitudes cabe en 8 MB), no hay librerías de recomendación
  instaladas, no necesita hiperparámetros de rango/regularización y es interpretable
  ("quien vio A vio B").
* **Híbrido ponderado**: ``score = α·z(CF) + (1-α)·z(contenido)``, con cada componente
  normalizado por usuario (z-score sobre los candidatos no vistos), de modo que α pesa
  componentes en la misma escala.
* **Baselines**: aleatorio, popularidad global y popularidad por grupo de edad.
* **Arranque en frío**: usuario sin historial → popularidad de su grupo de edad (global si no
  hay grupo); ítem sin interacciones → solo contenido.

Evaluación (leave-one-out con validación): ``particionar_loo``, ``ordenar_candidatos``,
``metricas_ranking`` (Precision/Recall/NDCG/HitRate@K por usuario), ``cobertura_catalogo``,
``diversidad_intralista``, ``bootstrap_ic`` y el flujo completo ``ejecutar_pipeline``.

Convención: los ítems se indexan por la posición en el catálogo pasado a ``fit`` (999 títulos
únicos); las matrices usuario × ítem son ``scipy.sparse.csr_matrix``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import t as t_student
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer

from src import evaluation as ev

SEED = 42
K_DEFECTO = 10
REJILLA_ALPHA = tuple(round(a, 1) for a in np.arange(0.0, 1.0001, 0.1))
ALPHA_DEFECTO = 0.5
DEFINICIONES = ("cualquiera", "completa")
# Pesos por campo del modelo de contenido: decididos a priori (no se ajustan con datos). Overview
# es corto y ruidoso (mediana 24 palabras) → peso moderado; Genre es el campo más informativo
# y denso; Stars (4 actores) pesa algo más que Director (1 persona, 547 distintos).
PESOS_CAMPOS = {"Overview": 0.35, "Genre": 0.30, "Director": 0.15, "Stars": 0.20}
_JITTER = 1e-9  # desempate aleatorio reproducible (muy por debajo de cualquier diferencia real)


# ---------------------------------------------------------------------------------------------
# Modelo de contenido
# ---------------------------------------------------------------------------------------------
def _tokens_genero(texto: str) -> list[str]:
    return [t for t in texto.split(", ") if t]


def _tokens_unico(texto: str) -> list[str]:
    return [texto] if texto else []


def _tokens_estrellas(texto: str) -> list[str]:
    return [t for t in texto.split("|") if t]


def matriz_contenido(
    catalogo: pd.DataFrame, pesos: dict[str, float] | None = None
) -> sp.csr_matrix:
    """Matriz TF-IDF ponderada por campos (filas con norma 1), un vector por ítem.

    Cada campo se vectoriza por separado (TF-IDF con L2 por campo) y se concatena escalado por
    ``sqrt(peso)``. Así el producto escalar entre dos ítems es la suma ponderada de las
    similitudes coseno por campo, y vale entre 0 y 1 si los pesos suman 1 (se normalizan).

    * Overview: palabras (sin *stopwords* inglesas, ``sublinear_tf``).
    * Genre: cada género es un término (multi-hot con IDF: los géneros raros pesan más).
    * Director: el nombre completo es un único término.
    * Stars: Star1-Star4 son términos.

    Los vectorizadores se ajustan sobre los metadatos del catálogo (no usan interacciones ni
    etiquetas), por lo que no hay fuga de datos de evaluación.
    """
    pesos = dict(PESOS_CAMPOS if pesos is None else pesos)
    total = sum(pesos.values())
    pesos = {c: w / total for c, w in pesos.items()}
    estrellas = (
        catalogo[["Star1", "Star2", "Star3", "Star4"]].fillna("").astype(str).agg("|".join, axis=1)
    )
    campos = {
        "Overview": (
            catalogo["Overview"].fillna("").astype(str),
            TfidfVectorizer(stop_words="english", sublinear_tf=True, dtype=np.float64),
        ),
        "Genre": (
            catalogo["Genre"].fillna("").astype(str),
            TfidfVectorizer(analyzer=_tokens_genero, dtype=np.float64),
        ),
        "Director": (
            catalogo["Director"].fillna("").astype(str),
            TfidfVectorizer(analyzer=_tokens_unico, dtype=np.float64),
        ),
        "Stars": (estrellas, TfidfVectorizer(analyzer=_tokens_estrellas, dtype=np.float64)),
    }
    bloques = [
        np.sqrt(pesos.get(nombre, 0.0)) * vec.fit_transform(serie)
        for nombre, (serie, vec) in campos.items()
        if pesos.get(nombre, 0.0) > 0
    ]
    return sp.hstack(bloques, format="csr")


def similitud_contenido(
    catalogo: pd.DataFrame, pesos: dict[str, float] | None = None
) -> np.ndarray:
    """Similitud coseno ítem-ítem de contenido (matriz densa n×n, diagonal = 0)."""
    x = matriz_contenido(catalogo, pesos)
    s = (x @ x.T).toarray()
    np.clip(s, 0.0, 1.0, out=s)
    np.fill_diagonal(s, 0.0)
    return s


def vecinos_contenido(
    sim: np.ndarray, catalogo: pd.DataFrame, titulo: str, n: int = 10
) -> pd.DataFrame:
    """Los ``n`` ítems más parecidos por contenido a ``titulo`` (con su similitud coseno)."""
    pos = np.flatnonzero(catalogo["Series_Title"].to_numpy() == titulo)
    if len(pos) == 0:
        raise KeyError(titulo)
    i = int(pos[0])
    orden = np.argsort(-sim[i], kind="stable")[:n]
    cols = [c for c in ("movie_id", "Series_Title", "Released_Year", "Genre", "Director")
            if c in catalogo.columns]
    out = catalogo.iloc[orden][cols].copy()
    out.insert(2, "similitud", sim[i, orden])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Utilidades de puntuación y ranking
# ---------------------------------------------------------------------------------------------
def _z_filas(raw: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Z-score por fila usando solo las posiciones de ``mascara``; el resto queda en 0.

    Filas sin dispersión (desviación 0) devuelven 0 (sin información para ordenar).
    """
    m = mascara.sum(axis=1, keepdims=True).clip(min=1)
    mu = (raw * mascara).sum(axis=1, keepdims=True) / m
    dif = (raw - mu) * mascara
    sd = np.sqrt((dif**2).sum(axis=1, keepdims=True) / m)
    return np.divide(dif, sd, out=np.zeros_like(dif), where=sd > 1e-12)


def jitter_para(forma: tuple[int, int], seed: int = SEED) -> np.ndarray:
    """Matriz de desempate aleatorio reproducible (U(0, 1e-9)) para romper empates en rankings."""
    return np.random.default_rng(seed).random(forma) * _JITTER


def ordenar_candidatos(
    scores: np.ndarray,
    vistos: np.ndarray,
    k: int | None = None,
    jitter: np.ndarray | None = None,
    seed: int = SEED,
) -> np.ndarray:
    """Índices de ítems ordenados por puntuación descendente, excluyendo los ya vistos.

    Args:
        scores: matriz (usuarios × ítems).
        vistos: máscara booleana de la misma forma; los vistos quedan al final del orden.
        k: si se da, devuelve solo los ``k`` primeros (``argpartition``, más rápido).
        jitter: desempate aleatorio aditivo (ver ``jitter_para``); si es ``None`` se genera.
    """
    if jitter is None:
        jitter = jitter_para(scores.shape, seed)
    s = np.where(vistos, -np.inf, scores + jitter)
    n_items = s.shape[1]
    if k is None or k >= n_items:
        return np.argsort(-s, axis=1, kind="stable")
    parte = np.argpartition(-s, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(s, parte, axis=1)
    return np.take_along_axis(parte, np.argsort(-vals, axis=1, kind="stable"), axis=1)


def rangos_desde_orden(orden: np.ndarray, vistos: np.ndarray) -> np.ndarray:
    """Matriz de rangos (1 = primero) de cada ítem para cada usuario; vistos = -1."""
    n, ni = orden.shape
    r = np.empty((n, ni), dtype=np.int16)
    np.put_along_axis(r, orden, np.broadcast_to(np.arange(1, ni + 1, dtype=np.int16), (n, ni)), 1)
    r[vistos] = -1
    return r


def rango_objetivo(
    scores: np.ndarray, vistos: np.ndarray, objetivo: np.ndarray, jitter: np.ndarray
) -> np.ndarray:
    """Rango (1 = primero) del ítem ``objetivo`` de cada usuario entre sus candidatos no vistos.

    Equivale a ``rangos_desde_orden`` pero sin ordenar todo (cuenta ítems con mayor puntuación).
    """
    s = scores + jitter
    st = s[np.arange(len(s)), objetivo]
    return ((s > st[:, None]) & ~vistos).sum(axis=1) + 1


def metricas_ranking(topk: np.ndarray, relevantes: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """Precision@K, Recall@K, NDCG@K y HitRate@K por usuario (vectorizado).

    Args:
        topk: (usuarios × ≥k) índices recomendados en orden de ranking.
        relevantes: máscara booleana (usuarios × ítems) de ítems relevantes (retenidos).
        k: corte.

    Definiciones: ``precision = aciertos/k``; ``recall = aciertos/|relevantes|`` (0 si no hay);
    ``ndcg = DCG/IDCG`` con ganancia binaria y descuento ``1/log2(posición+1)``;
    ``hit = 1 si aciertos > 0``. Con UN solo relevante por usuario: recall = hit y
    precision = hit/k (no son evidencia independiente).
    """
    topk = topk[:, :k]
    n = len(topk)
    acierto = relevantes[np.arange(n)[:, None], topk]
    descuento = 1.0 / np.log2(np.arange(2, k + 2))
    aciertos = acierto.sum(axis=1)
    n_rel = relevantes.sum(axis=1)
    dcg = (acierto * descuento).sum(axis=1)
    idcg_tabla = np.concatenate([[0.0], np.cumsum(descuento)])
    idcg = idcg_tabla[np.minimum(n_rel, k)]
    return {
        "precision": aciertos / k,
        "recall": np.divide(aciertos, n_rel, out=np.zeros(n), where=n_rel > 0),
        "ndcg": np.divide(dcg, idcg, out=np.zeros(n), where=idcg > 0),
        "hit": (aciertos > 0).astype(float),
    }


def metricas_desde_rango(rango: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """Las 4 métricas por usuario con UN ítem relevante, a partir de su rango (1 = primero)."""
    hit = (rango <= k).astype(float)
    return {
        "precision": hit / k,
        "recall": hit,
        "ndcg": np.where(rango <= k, 1.0 / np.log2(np.maximum(rango, 1) + 1), 0.0),
        "hit": hit,
    }


def azar_teorico(n_candidatos: np.ndarray, k: int = K_DEFECTO) -> dict[str, float]:
    """Valor esperado de HitRate@K y NDCG@K de un ranking aleatorio con un solo relevante.

    Con ``N`` candidatos por usuario: HitRate = K/N y NDCG = (1/N)·Σ_{r=1..K} 1/log2(r+1);
    se promedia entre usuarios.
    """
    n_candidatos = np.asarray(n_candidatos, dtype=float)
    desc = (1.0 / np.log2(np.arange(2, k + 2))).sum()
    return {
        "hit": float(np.mean(np.minimum(k, n_candidatos) / n_candidatos)),
        "ndcg": float(np.mean(desc / n_candidatos)),
    }


def cobertura_catalogo(topk: np.ndarray, n_items: int) -> float:
    """Fracción del catálogo que aparece al menos una vez en alguna lista recomendada."""
    return float(len(np.unique(topk)) / n_items)


def diversidad_intralista(topk: np.ndarray, sim: np.ndarray) -> np.ndarray:
    """Diversidad intra-lista por usuario: 1 − similitud de contenido media entre pares de la lista.

    ``sim`` debe tener diagonal 0 (como ``similitud_contenido``), de modo que la suma de la
    submatriz k×k solo contiene pares distintos.
    """
    n, k = topk.shape
    sub = sim[topk[:, :, None], topk[:, None, :]]
    return 1.0 - sub.sum(axis=(1, 2)) / (k * (k - 1))


def bootstrap_ic(
    columnas: dict[str, np.ndarray],
    n_boot: int = 1000,
    seed: int = SEED,
    nivel: float = 0.95,
    bloque: int = 50,
) -> dict[str, dict]:
    """IC bootstrap percentil de la media de varias columnas, remuestreando USUARIOS.

    Todas las columnas comparten los mismos remuestreos (misma semilla y mismo ``n``), de modo
    que los IC de distintos recomendadores son comparables y las diferencias son pareadas.
    Devuelve ``{nombre: {"valor": media, "ic95": [inf, sup]}}``.
    """
    nombres = list(columnas)
    m = np.column_stack([np.asarray(columnas[c], dtype=float) for c in nombres])
    rng = np.random.default_rng(seed)
    medias = []
    for _ in range(0, n_boot, bloque):
        b = min(bloque, n_boot - len(medias) * bloque)
        idx = rng.integers(0, len(m), size=(b, len(m)))
        medias.append(m[idx].mean(axis=1))
    medias = np.concatenate(medias)[:n_boot]
    a = (1 - nivel) / 2
    lo, hi = np.quantile(medias, [a, 1 - a], axis=0)
    return {
        c: {"valor": float(m[:, j].mean()), "ic95": [float(lo[j]), float(hi[j])]}
        for j, c in enumerate(nombres)
    }


def comparar_pareado(
    met_a: dict[str, np.ndarray], met_b: dict[str, np.ndarray], n_boot: int = 1000,
    seed: int = SEED,
) -> dict:
    """Comparación pareada por usuario A − B en HitRate@K y NDCG@K.

    * HitRate: diferencia con IC bootstrap y prueba de McNemar exacta sobre los aciertos.
    * NDCG: diferencia con IC bootstrap y prueba de rangos con signo de Wilcoxon.

    ``a_mejor`` es True solo si la diferencia es positiva, el IC95 excluye 0 y p < 0,05 en la
    métrica correspondiente.
    """
    d_hit = met_a["hit"] - met_b["hit"]
    d_ndcg = met_a["ndcg"] - met_b["ndcg"]
    ic = bootstrap_ic({"hit": d_hit, "ndcg": d_ndcg}, n_boot=n_boot, seed=seed)
    mc = ev.mcnemar_exacto(np.ones(len(d_hit)), met_a["hit"], met_b["hit"])
    p_w = _p_wilcoxon(d_ndcg)
    return {
        "hit": {
            "diferencia": ic["hit"]["valor"], "ic95": ic["hit"]["ic95"],
            "p_mcnemar": mc["p_valor"], "a_acierta_b_falla": mc["a_acierta_b_falla"],
            "a_falla_b_acierta": mc["a_falla_b_acierta"],
            "a_mejor": bool(ic["hit"]["ic95"][0] > 0 and mc["p_valor"] < 0.05),
        },
        "ndcg": {
            "diferencia": ic["ndcg"]["valor"], "ic95": ic["ndcg"]["ic95"], "p_wilcoxon": p_w,
            "a_mejor": bool(ic["ndcg"]["ic95"][0] > 0 and p_w < 0.05),
        },
    }


def _p_wilcoxon(dif: np.ndarray) -> float:
    """p-valor bilateral de Wilcoxon sobre diferencias; 1 si todas son 0."""
    if not np.any(dif != 0):
        return 1.0
    return float(wilcoxon(dif, zero_method="wilcox").pvalue)


def prueba_permutacion(rangos: np.ndarray, objetivo: np.ndarray, k: int = K_DEFECTO,
                       n_perm: int = 1000, seed: int = SEED) -> dict:
    """Prueba de permutación de la existencia de señal usuario-ítem en HitRate@K y NDCG@K.

    H0: el ítem retenido no depende del usuario. Se simula barajando entre usuarios el ítem
    retenido (conserva la distribución de popularidad de los retenidos y rompe el vínculo
    usuario-ítem). Los pares barajados cuyo ítem ya estaba en el historial del usuario
    (rango -1) se descartan. ``rangos`` es la matriz usuarios × ítems de ``rangos_desde_orden``
    (se calcula una sola vez y se reutiliza en todas las réplicas).

    Returns:
        observados, distribución nula (medias y percentil 95), p-valores empíricos de una cola
        con corrección +1 ``(1 + #{nulo >= obs}) / (n_perm + 1)`` y la fracción de pares válidos.
    """
    n = len(objetivo)
    idx = np.arange(n)
    obs = metricas_desde_rango(rangos[idx, objetivo].astype(int), k)
    obs_hit, obs_ndcg = float(obs["hit"].mean()), float(obs["ndcg"].mean())
    rng = np.random.default_rng(seed)
    nulo_hit = np.empty(n_perm)
    nulo_ndcg = np.empty(n_perm)
    validos = np.empty(n_perm)
    for i in range(n_perm):
        rr = rangos[idx, objetivo[rng.permutation(n)]].astype(int)
        ok = rr > 0
        dentro = ok & (rr <= k)
        nulo_hit[i] = dentro.sum() / ok.sum()
        nulo_ndcg[i] = (1.0 / np.log2(np.maximum(rr, 1) + 1))[dentro].sum() / ok.sum()
        validos[i] = ok.mean()
    return {
        "hit_obs": obs_hit, "ndcg_obs": obs_ndcg,
        "hit_nulo_media": float(nulo_hit.mean()),
        "hit_nulo_p95": float(np.quantile(nulo_hit, 0.95)),
        "ndcg_nulo_media": float(nulo_ndcg.mean()),
        "ndcg_nulo_p95": float(np.quantile(nulo_ndcg, 0.95)),
        "p_hit": float((1 + (nulo_hit >= obs_hit).sum()) / (n_perm + 1)),
        "p_ndcg": float((1 + (nulo_ndcg >= obs_ndcg).sum()) / (n_perm + 1)),
        "n_perm": n_perm, "fraccion_pares_validos": float(validos.mean()),
        "nulo_hit": nulo_hit, "nulo_ndcg": nulo_ndcg,
    }


# ---------------------------------------------------------------------------------------------
# Recomendadores
# ---------------------------------------------------------------------------------------------
def pesos_por_colision(catalogo: pd.DataFrame, metadatos_crudos: pd.DataFrame) -> np.ndarray:
    """Peso de popularidad de cada ítem del catálogo ante títulos repetidos (colisión de títulos).

    ``interacciones.csv`` solo trae ``Series_Title``: si dos películas distintas comparten título
    (p. ej. "Drishyam" 2013 y 2015) y el catálogo conserva una sola fila por título, las
    interacciones de ambas se asignan a esa fila y su popularidad se infla. El peso es
    ``1 / n`` con ``n`` = nº de filas con ese título en los metadatos crudos (1.000 filas), de modo
    que el recuento de popularidad del ítem colisionado se reparte entre las películas que
    comparten el título. La colisión se detecta en los datos, no por nombre.

    Args:
        catalogo: catálogo con un ítem por título (``Series_Title``).
        metadatos_crudos: metadatos completos (``load_metadata(titulos_unicos=False)``).

    Returns:
        Vector de pesos en (0, 1], alineado con las filas de ``catalogo``.
    """
    n_por_titulo = metadatos_crudos["Series_Title"].value_counts()
    n = n_por_titulo.reindex(catalogo["Series_Title"].to_numpy()).fillna(1).to_numpy(float)
    return 1.0 / n


class _RecomendadorBase:
    """Estructura común: matrices usuario × ítem, popularidades y manejo del arranque en frío.

    Dos matrices dispersas binarias: ``vistos_`` (cualquier interacción; define qué se excluye
    de las recomendaciones) y ``pos_`` (interacción positiva según ``positivo``):

    * ``"cualquiera"``: toda película vista cuenta como positiva.
    * ``"completa"``: solo ``Watched_to_End = True``.
    """

    nombre = "base"

    def __init__(self, positivo: str = "cualquiera", seed: int = SEED, pesos_popularidad=None):
        if positivo not in DEFINICIONES:
            raise ValueError(f"positivo debe ser uno de {DEFINICIONES}")
        self.positivo = positivo
        self.seed = seed
        # Pesos por ítem (ver ``pesos_por_colision``) que se aplican SOLO al recuento de
        # popularidad (global y por edad); no afectan a CF ni a contenido. None = sin ajuste.
        self.pesos_popularidad = pesos_popularidad
        self._rng = np.random.default_rng(seed)

    # -- ajuste ---------------------------------------------------------------------------
    def fit(self, catalogo: pd.DataFrame, interacciones: pd.DataFrame, sim_contenido=None):
        """Ajusta el modelo con las interacciones de entrenamiento.

        Args:
            catalogo: metadatos de los ítems (fija el índice de ítems = posición de la fila).
            interacciones: columnas ``User_ID``, ``Series_Title``, ``Watched_to_End``,
                ``Age_Group``. Los ítems retenidos para validación/test NO deben estar aquí.
            sim_contenido: similitud de contenido precalculada (solo depende del catálogo; se
                puede compartir entre modelos para ahorrar tiempo).
        """
        self.catalogo_ = catalogo.reset_index(drop=True)
        self.titulos_ = self.catalogo_["Series_Title"].to_numpy()
        self.n_items_ = len(self.titulos_)
        titulo_a_idx = pd.Series(np.arange(self.n_items_), index=self.titulos_)
        usuarios, filas = np.unique(interacciones["User_ID"].to_numpy(), return_inverse=True)
        cols = titulo_a_idx.reindex(interacciones["Series_Title"].to_numpy()).to_numpy()
        if np.isnan(cols).any():
            raise ValueError("hay títulos de interacciones que no están en el catálogo")
        cols = cols.astype(int)
        completa = interacciones["Watched_to_End"].to_numpy(dtype=bool)
        uno = np.ones(len(filas), dtype=np.float64)
        forma = (len(usuarios), self.n_items_)
        self.vistos_ = sp.csr_matrix((uno, (filas, cols)), shape=forma)
        self.vistos_.data[:] = 1.0
        pos = completa if self.positivo == "completa" else np.ones(len(filas), bool)
        self.pos_ = sp.csr_matrix((uno[pos], (filas[pos], cols[pos])), shape=forma)
        self.pos_.data[:] = 1.0
        self.usuarios_ = usuarios
        edad_u = pd.Series(interacciones["Age_Group"].to_numpy()).groupby(filas).first()
        self.grupos_ = sorted(pd.unique(edad_u.to_numpy()))
        g_idx = {g: i for i, g in enumerate(self.grupos_)}
        self.edad_usuario_ = np.array([g_idx[g] for g in edad_u.to_numpy()])
        self.pop_global_ = np.asarray(self.pos_.sum(axis=0)).ravel()
        self.pop_edad_ = np.vstack(
            [
                np.asarray(self.pos_[self.edad_usuario_ == i].sum(axis=0)).ravel()
                for i in range(len(self.grupos_))
            ]
        )
        self.item_frio_ = self.pop_global_ == 0
        if self.pesos_popularidad is not None:
            w = np.asarray(self.pesos_popularidad, dtype=float)
            if w.shape != (self.n_items_,):
                raise ValueError("pesos_popularidad debe tener un valor por ítem del catálogo")
            self.pop_global_ = self.pop_global_ * w
            self.pop_edad_ = self.pop_edad_ * w[None, :]
        self._ajustar(sim_contenido)
        return self

    def _ajustar(self, sim_contenido) -> None:
        """Gancho para que las subclases construyan sus estructuras propias."""

    # -- puntuación -----------------------------------------------------------------------
    def _filas(self, user_ids) -> np.ndarray:
        """Fila de cada ``user_id`` en las matrices (-1 si el usuario es desconocido)."""
        pos = np.searchsorted(self.usuarios_, user_ids)
        pos = np.clip(pos, 0, len(self.usuarios_) - 1)
        return np.where(self.usuarios_[pos] == user_ids, pos, -1)

    def _indice_grupo(self, grupos) -> np.ndarray:
        """Índice del grupo de edad (-1 si es ``None`` o desconocido)."""
        g_idx = {g: i for i, g in enumerate(self.grupos_)}
        return np.array([g_idx.get(g, -1) if g is not None else -1 for g in grupos], dtype=int)

    def _puntuar_frio(self, idx_grupo: np.ndarray) -> np.ndarray:
        """Arranque en frío de usuario: popularidad de su grupo de edad (global si no hay)."""
        g = np.where(idx_grupo >= 0, idx_grupo, 0)
        por_edad = self.pop_edad_[g]
        return np.where(idx_grupo[:, None] >= 0, por_edad, self.pop_global_[None, :]).astype(float)

    def _puntuar_filas(self, pos_rows: sp.csr_matrix, vistos: np.ndarray, **kw) -> np.ndarray:
        raise NotImplementedError

    def puntuar_lote(self, user_ids, grupos_edad=None, **kw) -> tuple[np.ndarray, np.ndarray]:
        """Puntúa todos los ítems para varios usuarios a la vez.

        Args:
            user_ids: identificadores de usuario (conocidos o no).
            grupos_edad: grupo de edad de cada usuario (opcional; para desconocidos o sin
                historial positivo; los conocidos usan el que se vio en el ajuste).

        Returns:
            ``(scores, vistos)``: matrices (usuarios × ítems); ``vistos`` es True para los
            ítems ya vistos en el entrenamiento (deben excluirse al recomendar).
        """
        user_ids = np.asarray(user_ids)
        n = len(user_ids)
        filas = self._filas(user_ids)
        conocido = filas >= 0
        vistos = np.zeros((n, self.n_items_), dtype=bool)
        npos = np.zeros(n, dtype=int)
        grupos = np.full(n, -1, dtype=int)
        if grupos_edad is not None:
            grupos = self._indice_grupo(list(grupos_edad))
        if conocido.any():
            fr = filas[conocido]
            vistos[conocido] = self.vistos_[fr].toarray() > 0
            npos[conocido] = np.diff(self.pos_.indptr)[fr]
            grupos[conocido] = self.edad_usuario_[fr]
        con_hist = npos > 0
        scores = np.zeros((n, self.n_items_))
        if con_hist.any():
            scores[con_hist] = self._puntuar_filas(
                self.pos_[filas[con_hist]], vistos[con_hist], **kw
            )
        if (~con_hist).any():
            scores[~con_hist] = self._puntuar_frio_n(grupos[~con_hist])
        return scores, vistos

    def _puntuar_frio_n(self, idx_grupo: np.ndarray) -> np.ndarray:
        return self._puntuar_frio(idx_grupo)

    # -- API pública ----------------------------------------------------------------------
    def recommend(self, user_id, k: int = K_DEFECTO, grupo_edad=None, excluir_vistos=True):
        """Top-``k`` recomendaciones para un usuario, sin incluir lo ya visto.

        Si el usuario es desconocido (o no tiene historial positivo) se recurre a la
        popularidad de su ``grupo_edad`` (global si no se indica o no existe).

        Returns:
            DataFrame con ``movie_id``, ``Series_Title`` y ``score``, de mayor a menor.
        """
        scores, vistos = self.puntuar_lote([user_id], [grupo_edad])
        if not excluir_vistos:
            vistos = np.zeros_like(vistos)
        orden = ordenar_candidatos(scores, vistos, k=k, seed=self.seed)[0]
        orden = orden[~vistos[0, orden]]
        return pd.DataFrame(
            {
                "movie_id": self.catalogo_["movie_id"].to_numpy()[orden],
                "Series_Title": self.titulos_[orden],
                "score": scores[0, orden],
            }
        )

    def recommend_historial(self, titulos, k: int = K_DEFECTO, grupo_edad=None) -> pd.DataFrame:
        """Recomienda a partir de una lista de títulos vistos (usuario hipotético o nuevo).

        Útil para el arranque en frío con historial mínimo. Sin títulos válidos → popularidad
        por grupo de edad.
        """
        idx = np.flatnonzero(np.isin(self.titulos_, list(titulos)))
        vistos = np.zeros((1, self.n_items_), dtype=bool)
        vistos[0, idx] = True
        if len(idx):
            fila = sp.csr_matrix(
                (np.ones(len(idx)), (np.zeros(len(idx), int), idx)), shape=(1, self.n_items_)
            )
            scores = self._puntuar_filas(fila, vistos)
        else:
            g = self._indice_grupo([grupo_edad])
            scores = self._puntuar_frio(g)
        orden = ordenar_candidatos(scores, vistos, k=k, seed=self.seed)[0]
        return pd.DataFrame(
            {
                "movie_id": self.catalogo_["movie_id"].to_numpy()[orden],
                "Series_Title": self.titulos_[orden],
                "score": scores[0, orden],
            }
        )

    def tamano_en_memoria(self) -> int:
        """Bytes ocupados por los arrays y matrices dispersas del modelo (aprox.)."""
        total = 0
        for v in self.__dict__.values():
            if isinstance(v, np.ndarray):
                total += v.nbytes
            elif sp.issparse(v):
                total += v.data.nbytes + v.indices.nbytes + v.indptr.nbytes
        return total


class RandomRecommender(_RecomendadorBase):
    """Baseline aleatorio: puntuaciones uniformes (el valor esperado es el azar teórico)."""

    nombre = "Aleatorio"

    def _puntuar_filas(self, pos_rows, vistos, **kw):
        return self._rng.random(vistos.shape)

    def _puntuar_frio_n(self, idx_grupo):
        return self._rng.random((len(idx_grupo), self.n_items_))


class PopularityRecommender(_RecomendadorBase):
    """Baseline de popularidad global: nº de interacciones positivas de cada ítem."""

    nombre = "Popularidad global"

    def _puntuar_filas(self, pos_rows, vistos, **kw):
        return np.broadcast_to(self.pop_global_.astype(float), vistos.shape).copy()

    def _puntuar_frio_n(self, idx_grupo):
        forma = (len(idx_grupo), self.n_items_)
        return np.broadcast_to(self.pop_global_.astype(float), forma).copy()


class AgePopularityRecommender(_RecomendadorBase):
    """Baseline de popularidad por grupo de edad (también el recurso de arranque en frío)."""

    nombre = "Popularidad por edad"

    def _puntuar_filas(self, pos_rows, vistos, **kw):
        # El grupo de cada fila se resuelve en ``puntuar_lote``; aquí se recalcula por índice.
        raise RuntimeError("usar puntuar_lote")  # pragma: no cover

    def puntuar_lote(self, user_ids, grupos_edad=None, **kw):
        user_ids = np.asarray(user_ids)
        filas = self._filas(user_ids)
        n = len(user_ids)
        grupos = np.full(n, -1, dtype=int)
        if grupos_edad is not None:
            grupos = self._indice_grupo(list(grupos_edad))
        vistos = np.zeros((n, self.n_items_), dtype=bool)
        conocido = filas >= 0
        if conocido.any():
            vistos[conocido] = self.vistos_[filas[conocido]].toarray() > 0
            grupos[conocido] = self.edad_usuario_[filas[conocido]]
        return self._puntuar_frio(grupos), vistos

    def recommend_historial(self, titulos, k=K_DEFECTO, grupo_edad=None):
        idx = np.flatnonzero(np.isin(self.titulos_, list(titulos)))
        vistos = np.zeros((1, self.n_items_), dtype=bool)
        vistos[0, idx] = True
        scores = self._puntuar_frio(self._indice_grupo([grupo_edad]))
        orden = ordenar_candidatos(scores, vistos, k=k, seed=self.seed)[0]
        return pd.DataFrame(
            {
                "movie_id": self.catalogo_["movie_id"].to_numpy()[orden],
                "Series_Title": self.titulos_[orden],
                "score": scores[0, orden],
            }
        )


class HybridRecommender(_RecomendadorBase):
    """Híbrido ponderado ``score = α·z(CF) + (1-α)·z(contenido)``.

    * ``z(contenido)``: para cada candidato, suma de similitudes de contenido con los ítems
      positivos del usuario, normalizada (z-score) entre sus candidatos no vistos.
    * ``z(CF)``: igual con la similitud coseno ítem-ítem de la matriz usuario × ítem.
    * ``α = 0`` es puro contenido y ``α = 1`` es puro colaborativo.
    * Ítems sin interacciones (``item_frio_``): su componente CF no existe → se puntúan solo
      con contenido (``contenido_en_items_frios=True``); si es False quedan al final.
    * Usuarios sin historial positivo: popularidad de su grupo de edad.

    Los z-scores por usuario ponen a ambos componentes en la misma escala, de modo que α es
    interpretable como peso relativo y no depende del nº de ítems vistos ni de la popularidad.
    """

    nombre = "Híbrido"

    def __init__(self, alpha: float = ALPHA_DEFECTO, positivo: str = "cualquiera",
                 pesos_campos: dict[str, float] | None = None, seed: int = SEED,
                 contenido_en_items_frios: bool = True, pesos_popularidad=None):
        super().__init__(positivo=positivo, seed=seed, pesos_popularidad=pesos_popularidad)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha debe estar en [0, 1]")
        self.alpha = alpha
        self.pesos_campos = pesos_campos
        self.contenido_en_items_frios = contenido_en_items_frios

    def _ajustar(self, sim_contenido) -> None:
        self.sim_contenido_ = (
            similitud_contenido(self.catalogo_, self.pesos_campos)
            if sim_contenido is None else sim_contenido
        )
        co = (self.pos_.T @ self.pos_).toarray()
        norma = np.sqrt(np.diag(co))
        den = np.outer(norma, norma)
        self.sim_cf_ = np.divide(co, den, out=np.zeros_like(co), where=den > 0)
        np.fill_diagonal(self.sim_cf_, 0.0)

    def componentes(self, pos_rows: sp.csr_matrix, vistos: np.ndarray):
        """Componentes normalizados ``(z_contenido, z_cf)`` para filas de historial positivo.

        Se exponen para evaluar muchos α sin recalcular (la combinación es lineal).
        """
        cand = ~vistos
        z_c = _z_filas(np.asarray(pos_rows @ self.sim_contenido_), cand)
        z_cf = _z_filas(np.asarray(pos_rows @ self.sim_cf_), cand & ~self.item_frio_[None, :])
        return z_c, z_cf

    def combinar(self, z_c: np.ndarray, z_cf: np.ndarray, alpha: float) -> np.ndarray:
        """Combina los componentes con peso ``alpha`` y trata los ítems sin interacciones."""
        s = alpha * z_cf + (1.0 - alpha) * z_c
        if self.item_frio_.any():
            if self.contenido_en_items_frios:
                s[:, self.item_frio_] = z_c[:, self.item_frio_]
            else:
                s[:, self.item_frio_] = -1e6
        return s

    def _puntuar_filas(self, pos_rows, vistos, alpha=None, **kw):
        z_c, z_cf = self.componentes(pos_rows, vistos)
        return self.combinar(z_c, z_cf, self.alpha if alpha is None else alpha)


class ContentRecommender(HybridRecommender):
    """Solo contenido (híbrido con α = 0)."""

    nombre = "Contenido"

    def __init__(self, **kw):
        kw.pop("alpha", None)
        super().__init__(alpha=0.0, **kw)


class CollaborativeRecommender(HybridRecommender):
    """Solo colaborativo ítem-ítem (híbrido con α = 1); ítems sin interacciones al final."""

    nombre = "Colaborativo"

    def __init__(self, **kw):
        kw.pop("alpha", None)
        kw["contenido_en_items_frios"] = False
        super().__init__(alpha=1.0, **kw)


# ---------------------------------------------------------------------------------------------
# Partición leave-one-out con validación y pipeline
# ---------------------------------------------------------------------------------------------
def particionar_loo(interacciones: pd.DataFrame, seed: int = SEED, min_completas: int = 3) -> dict:
    """Partición leave-one-out con validación, sin fuga.

    Usuarios elegibles: con al menos ``min_completas`` vistas completas. De sus vistas completas
    (positivas bajo ambas definiciones) se retiran DOS al azar (semilla fija): una a validación
    y otra a test; el resto es train. Los usuarios no elegibles quedan íntegros en train (aportan
    a CF/popularidad y sirven para ilustrar el arranque en frío). No hay marca de tiempo: el
    retenido es aleatorio, no "el último".

    Returns:
        dict con ``train`` (DataFrame), ``val`` y ``test`` (una fila por usuario elegible, mismo
        orden de ``elegibles``, con todas las columnas), y ``elegibles`` (User_ID ordenados).
    """
    df = interacciones.reset_index(drop=True)
    comp = df[df["Watched_to_End"].astype(bool)]
    cuentas = comp.groupby("User_ID").size()
    elegibles = np.sort(cuentas[cuentas >= min_completas].index.to_numpy())
    cand = comp[comp["User_ID"].isin(elegibles)].copy()
    rng = np.random.default_rng(seed)
    cand["_clave"] = rng.random(len(cand))
    cand = cand.sort_values(["User_ID", "_clave"], kind="stable")
    cand["_rango"] = cand.groupby("User_ID").cumcount()
    val = cand[cand["_rango"] == 0].sort_values("User_ID")
    test = cand[cand["_rango"] == 1].sort_values("User_ID")
    train = df.drop(index=np.concatenate([val.index.to_numpy(), test.index.to_numpy()]))
    cols = list(df.columns)
    out = {
        "train": train.reset_index(drop=True),
        "val": val[cols].reset_index(drop=True),
        "test": test[cols].reset_index(drop=True),
        "elegibles": elegibles,
    }
    comprobar_sin_fuga(out)
    return out


def comprobar_sin_fuga(part: dict) -> None:
    """Lanza ``AssertionError`` si algún ítem retenido aparece en el entrenamiento del usuario."""
    tr = set(zip(part["train"]["User_ID"], part["train"]["Series_Title"], strict=False))
    for nombre in ("val", "test"):
        ret = set(zip(part[nombre]["User_ID"], part[nombre]["Series_Title"], strict=False))
        assert not (tr & ret), f"fuga: pares de {nombre} presentes en train"
    assert not (
        set(zip(part["val"]["User_ID"], part["val"]["Series_Title"], strict=False))
        & set(zip(part["test"]["User_ID"], part["test"]["Series_Title"], strict=False))
    ), "fuga: validación y test comparten pares"
    assert np.array_equal(part["val"]["User_ID"].to_numpy(), part["elegibles"])
    assert np.array_equal(part["test"]["User_ID"].to_numpy(), part["elegibles"])


def indices_objetivo(catalogo: pd.DataFrame, retenidos: pd.DataFrame) -> np.ndarray:
    """Índice de catálogo del ítem retenido de cada fila de ``retenidos``."""
    pos = pd.Series(np.arange(len(catalogo)), index=catalogo["Series_Title"].to_numpy())
    return pos.reindex(retenidos["Series_Title"].to_numpy()).to_numpy().astype(int)


def evaluar_scores(scores, vistos, objetivo, k, jitter, guardar_rangos=False) -> dict:
    """Ordena y mide un recomendador sobre usuarios con un ítem retenido (``objetivo``).

    Devuelve las métricas por usuario, el top-k, el rango del objetivo, el nº de candidatos por
    usuario y (opcional) la matriz completa de rangos, usada por la prueba de permutación.
    """
    n = len(scores)
    assert not vistos[np.arange(n), objetivo].any(), "fuga: el ítem retenido figura como visto"
    orden = ordenar_candidatos(scores, vistos, None, jitter)
    rangos = rangos_desde_orden(orden, vistos)
    r_obj = rangos[np.arange(n), objetivo].astype(int)
    rel = np.zeros(scores.shape, dtype=bool)
    rel[np.arange(n), objetivo] = True
    met = metricas_ranking(orden[:, :k], rel, k)
    return {
        "metricas": met,
        "topk": orden[:, :k],
        "rango_objetivo": r_obj,
        "n_candidatos": (~vistos).sum(axis=1),
        "rangos": rangos if guardar_rangos else None,
    }


def ajustar_modelos(catalogo, interacciones, positivo: str, alpha: float, sim_contenido,
                    seed: int = SEED, pesos_popularidad=None) -> dict[str, _RecomendadorBase]:
    """Ajusta los seis recomendadores con la misma definición de positivo y el α dado.

    ``pesos_popularidad`` (ver ``pesos_por_colision``) reparte la popularidad de los títulos
    colisionados; solo afecta a los recuentos de popularidad (global, por edad y arranque en frío).
    """
    kw = {"positivo": positivo, "seed": seed, "pesos_popularidad": pesos_popularidad}
    modelos = {
        "Aleatorio": RandomRecommender(**kw),
        "Popularidad global": PopularityRecommender(**kw),
        "Popularidad por edad": AgePopularityRecommender(**kw),
        "Contenido": ContentRecommender(**kw),
        "Colaborativo": CollaborativeRecommender(**kw),
        "Híbrido": HybridRecommender(alpha=alpha, **kw),
    }
    for m in modelos.values():
        m.fit(catalogo, interacciones, sim_contenido=sim_contenido)
    return modelos


def seleccionar_configuracion(
    catalogo, train, val, usuarios, sim_contenido, k: int = K_DEFECTO,
    rejilla=REJILLA_ALPHA, seed: int = SEED, p_umbral: float = 0.05,
) -> dict:
    """Elige definición de positivo y α SOLO con la validación (el test no interviene).

    Reglas fijadas de antemano:

    * Para cada definición se evalúa el híbrido en toda la rejilla de α (NDCG@K y HitRate@K en
      validación; α = 0 es contenido y α = 1 colaborativo).
    * α de cada definición = argmax del NDCG@K de validación **solo si** supera de forma
      significativa (Wilcoxon pareado, p < ``p_umbral``) a α = 0,5; si no, α = 0,5 (neutro).
    * Definición: ``"completa"`` solo si su híbrido supera significativamente a ``"cualquiera"``;
      en otro caso ``"cualquiera"`` (la más simple, usa todos los datos).

    Returns:
        dict con tablas de validación, α y definición elegidos y los p-valores de las decisiones.
    """
    objetivo = indices_objetivo(catalogo, val)
    jit = jitter_para((len(usuarios), len(catalogo)), seed)
    tablas, vect = {}, {}
    for d in DEFINICIONES:
        m = HybridRecommender(alpha=0.5, positivo=d, seed=seed).fit(
            catalogo, train, sim_contenido=sim_contenido
        )
        fr = m._filas(usuarios)
        vistos = m.vistos_[fr].toarray() > 0
        z_c, z_cf = m.componentes(m.pos_[fr], vistos)
        filas = []
        for a in rejilla:
            sc = m.combinar(z_c, z_cf, a)
            met = metricas_desde_rango(rango_objetivo(sc, vistos, objetivo, jit), k)
            vect[(d, a)] = met
            filas.append({"definicion": d, "alpha": a, "ndcg": float(met["ndcg"].mean()),
                          "hit": float(met["hit"].mean())})
        tablas[d] = filas
    alphas, p_alpha = {}, {}
    for d in DEFINICIONES:
        t = pd.DataFrame(tablas[d])
        mejor = float(t.loc[t["ndcg"].idxmax(), "alpha"])
        dif = vect[(d, mejor)]["ndcg"] - vect[(d, ALPHA_DEFECTO)]["ndcg"]
        p = _p_wilcoxon(dif) if mejor != ALPHA_DEFECTO else 1.0
        p_alpha[d] = {"alpha_max": mejor, "p_vs_0_5": p}
        alphas[d] = mejor if (mejor != ALPHA_DEFECTO and p < p_umbral) else ALPHA_DEFECTO
    dif_def = (
        vect[("completa", alphas["completa"])]["ndcg"]
        - vect[("cualquiera", alphas["cualquiera"])]["ndcg"]
    )
    p_def = _p_wilcoxon(dif_def)
    completa_gana = bool(dif_def.mean() > 0 and p_def < p_umbral)
    definicion = "completa" if completa_gana else "cualquiera"
    return {
        "tablas": tablas,
        "alpha_por_definicion": alphas,
        "p_alpha": p_alpha,
        "definicion": definicion,
        "alpha": alphas[definicion],
        "dif_definiciones": {"ndcg_completa_menos_cualquiera": float(dif_def.mean()),
                             "p_wilcoxon": p_def},
        "completa_gana_significativamente": completa_gana,
    }


def evaluar_test(catalogo, part: dict, config: dict, sim_contenido, k: int = K_DEFECTO,
                 seed: int = SEED, guardar_rangos: bool = False, pesos_popularidad=None) -> dict:
    """Evaluación UNA vez en test con la configuración ya fijada.

    Los modelos se ajustan con train + el ítem de validación (ya es historial conocido) y los
    candidatos son todos los ítems no vistos en ese historial. Devuelve por recomendador el
    resultado de ``evaluar_scores`` y el objeto ``modelos``.
    """
    fit_df = pd.concat([part["train"], part["val"]], ignore_index=True)
    modelos = ajustar_modelos(catalogo, fit_df, config["definicion"], config["alpha"],
                              sim_contenido, seed, pesos_popularidad)
    usuarios = part["elegibles"]
    objetivo = indices_objetivo(catalogo, part["test"])
    jit = jitter_para((len(usuarios), len(catalogo)), seed)
    res = {}
    for nombre, m in modelos.items():
        sc, vis = m.puntuar_lote(usuarios)
        res[nombre] = evaluar_scores(sc, vis, objetivo, k, jit, guardar_rangos)
    return {"resultados": res, "modelos": modelos, "objetivo": objetivo}


def ic_todos(resultados: dict, n_boot: int, seed: int = SEED) -> dict:
    """IC95 bootstrap de las 4 métricas para cada recomendador."""
    return {
        n: bootstrap_ic(r["metricas"], n_boot=n_boot, seed=seed) for n, r in resultados.items()
    }


def ejecutar_pipeline(catalogo, interacciones, k: int = K_DEFECTO, n_boot: int = 1000,
                      rejilla=REJILLA_ALPHA, seed: int = SEED, sim_contenido=None,
                      pesos_popularidad=None) -> dict:
    """Flujo completo: partición LOO → selección en validación → test único → IC y pareados.

    Es exactamente el mismo flujo que usa el notebook con datos reales, de modo que los datos
    sintéticos pasan por el mismo procedimiento. Devuelve la configuración elegida, las
    métricas con IC por recomendador y las comparaciones pareadas del híbrido frente a
    popularidad global y azar.
    """
    if sim_contenido is None:
        sim_contenido = similitud_contenido(catalogo)
    part = particionar_loo(interacciones, seed)
    sel = seleccionar_configuracion(catalogo, part["train"], part["val"], part["elegibles"],
                                    sim_contenido, k, rejilla, seed)
    cfg = {"definicion": sel["definicion"], "alpha": sel["alpha"]}
    te = evaluar_test(catalogo, part, cfg, sim_contenido, k, seed,
                      pesos_popularidad=pesos_popularidad)
    res = te["resultados"]
    ic = ic_todos(res, n_boot, seed)
    h = res["Híbrido"]["metricas"]
    pareados = {
        base: comparar_pareado(h, res[base]["metricas"], n_boot, seed)
        for base in ("Popularidad global", "Aleatorio")
    }
    return {
        "config": cfg, "seleccion": sel, "n_usuarios": len(part["elegibles"]), "ic": ic,
        "pareados": pareados,
        "azar": azar_teorico(res["Híbrido"]["n_candidatos"], k),
    }


# ---------------------------------------------------------------------------------------------
# Agregación entre semillas (validación con señal sembrada)
# ---------------------------------------------------------------------------------------------
def resumen_semillas(valores) -> dict:
    """Media, desviación típica e IC 95 % (t de Student) de un valor medido en varias semillas.

    El IC es el de la media **entre semillas** (variabilidad de generación y partición), no el
    bootstrap sobre usuarios de una sola semilla. Con una única semilla, sd = 0 y el IC se
    reduce a la media.

    Returns:
        ``{"valor": media, "ic95": [inf, sup], "sd": desviación (ddof=1), "n_semillas": n,
        "valores": [...]}``; ``valor``/``ic95`` mantienen la estructura del resto del JSON.
    """
    v = np.asarray(valores, dtype=float)
    n = len(v)
    media = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        mitad = float(t_student.ppf(0.975, n - 1)) * sd / np.sqrt(n)
    else:
        mitad = 0.0
    return {"valor": media, "ic95": [media - mitad, media + mitad], "sd": sd,
            "n_semillas": int(n), "valores": [float(x) for x in v]}


def agregar_pipelines(resultados: list[dict]) -> dict:
    """Agrega varias ejecuciones de ``ejecutar_pipeline`` (una por semilla) en una sola.

    * ``ic[recomendador][metrica]``: ``resumen_semillas`` de la métrica puntual de cada semilla.
    * ``pareados[base][hit|ndcg]``: diferencia híbrido − base como media entre semillas (con su
      sd e IC entre semillas); el p-valor reportado es la **mediana** de los p de las semillas
      (``p_semillas`` guarda todos, ``p_max`` el peor).
    * ``azar``: media del azar teórico; ``configs``: configuración elegida en cada semilla.
    """
    nombres = list(resultados[0]["ic"])
    metricas = list(resultados[0]["ic"][nombres[0]])
    ic = {n: {m: resumen_semillas([r["ic"][n][m]["valor"] for r in resultados])
              for m in metricas} for n in nombres}
    pareados = {}
    for base in resultados[0]["pareados"]:
        pareados[base] = {}
        for met, clave_p in (("hit", "p_mcnemar"), ("ndcg", "p_wilcoxon")):
            res = resumen_semillas([r["pareados"][base][met]["diferencia"] for r in resultados])
            ps = [float(r["pareados"][base][met][clave_p]) for r in resultados]
            res.update({"diferencia": res["valor"], clave_p: float(np.median(ps)),
                        "p_semillas": ps, "p_max": float(max(ps))})
            pareados[base][met] = res
    azar = {m: float(np.mean([r["azar"][m] for r in resultados])) for m in resultados[0]["azar"]}
    return {"ic": ic, "pareados": pareados, "azar": azar,
            "n_usuarios": resultados[0]["n_usuarios"],
            "configs": [r["config"] for r in resultados], "config": resultados[0]["config"]}


def curva_validacion_alpha(resultados: list[dict]) -> dict:
    """Curva de validación por α (NDCG@K y HitRate@K) y p frente a α = 0,5, entre semillas.

    Para cada definición de positivo: media y sd entre semillas del NDCG/HitRate de validación
    en cada α, α máximo de cada semilla y su p de Wilcoxon frente a 0,5, y el α finalmente usado.
    """
    salida = {}
    for d in resultados[0]["seleccion"]["tablas"]:
        alphas = [f["alpha"] for f in resultados[0]["seleccion"]["tablas"][d]]
        ndcg = np.array([[f["ndcg"] for f in r["seleccion"]["tablas"][d]] for r in resultados])
        hit = np.array([[f["hit"] for f in r["seleccion"]["tablas"][d]] for r in resultados])
        salida[d] = {
            "alpha": alphas,
            "ndcg_media": ndcg.mean(axis=0).tolist(),
            "ndcg_sd": (ndcg.std(axis=0, ddof=1) if len(ndcg) > 1 else 0 * ndcg[0]).tolist(),
            "hit_media": hit.mean(axis=0).tolist(),
            "alpha_max_por_semilla": [r["seleccion"]["p_alpha"][d]["alpha_max"]
                                      for r in resultados],
            "p_vs_0_5_por_semilla": [r["seleccion"]["p_alpha"][d]["p_vs_0_5"]
                                     for r in resultados],
            "alpha_usado_por_semilla": [r["seleccion"]["alpha_por_definicion"][d]
                                        for r in resultados],
        }
    return salida
