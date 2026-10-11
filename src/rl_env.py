"""Entorno de simulación de Nile Prime y agentes de aprendizaje por refuerzo (reto 4).

Contenido
---------
* ``cargar_entorno_original()``: importa ``agent.py`` (del profesor, que NO se modifica) con una
  lectura de CSV robusta, sin depender del directorio actual ni de las tildes NFD.
* ``NilePrimeEnv``: hereda de ``SimulationEnvironment`` y SOLO corrige los bugs reales.
* ``matriz_probabilidades`` / ``politica_optima``: probabilidad de ver completa cada película
  por grupo de edad (``P[estado, acción]``) y política óptima analítica.
* ``entrenar_bandit``: Q-learning tabular (y UCB de referencia) vectorizado sobre semillas.
* Utilidades: regret, tiempos de descubrimiento, intervalos de confianza entre semillas.

Cambios respecto a ``agent.py`` (cada uno documentado)
------------------------------------------------------
=====  ==============================================  =====================================
 N.º   Cambio                                          Motivo
=====  ==============================================  =====================================
 1     ``_get_state()`` implementado (índice del       ``step()`` lo llama y no existía:
       grupo de edad del usuario actual)               ``AttributeError`` (bug real).
 2     ``reset()`` devuelve el estado inicial          Devolvía ``None`` (bug real); un
                                                       entorno estilo Gym debe devolverlo.
 3     ``reset()`` pone ``done = False``               ``done`` quedaba en ``True`` tras el
                                                       primer ``step`` (consistencia).
 4     Generador propio ``np.random.RandomState``      Reproducibilidad por semilla. Es el
       en vez del estado global de ``np.random``       mismo generador que ``np.random.rand``,
                                                       así que con la misma semilla produce
                                                       EXACTAMENTE las mismas decisiones
                                                       (hay un test).
 5     ``modo`` ("original" por defecto)               "original" = regla del profesor
                                                       (comparación EXACTA de género); los
                                                       demás son análisis de sensibilidad.
=====  ==============================================  =====================================

La recompensa (1 si el usuario ve la película completa, 0 si no) y la probabilidad de abandono
de ``modo="original"`` son EXACTAMENTE las del profesor, incluida la comparación exacta de
``Genre``: no es un bug, es la regla con la que se generaron los datos de ``interacciones.csv``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _st

from src import data as _data

SEED = 42
GRUPOS = ("youth", "adults", "seniors")
MODOS = ("original", "contiene", "calibrado")

# Diccionario del profesor: probabilidad de NO ver hasta el final, por grupo y género EXACTO.
SEGMENTOS_PROFESOR = {
    "youth": {"Horror": 0.3, "Drama": 0.3},
    "adults": {"Action": 0.2, "Sci-Fi": 0.2},
    "seniors": {"Comedy": 0.25, "Romance": 0.25},
}
PROB_DEFECTO_PROFESOR = 0.8  # grupo de edad desconocido (nunca ocurre en los datos)
SHA256_AGENT = "ff102044ff4126604f44cfe03ea11eeb9b91f5768d4477add8847b24ec3fb376"

# Modo "calibrado": abandono observado en los datos (notebook 00/03): ~9,9 % de base y
# ~20,6 % (youth) / ~21,1 % (seniors) en el segmento. Adults no tiene película que cumpla la
# regla exacta; su valor de segmento es irrelevante y se fija en el de los otros grupos.
CALIBRADO_DEFECTO = {
    "base": 0.099,
    "segmento": {"youth": 0.206, "adults": 0.21, "seniors": 0.211},
}


# ---------------------------------------------------------------------------------------
# Importación robusta de agent.py
# ---------------------------------------------------------------------------------------
def sha256_agent() -> str:
    """Huella sha256 de ``agent.py`` (para comprobar que no se ha modificado)."""
    return hashlib.sha256((_data.project_root() / "agent.py").read_bytes()).hexdigest()


def cargar_entorno_original():
    """Importa ``agent.py`` y devuelve el módulo (con ``SimulationEnvironment``).

    ``agent.py`` lee ``./Metadatos Películas.csv`` y ``./interacciones.csv`` del directorio
    actual al importarse. Esos nombres están en ``data/raw`` con tildes en Unicode NFD, de modo
    que ni el directorio actual ni un ``chdir`` bastan de forma portable. Solución sin tocar el
    archivo: se carga por ruta (``importlib``) y, solo durante la importación, se sustituye
    ``pandas.read_csv`` por una función que resuelve esos dos nombres con ``glob`` en
    ``data/raw`` (``src.data.find_csv``). ``pandas.read_csv`` se restaura siempre en un
    ``finally``; no se cambia el directorio de trabajo. El módulo queda registrado como
    ``sys.modules["agent"]``, así que la segunda llamada es instantánea.
    """
    modulo = sys.modules.get("agent")
    if modulo is not None and hasattr(modulo, "SimulationEnvironment"):
        return modulo
    ruta = _data.project_root() / "agent.py"
    original = pd.read_csv

    def _read_csv(ruta_csv, *args, **kwargs):
        if isinstance(ruta_csv, str | os.PathLike) and not Path(ruta_csv).exists():
            nombre = unicodedata.normalize("NFD", Path(str(ruta_csv)).stem)
            patron = nombre.split(" ")[0] + "*.csv"
            ruta_csv = _data.find_csv(patron)
        return original(ruta_csv, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("agent", ruta)
    modulo = importlib.util.module_from_spec(spec)
    pd.read_csv = _read_csv
    try:
        sys.modules["agent"] = modulo
        spec.loader.exec_module(modulo)
    except BaseException:
        sys.modules.pop("agent", None)
        raise
    finally:
        pd.read_csv = original
    return modulo


_agent = cargar_entorno_original()
SimulationEnvironment = _agent.SimulationEnvironment


# ---------------------------------------------------------------------------------------
# Reglas de recompensa y matriz de probabilidades
# ---------------------------------------------------------------------------------------
def prob_ver(modo: str, grupo: str, genero: str, calib: dict | None = None) -> float:
    """Probabilidad de que el usuario vea la película completa.

    * ``original``: lógica del profesor tal cual (género EXACTO; grupo desconocido → 0,8).
    * ``contiene``: penaliza si el género de la película CONTIENE alguno de los géneros del
      segmento (``Genre`` se separa por ", "); misma penalización que el profesor. Con
      varias coincidencias se aplica la mayor (no se acumulan).
    * ``calibrado``: coincidencia exacta como el profesor, pero con las tasas de abandono
      observadas en los datos (``CALIBRADO_DEFECTO``): base ≈ 9,9 % y segmento ≈ 21 %.
    """
    if modo not in MODOS:
        raise ValueError(f"modo debe ser uno de {MODOS}")
    if grupo not in SEGMENTOS_PROFESOR:
        return PROB_DEFECTO_PROFESOR
    segmento = SEGMENTOS_PROFESOR[grupo]
    if modo == "original":
        return 1.0 - segmento.get(genero, 0)
    if modo == "contiene":
        tokens = set(genero.split(", "))
        return 1.0 - max([p for g, p in segmento.items() if g in tokens], default=0.0)
    calib = calib or CALIBRADO_DEFECTO
    abandono = calib["segmento"][grupo] if genero in segmento else calib["base"]
    return 1.0 - abandono


def matriz_probabilidades(modo: str, generos, calib: dict | None = None) -> np.ndarray:
    """Matriz ``P[estado, acción]`` de probabilidad de ver la película completa.

    Args:
        modo: ``original``, ``contiene`` o ``calibrado``.
        generos: columna ``Genre`` del catálogo, en el orden del entorno (acción = posición).
        calib: tasas de abandono del modo ``calibrado`` (por defecto ``CALIBRADO_DEFECTO``).

    Returns:
        Array (3, n_películas) con filas en el orden de ``GRUPOS``.
    """
    generos = list(generos)
    cache: dict[tuple[str, str], float] = {}
    P = np.empty((len(GRUPOS), len(generos)))
    for i, g in enumerate(GRUPOS):
        for j, gen in enumerate(generos):
            clave = (g, gen)
            if clave not in cache:
                cache[clave] = prob_ver(modo, g, gen, calib)
            P[i, j] = cache[clave]
    return P


def politica_optima(P: np.ndarray, pesos_estado: np.ndarray | None = None, tol: float = 1e-12):
    """Política óptima analítica de un bandit contextual con ``P[estado, acción]`` conocida.

    Con episodios de 1 paso, el valor óptimo de cada estado es ``max_a P[s, a]``: las acciones
    óptimas son las que alcanzan ese máximo y las "a evitar" son todas las que lo superan
    por más de ``tol`` (aquí: el segmento de riesgo del grupo de edad).

    Returns:
        Diccionario con ``valor_optimo`` y ``valor_azar`` por estado, sus medias ponderadas por
        ``pesos_estado`` (``valor_optimo_medio``, ``valor_azar_medio``), ``brecha``, y por
        estado ``optimas`` (índices) y ``evitar`` (índices), más la máscara ``riesgo`` (3, A).
    """
    n_estados = P.shape[0]
    w = np.full(n_estados, 1.0 / n_estados) if pesos_estado is None else np.asarray(pesos_estado)
    vmax = P.max(axis=1)
    riesgo = P < (vmax[:, None] - tol)
    out = {
        "valor_optimo": vmax,
        "valor_azar": P.mean(axis=1),
        "valor_optimo_medio": float(w @ vmax),
        "valor_azar_medio": float(w @ P.mean(axis=1)),
        "optimas": [np.flatnonzero(~riesgo[s]) for s in range(n_estados)],
        "evitar": [np.flatnonzero(riesgo[s]) for s in range(n_estados)],
        "riesgo": riesgo,
        "pesos_estado": w,
    }
    out["brecha"] = out["valor_optimo_medio"] - out["valor_azar_medio"]
    return out


def agrupar_por_genero(generos) -> tuple[np.ndarray, np.ndarray]:
    """Agrupa las películas por cadena EXACTA de ``Genre``.

    Returns:
        ``(ids, unicos)``: ``ids[j]`` es el grupo de la película ``j`` y ``unicos`` los
        valores de ``Genre`` (orden de primera aparición).
    """
    unicos, primeros, ids = np.unique(np.asarray(list(generos)), return_index=True,
                                      return_inverse=True)
    orden = np.argsort(primeros)
    rango = np.empty_like(orden)
    rango[orden] = np.arange(len(orden))
    return rango[ids], unicos[orden]


def matriz_por_genero(P: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Reduce ``P`` (3, películas) a (3, grupos de género); exige que ``P`` sea constante
    dentro de cada grupo (cierto en los tres modos: la regla solo depende de ``Genre``)."""
    n = int(ids.max()) + 1
    primero = np.array([np.flatnonzero(ids == k)[0] for k in range(n)])
    Pg = P[:, primero]
    if not np.allclose(P, Pg[:, ids]):
        raise ValueError("P no es constante dentro de los grupos de género.")
    return Pg


# ---------------------------------------------------------------------------------------
# Entorno (hereda del del profesor)
# ---------------------------------------------------------------------------------------
class NilePrimeEnv(SimulationEnvironment):
    """``SimulationEnvironment`` con los bugs reales corregidos y modos de recompensa.

    Ver la tabla de cambios en la docstring del módulo. Interfaz tipo bandit contextual::

        env = NilePrimeEnv(peliculas, interacciones, modo="original", seed=0)
        estado = env.reset()                    # índice del grupo de edad (0, 1, 2)
        estado, recompensa, done = env.step(a)  # a = posición de la película

    Args:
        movies_df: catálogo en el orden original (la acción es la posición de la fila).
        user_interactions_df: interacciones; ``reset`` muestrea una fila (como el profesor).
        modo: ``original`` (por defecto), ``contiene`` o ``calibrado``.
        seed: semilla del generador propio (``np.random.RandomState``).
        calib: tasas de abandono del modo ``calibrado``.
    """

    def __init__(self, movies_df, user_interactions_df, modo: str = "original",
                 seed: int | None = None, calib: dict | None = None):
        if modo not in MODOS:
            raise ValueError(f"modo debe ser uno de {MODOS}")
        super().__init__(movies_df, user_interactions_df)
        self.modo = modo
        self.calib = calib
        self._rng = np.random.RandomState(seed)

    def _get_state(self) -> int:
        """Estado = índice (en ``GRUPOS``) del grupo de edad del usuario actual (cambio 1)."""
        return GRUPOS.index(self.current_user["Age_Group"])

    def reset(self) -> int:
        """Muestrea un usuario (fila de interacciones) y devuelve su estado (cambios 2-4)."""
        self.current_user = self.user_interactions_df.sample(1, random_state=self._rng).iloc[0]
        self.done = False
        return self._get_state()

    def _simulate_user_watch(self, recommended_movie) -> bool:
        """Misma decisión que el profesor (``rand() < prob``) con generador propio (cambio 4)."""
        p = prob_ver(self.modo, self.current_user["Age_Group"], recommended_movie["Genre"],
                     self.calib)
        return self._rng.rand() < p


# ---------------------------------------------------------------------------------------
# Programas de ε
# ---------------------------------------------------------------------------------------
def programa_epsilon(tipo: str, n_pasos: int, eps0: float = 0.3, eps_min: float = 0.01,
                     frac: float = 0.5) -> np.ndarray:
    """Programa de exploración ``ε_t`` (array de longitud ``n_pasos``).

    * ``constante``: ``ε_t = eps0``.
    * ``exponencial``: decae geométricamente de ``eps0`` a ``eps_min`` al cabo de ``frac·T``
      pasos y se queda en ``eps_min``.
    * ``inverso``: ``ε_t = eps0 / (1 + t/τ)``, con ``τ`` tal que llega a ``eps_min`` en ``frac·T``.
    """
    t = np.arange(n_pasos, dtype=float)
    if tipo == "constante":
        return np.full(n_pasos, eps0)
    if tipo == "exponencial":
        return np.maximum(eps_min, eps0 * (eps_min / eps0) ** (t / (frac * n_pasos)))
    if tipo == "inverso":
        tau = frac * n_pasos / (eps0 / eps_min - 1.0)
        return np.maximum(eps_min, eps0 / (1.0 + t / tau))
    raise ValueError("tipo debe ser constante, exponencial o inverso")


# ---------------------------------------------------------------------------------------
# Bandit contextual vectorizado sobre semillas
# ---------------------------------------------------------------------------------------
def _ruido(sal_t: np.ndarray, brazos: np.ndarray) -> np.ndarray:
    """Ruido de desempate U(0, 1e-9) (S, A): hash (finalizador de murmur3) de brazo ⊕ sal."""
    x = brazos[None, :] ^ sal_t[:, None]
    x = x * np.uint32(0x9E3779B1)
    x ^= x >> np.uint32(16)
    x = x * np.uint32(0x85EBCA6B)
    x ^= x >> np.uint32(13)
    x = x * np.uint32(0xC2B2AE35)
    x ^= x >> np.uint32(16)
    return x * (1e-9 / 2.0**32)


def pasos_checkpoint(n_pasos: int, n: int = 160) -> np.ndarray:
    """Pasos (1-indexados) donde se observa la política: malla geométrica de 10 a ``n_pasos``."""
    g = np.unique(np.round(np.geomspace(10, n_pasos, n)).astype(int))
    return g


def entrenar_bandit(
    P: np.ndarray,
    grupo_fila: np.ndarray,
    semillas,
    n_pasos: int,
    epsilon: np.ndarray,
    alpha: float | None = 0.1,
    q0: float = 1.0,
    metodo: str = "egreedy",
    c_ucb: float = 1.0,
    topk_fila: np.ndarray | None = None,
    checkpoints: np.ndarray | None = None,
    umbral_ident: float = 0.9,
) -> dict:
    """Q-learning tabular (o UCB) en un bandit contextual de episodios de 1 paso.

    Equivale a entrenar con el entorno ``NilePrimeEnv`` (la recompensa es Bernoulli con
    probabilidad ``P[s, a]``), pero vectorizado sobre las semillas. Cada semilla usa su propio
    ``Generator`` (estados, exploración y recompensas), así que el resultado de una semilla no
    depende de qué otras se ejecuten con ella.

    Como cada episodio dura 1 paso no hay estado siguiente ni descuento: la actualización es
    ``Q[s,a] += α·(r − Q[s,a])``. Con ``alpha=None`` se usa ``α = 1/N(s,a)`` (media muestral).
    En ``egreedy``, los empates de la acción voraz se rompen con una prioridad aleatoria fija
    por (semilla, estado, acción).

    Args:
        P: (3, A) probabilidad de recompensa 1 por estado y acción (brazo).
        grupo_fila: estado de cada fila de interacciones; cada paso muestrea una fila (como
            ``reset`` del entorno), lo que fija la distribución de estados.
        semillas: lista de semillas.
        n_pasos: número de episodios (interacciones).
        epsilon: programa ``ε_t`` (longitud ``n_pasos``).
        alpha: tasa de aprendizaje constante, o ``None`` para 1/N(s,a).
        q0: valor inicial de Q (1,0 = optimista: cota superior de la recompensa).
        metodo: ``egreedy`` o ``ucb`` (UCB1 con bonus ``c·sqrt(ln N_s / n_sa)``).
        topk_fila: (filas, K) candidatos por fila (re-ranking): el agente solo elige entre los
            K candidatos de la fila muestreada. Con ``None`` elige entre todas las acciones.
        checkpoints: pasos donde registrar la política voraz (solo sin ``topk_fila``).
        umbral_ident: no se usa en el entrenamiento; se guarda por trazabilidad.

    Returns:
        Diccionario con ``esperada`` (S, T) = ``P[s, a]`` de la acción elegida, ``real`` (S, T)
        recompensa observada, ``estados`` (S, T), ``Q`` y ``N`` finales (S, 3, A) y, si hay
        ``checkpoints``: ``pasos``, ``n_grupo`` (S, C, 3) interacciones acumuladas por grupo,
        ``voraz_riesgo`` (S, C, 3), ``ident`` (S, C, 3) fracción de acciones de riesgo YA PROBADAS
        (N ≥ 1) con Q por debajo de la media de Q de las no penalizadas ya probadas, y
        ``valor_voraz`` (S, C).
    """
    semillas = list(semillas)
    S, (n_est, A) = len(semillas), P.shape
    if len(epsilon) != n_pasos:
        raise ValueError("epsilon debe tener longitud n_pasos")
    n_filas = len(grupo_fila)
    K = None if topk_fila is None else topk_fila.shape[1]
    amplitud = A if K is None else K

    filas = np.empty((S, n_pasos), dtype=np.int64)
    u_ex = np.empty((S, n_pasos))
    a_rand = np.empty((S, n_pasos), dtype=np.int64)
    u_r = np.empty((S, n_pasos))
    sal = np.empty((S, n_pasos), dtype=np.uint32)
    tie = np.empty((S, n_est, A))
    for i, sem in enumerate(semillas):
        rng = np.random.default_rng(sem)
        filas[i] = rng.integers(0, n_filas, n_pasos)
        u_ex[i] = rng.random(n_pasos)
        a_rand[i] = rng.integers(0, amplitud, n_pasos)
        u_r[i] = rng.random(n_pasos)
        sal[i] = rng.integers(0, 2**32, n_pasos, dtype=np.uint32)
        tie[i] = rng.random((n_est, A)) * 1e-9
    estados = grupo_fila[filas]

    Q = np.full((S, n_est, A), float(q0))
    N = np.zeros((S, n_est, A), dtype=np.int64)
    ar = np.arange(S)
    brazos = np.arange(A, dtype=np.uint32)
    esperada = np.empty((S, n_pasos), dtype=np.float32)
    real = np.empty((S, n_pasos), dtype=np.uint8)

    vmax = P.max(axis=1)
    riesgo = P < (vmax[:, None] - 1e-12)
    seguro = ~riesgo
    n_riesgo = riesgo.sum(axis=1)
    chk = np.zeros(n_pasos + 1, dtype=bool)
    if checkpoints is not None and K is None:
        chk[np.asarray(checkpoints)] = True
    cum = np.stack([np.cumsum(estados == g, axis=1) for g in range(n_est)], axis=-1)
    reg = {"pasos": [], "n_grupo": [], "voraz_riesgo": [], "ident": [], "valor_voraz": []}
    w_est = np.bincount(grupo_fila, minlength=n_est) / n_filas

    for t in range(n_pasos):
        s = estados[:, t]
        q = Q[ar, s]
        ruido = _ruido(sal[:, t], brazos)
        if metodo == "ucb":
            n_sa = N[ar, s]
            n_s = n_sa.sum(axis=1, keepdims=True) + 1.0
            with np.errstate(divide="ignore", invalid="ignore"):
                bonus = c_ucb * np.sqrt(np.log(n_s) / n_sa)
            puntaje = np.where(n_sa == 0, np.inf, q + bonus) + ruido
            a = puntaje.argmax(axis=1)
        elif K is None:
            voraz = (q + ruido).argmax(axis=1)
            a = np.where(u_ex[:, t] < epsilon[t], a_rand[:, t], voraz)
        else:
            cand = topk_fila[filas[:, t]]
            sc = np.take_along_axis(q, cand, 1) + np.take_along_axis(ruido, cand, 1)
            j = np.where(u_ex[:, t] < epsilon[t], a_rand[:, t], sc.argmax(axis=1))
            a = cand[ar, j]
        p = P[s, a]
        r = u_r[:, t] < p
        n = N[ar, s, a] + 1
        N[ar, s, a] = n
        paso = (1.0 / n) if alpha is None else alpha
        Q[ar, s, a] += paso * (r - Q[ar, s, a])
        esperada[:, t] = p
        real[:, t] = r
        if chk[t + 1]:
            voraz = (Q + tie).argmax(axis=2)  # (S, 3)
            pv = np.take_along_axis(np.broadcast_to(P, (S, n_est, A)), voraz[..., None], 2)[..., 0]
            reg["pasos"].append(t + 1)
            reg["n_grupo"].append(cum[:, t, :])
            reg["voraz_riesgo"].append(pv < (vmax[None, :] - 1e-12))
            vis = N > 0
            seg_vis = seguro[None] & vis
            n_sv = seg_vis.sum(axis=2)
            q_seg = np.where(seg_vis, Q, 0.0).sum(axis=2) / np.maximum(n_sv, 1)
            menor = (Q < q_seg[..., None]) & riesgo[None] & vis & (n_sv[..., None] > 0)
            reg["ident"].append(menor.sum(axis=2) / np.maximum(n_riesgo, 1)[None])
            reg["valor_voraz"].append(pv @ w_est)

    out = {"esperada": esperada, "real": real, "estados": estados, "filas": filas, "Q": Q, "N": N,
           "semillas": semillas, "umbral_ident": umbral_ident}
    if reg["pasos"]:
        out["pasos"] = np.array(reg["pasos"])
        out["n_grupo"] = np.stack(reg["n_grupo"], axis=1)
        out["voraz_riesgo"] = np.stack(reg["voraz_riesgo"], axis=1)
        out["ident"] = np.stack(reg["ident"], axis=1)
        out["valor_voraz"] = np.stack(reg["valor_voraz"], axis=1)
    return out


# ---------------------------------------------------------------------------------------
# Métricas: regret, descubrimiento, convergencia, intervalos
# ---------------------------------------------------------------------------------------
def regret_por_paso(esperada: np.ndarray, estados: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Pseudo-regret instantáneo ``max_a P[s,a] − P[s,a_t]`` (S, T) usando ``P`` conocida."""
    return P.max(axis=1)[estados] - esperada


def media_movil(x: np.ndarray, ventana: int) -> np.ndarray:
    """Media móvil causal por filas (S, T) → (S, T - ventana + 1)."""
    c = np.cumsum(np.pad(x, ((0, 0), (1, 0))), axis=1, dtype=np.float64)
    return (c[:, ventana:] - c[:, :-ventana]) / ventana


def tiempo_convergencia(regret: np.ndarray, regret_azar: float, fraccion: float = 0.10,
                        ventana: int = 2000) -> np.ndarray:
    """Primer paso en que la media móvil del regret baja de ``fraccion × regret_azar``.

    Criterio de convergencia: el agente ha recuperado ``1 − fraccion`` (90 % por defecto) de la
    brecha entre la política aleatoria y la óptima, medido con ventana de ``ventana`` pasos.
    Devuelve (S,) con NaN si no se alcanza.
    """
    mm = media_movil(regret, ventana)
    ok = mm <= fraccion * regret_azar
    primero = ok.argmax(axis=1)
    return np.where(ok.any(axis=1), primero + ventana, np.nan).astype(float)


def tiempos_descubrimiento(res: dict, umbral: float = 0.9) -> dict:
    """Interacciones DEL GRUPO hasta (a) evitar el riesgo y (b) identificarlo, por semilla.

    (a) ``evitar``: primer checkpoint tras el cual la política voraz no recomienda ninguna
    película de riesgo en los checkpoints posteriores (estable). NaN si al final aún lo hace.
    (b) ``identificar``: primer checkpoint con ≥ ``umbral`` de las películas de riesgo probadas
    y con Q por debajo de la media de Q de las no penalizadas probadas (``ident``). NaN si no
    se alcanza.
    Devuelve arrays (S, 3) con NaN cuando no aplica o no se alcanza.
    """
    gr, ident, ng = res["voraz_riesgo"], res["ident"], res["n_grupo"]
    S, C, G = gr.shape
    inv = gr[:, ::-1, :]
    ultimo = C - 1 - inv.argmax(axis=1)  # último checkpoint con riesgo
    hay = gr.any(axis=1)
    idx_a = np.where(hay, ultimo + 1, 0)
    ev_a = np.where(hay & (idx_a >= C), np.nan,
                    np.take_along_axis(ng, np.minimum(idx_a, C - 1)[:, None, :], 1)[:, 0, :])
    alc = ident >= umbral
    primero = alc.argmax(axis=1)
    ev_b = np.where(alc.any(axis=1),
                    np.take_along_axis(ng, primero[:, None, :], 1)[:, 0, :], np.nan)
    alc50 = ident >= 0.5
    p50 = alc50.argmax(axis=1)
    ev_50 = np.where(alc50.any(axis=1),
                     np.take_along_axis(ng, p50[:, None, :], 1)[:, 0, :], np.nan)
    return {"evitar": ev_a.astype(float), "identificar": ev_b.astype(float),
            "identificar_50": ev_50.astype(float)}


def ic_t(valores, nivel: float = 0.95) -> dict:
    """Media e IC (t de Student) entre semillas: ``{"valor", "ic95", "sd", "n"}``."""
    v = np.asarray(valores, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    m = float(v.mean()) if n else float("nan")
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        h = float(_st.t.ppf(0.5 + nivel / 2, n - 1) * sd / np.sqrt(n))
    else:
        h = 0.0
    return {"valor": m, "ic95": [m - h, m + h], "sd": sd, "n": int(n)}


# ---------------------------------------------------------------------------------------
# Recomendador híbrido como política "sin RL"
# ---------------------------------------------------------------------------------------
def topk_recomendador(catalogo_unico: pd.DataFrame, interacciones: pd.DataFrame, alpha: float,
                      positivo: str, k: int = 10, lote: int = 1000, sim_contenido=None):
    """Top-``k`` del recomendador híbrido para cada usuario, como ``movie_id`` del catálogo.

    Returns:
        ``(usuarios, topk)``: ``usuarios`` ordenados y ``topk`` (n_usuarios, k) con índices de
        película del catálogo original (la acción del entorno). Excluye lo ya visto.
    """
    from src import recommender as rec

    modelo = rec.HybridRecommender(alpha=alpha, positivo=positivo).fit(
        catalogo_unico, interacciones, sim_contenido=sim_contenido)
    usuarios = modelo.usuarios_
    grupo = interacciones.drop_duplicates("User_ID").set_index("User_ID")["Age_Group"]
    mid = catalogo_unico["movie_id"].to_numpy()
    tops = []
    for i in range(0, len(usuarios), lote):
        u = usuarios[i:i + lote]
        scores, vistos = modelo.puntuar_lote(u, grupo.loc[u].to_numpy())
        orden = rec.ordenar_candidatos(scores, vistos, k=k, seed=SEED)
        tops.append(mid[orden])
    return usuarios, np.vstack(tops)


def auc_separacion(Q: np.ndarray, N: np.ndarray, riesgo: np.ndarray) -> np.ndarray:
    """AUC con que ``Q`` final separa las películas de riesgo de las no penalizadas.

    Es ``P(Q_riesgo < Q_seguro)`` (empates cuentan 0,5) entre las acciones ya probadas
    (``N ≥ 1``) de cada estado: 0,5 = ``Q`` no distingue; 1 = separa perfectamente. Es una
    medida de cuánto *sabe* el agente del segmento, a diferencia del regret (que sólo mide lo
    que recomienda). Devuelve (S, estados) con NaN si algún grupo no tiene probadas.
    """
    S, G, _ = Q.shape
    out = np.full((S, G), np.nan)
    for i in range(S):
        for g in range(G):
            vis = N[i, g] > 0
            r, s = Q[i, g, vis & riesgo[g]], Q[i, g, vis & ~riesgo[g]]
            if len(r) and len(s):
                rangos = _st.rankdata(np.concatenate([r, s]))
                u = rangos[: len(r)].sum() - len(r) * (len(r) + 1) / 2
                out[i, g] = 1.0 - u / (len(r) * len(s))
    return out
