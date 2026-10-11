"""Construcción de features para el reto 3 (abandono) y transformadores sin fuga.

Unidad de análisis: una interacción (usuario, título). Clase positiva: ``abandona`` =
``~Watched_to_End``. Contenido:

- ``construir_tabla``: une interacciones con metadatos (por título único) y crea features
  numéricas, géneros multi-hot, grupo de edad one-hot e interacciones edad x género.
- ``TasaAbandonoTitulo``: *target encoding* de la tasa de abandono por título, con
  suavizado hacia la media global y codificación **out-of-fold por usuario** dentro de
  ``fit_transform``. Vive dentro del ``Pipeline``: cada fold de la CV lo ajusta solo con su
  train.
- ``crear_pipeline``: ``ColumnTransformer`` (imputación + binarias + target encoding) + estimador.
- Features de "techo de señal" (T05b): ``genero_exacto`` (cadena completa de ``Genre``),
  ``n_generos``, ``genero_unico``, ``edad_genero_exacto`` (clave edad x género exacto) y
  ``ng_<edad>`` (``n_generos`` x grupo de edad). La tasa de abandono por clave se codifica con
  ``TasaAbandonoTitulo`` (generalizada a cualquier clave) dentro del Pipeline
  (``crear_pipeline_techo``). No se codifica a mano ninguna regla: el modelo recibe la
  representación (género exacto x edad) y aprende las tasas fuera de pliegue.
- ``particion_por_usuario``: train / validación / test donde ningún usuario aparece en dos.

Por qué NO se usa la tasa de abandono propia del usuario como feature: con la partición por
usuario, los usuarios de validación/test no existen en train, de modo que su historial no está
disponible en entrenamiento. Calcularla sobre las propias etiquetas del usuario (aunque fuera
leave-one-out) usaría información de la variable objetivo del mismo usuario que se quiere
predecir (fuga) y en producción solo existiría para usuarios con historial previo y marcas de
tiempo (aquí no hay marca de tiempo, por lo que ni siquiera se puede separar "pasado" de
"futuro").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
EDADES = ("youth", "adults", "seniors")
COLS_NUM = ["Runtime", "Released_Year", "IMDB_Rating", "No_of_Votes", "Meta_score"]
COL_TITULO = "Series_Title"
COL_USUARIO = "User_ID"
COL_EDAD = "Age_Group"
COL_Y = "abandona"
COL_GENERO_EXACTO = "genero_exacto"
COL_EDAD_GENERO = "edad_genero_exacto"
COLS_TECHO_NUM = ["n_generos", "genero_unico", *(f"ng_{e}" for e in EDADES)]


def lista_generos(metadatos: pd.DataFrame) -> list[str]:
    """Devuelve los géneros distintos de ``metadatos["Genre"]`` (separador ", "), ordenados."""
    return sorted({g for fila in metadatos["Genre"].dropna() for g in fila.split(", ") if g})


def columnas_binarias(generos: list[str]) -> list[str]:
    """Columnas binarias: géneros (``g_``), edad (``edad_``) y edad x género (``ag_``)."""
    cols = [f"g_{g}" for g in generos]
    cols += [f"edad_{e}" for e in EDADES]
    cols += [f"ag_{e}_{g}" for e in EDADES for g in generos]
    return cols


def construir_tabla(interacciones: pd.DataFrame, metadatos: pd.DataFrame) -> pd.DataFrame:
    """Une interacciones con metadatos y construye las features de abandono.

    Args:
        interacciones: ``User_ID``, ``Series_Title``, ``Watched_to_End`` (bool), ``Age_Group``.
        metadatos: salida de ``data.load_metadata(titulos_unicos=True)`` (un título por fila).

    Returns:
        DataFrame con ``User_ID``, ``Series_Title``, ``Age_Group``, ``Genre`` (texto, solo para
        auditoría), ``abandona`` (1 si NO vio hasta el final), las 5 numéricas de ``COLS_NUM``
        (``Meta_score`` con NaN sin imputar), los géneros multi-hot ``g_*``, la edad one-hot
        ``edad_*`` y las interacciones ``ag_<edad>_<género>`` (1 si el usuario es de ese grupo de
        edad Y la película tiene ese género). Además, para el análisis de techo de señal:
        ``genero_exacto`` (cadena completa de ``Genre``), ``edad_genero_exacto``
        (``"<edad>|<Genre>"``), ``n_generos``, ``genero_unico`` (1 si ``n_generos == 1``) y
        ``ng_<edad>`` (``n_generos`` si el usuario es de ese grupo, 0 si no).

    Raises:
        ValueError: si algún título de las interacciones no está en los metadatos o si los
            metadatos tienen títulos repetidos.
    """
    if metadatos["Series_Title"].duplicated().any():
        raise ValueError("metadatos con títulos repetidos: usa load_metadata(titulos_unicos=True)")
    generos = lista_generos(metadatos)
    meta = metadatos[[COL_TITULO, "Genre", *COLS_NUM]].copy()
    tabla = interacciones[[COL_USUARIO, COL_TITULO, COL_EDAD, "Watched_to_End"]].merge(
        meta, on=COL_TITULO, how="left", validate="many_to_one"
    )
    if tabla["Genre"].isna().any():
        raise ValueError("hay títulos de interacciones sin metadatos")

    tabla[COL_Y] = (~tabla["Watched_to_End"].astype(bool)).astype("int8")
    nuevas = {}
    gen_mat = {}
    for g in generos:
        col = tabla["Genre"].str.split(", ").map(lambda lst, g=g: g in lst).to_numpy(np.int8)
        gen_mat[g] = col
        nuevas[f"g_{g}"] = col
    for e in EDADES:
        nuevas[f"edad_{e}"] = (tabla[COL_EDAD] == e).to_numpy(np.int8)
    for e in EDADES:
        for g in generos:
            nuevas[f"ag_{e}_{g}"] = nuevas[f"edad_{e}"] * gen_mat[g]
    # Features de "techo de señal": cadena exacta de género y su interacción con la edad.
    n_gen = tabla["Genre"].str.split(", ").map(len).to_numpy(np.int16)
    nuevas["n_generos"] = n_gen
    nuevas["genero_unico"] = (n_gen == 1).astype(np.int8)
    for e in EDADES:
        nuevas[f"ng_{e}"] = nuevas[f"edad_{e}"].astype(np.int16) * n_gen
    nuevas[COL_GENERO_EXACTO] = tabla["Genre"].to_numpy()
    nuevas[COL_EDAD_GENERO] = (tabla[COL_EDAD] + "|" + tabla["Genre"]).to_numpy()
    tabla = pd.concat([tabla.drop(columns="Watched_to_End"), pd.DataFrame(nuevas)], axis=1)
    return tabla


def particion_por_usuario(
    tabla: pd.DataFrame,
    fracciones: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    """Particiona por USUARIO en train / validación / test (``GroupShuffleSplit`` encadenado).

    Args:
        tabla: tabla con ``User_ID``.
        fracciones: proporciones (train, validación, test) de usuarios; suman 1.
        seed: semilla.

    Returns:
        ``{"train": idx, "validacion": idx, "test": idx}`` con posiciones (``iloc``) de filas.
        Garantiza que ningún usuario aparece en dos particiones (``AssertionError`` si no).
    """
    f_tr, f_va, f_te = fracciones
    if not np.isclose(f_tr + f_va + f_te, 1.0):
        raise ValueError("las fracciones deben sumar 1")
    grupos = tabla[COL_USUARIO].to_numpy()
    gss1 = GroupShuffleSplit(n_splits=1, test_size=f_te, random_state=seed)
    resto, test = next(gss1.split(tabla, groups=grupos))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=f_va / (f_tr + f_va), random_state=seed)
    tr_rel, va_rel = next(gss2.split(resto, groups=grupos[resto]))
    out = {"train": resto[tr_rel], "validacion": resto[va_rel], "test": test}
    usuarios = {k: set(grupos[v]) for k, v in out.items()}
    assert not (usuarios["train"] & usuarios["validacion"]), "usuarios en train y validación"
    assert not (usuarios["train"] & usuarios["test"]), "usuarios en train y test"
    assert not (usuarios["validacion"] & usuarios["test"]), "usuarios en validación y test"
    return out


class TasaAbandonoTitulo(BaseEstimator, TransformerMixin):
    """Target encoding de la tasa de abandono por título, suavizado y sin autocontribución.

    Para cada título ``t``: ``(suma_t + m * prior) / (n_t + m)``, donde ``prior`` es la tasa
    global de abandono del conjunto de ajuste y ``m`` = ``suavizado`` (pseudo-observaciones).
    Títulos no vistos en ``fit`` reciben ``prior``.

    **Sesgo conocido y cómo se evita.** Si se ajusta con las filas de train y se aplica a esas
    mismas filas, cada fila contribuye a su propia codificación (su etiqueta entra en el
    promedio de su título): el modelo ve una feature que ya "contiene" la respuesta y aprende a
    confiar en ella, sin que generalice. Por eso ``fit_transform`` (lo que usa ``Pipeline`` en
    train) codifica **out-of-fold por usuario**: divide las filas en ``n_splits`` pliegues
    agrupados por usuario y codifica cada pliegue con estadísticas de los demás. Así la fila
    nunca ve su propia etiqueta ni la de su usuario. ``transform`` (validación/test) usa las
    estadísticas del ajuste completo.

    Recibe un DataFrame con dos columnas: título y usuario (grupo), en ese orden o por nombre.

    Args:
        suavizado: ``m``, fuerza del encogimiento hacia la media global.
        n_splits: pliegues agrupados de la codificación out-of-fold.
        col_titulo, col_grupo: nombres de las columnas del DataFrame de entrada. ``col_titulo``
            puede ser **cualquier clave categórica** (p. ej. ``edad_genero_exacto``): el
            transformador codifica la tasa de abandono por esa clave, con las mismas garantías.
        nombre_salida: nombre de la columna de salida (distinto por cada clave codificada).
    """

    def __init__(
        self,
        suavizado: float = 20.0,
        n_splits: int = 5,
        col_titulo: str = COL_TITULO,
        col_grupo: str = COL_USUARIO,
        nombre_salida: str = "tasa_abandono_titulo",
    ):
        self.suavizado = suavizado
        self.n_splits = n_splits
        self.col_titulo = col_titulo
        self.col_grupo = col_grupo
        self.nombre_salida = nombre_salida

    @staticmethod
    def _estadisticas(titulos: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, float]:
        df = pd.DataFrame({"t": titulos, "y": y})
        agg = df.groupby("t")["y"].agg(["sum", "count"])
        return agg, float(y.mean())

    def _codificar(self, titulos: np.ndarray, agg: pd.DataFrame, prior: float) -> np.ndarray:
        suma = pd.Series(titulos).map(agg["sum"]).to_numpy(dtype=float)
        n = pd.Series(titulos).map(agg["count"]).to_numpy(dtype=float)
        suma = np.nan_to_num(suma, nan=0.0)
        n = np.nan_to_num(n, nan=0.0)
        return (suma + self.suavizado * prior) / (n + self.suavizado)

    def _extraer(self, X) -> tuple[np.ndarray, np.ndarray | None]:
        if isinstance(X, pd.DataFrame):
            grupos = X[self.col_grupo].to_numpy() if self.col_grupo in X else None
            return X[self.col_titulo].to_numpy(), grupos
        X = np.asarray(X, dtype=object)
        return X[:, 0], (X[:, 1] if X.shape[1] > 1 else None)

    def fit(self, X, y):
        """Ajusta las estadísticas por título con TODAS las filas recibidas."""
        titulos, _ = self._extraer(X)
        self.agg_, self.prior_ = self._estadisticas(titulos, np.asarray(y, dtype=float))
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Ajusta y devuelve la codificación **out-of-fold por usuario** (sin autocontribución)."""
        titulos, grupos = self._extraer(X)
        y = np.asarray(y, dtype=float)
        self.fit(X, y)
        if grupos is None:
            raise ValueError("fit_transform necesita la columna de usuario para el out-of-fold")
        salida = np.empty(len(y))
        gkf = GroupKFold(n_splits=min(self.n_splits, len(np.unique(grupos))))
        for tr, te in gkf.split(titulos, y, groups=grupos):
            agg, prior = self._estadisticas(titulos[tr], y[tr])
            salida[te] = self._codificar(titulos[te], agg, prior)
        return salida.reshape(-1, 1)

    def transform(self, X):
        """Codifica con las estadísticas del ajuste; títulos no vistos reciben la media global."""
        titulos, _ = self._extraer(X)
        return self._codificar(titulos, self.agg_, self.prior_).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        """Nombre de la única columna de salida."""
        return np.array([self.nombre_salida], dtype=object)


def crear_pipeline(
    estimador,
    generos: list[str],
    escalar: bool = False,
    usar_te: bool = True,
    suavizado: float = 20.0,
) -> Pipeline:
    """Pipeline sin fuga: imputación (mediana) + binarias + target encoding + estimador.

    Todo lo que aprende de los datos (mediana de ``Meta_score``, tasas por título, escalado)
    se ajusta solo con los datos que recibe ``fit``. ``User_ID`` solo lo ve el target encoding
    (para el out-of-fold) y no llega al estimador.

    Args:
        estimador: clasificador sklearn (paso final ``clf``).
        generos: lista de géneros (de ``lista_generos``) para fijar las columnas binarias.
        escalar: añade ``StandardScaler`` tras el ``ColumnTransformer`` (para modelos lineales).
        usar_te: incluye la tasa de abandono por título.
        suavizado: parámetro ``m`` del target encoding.
    """
    pasos = [
        ("num", SimpleImputer(strategy="median"), COLS_NUM),
        ("bin", "passthrough", columnas_binarias(generos)),
    ]
    if usar_te:
        pasos.append(("te", TasaAbandonoTitulo(suavizado=suavizado), [COL_TITULO, COL_USUARIO]))
    ct = ColumnTransformer(pasos, remainder="drop", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    etapas = [("prep", ct)]
    if escalar:
        etapas.append(("esc", StandardScaler()))
    etapas.append(("clf", estimador))
    return Pipeline(etapas)


def crear_pipeline_techo(
    estimador, generos: list[str], suavizado: float = 20.0, usar_te_titulo: bool = True
) -> Pipeline:
    """Pipeline sin fuga con las features de género exacto (análisis de techo de señal).

    Añade a ``crear_pipeline``: tasa de abandono por ``edad_genero_exacto`` y por
    ``genero_exacto`` (``TasaAbandonoTitulo`` con esas claves: suavizada y **out-of-fold por
    usuario** en ``fit_transform``, categorías no vistas -> media global) y, sin transformar,
    ``n_generos``, ``genero_unico`` y ``ng_<edad>`` (n_generos x edad). Todo se ajusta solo con
    los datos de ``fit``. No codifica ninguna regla a mano.

    Args:
        estimador: clasificador sklearn (paso final ``clf``).
        generos: lista de géneros (de ``lista_generos``).
        suavizado: ``m`` del target encoding.
        usar_te_titulo: mantiene la tasa por título de T05.
    """
    pasos = [
        ("num", SimpleImputer(strategy="median"), COLS_NUM),
        ("bin", "passthrough", columnas_binarias(generos)),
        ("ng", "passthrough", COLS_TECHO_NUM),
        ("te_par", TasaAbandonoTitulo(suavizado=suavizado, col_titulo=COL_EDAD_GENERO,
                                      nombre_salida="tasa_edad_genero_exacto"),
         [COL_EDAD_GENERO, COL_USUARIO]),
        ("te_gen", TasaAbandonoTitulo(suavizado=suavizado, col_titulo=COL_GENERO_EXACTO,
                                      nombre_salida="tasa_genero_exacto"),
         [COL_GENERO_EXACTO, COL_USUARIO]),
    ]
    if usar_te_titulo:
        pasos.append(("te", TasaAbandonoTitulo(suavizado=suavizado), [COL_TITULO, COL_USUARIO]))
    ct = ColumnTransformer(pasos, remainder="drop", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return Pipeline([("prep", ct), ("clf", estimador)])


def columnas_entrada_techo(generos: list[str]) -> list[str]:
    """Columnas que consume ``crear_pipeline_techo`` (las de T05 más las de género exacto)."""
    return [*columnas_entrada(generos), COL_GENERO_EXACTO, COL_EDAD_GENERO, *COLS_TECHO_NUM]


def columnas_entrada(generos: list[str]) -> list[str]:
    """Columnas de la tabla que consume ``crear_pipeline`` (incluye título y usuario)."""
    return [COL_USUARIO, COL_TITULO, *COLS_NUM, *columnas_binarias(generos)]
