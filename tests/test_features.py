"""Tests de features de abandono, target encoding sin fuga y partición por usuario."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features as ft
from src import synthetic as syn


def meta_mini(n: int = 30) -> pd.DataFrame:
    gens = ["Drama", "Action", "Comedy"]
    return pd.DataFrame({
        "Series_Title": [f"T{i}" for i in range(n)],
        "Genre": [", ".join(sorted({gens[i % 3], gens[(i + 1) % 3]})) if i % 2 else gens[i % 3]
                  for i in range(n)],
        "Runtime": 90 + np.arange(n), "Released_Year": 1990 + np.arange(n),
        "IMDB_Rating": 7.0 + (np.arange(n) % 10) / 10, "No_of_Votes": 1000 + 10 * np.arange(n),
        "Meta_score": [np.nan if i % 7 == 0 else 60.0 + i for i in range(n)],
    })


def inter_mini(n_users: int = 200, por_user: int = 8, n_t: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    edades = np.array(ft.EDADES)
    filas = []
    for u in range(n_users):
        e = edades[u % 3]
        for t in rng.choice(n_t, size=por_user, replace=False):
            filas.append((u, f"T{t}", bool(rng.random() > 0.1), e))
    return pd.DataFrame(filas, columns=["User_ID", "Series_Title", "Watched_to_End", "Age_Group"])


@pytest.fixture(scope="module")
def tabla():
    return ft.construir_tabla(inter_mini(), meta_mini())


def test_tabla_columnas_y_etiqueta(tabla):
    inter = inter_mini()
    assert len(tabla) == len(inter)
    assert set(tabla["abandona"].unique()) <= {0, 1}
    assert (tabla["abandona"].to_numpy() == (~inter["Watched_to_End"]).to_numpy()).all()
    generos = ft.lista_generos(meta_mini())
    assert generos == ["Action", "Comedy", "Drama"]
    assert all(c in tabla for c in ft.columnas_binarias(generos))
    assert tabla["Meta_score"].isna().any()  # sin imputar al construir


def test_multihot_y_interaccion_consistentes(tabla):
    # ag_<edad>_<género> == edad_<edad> * g_<género>; un grupo de edad por fila
    assert (tabla[[f"edad_{e}" for e in ft.EDADES]].sum(axis=1) == 1).all()
    for e in ft.EDADES:
        for g in ("Action", "Comedy", "Drama"):
            assert (tabla[f"ag_{e}_{g}"] == tabla[f"edad_{e}"] * tabla[f"g_{g}"]).all()
    esperado = tabla["Genre"].str.split(", ").map(len)
    assert (tabla[[f"g_{g}" for g in ("Action", "Comedy", "Drama")]].sum(axis=1) == esperado).all()


def test_construir_tabla_titulo_sin_metadatos_falla():
    inter = inter_mini()
    with pytest.raises(ValueError):
        ft.construir_tabla(inter, meta_mini(n=20))


def test_particion_por_usuario_sin_solapes(tabla):
    p = ft.particion_por_usuario(tabla, seed=1)
    todos = np.concatenate(list(p.values()))
    assert len(todos) == len(set(todos)) == len(tabla)  # cobertura exacta, sin repetir filas
    us = {k: set(tabla["User_ID"].iloc[v]) for k, v in p.items()}
    assert not (us["train"] & us["validacion"] or us["train"] & us["test"]
                or us["validacion"] & us["test"])
    again = ft.particion_por_usuario(tabla, seed=1)
    assert all((p[k] == again[k]).all() for k in p)  # reproducible
    assert abs(len(p["train"]) / len(tabla) - 0.6) < 0.08


def test_te_titulos_no_vistos_y_suavizado():
    X = pd.DataFrame({"Series_Title": ["a", "a", "b", "b"], "User_ID": [1, 2, 3, 4]})
    y = np.array([1, 1, 0, 0])
    te = ft.TasaAbandonoTitulo(suavizado=2.0).fit(X, y)
    nuevo = pd.DataFrame({"Series_Title": ["a", "b", "zzz"], "User_ID": [9, 9, 9]})
    r = te.transform(nuevo).ravel()
    assert r[2] == pytest.approx(0.5)  # no visto -> media global
    assert r[0] == pytest.approx((2 + 2 * 0.5) / (2 + 2))  # suavizado hacia la media
    assert r[1] == pytest.approx((0 + 2 * 0.5) / (2 + 2))
    assert te.get_feature_names_out().tolist() == ["tasa_abandono_titulo"]


def test_te_fit_transform_oof_no_ve_su_etiqueta():
    # Etiquetas puramente aleatorias con pocos usuarios por título: la codificación ingenua
    # (fit y transform sobre las mismas filas) se correlaciona con y; la out-of-fold no.
    rng = np.random.default_rng(0)
    n_u, n_t = 3000, 400
    X = pd.DataFrame({"Series_Title": rng.integers(0, n_t, n_u * 3).astype(str),
                      "User_ID": np.repeat(np.arange(n_u), 3)})
    y = (rng.random(len(X)) < 0.1).astype(int)
    te = ft.TasaAbandonoTitulo(suavizado=2.0)
    oof = te.fit_transform(X, y).ravel()
    ingenua = ft.TasaAbandonoTitulo(suavizado=2.0).fit(X, y).transform(X).ravel()
    assert np.corrcoef(ingenua, y)[0, 1] > 0.1  # autocontribución: sesgo demostrado
    assert abs(np.corrcoef(oof, y)[0, 1]) < 0.05  # out-of-fold: sin sesgo
    # transform posterior usa el ajuste completo (no el OOF)
    assert not np.allclose(te.transform(X).ravel(), oof)


def test_te_fit_solo_con_train_dentro_del_pipeline(tabla):
    from sklearn.dummy import DummyClassifier

    gens = ft.lista_generos(meta_mini())
    pipe = ft.crear_pipeline(DummyClassifier(), gens)
    cols = ft.columnas_entrada(gens)
    p = ft.particion_por_usuario(tabla, seed=0)
    tr, va = tabla.iloc[p["train"]], tabla.iloc[p["validacion"]]
    pipe.fit(tr[cols], tr["abandona"])
    te = pipe.named_steps["prep"].named_transformers_["te"]
    assert te.prior_ == pytest.approx(tr["abandona"].mean())  # solo train
    Xv = pipe.named_steps["prep"].transform(va[cols])
    assert "User_ID" not in Xv.columns and "tasa_abandono_titulo" in Xv.columns
    assert not Xv.isna().any().any()  # Meta_score imputado dentro del pipeline


def test_generador_sembrado_reproducible_y_prevalencia():
    big = ft.construir_tabla(inter_mini(1500, 8), meta_mini())
    y1, p1 = syn.generar_abandono_sembrado(big, 1.0, 0.10, seed=3)
    y2, p2 = syn.generar_abandono_sembrado(big, 1.0, 0.10, seed=3)
    y3, _ = syn.generar_abandono_sembrado(big, 1.0, 0.10, seed=4)
    assert (y1 == y2).all() and np.allclose(p1, p2)
    assert not (y1 == y3).all()
    assert p1.mean() == pytest.approx(0.10, abs=1e-6)
    assert abs(y1.mean() - 0.10) < 0.01
    assert p1.std() > 0.02  # hay heterogeneidad


def test_generador_intensidad_cero_sin_senal():
    big = ft.construir_tabla(inter_mini(1500, 8), meta_mini())
    y, p = syn.generar_abandono_sembrado(big, 0.0, 0.12, seed=1)
    assert np.allclose(p, 0.12)
    y2, p2 = syn.generar_abandono_sembrado(big, 2.0, 0.12, seed=1)
    assert p2.std() > p.std()  # la intensidad aumenta la señal
    with pytest.raises(ValueError):
        syn.generar_abandono_sembrado(big, -1.0)
    with pytest.raises(ValueError):
        syn.generar_abandono_sembrado(big, 1.0, prevalencia=1.5)


# --- T05b: género exacto, n_generos y su interacción con la edad ---------------------------


def test_features_genero_exacto_y_n_generos(tabla):
    inter, meta = inter_mini(), meta_mini()
    gen = inter.merge(meta[["Series_Title", "Genre"]], on="Series_Title")["Genre"]
    assert (tabla["genero_exacto"].to_numpy() == gen.to_numpy()).all()  # cadena completa
    n = gen.str.split(", ").map(len).to_numpy()
    assert (tabla["n_generos"].to_numpy() == n).all()
    assert (tabla["genero_unico"].to_numpy() == (n == 1)).all()
    assert (tabla["edad_genero_exacto"] == tabla["Age_Group"] + "|" + tabla["Genre"]).all()
    for e in ft.EDADES:  # n_generos x edad: n si es de ese grupo, 0 si no
        assert (tabla[f"ng_{e}"] == tabla[f"edad_{e}"] * tabla["n_generos"]).all()
    # "Drama" a secas y "Drama, Action" son claves distintas (igualdad exacta, no "contiene")
    assert tabla["genero_exacto"].nunique() > len(ft.lista_generos(meta_mini()))


def test_te_clave_arbitraria_oof_sin_sesgo_y_nombre():
    # La misma garantía que para el título, con una clave edad|género de pocas categorías
    # pero etiquetas aleatorias: ingenua = r > 0 solo si hay pocas filas por clave.
    rng = np.random.default_rng(1)
    n_u = 3000
    X = pd.DataFrame({"clave": rng.integers(0, 600, n_u * 2).astype(str),
                      "User_ID": np.repeat(np.arange(n_u), 2)})
    y = (rng.random(len(X)) < 0.1).astype(int)
    mk = lambda: ft.TasaAbandonoTitulo(suavizado=1.0, col_titulo="clave",  # noqa: E731
                                       nombre_salida="tasa_clave")
    oof = mk().fit_transform(X, y).ravel()
    ingenua = mk().fit(X, y).transform(X).ravel()
    assert np.corrcoef(ingenua, y)[0, 1] > 0.1
    assert abs(np.corrcoef(oof, y)[0, 1]) < 0.05
    assert mk().get_feature_names_out().tolist() == ["tasa_clave"]
    # categoría no vista -> media global
    te = mk().fit(X, y)
    nuevo = pd.DataFrame({"clave": ["__nueva__"], "User_ID": [0]})
    assert te.transform(nuevo).ravel()[0] == pytest.approx(y.mean())


def test_pipeline_techo_sin_fuga_y_aprende_el_par(tabla):
    from sklearn.ensemble import RandomForestClassifier

    gens = ft.lista_generos(meta_mini())
    cols = ft.columnas_entrada_techo(gens)
    # señal: jóvenes que ven género exactamente "Drama" abandonan mucho más
    rng = np.random.default_rng(3)
    seg = ((tabla["Age_Group"] == "youth") & (tabla["Genre"] == "Drama")).to_numpy()
    y = (rng.random(len(tabla)) < np.where(seg, 0.6, 0.08)).astype(int)
    t = tabla.assign(abandona=y)
    p = ft.particion_por_usuario(t, seed=0)
    tr, va = t.iloc[p["train"]], t.iloc[p["validacion"]]
    pipe = ft.crear_pipeline_techo(RandomForestClassifier(50, random_state=0), gens)
    pipe.fit(tr[cols], tr["abandona"])
    prep = pipe.named_steps["prep"]
    # los codificadores se ajustan solo con train
    assert prep.named_transformers_["te_par"].prior_ == pytest.approx(tr["abandona"].mean())
    Xv = prep.transform(va[cols])
    assert {"tasa_edad_genero_exacto", "tasa_genero_exacto", "n_generos"} <= set(Xv.columns)
    assert "User_ID" not in Xv.columns and not Xv.isna().any().any()
    # la tasa del par youth|Drama aprendida en train es alta frente a otro par
    enc = prep.named_transformers_["te_par"]
    tasa = np.asarray(enc.transform(pd.DataFrame(
        {"edad_genero_exacto": ["youth|Drama", "adults|Drama"], "User_ID": [0, 0]}))).ravel()
    assert tasa[0] > tasa[1] + 0.2
