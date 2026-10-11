"""Tests de src/data.py: tipos, filas, nulos, duplicados y modo rápido."""

import pandas as pd
import pytest

from src import data


@pytest.fixture(scope="session")
def resenas():
    return data.load_reviews(rapido=False)


@pytest.fixture(scope="session")
def resenas_unicas():
    return data.load_reviews(sin_duplicados=True, rapido=False)


@pytest.fixture(scope="session")
def meta():
    return data.load_metadata()


@pytest.fixture(scope="session")
def inter():
    return data.load_interactions(rapido=False)


def test_raiz_y_glob():
    assert (data.project_root() / "data" / "raw").is_dir()
    assert data.find_csv("Rese*.csv").exists()
    assert data.find_csv("interacciones*.csv").exists()
    assert data.find_csv("Metadatos*.csv").exists()


def test_filas_completas(resenas, meta, inter):
    assert len(resenas) == data.N_RESENAS == 50_000
    assert len(meta) == data.N_METADATOS == 1_000
    assert len(inter) == data.N_INTERACCIONES == 100_624


def test_resenas_sin_html_y_etiqueta_intacta(resenas):
    assert not resenas["review"].str.contains(r"<[^>]+>", regex=True).any()
    assert resenas["sentiment"].value_counts().to_dict() == {
        "positive": 25_000,
        "negative": 25_000,
    }
    assert resenas["review"].isna().sum() == 0


def test_resenas_duplicados(resenas, resenas_unicas):
    # 418 duplicados exactos en bruto + los que solo aparecen tras limpiar el HTML.
    assert len(resenas_unicas) == data.N_RESENAS_UNICAS
    # Se deduplica solo por texto: ninguna reseña puede repetirse entre train y test.
    assert resenas_unicas["review"].is_unique
    # Ningún texto con etiquetas contradictorias en los datos completos.
    assert resenas.groupby("review")["sentiment"].nunique().max() == 1


def test_metadatos_tipos(meta):
    assert pd.api.types.is_integer_dtype(meta["Runtime"])
    assert pd.api.types.is_integer_dtype(meta["Released_Year"])
    assert meta["Gross"].dtype == "float64"
    assert meta["Meta_score"].dtype == "float64"
    assert meta["Runtime"].between(30, 400).all()
    assert meta["Released_Year"].between(1900, 2030).all()


def test_apollo13_corregido(meta):
    assert meta.loc[meta["Series_Title"] == "Apollo 13", "Released_Year"].item() == 1995


def test_metadatos_nulos(meta):
    assert meta["Certificate"].isna().sum() == 0
    assert (meta["Certificate"] == "Desconocido").sum() == 101
    assert meta["Meta_score"].isna().sum() == 157  # sin imputar
    assert meta["Gross"].isna().sum() == 169  # sin imputar


def test_metadatos_drishyam_y_movie_id(meta):
    assert meta["movie_id"].is_unique
    dris = meta[meta["Series_Title"] == "Drishyam"]
    assert sorted(dris["Released_Year"]) == [2013, 2015]
    unicos = data.load_metadata(titulos_unicos=True)
    assert len(unicos) == 999
    assert unicos["Series_Title"].is_unique
    fila = unicos[unicos["Series_Title"] == "Drishyam"].iloc[0]
    assert fila["Released_Year"] == 2015  # la de más votos


def test_movie_id_estable_con_titulos_unicos(meta):
    unicos = data.load_metadata(titulos_unicos=True)
    # Cada fila conserva el mismo movie_id (y el mismo contenido) que en la carga completa.
    esperado = meta[meta["movie_id"].isin(unicos["movie_id"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(unicos, esperado)
    id_2015 = meta.loc[
        (meta["Series_Title"] == "Drishyam") & (meta["Released_Year"] == 2015), "movie_id"
    ].item()
    assert unicos.loc[unicos["Series_Title"] == "Drishyam", "movie_id"].item() == id_2015


def test_interacciones_duplicados(inter):
    assert inter.isna().sum().sum() == 0
    assert inter.duplicated().sum() == 1
    assert inter.loc[inter.duplicated(keep=False), "User_ID"].unique().tolist() == [6537]
    limpias = data.load_interactions(sin_duplicados=True, rapido=False)
    assert len(limpias) == 100_623
    assert not limpias.duplicated().any()


def test_modo_rapido_resenas(resenas):
    a = data.load_reviews(rapido=True)
    b = data.load_reviews(rapido=True)
    assert len(a) == data.FAST_N_RESENAS
    assert a.equals(b)
    assert a["sentiment"].value_counts().nunique() == 1  # estratificada
    assert len(a) < len(resenas)


def test_modo_rapido_resenas_sin_duplicados():
    a = data.load_reviews(sin_duplicados=True, rapido=True)
    b = data.load_reviews(sin_duplicados=True, rapido=True)
    assert len(a) == data.FAST_N_RESENAS
    assert a.equals(b)
    assert a["review"].is_unique
    assert a["sentiment"].value_counts().nunique() == 1  # sigue estratificada


def test_modo_rapido_interacciones(inter):
    a = data.load_interactions(rapido=True)
    b = data.load_interactions(rapido=True)
    assert a.equals(b)
    assert a["User_ID"].nunique() == round(inter["User_ID"].nunique() * data.FAST_FRAC_USUARIOS)
    # usuarios completos: mismas filas que en la carga completa
    assert len(a) == inter["User_ID"].isin(a["User_ID"].unique()).sum()


def test_modo_rapido_por_variable_de_entorno(monkeypatch):
    monkeypatch.setenv("NP_FAST", "1")
    assert data.is_fast()
    assert len(data.load_metadata()) == 1_000  # metadatos siempre completos
    assert len(data.load_reviews()) == data.FAST_N_RESENAS


def test_sin_variable_todo(monkeypatch):
    monkeypatch.delenv("NP_FAST", raising=False)
    assert not data.is_fast()
    assert len(data.load_interactions()) == 100_624
