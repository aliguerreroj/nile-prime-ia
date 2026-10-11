"""Tests del generador sintético con señal sembrada."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import synthetic as syn
from tests.conftest import catalogo_mini, referencia_mini


def mismo_genero(df: pd.DataFrame, cat: pd.DataFrame) -> float:
    """Fracción de pares de títulos del mismo usuario que comparten algún género."""
    gen = dict(zip(cat["Series_Title"], cat["Genre"].str.split(", ").map(set), strict=True))
    comparte, total = 0, 0
    for _, g in df.groupby("User_ID"):
        sets = [gen[t] for t in g["Series_Title"]]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                comparte += bool(sets[i] & sets[j])
                total += 1
    return comparte / total


def test_reproducible_y_semilla():
    cat, ref = catalogo_mini(), referencia_mini()
    a = syn.generar_interacciones_sembradas(cat, ref, 0.6, seed=1)
    b = syn.generar_interacciones_sembradas(cat, ref, 0.6, seed=1)
    c = syn.generar_interacciones_sembradas(cat, ref, 0.6, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a["Series_Title"].equals(c["Series_Title"])


def test_tamanos_y_esquema():
    cat, ref = catalogo_mini(), referencia_mini()
    df = syn.generar_interacciones_sembradas(cat, ref, 0.5)
    assert list(df.columns) == syn.COLUMNAS
    assert len(df) == len(ref) and df["User_ID"].nunique() == ref["User_ID"].nunique()
    esperado = ref.groupby("User_ID").size()
    assert (df.groupby("User_ID").size() == esperado).all()
    assert not df.duplicated(["User_ID", "Series_Title"]).any()
    assert set(df["Series_Title"]) <= set(cat["Series_Title"])
    assert (df.groupby("User_ID")["Age_Group"].first()
            == ref.groupby("User_ID")["Age_Group"].first()).all()
    assert all(len(v) == 2 for v in df.attrs["generos_preferidos"].values())


def test_p_pref_cero_sin_estructura_y_alto_con_estructura():
    cat, ref = catalogo_mini(), referencia_mini(300)
    s0 = mismo_genero(syn.generar_interacciones_sembradas(cat, ref, 0.0, seed=3), cat)
    s9 = mismo_genero(syn.generar_interacciones_sembradas(cat, ref, 0.9, seed=3), cat)
    # barajado entre usuarios (sin estructura) como referencia nula
    rng = np.random.default_rng(0)
    base = syn.generar_interacciones_sembradas(cat, ref, 0.0, seed=3).copy()
    base["Series_Title"] = rng.permutation(base["Series_Title"].to_numpy())
    nulo = mismo_genero(base, cat)
    assert abs(s0 - nulo) < 0.03
    assert s9 > s0 + 0.1


def test_p_pref_invalido():
    with pytest.raises(ValueError):
        syn.generar_interacciones_sembradas(catalogo_mini(), referencia_mini(), 1.5)
