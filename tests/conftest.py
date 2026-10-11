"""Fixtures y utilidades compartidas por los tests (catálogo e interacciones de juguete)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import synthetic as syn


def catalogo_mini(n: int = 80) -> pd.DataFrame:
    """Catálogo de juguete: 8 géneros, hasta 2 géneros por título."""
    generos = ["Drama", "Action", "Comedy", "Horror", "Sci-Fi", "Western", "Crime", "War"]
    filas = []
    for i in range(n):
        g = [generos[i % 8], generos[(i // 8 + i) % 8]]
        g = sorted(set(g), key=g.index)
        filas.append({
            "movie_id": i, "Series_Title": f"T{i}", "Genre": ", ".join(g),
            "Overview": f"palabra{i % 7} historia {g[0].lower()} personaje{i}",
            "Director": f"Dir{i % 10}", "Star1": f"A{i % 13}", "Star2": f"B{i % 11}",
            "Star3": f"C{i % 5}", "Star4": f"D{i}",
        })
    return pd.DataFrame(filas)


def referencia_mini(n_usuarios: int = 200, por_usuario: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    edades = np.array(["youth", "adults", "seniors"])
    filas = []
    for u in range(n_usuarios):
        e = edades[u % 3]
        for t in rng.choice(80, size=por_usuario, replace=False):
            filas.append((u, f"T{t}", bool(rng.random() < 0.9), e))
    return pd.DataFrame(filas, columns=syn.COLUMNAS)
