"""Tests de src/plots.py."""

import matplotlib.pyplot as plt

from src import plots


def test_save_fig_dpi_y_ruta(tmp_path, monkeypatch):
    monkeypatch.setattr(plots, "project_root", lambda: tmp_path)
    plots.set_style()
    fig, ax = plt.subplots()
    ax.plot([1, 2])
    ruta = plots.save_fig(fig, "99_prueba")
    plt.close(fig)
    assert ruta == tmp_path / "reports" / "figures" / "99_prueba.png"
    assert ruta.exists() and ruta.stat().st_size > 0


def test_paleta():
    assert len(plots.COLORES) >= 4 and plots.DPI == 150


def test_paleta_unica_y_consistente():
    import re

    hexa = re.compile(r"^#[0-9a-f]{6}$")
    assert set(plots.GRUPOS_EDAD) == {"youth", "adults", "seniors"}
    assert all(hexa.match(c) for c in plots.PALETA.values())
    assert len(set(plots.GRUPOS_EDAD.values())) == 3  # colores distintos por grupo
    for clave in ("principal", "referencia", "enfasis", "neutro"):
        assert clave in plots.PALETA
    assert all(c in plots.PALETA.values() for c in plots.COLORES)
