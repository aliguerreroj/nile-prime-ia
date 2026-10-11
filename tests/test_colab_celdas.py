"""Celdas iniciales de los 5 notebooks: arranque en Colab sin subir nada."""
from pathlib import Path

import pytest

PY = Path(__file__).resolve().parents[1] / "notebooks" / "py"
FUENTES = sorted(PY.glob("0[0-4]_*.py"))


def test_hay_cinco_fuentes():
    assert len(FUENTES) == 5


@pytest.mark.parametrize("fuente", FUENTES, ids=lambda p: p.name)
def test_celdas_de_arranque_colab(fuente):
    texto = fuente.read_text(encoding="utf-8")
    assert 'REPO_URL = "https://github.com/aliguerreroj/nile-prime-ia.git"' in texto
    assert "NilePrime_AliGuerrero_codigo.zip" in texto
    assert "## Requisitos" in texto
    assert "git" in texto and "clone" in texto
    assert "Instalación de dependencias" in texto
