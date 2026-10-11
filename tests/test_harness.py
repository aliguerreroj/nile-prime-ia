"""Tests del arnés: reglas de tasks.json y comprobaciones de validate.py."""

import common
import validate


def _tarea(tid, estado="pending", depende_de=()):
    return {
        "id": tid,
        "titulo": tid,
        "fase": "entrega",
        "estado": estado,
        "depende_de": list(depende_de),
        "criterios_aceptacion": ["criterio"],
        "artifacts": [],
        "validacion": "",
    }


def test_tasks_json_cumple_el_esquema():
    assert common.errores_tareas(common.cargar_tareas()) == []


def test_tasks_json_tiene_las_tareas_previstas():
    tareas = common.cargar_tareas()
    entrega = [t["id"] for t in tareas if t["fase"] == "entrega"]
    escalado = [t for t in tareas if t["fase"] == "escalado"]
    # T05b se añadió después de T06 (techo de señal del reto 3); va tras T06 en el orden.
    previstas = [f"T{i:02d}" for i in range(11)]
    previstas.insert(previstas.index("T06") + 1, "T05b")
    assert entrega == previstas
    assert [t["id"] for t in escalado] == [f"E{i:02d}" for i in range(1, 6)]
    assert all(t["estado"] == "backlog" for t in escalado)


def test_solo_una_tarea_in_progress():
    tareas = [_tarea("T01", "in_progress"), _tarea("T02", "in_progress")]
    assert any("in_progress" in e for e in common.errores_tareas(tareas))


def test_dependencia_inexistente_es_error():
    assert common.errores_tareas([_tarea("T01", depende_de=["T99"])])


def test_detecta_ciclos_de_dependencias():
    ciclo = [
        _tarea("T01", depende_de=["T03"]),
        _tarea("T02", depende_de=["T01"]),
        _tarea("T03", depende_de=["T02"]),
    ]
    assert common.ciclos_dependencias(ciclo) == [["T01", "T03", "T02", "T01"]]
    assert any("ciclo" in e for e in common.errores_tareas(ciclo))
    assert any("ciclo" in e for e in common.errores_tareas([_tarea("T01", depende_de=["T01"])]))
    sin_ciclo = [
        _tarea("T01"),
        _tarea("T02", depende_de=["T01"]),
        _tarea("T03", depende_de=["T01", "T02"]),
    ]
    assert common.ciclos_dependencias(sin_ciclo) == []


def test_siguiente_tarea_respeta_dependencias():
    tareas = [
        _tarea("T01", "done"),
        _tarea("T02", "pending", ["T03"]),
        _tarea("T03", "pending", ["T01"]),
    ]
    assert common.siguiente_tarea(tareas)["id"] == "T03"
    bloqueadas = [_tarea("T01", "pending", ["T02"]), _tarea("T02", "blocked")]
    assert common.siguiente_tarea(bloqueadas) is None


def test_esquema_de_metricas_de_reto():
    completo = {
        "modo": "full",
        "config": {"semilla": 42},
        "baseline": {"nombre": "dummy"},
        "metricas": {"f1": {"valor": 0.9, "ic95": [0.89, 0.91]}},
        "umbral": {"valor": 0.5, "justificacion": "costos simétricos"},
    }
    assert validate.errores_metricas(completo, "abandono", "full") == []
    sin_umbral = {k: v for k, v in completo.items() if k != "umbral"}
    assert validate.errores_metricas(sin_umbral, "recomendador", "full") == []
    assert any("umbral" in e for e in validate.errores_metricas(sin_umbral, "abandono", "full"))
    mal_umbral = {**completo, "umbral": {"valor": 0.5}}
    assert any("justificacion" in e for e in validate.errores_metricas(mal_umbral, "rl", "full"))
    for clave in ("config", "baseline", "metricas"):
        incompleto = {k: v for k, v in completo.items() if k != clave}
        assert any(clave in e for e in validate.errores_metricas(incompleto, "sentimiento", "full"))


def test_esquema_de_metricas_modo():
    assert validate.errores_metricas({"modo": "full"}, "eda", "full") == []
    assert validate.errores_metricas({"modo": "fast"}, "eda", "quick") == []
    assert validate.errores_metricas({"modo": "fast"}, "eda", "full")
    assert validate.errores_metricas({}, "resumen", "quick")
    assert validate.errores_metricas([], "eda", "quick")


def test_detecta_rutas_absolutas():
    windows = "C:" + "\\" + "Users" + "\\" + "ana"
    assert validate.rutas_absolutas(f"pd.read_csv(r'{windows}')")
    assert validate.rutas_absolutas("open('/" + "home/ana/datos.csv')")
    assert validate.rutas_absolutas("open('D:" + "/datos/x.csv')")
    assert not validate.rutas_absolutas("ROOT / 'data' / 'raw'  # https://colab.research.google.com")
    assert not validate.rutas_absolutas("print('Total:\\n')")
