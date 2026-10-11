# Nile Prime — Caso práctico Módulo 10 (IA para la Empresa, IMMUNE)

Caso individual del Máster en Data Science & Business Analytics. Nile Prime es una plataforma
de streaming ficticia; como consultora resolvemos cuatro retos:

1. **Análisis de sentimiento** de reseñas (NLP), comparando varios modelos.
2. **Recomendador híbrido** (contenido + comportamiento del usuario).
3. **Predicción de abandono** con Random Forest.
4. **Mejora del recomendador con aprendizaje por refuerzo** sobre el entorno `agent.py`.

Entregables: notebooks ejecutables de principio a fin (también en Google Colab) y un informe
PDF para público no técnico.


## Para el profesor
Repositorio público: <https://github.com/aliguerreroj/nile-prime-ia> (rama `main`).

**Opción 1 — Colab con un clic (no hay que subir nada)**
1. Pulsa el badge *Open in Colab* del notebook que quieras (tabla de [Notebooks](#notebooks)).
2. *Entorno de ejecución → Ejecutar todo*.
3. La primera celda clona el repositorio con los datos e instala solo lo que falte; unos 20–30 min
   en total los cinco notebooks.

**Opción 2 — Local con el zip**
1. Descomprime `NilePrime_AliGuerrero_codigo.zip`.
2. `python harness/init.py --install` (o `pip install -r requirements.txt`, Python ≥ 3.11).
3. Ejecuta los notebooks de `notebooks/` con *Restart & Run All*.

## Estructura
```
agent.py            entorno de simulación original (no se modifica)
data/raw/           los 3 CSV originales
src/                lógica reutilizable (carga, features, recomendador, entorno RL, gráficos)
notebooks/          notebooks .ipynb generados;  notebooks/py/  fuentes jupytext
tests/              pytest
reports/            figures/ y metrics/ que alimentan el informe
docs/               contexto del caso y referencia del arnés
harness/            init.py, validate.py, tasks.json
progress/           estado, notas de sesión y decisiones (no viaja en el zip de la entrega)
```

## Puesta en marcha
Requiere Python ≥ 3.11. El primer comando crea el entorno virtual `.venv` e instala las
dependencias fijadas; después los scripts lo usan solos, sin activarlo.

```bash
python harness/init.py --install
```

```bash
python harness/validate.py --quick
```

`init.py` verifica el entorno y los datos y muestra la siguiente tarea; `validate.py` es la
puerta de calidad (ruff, pytest, ejecución de notebooks y artifacts). Detalle en
[docs/ARNES.md](docs/ARNES.md); el caso y los hallazgos, en
[docs/CONTEXTO_CASO.md](docs/CONTEXTO_CASO.md).

## Cómo ejecutar
La entrega es un zip (`NilePrime_AliGuerrero_codigo.zip`) con el código, los datos y el informe PDF.
Cada notebook es independiente y se ejecuta con *Restart & Run All*, en cualquier orden.

**Local (3 pasos)**
1. Descomprime el zip y crea el entorno: `python harness/init.py --install`
   (o `pip install -r requirements.txt` en un entorno con Python ≥ 3.11).
2. Abre la carpeta `notebooks/` y ejecuta cada `.ipynb` con *Restart & Run All* (kernel del entorno creado).
3. Resultados en `reports/` (`figures/` y `metrics/`); el informe es `informe_nile_prime.pdf`.

**Google Colab (3 pasos)**
1. Abre el notebook con su badge *Open in Colab* (tabla de [Notebooks](#notebooks)); no hay que subir nada.
2. *Entorno de ejecución → Ejecutar todo*: si no encuentra `data/raw`, la primera celda clona
   automáticamente <https://github.com/aliguerreroj/nile-prime-ia> (`git clone --depth 1`) y la celda
   siguiente instala con `%pip` solo lo que falte (no reinstala numpy, pandas, scikit-learn,
   matplotlib ni torch).
3. Revisa las salidas. El notebook 01 descarga DistilBERT de Hugging Face la primera vez.

*Alternativa sin GitHub:* sube `NilePrime_AliGuerrero_codigo.zip` a `/content` (panel de archivos) junto
con el notebook; la primera celda lo descomprime y lo usa en lugar de clonar. Si no hay internet ni zip,
la celda falla con un mensaje que indica ambas salidas.

Para regenerar el paquete (mantenimiento): `python harness/validate.py --full` y después
`python harness/empaquetar_entrega.py` (empaqueta y verifica el zip en una carpeta temporal; escribe
`entrega/`, que está en `.gitignore` porque el zip pesa decenas de MB).

## Notebooks
Tiempos de ejecución medidos (modo full, *Restart & Run All* desde el zip descomprimido en una
carpeta limpia; equipo de 16 núcleos, sin GPU, con DistilBERT ya descargado). Varían unos segundos entre ejecuciones. **En total, unos
20 minutos**; en un equipo más modesto o en Colab, cuenta con 25–30 minutos y con la descarga de
DistilBERT (268 MB) la primera vez.

| Notebook | Reto | Tiempo medido | Colab |
|---|---|---|---|
| `00_EDA_y_preparacion` | Exploración, calidad y limpieza de los datos | 13 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliguerreroj/nile-prime-ia/blob/main/notebooks/00_EDA_y_preparacion.ipynb) |
| `01_sentimiento` | Análisis de sentimiento: modelos clásicos y DistilBERT | 4 min 54 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliguerreroj/nile-prime-ia/blob/main/notebooks/01_sentimiento.ipynb) |
| `02_recomendador` | Recomendador híbrido (contenido + comportamiento) | 3 min 38 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliguerreroj/nile-prime-ia/blob/main/notebooks/02_recomendador.ipynb) |
| `03_abandono_random_forest` | Predicción de abandono con Random Forest | 6 min 39 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliguerreroj/nile-prime-ia/blob/main/notebooks/03_abandono_random_forest.ipynb) |
| `04_aprendizaje_por_refuerzo` | Mejora del recomendador con refuerzo sobre `agent.py` | 4 min 35 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliguerreroj/nile-prime-ia/blob/main/notebooks/04_aprendizaje_por_refuerzo.ipynb) |
| **Total** | | **20 min 2 s** | |

## Resultados principales
Cifras de la ejecución completa (`reports/metrics/*.json`, modo `full`); el detalle, los intervalos
de confianza y las limitaciones están en cada notebook.

| Reto | Resultado |
|---|---|
| 1 · Sentimiento | LinearSVC calibrado con umbral 0,60 (costos FP:FN = 2:1): accuracy 0,910 [0,904; 0,915] y F1 0,908 en test. DistilBERT no mejora de forma significativa y es entre 13 y 20 veces más lento por reseña suelta (la razón exacta varía entre ejecuciones). |
| 2 · Recomendador | En los datos reales ningún método supera al azar (HitRate@10 ≈ 1 %): el histórico no contiene preferencias. Con señal sembrada el híbrido sí la recupera (≈ 2,7 veces los aciertos de la popularidad con `p_pref` = 0,6). |
| 3 · Abandono | Random Forest con ROC-AUC 0,523 [0,510; 0,536]: señal detectable pero sin utilidad para actuar usuario a usuario; está en el techo de la regla que generó los datos (0,518). Segmentos de riesgo: jóvenes × drama puro (20,6 %) y mayores × comedia pura (22,0 %) frente al 10,3 % global. |
| 4 · Refuerzo | En el simulador de `agent.py` (corregidos dos fallos, sin cambiar sus reglas) el agente de Q-learning alcanza una recompensa de 0,99995 frente a 0,990 del azar y 0,974 del recomendador estático; el margen es pequeño en el modo original y mayor (+13 puntos frente al recomendador estático) en el análisis de sensibilidad. |

## Informe
El informe para público no técnico es `informe_nile_prime.pdf` (en el repositorio,
`reports/informe_nile_prime.pdf`). En el repositorio (el script no viaja en el zip de la entrega) se regenera con `python reports/informe/build_informe.py`, que
consolida las métricas en `reports/metrics/resumen.json` y necesita Edge o Chrome instalados (la
variable `NP_CHROME` indica la ruta si no lo encuentra).

## Modelos externos
El notebook 01 usa modelos públicos de Hugging Face, que se descargan solos la primera vez:

- DistilBERT (se ejecuta siempre, en CPU): <https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english>
- RoBERTa-large (solo si hay GPU; sin GPU se omite): <https://huggingface.co/siebert/sentiment-roberta-large-english>
