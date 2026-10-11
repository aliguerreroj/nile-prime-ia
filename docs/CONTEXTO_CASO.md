# Contexto del caso — Módulo 10 · "Nile Prime" (IA para la Empresa, IMMUNE)

> Detalle del caso, hallazgos verificados y enfoque por reto. Movido aquí desde el
> `CLAUDE.md` original (escrito el 2026-10-10) sin perder contenido; `CLAUDE.md` es ahora
> el índice corto. "Entrega: mañana" equivale a **2026-10-11**.
> Las secciones 1-6 son el texto original; la 7 y la 8 se añadieron al montar el arnés.

## 1. Contexto
Proyecto INDIVIDUAL del Máster en Data Science & Business Analytics (IMMUNE). Entrega: mañana.
Nile Prime es una plataforma de streaming ficticia. Somos la consultora que debe resolver 4 retos:
1. **Análisis de sentimiento** de reseñas (NLP). Comparar varios modelos y elegir el mejor.
2. **Recomendador basado en contenido**, idealmente **híbrido** (contenido + comportamiento del usuario).
3. **Predicción de abandono** (no ver la película hasta el final). **Random Forest es obligatorio.**
4. **Mejora del recomendador con aprendizaje por refuerzo**, usando el entorno simulado de `agent.py` (Q-learning y/o DQN).

Entregables: (a) notebooks Python bien comentados que **se ejecuten de principio a fin sin errores**, también en Google Colab,
y (b) un **informe en PDF** para **público NO técnico** (al final del proyecto).

## 2. Reglas de calidad (no negociables)
- Idioma: **español** en markdown, comentarios y gráficos. Los nombres de variables y funciones pueden ir en inglés.
- Cada notebook empieza con: título, objetivo de negocio, celda `%pip install -q ...` (versiones fijadas como en `requirements.txt`), imports y semilla global (`SEED = 42`).
- **Cada notebook es independiente**: se ejecuta solo con "Restart & Run All" y no depende de que otro se haya ejecutado antes. La lógica compartida (carga, limpieza, features, entorno) va en `src/` como funciones reutilizables con docstrings.
- **Compatibilidad con Colab**: la primera celda detecta si corre en Colab (`'google.colab' in sys.modules`), prepara el proyecto (si no encuentra `data/raw`, descomprime el zip de la entrega subido a `/content`; la entrega es por Classroom, sin GitHub) y resuelve las rutas. Nunca usar rutas absolutas de Windows.
- Antes de cada sección de código, explicar en markdown QUÉ se hace y POR QUÉ (base conceptual primero, después el código). Cerrar cada sección con una conclusión en lenguaje sencillo.
- Gráficos: título, ejes con etiqueta y unidades en español, y paleta consistente. Guardar cada figura relevante en `reports/figures/NN_nombre.png` (dpi=150) para usarla después en el informe.
- Guardar las métricas finales de cada reto en `reports/metrics/<reto>.json` para alimentar el informe.
- Validación correcta: separación train/test sin fuga de información (*leakage*), validación cruzada donde aplique y una línea base (*baseline*) contra la cual comparar cada modelo.
- Verificar SIEMPRE al terminar un notebook: `jupyter nbconvert --to notebook --execute --inplace <nb>.ipynb` sin errores. Los tiempos deben ser razonables en CPU; si algo necesita GPU, se detecta con `torch.cuda.is_available()` y hay una alternativa que funciona en CPU.
- Nada de celdas con errores "esperados" sin capturar: si se demuestra un bug, se hace dentro de `try/except` y se imprime el mensaje.

## 3. Estructura objetivo
```
data/raw/            # CSV originales (no modificar)
src/                 # data.py, features.py, recommender.py, rl_env.py, plots.py
notebooks/00_EDA_y_preparacion.ipynb
notebooks/01_sentimiento.ipynb
notebooks/02_recomendador.ipynb
notebooks/03_abandono_random_forest.ipynb
notebooks/04_aprendizaje_por_refuerzo.ipynb
reports/figures/  reports/metrics/
agent.py             # entorno original del profesor: NO modificarlo
requirements.txt  README.md
```

## 4. Datos (hallazgos ya verificados)
- `Reseñas Películas.csv`: 50.000 filas, columnas `review` y `sentiment` (25k positivas / 25k negativas; dataset IMDB). Contiene HTML (`<br />`). **No incluye el título de la película**, así que no se puede cruzar con el catálogo: es un módulo independiente.
- `Metadatos Películas.csv`: 1.000 películas (IMDB Top 1000). `Runtime` viene como texto ("142 min"), `Gross` con comas, y hay nulos en `Certificate` (101), `Meta_score` (157) y `Gross` (169). Hay 1 `Series_Title` duplicado. `Genre` es multi-etiqueta ("Action, Crime, Drama").
- `interacciones.csv`: 100.624 filas, columnas `User_ID`, `Series_Title`, `Watched_to_End`, `Age_Group`. Son 10.000 usuarios (~10 interacciones cada uno, entre 1 y 24) y 999 títulos, todos presentes en el catálogo. Cada usuario pertenece a un único grupo de edad (youth, adults o seniors, balanceados). Tasa de abandono: **10,3%** (clases desbalanceadas). La señal por género y edad es débil (tasa de visionado completo entre 0,886 y 0,905 por género y entre 0,892 y 0,902 por edad; V de Cramér 0,009 y 0,015). Las elecciones de título tampoco muestran preferencias: son independientes de la edad y sin coherencia de género por usuario frente a un baseline barajado (notebook 00, "Estructura de preferencias"). **No existe una columna de tiempo visto**, aunque el PDF la menciona.
- Los nombres de archivo llevan tildes y ñ: cargarlos de forma robusta (por ejemplo, con `glob`) y con `encoding='utf-8'`.

## 5. Hallazgos sobre `agent.py` (documentarlos en el notebook 04 y en el informe)
Corregido el 2026-10-10 tras verificarlo con los datos (ver `progress/decisions.md`).

**Bugs reales** (impiden ejecutar el entorno):
1. `step()` llama a `self._get_state()`, que **no existe**, y lanza `AttributeError`.
2. `reset()` no devuelve un estado (y toma una fila de interacciones, no un usuario).

**La comparación EXACTA de género NO es un bug**: reproduce el proceso que generó `interacciones.csv`. La regla penaliza solo cuando `Genre` es exactamente la cadena indicada (película de un único género), y los datos muestran justo ese patrón:
- youth + `Genre` exactamente "Drama" u "Horror": abandono 20,6 % (n = 2.937; IC 95 % 19,2–22,1). Drama: 85 películas, 20,8 %; Horror: 2 películas, n = 80, 13,8 % (muestra pequeña).
- seniors + exactamente "Comedy" o "Romance": 21,1 % (n = 474; IC 95 % 17,7–25,0). Son todas "Comedy" (13 películas); no hay ninguna película con exactamente "Romance".
- adults + exactamente "Action" o "Sci-Fi": 0 películas en el catálogo, así que nunca se penaliza.
- Resto: 9,8–10,0 %. Las películas que solo CONTIENEN el género (multi-etiqueta) no se penalizan: 9,9 % (n = 39.541). Por número de géneros: 1 → 13,1 %, 2 → 10,1 %, 3 → 9,8 %.
- AUC "oráculo" usando solo esta regla ≈ 0,52 (0,519 sobre todas las interacciones): el Random Forest del reto 3 (0,523 [0,510; 0,536]) ya está en el techo de la señal disponible.

**Inconsistencias entre el simulador y los datos** (no son bugs; hay que declararlas):
- En el simulador los géneros no penalizados tienen 0 % de abandono; en los datos, ~10 %.
- Las penalizaciones del simulador son 0,30 (youth) y 0,25 (seniors); en los datos el segmento abandona ~21 %.
- El valor por defecto de 0,8 nunca se usa, porque los tres grupos de edad están en el diccionario.

→ Crear `src/rl_env.py` con una clase `NilePrimeEnv` que herede del entorno original y solo corrija lo imprescindible: implementa `_get_state()` (codificación del grupo de edad) y hace que `reset()` devuelva el estado. `modo="original"` (por defecto) mantiene la comparación EXACTA, fiel al profesor y a los datos; `modo="contiene"` (pertenencia) queda como análisis de sensibilidad y, opcionalmente, `modo="calibrado"` usa las probabilidades estimadas de los datos (base ~10 %, segmento ~21 %). Documentar cada cambio.

## 6. Enfoque por reto
- **00 EDA**: limpieza, tipos de datos, nulos, duplicados, distribuciones, abandono por grupo de edad y género, y popularidad. Debe dejar listas las funciones de `src/data.py`.
- **01 Sentimiento**: limpieza de texto. Líneas base: TF-IDF (1–2 gramas) + Regresión Logística, LinearSVC y Naive Bayes, sobre los datos completos con train/test estratificado. Transformer preentrenado (por ejemplo `siebert/sentiment-roberta-large-english` y/o `distilbert-base-uncased-finetuned-sst-2-english`) evaluado sobre una muestra estratificada del test, con detección de GPU. Métricas: accuracy, F1, matriz de confusión y tiempo de inferencia. Elegir el modelo final explicando el equilibrio entre precisión y costo.
- **02 Recomendador**: (a) contenido: TF-IDF o embeddings de Overview + Genre + Director + Stars y similitud coseno; (b) colaborativo: matriz usuario×ítem implícita con `Watched_to_End`; (c) híbrido ponderado, con arranque en frío resuelto con popularidad y contenido. Evaluación *leave-one-out* con Precision@K, Recall@K, NDCG@K y HitRate@K, comparada contra popularidad y aleatorio. Exponer una función `recommend(user_id, k)`.
- **03 Abandono**: unir con los metadatos. Features: runtime, año, rating, votos, Meta_score, géneros multi-hot, grupo de edad y la interacción grupo de edad × género. Las features de historial del usuario se calculan sin fuga, separando por usuario con `GroupKFold`/`GroupShuffleSplit`. Manejar el desbalance con `class_weight`. Hacer *tuning* con validación cruzada. Métricas: ROC-AUC, PR-AUC, F1 y recall de la clase "abandona", ajuste del umbral, e importancia de variables (por impurity y por permutation). Si la señal es débil, decirlo con honestidad y explicar qué datos harían falta.
- **04 RL**: el problema es un *bandit* contextual (episodios de 1 paso). Implementar Q-learning tabular (estado = grupo de edad; acción = película o, mejor, re-ranking del top-K del recomendador híbrido) y opcionalmente un DQN pequeño. Comparar contra una política aleatoria y contra el recomendador sin RL. Mostrar curvas de recompensa media y la política aprendida por grupo de edad.

## 7. Enunciado del caso (PDF) — requisitos de la entrega
Fuente: `Caso práctico _ Inteligencia Artificial para la Empresa.pdf` (2 páginas, en la raíz).
- Entregable 1: **un informe** que cubra todos los puntos del desarrollo, con gráficos, tablas y conclusiones basadas en el análisis.
- Entregable 2: **código Python** (notebooks o `.py`) que documente cada etapa y los métodos usados, bien comentado y fácil de seguir.
- **El código debe ejecutarse sin errores.** Si se usan librerías, incluir el `pip install` correspondiente. Si se usan modelos externos, usar **enlaces accesibles**.
- Pistas del enunciado: (1) probar varios modelos de NLP y elegir el que mejor se adapte; (2) modelo híbrido de contenido + datos del usuario; (3) Random Forest, donde la selección e ingeniería de características es crucial; (4) agente de RL (Q-learning o DQN) en el entorno simulado.
- El PDF describe las reseñas "junto con las calificaciones asociadas" (en los datos solo hay `sentiment`) y las interacciones con "cuánto tiempo las han visto" (esa columna no existe): son diferencias entre el enunciado y los datos que hay que declarar.

## 8. Hallazgos al montar el arnés (2026-10-10)
- Los nombres `Metadatos Películas.csv` y `Reseñas Películas.csv` están en **Unicode NFD** (la tilde y la virgulilla son caracteres combinantes). Un literal escrito a mano (NFC) puede no abrir el archivo: localizar siempre con `glob` (p. ej. `Metadatos*.csv`, `Rese*.csv`).
- Los CSV viven ahora en `data/raw/`; su tamaño y sha256 están en `harness/data_manifest.json`.
- `agent.py` ejecuta `pd.read_csv('./Metadatos Películas.csv')` y `pd.read_csv('./interacciones.csv')` **al importarse**, relativo al directorio actual. Como no se modifica, `src/rl_env.py` (T06) debe importarlo de forma que no falle (p. ej. cambiando temporalmente el directorio de trabajo a `data/raw/`) y pasar sus propios DataFrames limpios al constructor.
- **El abandono se comporta como ruido** (verificado el 2026-10-10 sobre las 100.624 interacciones, tasa global 10,25 %): la correlación del abandono con la tasa *leave-one-out* del propio usuario es 0,006 y con la del título 0,014; la tasa es plana, entre 9,9 % y 10,5 %, por cuartiles de duración, rating, año y votos. Para el reto 3 se espera **AUC ≈ 0,5**, y la conclusión de negocio es **qué datos recoger** (tiempo visto, marcas de tiempo, dispositivo, momento del abandono, etc.), no un modelo predictivo útil.
- **Regla de `%pip` actualizada** (sustituye a la de la sección 2): en Colab no se reinstalan las librerías que ya trae; solo se instalan las que faltan, con versión fijada. Ver "Dependencias por entorno" en `docs/ARNES.md`.
- sha256 de `agent.py` original: `ff102044ff4126604f44cfe03ea11eeb9b91f5768d4477add8847b24ec3fb376`.
