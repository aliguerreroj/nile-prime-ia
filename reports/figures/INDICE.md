# Índice de figuras

Todas las figuras se generan al ejecutar el notebook indicado (fuente en `notebooks/py/`), con la
paleta común de `src/plots.py`, textos en español y 150 dpi.

| Figura | Notebook de origen | Mensaje de negocio |
|---|---|---|
| `00_nulos_metadatos.png` | 00 · EDA y preparación | Los metadatos son casi completos; los pocos nulos (puntuación de crítica, recaudación) no impiden recomendar por contenido. |
| `00_abandono_por_edad.png` | 00 · EDA y preparación | La tasa de abandono apenas cambia por grupo de edad: la edad por sí sola no identifica a quién se irá. |
| `00_abandono_por_genero.png` | 00 · EDA y preparación | Tampoco hay géneros que concentren el abandono de forma clara; la señal para predecirlo es débil. |
| `00_popularidad_titulos.png` | 00 · EDA y preparación | Todos los títulos reciben un número de visionados casi igual (≈ 100): no hay éxitos ni cola larga que explotar. |
| `00_estructura_preferencias.png` | 00 · EDA y preparación | Lo que ve cada usuario no depende de su edad ni de sus gustos previos: el histórico no contiene preferencias que aprender. |
| `01_umbral_costo.png` | 01 · Sentimiento | El umbral de decisión se fija por el costo del error (dejar pasar una queja cuesta más que una falsa alarma), no por defecto en 0,5. |
| `01_matriz_confusion.png` | 01 · Sentimiento | De cada 100 reseñas se clasifican bien ≈ 91; el modelo falla en ambos sentidos de forma pareja. |
| `01_calibracion.png` | 01 · Sentimiento | Las probabilidades del modelo son fiables, de modo que se puede priorizar la revisión humana por nivel de confianza. |
| `01_comparacion_modelos.png` | 01 · Sentimiento | Un modelo clásico y ligero iguala al transformer preentrenado en acierto con una fracción del costo de cómputo. |
| `02_comparacion_recomendadores.png` | 02 · Recomendador | Con los datos reales ningún recomendador supera al azar ni a «lo más popular»: el histórico no tiene señal de gustos. |
| `02_senal_sembrada.png` | 02 · Recomendador | Si los usuarios tuvieran gustos reconocibles, el híbrido los detectaría y superaría claramente a «lo más popular»: el límite está en los datos, no en el método. |
| `03_prueba_permutacion.png` | 03 · Abandono (Random Forest) | La capacidad de predecir el abandono con los datos disponibles es apenas distinguible del azar. |
| `03_umbral.png` | 03 · Abandono (Random Forest) | El umbral de aviso se elige por el costo de perder a un cliente frente al de una acción de retención innecesaria. |
| `03_curvas_roc_pr.png` | 03 · Abandono (Random Forest) | El modelo discrimina poco mejor que el azar: sus avisos de riesgo deben tomarse como orientativos. |
| `03_calibracion.png` | 03 · Abandono (Random Forest) | Las probabilidades de abandono están calibradas, aunque sean poco informativas. |
| `03_importancia_variables.png` | 03 · Abandono (Random Forest) | La duración y pocas variables de uso son las que más pesan, y su peso es modesto. |
| `03_senal_sembrada.png` | 03 · Abandono (Random Forest) | Con una señal conocida el Random Forest la recupera: la debilidad en datos reales es una propiedad de los datos. |
| `03_importancia_genero_exacto.png` | 03 · Abandono (Random Forest) | La combinación edad × género aporta la única señal ligera detectable; hay que validarla antes de actuar sobre ella. |
| `03_segmentos_riesgo.png` | 03 · Abandono (Random Forest) | Las diferencias entre segmentos de riesgo son pequeñas; las campañas de retención por segmento tendrían efecto limitado. |
| `04_entorno_original_vs_corregido.png` | 04 · Aprendizaje por refuerzo | El entorno original del profesor tiene fallos que alteran la recompensa; con el corregido la comparación entre políticas es justa. |
| `04_curvas_recompensa.png` | 04 · Aprendizaje por refuerzo | Aprender explorando en línea mejora la recompensa media frente a recomendar sin aprender. |
| `04_regret_acumulado.png` | 04 · Aprendizaje por refuerzo | El «arrepentimiento» acumulado se frena con el aprendizaje: el coste de explorar se amortiza. |
| `04_politica_por_edad.png` | 04 · Aprendizaje por refuerzo | La política aprendida adapta qué recomendar a cada grupo de edad, algo que el histórico no permitía descubrir. |
