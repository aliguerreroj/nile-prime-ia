# Prueba en Google Colab (plantilla)

Rellenar a mano tras ejecutar los 5 notebooks en Colab con `NilePrime_AliGuerrero_codigo.zip`.

## Entorno
| Campo | Valor |
|---|---|
| Fecha | |
| Tipo de entorno (CPU / GPU, modelo) | |
| RAM del runtime | |
| Python | |
| numpy / pandas / scikit-learn | |
| matplotlib / torch | |
| transformers (instalado por el notebook 01) | |

## Pasos seguidos
1. Subir `NilePrime_AliGuerrero_codigo.zip` y el `.ipynb` a `/content`.
2. *Entorno de ejecución → Ejecutar todo*; la primera celda descomprime el zip e instala lo que falte.
3. (anotar aquí cualquier paso adicional)

## Resultado por notebook
| Notebook | Resultado (OK / FAIL) | Tiempo | Incidencias |
|---|---|---|---|
| `00_EDA_y_preparacion` | | | |
| `01_sentimiento` | | | |
| `02_recomendador` | | | |
| `03_abandono_random_forest` | | | |
| `04_aprendizaje_por_refuerzo` | | | |

## Observaciones
(avisos, reinicios de runtime, diferencias de versiones, descarga de DistilBERT, etc.)

## Verificación local del paquete
Antes de subir a Classroom, `python harness/empaquetar_entrega.py` descomprime el zip en una
carpeta temporal fuera del repo y ejecuta allí los 5 notebooks desde cero (sin salidas previas,
modo full, kernel de `.venv`). El registro (resultado, segundos y nº de celdas por notebook) queda en
`entrega/verificacion.json`.
