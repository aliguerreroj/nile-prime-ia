# Estándares de ingeniería (nivel AI Engineer) — OBLIGATORIOS

Reglas transversales para todo modelo del proyecto. El implementador las aplica, el revisor
las comprueba (su incumplimiento es FAIL) y `validate.py` verifica el esquema de los JSON de
métricas. Los criterios concretos de cada reto están en `harness/tasks.json`.

## 1. Baseline y significancia
- Todo modelo se compara contra un **baseline** con los mismos datos y la misma métrica.
- La mejora se **valida estadísticamente**, según el caso:
  - *bootstrap* con IC 95 % y ≥ 1.000 remuestreos (métrica sobre un conjunto de test);
  - test pareado (p. ej. por usuario, Wilcoxon o t pareado) cuando ambos modelos se evalúan
    sobre las mismas unidades;
  - prueba de permutación cuando la pregunta es si existe señal.
- "Mejor" significa que el IC o el p-valor lo respaldan; si no, se dice "sin diferencia
  significativa".

## 2. Particiones: el test se usa UNA vez
- Tres roles: **train** (ajustar), **validación o CV** (elegir hiperparámetros, umbrales,
  pesos y modelo) y **test** (medir, al final).
- **Nunca tuning sobre test**: ni hiperparámetros, ni umbral, ni selección de modelo, ni peso α.
- Toda transformación que aprende de los datos (vectorizador, escalador, target encoding) se
  ajusta solo con train, dentro de un `Pipeline` o dentro de cada fold.
- Con datos agrupados (usuario), las particiones respetan el grupo.

## 3. Búsqueda de hiperparámetros
- Con CV: `RandomizedSearchCV` o `GridSearchCV`, con el **espacio de búsqueda documentado** en
  el notebook y guardado en `config`.
- En modo FAST (`NP_FAST=1`) se usa un espacio reducido y menos iteraciones; el flujo es el mismo.

## 4. Umbrales de decisión
- **Nunca 0,5 por defecto.** El umbral se elige **en validación** por:
  - costo de negocio documentado (matriz de costos FP/FN con su supuesto explícito), o
  - F-beta con el beta justificado.
- Si tras ese análisis el umbral resulta 0,5, se documenta por qué (p. ej. clases balanceadas
  y costos simétricos). Lo que se prohíbe es no justificarlo.

## 5. Incertidumbre
- Toda métrica final lleva **IC 95 %**.
- En procesos estocásticos (RL, entrenamiento con inicialización aleatoria): **≥ 5 semillas**,
  reportando media ± desviación estándar.

## 6. Calibración
- En clasificadores probabilísticos: **Brier score** y **curva de calibración**.

## 7. Costo operativo
- Tiempo de entrenamiento, **latencia de inferencia por ítem** y **tamaño del modelo** en
  disco o en memoria. Forman parte de la decisión de qué modelo elegir.

## 8. Análisis de errores
- Cualitativo: ejemplos concretos mal clasificados o mal recomendados, con una lectura de por
  qué falla el modelo.

## 9. Esquema de `reports/metrics/<reto>.json`
Retos: `sentimiento`, `recomendador`, `abandono`, `rl`. Claves que `validate.py` exige:

| Clave | Contenido |
|---|---|
| `modo` | `"fast"` o `"full"` |
| `config` | objeto: hiperparámetros (y espacio de búsqueda), semilla(s), versiones de librerías |
| `baseline` | objeto: nombre del baseline y sus métricas |
| `metricas` | objeto: métricas finales, cada una con su valor y su IC 95 % |
| `umbral` | objeto con `valor` y `justificacion`. Obligatorio en `sentimiento` y `abandono`; en los demás, solo si aplica |

Ejemplo mínimo:
```json
{
  "modo": "full",
  "config": {"semilla": 42, "hiperparametros": {"C": 1.0}, "versiones": {"scikit-learn": "1.7.2"}},
  "baseline": {"nombre": "DummyClassifier", "metricas": {"f1": 0.5}},
  "metricas": {"f1": {"valor": 0.9, "ic95": [0.89, 0.91]}},
  "umbral": {"valor": 0.5, "justificacion": "clases balanceadas y costos simétricos"}
}
```
`eda.json` y `resumen.json` no son JSON de reto: solo necesitan `modo`.

## 10. Ficha del modelo
Al final de cada notebook de reto, una **ficha del modelo** (model card) breve: propósito,
datos, métricas (con IC), limitaciones y riesgos.

## 11. Prohibido maquillar
Si una meta no se alcanza, **se reporta tal cual y se explica**. No se cambian la métrica, la
partición ni el umbral para que el número salga mejor, y no se omiten resultados negativos.

## 12. Política de plazo
Entrega el 2026-10-11. Confirmada por Ali el 2026-10-10.

**Umbral por reto**
- `sentimiento` y `abandono`: `umbral` obligatorio (valor y justificación).
- `recomendador`: no hay umbral; se declara `K = 10` en `config` con su justificación.
- `rl`: no aplica.

**Nunca se relaja**
- Notebooks sin errores.
- Sin fuga de datos.
- Baseline.
- IC en las métricas principales.
- Umbral justificado.
- Ficha del modelo.
- Informe PDF.

**Presupuesto por tarea**

| Tarea | T01 | T02 | T03 | T04 | T05 | T06 |
|---|---|---|---|---|---|---|
| Presupuesto | 30 min | 1 h | 1,5 h | 2 h | 1,5 h | 2 h |

**Escalera de relajación** — solo si una tarea supera su presupuesto en un 50 %, y en este orden:
1. Omitir el DQN.
2. Bootstrap de 1.000 a 500 remuestreos.
3. `n_iter` de la búsqueda de hiperparámetros de 20 a 10.
4. Análisis de errores de 10 a 5 ejemplos.
5. Diversidad y cobertura del recomendador pasan a métricas secundarias.

Cada escalón usado se anota en `progress/decisions.md` y se menciona en el informe como
limitación.

## 13. Audiencia
- **Notebooks y código: audiencia técnica** (profesor, AI Engineer, data scientist). Markdown
  técnico, preciso y conciso: objetivo, supuestos, decisiones de diseño y trade-offs, métricas
  con su interpretación, riesgos (fuga, desbalance, sesgo) y limitaciones. Terminología
  correcta; no se simplifican los conceptos.
- Cada sección cierra con una **Conclusión técnica** y una línea **Implicación de negocio**
  (una frase, que alimenta el informe).
- **Informe PDF (T09): único entregable para público no técnico.**
