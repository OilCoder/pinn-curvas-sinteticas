# 8. Modelos alternativos — XGBoost y LSTM físico-informados

Extensión del método físico-informado a otras dos arquitecturas, a raíz de una sugerencia
externa (comentario de José Emiliano Flores Pérez en LinkedIn): probar la idea en **gradient
boosting** y en una **red secuencial**, y dar al modelo una señal del estado del pozo. La
pregunta de este capítulo no es *cuál arquitectura gana*, sino si el término físico que ayudó
al MLP **se transfiere** a XGBoost y a una LSTM. El MLP/PINN de los capítulos anteriores queda
como **referencia**, no como rival.

Todo corre sobre el **mismo protocolo** del informe: LOWO de 27 pozos, los mismos 3 pozos ciegos
(`field_split(n_external=3, seed=42)`), normalización por pozo sin fuga y métricas en g/cc vía
`src/evaluate.py`.

---

## 8.1 Invariante del experimento

En las tres arquitecturas, **$\lambda = 0$ reproduce exactamente el modelo plano** (solo datos).
El término físico es un *añadido* sobre ese modelo, nunca una arquitectura distinta. Así, la
mejora de $\lambda = 0$ a $\lambda > 0$ aísla lo que aporta la física, sin nada más cambiando.

La física reutiliza la misma relación calibrada del PINN (`src/physics.py`):

$$\hat{y}^{fís} = A \cdot \text{NPHI} + D \cdot (\text{NPHI} \times \text{GR}), \quad A = -0.556,\; D = 0.086$$

ponderada por el peso de caliper $w$ (apaga la física en *washout*), igual que el `DCAL_WEIGHT`
del PINN.

---

## 8.2 El caliper como entrada (decisión de diseño)

El comentario sugería un *flag de bad hole* (marca binaria de derrumbe). Un flag obliga a fijar
a mano un umbral de diámetro para decidir qué cuenta como *washout*, y ese umbral es discutible.
La alternativa adoptada es más práctica: **introducir el caliper diferencial como sexta entrada
continua** (`DCAL_NORM`, z-score por pozo) y dejar que el modelo aprenda solo dónde el agujero
ensanchado degrada la fiabilidad de la densidad. Misma señal, sin umbral inventado.

Implementación: `src/features_caliper.py` — `preprocess_well_with_caliper()` envuelve el
preprocesamiento estándar añadiendo `DCAL_NORM` alineado a las filas sobrevivientes;
`FEATURE_COLS_EXT = FEATURE_COLS + ["DCAL_NORM"]`; `build_arrays()` arma `(x, y, nphi, gr, w, depth)`.
El MLP/PINN publicado **no se toca** (sigue con 5 entradas); la extensión es aditiva.

---

## 8.3 El término físico por arquitectura

El término físico entra en cada modelo según cómo aprende:

- **LSTM** (`src/model_lstm.py`): como el MLP, se entrena minimizando una función de coste por
  descenso de gradiente; el término físico es un sumando más y el entrenamiento lo deriva solo
  (`loss = MSE + λ · physics_loss`). Red de 1 capa, `hidden=64`, ventanas de 32 muestras en
  profundidad **sin cruzar fronteras de pozo** (`build_sequences`).

- **XGBoost** (`src/model_trees.py`): no minimiza una función de coste con autograd, sino que
  construye árboles greedily y solo necesita el gradiente y el hessiano del error respecto a la
  predicción. No se le puede "entregar" una loss. La solución es un **objetivo a medida** con las
  derivadas del error combinado datos + física calculadas analíticamente:

$$g = (\hat{y} - y) + \lambda\, w\, (\hat{y} - \hat{y}^{fís}), \qquad h = 1 + \lambda\, w$$

  Como $\hat{y}^{fís}$ es fijo por muestra, las derivadas son sencillas y $h$ se mantiene positivo
  (entrenamiento estable). Con $\lambda = 0$ el objetivo colapsa a `reg:squarederror` exacto.
  Árboles modestos: `max_depth=5`, `eta=0.05`, 400 rondas.

Ambos modelos son **deliberadamente modestos**, comparables en robustez al MLP `5→64→64→32→1`.

---

## 8.4 El λ óptimo depende de la arquitectura

No se asumió que el $\lambda$ óptimo del MLP (0.5) sirviera para las demás arquitecturas: se
optimizó $\lambda$ **por arquitectura**, probando un rango de valores. Resultó importar mucho.

| Modelo | MAE λ=0 → mejor λ | R² λ=0 → mejor λ | λ óptimo |
|---|---|---|---:|
| MLP / PINN *(referencia)* | 0.1396 → **0.1347** | 0.276 → **0.327** | 0.5 |
| XGBoost | 0.1286 → **0.1284** | 0.377 → 0.376 | 0.25 |
| LSTM | 0.1484 → **0.1299** | 0.168 → **0.337** | 2.0 |

Tres comportamientos distintos, los tres con sentido:

- **XGBoost — curva plana.** Ya parte del mejor MAE de los tres sin física; los árboles exprimen
  la estructura tabular y la física añade poca información nueva. No empeora; no hace falta forzarla.
- **LSTM — mejora sostenida, la que más gana.** Sin física es la peor (R² 0.168); la física la
  lleva a la par y sigue mejorando hasta $\lambda = 2$ (cuatro veces el del MLP). Una red secuencial
  con pocos pozos se ajusta a las secuencias vistas y generaliza mal (mucha varianza); el término
  físico le impone una relación que se cumple en cualquier pozo, dándole el marco que los datos
  escasos no le dan.
- **MLP — punto intermedio**, ya conocido (capítulo 5), estable cerca de $\lambda = 0.5$.

**Conclusión:** no existe un $\lambda$ universal. Cuanto más le cuesta el problema a la arquitectura
por su cuenta, más peso físico admite y más gana.

---

## 8.5 Validación externa — 3 pozos ciegos

Igual que el PINN, los 3 pozos ciegos se predicen bajo **dos protocolos** (ensemble de 27 modelos
LOWO y modelo único entrenado en 27 pozos), como control de robustez.

| Modelo · protocolo | MAE λ=0 → mejor λ | R² λ=0 → mejor λ |
|---|---|---|
| XGBoost · ensemble | 0.1510 → **0.1501** | 0.285 → **0.290** |
| XGBoost · único | 0.1514 → **0.1509** | 0.282 → **0.285** |
| LSTM · ensemble | 0.1519 → **0.1475** | 0.262 → **0.298** |
| LSTM · único | 0.1754 → **0.1500** | 0.040 → **0.287** |
| MLP/PINN · ensemble *(ref.)* | 0.157 → 0.153 | 0.233 → 0.271 |
| MLP/PINN · único *(ref.)* | 0.162 → 0.155 | 0.171 → 0.257 |

El caso más elocuente es la **LSTM con modelo único**: sin física apenas generaliza (R² 0.040) y
con física salta a 0.287, al nivel del resto. La física no le da más potencia, le da una brújula
cuando los datos de un pozo nuevo no alcanzan. En XGBoost la mejora externa es pequeña pero
consistente, en línea con su curva plana.

---

## 8.6 ¿Sirvió el caliper?

XGBoost permite leer la importancia de cada entrada (gain medio sobre los 27 modelos). El orden:

| Entrada | Rol |
|---|---|
| **NPHI** | Domina — es el predictor físico de la densidad |
| **RILM** | Segunda — una de las resistividades |
| **DCAL_NORM** *(caliper)* | Tercera — por delante de RT, GR y SP |
| RT, GR, SP | Resto |

El caliper entra con un peso modesto pero real: ni domina ni es decorativo. El modelo lo usa para
ajustar su confianza donde el agujero está ensanchado, que es exactamente lo que se buscaba.

---

## 8.7 Conclusiones

1. **El método físico-informado se transfiere.** No era un truco del MLP: en las tres arquitecturas
   $\lambda > 0$ iguala o mejora a $\lambda = 0$, sin empeorar ningún caso. Es una idea de método.
2. **El $\lambda$ óptimo es propio de cada arquitectura** y se correlaciona con cuánto le cuesta el
   problema sola: plano en XGBoost (0.25), intermedio en el MLP (0.5), alto y muy rentable en la
   LSTM (2.0).
3. **El caliper como entrada aporta** contexto de calidad del dato que el modelo usa de verdad, sin
   desplazar a las señales físicas.
4. **Honestidad sobre el alcance.** Las ganancias externas son modestas en absoluto —generalizar a
   un pozo nuevo con pocos pozos de entrenamiento sigue siendo difícil—, pero la dirección es
   consistente en todo el experimento. Modelar la incertidumbre de forma explícita (predicción con
   su confianza) queda como paso siguiente.

---

## 8.8 Reproducibilidad

```bash
# XGBoost físico-informado: LOWO + barrido de λ, y validación externa
docker compose run --rm dev python scripts/11_train_xgboost.py --lambdas 0,0.1,0.25,0.5,1.0
docker compose run --rm dev python scripts/12_eval_external_xgboost.py --lambdas 0.0,0.25

# LSTM físico-informada: LOWO + barrido de λ, y validación externa
docker compose run --rm dev python scripts/13_train_lstm.py --lambdas 0,0.1,0.25,0.5,1.0,1.5,2.0
docker compose run --rm dev python scripts/14_eval_external_lstm.py --lambdas 0,2.0

# Figuras del tab "Modelos alternativos" del sitio
docker compose run --rm dev python scripts/15_plot_alternatives.py
```

Módulos nuevos: `src/features_caliper.py`, `src/model_trees.py`, `src/model_lstm.py` (con sus tests
en `tests/`). Resultados crudos en `outputs/experiments/{xgboost,lstm}/`. La narrativa completa,
bilingüe, está en el tab **Modelos alternativos** del sitio (`docs/baselines.html`).

---

*Campo: Kraft Prusa, Barton County, Kansas (Kansas Geological Survey).*
*Stack: Python 3.11 · PyTorch · XGBoost · scikit-learn · lasio · matplotlib.*
