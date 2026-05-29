# PINN para Curvas Sintéticas de Registros de Pozo

Predice la curva de densidad **RHOB (DEN)** a partir de cinco registros de pozo convencionales usando dos arquitecturas comparadas bajo el protocolo **Leave-One-Well-Out (LOWO)**:

| Modelo | Descripción |
|--------|-------------|
| **Baseline MLP** | Red neuronal supervisada pura (MSE) |
| **PINN** | Mismo MLP + término físico `λ · L_física` que penaliza desviaciones de la relación DEN–NPHI |

La hipótesis central: embeber física en la función de pérdida mejora la generalización en **pozos ciegos** que el modelo nunca vio durante el entrenamiento.

---

## Problema

En exploración de petróleo y gas, la curva de densidad bulk (RHOB) a veces falla o no se registra. Este proyecto entrena un modelo que la reconstruye a partir de cinco registros que sí están disponibles:

| Entrada | Registro | Unidad |
|---------|----------|--------|
| GR | Gamma Ray | GAPI |
| RT | Resistividad profunda (RILD) | Ohm·m |
| RILM | Resistividad media | Ohm·m |
| NPHI | Neutrón compensado (CNLS) | v/v |
| SP | Potencial espontáneo | mV |

La **restricción física** se basa en la relación empírica lineal entre densidad bulk y porosidad neutrón, calibrada sobre el dataset:

```
DEN_esperada = a · NPHI + b
L_física = mean((DEN_pred - DEN_esperada)²)
L_total = L_MSE + λ · L_física
```

---

## Dataset

**Campo:** Kraft Prusa — Kansas Geological Survey  
**Pozos:** 25 wells con las 6 curvas requeridas  
**Registros totales:** ~157,000 puntos de profundidad  
**Profundidad:** ~200–2,600 ft (varía por pozo)

Los archivos LAS originales se descargan por separado y se colocan en `data/raw/` (no están versionados).

---

## Arquitectura

```
Entradas [5] → FC(64) → ReLU → FC(64) → ReLU → FC(32) → ReLU → FC(1)
```

- Optimizador: Adam, lr=1e-3  
- Normalización: min-max por pozo  
- Preprocesamiento: log₁₀(RT), log₁₀(RILM)  
- Evaluación: LOWO — cada pozo actúa como test una vez  
- Seed fija: 42

---

## Estructura del repositorio

```
src/                  Módulos de biblioteca
  data_loader.py      Carga LAS → DataFrames con nombres canónicos
  preprocessing.py    Log-RT, normalización min-max por pozo
  lowo.py             Generador de splits Leave-One-Well-Out
  dataset.py          WellDataset (torch.utils.data.Dataset)
  model.py            Clase MLP parametrizable
  train.py            Loop de entrenamiento con soporte λ_física
  evaluate.py         Métricas: MAE, RMSE, R², PE_90
  physics.py          Restricción física DEN–NPHI

tests/                Tests unitarios por módulo (pytest)
scripts/              Scripts reproducibles one-shot
  01_inspect_las.py   Inventario de curvas y calidad por pozo LAS
  02_run_eda.py       EDA: distribuciones, crossplots, calidad por pozo
  03_train_baseline.py  Entrena MLP en LOWO completo
  04_train_pinn.py    Entrena PINN con λ específico
  05_sweep_lambda.py  Sweep λ ∈ {0, 0.01, 0.05, 0.1, 0.5, 1.0}
  06_compare_baseline_vs_pinn.py  Comparación pareada por pozo
  07_plot_results.py  Figuras finales

data/raw/             LAS originales (gitignored)
data/processed/       Parquet por pozo (gitignored)
outputs/              Métricas, predicciones, figuras (gitignored)
documentation/        Reportes por fase en Markdown
```

---

## Instalación y uso (Docker)

**Requisitos:** Docker con NVIDIA Container Toolkit, GPU con CUDA 12.x

```bash
# Clonar el repositorio
git clone https://github.com/OilCoder/pinn-curvas-sinteticas.git
cd pinn-curvas-sinteticas

# Colocar los archivos LAS en data/raw/
# (descargar del Kansas Geological Survey)

# Construir la imagen
docker compose build

# Entrar al contenedor
docker compose run --rm dev

# Dentro del contenedor — pipeline completo
python scripts/02_run_eda.py
python scripts/03_train_baseline.py
python scripts/05_sweep_lambda.py
python scripts/06_compare_baseline_vs_pinn.py
python scripts/07_plot_results.py
```

---

## Evaluación

Protocolo de dos niveles para garantizar evaluación sin sesgo:

**Nivel 1 — Leave-One-Well-Out (LOWO)** sobre 22 pozos  
Cada fold entrena en N-1 pozos y predice en el restante. Sirve para comparar MLP vs PINN y seleccionar el mejor λ.

**Nivel 2 — External Validation Set** (3 pozos, seed=42)  
Separados antes de cualquier entrenamiento. Se usan únicamente para reportar el rendimiento final sin sesgo. Ningún modelo los vio durante LOWO ni durante el sweep de λ.

```python
train_pool, external = field_split(wells, n_external=3, seed=42)
# LOWO solo sobre train_pool (22 pozos)
# Evaluación final sobre external (3 pozos)
```

Métricas reportadas por fold y como media ± std:

| Métrica | Descripción |
|---------|-------------|
| MAE | Error absoluto medio (g/cc) |
| RMSE | Raíz del error cuadrático medio (g/cc) |
| R² | Coeficiente de determinación |
| PE_90 | Percentil 90 del error absoluto |

---

## Resultados

> _Resultados pendientes — se actualizan al completar la Fase 4._

---

## Tests y calidad de código

```bash
# Dentro del contenedor
pytest -q tests/
ruff check src/ tests/
mypy src/
```

---

## Licencia

MIT
