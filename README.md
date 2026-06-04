# PINN para curvas sintéticas de registros de pozo

Predice la curva de densidad **RHOB (DEN)** a partir de cinco registros de pozo convencionales, comparando dos arquitecturas bajo el protocolo **Leave-One-Well-Out (LOWO)**:

| Modelo | Descripción |
|--------|-------------|
| **Baseline MLP** | Red neuronal supervisada pura (MSE) |
| **PINN** | Mismo MLP + término físico `λ · L_física` que penaliza desviaciones de la relación DEN–NPHI |

La hipótesis central: embeber física en la función de pérdida mejora la generalización en **pozos ciegos** que el modelo nunca vio durante el entrenamiento — el punto débil conocido de los modelos de machine learning puros en geociencias.

> 🌐 **Sitio del proyecto (informe interactivo):** <https://oilcoder.github.io/pinn-curvas-sinteticas/> · [English](https://oilcoder.github.io/pinn-curvas-sinteticas/index.en.html)
> 📄 **Reporte técnico completo:** [`documentation/`](documentation/) (capítulos 00–06)
> 📚 **Paper de referencia:** Pothana & Ling (2025), *Energy Geoscience* — [doi.org/10.1016/j.engeos.2025.100410](https://doi.org/10.1016/j.engeos.2025.100410)

---

## Problema

En evaluación de formaciones, la curva de densidad bulk (RHOB) a veces falta o viene dañada (p. ej. por derrumbe del pozo). Este proyecto entrena un modelo que la reconstruye a partir de cinco registros que sí están disponibles:

| Entrada | Registro | Unidad |
|---------|----------|--------|
| GR | Gamma Ray | GAPI |
| RT | Resistividad profunda (RILD) | Ohm·m |
| RILM | Resistividad media | Ohm·m |
| NPHI | Neutrón compensado (CNLS) | v/v |
| SP | Potencial espontáneo | mV |

La **restricción física** se basa en la relación densidad–neutrón, calibrada por regresión sobre los 27 pozos de entrenamiento en el espacio normalizado (bivariada, con un término de interacción NPHI×GR que captura el efecto de la arcillosidad):

```
DEN_esperada = A · NPHI + D · (NPHI × GR)        # A = −0.556, D = 0.086, R² = 0.338
L_física     = mean( w · (DEN_pred − DEN_esperada)² )   # w = peso de caliper (DCAL): → 0 en washout
L_total      = L_datos + λ · L_física
```

El peso `w` (DCAL_WEIGHT) apaga la física donde el pozo está derrumbado y la lectura es poco fiable, lo que permite usar un λ alto sin degradar los pozos difíciles. `λ = 0` reproduce exactamente el baseline (ablación controlada).

---

## Dataset

**Campo:** Kraft Prusa — Kansas Geological Survey (KGS)
**Pozos:** 30 pozos limpios (27 train pool + 3 externos ciegos)
**Registros:** ~154,000 puntos de profundidad
**Profundidad:** en pies (varía por pozo)

Los archivos LAS originales se descargan por separado del KGS y se colocan en `data/raw/` (no versionados).

---

## Arquitectura y preprocesamiento

```
Entradas [5] → FC(64) → ReLU → FC(64) → ReLU → FC(32) → ReLU → FC(1)
```

- Optimizador: Adam, lr = 1e-3
- Normalización **per-pozo y per-columna**: Yeo-Johnson para curvas sesgadas (GR, SP, DEN), z-score para las simétricas (RT, RILM, NPHI)
- Preprocesamiento: filtro de washout por caliper DCAL, outliers por consenso de 5 detectores (≥2 → interpolación), `log₁₀(RT)`/`log₁₀(RILM)`, filtro de DEN a su rango físico [1.5, 3.1] g/cc
- Evaluación: LOWO sobre los 27 pozos + 3 pozos externos ciegos
- Seed fija: 42 (sin fuga entre pozos: cada pozo se normaliza con su propio scaler)

---

## Estructura del repositorio

```
src/                  Módulos de biblioteca
  data_loader.py      Carga LAS → DataFrames con nombres canónicos
  preprocessing.py    Washout, outliers, log-RT, normalización per-pozo, DCAL_WEIGHT
  lowo.py             Splits Leave-One-Well-Out + field_split (set externo)
  dataset.py          WellDataset (torch.utils.data.Dataset)
  model.py            Clase MLP parametrizable
  train.py            Loop de entrenamiento GPU-resident con soporte λ_física
  evaluate.py         Métricas: MAE, RMSE, R², PE_90
  physics.py          Restricción física DEN–NPHI bivariada (con tests propios)
  external_eval.py    Inferencia externa: ensemble LOWO y modelo único

tests/                Tests unitarios por módulo (pytest)
scripts/              Scripts reproducibles (orden de ejecución por prefijo)
  01_inspect_las.py · 02_run_eda.py
  03_train_baseline.py · 04_train_pinn.py · 05_sweep_lambda.py
  06_compare_baseline_vs_pinn.py · 07_plot_results.py
  08_eval_external.py · 09_plot_diagnostics.py · 10_external_validation.py
  run_parallel_sweep.sh   Sweep paralelo (3 jobs concurrentes, GPU-resident)

data/raw/             LAS originales (gitignored)
data/processed/       Parquet por pozo (gitignored)
outputs/              Métricas, predicciones, figuras (gitignored)
documentation/        Reporte técnico por capítulos (00–06)
docs/                 Sitio del proyecto (GitHub Pages, ES + EN)
```

---

## Instalación y uso (Docker)

**Requisitos:** Docker con NVIDIA Container Toolkit, GPU con CUDA 12.x

```bash
git clone https://github.com/OilCoder/pinn-curvas-sinteticas.git
cd pinn-curvas-sinteticas

# Colocar los archivos LAS del KGS (campo Kraft Prusa) en data/raw/

docker compose build
docker compose run --rm dev          # entrar al contenedor

# Dentro del contenedor — pipeline completo
python scripts/02_run_eda.py
python scripts/03_train_baseline.py
python scripts/05_sweep_lambda.py            # o scripts/run_parallel_sweep.sh
python scripts/06_compare_baseline_vs_pinn.py
python scripts/10_external_validation.py     # validación en los 3 pozos ciegos
```

---

## Evaluación

Protocolo de dos niveles para una evaluación sin sesgo:

**Nivel 1 — Leave-One-Well-Out (LOWO)** sobre los **27 pozos** del train pool.
Cada fold entrena en 26 pozos y predice en el restante. Sirve para comparar MLP vs PINN y seleccionar el mejor λ.

**Nivel 2 — Set externo ciego** (3 pozos, seed=42), separados antes de cualquier entrenamiento. Ningún modelo los vio durante LOWO ni el sweep de λ. Se evalúan bajo dos protocolos (ensemble de los 27 modelos LOWO y modelo único entrenado en los 27) como análisis de robustez.

```python
train_pool, external = field_split(wells, n_external=3, seed=42)
# LOWO solo sobre train_pool (27 pozos); evaluación final sobre external (3 pozos)
```

Métricas reportadas por fold y como media ± std: **MAE** y **RMSE** (g/cc), **R²**, **PE_90** (percentil 90 del error absoluto).

---

## Resultados

**LOWO (27 pozos), λ óptimo = 0.5** — la física mejora **22 de 27** pozos:

| | Baseline (λ=0) | PINN (λ=0.5) |
|---|---|---|
| MAE (g/cc) | 0.140 | **0.135** |
| R² | 0.276 | **0.327** |

**Pozos ciegos (3 externos), dos protocolos de inferencia** — el PINN mejora bajo ambos:

| Protocolo | MAE (base → PINN) | R² (base → PINN) |
|---|---|---|
| Ensemble (27 modelos LOWO) | 0.157 → **0.153** | 0.233 → **0.271** |
| Modelo único (27 pozos) | 0.162 → **0.155** | 0.171 → **0.257** |

El PINN mejora los **3 pozos ciegos** con las dos formas de evaluar, así que la ventaja física es real y no un artefacto del método de agregación. Análisis completo (incluido el diagnóstico in-sample vs out-of-sample) en [`documentation/06_resultados.md`](documentation/06_resultados.md) y en el [sitio del proyecto](https://oilcoder.github.io/pinn-curvas-sinteticas/).

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
