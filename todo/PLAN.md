# PINN para curvas sintéticas de registros de pozo

## Context

Proyecto de portafolio que implementa una Physics-Informed Neural Network para predecir RHOB (densidad bulk) a partir de [GR, RILD, RILM, CNLS, SP] usando la relación física RHOB-CNLS como término regularizador en la pérdida. El objetivo central es demostrar — con un benchmark riguroso Leave-One-Well-Out — que embeber física en la función de pérdida mejora la generalización en pozos ciegos respecto a un MLP supervisado puro.

Dataset confirmado: campo Kraft Prusa (Kansas Geological Survey), 25 pozos limpios, ~157k registros, profundidad en pies. Curvas reales: GR · RILD · RILM · CNLS/CNPOR · SP → RHOB. Arquitectura: 5→64→64→32→1.

El producto final es un repo público en GitHub OilCoder reproducible vía Docker (con CUDA para RTX 4080), sin notebooks, sin Hugging Face Spaces y sin código instalado en WSL.

Flujo de trabajo: autónomo por fase. Al terminar cada fase se presentan resultados y se espera revisión humana antes de avanzar.

## Goal

Entregar un repo público con baseline MLP y variante PINN entrenados y evaluados con protocolo LOWO sobre un dataset real de registros de pozo, comparados con métricas pareadas y documentados.

## Stack

| Capa | Tecnología |
|---|---|
| Lenguaje | Python 3.11 |
| Deep Learning | PyTorch 2.x + CUDA |
| Datos | lasio, pandas, numpy |
| Métricas y plots | scikit-learn, matplotlib |
| Test / Lint / Type | pytest, ruff, mypy |
| Entorno | Docker (imagen base CUDA) + docker-compose |
| Hardware | RTX 4080 16GB, WSL2 |
| Repo | GitHub OilCoder/pinn-curvas-sinteticas |

## Structure

```
src/                  Library: data_loader, dataset, preprocessing, lowo, model, train, evaluate, physics
tests/                pytest unit tests (uno por módulo de src/)
scripts/              Scripts one-shot: EDA, entrenamientos, sweeps, plots de resultados
data/raw/             LAS files originales (gitignored)
data/processed/       DataFrames limpios por pozo, parquet (gitignored)
outputs/              Métricas, predicciones, figuras, checkpoints (gitignored)
documentation/        Reportes por fase: 01_eda.md, 02_baseline.md, 03_pinn.md, 04_results.md
todo/                 PLAN.md + bitacora-YYYY-MM-DD.md
docs/                 Reservado para GitHub Pages (no se llena en MVP)
Dockerfile            Imagen CUDA + PyTorch
docker-compose.yml    Servicio dev con bind mount al proyecto
requirements.txt      Pin de dependencias
README.md             Overview, instalación Docker, uso, resultados
.gitignore            Existente; extender con data/, outputs/, *.pt
```

## Phases

### Phase 0 — Verificación de datos y entorno Docker

- [x] Confirmar ruta de LAS files con el usuario y crear `data/raw/` con un README que documente origen y conteo de pozos (2026-05-28)
- [x] Implementar `scripts/01_inspect_las.py` que liste por pozo: curvas presentes, unidades, rango de profundidad, % NaN por curva (2026-05-28)
- [x] Generar `outputs/eda/las_inventory.csv` con el resumen anterior (2026-05-28)
- [x] Filtrar pozos con las 4 curvas objetivo (GR, RT, NPHI, DEN o equivalentes) y gaps <10% (2026-05-28)
- [x] Documentar mapeo de nombres reales de curvas → nombres canónicos en `documentation/00_dataset.md` (2026-05-28)
- [x] Documentar unidades de profundidad y rangos en `documentation/00_dataset.md` (2026-05-28)
- ~~Generar `scripts/check_den_nphi_relation.py` que produce crossplot DEN vs NPHI por pozo y ajuste lineal~~ (discarded 2026-05-28: absorbido en scripts/02_run_eda.py)
- [x] Decidir si el dataset es viable: ≥8 pozos válidos y relación DEN-NPHI visible; si no, documentar pivote en `documentation/00_dataset.md` (2026-05-28)
- [x] Escribir `Dockerfile` basado en `pytorch/pytorch:2.x-cuda12.x-cudnn-runtime` con ruff/mypy/pytest/lasio/jupyter-less stack (2026-05-28)
- [x] Escribir `docker-compose.yml` con `runtime: nvidia`, bind mount al proyecto, working dir `/workspace` (2026-05-28)
- [x] Escribir `requirements.txt` con versiones pineadas (2026-05-28)
- [x] Verificar acceso a GPU desde el contenedor: `nvidia-smi` y `python -c "import torch; assert torch.cuda.is_available()"` (2026-05-28)
- [x] Escribir `documentation/00_environment.md` con instrucciones de build/run del contenedor (2026-05-28)

### Phase 1 — Pipeline de datos y EDA

- [x] Implementar `src/data_loader.py`: lectura de LAS con lasio, mapeo de curvas a nombres canónicos, retorno de DataFrame por pozo (2026-05-28)
- [x] Implementar `src/preprocessing.py`: limpieza de NaN, transformación logarítmica de RT (decisión registrada en docs), normalización por pozo (min-max vs z-score decidido aquí) (2026-05-28)
- [x] Implementar `src/lowo.py`: generador de splits LOWO + `field_split` para external validation set (n=3, seed=42) (2026-05-28)
- [x] Implementar `src/dataset.py`: `WellDataset(torch.utils.data.Dataset)` que toma DataFrames procesados y entrega tensores [GR, RT, RILM, NPHI, SP] → DEN (2026-05-28)
- [x] Implementar `tests/test_data_loader.py`: carga de un LAS dummy y validación de columnas (2026-05-28)
- [x] Implementar `tests/test_preprocessing.py`: cobertura de NaN, log-RT, normalización per-well (2026-05-28)
- [x] Implementar `tests/test_lowo.py`: verifica LOWO (test exactamente una vez, sin leakage) + field_split (sizes, sin overlap, reproducibilidad) (2026-05-28)
- [x] Implementar `tests/test_dataset.py`: shapes, dtypes, longitud (2026-05-28)
- [x] Implementar `scripts/02_run_eda.py`: distribuciones por curva, crossplots, calidad por pozo; figuras a `outputs/eda/` (2026-05-28)
- [x] Exportar DataFrames procesados a `data/processed/{well_id}.parquet` (2026-05-28)
- [x] Escribir `documentation/01_eda.md` con hallazgos y decisiones (log-RT sí/no, normalización elegida y por qué) (2026-05-28)
- [x] Verificación gate: `pytest -q tests/`, `ruff check src/ tests/`, `mypy src/` (2026-05-28)

### Phase 2 — Modelo base (MLP supervisado)

- [ ] Implementar `src/model.py`: clase `MLP` parametrizable (hidden_dims, dropout opcional), defaults 5→64→64→32→1 ReLU
- [ ] Implementar `src/train.py`: loop con seed fija (42), Adam, MSE, early stopping opcional, checkpoint del mejor modelo a `outputs/checkpoints/`
- [ ] Implementar `src/evaluate.py`: cálculo de MAE, RMSE, R², PE_90; salida a dict serializable
- [ ] Implementar `tests/test_model.py`: forward pass con shapes esperados
- [ ] Implementar `tests/test_evaluate.py`: métricas contra valores conocidos
- [ ] Implementar `tests/test_train.py`: smoke test que entrena 2 épocas sobre datos sintéticos sin crash
- [ ] Implementar `scripts/03_train_baseline.py`: ejecuta LOWO completo, guarda métricas por fold en `outputs/baseline/metrics.json` y predicciones por pozo en `outputs/baseline/predictions/{well_id}.parquet`
- [ ] Escribir `documentation/02_baseline.md` con tabla de métricas por pozo, agregados (media ± std) y discusión
- [ ] Verificación gate: `pytest -q tests/`, `ruff check src/ tests/`, `mypy src/`

### Phase 3 — PINN y barrido de λ

- [ ] Implementar `src/physics.py`: función `den_from_nphi(nphi) -> den_expected` (relación lineal a calibrar en Fase 0) y `physics_loss(den_pred, nphi_obs) -> tensor`
- [ ] Implementar `tests/test_physics.py`: verifica `den_from_nphi` con valores tabulados, gradiente del loss físico, comportamiento con tensor batch
- [ ] Extender `src/train.py` con parámetro `lambda_phys` que añade `λ · physics_loss` al MSE
- [ ] Implementar `scripts/04_train_pinn.py`: una corrida LOWO con un λ específico, guarda a `outputs/pinn/lambda_{λ}/`
- [ ] Implementar `scripts/05_sweep_lambda.py`: ejecuta sweep sobre λ ∈ {0.0, 0.01, 0.05, 0.1, 0.5, 1.0}, agrega métricas a `outputs/pinn/lambda_sweep.json`
- [ ] Verificar control pareado: λ=0 reproduce baseline dentro de tolerancia numérica (test en `tests/test_train.py`)
- [ ] Escribir `documentation/03_pinn.md` con curva tradeoff λ vs error, análisis por pozo, discusión de límites de la relación lineal en zonas arcillosas
- [ ] Verificación gate: `pytest -q tests/`, `ruff check src/ tests/`, `mypy src/`

### Phase 4 — Análisis final, README y publicación

- [ ] Implementar `scripts/06_compare_baseline_vs_pinn.py`: comparación pareada por pozo (Δ MAE, signo, significancia), salida a `outputs/figures/comparison_table.csv`
- [ ] Implementar `scripts/07_plot_results.py`: curvas DEN real vs base vs PINN en profundidad, crossplot con línea física, λ vs error agregado; figuras a `outputs/figures/`
- [ ] Escribir `documentation/04_results.md` con análisis cualitativo, escenarios A/B/C y conclusiones
- [ ] Escribir `README.md`: descripción del problema, instalación vía Docker, comando de reproducción end-to-end, resumen de resultados con figura clave embebida
- [ ] Crear repo público `OilCoder/pinn-curvas-sinteticas` en GitHub y configurar remote
- [ ] Push inicial con tag de release `v0.1.0`
- [ ] Verificar reproducibilidad: clonar en directorio limpio, `docker compose up`, ejecutar `scripts/03_train_baseline.py` y `scripts/05_sweep_lambda.py` sin intervención
- [ ] Verificación gate final: `pytest -q tests/`, `ruff check src/ tests/`, `mypy src/`, build de Docker exitoso

## Conventions

- Código y docstrings en inglés; PLAN.md, bitácora y documentation/*.md en español
- Una tarea por checkbox, plano sin sub-items; cada tarea menciona el archivo/módulo que toca
- Scripts en `scripts/` son one-shot reproducibles; la lógica reusable vive en `src/`
- Todo entrenamiento usa seed=42 y guarda métricas a JSON para análisis downstream
- Sin notebooks: cualquier análisis va a `scripts/*.py` + reporte en `documentation/*.md`
- Sin instalaciones en WSL: todo dentro del contenedor Docker
- Protocolo LOWO en todos los benchmarks: nunca splits aleatorios cross-well
- Fases secuenciales: cada una asume completados los gates de verificación de la anterior
- Al cierre de cada fase: actualizar `documentation/`, marcar tareas `[x]`, ejecutar `/checkpoint` para commit + bitácora

## Verification

Cada fase tiene su propio verification gate definido por `project-guidelines.md`:

```bash
pytest -q tests/
ruff check src/ tests/
mypy src/
ruff format src/ tests/
```

Verificación end-to-end del producto final (Fase 4):

```bash
docker compose build
docker compose run --rm dev pytest -q tests/
docker compose run --rm dev python scripts/02_run_eda.py
docker compose run --rm dev python scripts/03_train_baseline.py
docker compose run --rm dev python scripts/05_sweep_lambda.py
docker compose run --rm dev python scripts/06_compare_baseline_vs_pinn.py
docker compose run --rm dev python scripts/07_plot_results.py
```

Una corrida limpia desde repo clonado debe producir todas las figuras y métricas sin intervención manual.

## Open items deferred to execution

- Coeficientes finales de `RHOB = a · CNLS + b` (calibrar con regresión lineal en Fase 0 sobre Kraft Prusa)
- Normalización per-well: min-max vs z-score (decidir en Fase 1 con EDA)
- Transformación log de RILD (decidir en Fase 1 con histograma)
