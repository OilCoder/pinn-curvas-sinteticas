# Documentación — PINN para Predicción de Densidad en Registros de Pozo

Reporte técnico del proyecto que predice la **densidad bulk** (DEN, g/cc) a partir de
cinco registros de pozo `[GR, RT, RILM, NPHI, SP]` mediante una **red neuronal informada
por física (PINN)**. La restricción física embebida es la relación empírica DEN–NPHI
calibrada en el campo Kraft Prusa, con corrección litológica por GR.

---

## Cómo leer esta documentación

Los documentos están numerados para lectura secuencial, como un tutorial que sigue el
flujo real del proyecto: de los datos crudos a los resultados finales.

| # | Documento | Contenido |
|---|---|---|
| 1 | [`00_dataset.md`](00_dataset.md) | Origen de los datos, campo Kraft Prusa, mapeo de curvas, restricción física |
| 2 | [`01_eda.md`](01_eda.md) | Análisis exploratorio: distribuciones, asimetría, calibración DEN–NPHI |
| 3 | [`02_preprocessing.md`](02_preprocessing.md) | Pipeline de limpieza: washout, outliers por consenso, normalización |
| 4 | [`03_baseline.md`](03_baseline.md) | Modelo base MLP, protocolo LOWO, resultados por pozo |
| 5 | [`04_pinn.md`](04_pinn.md) | Formulación del PINN, barrido de λ, comparación pareada |
| 6 | [`05_methodology.md`](05_methodology.md) | Síntesis del pipeline completo y decisiones de diseño |
| — | [`00_environment.md`](00_environment.md) | Configuración del entorno Docker (reproducibilidad) |

> **¿Primera vez?** Empieza por [`05_methodology.md`](05_methodology.md) para una visión
> general del pipeline completo, luego vuelve al documento 1 para el detalle.

---

## Resumen del proyecto

### Problema

La densidad bulk (DEN/RHOB) es una curva costosa de adquirir y frecuentemente ausente en
pozos antiguos. Predecirla a partir de registros más comunes permite completar conjuntos
de datos para caracterización de yacimientos. El reto: **generalizar a pozos no vistos**,
donde un modelo supervisado puro tiende a sobreajustar.

### Enfoque

$$\mathcal{L}_{total} = \mathcal{L}_{datos} + \lambda \cdot \mathcal{L}_{física}$$

El término físico restringe las predicciones hacia la relación geológica conocida
DEN–NPHI, actuando como regularizador. La restricción usa una corrección litológica
bivariate y se pondera por calidad de caliper (DCAL):

$$\hat{y}^{fís} = A \cdot \text{NPHI} + D \cdot (\text{NPHI} \times \text{GR})$$

### Resultados (LOWO, 27 pozos)

| Modelo | MAE (g/cc) | R² | Pozos con R² > 0 |
|---|---:|---:|---:|
| Baseline MLP (λ=0) | 0.1396 | 0.276 | 20/27 |
| **PINN (λ=0.5)** | **0.1347** | **0.327** | — |

La PINN con λ=0.5 mejora el **81.5 %** de los pozos (22/27) respecto al baseline. La
restricción física bivariate ponderada por caliper DCAL mejora monotónicamente hasta
λ≈0.5 y satura — el DCAL_WEIGHT desactiva la física en zonas de *washout*, lo que
permite usar λ alto sin degradación.

En los **3 pozos ciegos** (validación externa), el PINN mejora los 3, con
MAE 0.157→0.153 g/cc y R² 0.233→0.271.

---

## Reproducibilidad

Todo el pipeline corre dentro de Docker (ver [`00_environment.md`](00_environment.md)):

```bash
docker compose run --rm dev python scripts/02_run_eda.py            # EDA + figuras
docker compose run --rm dev python scripts/03_train_baseline.py     # baseline LOWO
docker compose run --rm dev python scripts/05_sweep_lambda.py       # barrido de λ
docker compose run --rm dev python scripts/06_compare_baseline_vs_pinn.py
docker compose run --rm dev python scripts/07_plot_results.py       # figuras finales
```

---

*Campo: Kraft Prusa, Barton County, Kansas (Kansas Geological Survey).*
*Stack: Python 3.11 · PyTorch · scikit-learn · lasio · matplotlib.*
