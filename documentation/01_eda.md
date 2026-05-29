# EDA — Campo Kraft Prusa

## Dataset cargado

| Métrica | Valor |
|---------|-------|
| Pozos válidos (`data_loader`) | 32 |
| Registros totales | 172,956 |
| Rango de profundidad | 0 – 3,483 ft |
| Paso de muestreo | 0.5 ft (mayoría) |

> Nota: el inventario inicial (`01_inspect_las.py`) reportó 25 pozos con filtro <10% NaN. El `data_loader` usa el filtro más permisivo de ≥100 filas limpias tras `dropna`, resultando en 32 pozos. Se conservan los 32 para maximizar datos de entrenamiento.

---

## Transformación logarítmica de resistividad (decisión: SÍ)

El histograma comparativo confirma que RT y RILM siguen distribución log-normal:

- **Distribución raw**: fuertemente asimétrica hacia valores bajos (0.3–2 Ohm·m), con cola larga hacia 500+ Ohm·m.
- **Distribución log₁₀**: aproximadamente simétrica, centrada en ~0.5–1.0.

**Decisión: aplicar log₁₀ a RT y RILM antes de normalización.** Esto mejora los gradientes durante el entrenamiento y elimina la dominancia de outliers de alta resistividad.

Implementado en `src/preprocessing.py → apply_log_rt()`.

---

## Normalización per-well min-max (decisión: SÍ)

**Motivación**: cada pozo tiene rangos de curvas distintos dependiendo de la litología local, la profundidad y el estado del hoyo. La normalización global penalizaría pozos con valores extremos y distorsionaría las distribuciones de pozos con rangos normales.

**Decisión: normalización min-max independiente por pozo**, aplicada después de log-RT.

| Alternativa | Razón de rechazo |
|-------------|-----------------|
| Z-score global | Pozos con outliers extremos dominan la media/std global |
| Z-score per-well | Min-max da rango garantizado [0,1], preferible para ReLU |
| Min-max global | Pierde variabilidad relativa intra-pozo |

Implementado en `src/preprocessing.py → WellScaler`.

---

## Relación DEN–NPHI: hallazgo crítico

### Correlación global vs per-well

| Contexto | R² | Pendiente |
|----------|----|-----------|
| Raw global (todos los pozos mezclados) | 0.017 | -0.0009 |
| Normalizado global (pooled en espacio [0,1]) | 0.050 | -0.2144 |
| Per-well mediana | **0.277** | negativa (31/32 pozos) |
| Per-well media | 0.298 | negativa (31/32 pozos) |

**Interpretación**: el R² global bajo es un artefacto del mezclado de pozos. Cada pozo tiene su propio nivel absoluto de DEN y NPHI dependiendo de la litología dominante. Al concatenar todos los pozos, las diferencias de offset entre ellos destruyen la correlación intra-pozo.

Dentro de cada pozo, la relación negativa es consistente y físicamente sensata: a mayor porosidad neutrón (NPHI), menor densidad bulk (DEN).

### Excepción: Dolecheck_1

Único pozo con pendiente positiva en espacio normalizado (+1.03). Probablemente refleja una sección con litología anómala o efecto de gas en el log neutrón (gas baja NPHI mientras también baja DEN, invirtiendo la relación normal). Registrado como caso de análisis en Phase 3.

### Distribución de R² per-well

```
R² ≥ 0.50  →  4 pozos  (relación fuerte)
R² ∈ [0.30, 0.50)  →  8 pozos  (relación moderada)
R² ∈ [0.10, 0.30)  → 18 pozos  (relación débil)
R² < 0.10  →  2 pozos  (sin relación o invertida)
```

---

## Coeficientes de la restricción física para el PINN

La restricción física del PINN opera en el **espacio normalizado** (valores en [0,1] tras log-RT y min-max per-well), que es donde el modelo produce sus predicciones.

**Regresión lineal en espacio normalizado pooled** (172,956 puntos de todos los pozos):

```
DEN_norm_esperada = -0.2144 · NPHI_norm + 0.8343
R² = 0.050
```

Guardados en `outputs/eda/den_nphi_coefficients_normalized.csv`.

**Limitaciones reconocidas**:
- R²=0.050 implica una restricción débil a nivel global
- Los coeficientes per-well varían significativamente (pendiente: -0.10 a -0.69)
- El sweep de λ (Phase 3) determinará empíricamente si este término mejora la generalización

**Estrategia**: usar estos coeficientes globales como prior suave en la función de pérdida. Si el PINN no mejora sobre el baseline con ningún λ, el resultado documenta los límites de esta restricción lineal para el campo Kraft Prusa.

---

## Estadísticas descriptivas de las curvas (espacio raw)

| Curva | Media | Std | Min | P25 | P50 | P75 | Max |
|-------|-------|-----|-----|-----|-----|-----|-----|
| GR (GAPI) | 51.3 | 25.6 | 0.0 | 32.1 | 46.1 | 65.7 | 267.0 |
| RT (Ohm·m) | 16.8 | 42.5 | 0.1 | 3.5 | 7.5 | 16.4 | 1995.5 |
| RILM (Ohm·m) | 14.5 | 32.9 | 0.2 | 3.5 | 7.4 | 15.2 | 1571.1 |
| NPHI (v/v) | 0.267 | 0.073 | -0.015 | 0.220 | 0.265 | 0.315 | 0.600 |
| SP (mV) | -31.8 | 22.7 | -121.5 | -47.7 | -33.0 | -16.8 | 90.5 |
| DEN (g/cc) | 2.419 | 0.121 | 1.690 | 2.340 | 2.422 | 2.502 | 2.900 |

---

## Archivos generados

| Archivo | Contenido |
|---------|-----------|
| `outputs/eda/distributions_raw.png` | Histogramas de las 6 curvas en escala raw |
| `outputs/eda/log_rt_comparison.png` | RT y RILM: distribución raw vs log₁₀ |
| `outputs/eda/den_nphi_crossplot.png` | Crossplot DEN vs NPHI global con fit lineal |
| `outputs/eda/den_nphi_by_well.png` | Crossplot DEN vs NPHI por pozo (small multiples) |
| `outputs/eda/profile_*.png` | Perfiles de profundidad de 4 pozos ejemplo |
| `outputs/eda/well_quality.csv` | Métricas de calidad por pozo |
| `outputs/eda/curve_statistics.csv` | Estadísticas descriptivas de las 6 curvas |
| `outputs/eda/den_nphi_coefficients.csv` | Coeficientes raw: a=-0.0009, b=2.2636, R²=0.017 |
| `data/processed/*.parquet` | DataFrames limpios por pozo (32 archivos) |

---

## Decisiones documentadas

| Decisión | Elección | Alternativa descartada |
|----------|---------|----------------------|
| Log-RT | Sí (log₁₀ a RT y RILM) | No aplicar |
| Normalización | Min-max per-well | Z-score global, Z-score per-well |
| Coeficientes física | Espacio normalizado: a=-0.2144, b=0.8343 | Espacio raw (R² irrelevante) |
| Pozos a usar | 32 (filtro ≥100 filas) | 25 (filtro <10% NaN) |

*Fuente del análisis: `scripts/02_run_eda.py`*  
*Fuente del código: `src/data_loader.py`, `src/preprocessing.py`*
