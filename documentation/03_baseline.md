# Modelo Base MLP — Evaluación LOWO

Establece el rendimiento de referencia del MLP supervisado puro (λ=0) sobre el campo Kraft Prusa
mediante validación cruzada Leave-One-Well-Out. Toda métrica de Phase 3 (PINN) se compara contra
estos valores.

---

## Protocolo de evaluación

### Leave-One-Well-Out (LOWO)

```
30 pozos totales (2 duplicados eliminados: Weber_'A'_13_part1, Frees-Burmeister_13_part1)
├── 3 pozos externos (field_split, seed=42) — reservados para Phase 4
│   Arensman_2 · Burmeister_1 · Rous_'F'_2
└── 27 pozos train pool — usados en LOWO
    └── 27 folds × (26 pozos train | 1 pozo test)
```

Cada fold es completamente independiente:

1. Los 26 pozos de entrenamiento se preprocesan individualmente (cada uno con su propio `WellScaler`).
2. El pozo de prueba se preprocesa con su propio `WellScaler` — sin información cruzada del conjunto de entrenamiento.
3. El modelo se inicializa desde cero con `set_seed(42)`.
4. Las predicciones se invierten a g/cc antes de calcular métricas.

Esta estrategia garantiza que no hay data leakage entre pozos en ninguna etapa.

**Fuentes:** `src/lowo.py` · `scripts/03_train_baseline.py`

---

## Preprocesamiento de outliers

Los outliers se detectan por **consenso de votación** con 5 detectores independientes:

| Detector | Escala | Criterio |
|---|---|---|
| MAD (threshold=3.5) | lineal / log₁₀ para RT, RILM | desviación respecto a mediana |
| IQR (k=3.0) | lineal / log₁₀ | fuera de [Q1−k·IQR, Q3+k·IQR] |
| Z-score (threshold=3.0) | lineal / log₁₀ | más de 3σ del promedio |
| Percentil (p1.5–p98.5) | lineal / log₁₀ | fuera del rango [1.5%, 98.5%] |
| IsolationForest (contamination=0.05) | lineal / log₁₀ | anomalía según árbol de aislamiento |

Un punto se marca como outlier si **≥ 2 detectores** coinciden. Outliers → NaN → interpolación lineal
(límite 5 pasos) + ffill + bfill. Evaluación por pozo: sin cruces entre pozos.

RT y RILM se evalúan en escala logarítmica (distribución log-normal); el resto en escala lineal.

**Fuente:** `src/preprocessing.py` · `flag_outliers_consensus()`

---

## Arquitectura del modelo

```
Entrada: [GR_norm, log10(RT)_norm, RILM_norm, NPHI_norm, SP_norm]  → dim 5
Capa 1:  Linear(5, 64)  + ReLU
Capa 2:  Linear(64, 64) + ReLU
Capa 3:  Linear(64, 32) + ReLU
Salida:  Linear(32, 1)              → DEN normalizado (sin activación)
```

Salida lineal: el modelo puede predecir cualquier valor real; la inversión `WellScaler` lo devuelve a g/cc.

**Fuente:** `src/model.py`

---

## Configuración de entrenamiento

| Parámetro | Valor |
|---|---|
| Optimizador | Adam |
| Tasa de aprendizaje | 1 × 10⁻³ |
| Loss | MSE (espacio normalizado) |
| Épocas máximas | 500 |
| Early stopping patience | 30 épocas |
| min_delta | 1 × 10⁻⁵ |
| Fracción validación interna | 15 % |
| Batch size | 256 |
| λ_phys | 0.0 (baseline puro, sin física) |
| Semilla | 42 (aplicada antes de cada fold) |

El checkpoint del mejor modelo (menor val loss) se restaura al final del entrenamiento.

**Fuente:** `src/train.py` · `TrainConfig`

---

## Resultados por pozo

Ordenados por R² descendente. Las métricas están en **g/cc** (post inverse-transform).

| Pozo | MAE | RMSE | R² | PE_90 |
|---|---:|---:|---:|---:|
| Oeser_2 | 0.063 | 0.099 | 0.816 | 0.142 |
| Hoffman_2 | 0.067 | 0.103 | 0.778 | 0.141 |
| Oeser,_R__1 | 0.073 | 0.132 | 0.671 | 0.149 |
| Hoffman_Trust_1 | 0.063 | 0.104 | 0.669 | 0.131 |
| Nadine_1 | 0.069 | 0.107 | 0.667 | 0.146 |
| Grossardt_3 | 0.072 | 0.121 | 0.664 | 0.168 |
| Schneweis_10 | 0.041 | 0.058 | 0.644 | 0.089 |
| Soeken_12 | 0.154 | 0.200 | 0.637 | 0.315 |
| Beaver_S-Reif_1-22 | 0.079 | 0.118 | 0.625 | 0.161 |
| Kraft-Prusa_Unit_16 | 0.072 | 0.115 | 0.604 | 0.149 |
| Kroutwurst_19 | 0.040 | 0.061 | 0.577 | 0.086 |
| Woydziak-Kirmer_Unit_1 | 0.054 | 0.080 | 0.490 | 0.138 |
| Frees-Burmeister_13 | 0.044 | 0.068 | 0.487 | 0.091 |
| Rupe-Woydziak_Unit_1 | 0.133 | 0.184 | 0.485 | 0.286 |
| Rous_1-28 | 0.051 | 0.084 | 0.484 | 0.109 |
| Schneweis_3 | 0.134 | 0.193 | 0.471 | 0.308 |
| Wirth_5 | 0.108 | 0.172 | 0.437 | 0.238 |
| Holder_'A'_5 | 0.165 | 0.218 | 0.424 | 0.333 |
| Krier_'C'_6 | 0.238 | 0.301 | 0.332 | 0.487 |
| Weber_'A'_13 | 0.167 | 0.235 | 0.331 | 0.419 |
| Woydziak_'A'_1 | 0.200 | 0.269 | 0.258 | 0.465 |
| Demel_3 | 0.215 | 0.297 | 0.161 | 0.517 |
| Bieberle_Trust_2 | 0.278 | 0.388 | 0.093 | 0.716 |
| Kroutwurst_20 | 0.269 | 0.323 | 0.083 | 0.508 |
| Esfeld_9 | 0.165 | 0.226 | 0.082 | 0.357 |
| Kroutwurst_21 | 0.203 | 0.283 | −0.162 | 0.461 |
| Dolecheck_1 | 0.395 | 0.473 | −0.640 | 0.793 |

---

## Métricas agregadas

| Métrica | Media | Desv. std |
|---|---:|---:|
| MAE (g/cc) | **0.1338** | 0.0882 |
| RMSE (g/cc) | **0.1857** | 0.1056 |
| R² | **0.4137** | 0.3136 |
| PE_90 (g/cc) | **0.2927** | 0.1923 |

> **25/27 pozos con R² positivo**. La **mediana R² = 0.485** es representativa del
> comportamiento típico. Solo Kroutwurst_21 (−0.162) y Dolecheck_1 (−0.640) tienen R²
> negativo; Dolecheck_1 está documentado como anómalo (NPHI fuera de rango).
>
> Estos resultados (Yeo-Johnson + fix NaN) son significativamente mejores que los runs
> previos con min-max (MAE=0.183, R²=−0.174): la normalización por columna eliminó el
> sesgo de distribución y el fix de dominio eliminó predicciones NaN en 13 folds.

---

## Discusión

### Pozos con buen rendimiento (R² > 0.4)

Dieciocho pozos alcanzan R² > 0.4, con el mejor en Oeser_2 (0.816). Estos pozos comparten:
- Registros de buena calidad sin anomalías de unidad
- Varianza de DEN suficientemente alta para que R² sea informativo
- Respuesta de NPHI correlacionada con DEN (relación física activa)

Son los pozos donde Phase 3 debe mantener o mejorar el rendimiento.

### R² negativo con MAE bajo (Kroutwurst_19, Frees-Burmeister_13, Rous_1-28)

Tres pozos muestran este patrón: MAE/RMSE razonables pero R² muy negativo.
Indica formación con varianza de DEN muy baja (registro casi plano). El modelo comete
un sesgo pequeño en absoluto, pero ese sesgo supera la varianza del target, produciendo R² < 0.

- Kroutwurst_19: MAE=0.056 g/cc con R²=0.283 → varianza moderada, comportamiento normal.
- Frees-Burmeister_13: MAE=0.178 pero R²=−3.58 → casi sin varianza en DEN.
- Rous_1-28: MAE=0.209 pero R²=−3.07 → mismo patrón.

En estos casos MAE es más honesto que R².

### Outlier genuino: Kroutwurst_21

Kroutwurst_21 tiene tanto MAE=0.443 como R²=−2.86 malos simultáneamente, indicando
fallo real del modelo (no solo varianza baja). Posibles causas: litología no representada
en los 26 pozos de entrenamiento, o señal NPHI degradada. Candidato clave para observar
si λ > 0 aporta regularización adicional.

### Dolecheck_1 y Esfeld_9 (R² ≈ −0.49)

Dolecheck_1 está documentado como anómalo desde Phase 1 (NPHI 0.81–5.69 v/v, inconsistencia
de unidad en el LAS original). Tras el voting-consensus y la normalización, NPHI queda
prácticamente constante → el modelo no recibe información de porosidad real. R²=−0.49 es esperado.

Esfeld_9 muestra el mismo nivel de R², con MAE=0.218 — heterogeneidad litológica probable.

### Bieberle_Trust_2

MAE=0.292, RMSE=0.439, PE_90=0.866 — el PE_90 más alto del conjunto. Este pozo tiene la
cola de error más ancha, indicando que el modelo falla en profundidades específicas (posiblemente
zonas arcillosas donde NPHI pierde correlación con DEN).

---

## Implicaciones para Phase 3

| Observación | Consecuencia para PINN |
|---|---|
| 25/27 pozos con R² positivo | PINN parte de una base sólida; objetivo: mantener o mejorar |
| Mediana R² = 0.485 | Umbral representativo; PINN λ=0.1 logra mediana ~0.50 |
| Kroutwurst_21 (R²=−0.162), Dolecheck_1 (R²=−0.640) | Candidatos para mejora con regularización física |
| MAE medio = 0.134 g/cc | Umbral de referencia; PINN λ=0.1 logra 0.131 g/cc |
| PE_90 medio = 0.293 g/cc | El 90% de los errores queda bajo ~0.29 g/cc |
| External set: {Arensman_2, Burmeister_1, Rous_'F'_2} | Reservados; evaluar PINN final vs baseline en estos 3 pozos |

La calibración de los coeficientes DEN-NPHI en **espacio normalizado** (A=−0.2939, B=0.7608, R²=0.125)
ya está implementada en `src/physics.py`. Phase 3 puede proceder directamente al sweep de λ.

---

## Fuentes

| Módulo | Ruta |
|---|---|
| Script de entrenamiento | `scripts/03_train_baseline.py` |
| Modelo | `src/model.py` |
| Loop de entrenamiento | `src/train.py` |
| Métricas | `src/evaluate.py` |
| Preprocesamiento | `src/preprocessing.py` |
| Splits LOWO | `src/lowo.py` |
| Resultados raw | `outputs/baseline/metrics.json` |
| Predicciones por pozo | `outputs/baseline/predictions/*.parquet` |
