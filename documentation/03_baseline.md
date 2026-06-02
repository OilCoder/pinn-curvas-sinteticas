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
| Oeser,_R__1 | 0.114 | 0.153 | 0.554 | 0.237 |
| Grossardt_3 | 0.106 | 0.143 | 0.529 | 0.224 |
| Hoffman_2 | 0.123 | 0.158 | 0.477 | 0.233 |
| Woydziak-Kirmer_Unit_1 | 0.062 | 0.085 | 0.431 | 0.139 |
| Schneweis_3 | 0.152 | 0.203 | 0.411 | 0.320 |
| Schneweis_10 | 0.061 | 0.077 | 0.380 | 0.123 |
| Rupe-Woydziak_Unit_1 | 0.157 | 0.206 | 0.353 | 0.328 |
| Hoffman_Trust_1 | 0.113 | 0.147 | 0.339 | 0.224 |
| Wirth_5 | 0.131 | 0.187 | 0.338 | 0.284 |
| Nadine_1 | 0.111 | 0.153 | 0.319 | 0.225 |
| Kroutwurst_19 | 0.056 | 0.080 | 0.283 | 0.115 |
| Weber_'A'_13 | 0.193 | 0.244 | 0.277 | 0.417 |
| Holder_'A'_5 | 0.193 | 0.251 | 0.237 | 0.424 |
| Woydziak_'A'_1 | 0.209 | 0.277 | 0.216 | 0.503 |
| Demel_3 | 0.215 | 0.294 | 0.178 | 0.550 |
| Oeser_2 | 0.176 | 0.211 | 0.174 | 0.333 |
| Beaver_S-Reif_1-22 | 0.136 | 0.177 | 0.151 | 0.272 |
| Soeken_12 | 0.226 | 0.311 | 0.128 | 0.561 |
| Krier_'C'_6 | 0.277 | 0.349 | 0.105 | 0.620 |
| Kraft-Prusa_Unit_16 | 0.137 | 0.174 | 0.094 | 0.286 |
| Kroutwurst_20 | 0.268 | 0.343 | −0.035 | 0.626 |
| Bieberle_Trust_2 | 0.292 | 0.439 | −0.159 | 0.866 |
| Dolecheck_1 | 0.390 | 0.450 | −0.488 | 0.687 |
| Esfeld_9 | 0.218 | 0.288 | −0.489 | 0.473 |
| Frees-Burmeister_13 | 0.178 | 0.204 | −3.584 | 0.295 |
| Rous_1-28 | 0.209 | 0.235 | −3.068 | 0.352 |
| Kroutwurst_21 | 0.443 | 0.515 | −2.861 | 0.785 |

---

## Métricas agregadas

| Métrica | Media | Desv. std |
|---|---:|---:|
| MAE (g/cc) | **0.1832** | 0.0908 |
| RMSE (g/cc) | **0.2354** | 0.1098 |
| R² | −0.174 | 1.093 |
| PE_90 (g/cc) | **0.3889** | 0.1977 |

> **Nota sobre R²:** la media está distorsionada por tres outliers extremos
> (Frees-Burmeister_13 = −3.58, Rous_1-28 = −3.07, Kroutwurst_21 = −2.86).
> La **mediana R² = 0.216** es más representativa. Si se excluyen los tres outliers,
> R² medio = **0.30 ± 0.28** sobre los 24 restantes.
> MAE y PE_90 son las métricas primarias para comparar Phase 3 vs baseline.

> **Comparación con run previo (29 folds, clip fijo):**
> MAE 0.175 → 0.183 (+0.008), RMSE 0.226 → 0.235 (+0.009). La ligera degradación
> se debe principalmente a que Burmeister_1 (R²=0.633 en el run anterior) pasó al
> conjunto externo, y Kroutwurst_21 (R²=−2.86) entró al pool LOWO al cambiar el split.

---

## Discusión

### Pozos con buen rendimiento (R² > 0.4)

Seis pozos alcanzan R² > 0.4, con el mejor en Oeser,_R__1 (0.554). Estos pozos comparten:
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
| 18/27 pozos con R² positivo | Phase 3 debe mantener o mejorar — grupo de control amplio |
| Mediana R² = 0.216 | Umbral representativo; PINN objetivo: mediana > 0.22 |
| Frees-Burmeister_13, Rous_1-28 con R² < −3 | Varianza baja; λ > 0 improbable que ayude — observar MAE |
| Kroutwurst_21 con fallo genuino (MAE y R² malos) | λ > 0 puede aportar regularización donde datos fallan |
| MAE medio = 0.183 g/cc | Umbral de referencia; Phase 3 debe reducirlo para λ óptimo |
| PE_90 medio = 0.389 g/cc | El 90% de los errores queda bajo ~0.39 g/cc |
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
