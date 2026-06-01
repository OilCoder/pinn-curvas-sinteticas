# Modelo Base MLP — Evaluación LOWO

Establece el rendimiento de referencia del MLP supervisado puro (λ=0) sobre el campo Kraft Prusa
mediante validación cruzada Leave-One-Well-Out. Toda métrica de Phase 3 (PINN) se compara contra
estos valores.

---

## Protocolo de evaluación

### Leave-One-Well-Out (LOWO)

```
32 pozos totales
├── 3 pozos externos (field_split, seed=42) — reservados para Phase 4
│   Beaver_S-Reif_1-22 · Frees-Burmeister_13 · Kroutwurst_21
└── 29 pozos train pool — usados en LOWO
    └── 29 folds × (28 pozos train | 1 pozo test)
```

Cada fold es completamente independiente:

1. Los 28 pozos de entrenamiento se preprocesan individualmente (cada uno con su propio `WellScaler`).
2. El pozo de prueba se preprocesa con su propio `WellScaler` — sin información cruzada del conjunto de entrenamiento.
3. El modelo se inicializa desde cero con `set_seed(42)`.
4. Las predicciones se invierten a g/cc antes de calcular métricas.

Esta estrategia garantiza que no hay data leakage entre pozos en ninguna etapa.

**Fuentes:** `src/lowo.py` · `scripts/03_train_baseline.py`

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
| Burmeister_1 | 0.093 | 0.127 | 0.633 | 0.177 |
| Weber_'A'_13 | 0.136 | 0.188 | 0.571 | 0.309 |
| Kraft-Prusa_Unit_16 | 0.097 | 0.125 | 0.536 | 0.194 |
| Woydziak-Kirmer_Unit_1 | 0.056 | 0.078 | 0.518 | 0.124 |
| Oeser_2 | 0.141 | 0.168 | 0.473 | 0.258 |
| Hoffman_2 | 0.135 | 0.166 | 0.423 | 0.261 |
| Rupe-Woydziak_Unit_1 | 0.157 | 0.197 | 0.406 | 0.297 |
| Schneweis_3 | 0.152 | 0.203 | 0.410 | 0.315 |
| Schneweis_10 | 0.059 | 0.075 | 0.406 | 0.114 |
| Soeken_12 | 0.188 | 0.263 | 0.377 | 0.438 |
| Oeser,_R__1 | 0.152 | 0.184 | 0.362 | 0.280 |
| Nadine_1 | 0.121 | 0.160 | 0.252 | 0.247 |
| Woydziak_'A'_1 | 0.205 | 0.271 | 0.250 | 0.491 |
| Grossardt_3 | 0.147 | 0.181 | 0.249 | 0.286 |
| Kroutwurst_20 | 0.229 | 0.296 | 0.229 | 0.503 |
| Demel_3 | 0.214 | 0.291 | 0.196 | 0.553 |
| Arensman_2 | 0.168 | 0.227 | 0.171 | 0.354 |
| Esfeld_9 | 0.161 | 0.216 | 0.164 | 0.318 |
| Krier_'C'_6 | 0.285 | 0.357 | 0.063 | 0.628 |
| Rous_'F'_2 | 0.232 | 0.326 | 0.014 | 0.621 |
| Wirth_5 | 0.177 | 0.229 | 0.002 | 0.362 |
| Holder_'A'_5 | 0.223 | 0.290 | −0.012 | 0.470 |
| Bieberle_Trust_2 | 0.291 | 0.412 | −0.020 | 0.802 |
| Dolecheck_1 | 0.339 | 0.391 | −0.121 | 0.616 |
| Kroutwurst_19 | 0.076 | 0.104 | −0.223 | 0.167 |
| Frees-Burmeister_13_part1 | 0.109 | 0.131 | −0.877 | 0.200 |
| Rous_1-28 | 0.233 | 0.268 | −4.265 | 0.386 |
| Hoffman_Trust_1 | 0.352 | 0.430 | −4.659 | 0.660 |
| Weber_'A'_13_part1 ⚠️ | 0.136 | 0.188 | 0.571 | 0.309 |

---

## Métricas agregadas

| Métrica | Media | Desv. std |
|---|---:|---:|
| MAE (g/cc) | **0.1746** | 0.0751 |
| RMSE (g/cc) | **0.2256** | 0.0940 |
| R² | −0.100 | 1.226 |
| PE_90 (g/cc) | **0.3704** | 0.1742 |

> **Nota sobre R²:** la media y el std de R² están distorsionados por dos outliers extremos
> (Hoffman_Trust_1 = −4.66, Rous_1-28 = −4.26). Si se excluyen ambos, R² medio = **0.20 ± 0.38**.
> MAE y PE_90 son las métricas primarias para comparar Phase 3 vs baseline por ser más robustas.

---

## Discusión

### Pozos con buen rendimiento (R² > 0.4)

Once pozos alcanzan R² > 0.4, con el mejor en Burmeister_1 (0.633). Estos pozos comparten:
- Registros de buena calidad sin anomalías de unidad
- Varianza de DEN suficientemente alta para que R² sea informativo
- Respuesta de NPHI correlacionada con DEN (relación física activa)

Son los pozos donde Phase 3 debería mantener o mejorar el rendimiento.

### R² negativo con MAE bajo (Kroutwurst_19, Frees-Burmeister_13_part1)

Kroutwurst_19 tiene MAE = 0.076 g/cc pero R² = −0.22. Este patrón indica que el pozo tiene
varianza de DEN muy baja (formación uniforme). El modelo comete un error de sesgo pequeño
en absoluto, pero ese sesgo supera la varianza del target, produciendo R² < 0.

En estos casos MAE es más honesto que R²: el error absoluto de 0.076 g/cc es técnicamente
aceptable para predicción de densidad.

### Outliers extremos de R² (Hoffman_Trust_1, Rous_1-28)

Estos dos pozos tienen R² < −4, lo que indica que el modelo falla completamente en capturar
la variabilidad del DEN. Posibles causas:

- **Heterogeneidad litológica no representada en el train:** si estos pozos tienen facies
  únicas no presentes en los 28 folds de entrenamiento, el modelo no puede extrapolarse.
- **Señal de NPHI degradada:** correlación DEN-NPHI débil en zonas arcillosas.
- **Artefactos de la normalización per-well:** si el scaler del pozo de prueba captura mal
  los percentiles locales, el espacio normalizado queda desalineado.

Estos pozos son los candidatos más relevantes para observar si la pérdida física (Phase 3)
aporta regularización adicional que ayude a generalizar.

### Duplicado: Weber_'A'_13 y Weber_'A'_13_part1

El data loader generó dos entradas para el mismo pozo físico con sufijos `_part1`. Las métricas
son idénticas (hasta 16 dígitos), confirmando que son el mismo conjunto de datos evaluado dos
veces. El pozo físico único es Weber_'A'_13; el fold `_part1` es redundante. Esto reduce
el conteo efectivo de folds únicos a **28**, aunque no afecta la validez del benchmark
(el modelo nunca vio el pozo en ninguno de sus dos fold evaluados).

**Acción pendiente para Phase 3:** investigar y corregir el origen del sufijo `_part1` en
`src/data_loader.py` o en los nombres de archivo LAS.

### Dolecheck_1

Pozo documentado como anómalo desde Phase 1 (NPHI 0.81–5.69 v/v, inconsistencia de unidad
en el LAS original). NPHI se clipeó a 0.80 y queda constante = 1.0 tras normalización.
El modelo no recibe información de porosidad real para este pozo → R² = −0.12 esperable.

---

## Implicaciones para Phase 3

| Observación | Consecuencia para PINN |
|---|---|
| 11 pozos con R² > 0.4 | Phase 3 debe mantener o mejorar — son el grupo de control |
| Rous_1-28 y Hoffman_Trust_1 con R² < −4 | λ > 0 puede aportar regularización física donde los datos fallan |
| MAE medio = 0.175 g/cc | Umbral de referencia; Phase 3 debe reducirlo para λ óptimo |
| Duplicado Weber_'A'_13 | Corregir antes de Phase 3 para métricas limpias |
| PE_90 medio = 0.370 g/cc | El 90% de los errores queda bajo ~0.37 g/cc — benchmark claro |

La calibración de los coeficientes DEN-NPHI en **espacio normalizado** (no en raw) es el
primer paso de Phase 3, antes de implementar `src/physics.py`.

---

## Fuentes

| Módulo | Ruta |
|---|---|
| Script de entrenamiento | `scripts/03_train_baseline.py` |
| Modelo | `src/model.py` |
| Loop de entrenamiento | `src/train.py` |
| Métricas | `src/evaluate.py` |
| Splits LOWO | `src/lowo.py` |
| Resultados raw | `outputs/baseline/metrics.json` |
| Predicciones por pozo | `outputs/baseline/predictions/*.parquet` |
