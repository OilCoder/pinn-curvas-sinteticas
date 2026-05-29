# EDA — Campo Kraft Prusa

## Dataset cargado

| Métrica | Valor |
|---------|-------|
| Pozos válidos (`data_loader`) | 32 |
| Registros totales | 172,618 |
| Rango de profundidad | 0 – 3,483 ft |
| Paso de muestreo | 0.5 ft (mayoría) |

> Nota: el inventario inicial (`01_inspect_las.py`) reportó 25 pozos con filtro <10% NaN. El `data_loader` usa el filtro más permisivo de ≥100 filas limpias tras `dropna`, resultando en 32 pozos. Se conservan los 32 para maximizar datos de entrenamiento.

---

## Estrategia de preprocesamiento (pipeline completo)

El pipeline sigue este orden estricto, justificado empíricamente por la auditoría de los 32 pozos:

```
1. Conversión de unidades NPHI  →  % a v/v (data_loader)
2. Centinelas grandes RT/RILM > 1e6  →  NaN (data_loader)
3. dropna inicial  →  filas con NaN en curvas canónicas (data_loader)

— A partir de aquí en preprocess_well:

4. Clip de features a FEATURE_BOUNDS  →  preserva filas
5. Drop de filas con DEN fuera de TARGET_BOUNDS  →  no entrenar con target roto
6. log₁₀ en RT y RILM
7. Normalización min-max per-well sobre las 6 curvas
```

**Decisión clave**: el clip y el drop son operaciones SEPARADAS con propósitos distintos:
- **Features** (GR, RT, RILM, NPHI, SP) → se **clipean al borde** (preserva la fila): un spike de centinela no debe tirar el registro completo
- **Target** (DEN) → se **eliminan las filas** con DEN inválido: no podemos entrenar con valores de DEN falsos

Esta separación nace de la auditoría: la versión anterior (NaN universal + dropna) eliminaba el 100% de algunos pozos (p.ej., Bieberle_Trust_2) porque tenían un solo curva con problema de escala. El clip de features rescata 89.5% del dataset vs 75% de la versión NaN-todo.

---

## Unidades de NPHI — corrección aplicada

**Problema detectado**: todos los archivos LAS del campo Kraft Prusa almacenan CNLS en porcentaje (valores típicos 14–31), no en fracción v/v (valores típicos 0.14–0.31).

**Diagnóstico**: medianas por pozo ~20 (imposible en v/v, confirma escala %):
- Antes de corrección: NPHI media = 36.6 (claramente % mal declarado)
- Después de corrección: NPHI media = 0.27 v/v (físicamente correcto)

**Solución**: en `src/data_loader.py`, si la unidad del LAS contiene "%" o la mediana del NPHI bruto supera 1.5, se divide por 100 antes de cualquier otro procesamiento.

---

## Valores centinela de resistividad

Algunos archivos KGS codifican valores nulos como 1e9 o 1e11 Ohm·m (artefacto de herramientas antiguas). Los centinelas clásicos `{-9999, -999.25}` no los capturan.

**Solución**: valores |RT| > 1e6 o |RILM| > 1e6 → NaN inmediatamente tras la carga, antes de `dropna`.

---

## Clip de features y filtro de target

### Feature bounds — `clip_features_to_bounds()`

Los valores extremos fuera de estos rangos se **clippean al borde** (no se eliminan filas). Los límites se eligieron amplios para neutralizar centinelas y artefactos sin descartar litologías legítimas:

| Curva | Min | Max | Justificación |
|-------|-----|-----|---------------|
| GR | 0 GAPI | 400 GAPI | >400 = centinela; spikes de arcilla 200–300 son legítimos |
| RT | 0.05 Ohm·m | 50,000 Ohm·m | Captura centinelas (100,000+) sin restringir formaciones resistivas |
| RILM | 0.05 Ohm·m | 50,000 Ohm·m | Ídem |
| NPHI | -0.15 v/v | 0.80 v/v | Generoso para gas (negativo) y problemas residuales de unidad |
| SP | -1000 mV | 1000 mV | Muy loose: algunos pozos registran SP en escala absoluta (0–666 mV) y la normalización per-well absorbe el offset |

**Por qué clip y no NaN**: la auditoría reveló que con NaN universal los pozos con un solo problema sistemático perdían el 100% de las filas. Ejemplo: Bieberle_Trust_2 tenía SP en escala 0–666 mV; con el clip al borde, todas las filas se conservan y la normalización per-well las escala correctamente.

### Target bounds — `filter_invalid_target_rows()`

| Curva | Min | Max | Acción |
|-------|-----|-----|--------|
| DEN | 1.5 g/cc | 3.1 g/cc | Filas fuera de este rango se **eliminan** |

DEN < 1.5 g/cc casi siempre es washout o lectura corrupta; DEN > 3.1 es error de herramienta. Entrenar con estos valores corrompería la pérdida.

### Resultado de la auditoría (32 pozos)

| Métrica | Estrategia NaN-todo | Estrategia clip+filter |
|---------|--------------------:|---------------------:|
| Pozos vivos | 30 / 32 (2 vacíos) | 32 / 32 |
| Filas conservadas | 129,661 (75%) | **154,451 (89.5%)** |
| Inverse transform DEN | OK | OK (error = 0) |
| Test wells en [0,1] tras norm | sí | sí |

**Caso atípico — Dolecheck_1**: 99% de los valores NPHI superan 0.80 v/v (rango 0.81–5.69). La unidad de origen del LAS es inconsistente. Sus features se clippean a la boundary; la normalización per-well aún recupera información, pero NPHI normalizado para este pozo es esencialmente constante en 1.0. Se mantiene en el dataset pero se documenta como anómalo para discusión en Phase 3.

---

## Transformación logarítmica de resistividad (decisión: SÍ)

El histograma confirmó que RT y RILM siguen distribución log-normal:

| Estadístico | RT (raw) | RT (log₁₀) |
|-------------|----------|------------|
| Asimetría | 8.65 | ~0 (simétrica) |
| Rango | 0.1 – 10,000 Ohm·m | -1 – 4 |

**Decisión: aplicar log₁₀ a RT y RILM antes de la normalización.** Elimina la dominancia de outliers de alta resistividad y produce una distribución aproximadamente normal.

Implementado en `src/preprocessing.py → apply_log_rt()`.

---

## Normalización per-well min-max (decisión: SÍ)

**Motivación**: cada pozo refleja su propia litología dominante, herramienta y profundidad. Los rangos absolutos de GR, RT, DEN difieren pozo a pozo por razones geológicas, no por error. Normalizar globalmente significaría que pozos con valores extremos dictarían el rango para todos.

**Decisión: normalización min-max independiente por pozo**, aplicada después de clip + log-RT.

| Alternativa | Razón de rechazo |
|-------------|-----------------|
| Z-score global | Outliers en un pozo distorsionan la std global |
| Z-score per-well | Min-max da rango garantizado [0,1], preferible para ReLU sin activación negativa |
| Min-max global | Pierde variabilidad relativa intra-pozo; test well fuera de rango si tiene valores extremos |

El scaler de cada pozo se ajusta **solo** con los datos de ese pozo. En el LOWO:
- Pozos de entrenamiento: scaler propio (sin contaminación cruzada)
- Pozo de prueba: scaler propio (no usa información de los pozos de entrenamiento)

Implementado en `src/preprocessing.py → WellScaler`.

---

## Relación DEN–NPHI

### Correlación global vs per-well

| Contexto | R² | Pendiente |
|----------|----|-----------|
| Raw global (todos los pozos mezclados, NPHI en v/v) | 0.017 | -0.090 |
| Per-well mediana | **0.277** | negativa (31/32 pozos) |
| Per-well media | 0.298 | negativa (31/32 pozos) |

**Interpretación**: el R² global bajo es un artefacto del mezclado de pozos con distintos offsets absolutos de DEN y NPHI. Dentro de cada pozo la relación negativa es consistente y físicamente correcta: a mayor porosidad neutrón, menor densidad bulk.

### Excepción: Dolecheck_1

Único pozo con pendiente positiva en espacio normalizado (+1.03). Probable efecto de gas en el log neutrón (gas baja NPHI y DEN simultáneamente, invirtiendo la relación). Registrado como caso de análisis en Phase 3.

---

## Estadísticas descriptivas de las curvas (espacio raw, post-corrección de unidades)

| Curva | Media | Std | Asimetría | P25 | P50 | P75 | Max |
|-------|-------|-----|-----------|-----|-----|-----|-----|
| GR (GAPI) | 69.1 | 30.8 | +1.50 | 46.1 | 66.5 | 88.0 | 683.1 |
| RT (Ohm·m) | 1300 | 11,257 | +8.65 | 2.77 | 4.41 | 9.56 | 100,000 |
| RILM (Ohm·m) | 3456 | 18,237 | +5.10 | 2.63 | 3.83 | 7.75 | 100,000 |
| NPHI (v/v) | 0.366 | 0.609 | +4.75 | 0.140 | 0.249 | 0.317 | 6.66 |
| SP (mV) | -24.5 | 124.7 | +3.20 | -80.5 | -52.1 | -5.5 | 666.6 |
| DEN (g/cc) | 2.231 | 0.424 | -1.14 | 2.054 | 2.382 | 2.545 | 2.979 |

> Los máximos extremos de RT, RILM, NPHI y SP corresponden a valores residuales que se eliminan con `clip_physical_ranges()` antes del entrenamiento. Los parquets almacenan los datos post-corrección de unidades pero pre-clip para inspección.

---

## Archivos generados

| Archivo | Contenido |
|---------|-----------|
| `outputs/eda/distributions_raw.png` | Histogramas de las 6 curvas en escala raw |
| `outputs/eda/log_rt_comparison.png` | RT y RILM: distribución raw vs log₁₀ |
| `outputs/eda/den_nphi_crossplot.png` | Crossplot DEN vs NPHI global con fit lineal |
| `outputs/eda/den_nphi_by_well.png` | Crossplot DEN vs NPHI por pozo (small multiples) |
| `outputs/eda/per_well_boxplots.png` | Boxplot por curva mostrando variación inter-pozo |
| `outputs/eda/profile_*.png` | Perfiles de profundidad de 4 pozos ejemplo |
| `outputs/eda/well_quality.csv` | Métricas de calidad por pozo |
| `outputs/eda/curve_statistics.csv` | Estadísticas descriptivas + asimetría de las 6 curvas |
| `outputs/eda/den_nphi_coefficients.csv` | Coeficientes raw: a=-0.090, b=2.264, R²=0.017 |
| `data/processed/*.parquet` | DataFrames limpios por pozo, post-corrección de unidades (32 archivos) |

---

## Decisiones documentadas

| Decisión | Elección | Alternativa descartada |
|----------|---------|----------------------|
| Corrección NPHI | Dividir por 100 si mediana > 1.5 o unidad "%" | Usar valores en % (error en el modelo) |
| Centinelas grandes | RT/RILM > 1e6 → NaN | Solo centinelas conocidos (-9999, -999.25) |
| Clip físico | Límites por curva antes de normalización | Sin clip (la normalización se distorsiona) |
| Log-RT | Sí (log₁₀ a RT y RILM) | No aplicar |
| Normalización | Min-max per-well | Z-score global, Z-score per-well, Min-max global |
| Coeficientes física PINN | Espacio normalizado (re-calibrar en Phase 3) | Espacio raw (R² irrelevante) |
| Pozos a usar | 32 (filtro ≥100 filas) | 25 (filtro <10% NaN) |

*Fuente del análisis: `scripts/02_run_eda.py`*  
*Fuente del código: `src/data_loader.py`, `src/preprocessing.py`*
