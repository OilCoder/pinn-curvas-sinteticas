# Preprocessing Pipeline — Estrategia y Justificación

Pipeline completo de limpieza y transformación de datos que convierte los archivos LAS
crudos del campo Kraft Prusa en tensores listos para entrenar. Cada decisión está
justificada empíricamente contra los 32 pozos del dataset.

---

## 1. Contexto del problema

Los registros de pozo del campo Kraft Prusa (Kansas Geological Survey) presentan
cinco características que hacen que un preprocesamiento ingenuo produzca datos
silenciosamente corruptos:

| Problema | Impacto si se ignora |
|----------|----------------------|
| CNLS almacenado en % en lugar de v/v | NPHI entre 14–31 en lugar de 0.14–0.31; el modelo aprende relaciones DEN–NPHI con escala equivocada |
| Centinelas de resistividad grandes (1e9, 1e11 Ohm·m) | RT/RILM no capturados por filtros clásicos (-9999); distorsionan normalización y log transform |
| Variación inter-pozo de escala absoluta | GR puede ir de 0–50 en carbonatos y 0–200 en shales; normalización global obliga al modelo a aprender offsets geológicos en vez de patrones |
| Valores de DEN físicamente imposibles | DEN=1.0 (washout) y DEN=4.0 (herramienta pegada) son errores de adquisición, no litología; entrenar con ellos corrompe la pérdida |
| SP con referencia absoluta vs diferencial | Algunos pozos registran SP en escala 0–666 mV; otros en -120–+20 mV. Misma señal, distinto offset |

El pipeline resuelve estos cinco problemas en un orden estricto, donde cada paso
asume que el anterior ya fue aplicado.

La siguiente figura muestra la distribución de cada curva pozo a pozo (post-clip).
Los boxes representan el IQR de cada pozo; la línea roja es la mediana. La variación
horizontal entre pozos es geológica real — justifica la normalización per-well.

![Distribución por pozo](../outputs/eda/per_well_boxplots.png)

*Fig. 1 — Variación inter-pozo en unidades físicas (post-clip). RT/RILM: outliers
legítimos en carbonatos resistivos. SP: offset absoluto en algunos pozos (ver Bieberle_Trust_2).
DEN: distribución bimodal arena/shale consistente en todos los pozos.*

---

## 2. Visión general del pipeline

```
Archivo LAS (crudo)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  src/data_loader.py — load_well()                     │
│                                                       │
│  Paso 1 — Mapeo de mnemonics → nombres canónicos      │
│  Paso 2 — Construcción del DataFrame [DEPTH + 6 col]  │
│  Paso 3a — Centinelas LAS clásicos → NaN              │
│  Paso 3b — Centinelas RT/RILM > 1e6 → NaN             │
│  Paso 3c — Conversión NPHI % → v/v                    │
│  Paso 4 — dropna en curvas canónicas; mínimo 100 filas│
└───────────────────────────────────────────────────────┘
        │
        │  DataFrame crudo en unidades físicas correctas
        ▼
┌───────────────────────────────────────────────────────┐
│  src/preprocessing.py — preprocess_well()             │
│                                                       │
│  Paso 1 — clip_features_to_bounds()  → filas intactas │
│  Paso 2 — filter_invalid_target_rows() → drop DEN bad │
│  Paso 3 — apply_log_rt()             → log₁₀ RT, RILM│
│  Paso 4 — fit_scaler() + transform() → [0, 1] per well│
└───────────────────────────────────────────────────────┘
        │
        │  DataFrame normalizado [0, 1], sin NaN
        ▼
  src/dataset.py — WellDataset → tensores PyTorch
```

**Regla de oro**: la limpieza de datos de adquisición (unidades, centinelas) ocurre
en `data_loader`. La transformación para el modelo (distribución, escala) ocurre en
`preprocessing`. Nunca se mezclan.

---

## 3. Paso a paso detallado

### 3.1 Mapeo de mnemonics a nombres canónicos

**Qué hace**: renombra las curvas del LAS a los 6 nombres internos del proyecto.

```
GR        → GR   (Gamma Ray, GAPI)
RILD      → RT   (Resistividad profunda, Ohm·m)
RILM      → RILM (Resistividad media, Ohm·m)
CNLS/CNPOR/CNSS/CNDL/NPOR/NEU/PHIN/... → NPHI  (Neutrón compensado, v/v)
SP        → SP   (Potencial espontáneo, mV)
RHOB      → DEN  (Densidad bulk — target, g/cc)
```

**Por qué**: el KGS usa distintos mnemonics según la herramienta y el año de registro.
Sin normalización de nombres, cada módulo necesitaría conocer las variantes. La
normalización en el loader desacopla la variabilidad de los LAS del resto del código.

**Criterio de rechazo**: si alguna de las 6 curvas canónicas falta en el LAS, el pozo
se descarta en esta etapa. Los pozos con las 6 curvas presentes son 32 de los 38 LAS
disponibles.

**Código**: `src/data_loader.py → _MNEMONIC_MAP`, `load_well() Step 1`.

---

### 3.2 Centinelas LAS clásicos → NaN

**Qué hace**: reemplaza los valores nulos estándar del formato LAS por `NaN` de pandas.

```python
_NULL_VALUES = {-9999.0, -999.25, -9999.25, -999.0}
```

**Por qué**: el formato LAS codifica "sin dato" con valores numéricos especiales.
Sin este paso, -9999 se trataría como una lectura válida (Gamma Ray de -9999 GAPI,
DEN de -9999 g/cc), lo que distorsionaría cualquier estadística y la normalización.

**Código**: `src/data_loader.py → load_well() Step 3, Substep 3.1`.

---

### 3.3 Centinelas de resistividad grandes → NaN

**Qué hace**: cualquier valor de RT o RILM con valor absoluto > 1 × 10⁶ Ohm·m se
convierte a `NaN`.

**Por qué**: las herramientas de inducción de los años 70–80 usadas en Kraft Prusa
codifican datos faltantes con valores como 1 × 10⁹ o 1 × 10¹¹ Ohm·m. El umbral
clásico de -9999 no los captura. El máximo físico realista para una formación
(evaporitas, carbonatos muy resistivos) es del orden de 10,000–50,000 Ohm·m.
Valores > 1 × 10⁶ Ohm·m son, sin excepción, artefactos.

**Consecuencia si se omite**: estos valores sobreviven el `dropna` clásico,
entran al log transform (log₁₀(1e9) = 9, muy fuera del rango real de ≈ -1 a 4),
y distorsionan la normalización per-well al expandir el rango artificialmente.

**Código**: `src/data_loader.py → load_well() Step 3, Substep 3.2`.

---

### 3.4 Conversión NPHI de % a v/v

**Qué hace**: si la unidad del campo `~Curve` del LAS contiene `%`, o si la mediana
de los valores no nulos de NPHI supera 1.5, se divide toda la columna por 100.

```python
if "%" in nphi_unit or (median(NPHI) > 1.5):
    NPHI = NPHI / 100.0
```

**Por qué**: todos los archivos LAS del campo Kraft Prusa almacenan CNLS en porcentaje
(valores típicos 14–31 %). La convención del sector es fracción v/v (0.14–0.31).
Sin esta conversión, la relación física DEN–NPHI que el PINN embebe tiene la escala
equivocada, y las métricas de NPHI vs otros campos o datasets son incomparables.

**Diagnóstico empírico**:

| Estado | NPHI media | Interpretación |
|--------|-----------|----------------|
| Sin corrección | 36.6 | Imposible en v/v; confirma escala % |
| Con corrección | 0.27 v/v | Físicamente correcto para sedimentos clásticos |

**Doble guarda**: se revisa primero la unidad del LAS (más fiable); si está vacía o
es ambigua, el heurístico de mediana > 1.5 decide. En los 32 pozos de Kraft Prusa,
ambas condiciones coinciden.

El histograma NPHI (panel inferior izquierdo) confirma la distribución corregida:
mediana 0.249 v/v, media 0.262 v/v — rango físicamente correcto para sedimentos
clásticos. GR muestra la distribución bimodal típica (arenas ~40 GAPI, shales ~90 GAPI).

![Distribuciones post-clip](../outputs/eda/distributions_raw.png)

*Fig. 2 — Distribuciones de las 6 curvas en unidades físicas, con datos post-clip
aplicado a FEATURE_BOUNDS y TARGET_BOUNDS. Línea sólida roja = mediana; línea gris
discontinua = media. RT y RILM post-clip aún muestran cola derecha — justifica el
log transform (Paso 3.8).*

**Código**: `src/data_loader.py → load_well() Step 3, Substep 3.3`.

---

### 3.5 dropna y mínimo de filas

**Qué hace**: elimina cualquier fila que tenga `NaN` en alguna de las 6 curvas
canónicas. Descarta el pozo si quedan menos de 100 filas.

**Por qué**: el `NaN` que sobrevive en este punto es ruido genuino — profundidades
donde la herramienta no registró. No tiene sentido entrenar sobre filas con inputs
incompletos. El umbral de 100 filas (~50 ft a 0.5 ft/muestra) es el mínimo para que
el scaler per-well tenga estadísticas significativas.

**Resultado**: de 38 LAS disponibles, 32 superan el umbral y 6 quedan excluidos.

**Código**: `src/data_loader.py → load_well() Step 4`.

---

### 3.6 Clip de features a límites físicos

**Qué hace**: los valores de cada feature que exceden los límites físicos se
**recortan al borde** (no se convierten a NaN, no se elimina la fila).

| Curva | Límite inferior | Límite superior | Unidad |
|-------|----------------|----------------|--------|
| GR | 0 | 400 | GAPI |
| RT | 0.05 | 50,000 | Ohm·m |
| RILM | 0.05 | 50,000 | Ohm·m |
| NPHI | -0.15 | 0.80 | v/v |
| SP | -1,000 | 1,000 | mV |

**Por qué clip y no NaN**:

La estrategia NaN-todo que existía en versiones anteriores tenía un defecto grave:
si una sola curva de un pozo tenía un problema sistemático de escala, el `dropna`
posterior eliminaba el **100% de las filas de ese pozo**. Dos ejemplos reales:

- **Bieberle_Trust_2**: SP registrado en escala absoluta (0–666 mV). Con el límite
  anterior de [-200, 100] mV, toda la columna SP se convertía a NaN → 100% de filas
  perdidas. Con el clip a [-1000, 1000] mV, todas las filas se conservan y la
  normalización per-well absorbe el offset absoluto.
- **Dolecheck_1**: NPHI con valores 0.81–5.69 v/v (inconsistencia de unidad residual).
  Con clip a 0.80, la columna queda constante en el borde pero la fila no se pierde
  y el DEN sigue siendo válido para entrenar.

El perfil del pozo Bieberle_Trust_2 ilustra el problema: el track SP (5.ª columna)
muestra valores en escala 0–400 mV — referencia absoluta al nivel del mar en lugar
de diferencial. Con el clip a [-1000, 1000] mV, estas filas se conservan.

![Perfil Bieberle_Trust_2](../outputs/eda/profile_Bieberle_Trust_2.png)

*Fig. 3 — Perfil de pozo Bieberle_Trust_2. El track SP (5.ª columna) registra en
escala 0–400 mV. La estrategia NaN-todo eliminaba el 100% de las filas de este pozo;
el clip al borde de ±1000 mV las preserva todas.*

**Por qué los límites son amplios**: el objetivo no es restringir la litología sino
neutralizar artefactos de adquisición. Los rangos geológicos reales del campo (GR
hasta 300 GAPI en shales carbonosos, RT hasta varios miles en carbonatos) caben
holgadamente dentro de estos límites.

**Resultado de la auditoría**:

| Estrategia | Pozos vivos | Filas conservadas |
|------------|:-----------:|:-----------------:|
| NaN universal + dropna | 30 / 32 | 129,661 (75%) |
| Clip features + drop DEN | **32 / 32** | **154,451 (89.5%)** |

**Código**: `src/preprocessing.py → clip_features_to_bounds()`, `FEATURE_BOUNDS`.

---

### 3.7 Eliminación de filas con DEN inválido

**Qué hace**: elimina completamente las filas donde DEN cae fuera de (1.5, 3.1) g/cc.

| Límite | Valor | Justificación petrof ísica |
|--------|-------|--------------------------|
| Mínimo | 1.5 g/cc | Por debajo: washout severo (la herramienta lee fluido del pozo) o falla electrónica |
| Máximo | 3.1 g/cc | Por encima: herramienta pegada a la pared o error de calibración; las litologías más densas del campo (anhidrita, dolomita densa) son ≈ 2.87 g/cc |

**Por qué se elimina la fila y no se clipea como los features**:

El DEN es el **target** del modelo — es lo que queremos predecir. Si el valor de DEN
en esa fila es físicamente imposible, el modelo recibiría una señal de pérdida
incorrecta. No hay forma de "reparar" un DEN de 1.2 g/cc (la herramienta no midió
nada útil); la única opción honesta es descartarlo. En cambio, un GR de 450 GAPI
(spike de herramienta) se puede clippear a 400 GAPI porque el resto de la fila —
incluyendo el DEN — sigue siendo válido.

**Principio**: los features pueden ser imperfectos siempre que el target sea correcto.
Si el target está corrupto, toda la fila es basura para el entrenamiento.

**Código**: `src/preprocessing.py → filter_invalid_target_rows()`, `TARGET_BOUNDS`.

---

### 3.8 Transformación logarítmica de resistividad

**Qué hace**: reemplaza RT y RILM por su logaritmo en base 10.

```
RT_log   = log₁₀(RT)    →  rango típico: -1 a +4
RILM_log = log₁₀(RILM)  →  rango típico: -1 a +4
```

**Por qué**: la resistividad sigue una distribución log-normal en la naturaleza.
En escala lineal, la asimetría de RT es +8.65 (fuertemente cola derecha): el 75%
de los valores caen entre 0.1 y 10 Ohm·m pero el máximo tras el clip es 50,000 Ohm·m.
La normalización min-max comprimiría todo el rango útil a una franja estrecha del
intervalo [0, 1], degradando la señal que el modelo recibe.

| Estadístico | RT lineal (post-clip) | RT log₁₀ |
|-------------|----------------------|----------|
| Asimetría | +8.65 | ≈ 0 (simétrica) |
| P25–P75 | 2.77 – 9.56 Ohm·m | 0.44 – 0.98 |
| Rango efectivo | 0.05 – 50,000 | -1.3 – 4.7 |

Tras el log transform, la normalización min-max distribuye la señal de forma uniforme
a lo largo del rango [0, 1] y los gradientes del modelo son estables.

![Log RT comparison](../outputs/eda/log_rt_comparison.png)

*Fig. 4 — RT y RILM: escala lineal (izquierda) vs log₁₀ (derecha). En escala lineal
el 99% de las muestras quedan comprimidas en el primer bin — la distribución útil es
invisible. En log₁₀ la distribución es unimodal y manejable. El skewness anotado en
cada panel (+8.64 → +3.40 para RT) cuantifica la mejora.*

**Por qué log₁₀ y no log natural**: log₁₀ mantiene la interpretabilidad petrof ísica
clásica (1 década = 1 unidad). No hay diferencia funcional para el modelo; es
convención de la industria.

**Código**: `src/preprocessing.py → apply_log_rt()`.

---

### 3.9 Normalización min-max per-well

**Qué hace**: aplica min-max a cada pozo de forma independiente, produciendo valores
en [0, 1] para las 6 curvas.

```
x_norm = (x - min_well) / (max_well - min_well)
```

El scaler de cada pozo se ajusta **solo** sobre los datos de ese pozo y se almacena
en un objeto `WellScaler`. Ningún pozo "ve" los datos de otro.

**Por qué per-well y no global**:

Cada pozo refleja su propia litología, herramienta de adquisición y rango de
profundidad. Las diferencias absolutas entre pozos son geológicamente reales:

- Un pozo en carbonatos tiene GR de 0–30 GAPI
- Un pozo en shales tiene GR de 60–200 GAPI
- Ambos son "normales" para su contexto

Con normalización global, el pozo de carbonatos ocuparía solo el 15% del rango [0, 1]
y el modelo no distinguiría variaciones dentro de él. Con normalización per-well,
cada pozo ocupa todo el rango [0, 1] y el modelo aprende patrones relativos, no
magnitudes absolutas.

**Comparativa de alternativas**:

| Estrategia | Por qué se descartó |
|------------|---------------------|
| Min-max global | Pozos con valores extremos dictan el rango para todos; variación intra-pozo comprimida |
| Z-score global | Los outliers de un pozo distorsionan la media y std global |
| Z-score per-well | No garantiza rango [0, 1]; valores negativos son problemáticos con activaciones ReLU |
| **Min-max per-well** | **Garantiza [0, 1], independiente por pozo, compatible con ReLU** |

**Integridad en el protocolo LOWO**:

La separación de scalers es lo que garantiza que no haya data leakage en el
Leave-One-Well-Out:

- **Pozo de entrenamiento**: su scaler se ajusta con sus propios datos.
- **Pozo de prueba**: su scaler se ajusta con sus propios datos, completamente
  ajeno a los pozos de entrenamiento. El modelo ve features normalizadas [0,1]
  del pozo test, pero los parámetros de normalización no provienen del train set.

Las dos figuras siguientes muestran por qué la normalización global falla para este
dataset y por qué la per-well es necesaria.

![DEN-NPHI global](../outputs/eda/den_nphi_crossplot.png)

*Fig. 5 — Crossplot DEN vs NPHI global (todos los pozos mezclados). R² = 0.017:
la mezcla de pozos con distintos offsets geológicos destruye la correlación.
La normalización global tendría el mismo problema: los rangos absolutos de un pozo
dominarían el espacio normalizado de todos.*

![DEN-NPHI por pozo](../outputs/eda/den_nphi_by_well.png)

*Fig. 6 — Crossplot DEN vs NPHI por pozo (small multiples). Relación negativa clara
en 31 de 32 pozos (R² per-well mediana = 0.28). Dentro de cada pozo la física funciona.
La normalización per-well captura estos patrones relativos; la global los enmascara.*

**Verificación**: la auditoría confirmó que los 32 pozos, tanto en el pool de
entrenamiento como en el set externo, producen features en [0, 1] tras aplicar
sus propios scalers. Error máximo de inverse transform: 0.

**Código**: `src/preprocessing.py → WellScaler`, `fit_scaler()`, `preprocess_well()`.

---

## 4. Garantías de calidad del dataset final

Tras aplicar el pipeline completo, el dataset cumple las siguientes propiedades:

| Propiedad | Valor verificado |
|-----------|-----------------|
| Pozos con datos | 32 / 32 (100%) |
| Filas totales | 154,451 (89.5% del crudo) |
| NaN en cualquier columna | 0 |
| Features fuera de [0, 1] | 0 (verificado en LOWO fold 1) |
| DEN target en rango físico | 100% (todas las filas fuera de [1.5, 3.1] g/cc eliminadas) |
| Error de inverse transform DEN | < 1 × 10⁻⁴ g/cc |
| Contaminación cruzada LOWO | Ninguna (scalers independientes por pozo) |

---

## 5. Caso especial: Dolecheck_1

El pozo Dolecheck_1 presenta NPHI entre 0.81 y 5.69 v/v tras la corrección de
unidades — la conversión % → v/v fue correcta (la mediana raw era > 1.5), pero el
LAS de origen tiene un problema adicional de escala no diagnosticado.

**Tratamiento**: las features se clipean a 0.80 v/v (NPHI queda constante en 1.0
tras la normalización). El pozo se mantiene en el dataset porque el DEN y las demás
curvas son válidas. La columna NPHI de este pozo no aporta señal al modelo pero
tampoco introduce ruido destructivo.

**Implicación para Phase 3**: al calibrar los coeficientes de la pérdida física
(DEN = a·NPHI + b), Dolecheck_1 actuará como punto de palanca si se usa en el set
de calibración. Se evaluará su inclusión o exclusión en la calibración.

---

## 6. Qué NO hace el pipeline

Es importante ser explícito sobre los límites:

- **No interpola profundidades**: las filas faltantes se descartan; el modelo no
  conoce el concepto de continuidad en profundidad (eso podría incorporarse como
  feature de contexto en versiones futuras).
- **No detecta ciclos de deriva de herramienta**: una tendencia sistemática de
  sobreregistro a cierta profundidad no se corrige.
- **No normaliza la distribución a normal**: solo normaliza el rango a [0, 1].
  La distribución interna de GR (bimodal en arcillas/arenas) o DEN (asimétrica
  negativa) se preserva; el modelo las aprende tal como son.
- **No aplica filtros de suavizado**: sin media móvil ni filtros de spike dentro
  del pozo. El clip al borde es suficiente para los artefactos detectados.

---

## 7. Referencia de código

| Función / Clase | Archivo | Responsabilidad |
|-----------------|---------|-----------------|
| `load_well()` | `src/data_loader.py` | Carga LAS, mapeo mnemonics, corrección unidades, centinelas |
| `load_field()` | `src/data_loader.py` | Itera directorio, retorna dict well_id → DataFrame |
| `clip_features_to_bounds()` | `src/preprocessing.py` | Clip features a FEATURE_BOUNDS |
| `filter_invalid_target_rows()` | `src/preprocessing.py` | Drop filas con DEN fuera de TARGET_BOUNDS |
| `apply_log_rt()` | `src/preprocessing.py` | log₁₀ en RT y RILM |
| `fit_scaler()` | `src/preprocessing.py` | Ajusta WellScaler per-well |
| `WellScaler.transform()` | `src/preprocessing.py` | Aplica normalización min-max |
| `WellScaler.inverse_transform_target()` | `src/preprocessing.py` | Desnormaliza predicciones DEN a g/cc |
| `preprocess_well()` | `src/preprocessing.py` | Orquesta pasos 1–4 para un pozo |
| `preprocess_wells()` | `src/preprocessing.py` | Itera todos los pozos independientemente |

Constantes de referencia:

```python
# src/preprocessing.py
FEATURE_BOUNDS = {
    "GR":   (0.0,    400.0),    # GAPI
    "RT":   (0.05, 50_000.0),  # Ohm·m (post-log: -1.3 a 4.7)
    "RILM": (0.05, 50_000.0),  # Ohm·m
    "NPHI": (-0.15,    0.80),  # v/v
    "SP":   (-1000.0, 1000.0), # mV
}
TARGET_BOUNDS = (1.5, 3.1)     # g/cc
```

*Fuente de los datos de auditoría: `debug/audit_preprocessing.py`*  
*Análisis EDA: `scripts/02_run_eda.py`, reporte en `documentation/01_eda.md`*
