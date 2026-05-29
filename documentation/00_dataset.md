# Dataset — Campo Kraft Prusa

## Origen

**Fuente:** Kansas Geological Survey (KGS) — Oil and Gas Wells Database  
**Campo:** Kraft Prusa, condado de Barton, Kansas, Estados Unidos  
**Formato original:** LAS 2.0 (Log ASCII Standard)

---

## Proceso de selección

Se evaluaron 4 campos del KGS. Kraft Prusa fue seleccionado por ser el de mayor cobertura de curvas y cantidad de pozos viables:

| Campo | Pozos totales | Pozos con 4 curvas | Pozos válidos |
|-------|--------------|-------------------|---------------|
| Kraft Prusa | 54 | 32 | 25 |
| Arroyo | — | — | < 8 |
| Cooper | — | — | < 8 |
| Thrall Aagard | — | — | < 8 |

Criterios de exclusión aplicados por `scripts/01_inspect_las.py` y `src/data_loader.py`:

- Curva faltante (alguna de las 6 requeridas ausente)
- Más del 10% de valores NaN en cualquier curva requerida
- Menos de 100 filas limpias tras eliminar NaN

---

## Estadísticas del dataset final

| Métrica | Valor |
|---------|-------|
| Pozos válidos | 25 |
| Registros totales (estimado) | ~157,000 |
| Promedio de registros por pozo | ~6,280 |
| Profundidad mínima | ~0 ft |
| Profundidad máxima | ~3,480 ft |
| Unidades de profundidad | **Pies (ft)** |
| Paso de muestreo | 0.5 ft (mayoría de pozos) |

---

## Mapeo de curvas: mnemonics reales → nombres canónicos

El código aplica este mapeo en tiempo de carga (`src/data_loader.py`, `_MNEMONIC_MAP`):

| Mnemónico en LAS | Nombre canónico | Descripción | Unidad | Rol |
|-----------------|-----------------|-------------|--------|-----|
| `GR` | `GR` | Gamma Ray | GAPI | Entrada |
| `RILD` | `RT` | Resistividad profunda (Deep Induction) | Ohm·m | Entrada |
| `RILM` | `RILM` | Resistividad media (Medium Induction) | Ohm·m | Entrada |
| `CNLS`, `CNPOR`, `CNSS`, `CNDL`, `NPOR`, `NEU`, `NEUT`, `PHIN`, `CNCF`, `TNPH` | `NPHI` | Neutrón compensado | v/v | Entrada |
| `SP` | `SP` | Potencial espontáneo | mV | Entrada |
| `RHOB` | `DEN` | Densidad bulk | g/cc | **Objetivo** |

> El mnemónico predominante en Kraft Prusa para neutrón es `CNLS`. Se mapean 10 variantes para cubrir variaciones entre pozos.

---

## Curvas descartadas como entradas

| Curva | Razón de exclusión |
|-------|--------------------|
| `DPOR` / `RHOC` | Derivadas directamente de `RHOB` — data leak garantizado |
| `DT` | Sónico — cobertura ~40% en los 25 pozos válidos |
| `CALI` | Calibre del hoyo — no aporta información petrrofísica relevante para DEN |
| `DCAL` | Corrección de calibre — derivada de CALI |
| `RLL3` / `RXORT` | Resistividades adicionales — alta correlación con RILD y RILM; no agregan información independiente |

---

## Rangos físicos esperados por curva

Rangos derivados del inventario sobre los 25 pozos válidos:

| Curva | Mín típico | Máx típico | Notas |
|-------|-----------|-----------|-------|
| GR | ~10 GAPI | ~150 GAPI | Arenas limpias ~20–40, lutitas ~80–150 |
| RT (RILD) | ~0.3 Ohm·m | ~500 Ohm·m | Distribución log-normal → se aplica log₁₀ |
| RILM | ~0.3 Ohm·m | ~500 Ohm·m | Distribución log-normal → se aplica log₁₀ |
| NPHI | ~0.05 v/v | ~0.45 v/v | Aumenta con arcillosidad y porosidad |
| SP | ~−80 mV | ~+20 mV | Negativo en arenas, cercano a cero en lutitas |
| DEN (RHOB) | ~2.0 g/cc | ~2.9 g/cc | Agua ~1.0, lutita ~2.6, caliza ~2.71 |

---

## Decisión de preprocesamiento

| Decisión | Valor | Razón |
|----------|-------|-------|
| Transformación logarítmica | Aplicada a RT y RILM | Distribución log-normal; log₁₀ linealiza el rango y mejora gradientes |
| Normalización | Min-max per-well | Preserva variaciones relativas dentro del pozo; evita sesgo de pozos con rangos extremos |
| Tratamiento de NaN | `dropna` sobre las 6 curvas canónicas | Cualquier fila con valor faltante en cualquier curva requerida se descarta |
| Sentinelas LAS | `{-9999.0, -999.25, -9999.25, -999.0}` → NaN | Valores nulos estándar del formato LAS 2.0 |

---

## Restricción física DEN–NPHI

La relación lineal entre densidad bulk y porosidad neutrón es la base del término regularizador del PINN:

```
DEN_esperada = a · NPHI + b
```

Los coeficientes `a` y `b` se calibran empíricamente mediante regresión lineal sobre todos los pozos del dataset. Los valores finales se determinan al ejecutar `scripts/02_run_eda.py` y quedan registrados en `outputs/eda/den_nphi_coefficients.csv`.

> **Pendiente:** actualizar esta sección con los coeficientes finales tras ejecutar el EDA en Docker.

---

*Fuente del código de carga: `src/data_loader.py`*  
*Fuente del código de preprocesamiento: `src/preprocessing.py`*
