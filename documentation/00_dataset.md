# 1. Dataset — Campo Kraft Prusa

Descripción del conjunto de datos utilizado para entrenar y evaluar la PINN de predicción
de densidad bulk (DEN). Cubre la selección del campo, el inventario de pozos, el mapeo de
curvas y las restricciones físicas que fundamentan el término regularizador.

---

## 1.1 Introducción

El campo **Kraft Prusa** (condado de Barton, Kansas, EE. UU.) fue seleccionado como
dataset de referencia para este proyecto por tres razones principales:

1. **Disponibilidad pública**: Kansas Geological Survey (KGS) publica los archivos LAS
   libremente a través de su base de datos de pozos de petróleo y gas.
2. **Cobertura de curvas**: es uno de los pocos campos del KGS con cobertura simultánea
   de GR, RILD/RILM, CNLS, SP y RHOB en un número estadísticamente significativo de pozos.
3. **Homogeneidad geológica**: los pozos comparten la misma formación objetivo (Arbuckle),
   lo que facilita el aprendizaje de patrones petrográficos consistentes por el modelo.

El **Kansas Geological Survey (KGS)** es una agencia científica de la Universidad de
Kansas que mantiene registros de pozos, datos de producción e información geológica del
estado desde 1889. Los datos son de acceso libre en formato LAS 2.0.

---

## 1.2 Proceso de selección de campo

Se evaluaron cuatro campos del KGS. La selección se basó en el número de pozos con las
seis curvas canónicas presentes, sin exceder el 10 % de valores NaN por curva.

| Campo | Pozos totales en KGS | Pozos con 6 curvas | Pozos válidos |
|---|---:|---:|---:|
| **Kraft Prusa** | 54 | 32 | **30** |
| Arroyo | — | — | < 8 |
| Cooper | — | — | < 8 |
| Thrall Aagard | — | — | < 8 |

Kraft Prusa fue el único campo con suficientes pozos válidos para el protocolo
Leave-One-Well-Out (LOWO). Los campos alternativos no alcanzaban la masa crítica
de ≥ 20 pozos que garantiza folds de entrenamiento representativos.

Los criterios de exclusión se aplican en `src/data_loader.py`:

- Curva faltante (alguna de las seis curvas canónicas ausente en el LAS).
- Menos de 100 filas limpias tras eliminar NaN en las curvas canónicas.
- Archivo LAS no parseable (cabecera malformada).

Adicionalmente, se eliminaron 2 registros duplicados detectados como divisiones
del mismo pozo físico:

| Archivo excluido | Motivo |
|---|---|
| `Weber_'A'_13_part1.las` | Duplicado parcial de `Weber_'A'_13.las` |
| `Frees-Burmeister_13_part1.las` | Duplicado parcial de `Frees-Burmeister_13.las` |

El dataset final contiene **30 pozos únicos**.

---

## 1.3 Inventario del dataset

El dataset se divide en dos conjuntos. El **set externo** (3 pozos) se reserva para
validación final después del entrenamiento completo; el **train pool** (27 pozos) se
usa exclusivamente en el protocolo LOWO.

### 1.3.1 Set externo (3 pozos — reservados)

Seleccionados con `src/lowo.py → field_split(seed=42)`. No participan en ningún fold
de entrenamiento ni en la calibración de hiperparámetros.

| Pozo | Registros (estimado) |
|---|---:|
| Arensman_2 | ~5,000 |
| Burmeister_1 | ~4,500 |
| Rous_'F'_2 | ~5,200 |

### 1.3.2 Train pool LOWO (27 pozos)

| Pozo | Pozo | Pozo |
|---|---|---|
| Beaver_S-Reif_1-22 | Bieberle_Trust_2 | Demel_3 |
| Dolecheck_1 | Esfeld_9 | Frees-Burmeister_13 |
| Grossardt_3 | Hoffman_2 | Hoffman_Trust_1 |
| Holder_'A'_5 | Kraft-Prusa_Unit_16 | Krier_'C'_6 |
| Kroutwurst_19 | Kroutwurst_20 | Kroutwurst_21 |
| Nadine_1 | Oeser_2 | Oeser,_R__1 |
| Rous_1-28 | Rupe-Woydziak_Unit_1 | Schneweis_10 |
| Schneweis_3 | Soeken_12 | Weber_'A'_13 |
| Wirth_5 | Woydziak_'A'_1 | Woydziak-Kirmer_Unit_1 |

### 1.3.3 Estadísticas generales

| Métrica | Valor |
|---|---|
| Pozos totales | 30 |
| Registros totales (estimado) | ~164,000 |
| Promedio de registros por pozo | ~5,470 |
| Rango de profundidad | 0 – 3,483 ft |
| Paso de muestreo | 0.5 ft (mayoría de pozos) |
| Unidad de profundidad | Pies (ft) |

---

## 1.4 Mapeo de curvas

El cargador `src/data_loader.py` aplica el diccionario `_MNEMONIC_MAP` para renombrar
las curvas del LAS a nombres canónicos internos. El primer mnemónico que coincida para
cada nombre canónico se usa; los duplicados se descartan.

| Mnemónicos en LAS | Nombre canónico | Descripción | Unidad | Rol |
|---|---|---|---|---|
| `GR` | `GR` | Gamma Ray | GAPI | Feature de entrada |
| `RILD` | `RT` | Resistividad profunda (Deep Induction) | Ohm·m | Feature de entrada |
| `RILM` | `RILM` | Resistividad media (Medium Induction) | Ohm·m | Feature de entrada |
| `CNLS`, `CNPOR`, `CNSS`, `CNDL`, `NPOR`, `NEU`, `NEUT`, `PHIN`, `CNCF`, `TNPH` | `NPHI` | Porosidad neutrón compensada | v/v | Feature de entrada |
| `SP` | `SP` | Potencial espontáneo | mV | Feature de entrada |
| `RHOB` | `DEN` | Densidad bulk | g/cc | **Variable objetivo** |
| `DCAL` | `DCAL` | Corrección diferencial de calibre | in | Peso de calidad (washout) |

> El mnemónico predominante en Kraft Prusa para la curva neutrón es `CNLS`.
> Se mapean 10 variantes para cubrir variaciones de herramienta y época de registro.
> DCAL es opcional: si está presente en el LAS, se carga y se usa para filtrar
> intervalos de hoyo ensanchado (*borehole washout*); si no está, el pipeline opera
> sin él.

---

## 1.5 Rangos físicos esperados

Rangos derivados de la auditoría sobre los 30 pozos del dataset, con interpretación
geológica para el contexto de la formación Arbuckle y sedimentos clásticos de Kansas.

| Curva | Unidad | Mín típico | Máx típico | Interpretación geológica |
|---|---|---:|---:|---|
| GR | GAPI | ~10 | ~150 | Arenas limpias 20–40; lutitas 80–150; shales carbonosos hasta 300 |
| RT | Ohm·m | ~0.3 | ~10,000 | Distribución log-normal; carbonatos resistivos pueden superar 1,000 |
| RILM | Ohm·m | ~0.3 | ~10,000 | Igual que RT; lectura a menor profundidad de investigación |
| NPHI | v/v | ~0.05 | ~0.45 | Aumenta con arcillosidad y porosidad; gas puede producir valores bajos o negativos |
| SP | mV | ~−80 | ~+20 | Negativo en arenas permeables; próximo a cero en lutitas |
| DEN | g/cc | ~2.0 | ~2.9 | Agua ≈ 1.0; lutita ≈ 2.6; caliza densa ≈ 2.71; anhidrita ≈ 2.87 |
| DCAL | in | ~0 | ~3 | Diferencia entre calibre medido y nominal del trépano; > Q75+1.5·IQR indica washout |

---

## 1.6 Restricción física DEN–NPHI

La relación entre densidad bulk y porosidad neutrón es la base del término
regularizador de la PINN. La formulación bivariate incluye un término de interacción
con GR para capturar la atenuación litológica dependiente de arcillosidad:

$$\text{DEN}_{esperado} = A \cdot \text{NPHI} + D \cdot (\text{NPHI} \times \text{GR})$$

donde los coeficientes se calibran empíricamente por regresión lineal sobre los 27
pozos del train pool en espacio Yeo-Johnson + z-score normalizado:

| Coeficiente | Valor | Interpretación física |
|---|---:|---|
| $A$ | −0.5563 | Pendiente NPHI→DEN: mayor porosidad → menor densidad |
| $D$ | +0.0864 | Corrección litológica: en zonas arcillosas (GR alto), la pendiente se reduce |
| $R^2$ | 0.338 | Fuerza de la relación en espacio normalizado |

**El término de interacción** $D \cdot (\text{NPHI} \times \text{GR})$ mejora $R^2$ de 0.330
(modelo univariado NPHI solo) a 0.338. En litologías arcillosas (GR alto), los minerales
de arcilla tienen alta porosidad aparente neutrón pero densidad bulk intermedia,
atenuando la pendiente NPHI–DEN. El producto NPHI×GR captura este efecto.

**Intercepto nulo**: el intercepto es cero por construcción. La normalización
Yeo-Johnson + StandardScaler centra todas las variables en cero, por lo que la
regresión sin intercepto es la formulación correcta.

**El peso de caliper DCAL_WEIGHT**: en zonas de *washout* (hoyo ensanchado), las
herramientas de densidad y neutrón registran valores poco fiables. El peso

$$w_i = \text{clip}\!\left(1 - \frac{\text{DCAL}_i - Q_{25}}{Q_{90} - Q_{25}},\; 0,\; 1\right)$$

reduce la contribución del loss físico donde la relación DEN–NPHI es menos confiable.
El loss físico ponderado se implementa en `src/physics.py → physics_loss()`.

---

*Fuente del código de carga: `src/data_loader.py`*
*Fuente del código de preprocesamiento: `src/preprocessing.py`*
*Fuente de la calibración física: `src/physics.py`*
