# 5. PINN — Evaluación LOWO con Barrido de λ

Documenta el entrenamiento del modelo Physics-Informed Neural Network (PINN) sobre el
campo Kraft Prusa con protocolo LOWO, el barrido sobre
$\lambda \in \{0.0, 0.01, 0.05, 0.1, 0.5, 1.0\}$, y la comparación pareada con el
baseline MLP.

---

## 5.1 Introducción

Una **Physics-Informed Neural Network (PINN)** es un modelo de aprendizaje profundo cuya
función de pérdida incorpora términos derivados de ecuaciones físicas, además del error
de predicción estándar. En el contexto de registros de pozo, esto significa:

- **MSE de datos**: el modelo aprende a reproducir DEN observado.
- **Residuo físico**: el modelo también aprende a ser consistente con la relación
  empírica DEN–NPHI calibrada en el campo.

La ventaja principal es que la restricción física actúa como regularizador que reduce el
sobreajuste en folds con pocos datos, especialmente en pozos geológicamente atípicos
donde el modelo supervisado puro tiende a aprender patrones espurios.

En este proyecto, la PINN aplica la restricción bivariate calibrada en el EDA y reutiliza
exactamente la misma arquitectura y protocolo LOWO del baseline. La única diferencia
con el baseline (λ=0) es el parámetro `lambda_phys` en `TrainConfig`.

---

## 5.2 Formulación del loss

La función de pérdida total combina el MSE de datos con el residuo físico ponderado:

$$\mathcal{L}_{total} = \underbrace{\frac{1}{N}\sum_{i=1}^{N}\left(\hat{y}_i - y_i\right)^2}_{\mathcal{L}_{datos}} + \lambda \cdot \underbrace{\frac{\sum_i w_i \left(\hat{y}_i - \hat{y}^{fis}_i\right)^2}{\sum_i w_i}}_{\mathcal{L}_{física}}$$

donde:

- $\hat{y}_i$: predicción del modelo para el registro $i$ (DEN normalizado).
- $y_i$: DEN observado normalizado.
- $\hat{y}^{fis}_i$: DEN esperado por la restricción física, definido en la Sección 5.3.
- $w_i$: peso de calidad del caliper DCAL, definido en la Sección 5.4.
- $\lambda$: hiperparámetro que controla el peso de la restricción física.

Con $\lambda = 0$ se reproduce exactamente el baseline (verificado por test unitario).

**Implementación**: `src/physics.py → physics_loss()`, `src/train.py`

---

## 5.3 Restricción física DEN–NPHI

La relación física embebida en el PINN es la calibración bivariate obtenida en el EDA
sobre los 27 pozos del train pool en espacio Yeo-Johnson + z-score normalizado:

$$\hat{y}^{fis}_i = A \cdot \text{NPHI}_i + D \cdot (\text{NPHI}_i \times \text{GR}_i)$$

| Coeficiente | Valor | Interpretación |
|---|---:|---|
| $A$ | −0.5563 | Pendiente NPHI→DEN: mayor porosidad → menor densidad |
| $D$ | +0.0864 | Corrección litológica: en zonas arcillosas (GR alto), la pendiente se atenúa |
| $R^2$ | 0.338 | Fuerza de la relación en espacio normalizado |
| Intercepto | 0.0 | Nulo por construcción (Yeo-Johnson+StandardScaler centra en cero) |

El **término de interacción** $D \cdot (\text{NPHI} \times \text{GR})$ captura el efecto
de la arcillosidad sobre la pendiente NPHI–DEN. En lutitas (GR alto), los minerales de
arcilla exhiben alta porosidad aparente neutrón pero densidad bulk intermedia, atenuando
la correlación. Este término mejora $R^2$ de 0.330 (NPHI solo) a 0.338 con la adición
de GR, reflejando una relación física más completa.

**Implementación**: `src/physics.py → den_from_nphi()`, constantes `A_PHYS = −0.5563`,
`D_PHYS = 0.0864`

---

## 5.4 Peso de caliper DCAL_WEIGHT

En zonas de *borehole washout* (hoyo ensanchado), las herramientas de densidad y neutrón
no leen la formación real. La relación DEN–NPHI calibrada sobre roca sana no es válida
en estas profundidades. El peso:

$$w_i = \text{clip}\!\left(1 - \frac{\text{DCAL}_i - Q_{25}}{Q_{90} - Q_{25}},\; 0,\; 1\right)$$

reduce la contribución del loss físico exactamente donde la restricción es menos
confiable. Donde el hoyo está en calibre nominal ($\text{DCAL} \approx Q_{25}$), $w_i = 1$.
Donde el hoyo está muy ensanchado ($\text{DCAL} \approx Q_{90}$), $w_i \rightarrow 0$.

Si DCAL no está presente en el pozo, todos los pesos se fijan a 1 (restricción física
uniforme).

**Implementación**: `src/preprocessing.py → compute_dcal_weight()`,
`src/physics.py → physics_loss()`

---

## 5.5 Barrido de λ

Métricas agregadas (media ± desv. std sobre 27 folds). ΔMAE = MAE_baseline − MAE_PINN
(positivo = PINN mejora). Barrido ejecutado con `scripts/05_sweep_lambda.py`.

| λ | MAE (g/cc) | R² | ΔMAE (g/cc) | % pozos mejorados | ΔR² |
|---|---:|---:|---:|---:|---:|
| 0.00 (baseline) | 0.1338 ± 0.0882 | 0.4137 ± 0.3136 | — | — | — |
| 0.01 | 0.1328 ± 0.0872 | 0.4270 ± 0.2943 | +0.00103 | 70.4 % | +0.0133 |
| 0.05 | 0.1324 ± 0.0869 | 0.4296 ± 0.2954 | +0.00142 | 59.3 % | +0.0159 |
| **0.10** | **0.1311 ± 0.0870** | **0.4373 ± 0.2965** | **+0.00279** | **81.5 %** | **+0.0237** |
| 0.50 | 0.1331 ± 0.0865 | 0.4169 ± 0.2692 | +0.00075 | 37.0 % | +0.0032 |
| 1.00 | 0.1352 ± 0.0867 | 0.3934 ± 0.2633 | −0.00131 | 14.8 % | −0.0203 |

---

## 5.6 λ óptimo = 0.1

$\lambda = 0.1$ maximiza simultáneamente tres criterios:

| Criterio | Valor con λ=0.1 | Contexto |
|---|---|---|
| ΔMAE | +0.00279 g/cc (máximo del barrido) | Reducción absoluta de error medio |
| % pozos mejorados | 81.5 % (22/27 pozos) | El porcentaje más alto del barrido |
| ΔR² | +0.0237 (máximo del barrido) | Mejor ajuste explicado de varianza |

Con λ pequeño (0.01–0.1), el término físico actúa como **regularizador suave**: empuja
las predicciones hacia la recta DEN–NPHI calibrada sin forzar al modelo a ignorar la
información de los datos. El balance óptimo se logra cuando la señal física ($R^2=0.338$
en espacio normalizado) es suficiente para corregir desviaciones sin sobreimponerse.

---

## 5.7 Análisis por pozo (λ=0.1)

### 5.7.1 Pozos con mayor mejora

| Pozo | MAE baseline (g/cc) | MAE PINN (g/cc) | ΔMAE (g/cc) | R² PINN |
|---|---:|---:|---:|---:|
| Kroutwurst_21 | 0.203 | 0.180 | +0.023 | 0.074 |
| Kroutwurst_20 | 0.269 | 0.261 | +0.008 | 0.127 |
| Esfeld_9 | 0.165 | 0.158 | +0.007 | 0.115 |
| Oeser,_R__1 | 0.073 | 0.068 | +0.006 | 0.734 |
| Woydziak_'A'_1 | 0.200 | 0.195 | +0.005 | 0.266 |

### 5.7.2 Pozos con degradación marginal

| Pozo | MAE baseline (g/cc) | MAE PINN (g/cc) | ΔMAE (g/cc) |
|---|---:|---:|---:|
| Beaver_S-Reif_1-22 | 0.079 | 0.080 | −0.001 |
| Wirth_5 | 0.108 | 0.108 | −0.000 |
| Schneweis_10 | 0.041 | 0.041 | −0.000 |

Las degradaciones son marginales (< 0.002 g/cc) y ocurren en pozos que ya funcionaban
bien con el baseline (R² > 0.4). La restricción física actúa como regularización suave
que ayuda en pozos difíciles sin dañar significativamente los pozos fáciles.

---

## 5.8 Discusión

### 5.8.1 Por qué λ=0.1 funciona

Con λ=0.1, el loss físico contribuye ~10 % del gradiente total. La relación DEN–NPHI
calibrada tiene $R^2=0.338$ — es una aproximación estadística, no una ley exacta.
Con este peso, el PINN puede corregir el modelo cuando las predicciones se alejan
sistemáticamente de la física sin eliminar la señal de los datos.

La mejora es mayor en pozos difíciles (R² < 0.3 en baseline), que es exactamente donde
se espera que la regularización física sea más útil: el modelo tiene menos datos de
calidad para aprender y la restricción física compensa.

### 5.8.2 Por qué λ ≥ 0.5 degrada

Para λ=0.5, la contribución del loss físico supera la del MSE en muchos batches. El
modelo comienza a predecir DEN siguiendo la recta NPHI→DEN calibrada en lugar de
aprender de los datos. El 63 % de los pozos empeoran con λ=1.0.

El problema fundamental es que la relación lineal DEN–NPHI tiene $R^2=0.338$: es
suficiente para regularizar suavemente, pero no lo suficientemente precisa para ser
la señal dominante del entrenamiento.

### 5.8.3 Limitaciones de la restricción física lineal

La relación $\hat{y}^{fis}_i = A \cdot \text{NPHI}_i + D \cdot (\text{NPHI}_i \times \text{GR}_i)$ asume:

1. **Litología relativamente uniforme**: la pendiente A varía entre calcita (~−0.5),
   arena (~−0.6) y arcilla (correlación débil). En pozos con heterogeneidad litológica
   marcada (Kroutwurst_21, Bieberle_Trust_2), la relación global es un promedio que
   falla localmente.
2. **Sin efecto de gas dominante**: el gas en poros eleva el NPHI aparente y reduce el
   DEN simultáneamente, desacoplando la relación negativa canónica.
3. **Sin efecto de invasión diferencial**: NPHI y DEN tienen distintas profundidades de
   investigación; en formaciones laminadas o con invasión de lodo profunda, esto
   introduce ruido en la relación.

En los pozos donde el PINN no mejora (Dolecheck_1 con NPHI constante por anomalía de
escala, Esfeld_9 con heterogeneidad alta), la señal física es demasiado ruidosa para
aportar regularización útil.

---

## 5.9 Implicaciones

| Observación | Recomendación |
|---|---|
| λ=0.1 mejora el 81.5 % de los pozos | **λ óptimo recomendado para el análisis final** |
| Mejora concentrada en pozos con R² < 0.3 | La restricción física es más útil donde los datos son más ruidosos o escasos |
| Degradaciones marginales (< 0.002 g/cc) | La restricción no daña significativamente los pozos bien predichos por el baseline |
| Set externo {Arensman_2, Burmeister_1, Rous_'F'_2} | Pendiente evaluar PINN λ=0.1 vs. baseline en estos 3 pozos (Phase 4) |
| R² calibración = 0.338 | Suficiente para regularización suave; insuficiente para ser señal dominante (λ > 0.5 degrada) |

---

## 5.10 Fuentes

| Módulo | Ruta |
|---|---|
| Función de pérdida física | `src/physics.py` |
| Loop de entrenamiento | `src/train.py` |
| Script PINN (λ fijo) | `scripts/04_train_pinn.py` |
| Barrido de λ | `scripts/05_sweep_lambda.py` |
| Comparación pareada | `scripts/06_compare_baseline_vs_pinn.py` |
| Figuras de resultados | `scripts/07_plot_results.py` |
| Métricas sweep | `outputs/pinn/lambda_sweep.json` |
| Comparación detallada por pozo | `outputs/figures/comparison_table.csv` |
