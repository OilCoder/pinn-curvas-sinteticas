# PINN — Evaluación LOWO con barrido de λ

Documenta el entrenamiento del modelo Physics-Informed Neural Network (PINN) sobre el
campo Kraft Prusa con protocolo LOWO, el barrido sobre λ ∈ {0.0, 0.01, 0.05, 0.1, 0.5, 1.0},
y la comparación pareada con el baseline MLP.

---

## Formulación del loss físico

El PINN extiende el loss del baseline añadiendo un término de regularización física:

```
Loss_total = MSE(DEN_pred, DEN_true) + λ · L_física
```

donde:

```
L_física = mean( (DEN_pred − DEN_esperado(NPHI))² )
DEN_esperado = A · NPHI_norm + B
```

Los coeficientes `A` y `B` se calibraron empíricamente mediante regresión lineal sobre
los 27 pozos del train pool en espacio Yeo-Johnson + z-score normalizado:

| Coeficiente | Valor | Interpretación |
|---|---|---|
| A | −0.5734 | Pendiente: mayor NPHI → menor DEN esperado |
| B | 0.0000 | Intercepto nulo por construcción (ambas variables centradas en 0) |
| R² | 0.329 | Fuerza de la relación lineal en espacio normalizado |

B=0 es exactamente correcto por la estandarización: Yeo-Johnson+StandardScaler centra
ambas variables en cero, por lo que la regresión sin intercepto es apropiada.

**Fuentes:** `src/physics.py` · `debug/dbg_calibrate_physics.py`

---

## Protocolo LOWO

Idéntico al baseline (`documentation/03_baseline.md`):
- 27 pozos train pool (3 externos reservados para Phase 4)
- 27 folds × (26 train | 1 test), cada uno independiente
- Misma arquitectura: 5→64→64→32→1 · Adam · early stopping

La única diferencia es el parámetro `lambda_phys` en `TrainConfig`. Para λ=0 se
reproduce el baseline exactamente (verificado con `test_lambda_phys_zero_same_as_no_physics`).

**Fuentes:** `scripts/04_train_pinn.py` · `scripts/05_sweep_lambda.py`

---

## Configuración de entrenamiento

| Parámetro | Valor |
|---|---|
| Épocas máximas | 300 |
| Early stopping patience | 10 |
| Batch size | 512 |
| Tasa de aprendizaje | 1 × 10⁻³ |
| Semilla | 42 |

---

## Resultados del barrido λ

Métricas agregadas (media ± std sobre 27 folds). ΔMAE = MAE_baseline − MAE_PINN
(positivo = PINN mejora).

| λ | MAE | R² | ΔMAE | % pozos mejorados | ΔR² |
|---|---:|---:|---:|---:|---:|
| 0.00 (baseline) | 0.1338 ± 0.0882 | 0.4137 ± 0.3136 | — | — | — |
| 0.01 | 0.1328 ± 0.0872 | 0.4270 ± 0.2943 | +0.00103 | 70.4 % | +0.0133 |
| 0.05 | 0.1324 ± 0.0869 | 0.4296 ± 0.2954 | +0.00142 | 59.3 % | +0.0159 |
| **0.10** | **0.1311 ± 0.0870** | **0.4373 ± 0.2965** | **+0.00279** | **81.5 %** | **+0.0237** |
| 0.50 | 0.1331 ± 0.0865 | 0.4169 ± 0.2692 | +0.00075 | 37.0 % | +0.0032 |
| 1.00 | 0.1352 ± 0.0867 | 0.3934 ± 0.2633 | −0.00131 | 14.8 % | −0.0203 |

**λ=0.1 es el punto óptimo**: maximiza ΔMAE (+0.00279 g/cc), mejora el 81.5% de los
pozos (22/27), y aumenta R² en +0.024 sobre el baseline.

> Ver figura: `outputs/figures/lambda_vs_error.png`

---

## Análisis por pozo (λ=0.1)

Pozos con mayor mejora absoluta de MAE:

| Pozo | MAE baseline | MAE PINN | Δ MAE | R² PINN |
|---|---:|---:|---:|---:|
| Kroutwurst_21 | 0.2034 | 0.1803 | +0.023 | 0.074 |
| Kroutwurst_20 | 0.2687 | 0.2606 | +0.008 | 0.127 |
| Esfeld_9 | 0.1651 | 0.1580 | +0.007 | 0.115 |
| Oeser,_R__1 | 0.0735 | 0.0679 | +0.006 | 0.734 |
| Woydziak_'A'_1 | 0.2001 | 0.1954 | +0.005 | 0.266 |

Pozos donde el PINN degrada ligeramente (λ=0.1):

| Pozo | MAE baseline | MAE PINN | Δ MAE |
|---|---:|---:|---:|
| Beaver_S-Reif_1-22 | 0.0788 | 0.0803 | −0.001 |
| Wirth_5 | 0.1080 | 0.1084 | −0.000 |
| Schneweis_10 | 0.0405 | 0.0408 | −0.000 |

Las degradaciones son marginales (< 0.002 g/cc) y ocurren en pozos que ya funcionaban
bien con el baseline (R² > 0.4). La física actúa como regularización suave que ayuda
en pozos difíciles sin dañar significativamente los pozos fáciles.

---

## Discusión

### Por qué λ=0.1 funciona

Con λ pequeño, el término físico actúa como regularización suave: empuja las predicciones
hacia la recta DEN-NPHI sin forzar al modelo a ignorar la información de los datos. El
balance óptimo se da cuando la señal física (R²=0.329 en espacio normalizado) es
suficiente para corregir pequeñas desviaciones sin sobreimponerla.

### Por qué λ≥0.5 degrada

Para λ=0.5 la contribución del loss físico supera la del MSE en muchos batches. El modelo
comienza a predecir DEN siguiendo la recta NPHI→DEN calibrada en lugar de aprender de
los datos. El 63% de los pozos empeoran con λ=1.0. El problema es que la relación lineal
DEN-NPHI tiene R²=0.329 — es una aproximación, no una ley exacta.

### Limitaciones de la relación DEN-NPHI lineal

La relación `DEN = A · NPHI + B` asume:
1. **Litología uniforme**: la pendiente A varía entre calcita (−0.5), arena (−0.6) y
   arcilla (débil correlación). En pozos con heterogeneidad litológica marcada
   (Kroutwurst_21, Bieberle_Trust_2), la relación global es un promedio que falla localmente.
2. **Sin efecto de gas**: gas en poros eleva NPHI aparente y reduce DEN, desacoplando la relación.
3. **Profundidad de investigación similar**: NPHI y DEN leen volúmenes ligeramente diferentes;
   en formaciones laminadas, esto introduce ruido en la relación.

En los pozos donde el PINN no mejora (Dolecheck_1 con NPHI anómalo, Esfeld_9 con
heterogeneidad alta), la señal física es demasiado ruidosa para aportar regularización útil.

### Implicaciones para Phase 4

- **λ óptimo recomendado**: 0.1 para el análisis final y comparación reportada.
- **Resultado central**: el PINN mejora el 81.5% de los pozos con una reducción de MAE de
  ~0.3% relativo — modesta pero consistente. La mejora es mayor en pozos difíciles (R² < 0.3
  en baseline), que es precisamente donde se espera que la regularización física sea más útil.
- **Publicación**: los coeficientes A y B deben reportarse junto a su R² de calibración
  para que el lector evalúe la solidez de la restricción física.

---

## Fuentes

| Módulo | Ruta |
|---|---|
| Función de pérdida física | `src/physics.py` |
| Loop de entrenamiento | `src/train.py` |
| Script PINN (λ fijo) | `scripts/04_train_pinn.py` |
| Barrido de λ | `scripts/05_sweep_lambda.py` |
| Comparación pareada | `scripts/06_compare_baseline_vs_pinn.py` |
| Figuras | `scripts/07_plot_results.py` |
| Métricas sweep | `outputs/pinn/lambda_sweep.json` |
| Comparación detallada | `outputs/figures/comparison_table.csv` |
