
<img width="950" height="200" alt="image" src="https://github.com/user-attachments/assets/be0944e4-6832-4f6c-88a1-86610d295907" />

# Cancer_Regression_Analysis

Implementación manual de un algoritmo de regresión lineal sin el uso de frameworks de machine learning.  

## Descripción del dataset

El dataset contiene información de características celulares obtenidas a partir de imágenes de tumores, con el objetivo de predecir el tamaño del área media del tumor (`area_mean`). El dataset está organizado de la siguiente manera:

| Variable                  | Descripción |
|----------------------------|-------------|
| `radius_mean`              | Radio promedio de las células |
| `perimeter_mean`           | Perímetro promedio de las células |
| `smoothness_mean`          | Suavidad promedio de las células |
| `compactness_mean`         | Compacidad promedio |
| `concavity_mean`           | Concavidad promedio |
| `texture_mean`             | Textura promedio |
| `area_mean`                | Área promedio de las células (variable objetivo) |

Fuente: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

## Objetivo

Predecir el tamaño del área media del tumor utilizando un modelo de regresión lineal implementado desde cero, evaluando el impacto de las características seleccionadas sobre la variable objetivo mediante **MSE** y **R²**, y analizar la calidad de las predicciones mediante visualizaciones gráficas.

---

## Procedimiento

### ETL y Preprocesamiento de Datos

1. **Carga de datos:** Se leyó el archivo CSV y se creó un DataFrame de pandas.  
2. **Análisis inicial:** Se exploraron valores nulos y tipos de datos; la columna `diagnosis` se eliminó para la predicción de `area_mean`.  
3. **Selección de variables:** Variables más representativas basadas en correlación y relevancia clínica:  
   - `radius_mean`, `perimeter_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`, `texture_mean`.  
4. **Normalización:** Variables escaladas con **Z-score** para mantener la misma escala y evitar dominancia en gradiente descendente.  
5. **División de datos:** 80% entrenamiento, 20% prueba.

---

### Implementación del Modelo

- Modelo de **regresión lineal desde cero**:  
  - Función hipótesis: \(y = X \cdot \theta + b\)  
  - Función de costo: **MSE**  
  - Gradiente descendente: actualización iterativa de \(\theta\) y b.  
- Registro de MSE por época para seguimiento de convergencia.

---

### Entrenamiento

- **Learning rate:** 0.01  
- **Épocas:** 2000  

---

### Evaluación

1. **Métricas obtenidas:**
   - MSE Entrenamiento: 2844.00  
   - MSE Prueba: 2081.69  
   - R² Entrenamiento: 0.9774  
   - R² Prueba: 0.9819

2. **Predicciones de ejemplo:**

| Real  | Predicho |
|-------|----------|
| 481.90 | 492.81  |
| 1130.00 | 1122.88 |
| 748.90 | 809.41  |
| 467.80 | 471.51  |
| 402.90 | 391.12  |

---

## Resultados gráficos

### Evolución del error durante el entrenamiento
![Evolución MSE](grafica_mse.png)  
Esta gráfica muestra cómo el **Error Cuadrático Medio (MSE)** disminuye iteración por iteración durante el entrenamiento. Una pendiente descendente estable indica que el modelo está aprendiendo correctamente, que los pesos convergen hacia valores óptimos y que el gradiente descendente está funcionando de manera efectiva. La ausencia de oscilaciones grandes sugiere que el learning rate elegido es adecuado.

### Correlación entre variables
![Correlación](grafica_correlacion.png)  
Visualiza las relaciones lineales entre las variables independientes y la variable objetivo. Valores cercanos a 1 o -1 indican alta correlación positiva o negativa. Esta gráfica es útil para identificar redundancias entre variables, detectar posibles multicolinealidades y guiar la selección de características más significativas para la predicción.

### Predicciones vs valores reales
![Predicciones vs Reales](predicciones_vs_reales.png)  
Muestra la relación entre los valores predichos por el modelo y los valores reales del área media de los tumores. Los puntos deberían alinearse cerca de la línea diagonal (y=x), lo que indicaría predicciones precisas. Desviaciones significativas pueden indicar casos atípicos o limitaciones del modelo lineal.

### Distribución de errores (residuals)
![Distribución de errores](distribucion_error.png)  
Esta gráfica muestra los errores (residuals) de las predicciones. Una distribución centrada alrededor de cero y sin patrones visibles indica que el modelo no presenta sesgo sistemático, es decir, no subestima ni sobreestima consistentemente los valores. También permite detectar heterocedasticidad o outliers.

---

## Limitaciones

- Captura únicamente relaciones lineales entre variables; no modela interacciones complejas o no lineales.  
- Algunas características podrían estar correlacionadas, afectando la interpretación de pesos individuales.  
- Conjunto de variables seleccionado manualmente; podrían incluirse más características relevantes.  
- Sensible a valores atípicos (outliers).  
- Validación simple (entrenamiento/prueba), sin cross-validation ni técnicas robustas de evaluación.  
- Dataset relativamente pequeño y específico, lo que limita la generalización.  
- No se incorporan regularizaciones ni métricas adicionales como MAE o RMSLE.

---

## Conclusión

El modelo de regresión lineal implementado desde cero logró predecir correctamente el área media de los tumores con un **R²>0.97**, explicando la mayoría de la varianza de los datos. Las gráficas muestran que el modelo converge adecuadamente, produce predicciones consistentes y no evidencia sesgo sistemático. Este enfoque permite comprender cómo cada característica impacta en la predicción y brinda un aprendizaje práctico sobre implementación manual de gradiente descendente y análisis de errores. Aunque los resultados son satisfactorios, el modelo puede beneficiarse de mayor complejidad, selección automática de variables, y técnicas de regularización para mejorar la generalización y precisión.

---

## Mejoras a futuro

Para futuras versiones, se plantea implementar regularización (Ridge, Lasso) para reducir sobreajuste, validar el modelo con cross-validation para evaluar estabilidad y robustez, explorar modelos no lineales como regresión polinómica, árboles de decisión o redes neuronales simples para capturar relaciones más complejas, automatizar la selección de variables mediante técnicas estadísticas o de feature importance, reducir correlaciones problemáticas y outliers mediante preprocesamiento avanzado, y desarrollar dashboards interactivos para visualizar predicciones, errores y residuales de manera dinámica. Además, se considerará integrar métricas complementarias (MAE, RMSLE) para evaluar mejor la precisión y confiabilidad del modelo, y ampliar el dataset para mejorar la generalización a nuevos casos clínicos.
