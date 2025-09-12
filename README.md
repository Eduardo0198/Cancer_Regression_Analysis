
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



2. **Predicciones de ejemplo:**


### Distribución de errores (residuals)
![Distribución de errores](distribucion_error.png)  


## Mejoras a futuro

Para futuras versiones, se plantea implementar regularización (Ridge, Lasso) para reducir sobreajuste, validar el modelo con cross-validation para evaluar estabilidad y robustez, explorar modelos no lineales como regresión polinómica, árboles de decisión o redes neuronales simples para capturar relaciones más complejas, automatizar la selección de variables mediante técnicas estadísticas o de feature importance, reducir correlaciones problemáticas y outliers mediante preprocesamiento avanzado, y desarrollar dashboards interactivos para visualizar predicciones, errores y residuales de manera dinámica. Además, se considerará integrar métricas complementarias (MAE, RMSLE) para evaluar mejor la precisión y confiabilidad del modelo, y ampliar el dataset para mejorar la generalización a nuevos casos clínicos.
