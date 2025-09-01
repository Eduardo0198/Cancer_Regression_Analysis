<img width="950" height="200" alt="image" src="https://github.com/user-attachments/assets/040b69f6-96a2-4a9b-929b-b1e523766df6" />

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

El objetivo de este proyecto es **predecir el tamaño del área media del tumor** utilizando un modelo de regresión lineal implementado desde cero. Se busca explorar el impacto de las características seleccionadas sobre la variable objetivo y evaluar el desempeño del modelo mediante **MSE** y **R²**.

---

## Procedimiento

### ETL y Preprocesamiento de Datos

1. **Carga de datos:**  
   Se leyó el archivo CSV original y se creó un DataFrame de pandas.

2. **Análisis inicial:**  
   Se exploraron valores nulos y tipos de datos. La columna `diagnosis` fue eliminada para el modelo, ya que no se utiliza en la predicción de `area_mean`.

3. **Selección de variables:**  
   Se eligieron las variables más representativas basadas en correlación y relevancia clínica:  
   - `radius_mean`, `perimeter_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`, `texture_mean`.

4. **Normalización:**  
   - Las variables independientes fueron escaladas con **Z-score**:  
     \[
     Z = \frac{X - \text{media}}{\text{desviación estándar}}
     \]  
   - Esto asegura que todas las variables tengan la misma escala y evita que una domine el gradiente descendente.

5. **División de datos:**  
   - **80% entrenamiento** y **20% prueba**, para evaluar la generalización del modelo.

---

### Implementación del Modelo

1. Se implementó un modelo de **regresión lineal desde cero**, usando:  
   - Función hipótesis: \(y = X \cdot \theta + b\)  
   - Función de costo: **Error Cuadrático Medio (MSE)**  
   - Gradiente descendente: actualización iterativa de los pesos (\(\theta\)) y el bias (b) para minimizar el MSE.  

2. Se registró el MSE en cada época para monitorear la convergencia.

---

### Entrenamiento del Modelo

- **Learning rate:** 0.01  
- **Épocas:** 2000  
- Se utilizó todo el conjunto de entrenamiento para actualizar los pesos en cada iteración.

---

### Evaluación del Modelo

1. **Métricas obtenidas:**
   - MSE Entrenamiento: 2844.00  
   - MSE Prueba: 2081.69  
   - R² Entrenamiento: 0.9774  
   - R² Prueba: 0.9819

2. **Predicciones de ejemplo:**  
   Se compararon los valores reales y predichos de la variable `area_mean` para las primeras cinco muestras:

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

### Correlación entre variables
![Correlación](grafica_correlacion.png)

---

## Limitaciones

- El modelo solo captura relaciones lineales entre variables.  
- Algunas características podrían estar correladas, lo que puede afectar la interpretación de los pesos individuales.  
- Se usó un conjunto limitado de variables seleccionadas manualmente; incluir más variables podría mejorar el desempeño.  

---

## Conclusión

La regresión lineal implementada desde cero logró predecir correctamente el área media de los tumores, con un **R² superior a 0.97** en entrenamiento y prueba, indicando que el modelo explica la mayoría de la varianza de los datos.  

El enfoque permite comprender cómo cada característica impacta en la predicción y sirve como base para futuras mejoras, como probar técnicas de selección automática de variables, regularización o modelos más complejos.
