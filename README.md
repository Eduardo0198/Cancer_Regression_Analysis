
<img width="950" height="200" alt="image" src="https://github.com/user-attachments/assets/be0944e4-6832-4f6c-88a1-86610d295907" />

# Cancer_Regression_Analysis
# Predicción de Área de Tumores usando Modelos de Regresión

## Descripción
Este proyecto tiene como objetivo predecir el área promedio de tumores (`area_mean`) a partir de un conjunto de características celulares, utilizando tres enfoques de regresión diferentes. Se comparan modelos manuales y basados en frameworks para evaluar su desempeño, interpretar resultados y analizar el impacto de la optimización de hiperparámetros. Además, se consideran aspectos éticos relacionados con el uso de datos médicos sensibles.

## Archivos del proyecto
1. **`main_lineal_regression.py`**  
   Implementa una regresión lineal múltiple desde cero, utilizando gradiente descendente y early stopping. Permite analizar el comportamiento del aprendizaje paso a paso, incluyendo la convergencia del error, bias y varianza.

2. **`main_rf.py`**  
   Contiene la implementación de un Random Forest Regressor básico usando `scikit-learn` con hiperparámetros fijos. Este modelo sirve como referencia para comparar desempeño frente a la regresión lineal manual y evaluar la capacidad de generalización de un modelo de ensamble.

3. **`main_rf_gs.py`**  
   Implementa un Random Forest optimizado mediante GridSearchCV, explorando combinaciones de hiperparámetros para maximizar el R² y minimizar el error. Incluye análisis de importancia de features y gráficos de predicciones vs valores reales para interpretar mejor los resultados.

## Requisitos
- Python 3.9 o superior
- Librerías: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

## Uso
1. Colocar el dataset `Cancer_Data.csv` en la misma carpeta que los scripts.  
2. Ejecutar cada script por separado para entrenar y evaluar los modelos.  
3. Los resultados incluyen métricas (R², MSE), gráficos de predicción vs valores reales e importancia de features.  

## Licencia
Este proyecto se realiza con fines académicos y educativos. Los datos utilizados son simulados o anonimizados para garantizar privacidad.
