
# ////////////////////////////////////////////////////
# jose eduardo viveros escamia - A01710605
# Regresión Lineal Múltiple desde Cero con Validación
# ////////////////////////////////////////////////////

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. Cargar datos
df = pd.read_csv(r"C:\Users\josed\Documents\Benj\Cancer_Data.csv")

def save_plot(nombre):
    ruta = os.path.join(carpeta, f"{nombre}.png")
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    print(f"Gráfica guardada en: {ruta}")

carpeta = "images_lr"
os.makedirs(carpeta, exist_ok=True)

# 2. Selección de variables

features_bar = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
                'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
                'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

target = "area_mean"

# Visualizar cuales fetures tienen mayor correlación con el target y cuales 
# pueden tener mucha colinealidad
cor_target = df[features_bar].corrwith(df[target]).abs().sort_values(ascending=False)

plt.figure(figsize=(10,5))
cor_target.plot(kind="bar", color="skyblue")
plt.title(f"Correlación absoluta de las features con {target}")
plt.ylabel("Correlación |r|")
plt.tight_layout()
save_plot("barras_todas_features.png")
plt.show()

# Visualisando los datos de la barras podemos elegir ahora si cuales fetures son los mejores
# para predecir el target y cuales no elegir debido a su baja correlación y a una alta colinealidad
features = [
    'texture_mean',       # Diversidad de textura
    'smoothness_mean',    # Suavidad
    'compactness_mean',   # Compacidad  
    'concavity_mean',     # Concavidad
    'symmetry_mean',      # Simetría
    'fractal_dimension_mean',  # Dimensión fractal
    'texture_worst',      # Peor textura
    'smoothness_worst',   # Peor suavidad
    'compactness_worst',  # Peor compacidad
    'concavity_worst',    # Peor concavidad
    'symmetry_worst',     # Peor simetría
    'fractal_dimension_se', # Peor dimensión fractal 
    'radius_mean',      # Radio promedio
#    'perimeter_mean',   # Perímetro promedio
#    'radius_se',    # Peor radio
#    'perimeter_se'   # Peor perímetro
]

# Las features que estan comentadas tienen una alta colinealidad con otras features
# y a pesar de que tienen una buena correlación con el target, no se consideran
# para evitar problemas de multicolinealidad en la regresión lineal, asi que en ese 
# sentido se prefiere tener menos features pero que sean independientes entre si, y solo agregamos 
# un de esas fetures para ayudar que la R_2 suba un poco mas -> 'radius_mean'

# Calcular matriz de correlación
corr_matrix = df[features].corr()

# Visualización
plt.figure(figsize=(15, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0)
plt.title("Matriz de correlación de features")
plt.tight_layout()
save_plot("matriz_fetures_seleccionadas.png")
plt.show()

# Selección de features y target
# para poder usarlas con las funcion de GD
X = df[features].values
y = df[target].values.reshape(-1, 1)


print(f"\nVariables independientes: {X.shape}")
print(f"Variable dependiente: {y.shape}")


# 3. Escalado de característica
# Escalamos las características para que todas esten en la misma magnitud
# Esto ayuda a la convergencia del gradiente descendente
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_original = y.flatten()


# 4. División de datos en las 3 secciones
# Train - (60%) 
# Validation - (20%) 
# Test - (20%)

X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y_original, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  
)

print(f"\n=== DIVISIÓN DE DATOS ===")
print(f" - Entrenamiento: {X_train.shape[0]} muestras")
print(f" - Validación: {X_val.shape[0]} muestras")
print(f" - Prueba: {X_test.shape[0]} muestras")


# 5. Funciones del modelo (sin cambios)
def hipotesis(X, theta, b):
    return b + (X @ theta)

def MSE(X, y, theta, b):
    y_pred = hipotesis(X, theta, b)
    error = y_pred - y
    return np.mean(error ** 2)

def calcular_gradientes(X_batch, y_batch, theta, b):
    predicciones = hipotesis(X_batch, theta, b)
    error = predicciones - y_batch
    grad_b = np.mean(error)
    grad_theta = np.zeros_like(theta)
    for j in range(len(theta)):
        grad_theta[j] = np.mean(error * X_batch[:, j])
    return grad_theta, grad_b

# 6. Entrenamiento con validación 
def GD(X_train, y_train, X_val, y_val, theta_inicial, b_inicial, lr, epochs, paciencia=50):
    theta = theta_inicial.copy()
    b = b_inicial
    historial_error_train = []
    historial_error_val = []
    mejor_error_val = float('inf')
    mejor_theta = theta.copy()
    mejor_b = b
    epochs_sin_mejora = 0
    
    for epoch in range(epochs):
        # Calcular gradientes y actualizar parámetros
        grad_theta, grad_b = calcular_gradientes(X_train, y_train, theta, b)
        theta -= lr * grad_theta
        b -= lr * grad_b
        
        # Calcular errores
        error_train = MSE(X_train, y_train, theta, b)
        error_val = MSE(X_val, y_val, theta, b)
        
        historial_error_train.append(error_train)
        historial_error_val.append(error_val)
        
        # Early stopping
        if error_val < mejor_error_val:
            mejor_error_val = error_val
            mejor_theta = theta.copy()
            mejor_b = b
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1
            
        if epochs_sin_mejora >= paciencia:
            print(f"Early stopping en epoch {epoch}")
            break
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train MSE = {error_train:.6f}, Val MSE = {error_val:.6f}")
    
    return mejor_theta, mejor_b, historial_error_train, historial_error_val

# 7. Entrenamiento con validación
n_caracteristicas = X_train.shape[1]
theta_inicial = np.zeros(n_caracteristicas)
b_inicial = 0.0
learning_rate = 0.01
epochs = 800

print("\n=== ENTRENAMIENTO CON VALIDACIÓN ===")
theta_entrenado, b_entrenado, error_train_hist, error_val_hist = GD(
    X_train, y_train, X_val, y_val, theta_inicial, b_inicial, learning_rate, epochs
)

# 8. Evaluación en los tres conjuntos

# TRAIN
y_pred_train = hipotesis(X_train, theta_entrenado, b_entrenado)
mse_train = MSE(X_train, y_train, theta_entrenado, b_entrenado)
r2_train = r2_score(y_train, y_pred_train)

# VALIDATION
y_pred_val = hipotesis(X_val, theta_entrenado, b_entrenado)
mse_val = MSE(X_val, y_val, theta_entrenado, b_entrenado)
r2_val = r2_score(y_val, y_pred_val)

# TEST
mse_test = MSE(X_test, y_test, theta_entrenado, b_entrenado)
y_pred_test = hipotesis(X_test, theta_entrenado, b_entrenado)
r2_test = r2_score(y_test, y_pred_test)


# 9. Resultados en terminal
print("\n//////// MODELO SIN FRAMEWORK /////////")
print("\n" + "="*50)
print("RESULTADOS FINALES")
print("="*50)
print(f"MSE Train: {mse_train:.6f}")
print(f"MSE Validation: {mse_val:.6f}")
print(f"MSE Test: {mse_test:.6f}")
print("-"*50)
print(f"R² Train: {r2_train:.4f}")
print(f"R² Validation: {r2_val:.4f}")
print(f"R² Test: {r2_test:.4f}")
print("-"*50)
print(f"Número de épocas ejecutadas: {len(error_train_hist)}")
print(f"Mejor error de validación: {min(error_val_hist):.6f}")


# 10. Gráficas de evaluación
plt.figure(figsize=(15, 10))

# Gráfica 1: Evolución del error durante entrenamiento
plt.subplot(2, 2, 1)
plt.plot(error_train_hist, label='Train MSE', alpha=0.8)
plt.plot(error_val_hist, label='Validation MSE', alpha=0.8)
plt.xlabel('Épocas')
plt.ylabel('Error Cuadrático Medio')
plt.title('Evolución del Error durante el Entrenamiento')
plt.legend()
plt.grid(True)

# Gráfica 2: Predicciones vs Valores reales (Train)
plt.subplot(2, 2, 2)
plt.scatter(y_train, y_pred_train, alpha=0.5, label='Train', color='blue')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Valor Real')
plt.ylabel('Predicción')
plt.title(f'Train: R² = {r2_train:.4f}')
plt.legend()

# Gráfica 3: Predicciones vs Valores reales (Validation)
plt.subplot(2, 2, 3)
plt.scatter(y_val, y_pred_val, alpha=0.5, label='Validation', color='green')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('Valor Real')
plt.ylabel('Predicción')
plt.title(f'Validation: R² = {r2_val:.4f}')
plt.legend()

# Gráfica 4: Predicciones vs Valores reales (Test)
plt.subplot(2, 2, 4)
plt.scatter(y_test, y_pred_test, alpha=0.5, label='Test', color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valor Real')
plt.ylabel('Predicción')
plt.title(f'Test: R² = {r2_test:.4f}')
plt.legend()

plt.tight_layout()
save_plot("evaluacion_completa_lineal_regression")
plt.show()

# 11. Distribución de errores
plt.figure(figsize=(12, 4))

# Distribución de errores en los tres conjuntos
residuals_train = y_train - y_pred_train
residuals_val = y_val - y_pred_val
residuals_test = y_test - y_pred_test

plt.subplot(1, 3, 1)
sns.histplot(residuals_train, kde=True, bins=30, color='blue')
plt.title('Distribución de Error - Train')
plt.xlabel('Error')

plt.subplot(1, 3, 2)
sns.histplot(residuals_val, kde=True, bins=30, color='green')
plt.title('Distribución de Error - Validation')
plt.xlabel('Error')

plt.subplot(1, 3, 3)
sns.histplot(residuals_test, kde=True, bins=30, color='orange')
plt.title('Distribución de Error - Test')
plt.xlabel('Error')

plt.tight_layout()
save_plot("distribucion_errores_lineal_regression.png")
plt.show()

# 12. Ejemplos de predicciones
print("\n=== EJEMPLOS DE PREDICCIONES ===")
print("Test set (primeras 5 muestras):")
for i in range(5):
    print(f"Real: {y_test[i]:.2f} - Predicho: {y_pred_test[i]:.2f} - Error: {abs(y_test[i] - y_pred_test[i]):.2f}")

# 13. Resumen de métricas por conjunto
print("\n=== RESUMEN DE MÉTRICAS ===")
metricas = pd.DataFrame({
    'Conjunto': ['Train', 'Validation', 'Test'],
    'MSE': [mse_train, mse_val, mse_test],
    'R²': [r2_train, r2_val, r2_test],
    'Muestras': [len(y_train), len(y_val), len(y_test)]
})
print(metricas)

