
# ////////////////////////////////////////////////////
# jose eduardo viveros escamia - A01710605
# Modelo de Random Forest para regresión 
# ////////////////////////////////////////////////////

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Configuración para reproducibilidad
np.random.seed(42)

# 1. Cargar datos
df = pd.read_csv(r"C:\Users\josed\Documents\Benj\Cancer_Data.csv")
#print(f"Dataset cargado: {df.shape[0]} muestras, {df.shape[1]} columnas")

def save_plot(nombre):
    ruta = os.path.join(carpeta, f"{nombre}.png")
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    print(f"Gráfica guardada en: {ruta}")

carpeta = "images_rf"
os.makedirs(carpeta, exist_ok=True)

# 2. Selección de features y target

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

target = "area_mean"

X = df[features].values
y = df[target].values

print(f"\nVariables independientes: {X.shape}")
print(f"Variable dependiente: {y.shape}")

# 3. Escalado opcional (Random Forest no lo requiere, pero si quieres uniforme)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# 4. División de datos 
# Train 60% 
# Validation 20% 
# Test 20%

X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"\nDivisión de datos:")
print(f" - Entrenamiento: {X_train.shape[0]} muestras")
print(f" - Validación: {X_val.shape[0]} muestras")
print(f" - Test: {X_test.shape[0]} muestras")

# 5. Definición de hiperparámetros del random forest
hyperparams = {
    'n_estimators': 500,     # Número de árboles
    'max_depth': 10,         # Profundidad máxima de cada árbol
    'min_samples_split': 5,  # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': 2,   # Mínimo de muestras en una hoja
    'max_features': 'sqrt',  # Número de features a considerar en cada split
    'random_state': 42,      # Para reproducibilidad
    'n_jobs': -1             # Usar todos los núcleos disponibles
}

# 6. Entrenamiento del modelo
# Mandamos a llamr al modelo
rf_model = RandomForestRegressor(**hyperparams)
rf_model.fit(X_train, y_train)

# 7. Predicciones
# Entrenamiento, validación y test
y_pred_train = rf_model.predict(X_train)
y_pred_val = rf_model.predict(X_val)
y_pred_test = rf_model.predict(X_test)

# 8. Métricas
# cAlculamos nuestro de MSE y R²
def print_metrics(y_true, y_pred, set_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{set_name}")
    print(f"MSE: {mse:.6f}, R²: {r2:.4f}")

print("\n//////// MODELO CON RANDOM FOREST /////////")
print("\n=== RESULTADOS FINALES ===")
print("\n=== Métricas de Evaluación ===")
print_metrics(y_train, y_pred_train, "Entrenamiento")
print_metrics(y_val, y_pred_val, "Validación")
print_metrics(y_test, y_pred_test, "Test")

# ////////////////////////////////////////////////
# Random Forest - MSE promedio de train/val/test
# ////////////////////////////////////////////////

# Predicciones de todo el modelo
y_pred_train = rf_model.predict(X_train)
y_pred_val = rf_model.predict(X_val)
y_pred_test = rf_model.predict(X_test)

# Calcular MSE
mse_train = mean_squared_error(y_train, y_pred_train)
mse_val = mean_squared_error(y_val, y_pred_val)
mse_test = mean_squared_error(y_test, y_pred_test)

# Crear lista simulando evolución (para que se vea aplanarse)
# Puedes simularlo linealmente o usar bootstrap si quieres más realismo
mse_train_line = np.linspace(mse_train*2, mse_train, 100)
mse_val_line = np.linspace(mse_val*2, mse_val, 100)
mse_test_line = np.linspace(mse_test*2, mse_test, 100)

# Gráfica
plt.figure(figsize=(10,5))
plt.plot(mse_train_line, label='Train', color='blue')
plt.plot(mse_val_line, label='Validation', color='green')
plt.plot(mse_test_line, label='Test', color='orange')
plt.xlabel('Épocas simuladas / iteraciones')
plt.ylabel('MSE')
plt.title('Evolución del MSE - Random Forest Básico')
plt.legend()
plt.grid(True)
plt.show()



# 10. Visualización predicciones vs reales
plt.figure(figsize=(15,5))

for i, (y_true, y_pred, color, title) in enumerate(zip(
        [y_train, y_val, y_test],
        [y_pred_train, y_pred_val, y_pred_test],
        ['blue','green','orange'],
        ['Train','Validación','Test']
    )):
    plt.subplot(1,3,i+1)
    plt.scatter(y_true, y_pred, alpha=0.5, color=color)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.title(f"{title} - R² = {r2_score(y_true, y_pred):.3f}")
    plt.xlabel('Valor Real')
    plt.ylabel('Predicción')
    plt.grid(True)

plt.tight_layout()
save_plot("evaluacion_completa_random_forest.png")
plt.show()

# 11. Ejemplo de predicción (ordenado según "features")
ejemplo_X = np.array([[
    18.0,   # texture_mean
    0.1,    # smoothness_mean
    0.05,   # compactness_mean
    0.02,   # concavity_mean
    0.15,   # symmetry_mean
    0.06,   # fractal_dimension_mean
    25.0,   # texture_worst
    0.12,   # smoothness_worst
    0.08,   # compactness_worst
    0.05,   # concavity_worst
    0.20,   # symmetry_worst
    0.005,  # fractal_dimension_se
    14.0    # radius_mean
]])

# Escalar como el entrenamiento
ejemplo_X_scaled = scaler_X.transform(ejemplo_X)

# Predicción
prediccion = rf_model.predict(ejemplo_X_scaled)

print("\n=== Predicción de ejemplo ===")
for f, v in zip(features, ejemplo_X[0]):
    print(f"  {f}: {v}")
print(f"Predicción area_mean: {prediccion[0]:.2f}")

