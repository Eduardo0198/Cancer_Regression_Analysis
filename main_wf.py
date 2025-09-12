
# jose eduardo viveros escamia - A01710605
# Random Forest Regression para predecir area_mean en dataset de cáncer

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
print(f"Dataset cargado: {df.shape[0]} muestras, {df.shape[1]} columnas")

# 2. Selección de features y target
features = [
    'texture_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'symmetry_mean', 'fractal_dimension_mean', 'texture_worst', 'smoothness_worst'
]
target = "area_mean"

X = df[features].values
y = df[target].values

print(f"\nVariables independientes: {X.shape}")
print(f"Variable dependiente: {y.shape}")

# 3. Escalado opcional (Random Forest no lo requiere, pero si quieres uniforme)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# 4. División de datos (Train 60%, Validation 20%, Test 20%)
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"\nDivisión de datos:")
print(f" - Entrenamiento: {X_train.shape[0]} muestras")
print(f" - Validación: {X_val.shape[0]} muestras")
print(f" - Test: {X_test.shape[0]} muestras")

# 5. Definición de hiperparámetros
hyperparams = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

# 6. Entrenamiento del modelo
rf_model = RandomForestRegressor(**hyperparams)
rf_model.fit(X_train, y_train)

# 7. Predicciones
y_pred_train = rf_model.predict(X_train)
y_pred_val = rf_model.predict(X_val)
y_pred_test = rf_model.predict(X_test)

# 8. Métricas
def print_metrics(y_true, y_pred, set_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{set_name:12} - MSE: {mse:.6f}, R²: {r2:.4f}")

print("\n=== Métricas de Evaluación ===")
print_metrics(y_train, y_pred_train, "Entrenamiento")
print_metrics(y_val, y_pred_val, "Validación")
print_metrics(y_test, y_pred_test, "Test")

# 9. Importancia de features
print("\n=== Importancia de características ===")
importancia = rf_model.feature_importances_
for i, col in enumerate(features):
    print(f"  {col}: {importancia[i]:.4f}")

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
plt.show()

# 11. Ejemplo de predicción
ejemplo_X = np.array([[15.0, 0.1, 0.05, 0.02, 0.1, 0.06, 18.0, 0.12]])  # Ajusta según tus features
prediccion = rf_model.predict(ejemplo_X)
print("\n=== Predicción de ejemplo ===")
for i, col in enumerate(features):
    print(f"  {col}: {ejemplo_X[0,i]}")
print(f"Predicción area_mean: {prediccion[0]:.2f}")
