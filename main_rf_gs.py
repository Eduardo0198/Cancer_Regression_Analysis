
# ////////////////////////////////////////////////////
# jose eduardo viveros escamia - A01710605
# Random Forest Mejorado con GridSearchCV y gráfica de ajuste
# ////////////////////////////////////////////////////

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Reproducibilidad
np.random.seed(42)

# 1. Cargar datos
df = pd.read_csv(r"C:\Users\josed\Documents\Benj\Cancer_Data.csv")

# 2. Carpeta para guardar imágenes
carpeta = "images_rf_cv"
os.makedirs(carpeta, exist_ok=True)
def save_plot(nombre):
    ruta = os.path.join(carpeta, f"{nombre}.png")
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    print(f"Gráfica guardada en: {ruta}")

# 3. Features y target
features = [
    'texture_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'symmetry_mean', 'fractal_dimension_mean', 'texture_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'symmetry_worst', 'fractal_dimension_se',
    'radius_mean'
]
target = "area_mean"

X = df[features].values
y = df[target].values

# 4. Escalado opcional
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# 5. División en Train/Val/Test
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# 6. Definir Random Forest y GridSearchCV
rf = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [300, 500, 1000],
    'max_depth': [None, 7, 12, 17],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [5, 6]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# 7. Entrenar con GridSearch
grid_search.fit(X_train, y_train)

print("\n=== Mejor combinación de hiperparámetros ===")
print(grid_search.best_params_)

# 8. Modelo final con mejores hiperparámetros
best_rf = grid_search.best_estimator_

# Predicciones
y_pred_train = best_rf.predict(X_train)
y_pred_val = best_rf.predict(X_val)
y_pred_test = best_rf.predict(X_test)

# Métricas
def print_metrics(y_true, y_pred, set_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{set_name} - MSE: {mse:.6f}, R²: {r2:.4f}")

print("\n=== RESULTADOS RANDOM FOREST MEJORADO ===")
print_metrics(y_train, y_pred_train, "Train")
print_metrics(y_val, y_pred_val, "Validation")
print_metrics(y_test, y_pred_test, "Test")

# 9. Importancia de features
importancia = best_rf.feature_importances_
plt.figure(figsize=(10,5))
sns.barplot(x=importancia, y=features)
plt.title("Importancia de Features - Random Forest Mejorado")
plt.xlabel("Importancia")
plt.ylabel("Features")
plt.tight_layout()
save_plot("feature_importances_rf_mejorado")
plt.show()

# 10. Gráfica predicciones vs valores reales (ajuste modelo)
plt.figure(figsize=(15,5))
for i, (y_true, y_pred, color, title) in enumerate(zip(
        [y_train, y_val, y_test],
        [y_pred_train, y_pred_val, y_pred_test],
        ['blue','green','orange'],
        ['Train','Validation','Test']
    )):
    plt.subplot(1,3,i+1)
    plt.scatter(y_true, y_pred, alpha=0.5, color=color)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.title(f"{title} - R² = {r2_score(y_true, y_pred):.3f}")
    plt.xlabel('Valor Real')
    plt.ylabel('Predicción')
    plt.grid(True)

plt.tight_layout()
save_plot("predicciones_vs_reales_rf_mejorado")
plt.show()
