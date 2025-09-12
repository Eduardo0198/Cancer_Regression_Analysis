
# jose eduardo viveros escamia - A01710605
# Regresión Lineal Múltiple desde Cero

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. Cargar datos
df = pd.read_csv(r"C:\Users\josed\Documents\Benj\Cancer_Data.csv")

# 2. Matriz de correlación
print("\n=== MATRIZ DE CORRELACIÓN ===")


# 3. Selección de variables
# Basicamente seleccioné las variables con mayor correlación con 'area_mean' osea nuestras x
features = ["radius_mean", "perimeter_mean", "smoothness_mean", 
            "compactness_mean", "concavity_mean", "texture_mean"]
# area_mean es nuestra y
target = "area_mean"


# Selecciona solo las features numéricas + target
cols_numericas = features + [target]
correlation_matrix = df[cols_numericas].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".3f", center=0)
plt.title('Matriz de Correlación')
plt.tight_layout()
plt.show()

X = df[features].values
y = df[target].values.reshape(-1, 1)

print(f"\nVariables independientes: {X.shape}")
print(f"Variable dependiente: {y.shape}")

# 4. Escalado de características
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_original = y.flatten()  # Shape (n,)

print("\n=== Después del escalamiento ===")
print("X - Medias:", scaler_X.mean_)
print("X - Escalas:", scaler_X.scale_)
print("y - (primeros valores):", y_original[:5])

# 5. División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_original, test_size=0.2, random_state=42
)
print(f"\nDivisión de datos:")
print(f" - Entrenamiento: {X_train.shape[0]} muestras")
print(f" - Prueba: {X_test.shape[0]} muestras")

# 6. Funciones del modelo
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


def GD(X, y, theta_inicial, b_inicial, lr, epochs):
    theta = theta_inicial.copy()
    b = b_inicial
    historial_error = []
    mejores_parametros = None
    mejor_error = float('inf')
    for epoch in range(epochs):
        grad_theta, grad_b = calcular_gradientes(X, y, theta, b)
        theta -= lr * grad_theta
        b -= lr * grad_b
        error_epoch = MSE(X, y, theta, b)
        historial_error.append(error_epoch)
        if error_epoch < mejor_error:
            mejor_error = error_epoch
            mejores_parametros = (theta.copy(), b)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Error = {error_epoch:.6f}")
    return mejores_parametros[0], mejores_parametros[1], historial_error

# 7. Entrenamiento
n_caracteristicas = X_train.shape[1]
theta_inicial = np.zeros(n_caracteristicas)
b_inicial = 0.0
learning_rate = 0.01
epochs = 200

theta_entrenado, b_entrenado, historial_error = GD(
    X_train, y_train, theta_inicial, b_inicial, learning_rate, epochs
)

# 8. Evaluación
y_pred_train = hipotesis(X_train, theta_entrenado, b_entrenado)
y_pred_test = hipotesis(X_test, theta_entrenado, b_entrenado)

mse_train = MSE(X_train, y_train, theta_entrenado, b_entrenado)
mse_test = MSE(X_test, y_test, theta_entrenado, b_entrenado)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("\n========== RESULTADOS ==========")
print("Variables usadas:", features)
print("Pesos aprendidos:", theta_entrenado)
print("Bias:", b_entrenado)
print(f"MSE Entrenamiento: {mse_train:.6f}, MSE Test: {mse_test:.6f}")
print(f"R² Entrenamiento: {r2_train:.4f}, R² Test: {r2_test:.4f}")


# 9. Visualización de error
plt.figure(figsize=(8,5))
plt.plot(historial_error, label="MSE")
plt.xlabel("Iteraciones")
plt.ylabel("Error cuadrático medio")
plt.title("Evolución del error en entrenamiento")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafica_mse.png")  # Guardar imagen para README
plt.show()

# 10. Predicciones de ejemplo
print("\nEjemplo de predicciones:")
for i in range(5):
    print(f"Real: {y_test[i]:.2f} - Predicho: {y_pred_test[i]:.2f}")

# Gráfica de predicciones vs valores reales
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_test, alpha=0.5, color = 'green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw = 2)
plt.xlabel("Valor Real")
plt.ylabel("Predicción")
plt.title("Predicciones vs Valores Reales")
plt.savefig("predicciones_vs_reales.png")
plt.show()

# Distribución del error
residuals = y_test - y_pred_test
plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel("Error (Residual)")
plt.ylabel("Frecuencia")
plt.title("Distribución del Error")
plt.savefig("distribucion_error.png")
plt.show()

# 11. Matriz de correlación final
plt.figure(figsize=(10,8))
sns.heatmap(df[features + [target]].corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlación entre variables")
plt.tight_layout()
plt.savefig("grafica_correlacion.png")  # Guardar imagen para README
plt.show()
