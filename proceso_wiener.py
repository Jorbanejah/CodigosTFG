import numpy as np
import matplotlib.pyplot as plt

# Parámetros
N = 10000  # Número de simulaciones
T = 100    # Número de pasos de tiempo
h = 1 / T  # Tamaño del paso de tiempo

# Simulación de Wiener N veces
final_values = np.zeros(N)

for i in range(N):
    u = np.random.randn(T)  # Generamos T variables normales estándar
    W = np.cumsum(np.sqrt(h) * u)  # Proceso de Wiener acumulado
    final_values[i] = W[-1]  # Guardamos el valor final W(T)

# Elegir índices aleatorios para graficar
indices = np.random.choice(N, size=5, replace=False)

# Crear una figura con dos subgráficos
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 fila, 2 columnas

# Histograma del valor final de los procesos de Wiener
axes[0].hist(final_values, bins=50, density=True, alpha=0.6, color='b', edgecolor='black')
axes[0].set_xlabel(r'Valor final $W(T)$')
axes[0].set_ylabel('Densidad')
axes[0].set_title(f'Histograma de {N} simulaciones del proceso de Wiener')
axes[0].grid(True)

# Graficar algunos procesos de Wiener en el tiempo
for idx in indices:
    W = np.cumsum(np.sqrt(h) * np.random.randn(T))
    axes[1].plot(np.linspace(0, T, T), W, label=f'Proceso de Wiener {idx}')
axes[1].set_xlabel('Tiempo')
axes[1].set_ylabel(r'$W(t)$')
axes[1].set_title('Ejemplos de procesos de Wiener')
axes[1].legend()
axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')  # Dibuja el eje x
axes[1].set_xlim(0, T)


# Ajustar distribución de gráficos
plt.tight_layout()
plt.show()


