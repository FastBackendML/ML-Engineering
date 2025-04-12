import numpy as np
import matplotlib.pyplot as plt

# координаты четырех точек
x = np.array([0, 1, 2, 3])
y = np.array([0.5, 0.8, 0.6, 0.2])

x_est = np.arange(0, 3.1, 0.1)

# Треугольное (линейное) ядро
def K(r):
    return np.where(np.abs(r) <= 1, 1 - np.abs(r), 0)

h = 1
y_est = []

for i in x_est:
    distances = np.abs(x - i)
    weights = K(distances / h)
    if np.sum(weights) > 0:
        y_est.append(np.sum(weights * y) / np.sum(weights))
    else:
        y_est.append(0)  # на случай, если нет подходящих точек

# Визуализация
plt.plot(x, y, 'o', label='Исходные точки')
plt.plot(x_est, y_est, '-', label='Оценка функции (Nadaraya-Watson)')
plt.legend()
plt.grid()
plt.show()


