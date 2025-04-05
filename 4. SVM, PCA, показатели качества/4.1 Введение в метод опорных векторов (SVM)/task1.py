import numpy as np
import matplotlib.pyplot as plt
import time

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.02 * np.exp(-x) - 0.2 * np.sin(3 * x) + 0.5 * np.cos(2 * x) - 7


# здесь объявляйте необходимые функции


coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.01, 1e-3, 1e-4, 1e-5, 1e-6]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.array([0., 0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
X = np.vstack((np.ones(sz), coord_x, coord_x**2, coord_x**3, coord_x**4)).T
Qe = sum((X @ w.T) - coord_y) / sz# начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

Qe_value = []
# здесь продолжайте программу
for i in range(N):
    k = np.random.randint(0, sz - 1)
    Lk = ((w.T @ X[k]) - coord_y[k])**2
    dLk = 2 * ((w.T @ X[k]) - coord_y[k]) * X[k].T
    w -= eta * dLk
    Qe = lm * Lk + (1 - lm) * Qe
    Qe_value.append(Qe)


Q = sum(((X @ w.T) - coord_y)**2) / sz
print(Q)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
line1 = ax1.plot((X @ w.T), color='red', label='a(x)')
line2 = ax1.plot(coord_y, color='blue', label='Y')
line3 = ax2.plot(Qe_value, label='Qe', color='red')
ax1.legend()
ax2.legend()
fig.tight_layout()
plt.show()
