import numpy as np
import matplotlib.pyplot as plt
import time

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5


# здесь объявляйте необходимые функции


coord_x = np.arange(-4.0, 6.0, 0.1)  # значения по оси абсцисс [-4; 6] с шагом 0.1
coord_y = func(coord_x)  # значения функции по оси ординат

sz = len(coord_x)  # количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001])  # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.])  # начальные значения параметров модели
N = 500  # число итераций алгоритма SGD
lm = 0.02  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 50  # размер мини-батча (величина K = 50)

X = np.array([[1, x, x ** 2, x ** 3] for x in coord_x])
Qe = sum(((X @ w.T) - coord_y) ** 2) / sz  # начальное значение среднего эмпирического риска
np.random.seed(0)  # генерация одинаковых последовательностей псевдослучайных чисел
K = 50
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 10))
line1 = ax1.plot(coord_x, coord_y, label='func')
line2 = ax1.plot(coord_x, X @ w.T, label='func_a')[0]
ax1.legend()
ax1.grid(True)
ax1.set_title("Сравнение функции и модели")

# График Qe
line3 = ax2.plot([],[], label='Qe', color='r')[0]
line4 = ax2.plot([],[], label='Q', color='b')[0]
ax2.set_xlim(0, N + 20)
ax2.set_ylim(0, 3)
ax2.set_title("Изменение Qe по итерациям")
ax2.grid(True)
ax2.legend()

Qe_values = []
Q_values = []
for n in range(N):
    k = np.random.randint(0, sz - batch_size)

    Xk = X[range(k, k + batch_size)]
    Yk = coord_y[range(k, k + batch_size)]

    aXk = [x @ w.T for x in Xk]
    a = [x @ w.T for x in X]
    line2.set_ydata(a)
    line3.set_data(range(len(Qe_values)), Qe_values)
    line4.set_data(range(len(Q_values)), Q_values)

    dQk = 0
    Qk = 0
    for z, i in enumerate(Yk):
        dQk += (aXk[z] - i) * Xk[z].T
        Qk += (aXk[z] - i) ** 2

    dQk = 2 / K * dQk
    Qk = 1 / K * Qk

    w = w - eta * dQk
    Q = 1 / sz * sum(((X @ w.T) - coord_y) ** 2)
    Qe = lm * Qk + (1 - lm) * Qe
    Q_values.append(Q)
    Qe_values.append(Qe)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)

Q = 1 / sz * sum(((X @ w.T) - coord_y) ** 2)
print(w)
print(Qe)
print(Q)
plt.ioff()
plt.show()
