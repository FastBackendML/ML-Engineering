import numpy as np
import matplotlib.pyplot as plt
import time
# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.5 * x**2 - 0.1 * 1/np.exp(-x) + 0.5 * np.cos(2*x) - 2.


# здесь объявляйте необходимые функции


coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.01, 0.001, 0.0001, 0.01, 0.01]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.array([0., 0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего

X = np.array([[1, x, x**2, np.cos(2*x), np.sin(2*x)] for x in coord_x])

Qe = sum(((X @ w.T) - coord_y)**2) / sz  # начальное значение среднего эмпирического риска
np.random.seed(0)  # генерация одинаковых последовательностей псевдослучайных чисел
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 10))
line1, = ax1.plot(coord_x, coord_y, label='func')

def model(x, w):
    return w[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3] + w[4] * x[4]

coord_a = [model(x,w) for x in X]
line2 = ax1.plot(coord_x, X @ w.T, label='model_a')[0]
ax1.legend()
ax1.grid(True)
ax1.set_title("Сравнение функции и модели")

# График Qe
line3, = ax2.plot([], [], label='Qe', color='r')
ax2.set_xlim(0, N + 20)
ax2.set_ylim(0, 12)
ax2.set_title("Изменение Qe по итерациям")
ax2.grid(True)
ax2.legend()

Qe_values = []
print(line2)
for n in range(N):
    k = np.random.randint(0, sz)
    dL = 2 * ((X[k] @ w.T) - coord_y[k]) * X[k].T
    w = w - eta * dL
    Qe = lm * ((X[k] @ w.T) - coord_y[k])**2 + (1 - lm) * Qe
    Qe_values.append(Qe)
    coord_a = X @ w.T
    line2.set_ydata(coord_a)

    line3.set_data(range(len(Qe_values)), Qe_values)

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.2)


Q = 1/sz * sum(((X @ w.T) - coord_y)**2)
print(Q)
print(Qe)
plt.ioff()
plt.show()
