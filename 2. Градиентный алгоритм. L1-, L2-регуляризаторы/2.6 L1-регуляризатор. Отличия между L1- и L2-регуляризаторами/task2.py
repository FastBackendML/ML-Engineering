import numpy as np
import matplotlib.pyplot as plt
# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return -0.5 * x ** 2 + 0.1 * x ** 3 + np.cos(3 * x) + 7


# модель
def model(w, x):
    xv = np.array([x ** n for n in range(len(w))])
    return w.T @ xv


# функция потерь
def loss(w, x, y):
    return (model(w, x) - y) ** 2


# производная функции потерь
def dL(w, x, y):
    xv = np.array([x ** n for n in range(len(w))])
    return 2 * (model(w, x) - y) * xv


coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)

N = 5 # сложность модели (полином степени N-1)
lm_l1 = 2.0 # коэффициент лямбда для L1-регуляризатора
sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001, 0.000002]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.zeros(N) # начальные нулевые значения параметров модели
n_iter = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20 # размер мини-батча (величина K = 20)

X = np.array([model(w, x) for x in coord_x])
Qe = sum(loss(w, X[i], coord_y[i]) for i in range(sz)) / sz  # начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел
Qe_valuesL1 = []
# здесь продолжайте программу
for i in range(n_iter):
    k = np.random.randint(0, sz - batch_size - 1)
    Xk = X[range(k, k+batch_size)]
    Yk = coord_y[range(k, k+batch_size)]
    Qk = loss(w, Xk, Yk) / batch_size
    dQk = dL(w, Xk, Yk) / batch_size
    W = w.copy()
    W[0] = 0
    w -= eta * (dQk + lm_l1 * np.sign(W))
    Qe = lm * Qk + (1 - lm) * Qe
    Qe_valuesL1.append(Qe)

Q = sum(loss(w, X[i], coord_y[i]) for i in range(sz)) / sz
print(w)
print(Q)
plt.plot(Qe_valuesL1)
plt.title('Qe L1')
plt.grid()
plt.show()