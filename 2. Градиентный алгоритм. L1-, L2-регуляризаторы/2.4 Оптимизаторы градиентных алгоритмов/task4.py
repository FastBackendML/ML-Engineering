import numpy as np

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return -0.7 * x - 0.2 * x ** 2 + 0.05 * x ** 3 - 0.2 * np.cos(3 * x) + 2


# здесь объявляйте необходимые функции


coord_x = np.arange(-4.0, 6.0, 0.1) # значения по оси абсцисс [-4; 6] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001]) # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20 # размер мини-батча (величина K = 20)
gamma = 0.8 # коэффициент гамма для вычисления импульсов Нестерова
v = np.zeros(len(w))  # начальное значение [0, 0, 0, 0]
X = np.array([[1, x, x**2, x**3] for x in coord_x])
Qe = sum(((X @ w.T) - coord_y) ** 2) / sz # начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел
K = batch_size
# здесь продолжайте программу
for n in range(N):
    k = np.random.randint(0, sz - batch_size - 1)  # sz - размер выборки (массива coord_x)
    Xi = X[range(k, k+batch_size)]
    Yi = coord_y[range(k, k+batch_size)]
    dQk = 0
    Qk = 0
    for i in range(len(Xi)):
        dQk += ((w - gamma * v).T @ Xi[i] - Yi[i]) * Xi[i].T
        Qk += ((w.T @ Xi[i]) - Yi[i])**2

    dQk = (2 * dQk) / K
    Qk = Qk / K
    v = gamma * v + (1 - gamma) * eta * dQk
    w -= v
    Qe = lm * Qk + (1 - lm) * Qe

Q = sum(((X @ w.T) - coord_y)**2) / sz
print(Q)
print(Qe)
print(w)

