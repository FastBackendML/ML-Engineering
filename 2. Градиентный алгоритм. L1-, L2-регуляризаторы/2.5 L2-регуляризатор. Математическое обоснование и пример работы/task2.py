import numpy as np


# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.5 * x + 0.2 * x**2 - 0.05 * x**3 + 0.2 * np.sin(4*x) - 2.5


# модель
def model(w, x):
    xv = np.array([x ** n for n in range(len(w))])
    return w @ xv


# функция потерь
def loss(w, x):
    return (model(w, x) - func(x)) ** 2


# производная функции потерь
def dL(w, x):
    xv = np.array([x ** n for n in range(len(w))])
    return 2 * (model(w, x) - func(x)) * xv


def Qk(w, x_batch):
    return np.mean([loss(w, x) for x in x_batch])


def dQk(w, x_batch):
    return sum([dL(w, x) for x in x_batch]) / len(x_batch)


coord_x = np.arange(-4.0, 6.0, 0.1)  # значения по оси абсцисс [-4; 6] с шагом 0.1
# coord_y = func(coord_x)  # значения функции по оси ординат

n_iter = 500  # число итераций алгоритма SGD
sz = len(coord_x)  # количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001, 0.000002])  # шаг обучения для каждого параметра w0, w1, w2, w3, w4
lm = 0.02  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20  # размер мини-батча (величина K = 20)


# здесь продолжайте программу
def calculate_w_Qe_Q(lm_l2):  # коэффициент лямбда для L2-регуляризатора
    N = 5  # сложность модели (полином степени N-1)
    w = np.zeros(N)  # начальные нулевые значения параметров модели

    Qe = Qk(w, coord_x)  # начальное значение среднего эмпирического риска

    np.random.seed(0)  # генерация одинаковых последовательностей псевдослучайных чисел

    # Раскомментируй строку ниже и увидишь, что L2 ничего не дает. Успех L2 есть лишь для конкретных случайных чисел.
    # Наверняка есть такие случайные числа, когда без L2 будет лучше

    # [np.random.randint(0, sz - batch_size - 1) for _ in range(n_iter)]

    for _ in range(n_iter):
        k = np.random.randint(0, sz - batch_size - 1)
        x_batch = coord_x[k: k + batch_size]
        Qe = (1 - lm) * Qe + lm * Qk(w, x_batch)
        w_tilda = w.copy()
        w_tilda[0] = 0
        w -= eta * (dQk(w, x_batch) + lm_l2 * w_tilda)
        # Qe = (1 - lm) * Qe + lm * Qk(w, x_batch)

    Q = Qk(w, coord_x)

    return w, Qe, Q


w_without_L2, Qe_without_L2, Q_without_L2 = calculate_w_Qe_Q(lm_l2=0)
w, Qe, Q = calculate_w_Qe_Q(lm_l2=2)


# вывод результатов
print(f"w_without_L2={w_without_L2}, Qe_without_L2={Qe_without_L2}, Q_without_L2={Q_without_L2}")
print(f"w_with_L2={w}, Qe_with_L2={Qe}, Q_with_L2={Q}")


from matplotlib import pyplot as plt


plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")

plt.scatter(coord_x, func(coord_x), color="black", label="f(x)")
plt.plot(coord_x, [model(w_without_L2, x) for x in coord_x], color="red", label="a(x) without L2")
plt.plot(coord_x, [model(w, x) for x in coord_x], color="green", label="a(x) with L2")

plt.legend()
plt.show()