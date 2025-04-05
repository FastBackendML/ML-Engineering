import numpy as np

# исходные параметры распределений двух классов
np.random.seed(0)
mean1 = np.array([1, -2, 0])
mean2 = np.array([1, 3, 1])
r = 0.7
D = 2.0
V = [[D, D * r, D*r*r], [D*r, D, D*r], [D*r*r, D*r, D]] # три предиктора

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T

x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.zeros(N), np.ones(N)])

# вычисление оценок МО и общей ковариационной матрицы
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = np.hstack([(x1.T - mm1).T, (x2.T - mm2).T])
print(a)
VV = np.array([
    [np.dot(a[0], a[0]) / (N*2-1), np.dot(a[0], a[1]) / (N*2-1), np.dot(a[0], a[2]) / (N*2-1)],
    [np.dot(a[1], a[0]) / (N*2-1), np.dot(a[1], a[1]) / (N*2-1), np.dot(a[1], a[2]) / (N*2-1)],
    [np.dot(a[2], a[0]) / (N*2-1), np.dot(a[2], a[1]) / (N*2-1), np.dot(a[2], a[2]) / (N*2-1)]
])

# параметры для линейного дискриминанта Фишера
Py1, L1 = 0.5, 1  # вероятности появления классов
Py2, L2 = 1 - Py1, 1  # и величины штрафов неверной классификации

# Функции:
alpha = lambda x, v, m: m @ np.linalg.inv(v)
beta = lambda x, v, m, l, py: np.log(l * py) - 0.5 * m @ np.linalg.inv(v) @ m
am = lambda x: np.argmax([alpha(x, VV, mm1) @ x.T + beta(x, VV, mm1, L1, Py1),
                          alpha(x, VV, mm2) @ x.T + beta(x, VV, mm2, L2, Py2)]) # классификатор

# Ответ на вопрос задания
alpha1, alpha2  = alpha(x_train, VV, mm1), alpha(x_train, VV, mm2)
beta1, beta2 = beta(x_train, VV, mm1, L1, Py1), beta(x_train, VV, mm2, L2, Py2)

# Предсказание по модели
predict = [am(x) for x in x_train]
print(alpha1)
# Качество
Q = sum(predict != y_train)