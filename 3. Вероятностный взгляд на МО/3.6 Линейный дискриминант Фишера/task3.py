import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
mean1 = np.array([1, -2])
mean2 = np.array([-3, -1])
mean3 = np.array([1, 2])

r = 0.5
D = 1.0
V = [[D, D * r], [D*r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T
x3 = np.random.multivariate_normal(mean3, V, N).T

x_train = np.hstack([x1, x2, x3]).T
y_train = np.hstack([np.zeros(N), np.ones(N), np.ones(N) * 2])

# здесь вычисляйте векторы математических ожиданий и ковариационную матрицу по выборке x1, x2, x3
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)
mm3 = np.mean(x3.T, axis=0)

a = np.hstack([(x1.T - mm1).T, (x2.T - mm2).T, (x3.T - mm3).T])
VV = np.array([
    [np.dot(a[0], a[0]) / (N*2-1), np.dot(a[0], a[1]) / (N*2-1)],
    [np.dot(a[1], a[0]) / (N*2-1), np.dot(a[1], a[1]) / (N*2-1)]
])

# параметры для линейного дискриминанта Фишера
Py1, Py2, Py3 = 0.2, 0.4, 0.4
L1, L2, L3 = 1, 1, 1

# здесь продолжайте программу
alpha = lambda v, m: m @ np.linalg.inv(v)

beta = lambda l, m, py, v: np.log(l * py) - 0.5 * m @ np.linalg.inv(v) @ m

alpha1, alpha2, alpha3 = [alpha(VV, i) for i in [mm1, mm2, mm3]]

beta1, beta2, beta3 = [beta(i[0], i[1], i[2], VV) for i in [(L1, mm1, Py1), (L2, mm2, Py2), (L3, mm3, Py3)]]

ax = lambda x, a, b: a @ x.T + b
predict = []
for i in x_train:
    predict.append(np.argmax([ax(i, alpha1, beta1),
                              ax(i, alpha2, beta2),
                              ax(i, alpha3, beta3)
                              ]))

Q = sum(predict != y_train)
print(Q)
