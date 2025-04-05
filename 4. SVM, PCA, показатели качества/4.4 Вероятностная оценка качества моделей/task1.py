import numpy as np
from sklearn.model_selection import train_test_split


def loss(w, x, y):
    M = (x @ w) * y
    return np.exp(-M)


def df(w, x, y):
    return -1 * loss(w, x, y) * x.T * y


np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.4
D1 = 2.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 3.0
mean2 = [2, 3]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

data_x = np.array([[1, x[0], x[1]] for x in np.hstack([x1, x2]).T])
data_y = np.hstack([np.ones(N) * -1, np.ones(N)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.3, shuffle=True)

n_train = len(x_train)  # размер обучающей выборки
w = np.array([0.0, 0.0, 0.0])  # начальные весовые коэффициенты
nt = np.array([0.5, 0.01, 0.01])  # шаг обучения для каждого параметра w0, w1, w2
lm = 0.01  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
N = 500  # число итераций алгоритма SGD
batch_size = 10 # размер мини-батча (величина K = 10)
Qe = sum(loss(w, x_train[i], y_train[i]) for i in range(len(x_train))) / n_train

# здесь продолжайте программу
for i in range(N):
    k = np.random.randint(0, n_train - batch_size - 1)  # n_train - размер выборки (массива x_train)
    Xk = x_train[range(k, k+batch_size)]
    Yk = y_train[range(k, k+batch_size)]
    Qk = 0
    dQk = 0
    for i in range(len(Xk)):
        Qk += loss(w, Xk[i], Yk[i])
        dQk += df(w, Xk[i], Yk[i])
    Qk = Qk / batch_size
    dQk = dQk / batch_size
    w -= nt * dQk
    Qe = lm * Qk + (1 - lm) * Qe

mrgs = np.array([(x_test[i] @ w) * y_test[i] for i in range(len(x_test))])
mrgs.sort()
ax = [(x_test[i] @ w) * y_test[i] < 0 for i in range(len(x_test))]
preds = np.sign(x_test @ w)
acc = np.mean(preds == y_test)


import matplotlib.pyplot as plt

# Отрисовка точек двух классов
plt.figure(figsize=(8, 6))
plt.scatter(x_test[y_test == -1][:, 1], x_test[y_test == -1][:, 2], color='blue', label='Класс -1', alpha=0.5)
plt.scatter(x_test[y_test == 1][:, 1], x_test[y_test == 1][:, 2], color='red', label='Класс 1', alpha=0.5)

# Построение границы решений: w0 + w1*x1 + w2*x2 = 0 → x2 = -(w0 + w1*x1)/w2
x_vals = np.linspace(x_test[:, 1].min(), x_test[:, 1].max(), 100)
if w[2] != 0:
    y_vals = -(w[0] + w[1] * x_vals) / w[2]
    plt.plot(x_vals, y_vals, color='black', linewidth=2, label='Граница решений')

plt.xlabel('Признак 1 (x₁)')
plt.ylabel('Признак 2 (x₂)')
plt.title(f'Разделяющая граница (точность: {acc:.2f})')
plt.legend()
plt.grid(True)
plt.show()





