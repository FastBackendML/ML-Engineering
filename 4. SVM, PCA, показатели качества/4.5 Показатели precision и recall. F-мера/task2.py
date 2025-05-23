import numpy as np

# логарифмическая функция потерь: log2​(1+e−{w^T⋅xi​⋅yi}​)
loss = lambda w, x, y: np.log2(1 + np.exp(x @ w * y))

# её производная вектору w: −e−{w^T⋅xi​⋅yi​⋅x^iT​⋅yi}​​ / ((1+e^{-wT⋅xi​⋅yi}​)⋅ln(2))
df = lambda w, x, y: -np.exp(-x @ w * y) * x * y / (1 + np.exp(-x @ w * y)) / np.log(2)

data_x = [(5.8, 1.2), (5.6, 1.5), (6.5, 1.5), (6.1, 1.3), (6.4, 1.3), (7.7, 2.0), (6.0, 1.8), (5.6, 1.3), (6.0, 1.6), (5.8, 1.9), (5.7, 2.0), (6.3, 1.5), (6.2, 1.8), (7.7, 2.3), (5.8, 1.2), (6.3, 1.8), (6.0, 1.0), (6.2, 1.3), (5.7, 1.3), (6.3, 1.9), (6.7, 2.5), (5.5, 1.2), (4.9, 1.0), (6.1, 1.4), (6.0, 1.6), (7.2, 2.5), (7.3, 1.8), (6.6, 1.4), (5.6, 2.0), (5.5, 1.0), (6.4, 2.2), (5.6, 1.3), (6.6, 1.3), (6.9, 2.1), (6.8, 2.1), (5.7, 1.3), (7.0, 1.4), (6.1, 1.4), (6.1, 1.8), (6.7, 1.7), (6.0, 1.5), (6.5, 1.8), (6.4, 1.5), (6.9, 1.5), (5.6, 1.3), (6.7, 1.4), (5.8, 1.9), (6.3, 1.3), (6.7, 2.1), (6.2, 2.3), (6.3, 2.4), (6.7, 1.8), (6.4, 2.3), (6.2, 1.5), (6.1, 1.4), (7.1, 2.1), (5.7, 1.0), (6.8, 1.4), (6.8, 2.3), (5.1, 1.1), (4.9, 1.7), (5.9, 1.8), (7.4, 1.9), (6.5, 2.0), (6.7, 1.5), (6.5, 2.0), (5.8, 1.0), (6.4, 2.1), (7.6, 2.1), (5.8, 2.4), (7.7, 2.2), (6.3, 1.5), (5.0, 1.0), (6.3, 1.6), (7.7, 2.3), (6.4, 1.9), (6.5, 2.2), (5.7, 1.2), (6.9, 2.3), (5.7, 1.3), (6.1, 1.2), (5.4, 1.5), (5.2, 1.4), (6.7, 2.3), (7.9, 2.0), (5.6, 1.1), (7.2, 1.8), (5.5, 1.3), (7.2, 1.6), (6.3, 2.5), (6.3, 1.8), (6.7, 2.4), (5.0, 1.0), (6.4, 1.8), (6.9, 2.3), (5.5, 1.3), (5.5, 1.1), (5.9, 1.5), (6.0, 1.5), (5.9, 1.8)]
data_y = [-1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1]

x_train = np.array([[1, x[0], x[1]] for x in data_x])
y_train = np.array(data_y)

n_train = len(x_train)  # размер обучающей выборки
w = [0.0, 0.0, 0.0]  # начальные весовые коэффициенты
nt = np.array([0.5, 0.01, 0.01])  # шаг обучения для каждого параметра w0, w1, w2
lm = 0.01  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
N = 500  # число итераций алгоритма SGD

np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

for _ in range(N):
    k = np.random.randint(0, n_train-1) # n_train - размер выборки (массива x_train)
    grad = df(w, x_train[k], y_train[k])
    w -= nt * grad

# Предсказание по модели
predict = np.sign(x_train @ w)

TP = sum([p == y and p == 1 for p, y in zip(predict, y_train)])
TN = sum([p == y and p == -1 for p, y in zip(predict, y_train)])
FP = sum([p != y and p == 1 for p, y in zip(predict, y_train)])
FN = sum([p != y and p == -1 for p, y in zip(predict, y_train)])

precision = TP / (TP + FP)
recall = TP / (TP + FN)

import matplotlib.pyplot as plt

# Преобразуем данные
X = np.array(data_x)
Y = np.array(data_y)

# Разделим на классы для отрисовки
class1 = X[Y == 1]
class2 = X[Y == -1]

# Граница решений: создадим сетку точек
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

# Для каждой точки сетки считаем линейную комбинацию с w
grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
Z = grid @ w
Z = Z.reshape(xx.shape)

# Построим график
plt.figure(figsize=(10, 6))

# Классы
plt.scatter(class1[:, 0], class1[:, 1], color='blue', label='Класс +1', s=60, edgecolors='k')
plt.scatter(class2[:, 0], class2[:, 1], color='red', label='Класс -1', s=60, edgecolors='k')

# Граница решений
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='green')

# Оформление
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.title(f"Граница решений логистической регрессии\nPrecision: {precision:.2f}, Recall: {recall:.2f}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




