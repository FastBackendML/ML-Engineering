import numpy as np


def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5


def model(w, x):
    return w[0] + w[1] * x + w[2] * x ** 2 + w[3] * x ** 3


coord_x = np.arange(-4.0, 6.0, 0.1)

x_train = np.array([[_x**i for i in range(4)] for _x in coord_x]) # обучающая выборка
y_train = func(coord_x) # целевые выходные значения

# здесь продолжайте программу
XT_X = x_train.T @ x_train
XT_X = np.linalg.inv(XT_X)
XT_Y = x_train.T @ y_train
w = XT_X @ XT_Y
print(w)
Q = sum((y_train - (x_train @ w.T))**2) / len(y_train)
print(Q)