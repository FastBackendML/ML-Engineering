import numpy as np


def func(x):
    return 0.4 * x + 0.1 * np.sin(2*x) + 0.2 * np.cos(3*x)

# здесь объявляйте функцию df (производную) и продолжайте программу
def df(x):
    return 0.4 + 0.2 * np.cos(2 * x) - 0.6 * np.sin(3 * x)


eta = 1
N = 500
x0 = 4.0
y = 0.7
v = 0
x = x0
for i in range(N):
    v = y * v + (1 - y) * eta * df(x - y * v)
    x -= v
