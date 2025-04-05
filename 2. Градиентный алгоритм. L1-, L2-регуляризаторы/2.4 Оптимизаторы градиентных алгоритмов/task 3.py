import numpy as np


def func(x):
    return 2 * x + 0.1 * x ** 3 + 2 * np.cos(3 * x)


def df(x):
    return 2 + 0.3 * x ** 2 - 6 * np.sin(3 * x)


# здесь объявляйте функцию df (производную) и продолжайте программу
n = 0.5
N = 200
x0 = 4.0
a = 0.8
G = 0
ϵ = 0.01
x = x0
for i in range(N):
    G = a * G + (1 - a) * df(x) * df(x)
    x = x - n * df(x) / (np.sqrt(G) + ϵ)
