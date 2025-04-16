import numpy as np

x = np.arange(-2, 3, 0.1)
y = -x + 0.2 * x ** 2 - 0.5 * np.sin(4*x) + np.cos(2*x)


def HR_(y):
    return np.square(y.mean() - y).sum()


HR1 = HR_(y[x < 0])
HR2 = HR_(y[~(x < 0)])

IG = HR_(y) - (np.sum(x < 0) * HR1 + np.sum(~(x < 0)) * HR2) / len(y)
print(HR_(y), HR1, HR2, IG)
