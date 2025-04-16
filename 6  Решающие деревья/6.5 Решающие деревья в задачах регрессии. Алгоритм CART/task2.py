import numpy as np

x = np.arange(-2, 3, 0.1)
y = -x + 0.2 * x ** 2 - 0.5 * np.sin(4*x) + np.cos(2*x)

# здесь продолжайте программу
th = 0
IG = 0


def impurity(y):
    if len(y) == 0:
        return 0
    return np.square(y.mean() - y).sum()

HR = impurity(y)

for t in x:
    left = x < t
    right = x >= t
    HR1 = impurity(y[left])
    HR2 = impurity(y[right])
    ig = HR - (len(y[left]) / len(y)) * HR1 - (len(y[right]) / len(y)) * HR2
    if ig > IG:
        IG = ig
        th = t

print(th, IG)