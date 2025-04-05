import numpy as np

w = np.array([-330, 1, 3, 2, -0.15])
y = 1
x = np.array([1, 240, 80, 1, 1000])
M = w.T @ x * y
print(M)
P = 1/ (1 + np.exp(-M))
print(P)