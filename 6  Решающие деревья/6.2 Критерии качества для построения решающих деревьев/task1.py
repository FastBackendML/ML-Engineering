import numpy as np

np.random.seed(0)
X = np.random.randint(0, 2, size=200)

split = 150

def calc_gini(x):
    prob_0 = len(x[x == 0]) / len(x)
    prob_1 = len(x[x == 1]) / len(x)
    return 1 - prob_0**2 - prob_1**2

s_0 = calc_gini(X)
s_1 = calc_gini(X[:split])
s_2 = calc_gini(X[split:])
IG = s_0 - s_1 * len(X[:split]) / len(X) - s_2 * len(X[split:]) / len(X)
print(IG)