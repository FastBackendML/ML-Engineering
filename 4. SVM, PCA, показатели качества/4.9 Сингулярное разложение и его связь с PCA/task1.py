import numpy as np

np.random.seed(0)

n_total = 1000 # число образов выборки
n_features = 200 # число признаков

table = np.zeros(shape=(n_total, n_features))

for _ in range(100):
    i, j = np.random.randint(0, n_total), np.random.randint(0, n_features)
    table[i, j] = np.random.randint(1, 10)

# матрицу table не менять

# здесь продолжайте программу
L, W = np.linalg.eig(table.T @ table  / len(table))
WW = W[np.argsort(L)]
WW_nonzero = np.array([w for l, w in zip(L, WW) if l >= 0.01])
# W.shape, WW_nonzero.shape

data_x = table @ WW_nonzero.T
print(data_x)

