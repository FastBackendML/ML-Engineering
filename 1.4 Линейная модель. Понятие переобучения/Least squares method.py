import numpy as np

np.random.seed(0) # псевдослучайные числа образуют одну и ту же последовательность (при каждом запуске)
x = np.arange(-1.0, 1.0, 0.1) # аргумент [-1; 1] с шагом 0,1


model_a = lambda xx, ww: (ww[0] + ww[1] * xx) # модель
Y = -5.2 + 0.7 * x + np.random.normal(0, 0.1, len(x)) # вектор целевых значений

# здесь продолжайте программу

ones = np.ones(len(x))
train = np.column_stack([ones, x])

mx = np.sum(x)/len(x)
my = np.sum(Y)/len(x)
a2 = np.dot(x.T, x)/len(x)
a11 = np.dot(x.T, Y)/len(x)
kk = (a11 - mx*my)/(a2-mx**2)
bb = my-kk*mx
w = np.array([bb, kk])
print(w)