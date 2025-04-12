import numpy as np


def func(x):
    return 0.1 * x - np.cos(x/2) + 0.4 * np.sin(3*x) + 5


np.random.seed(0)

x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
y = func(x) + np.random.normal(0, 0.2, len(x)) # значения функции по оси ординат

K = lambda r: 1/np.sqrt(2 * np.pi) * np.exp(-r**2 / 2)
# здесь продолжайте программу
h = 0.5
y_est = []
for i in x:
    distance = np.abs(x - i)
    weight = K(distance / h)
    y_est.append(np.sum(weight * y) / np.sum(weight))


Q = sum((y_est - y)**2) / len(y_est)
print(Q)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
fig = plt.figure(figsize=(10,6))
line1 = plt.plot(y_est, color='blue', lw=2, ls='--', label='Y estimation')
line2 = plt.plot(y, color='red', lw=2, label='Y')
plt.title('Восстановление функции f(x) с помощью формулы ядерного сглаживания Надарая-Ватсона')
plt.legend()
plt.tight_layout()
plt.show()




