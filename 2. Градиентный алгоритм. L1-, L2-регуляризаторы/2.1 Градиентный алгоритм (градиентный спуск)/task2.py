import numpy as np
import matplotlib.pyplot as plt
import time

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.1 * x**2 - np.sin(x) + 5.


# здесь объявляйте необходимые функции


coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001]) # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.]) # начальные значения параметров модели
N = 200 # число итераций градиентного алгоритма

# здесь продолжайте программу
X = np.array([[1, x, x**2, x**3] for x in coord_x])

plt.ion()
fig, fx = plt.subplots()
fx.grid(True)
fx.plot(coord_x, coord_y)
approx = fx.plot(coord_x, X @ w.T, c="red")[0]

for n in range(N):
    dQ = 2/sz * X.T @ ((X @ w.T) - coord_y)
    w = w - eta * dQ
    approx.set_ydata(X @ w.T)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)

Q = 1/sz * sum(((X @ w.T) - coord_y)**2)
print(w)
print(Q)
plt.ioff()
plt.show()
