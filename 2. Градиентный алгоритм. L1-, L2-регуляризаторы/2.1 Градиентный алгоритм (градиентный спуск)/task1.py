import numpy as np
import matplotlib.pyplot as plt
import time

def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.1 * x ** 3


def df(x):
    return 0.5 + 0.4 * x - 0.3 * x**2


coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс
coord_y = func(coord_x) # значения по оси ординат (значения функции)

# здесь продолжайте программу
n = 0.01
N = 200
x = -4

f_plt = [func(x) for x in coord_x]
plt.ion()
fig,ax = plt.subplots()
ax.grid(True)
ax.plot(coord_x, f_plt)
point = ax.scatter(x, func(x), c='red')

for i in range(N):
        x = x - 0.01 * df(x)

        point.set_offsets([x, func(x)])

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.02)
