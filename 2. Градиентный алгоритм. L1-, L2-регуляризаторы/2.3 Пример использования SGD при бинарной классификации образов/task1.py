import numpy as np
import matplotlib.pyplot as plt
import time

# логарифмическая функция потерь
def loss(w, x, y):
    M = np.dot(w, x) * y
    return np.log2(1 + np.exp(-M))


# производная логарифмической функции потерь по вектору w
def df(w, x, y):
    M = np.dot(w, x) * y
    return -(np.exp(-M) * x.T * y) / ((1 + np.exp(-M)) * np.log(2))


data_x = [(3.0, 4.9), (2.7, 3.9), (3.0, 5.5), (2.6, 4.0), (2.9, 4.3), (3.1, 5.1), (2.2, 4.5), (2.3, 3.3), (2.7, 5.1), (3.3, 5.7), (2.8, 5.1), (2.8, 4.9), (2.5, 4.5), (2.8, 4.7), (3.2, 4.7), (3.2, 5.7), (2.8, 6.1), (3.6, 6.1), (2.8, 4.8), (2.9, 4.5), (3.1, 4.9), (2.3, 4.4), (3.3, 6.0), (2.6, 5.6), (3.0, 4.4), (2.9, 4.7), (2.8, 4.0), (2.5, 5.8), (2.4, 3.3), (2.8, 6.7), (3.0, 5.1), (2.3, 4.0), (3.1, 5.5), (2.8, 4.8), (2.7, 5.1), (2.5, 4.0), (3.1, 4.4), (3.8, 6.7), (3.1, 5.6), (3.1, 4.7), (3.0, 5.8), (3.0, 5.2), (3.0, 4.5), (2.7, 4.9), (3.0, 6.6), (2.9, 4.6), (3.0, 4.6), (2.6, 3.5), (2.7, 5.1), (2.5, 5.0), (2.0, 3.5), (3.2, 5.9), (2.5, 5.0), (3.4, 5.6), (3.4, 4.5), (3.2, 5.3), (2.2, 4.0), (2.2, 5.0), (3.3, 4.7), (2.7, 4.1), (2.4, 3.7), (3.0, 4.2), (3.2, 6.0), (3.0, 4.2), (3.0, 4.5), (2.7, 4.2), (2.5, 3.0), (2.8, 4.6), (2.9, 4.2), (3.1, 5.4), (2.5, 4.9), (3.2, 5.1), (2.8, 4.5), (2.8, 5.6), (3.4, 5.4), (2.7, 3.9), (3.0, 6.1), (3.0, 5.8), (3.0, 4.1), (2.5, 3.9), (2.4, 3.8), (2.6, 4.4), (2.9, 3.6), (3.3, 5.7), (2.9, 5.6), (3.0, 5.2), (3.0, 4.8), (2.7, 5.3), (2.8, 4.1), (2.8, 5.6), (3.2, 4.5), (3.0, 5.9), (2.9, 4.3), (2.6, 6.9), (2.8, 5.1), (2.9, 6.3), (3.2, 4.8), (3.0, 5.5), (3.0, 5.0), (3.8, 6.4)]
data_y = [1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1]

x_train = np.array([[1, x[0], x[1]] for x in data_x])
y_train = np.array(data_y)

n_train = len(x_train)  # размер обучающей выборки
w = [0.0, 0.0, 0.0]  # начальные весовые коэффициенты
nt = np.array([0.5, 0.01, 0.01])   # шаг обучения для каждого параметра w0, w1, w2
lm = 0.01  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
N = 1000  # число итераций алгоритма SGD


Qe = sum([loss(w, x_train[i], y_train[i]) for i in range(n_train)]) / n_train  # начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел
plt.ion()  # Включение интерактивного режима

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # Построение двух графиков на одной фигуре

coords_scatter_x = x_train[y_train == 1]  # Координаты точек, соответствующие y=1

coords_scatter_y = x_train[y_train == -1]  # Координаты точек, соответствующие y=-1

ax1.scatter(coords_scatter_x[:, 1], coords_scatter_x[:, 2], color='red', zorder=2)  # Построение точек

ax1.scatter(coords_scatter_y[:, 1], coords_scatter_y[:, 2], color='blue', zorder=2)  # Построение точек

coords_line_x = [min(x_train[:, 1]), max(x_train[:, 1])]  # Начальные координаты точек линии

coords_line_y = [0, 0]  # Начальные координаты точек линии

line_1, = ax1.plot(coords_line_x, coords_line_y, color='green', zorder=2, label='Разделяющая линия')  # Построение линии

ax1.set_xlim(min(x_train[:, 1]) - 0.1, max(x_train[:, 1]) + 0.1)  # Установление пределов по шкале

ax1.set_ylim(min(x_train[:, 2]) - 0.1, max(x_train[:, 2]) + 0.1)  # Установление пределов по шкале

line_2, = ax2.plot([], [], color='red', label='Qe')  # Построение графика Qe

ax2.set_xlim(0, N)  # Установление пределов по шкале

ax2.set_ylim(0, 2)  # Установление пределов по шкале

Qe_values = []  # Значения Qe для построения графика

ax1.set_title('Построение разделяющей линии')  # Установка названия графика

ax2.set_title('Отображение изменений графика Qe')  # Установка названия графика

ax1.legend()  # Отображение легенды на графике

ax2.legend()  # Отображение легенды на графике

ax1.grid(zorder=1)  # Отображение сетки на графике

ax2.grid(zorder=1)  # Отображение сетки на графике

# здесь продолжайте программу
for i in range(N):
    k = np.random.randint(0, n_train - 1)
    Lwk = loss(w, x_train[k], y_train[k])
    dLwk = df(w, x_train[k], y_train[k])
    w = w - nt * dLwk
    Qe = lm * Lwk + (1 - lm) * Qe
    Qe_values.append(Qe)  # Добавляем новые значения Qe для построения графика

    coords_line_y = [(-w[1] * x - w[0]) / w[2] for x in coords_line_x]  # Задание новых координат линии

    line_1.set_ydata(coords_line_y)  # Установка новых координат линии (обновление существующих координат)

    line_2.set_data(range(len(Qe_values)),
                    Qe_values)  # Установка новых координат для графика Qe (обновление существующих координат)

    fig.canvas.draw()  # Перерисовывает текущую фигуру

    fig.canvas.flush_events()  # Очистка внутренних событий

    time.sleep(0.02)  # Установка задержки перед выполнением следующей операции

Q = sum([np.dot(w.T, x_train[i]) * y_train[i] < 0 for i in range(n_train)]) / n_train
print(Q)
plt.ioff()  # Отключение интерактивного режима

plt.show()  # Отображение графиков


