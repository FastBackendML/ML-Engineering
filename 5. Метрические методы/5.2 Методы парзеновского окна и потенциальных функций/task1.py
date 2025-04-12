import numpy as np
from sklearn.model_selection import train_test_split

# Генерация данных
np.random.seed(0)
n_feature = 2

r1, D1, mean1 = 0.7, 3.0, [3, 3]
V1 = [[D1 * r1 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r2, D2, mean2 = 0.5, 2.0, [1, 1]
V2 = [[D2 * r2 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r3, D3, mean3 = -0.7, 1.0, [-2, -2]
V3 = [[D3 * r3 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

N1, N2, N3 = 200, 150, 190
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T
x3 = np.random.multivariate_normal(mean3, V3, N3).T

data_x = np.hstack([x1, x2, x3]).T
data_y = np.hstack([np.zeros(N1), np.ones(N2), np.ones(N3) * 2])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123, test_size=0.5, shuffle=True)

# Параметры метода
h = 1.0
classes = np.unique(y_train)

def gaussian_kernel(r):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-r**2 / 2)

# Предсказания
predict = []
for x in x_test:
    scores = []
    for c in classes:
        # фильтруем обучающие объекты текущего класса
        mask = (y_train == c)
        x_c = x_train[mask]

        # считаем расстояния: Манхэттен
        distances = np.sum(np.abs(x_c - x), axis=1)

        # применяем ядро
        kernel_vals = gaussian_kernel(distances / h)

        # итоговая сумма
        scores.append(np.sum(kernel_vals))

    # выбираем класс с максимальным score
    predicted_class = np.argmax(scores)
    predict.append(predicted_class)

# Подсчёт ошибки
predict = np.array(predict)
Q = np.mean(predict != y_test)

print(f"Качество классификации (ошибка): Q = {Q:.4f}")




