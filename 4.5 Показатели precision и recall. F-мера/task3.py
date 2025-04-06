import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

np.random.seed(0)

# исходные параметры распределений классов
r1 = 0.2
D1 = 3.0
mean1 = [2, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-1, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N1 = 2500
N2 = 1500
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.4, shuffle=True)

# здесь продолжайте программу
cfl = svm.SVC(kernel='linear')
cfl.fit(x_train, y_train)
prediction = cfl.predict(x_test)
w = [cfl.intercept_[0], *cfl.coef_[0]]
TP = sum([prediction[i] == 1 and y_test[i] == 1 for i in range(len(y_test))])
TN = sum([prediction[i] == -1 and y_test[i] == -1 for i in range(len(y_test))])
FP = sum([prediction[i] == 1 and y_test[i] == -1 for i in range(len(y_test))])
FN = sum([prediction[i] == -1 and y_test[i] == 1 for i in range(len(y_test))])
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F = (2 * precision * recall) / (precision + recall)
B = 0.5
Fb = ((1 + B**2) * precision * recall) / (B**2 * precision + recall)
print(precision, recall, F, Fb, sep='\n')

import matplotlib.pyplot as plt

# Разделим тестовые точки по предсказанному классу
x_test = np.array(x_test)
y_test = np.array(y_test)

pos = x_test[prediction == 1]
neg = x_test[prediction == -1]

# Построим сетку для отображения границы решений
x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
y_min, y_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = cfl.decision_function(grid).reshape(xx.shape)

# Визуализация
plt.figure(figsize=(10, 6))

# Фон — раскраска классов
plt.contourf(xx, yy, Z, levels=[-1e10, 0, 1e10], colors=["#FFEEEE", "#EEEEFF"], alpha=0.5)

# Граница решений
plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

# Точки
plt.scatter(pos[:, 0], pos[:, 1], color='blue', label='Класс +1', s=30, edgecolors='k')
plt.scatter(neg[:, 0], neg[:, 1], color='red', label='Класс -1', s=30, edgecolors='k')

# Подписи
plt.title("SVM с линейным ядром — Граница решений")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.legend()

# Метрики
plt.text(x_min + 0.5, y_max - 0.5,
         f"Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1: {F:.2f}\nF0.5: {Fb:.2f}",
         fontsize=12, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))

plt.grid(True)
plt.tight_layout()
plt.show()