import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
N1 = 1000
N2 = 1000
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.5, shuffle=True)

# здесь продолжайте программу
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
predict = clf.predict(x_test)
w = np.array([clf.intercept_[0], *clf.coef_[0]])
t = 2
x_test = np.array([[1, x[0], x[1]] for x in x_test])
ax = [np.sign(w.T @ x - t) for x in x_test]
TP = sum(ax[i] == 1 and y_test[i] == 1 for i in range(len(y_test)))
FN = sum(ax[i] == -1 and y_test[i] == 1 for i in range(len(y_test)))
FP = sum(ax[i] == 1 and y_test[i] == -1 for i in range(len(y_test)))
TN = sum(ax[i] == -1 and y_test[i] == -1 for i in range(len(y_test)))
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
print(TPR, FPR)


# Задаём параметры для классов
np.random.seed(0)

r1, D1 = 0.2, 3.0
mean1, V1 = [2, -2], [[D1, D1 * r1], [D1 * r1, D1]]

r2, D2 = 0.5, 2.0
mean2, V2 = [-1, -1], [[D2, D2 * r2], [D2 * r2, D2]]

N1, N2 = 1000, 1000
x1 = np.random.multivariate_normal(mean1, V1, N1)
x2 = np.random.multivariate_normal(mean2, V2, N2)

data_x = np.vstack([x1, x2])
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

# Разделим данные на train и test
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123, test_size=0.5, shuffle=True)

# Обучим линейный классификатор
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

# Получим предсказания и параметры гиперплоскости
w = clf.coef_[0]
b = clf.intercept_[0]

# Построим визуализацию
plt.figure(figsize=(10, 6))

# Отобразим точки из тестовой выборки
plt.scatter(x_test[y_test == -1][:, 0], x_test[y_test == -1][:, 1], color='red', label='Class -1', alpha=0.5)
plt.scatter(x_test[y_test == 1][:, 0], x_test[y_test == 1][:, 1], color='blue', label='Class +1', alpha=0.5)

# Построим разделяющую гиперплоскость и границы
xx = np.linspace(np.min(data_x[:, 0]), np.max(data_x[:, 0]), 100)
yy = -(w[0] * xx + b) / w[1]
yy_margin1 = -(w[0] * xx + b - 1) / w[1]
yy_margin2 = -(w[0] * xx + b + 1) / w[1]

plt.plot(xx, yy, 'k-', label='Decision boundary')
plt.plot(xx, yy_margin1, 'k--', alpha=0.5)
plt.plot(xx, yy_margin2, 'k--', alpha=0.5)

plt.title("SVM Classification with Linear Kernel")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
