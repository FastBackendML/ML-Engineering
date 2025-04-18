# @title Наивный Байес vs Линейная регрессия (МНК)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score

# Генерация данных с заданными матожиданиями и дисперсией
# Нули в дисперсии как раз показывают, что признаки независимы
X_class1 = np.random.multivariate_normal([5.0, 5.5], [[0.2, 0], [0, 0.2]], 100)
X_class2 = np.random.multivariate_normal([3.5, 3.5], [[0.8, 0], [0, 0.8]], 100)
X = np.vstack((X_class1, X_class2))
y = np.hstack((np.zeros(100), np.ones(100)))  # 0 - первый класс, 1 - второй класс

# Обучение наивного байесовского классификатора с гаусовским распределением
nb_model = GaussianNB()
nb_model.fit(X, y)

# Обучение линейной регрессии (МНК)
lr_model = LogisticRegression()
lr_model.fit(X, y)

# Подсчёт качества классификации (доли правильных предсказаний)
y_pred_nb = nb_model.predict(X)  # Предсказания Байеса
y_pred_lr = lr_model.predict(X)  # Предсказания МНК

q_nb = accuracy_score(y, y_pred_nb)  # Доля правильных предсказаний Байеса
q_lr = accuracy_score(y, y_pred_lr)  # Доля правильных предсказаний МНК

print(f"Доля правильных классификаций (Q):")
print(f"Наивный Байес: {q_nb:.2f}")
print(f"Логистическая регрессия (МНК): {q_lr:.2f}")

# Визуализация данных
plt.figure(figsize=(8, 8))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')
plt.legend(*scatter.legend_elements(), title="Классы")

# Построение границы решений Байеса
DecisionBoundaryDisplay.from_estimator(
    nb_model, X, cmap='coolwarm', alpha=0.2, ax=plt.gca(), response_method="predict"
)

# Построение разделяющей линии МНК
x_min, x_max = plt.xlim()
x_vals = np.linspace(x_min, x_max, 100)
y_vals = -(lr_model.coef_[0, 0] * x_vals + lr_model.intercept_[0]) / lr_model.coef_[0, 1]
plt.plot(x_vals, y_vals, color="green", label="Линейная регрессия (МНК)")

# Подписи и оформление
plt.title(f"Наивный Байес vs Линейная регрессия (МНК)\nQ (Байес)={q_nb:.2f}, Q (МНК)={q_lr:.2f}")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.legend()
plt.show()
