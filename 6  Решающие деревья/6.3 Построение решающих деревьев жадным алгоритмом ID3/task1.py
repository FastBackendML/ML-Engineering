import matplotlib.pyplot as plt
import numpy as np


data_x = [(5.8, 2.7), (6.7, 3.1), (5.7, 2.9), (5.5, 2.4), (4.8, 3.4), (5.4, 3.4), (4.8, 3.0), (5.5, 2.5), (5.3, 3.7), (7.0, 3.2), (5.6, 2.9), (4.9, 3.1), (4.8, 3.0), (5.0, 2.3), (5.2, 3.4), (5.1, 3.8), (5.0, 3.0), (5.0, 3.3), (4.6, 3.1), (5.5, 2.6), (5.0, 3.5), (6.7, 3.0), (6.0, 2.2), (4.8, 3.1), (6.4, 2.9), (5.6, 3.0), (4.4, 3.0), (4.9, 2.4), (5.6, 3.0), (5.0, 3.6), (5.1, 3.3), (5.8, 4.0), (5.5, 2.4), (5.2, 2.7), (5.1, 3.8), (5.1, 3.5), (5.5, 4.2), (4.9, 3.1), (5.9, 3.2), (5.7, 2.6), (4.7, 3.2), (5.4, 3.9), (5.8, 2.6), (5.1, 3.4), (6.4, 3.2), (5.8, 2.7), (5.6, 2.7), (5.7, 2.8), (5.4, 3.0), (5.0, 3.2), (4.6, 3.4), (6.0, 2.7), (6.6, 3.0), (4.9, 3.0), (4.9, 3.6), (4.4, 3.2), (5.4, 3.4), (6.0, 3.4), (5.9, 3.0), (6.1, 2.8), (5.1, 3.7), (5.5, 3.5), (6.1, 3.0), (6.2, 2.2), (5.7, 3.0), (5.2, 3.5), (5.4, 3.7), (4.6, 3.2), (5.2, 4.1), (5.0, 2.0), (6.8, 2.8), (5.0, 3.5), (6.7, 3.1), (6.3, 3.3), (6.0, 2.9), (4.7, 3.2), (6.6, 2.9), (5.6, 2.5), (4.4, 2.9), (6.2, 2.9), (6.1, 2.9), (4.3, 3.0), (6.9, 3.1), (5.7, 3.8), (5.4, 3.9), (6.1, 2.8), (4.6, 3.6), (5.5, 2.3), (4.8, 3.4), (6.5, 2.8), (6.3, 2.5), (5.1, 3.8), (5.7, 4.4), (5.0, 3.4), (4.5, 2.3), (5.7, 2.8), (5.1, 2.5), (5.1, 3.5), (6.3, 2.3), (5.0, 3.4)]
data_y = [1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1]

x_train = np.array(data_x)
y_train = np.array(data_y)


def calc_gini(y_subset):
    if len(y_subset) == 0:
        return 0
    pur_1 = np.sum(y_subset == 1) / len(y_subset)
    pur_2 = np.sum(y_subset == -1) / len(y_subset)
    return 1 - pur_1 ** 2 - pur_2 ** 2


best_feature = None
best_threshold = None
best_ig = -np.inf

s_0 = calc_gini(y_train)

for i in [0, 1]:  # признаки 0 и 1
    feature_values = x_train[:, i]
    thresholds = np.arange(min(feature_values) + 0.1, max(feature_values) - 0.1, 0.1)

    for t in thresholds:
        mask_left = feature_values < t
        mask_right = feature_values >= t

        y_left = y_train[mask_left]
        y_right = y_train[mask_right]

        s_left = calc_gini(y_left)
        s_right = calc_gini(y_right)

        ig = s_0 - (len(y_left) / len(y_train)) * s_left - (len(y_right) / len(y_train)) * s_right

        if ig > best_ig:
            best_ig = ig
            best_threshold = t
            best_feature = i

# Визуализация
plt.figure(figsize=(8, 6))

# Раскраска по меткам классов
colors = ['red' if y == -1 else 'blue' for y in y_train]
plt.scatter(x_train[:, 0], x_train[:, 1], c=colors, edgecolor='k', s=70, alpha=0.7)

# Рисуем разделяющую границу
if best_feature == 0:
    plt.axvline(x=best_threshold, color='green', linestyle='--', label=f'x0 = {best_threshold:.2f}')
else:
    plt.axhline(y=best_threshold, color='green', linestyle='--', label=f'x1 = {best_threshold:.2f}')

plt.xlabel('x0 (Признак 1)')
plt.ylabel('x1 (Признак 2)')
plt.title('Разделяющая граница по Gini-информации')
plt.legend()
plt.grid(True)
plt.show()




