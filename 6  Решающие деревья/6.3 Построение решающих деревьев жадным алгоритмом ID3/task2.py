import numpy as np
import matplotlib.pyplot as plt


data_x = [(5.8, 2.7), (6.7, 3.1), (5.7, 2.9), (5.5, 2.4), (4.8, 3.4), (5.4, 3.4), (4.8, 3.0), (5.5, 2.5), (5.3, 3.7), (7.0, 3.2), (5.6, 2.9), (4.9, 3.1), (4.8, 3.0), (5.0, 2.3), (5.2, 3.4), (5.1, 3.8), (5.0, 3.0), (5.0, 3.3), (4.6, 3.1), (5.5, 2.6), (5.0, 3.5), (6.7, 3.0), (6.0, 2.2), (4.8, 3.1), (6.4, 2.9), (5.6, 3.0), (4.4, 3.0), (4.9, 2.4), (5.6, 3.0), (5.0, 3.6), (5.1, 3.3), (5.8, 4.0), (5.5, 2.4), (5.2, 2.7), (5.1, 3.8), (5.1, 3.5), (5.5, 4.2), (4.9, 3.1), (5.9, 3.2), (5.7, 2.6), (4.7, 3.2), (5.4, 3.9), (5.8, 2.6), (5.1, 3.4), (6.4, 3.2), (5.8, 2.7), (5.6, 2.7), (5.7, 2.8), (5.4, 3.0), (5.0, 3.2), (4.6, 3.4), (6.0, 2.7), (6.6, 3.0), (4.9, 3.0), (4.9, 3.6), (4.4, 3.2), (5.4, 3.4), (6.0, 3.4), (5.9, 3.0), (6.1, 2.8), (5.1, 3.7), (5.5, 3.5), (6.1, 3.0), (6.2, 2.2), (5.7, 3.0), (5.2, 3.5), (5.4, 3.7), (4.6, 3.2), (5.2, 4.1), (5.0, 2.0), (6.8, 2.8), (5.0, 3.5), (6.7, 3.1), (6.3, 3.3), (6.0, 2.9), (4.7, 3.2), (6.6, 2.9), (5.6, 2.5), (4.4, 2.9), (6.2, 2.9), (6.1, 2.9), (4.3, 3.0), (6.9, 3.1), (5.7, 3.8), (5.4, 3.9), (6.1, 2.8), (4.6, 3.6), (5.5, 2.3), (4.8, 3.4), (6.5, 2.8), (6.3, 2.5), (5.1, 3.8), (5.7, 4.4), (5.0, 3.4), (4.5, 2.3), (5.7, 2.8), (5.1, 2.5), (5.1, 3.5), (6.3, 2.3), (5.0, 3.4)]
data_y = [1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1]

x_train = np.array(data_x)
y_train = np.array(data_y)

def calc_gini(y_subset):
    if len(y_subset) == 0:
        return 0
    p1 = np.sum(y_subset == 1) / len(y_subset)
    p2 = np.sum(y_subset == -1) / len(y_subset)
    return 1 - p1**2 - p2**2

def best_split(X, y):
    best_feature = None
    best_threshold = None
    best_ig = -np.inf
    s_0 = calc_gini(y)

    for i in range(X.shape[1]):
        thresholds = np.unique(X[:, i])
        for t in thresholds:
            mask_left = X[:, i] < t
            mask_right = X[:, i] >= t

            y_left = y[mask_left]
            y_right = y[mask_right]

            s_left = calc_gini(y_left)
            s_right = calc_gini(y_right)

            ig = s_0 - (len(y_left) / len(y)) * s_left - (len(y_right) / len(y)) * s_right

            if ig > best_ig:
                best_ig = ig
                best_threshold = t
                best_feature = i

    return best_feature, best_threshold

# Первый уровень разбиения
f1, t1 = best_split(x_train, y_train)

# Маски для левой и правой подгруппы
left_mask = x_train[:, f1] < t1
right_mask = ~left_mask

# Второй уровень разбиения
f2_left, t2_left = best_split(x_train[left_mask], y_train[left_mask])
f2_right, t2_right = best_split(x_train[right_mask], y_train[right_mask])

# Визуализация
plt.figure(figsize=(10, 8))
colors = ['red' if y == -1 else 'blue' for y in y_train]
plt.scatter(x_train[:, 0], x_train[:, 1], c=colors, edgecolor='k', s=70, alpha=0.7)

# Первая граница
if f1 == 0:
    plt.axvline(x=t1, color='green', linestyle='--', linewidth=2)
else:
    plt.axhline(y=t1, color='green', linestyle='--', linewidth=2)

# Вторая граница (в левой подгруппе)
if f2_left == 0:
    plt.axvline(x=t2_left, color='purple', linestyle='-.', linewidth=2)
else:
    plt.axhline(y=t2_left, color='purple', linestyle='-.', linewidth=2)

# Вторая граница (в правой подгруппе)
if f2_right == 0:
    plt.axvline(x=t2_right, color='orange', linestyle='-.', linewidth=2)
else:
    plt.axhline(y=t2_right, color='orange', linestyle='-.', linewidth=2)

plt.title('2D-решающее дерево (глубина = 2)')
plt.xlabel('x0')
plt.ylabel('x1')
plt.grid(True)
plt.show()