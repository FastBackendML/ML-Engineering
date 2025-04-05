import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns

def func(x):
    return np.sin(0.5*x) + 0.2 * np.cos(2*x) - 0.1 * np.sin(4 * x) + 3


# обучающая выборка
coord_x = np.expand_dims(np.arange(-4.0, 6.0, 0.1), axis=1)
coord_y = func(coord_x).ravel()

# здесь продолжайте программу
svr = svm.SVR(kernel='rbf')   # SVM с нелинейным ядром

x_train = coord_x[::3]
y_train = coord_y[::3]

svr.fit(x_train, y_train)
predict = svr.predict(coord_x)
Q = sum((predict - coord_y)**2) / len(coord_x)

figure = plt.figure(figsize=(10, 5))
sns.set_style('whitegrid')
line1 = plt.plot(coord_y, color='blue', label='original', lw=1, marker='.')
line2 = plt.plot(predict, color='red', label='approximate', ls='--', lw=1)
plt.tight_layout()
plt.legend()
plt.show()
