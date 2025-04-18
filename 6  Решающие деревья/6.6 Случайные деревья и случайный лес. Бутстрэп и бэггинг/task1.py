import numpy as np
from sklearn.ensemble import RandomForestRegressor

x = np.arange(-3, 3, 0.1)
y = 0.3 * x + np.cos(2*x) + 0.2 * np.sin(7*x) # + np.random.normal(0.0, 0.1, n_samples)
x = x.reshape(-1, 1)

T = 5  # число деревьев

# здесь продолжайте программу
rf = RandomForestRegressor(max_depth=8, n_estimators=T, random_state=1)
rf.fit(x, y)
pr_y = rf.predict(x)
Q = np.sum((pr_y - y)**2) / len(pr_y)
print(Q)