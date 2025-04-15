import numpy as np
from sklearn.cluster import DBSCAN

X = [(58, 138), (74, 96), (103, 82), (135, 76), (162, 79), (184, 97), (206, 111), (231, 118), (251, 118),  (275, 110), (298, 86), (320, 68), (344, 62), (376, 61), (403, 75), (414, 90), (430, 100), (254, 80), (219, 85), (288, 66), (260, 92), (201, 76), (162, 66), (127, 135), (97, 143), (83, 160), (82, 177), (88, 199), (105, 205), (135, 208), (151, 198), (157, 169), (153, 152), (117, 158), (106, 168), (106, 185), (123, 188), (125, 171), (139, 163), (139, 183), (358, 127), (328, 132), (313, 146), (300, 169), (300, 181), (308, 197), (326, 206), (339, 209), (370, 199), (380, 184), (380, 147), (343, 154), (329, 169), (332, 184), (345, 185), (363, 159), (361, 177), (344, 169), (311, 175), (351, 89), (134, 96)]
X = np.array(X)

# здесь продолжайте программу
clustering = DBSCAN(eps=35, min_samples=3, metric='euclidean')
res = clustering.fit_predict(X)

X1, X2, X3, Noise = [X[res==i] for i in [0, 1, 2, -1]]

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(X1[:,0], X1[:,1], color='red', label=f'Cluster {0}', alpha=0.6)
plt.scatter(X2[:,0], X2[:,1], color='blue', label=f'Cluster {1}', alpha=0.6)
plt.scatter(X3[:,0], X3[:,1], color='green', label=f'Cluster {2}', alpha=0.6)
plt.scatter(Noise[:,0], Noise[:,1], color='y', label=f'Cluster {-1}', alpha=0.6)
core_ind = X[clustering.core_sample_indices_]
plt.scatter(core_ind[:,0], core_ind[:,1], marker='x', color='black', label='Core points', alpha=0.6)
plt.legend()
plt.title('Алгоритм кластеризации DBSCAN')
plt.tight_layout()
plt.show()
