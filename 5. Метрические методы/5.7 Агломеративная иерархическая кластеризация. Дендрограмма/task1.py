import numpy as np
from sklearn.cluster import AgglomerativeClustering

X = [(189, 185), (172, 205), (156, 221), (154, 245), (164, 265), (183, 275), (204, 276), (227, 271), (241, 255), (250, 229), (240, 197), (217, 183), (194, 202), (179, 224), (179, 248), (199, 249), (197, 227), (211, 214), (211, 242), (210, 265), (226, 237), (218, 196), (79, 106), (97, 132), (117, 159), (138, 174), (148, 163), (140, 145), (121, 123), (112, 108), (89, 92), (282, 162), (298, 180), (344, 154), (344, 113), (362, 67), (397, 77), (412, 121), (379, 112), (377, 148), (312, 130)]

X = np.array(X)

K = 3
# здесь продолжайте программу
clustering = AgglomerativeClustering(n_clusters=K, linkage="ward", metric="euclidean")
res = clustering.fit_predict(X)
X1, X2, X3 = [X[res == i] for i in [0, 1, 2]]


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(X1[:,0], X1[:,1], color='red', label=f'Cluster {0}', alpha=0.6)
plt.scatter(X2[:,0], X2[:,1], color='blue', label=f'Cluster {1}', alpha=0.6)
plt.scatter(X3[:,0], X3[:,1], color='green', label=f'Cluster {2}', alpha=0.6)

plt.legend()
plt.title('Агломеративная иерархическая кластеризация. Дендрограмма')
plt.tight_layout()
plt.show()

