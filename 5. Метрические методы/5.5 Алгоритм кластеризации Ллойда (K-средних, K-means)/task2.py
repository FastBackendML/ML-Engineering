import numpy as np
import matplotlib.pyplot as plt

T = [[(365, 200), (390, 180), (350, 172), (400, 171)], [(77, 150), (100, 200), (50, 130)], [(250, 100), (170, 88), (280, 102), (230, 108)]]
data_x = [(48, 118), (74, 96), (103, 82), (135, 76), (162, 79), (184, 97), (206, 111), (231, 118), (251, 118), (275, 110), (298, 86), (320, 68), (344, 62), (376, 61), (403, 75), (424, 95), (440, 114), (254, 80), (219, 85), (288, 66), (260, 92), (201, 76), (162, 66), (127, 135), (97, 143), (83, 160), (82, 177), (88, 199), (105, 205), (135, 208), (151, 198), (157, 169), (153, 152), (117, 158), (106, 168), (106, 185), (123, 188), (125, 171), (139, 163), (139, 183), (358, 127), (328, 132), (313, 146), (300, 169), (300, 181), (308, 197), (326, 206), (339, 209), (370, 199), (380, 184), (380, 147), (343, 154), (329, 169), (332, 184), (345, 185), (363, 159), (361, 177), (344, 169), (311, 175), (351, 89), (134, 96)]

K = 3                       # число кластеров
N = 10
func = lambda m, k: np.mean((np.array(m) - np.array(k)) ** 2)

# здесь продолжайте программу
cl_centers = [np.mean(i, axis=0) for i in T]

for _ in range(N):
    X = [[] for i in range(K)]

    for k in data_x:
        dist = [func(m, k) for m in cl_centers]
        index = np.argmin(dist)
        X[index].append(k)

    cl_centers = [np.mean(i, axis=0) for i in X]

for i in range(K):
    X[i].extend(T[i])


# Visualisation
colors = ['r', 'b', 'g']
plt.figure(figsize=(10, 6))

for i, x in enumerate(X):
    xx = np.array(x)
    if len(xx) > 0:
        plt.scatter(xx[:, 0], xx[:, 1], c=colors[i], label=f'Cluster {i+1}', alpha=0.6)

cl_centers = np.array(cl_centers)
plt.scatter(cl_centers[:,0], cl_centers[:,1], c='black', marker='x', s=100, label='Final Centers')

plt.title('Clustering Result with Final Centers')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.legend()
plt.show()




