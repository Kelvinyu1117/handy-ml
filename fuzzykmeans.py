import numpy as np
import scipy.spatial.distance as scipy_dist
from scipy.spatial import distance_matrix
import numpy.linalg as linalg


def sqaure_dist(a, b):
    return (a - b) ** 2

# only for two clusters


def fuzzy_kmean(X, C_init):
    def get_weight(X, C, i):
        a = sqaure_dist(X, C[i])
        b = sqaure_dist(X, C[i])
        for j in range(len(C)):
            if(j == i):
                continue
            else:
                b = np.array([b, sqaure_dist(X, C[j])])

        b = np.sum(b, axis=0)
        return a/b

    def get_centroid(X, partition):
        p = partition ** 2
        c1 = np.sum(X * p[:, 0]) / np.sum(p[:, 0])
        c2 = np.sum(X * p[:, 1]) / np.sum(p[:, 1])

        return np.array([c1, c2])
    C = C_init

    for i in range(10):
        print(f'Iteration: {i + 1}, old centroid: {C}')
        partition = np.zeros((len(X), len(C)))

        partition[:, 0] = get_weight(X, C, 1)
        partition[:, 1] = get_weight(X, C, 0)

        C = get_centroid(X, partition)
        print(f'partition matrix: {partition}\n new centriod: {C}')
        print(f'----------------------------------------------------------')


if __name__ == '__main__':
    X = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    C = np.array([4, 11])
    fuzzy_kmean(X, C)
