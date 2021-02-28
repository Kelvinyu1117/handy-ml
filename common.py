import numpy as np
import scipy.spatial.distance as scipy_dist
from scipy.spatial import distance_matrix
import numpy.linalg as linalg


def norm(X, r=1):
    m = linalg.norm(X, ord=1)
    print(m)


def norm_pair(X, r=1):
    m = distance_matrix(X, X, p=r)
    print(m)


"""
    corr(x, y) = cov(x, y)/(std(x) * std(y))
"""


def correlation_coef(X):
    m = np.corrcoef(X)
    print(m)


"""
    cosine_dist = cos(x, y) = x dot y/(|x||y|)
"""


def cosine_distance(X):
    m = 1 - scipy_dist.pdist(X, 'cosine')
    print(m)


"""
row: # of observation
column: # of features
"""
if __name__ == '__main__':
