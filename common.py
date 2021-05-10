import numpy as np
import scipy.spatial.distance as scipy_dist
from scipy.spatial import distance_matrix
import numpy.linalg as linalg


def norm(X, r=1):
    m = linalg.norm(X, ord=r)
    print(m)


def norm_pair(X, r=1):
    m = distance_matrix(X, X, p=r)
    print(m)
    return m


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
X: nxm matrix (n features, m observations)
"""


def pca(X):
    mu = np.mean(X, axis=0)
    print(X)
    print(mu)
    x_tilde = X - mu

    x_tilde = x_tilde.T
    print(f'mu = {mu}, x_tilde = {x_tilde}')

    cov_X = np.cov(x_tilde, bias=True)
    print(f'cov_X = {cov_X}')

    eigen_values, eigen_vectors_norm = linalg.eig(cov_X)

    print(
        f'eigen_values = {eigen_values}, eigen_vectors_norm = {eigen_vectors_norm}')

    print(np.dot(X, eigen_vectors_norm[0]))


"""
    P: probability of each class
"""


def gini(P):
    sum_of_p_square = np.square(np.sum(P))

    res = 1 - sum_of_p_square

    print(res)

    return res


"""
    P: probability of each class
"""


def entropy(P):
    from scipy.stats import entropy

    res = entropy(P, base=2)

    print(res)

    return res


"""
row: # of observation
column: # of features
"""
if __name__ == '__main__':
    pass
