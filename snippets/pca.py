from numpy.linalg import svd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def mean_normalization(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled


def find_optimal_k(X, variance_retained):
    max = (100 - variance_retained) / 100
    X = mean_normalization(X)
    sigma = (1/X.shape[0]) * X.T @ X
    _, S, _ = svd(sigma, full_matrices=False)
    for k in range(1, S.shape[0]):
        current = 1 - np.sum(S[:k]) / np.sum(S)
        if current <= max:
            return k


def pca(X, new_dimension):
    X = mean_normalization(X)
    sigma = (1/X.shape[0]) * X.T @ X
    _, _, Vt = svd(sigma, full_matrices=False)
    return X @ Vt[:new_dimension].T * -1


def sklearn_pca(X, new_dimension):
    X = mean_normalization(X)
    pca = PCA(n_components=new_dimension)
    return pca.fit_transform(X)


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data

    # variance retained in percent (0-100)
    k = find_optimal_k(X, variance_retained=99)

    X_one = pca(X, k)
    X_two = sklearn_pca(X, k)
