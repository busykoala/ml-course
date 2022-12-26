from numpy.linalg import svd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def mean_normalization(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled


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

    X_one = pca(X, 2)
    X_two = sklearn_pca(X, 2)
