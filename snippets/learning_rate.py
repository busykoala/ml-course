import numpy as np
from sklearn import datasets


def find_good_learning_rate(X, y):
    """A systematical approach to likely find a decent alpha.

    The idea is to compare the square sum over two epochs. If the data is
    normalized it is likely that comparing two epoches makes sure the data is
    converging and does not jump forth and back.
    """
    X = np.concatenate((np.ones([X.shape[0],1], X.dtype), X), 1)
    m = len(X)
    n = len(X[0])

    def get_thetas(thetas, alpha):
        hypothesis = np.dot(X, thetas)
        cost = hypothesis - y
        return thetas - alpha / m * np.sum((cost * X.T), axis=1)

    alpha = 10
    converges = False

    while not converges:
        thetas_0 = np.zeros(n)
        thetas_1 = get_thetas(thetas_0, alpha)
        thetas_2 = get_thetas(thetas_1, alpha)
        thetas_3 = get_thetas(thetas_2, alpha)

        diff_0_1 = np.sum((thetas_1 - thetas_0)**2)
        diff_1_2 = np.sum((thetas_2 - thetas_1)**2)
        diff_2_3 = np.sum((thetas_3 - thetas_2)**2)

        if diff_2_3 < diff_1_2 and diff_1_2 < diff_0_1:
            print(f"Alpha: {alpha}")
            converges = True
        else:
            print(f"Alpha {alpha} too high")
            alpha = alpha * 0.7
    return alpha


def main():
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train = X[:-20]
    y_train = y[:-20]

    alpha = find_good_learning_rate(X_train, y_train)
    print(f"Alpha {alpha} seems to converge")


if __name__ == "__main__":
    main()
