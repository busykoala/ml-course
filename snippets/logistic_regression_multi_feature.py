import matplotlib
matplotlib.use('Qt5Agg')

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


def fit(X, y, alpha=0.001, epochs=100):
    X = np.concatenate((np.ones([X.shape[0],1], X.dtype), X), 1)
    n = len(X[0])
    m = len(X)
    thetas = np.zeros(n)
    for _ in range(epochs):
        g_hypo = 1/(1 + np.e**(-(thetas.T * X)))
        cost = g_hypo.T - y
        thetas = thetas - alpha / m * np.sum((cost * X.T), axis=1)
    return thetas


def run_multiD_set():
    X, y = datasets.load_iris(return_X_y=True)

    # normalize data by column (axis=0)
    X = preprocessing.normalize(X, axis=0, norm="l1")

    X_train = X[:-20]
    y_train = y[:-20]

    thetas = fit(X_train, y_train, epochs=100000)
    regr = LogisticRegression().fit(X_train, y_train)
    summ = np.sum((np.array(thetas[1:]) - regr.coef_)**2)
    print(f"sum of sqared diffs between gradient descent and scikit: {summ}")
    print("*" * 50)
    print(thetas)


def main():
    run_multiD_set()


if __name__ == "__main__":
    main()
