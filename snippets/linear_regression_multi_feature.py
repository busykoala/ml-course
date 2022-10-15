import matplotlib
matplotlib.use('Qt5Agg')

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


def gradient_descent(X, y, alpha=0.1, epochs=10):
    # add additional row of ones (theta_0 * 1)
    X = np.concatenate((np.ones([X.shape[0],1], X.dtype), X), 1)
    m = len(X)
    n = len(X[0])
    thetas = np.zeros(n)
    for _ in range(epochs):
        new_thetas = []
        for current_n in range(n):
            summ = 0
            for current_m in range(m):
                hyp = lambda x: sum([th * x for th, x in zip(thetas, x)])
                cc = (hyp(X[current_m]) - y[current_m]) * X[current_m][current_n]
                summ = summ + cc
            new_thetas.append(
                thetas[current_n] - alpha / m * summ
            )
        thetas = new_thetas
    return thetas


def gradient_descent_improved(X, y, alpha=0.6, epochs=10):
    # add additional row of ones (theta_0 * 1)
    X = np.concatenate((np.ones([X.shape[0],1], X.dtype), X), 1)
    m = len(X)
    n = len(X[0])
    thetas = np.zeros(n)
    for _ in range(epochs):
        hypothesis = np.dot(X, thetas)
        cost = hypothesis - y
        thetas = thetas - alpha / m * np.sum((cost * X.T), axis=1)
    return thetas


def plot_2D_set():
    X, y = datasets.load_diabetes(return_X_y=True)
    X = np.delete(X, np.s_[3:], axis=1)
    X = np.delete(X, np.s_[:2], axis=1)
    X_train = X[:-20]
    X_test = X[-20:]
    y_train = y[:-20]
    # y_test = y[-20:]

    thetas = gradient_descent(X_train, y_train, epochs=10000)
    pred = lambda x: thetas[0] + sum([th * x for th, x in zip(thetas[1:], x)])

    regr = LinearRegression().fit(X_train[:100], y_train[:100])
    scikit_pred = regr.predict(X_test)

    plt.scatter(X_train, y_train, label="Training Data")
    plt.scatter(X_test, [pred(x) for x in X_test], label="Test Data")
    plt.scatter(X_test, scikit_pred, label="Test Data Scikit")
    plt.show()


def run_multiD_set():
    X, y = datasets.load_diabetes(return_X_y=True)

    # normalize data
    minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    X = minmax_scaler.fit_transform(X)

    X_train = X[:-20]
    y_train = y[:-20]

    thetas = gradient_descent_improved(X_train, y_train, epochs=100000)
    regr = LinearRegression().fit(X_train, y_train)
    summ = np.sum((np.array(thetas[1:]) - regr.coef_)**2)
    print(f"sum of sqared diffs between gradient descent and scikit: {summ}")
    print("*" * 50)
    print(thetas)


def main():
    plot_2D_set()
    run_multiD_set()


if __name__ == "__main__":
    main()
