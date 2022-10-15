from sklearn import preprocessing
import numpy as np
from sklearn import datasets


def gradient_descent(X, y, alpha=0.6, epochs=100000):
    X = np.concatenate((np.ones([X.shape[0],1], X.dtype), X), 1)
    m = len(X)
    n = len(X[0])
    thetas = np.zeros(n)
    for _ in range(epochs):
        hypothesis = np.dot(X, thetas)
        cost = hypothesis - y
        thetas = thetas - alpha / m * np.sum((cost * X.T), axis=1)
    return thetas


def normal_equation(X, y):
    X = np.concatenate((np.ones([X.shape[0],1], X.dtype), X), 1)
    thetas = np.linalg.inv(X.T @ X) @ X.T @ y
    return thetas


def run_normal_eq():
    X, y = datasets.load_diabetes(return_X_y=True)
    minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    X = minmax_scaler.fit_transform(X)
    X_train = X[:-20]
    y_train = y[:-20]
    thetas_n = normal_equation(X_train, y_train)
    thetas_g = gradient_descent(X_train, y_train)
    print(f"Thetas normal equation: {[round(x, 4) for x in thetas_n]}")
    print("*" * 40)
    print(f"Thetas gradient descent: {[round(x, 4) for x in thetas_g]}")


def main():
    run_normal_eq()


if __name__ == "__main__":
    main()
