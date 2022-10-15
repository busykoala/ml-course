import matplotlib
matplotlib.use('Qt5Agg')

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


def gradient_descent(X, Y, alpha=0.01, epochs=30):
    """Calculate the coefficients for a linear function using gradient descent
    :param X: feature vector
    :param Y: target vector
    :param alpha: learning rate
    :param epochs: max number of iterations
    """
    theta0 = 0
    theta1 = 0
    m = len(X)
    for _ in range(epochs):
        temp0 = theta0 - alpha / m * sum([
            (theta0 + theta1 * x) - y
            for x, y in zip(X, Y)])
        temp1 = theta1 - alpha / m * sum([
            ((theta0 + theta1 * x) - y) * x
            for x, y in zip(X, Y)])
        theta0 = temp0
        theta1 = temp1
    return theta0, theta1


def print_result(theta0, theta1, hypothesis_func):
    print(f"h(x) = {theta0} + {theta1} * x")
    print(f"e.g. h(3.1) = {hypothesis_func(3.1)}")


def print_epoch_evolution(X, Y):
    th0_before = 0
    th1_before = 0
    for epochs in [10, 100, 1000, 10000, 100000]:
        theta0, theta1 = gradient_descent(X, Y, epochs=epochs)
        print(f"epochs: {epochs}")
        print(f"diff th0: {theta0 - th0_before} | diff th1: {theta1 - th1_before}")
        th0_before = theta0
        th1_before = theta1


def plot_original_vs_hypothesis(X, Y, hypothesis_func, hypothesis_func_scikit):
    Y_h = [hypothesis_func(x) for x in X]
    Y_h_sci = [hypothesis_func_scikit([[x]]) + 0.05 for x in X]  # add 0.05 to make line visible ;)
    plt.plot(X, Y, label="Original")
    plt.plot(X, Y_h, label="Hypothesis")
    plt.plot(X, Y_h_sci, label="Hypothesis Scikit")
    plt.legend()
    plt.show()


def scikit_linear_regression(X, Y):
    X_reshaped = X.reshape(-1, 1)  # each row is an array of features
    reg = LinearRegression().fit(X_reshaped, Y)
    return reg.predict


def main():
    X = np.array([0.1, 1, 2.3, 3.1, 4, 5.4])
    Y = np.array([0, 2.5, 4.1, 6.2, 8.1, 10.1])
    theta0, theta1 = gradient_descent(X, Y, epochs=100000)
    hypothesis_func = lambda x: theta0 + theta1 * x
    hypothesis_func_scikit = scikit_linear_regression(X, Y)
    print_result(theta0, theta1, hypothesis_func)
    print_epoch_evolution(X, Y)
    plot_original_vs_hypothesis(X, Y, hypothesis_func, hypothesis_func_scikit)


if __name__ == "__main__":
    main()
