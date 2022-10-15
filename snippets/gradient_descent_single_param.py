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


X = np.array([0.1, 1, 2.3, 3.1, 4, 5.4])
Y = np.array([0, 2.5, 4.1, 6.2, 8.1, 10.1])
theta0, theta1 = gradient_descent(X, Y, epochs=100000)
hypothesis_func = lambda x: theta0 + theta1 * x

print(f"h(x) = {theta0} + {theta1} * x")
print(f"e.g. h(3.1) = {hypothesis_func(3.1)}")

print("*" * 50)
th0_before = 0
th1_before = 0
for epochs in [10, 100, 1000, 10000, 100000]:
    theta0, theta1 = gradient_descent(X, Y, epochs=epochs)
    print(f"epochs: {epochs}")
    print(f"diff th0: {theta0 - th0_before} | diff th1: {theta1 - th1_before}")
    th0_before = theta0
    th1_before = theta1
