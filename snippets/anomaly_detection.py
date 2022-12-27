from scipy.stats import multivariate_normal
from scipy.stats import norm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
import numpy as np
import numpy as np


def gaussian(X, y, threshold):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    probabilities = norm.pdf(X_test, mu, sigma)
    y_pred = np.max((probabilities < threshold).astype(int), axis=1)
    # replace 1 with -1 and 0 with one according to the given y
    y_pred = np.where(y_pred == 1, -1, 1)
    print("Gaussian:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)


def multivariate_gaussian(X, y, threshold):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    mu = np.mean(X_train, axis=0)
    sigma = np.cov(X_train.T)
    probabilities = multivariate_normal.pdf(X_test, mean=mu, cov=sigma)
    y_pred = (probabilities < threshold).astype(int)
    # replace 1 with -1 and 0 with one according to the given y
    y_pred = np.where(y_pred == 1, -1, 1)
    print("Multivariate Gaussian:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)


def one_class_svm_example(X, y, threshold):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = OneClassSVM(kernel='rbf', nu=threshold)
    model.fit(X_train)
    y_pred = model.predict(X_test)
    print("OneClassSVM:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)


if __name__ == '__main__':
    anomaly_size = 100
    X_size = (10000, 3)
    X = np.random.normal(size=X_size)
    X[:anomaly_size] = np.random.uniform(low=-5, high=5, size=(anomaly_size, X_size[1]))
    y = np.ones(X.shape[0])
    y[:anomaly_size] = -1
    threshold = 0.01

    gaussian(X, y, threshold)
    multivariate_gaussian(X, y, threshold)
    one_class_svm_example(X, y, threshold)
