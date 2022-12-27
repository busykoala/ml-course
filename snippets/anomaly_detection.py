import numpy as np
from scipy.stats import norm
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, classification_report


def anomaly_detection(X, threshold):
    # calculate mean and standard deviation of each feature
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    # calculate probability of each sample under the Gaussian distribution
    probabilities = norm.pdf(X, mu, sigma)
    # classify samples as anomalies if probability is below the threshold
    anomalies = probabilities < threshold
    return anomalies


def one_class_svm_example():
    size = 20000
    X = np.random.normal(size=(size, 3))
    y = np.ones(size)
    # add some anomalies to the dataset
    X[:10] = np.random.uniform(low=-5, high=5, size=(10, 3))
    y[:10] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = OneClassSVM(kernel='rbf', nu=0.1)
    model.fit(X_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


X = np.random.normal(size=(100, 3))
threshold = 0.01
anomalies = anomaly_detection(X, threshold)

anomaly_indices = np.where(np.any(anomalies, axis=1))[0]
X_anomalies = X[anomaly_indices]
print("Gaussian Anomaly Detection - Anomalies Found:")
print(X_anomalies)
print()

print("One Class SVM:")
one_class_svm_example()
