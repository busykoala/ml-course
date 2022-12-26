import numpy as np
from scipy.stats import norm


def anomaly_detection(X, threshold):
    # calculate mean and standard deviation of each feature
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    # calculate probability of each sample under the Gaussian distribution
    probabilities = norm.pdf(X, mu, sigma)
    # classify samples as anomalies if probability is below the threshold
    anomalies = probabilities < threshold
    return anomalies


X = np.random.normal(size=(100, 3))
threshold = 0.01
anomalies = anomaly_detection(X, threshold)

anomaly_indices = np.where(np.any(anomalies, axis=1))[0]
X_anomalies = X[anomaly_indices]
print(X_anomalies)
