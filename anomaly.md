# Anomaly Detection

## Gaussian Distributed

The anomaly detection function works by fitting a Gaussian distribution to the
data in the array X, and then calculating the probability of each sample under
the distribution.
Samples with a low probability are classified as anomalies, based on the
threshold threshold passed as an argument to the function.

This approach to anomaly detection assumes that the data in X follows a
Gaussian distribution, and uses the mean and standard deviation of the features
to fit the distribution to the data. It then calculates the probability of
each sample under the distribution, and classifies samples as anomalies if
their probability is below the threshold. This allows you to identify unusual
or unexpected samples in the data.

Checkout [this script](./snippets/anomaly_detection.py) as a reference. There
is also an example with OneClassSVM (sklearn).
