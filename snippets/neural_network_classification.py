from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


iris = load_iris()
X = pd.DataFrame(iris.data)
y = pd.DataFrame(iris.target)


def main(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    # standardize train/test data
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # initialize a multilayer perceptron classifier
    # with 3 hidden layers with each 5 neurons
    classifier = MLPClassifier(
        solver="lbfgs",
        alpha=1e-5,
        hidden_layer_sizes=(5 for _ in range(3)),
        random_state=1,
    )
    # fit and predict
    model = classifier.fit(X_train, np.ravel(y_train))
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(np.ravel(y_test), y_pred)
    print(accuracy)


for _ in range(10):
    main(X, y)
