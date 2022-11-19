import matplotlib
matplotlib.use('Qt5Agg')

from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(0)
X, y = load_iris(return_X_y=True)
indices = np.arange(y.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]

param_range = np.logspace(-7, 3, 20)
train_scores, test_scores = validation_curve(
    Ridge(), X, y, param_name="alpha",
    param_range=param_range,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with Ridge")
plt.xlabel(r"$\alpha$")
plt.ylabel("Score")
lw = 2
plt.semilogx(
    param_range, train_scores_mean,
    label="Training score",
    color="darkorange",
    lw=lw
)
plt.fill_between(
    param_range,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)
plt.semilogx(
    param_range,
    test_scores_mean,
    label="Cross-validation score",
    color="navy",
    lw=lw
)
plt.fill_between(
    param_range,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=lw,
)
plt.legend(loc="best")
plt.show()
