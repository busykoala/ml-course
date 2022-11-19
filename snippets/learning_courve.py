import matplotlib
matplotlib.use('Qt5Agg')

from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(0)
X, y = load_iris(return_X_y=True)
indices = np.arange(y.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]

train_sizes = np.linspace(0.1, 1, 20)
train_sizes, train_scores, valid_scores = learning_curve(
    Ridge(), X, y,
    train_sizes=train_sizes,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)


plt.title("Learning Curve with Ridge")
plt.xlabel("Training Examples")
plt.ylabel("Score")
lw = 2
plt.semilogx(
    train_sizes, train_scores_mean,
    label="Training score",
    color="darkorange",
    lw=lw
)
plt.fill_between(
    train_sizes,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)
plt.semilogx(
    train_sizes,
    valid_scores_mean,
    label="Cross-validation score",
    color="navy",
    lw=lw
)
plt.fill_between(
    train_sizes,
    valid_scores_mean - valid_scores_std,
    valid_scores_mean + valid_scores_std,
    alpha=0.2,
    color="navy",
    lw=lw,
)
plt.legend(loc="best")
plt.show()
