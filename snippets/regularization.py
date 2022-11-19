# This follows the blog entry here:
# https://medium.com/coinmonks/regularization-of-linear-models-with-sklearn-f88633a93a2
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv('./assets/boston_housing.csv')

X = train_df.drop(["MEDV", "TOWN"], axis=1)
y = train_df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.3)

# linear regression without any scaling or regularization
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("*" * 50)
print("Linear regression without scaling or regularization")
print('Training score: {}'.format(lr_model.score(X_train, y_train)))
print('Test score: {}'.format(lr_model.score(X_test, y_test)))

# linear regression with feature scaling and polinomial features
pipeline = Pipeline([
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', LinearRegression())
])
pipeline.fit(X_train, y_train)
print("*" * 50)
print("linear regression with feature scaling and polinomial features")
print('Training score: {}'.format(pipeline.score(X_train, y_train)))
print('Test score: {}'.format(pipeline.score(X_test, y_test)))

# ridge adds penalty for higher polinomials to avoid overfitting
ridge_pipe = Pipeline([
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha=10, fit_intercept=True))
])
ridge_pipe.fit(X_train, y_train)
print("*" * 50)
print("ridge adds penalty for higher polinomials to avoid overfitting")
print('Training Score: {}'.format(ridge_pipe.score(X_train, y_train)))
print('Test Score: {}'.format(ridge_pipe.score(X_test, y_test)))

# Lasso helps eliminating irrelevant features driving down the
# coeffitients to zero (alpha 0.1-1)
lasso_pipe = Pipeline([
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Lasso(alpha=0.3, fit_intercept=True))
])
lasso_pipe.fit(X_train, y_train)
print("*" * 50)
print("lasso helps eliminating irrelevant fearures driving down coeff")
print('Training score: {}'.format(lasso_pipe.score(X_train, y_train)))
print('Test score: {}'.format(lasso_pipe.score(X_test, y_test)))
