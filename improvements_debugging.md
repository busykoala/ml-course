# Improvements & Debugging

## Possible Improvements

- More training examples
- Smaller sets of features
- Additional features
- Polynomial features
- Decreasing lambda
- Increasing lambda

## Choose Model

If a model is choosen based on the accuracy of different approaches there
should be a cross validation set additionally to the training and test set.
A typical split is 40% training data, 20% training set and 20% CV set.
The highest accuracy is then calculated based on the CV set and the accuracy
calculated based on the chosen model trained with the training data and
accuracy then measured using the test data.

## Bias/Variance

High bias (underfitting): both J_train and J_CV will be high.Also, J_CV≈J_train.

High variance (overfitting): J_train will be low and J_CV will be much greater
than J_train.

## Regularization

Checkout [this script](./snippets/regularization.py)
for a practical implementation of regularization.

## Validation & Learning Curve

To look at the validation and learning curve checkout:
- [Learning Curve](./snippets/learning_courve.py)
- [Validation Curve](./snippets/validation_curve.py)
