# Normal Equation

If the number of features is not too large normal equation can be the better
choice than gradient descent.
There is no need for the epoch iterations and alpha doesn't need to be choosen.
For a high number of features inverting a `n x n` matix is likely to be O(n^3).

## Background

The equation to get the thetas is `thetas = (X^T * X)^(-1) * X^T * y`. For the
same reason the column of ones is added to X this is done here too.

The implementation might look like this:

```python
def normal_equation(X, y):
    X = np.concatenate((np.ones([X.shape[0],1], X.dtype), X), 1)
    thetas = np.linalg.inv(X.T @ X) @ X.T @ y
    return thetas
```

## Examples

An example and comparence to gradient_descent is in
[this script](./snippets/normal_equation.py)
