# Linear Regression

## Example:

Estimate the price `y` of an appartment of size `x` based on a dataset (with
`m` datapoints) of appartment sizes `X` and prices `Y` assuming they
correlation in a linear manner.

## Approach

The goal is to find the linear `hypothesis` function `h(x) = a + b*x`
so that we can input a unknown size `x` and we get an estimate price `y`.

### Square Error Cost Function

Minimizing the cost function `J = (1/2*m) * sum([(h(x)-y)**2 for x,y in zip(X,Y)])` gets a pretty
decent result for linear regression problems. In the function the term `1/2*m`
makes it easier to minimize and has no inpact on finding the minimum.

### Gradient Descent

One way to minimize the constfunction `J` is applying gradient descent.
By repeating calculating the partial derivatives using updated `a` and `b` 
each time we can find decent values for `a` and `b` after a few repetitions.

-> See [single feature script](./snippets/linear_regression_single_feature.py) as a reference.

-> See [multi feature script](./snippets/linear_regression_multi_feature.py) as a reference.

### Code Example Linear Regression

```python
def linear_regression(X, y, alpha=0.6, epochs=100000):
    # add additional row of ones (theta_0 * 1)
    X = np.concatenate((np.ones([X.shape[0],1], X.dtype), X), 1)
    m = len(X)
    n = len(X[0])
    thetas = np.zeros(n)
    for _ in range(epochs):
        hypothesis = np.dot(X, thetas)
        cost = hypothesis - y
        thetas = thetas - alpha / m * np.sum((cost * X.T), axis=1)
    predict = lambda x: thetas[0] + sum([th * x for th, x in zip(thetas[1:], x)])
    return predict
```

### Learning Rate

The snippet [learning rate](./snippets/learning_rate.py) contains a simple
approach to find a good learning rate with little computation.
