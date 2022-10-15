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
