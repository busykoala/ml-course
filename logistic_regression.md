# Logistic Regression

## Example:

Get the probability of a cancer being malignant or not by the size of the
cancer.

## Approach

The data is fit into a range of `0 <= h(x) <=1` using the sigmoid function `g`
for the transpose of theta x.

```
h(x) = g(theta^T x)
z = theta^T x
g(z) = 1/(1 + e^(-z))
```

### Decision Boundry

![diagonal boundry](./assets/diagonal.png)
![circular boundry](./assets/circular.png)
