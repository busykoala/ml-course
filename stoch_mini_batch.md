# Stochastic Gradient Descent & Mini Batch Gradient Descent

Stochastic gradient descent (SGD) is an optimization algorithm used to find
the values of parameters (coefficients and biases) in a model that minimize the
loss function. It is a type of gradient descent, which is an iterative method
that moves in the direction of the negative gradient of a loss function with
respect to the model's parameters.

In SGD, the loss function is calculated for a small batch of training data,
rather than the entire training set, at each iteration. This makes SGD
computationally more efficient, especially when the training set is large.
It also introduces randomness into the process, which can help the model escape
from local minima and converge to a better solution.

To perform SGD, the following steps are generally followed:

- Initialize the model's parameters with random values.
- Shuffle the training data.
- Divide the training data into small batches or single items.
- For each batch or item of data:
  - Calculate the loss function for the batch/item.
  - Calculate the gradients of the loss function with respect to the model's parameters.
  - Update the parameters using the gradients and a learning rate.
- Repeat the above steps until the model converges or a pre-defined number of
  iterations is reached.
