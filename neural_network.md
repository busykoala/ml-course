# Neural Network

## Practical Example

[This snippet](./snippets/neural_network_classification.py) shows an example
for classification (iris dataset) with a multi-layer perceptron classifier.

In the example there are three hidden layers and each contains five neurons.

The theory is tricky to do in a markdown but consists of forward and
backpropagation. On forward propagation a chosen theta is multiplied with the
previous neuron for each connection. On backward propagation the delta between
the expected values are considered.
