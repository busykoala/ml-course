# Support Vector Machines

## Bias & Variance

**Term C:**
- Large C: lower bias, higher variance
- Small C: higher bias, lower variance

**Term sigma:**
- Large sigma: higher bias, lower variance
- Small sigma: lower bias, higher variance


## Kernels

In the context of SVMs, a kernel is used to map the original data points into
a higher-dimensional space, in which it becomes possible to find a linear
decision boundary that separates the different classes. The kernel function is
used to calculate the inner product between pairs of data points in the
higher-dimensional space, which can then be used to train the SVM.


## Snippet

There is a [SVM snippet here](./snippets/svm.py)
