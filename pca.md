# PCA (principle component analysis)

Principal component analysis (PCA) is a technique for reducing the
dimensionality of a dataset by projecting it onto a lower-dimensional space.
It is commonly used to visualize high-dimensional datasets, or to reduce the
complexity of the data for machine learning tasks.

In machine learning, PCA is often used as a preprocessing step to improve the
performance of algorithms that are sensitive to the scale and variance of the
features in the dataset. By reducing the dimensionality of the data, PCA can
also help to reduce the computational cost of training a model, and can improve
the interpretability of the results.

PCA works by finding the directions in the data that have the highest variance,
and projecting the data onto these directions. The resulting projections are
the principal components of the data, and they capture the most important
information in the dataset.

## Implementation

To perform PCA on a dataset, you need to first center the data by subtracting
the mean of each feature (mean normalization).
Then, you can calculate the covariance matrix (sigma) of the data, and compute
the eigenvalues and eigenvectors of the matrix.
The eigenvectors are the principal components of the data, and the eigenvalues
are the amount of variance explained by each component.

You can then select the number of principal components to keep, based on the
desired level of dimensionality reduction. Finally, you can project the data
onto the selected principal components to obtain the transformed dataset.

There is a sklearn and numpy implementation [in this snippet](./snippets/pca.py).

Also there is an example of finding the optimal k-value.
