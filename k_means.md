# K means

The idea is to choose k randomly placed centers for the cluster and then repeat
these two steps to find better places for these centers:
- assign each x to the center it is closest to
- move each center into the middle of the newly assigned x's


## Elbow courve

A way to possibly determine how many clusters should be chosen is to look at
the elbow courve.
Checkout the example [in here](./snippets/k_means.py) which also shows a
snippet of a k-mean implementation.
