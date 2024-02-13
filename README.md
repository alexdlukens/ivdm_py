# IVDM for Python

Based on paper [Improved Heterogeneous Distance Functions](https://arxiv.org/abs/cs/9701101) by D. Randall Wilson and Tony R. Martinez

Implements a Scikit-Learn interface for utilizing the IVDM (Interpolated Value Distance Metric) algorithm in Python. Calling fit() on the Estimator computes the minimum and maximum values for each feature in the training set, and sets up the conditional probability lookup table
as computed via the [VDM3 library](https://github.com/esmondhkchu)

## Example Usage

```python
from ivdm_py import InterpolatedValueDistanceMetric
from sklearn.datasets import load_iris

ivdm_metric = InterpolatedValueDistanceMetric(s=10, n_neighbors=5, norm=2)

X, y = load_iris(return_X_y=True)

ivdm_metric.fit(X[:100], y[:100])

predictions = ivdm_metric.predict(X[100:])


```
