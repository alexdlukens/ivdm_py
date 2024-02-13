import numpy as np
from ivdm_py import InterpolatedValueDistanceMetric


def test_setup():
    ivdm_metric = InterpolatedValueDistanceMetric(s=5)

    test_data = np.zeros((3, 3))
    test_data[0] = [1, 2, 3]
    test_data[1] = [4, 2, 4]
    test_data[2] = [0, 2, 100]

    labels = np.array([1, 2, 3])

    ivdm_metric.fit(X=test_data, y=labels)

    assert np.allclose(ivdm_metric.feature_ranges, np.array([[0, 4], [2, 2], [3, 100]]))

    # check computed feature spaces to ensure
    # they are compliant with min and max values
    for feature in range(test_data.shape[0]):
        assert np.min(ivdm_metric.feature_spaces[feature]) == np.min(
            ivdm_metric.feature_ranges[feature]
        )
        assert np.max(ivdm_metric.feature_spaces[feature]) == np.max(
            ivdm_metric.feature_ranges[feature]
        )

        assert len(ivdm_metric.feature_spaces[feature]) == ivdm_metric.s
