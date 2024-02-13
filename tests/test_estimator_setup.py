
import numpy as np
from ivdm_py import InterpolatedValueDistanceMetric


def test_setup():
    
    ivdm_metric = InterpolatedValueDistanceMetric(s=5)
    
    test_data = np.zeros((3,3))
    test_data[0] = [1, 2, 3]
    test_data[1] = [4, 2, 4]
    test_data[2] = [0, 2, 100]
    
    labels = np.array([1, 2, 3])
    
    ivdm_metric.fit(X=test_data, y=labels)
    
    assert np.allclose(ivdm_metric.feature_ranges, np.array([[0, 4], [2,2], [3,100]]))
    