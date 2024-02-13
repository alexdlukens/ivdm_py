import numpy as np


def get_feature_ranges(X: np.ndarray) -> np.ndarray:
    """Get the minimum and maximum values from the training set for each feature

    Args:
        X (np.ndarray): Training set of instances (2d)

    Returns:
        np.ndarray: 2d array of ranges. First dimension corresponds to each feature. 2nd dimension contains min and max values for the range
    """
    feature_count = X.shape[0]

    feature_ranges = np.ndarray(shape=(feature_count, 2), dtype=X.dtype)

    feature_ranges[:, 0] = np.min(X, axis=0)
    feature_ranges[:, 1] = np.max(X, axis=0)

    return feature_ranges


def configure_feature_windows(feature_ranges: np.ndarray, s: int):
    """Determine the regions for the feature ranges based on the configured number of intervals

    Args:
        feature_ranges (np.ndarray): Feature ranges min & max, previously computed
        s (int): Where s is the number of intervals to divide the feature range into
    """

    feature_count = feature_ranges.shape[0]

    feature_windows = []
    for i in range(feature_count):
        feature_space = np.linspace(
            start=feature_ranges[i][0], stop=feature_ranges[i][1], num=s
        )
        feature_windows.append(feature_space)

    return feature_windows
