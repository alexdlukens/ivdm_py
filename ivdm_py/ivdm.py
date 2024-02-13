import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ivdm_py.components import (
    configure_feature_windows,
    get_feature_ranges,
    clip_instances_to_feature_windows,
)


class InterpolatedValueDistanceMetric(BaseEstimator, ClassifierMixin):
    def __init__(self, s: int) -> None:
        self.s = s
        self.feature_ranges = None
        pass

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        self.feature_ranges = get_feature_ranges(X)
        self.feature_windows = configure_feature_windows(
            feature_ranges=self.feature_ranges, s=self.s
        )

        self.X_transformed = clip_instances_to_feature_windows(
            X=X,
            feature_windows=self.feature_windows,
            feature_ranges=self.feature_ranges,
        )

        # now ready to perform VDM algorithm on data

        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)

        X = check_array(X)

        X = clip_instances_to_feature_windows(
            X=X,
            feature_windows=self.feature_windows,
            feature_ranges=self.feature_ranges,
        )

        # now query VDM

        return None
