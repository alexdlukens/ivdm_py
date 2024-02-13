from functools import partial

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from vdm3.components import get_conditional_proba_nd, get_delta_nd

from ivdm_py.components import (
    clip_instances_to_feature_windows,
    configure_feature_windows,
    get_feature_ranges,
)


class InterpolatedValueDistanceMetric(BaseEstimator, ClassifierMixin):
    def __init__(self, s: int, n_neighbors: int, norm: int) -> None:
        self.s = s
        self.feature_ranges = None
        self.feature_windows = None
        self.cond_proba = None
        self.norm = norm
        self.n_neighbors = n_neighbors
        self.knn_ = None

    def get_distance(self, ins_1, ins_2):
        distance_metric = partial(get_delta_nd, self.cond_proba, norm=self.norm)

        distance_metric(ins_1=ins_1, ins_2=ins_2).sum() ** (1 / self.norm)

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
        self.cond_proba = get_conditional_proba_nd(self.X_transformed, y)

        self.knn_ = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            algorithm="brute",
            metric=self.get_distance,
            metric_params=None,
            p=self.norm,
            n_jobs=None,
            weights="uniform",
        )
        self.knn_.fit(X=self.X_transformed, y=y)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)

        X = check_array(X)

        X = clip_instances_to_feature_windows(
            X=X,
            feature_windows=self.feature_windows,
            feature_ranges=self.feature_ranges,
        )
        print(f"in predict")
        print(X)
        # now query VDM

        return self.knn_.predict(X)
