from ivdm_py import InterpolatedValueDistanceMetric
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score


def test_iris():
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    ivdm_metric = InterpolatedValueDistanceMetric(s=10, n_neighbors=2, norm=2)

    ivdm_metric.fit(X=X_train, y=y_train)

    predictions = ivdm_metric.predict(X=X_test)

    accuracy = accuracy_score(y_true=y_test, y_pred=predictions)
    precision = precision_score(y_true=y_test, y_pred=predictions, average="micro")

    assert accuracy == 0.98
    assert precision == 0.98
