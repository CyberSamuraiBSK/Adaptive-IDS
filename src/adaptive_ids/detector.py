"""Hybrid supervised and anomaly-based detection models."""
from __future__ import annotations

import threading
from collections import deque

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDClassifier


class AdaptiveIDS:
    """SGD attack classifier plus Isolation Forest anomaly detector."""

    def __init__(
        self,
        normal_class_id: int,
        contamination: float = 0.10,
        random_state: int = 42,
        reference_window: int = 5000,
    ) -> None:
        if not 0.0 < contamination <= 0.5:
            raise ValueError("contamination must be in the interval (0, 0.5].")
        self.normal_class_id = int(normal_class_id)
        self.sgd = SGDClassifier(
            loss="log_loss",
            random_state=random_state,
            class_weight="balanced",
        )
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            n_jobs=-1,
        )
        self._lock = threading.RLock()
        self._classes: np.ndarray | None = None
        self._normal_reference: deque[np.ndarray] = deque(maxlen=reference_window)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaptiveIDS":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2D and have the same number of rows as y.")
        if len(np.unique(y)) < 2:
            raise ValueError("The SGD classifier requires at least two classes.")

        with self._lock:
            self._classes = np.unique(y)
            self.sgd.fit(X, y)
            normal_rows = X[y == self.normal_class_id]
            if len(normal_rows) == 0:
                raise ValueError(
                    "No benign samples found in training data; check --normal-label."
                )
            self.isolation_forest.fit(normal_rows)
            self._normal_reference.clear()
            for row in normal_rows[-self._normal_reference.maxlen:]:
                self._normal_reference.append(row.copy())
            self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return multi-class labels and Isolation Forest's binary anomaly flags."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        with self._lock:
            self._require_fitted()
            class_predictions = self.sgd.predict(X)
            # sklearn: -1 means anomaly; 1 means inlier.
            raw_anomaly = self.isolation_forest.predict(X)
            anomaly = (raw_anomaly == -1).astype(np.int8)
            scores = -self.isolation_forest.score_samples(X)
        return {
            "class_prediction": class_predictions,
            "is_anomaly": anomaly,
            "anomaly_score": scores,
        }

    def partial_retrain(self, X: np.ndarray, y: np.ndarray) -> None:
        """Incrementally update SGD using confirmed labels, then refresh IF baseline."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        if len(X) == 0:
            return
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2D and have the same number of rows as y.")

        with self._lock:
            self._require_fitted()
            self.sgd.partial_fit(X, y, classes=self._classes)
            confirmed_normal = X[y == self.normal_class_id]
            for row in confirmed_normal:
                self._normal_reference.append(row.copy())
            # IsolationForest has no partial_fit. Refit it on a bounded benign
            # reference window rather than fitting it on known attacks.
            if len(self._normal_reference) >= 2:
                reference = np.asarray(self._normal_reference)
                self.isolation_forest.fit(reference)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before prediction or retraining.")
