"""Feature-space adversarial perturbations for controlled experiments."""
from __future__ import annotations

import numpy as np


def pgd_linf(
    classifier,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.05,
    step_size: float = 0.01,
    steps: int = 5,
) -> np.ndarray:
    """Create an L-infinity PGD-style perturbation for SGDClassifier(log_loss).

    Inputs are expected to be in the same standardized feature space used by
    the classifier. This is an experimental approximation; perturbed vectors
    are not guaranteed to map to valid or physically realizable network flows.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    if epsilon < 0 or step_size <= 0 or steps < 1:
        raise ValueError("epsilon must be >= 0, step_size > 0, and steps >= 1.")
    if not hasattr(classifier, "coef_") or not hasattr(classifier, "predict_proba"):
        raise ValueError("Classifier must be a fitted SGDClassifier using loss='log_loss'.")

    original = X.copy()
    adversarial = original.copy()
    classes = classifier.classes_
    class_to_col = {int(label): i for i, label in enumerate(classes)}
    if any(int(label) not in class_to_col for label in y):
        raise ValueError("y contains a class not known to the classifier.")

    for _ in range(steps):
        probabilities = classifier.predict_proba(adversarial)
        one_hot = np.zeros_like(probabilities)
        rows = np.arange(len(y))
        one_hot[rows, [class_to_col[int(label)] for label in y]] = 1.0
        # Gradient of log loss with respect to each input vector.
        # sklearn stores a single coefficient row for binary classification,
        # but one coefficient row per class for multiclass classification.
        if classifier.coef_.shape[0] == 1 and probabilities.shape[1] == 2:
            positive_class = classes[1]
            target_positive = (y == positive_class).astype(float)
            gradient = (
                (probabilities[:, 1] - target_positive)[:, None]
                * classifier.coef_[0][None, :]
            )
        else:
            gradient = (probabilities - one_hot) @ classifier.coef_
        adversarial = adversarial + step_size * np.sign(gradient)
        adversarial = np.minimum(np.maximum(adversarial, original - epsilon), original + epsilon)

    return adversarial
