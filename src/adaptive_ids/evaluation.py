"""Evaluation helpers for supervised and anomaly detection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_supervised(y_true: np.ndarray, y_pred: np.ndarray, label_encoder) -> dict:
    labels = np.arange(len(label_encoder.classes_))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=[str(x) for x in label_encoder.classes_],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def evaluate_anomaly(y_true: np.ndarray, y_pred_anomaly: np.ndarray, normal_class_id: int) -> dict:
    y_true_attack = (np.asarray(y_true) != normal_class_id).astype(int)
    y_pred_attack = np.asarray(y_pred_anomaly).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true_attack, y_pred_attack)),
        "precision": float(precision_score(y_true_attack, y_pred_attack, zero_division=0)),
        "recall": float(recall_score(y_true_attack, y_pred_attack, zero_division=0)),
        "f1": float(f1_score(y_true_attack, y_pred_attack, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_attack, y_pred_attack, labels=[0, 1]).tolist(),
    }
