"""Command-line entry point for Adaptive-IDS."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .attacks import pgd_linf
from .data import load_dataset
from .detector import AdaptiveIDS
from .evaluation import evaluate_anomaly, evaluate_supervised
from .retraining import AsyncRetrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the Adaptive-IDS prototype.")
    parser.add_argument("--dataset", default="data/cicids2017_combined_sampled.csv",
                        help="CSV path with numeric features and a Label column.")
    parser.add_argument("--label-column", default="Label")
    parser.add_argument("--normal-label", default="BENIGN")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--contamination", type=float, default=0.10)
    parser.add_argument("--pgd", action="store_true", help="Run a feature-space PGD-style test.")
    parser.add_argument("--feedback-replay", action="store_true",
                        help="Simulate updates from confirmed benchmark labels (offline only).")
    parser.add_argument("--retrain-batch-size", type=int, default=64)
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--report", default="reports/metrics.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = load_dataset(
        args.dataset,
        label_column=args.label_column,
        normal_label=args.normal_label,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = AdaptiveIDS(
        normal_class_id=data.normal_class_id,
        contamination=args.contamination,
        random_state=args.random_state,
    )
    model.fit(data.X_train, data.y_train)

    baseline = model.predict(data.X_test)
    report = {
        "dataset": str(args.dataset),
        "samples": {
            "train": int(len(data.y_train)),
            "test": int(len(data.y_test)),
            "features": int(data.X_train.shape[1]),
            "classes": [str(x) for x in data.label_encoder.classes_],
        },
        "static_baseline": {
            "supervised": evaluate_supervised(
                data.y_test, baseline["class_prediction"], data.label_encoder
            ),
            "isolation_forest": evaluate_anomaly(
                data.y_test, baseline["is_anomaly"], data.normal_class_id
            ),
        },
    }

    if args.pgd:
        adversarial = pgd_linf(model.sgd, data.X_test, data.y_test)
        adv_predictions = model.predict(adversarial)
        report["pgd_feature_space"] = {
            "epsilon": 0.05,
            "steps": 5,
            "supervised": evaluate_supervised(
                data.y_test, adv_predictions["class_prediction"], data.label_encoder
            ),
            "isolation_forest": evaluate_anomaly(
                data.y_test, adv_predictions["is_anomaly"], data.normal_class_id
            ),
            "note": "Feature-space perturbations; not guaranteed to be valid network flows.",
        }

    if args.feedback_replay:
        # Explicitly an offline simulation: benchmark labels are treated as if
        # they had been independently confirmed after initial predictions.
        retrainer = AsyncRetrainer(
            model,
            batch_size=args.retrain_batch_size,
            queue_size=args.queue_size,
        )
        try:
            # Stream test rows in order, enqueueing their labels as simulated feedback.
            chunk_size = max(1, args.retrain_batch_size)
            for start in range(0, len(data.X_test), chunk_size):
                end = min(start + chunk_size, len(data.X_test))
                retrainer.submit(data.X_test[start:end], data.y_test[start:end])
            # Ensure queued feedback is processed before final evaluation.
            retrainer._queue.join()
        finally:
            retrainer.close(wait=True)

        after = model.predict(data.X_test)
        report["feedback_replay"] = {
            "supervised": evaluate_supervised(
                data.y_test, after["class_prediction"], data.label_encoder
            ),
            "isolation_forest": evaluate_anomaly(
                data.y_test, after["is_anomaly"], data.normal_class_id
            ),
            "retraining": {
                "batches_completed": retrainer.stats.batches_completed,
                "samples_retrained": retrainer.stats.samples_retrained,
                "batches_dropped": retrainer.stats.batches_dropped,
                "last_error": retrainer.stats.last_error,
            },
            "note": (
                "Offline label-feedback simulation. Because labels from the evaluated "
                "test split are reused for updates, these post-update metrics are not an "
                "independent generalization estimate. Use a separate future holdout for "
                "a rigorous adaptive-learning comparison."
            ),
        }

    output_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(output_path),
        "static_supervised_accuracy": report["static_baseline"]["supervised"]["accuracy"],
        "static_supervised_macro_f1": report["static_baseline"]["supervised"]["macro_f1"],
        "static_anomaly_f1": report["static_baseline"]["isolation_forest"]["f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
