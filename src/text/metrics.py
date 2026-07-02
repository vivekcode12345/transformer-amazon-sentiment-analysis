"""Metrics computation for text-only sentiment analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def _to_label_ids(predictions: np.ndarray) -> np.ndarray:
    """Convert model outputs (logits) into hard class labels."""
    if np.asarray(predictions).ndim == 1:
        return np.asarray(predictions).astype(int)
    return np.argmax(np.asarray(predictions), axis=-1)


def compute_basic_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute core binary classification metrics using scikit-learn."""
    pred_ids = _to_label_ids(predictions)
    label_ids = np.asarray(labels)

    accuracy = accuracy_score(label_ids, pred_ids)
    precision = precision_score(label_ids, pred_ids, zero_division=0)
    recall = recall_score(label_ids, pred_ids, zero_division=0)
    f1 = f1_score(label_ids, pred_ids, zero_division=0)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_trainer_metrics(eval_pred: Any) -> dict[str, float]:
    """Compute metrics in the format expected by Hugging Face Trainer."""
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    return compute_basic_metrics(predictions=np.asarray(predictions), labels=np.asarray(labels))


def compute_detailed_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: tuple[str, str] = ("negative", "positive"),
) -> dict[str, Any]:
    """Compute research-grade diagnostics from predictions and labels."""
    pred_ids = _to_label_ids(np.asarray(predictions))
    label_ids = np.asarray(labels)

    cm = confusion_matrix(label_ids, pred_ids)
    report_dict = classification_report(
        label_ids,
        pred_ids,
        target_names=list(label_names),
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        label_ids,
        pred_ids,
        target_names=list(label_names),
        output_dict=False,
        zero_division=0,
    )

    metrics = compute_basic_metrics(predictions=pred_ids, labels=label_ids)

    return {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "classification_report_text": report_text,
    }


def print_detailed_metrics(detailed_metrics: dict[str, Any]) -> None:
    """Pretty-print detailed metric outputs for terminal usage."""
    print("Evaluation Metrics")
    print("-" * 60)
    print(f"Accuracy : {detailed_metrics['accuracy']:.4f}")
    print(f"Precision: {detailed_metrics['precision']:.4f}")
    print(f"Recall   : {detailed_metrics['recall']:.4f}")
    print(f"F1 Score : {detailed_metrics['f1']:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(np.asarray(detailed_metrics["confusion_matrix"]))
    print("\nClassification Report:")
    print(detailed_metrics["classification_report_text"])
