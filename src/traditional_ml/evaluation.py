"""
Unified Evaluation Module for Traditional ML Models
====================================================

This module provides a single, reusable evaluation function for all traditional
ML models in the pipeline. It eliminates code duplication by centralizing all
evaluation logic in one place.

WHY THIS MODULE?
Previously, each model (logistic_regression.py, linear_svm.py, etc.) had its
own evaluate_*() function with duplicated code. This module consolidates all
evaluation logic into one reusable function, following the DRY (Don't Repeat
Yourself) principle.

WHAT DOES IT DO?
- Computes standard classification metrics (accuracy, precision, recall, F1)
- Generates confusion matrix
- Creates classification reports (both dict and text formats)
- Optionally computes confidence statistics from decision scores
- Returns all metrics in a standardized dictionary format

REUSES:
- Metrics computation from sklearn
- Used by all traditional ML models
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    y_true: Any,
    y_pred: Any,
    y_pred_proba: Optional[Any] = None,
    decision_scores: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Evaluate a classification model and return comprehensive metrics.
    
    This is the main evaluation function used by all traditional ML models.
    It computes standard classification metrics and optionally includes
    confidence statistics if decision scores are provided.
    
    Args:
        y_true: True labels (ground truth) - array-like of shape (n_samples,)
        y_pred: Predicted labels (0 or 1) - array-like of shape (n_samples,)
        y_pred_proba: Predicted probabilities (optional) - array-like of shape (n_samples, 2)
            Not used in current implementation but kept for future extensibility
        decision_scores: Decision function values (optional) - array-like of shape (n_samples,)
            If provided, computes mean_confidence metric
            
    Returns:
        Dictionary containing all evaluation metrics:
        - accuracy: Overall accuracy (float)
        - precision: Precision score (float)
        - recall: Recall score (float)
        - f1: F1 score (float)
        - confusion_matrix: Confusion matrix as list (JSON serializable)
        - classification_report: Classification report as dict (JSON serializable)
        - classification_report_text: Classification report as formatted string
        - mean_confidence: (optional) Average absolute decision score, if decision_scores provided
    
    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.metrics import accuracy_score
        >>> 
        >>> # Train a model
        >>> model = LogisticRegression()
        >>> model.fit(X_train, y_train)
        >>> 
        >>> # Make predictions
        >>> y_pred = model.predict(X_test)
        >>> y_pred_proba = model.predict_proba(X_test)
        >>> decision_scores = model.decision_function(X_test)
        >>> 
        >>> # Evaluate
        >>> metrics = evaluate_model(y_test, y_pred, y_pred_proba, decision_scores)
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
        >>> print(f"F1 Score: {metrics['f1']:.2%}")
    """
    
    # =========================================================================
    # Calculate basic classification metrics
    # =========================================================================
    
    # Accuracy: Percentage of correct predictions
    # "How often is the model correct?"
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision: Of all positive predictions, how many were actually positive?
    # "When the model says positive, how often is it right?"
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    # Recall: Of all actual positives, how many did we find?
    # "Of all positive reviews, how many did we correctly identify?"
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # F1 Score: Harmonic mean of precision and recall
    # "Balanced measure of model performance"
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # =========================================================================
    # Calculate confusion matrix
    # =========================================================================
    # Confusion Matrix shows:
    # - True Negatives (TN): Correctly predicted negative
    # - False Positives (FP): Negative predicted as positive
    # - False Negatives (FN): Positive predicted as negative
    # - True Positives (TP): Correctly predicted positive
    cm = confusion_matrix(y_true, y_pred)
    
    # =========================================================================
    # Generate classification report
    # =========================================================================
    # This provides precision, recall, F1 for each class
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Negative", "Positive"],
        output_dict=True,
        zero_division=0,
    )
    
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=["Negative", "Positive"],
        output_dict=False,
        zero_division=0,
    )
    
    # =========================================================================
    # Package all metrics into a dictionary
    # =========================================================================
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),  # Convert to list for JSON serialization
        "classification_report": report,
        "classification_report_text": report_text,
    }
    
    # =========================================================================
    # Add confidence statistics if decision scores are provided
    # =========================================================================
    # decision_function() returns signed distances from the decision boundary
    # Positive values = predicted as positive class
    # Negative values = predicted as negative class
    # Absolute values = confidence (distance from boundary)
    
    if decision_scores is not None:
        mean_confidence = float(np.abs(decision_scores).mean())
        metrics["mean_confidence"] = mean_confidence
        metrics["note"] = "Confidence scores are from decision_function(), not probabilities"
    
    return metrics


def print_evaluation_results(metrics: Dict[str, Any]) -> None:
    """
    Pretty-print evaluation metrics to console.
    
    This function provides a standardized way to display evaluation results
    across all models.
    
    Args:
        metrics: Dictionary of metrics returned by evaluate_model()
    
    Example:
        >>> metrics = evaluate_model(y_test, y_pred)
        >>> print_evaluation_results(metrics)
    """
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']:.2%})")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']:.2%})")
    print(f"F1 Score:  {metrics['f1']:.4f} ({metrics['f1']:.2%})")
    
    if "mean_confidence" in metrics:
        print(f"Mean Confidence: {metrics['mean_confidence']:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(metrics['classification_report_text'])
    print("=" * 70)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Demonstrate the unified evaluation function.
    
    This shows how to use evaluate_model() with different types of classifiers.
    """
    
    print("\n" + "=" * 70)
    print("UNIFIED EVALUATION MODULE - EXAMPLE")
    print("=" * 70)
    
    # Example 1: Model with predict_proba (e.g., Logistic Regression)
    print("\n[Example 1] Model with predict_proba()")
    print("-" * 70)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_pred_proba=y_pred_proba)
    print_evaluation_results(metrics)
    
    # Example 2: Model with decision_function (e.g., LinearSVC)
    print("\n[Example 2] Model with decision_function()")
    print("-" * 70)
    
    from sklearn.svm import LinearSVC
    
    # Train model
    model = LinearSVC()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    decision_scores = model.decision_function(X_test)
    
    # Evaluate with decision scores
    metrics = evaluate_model(y_test, y_pred, decision_scores=decision_scores)
    print_evaluation_results(metrics)
    
    # Example 3: Model with only predictions (no probabilities or decision scores)
    print("\n[Example 3] Model with only predictions")
    print("-" * 70)
    
    from sklearn.tree import DecisionTreeClassifier
    
    # Train model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate without probabilities or decision scores
    metrics = evaluate_model(y_test, y_pred)
    print_evaluation_results(metrics)
    
    print("\n" + "=" * 70)
    print("✓ Example complete!")
    print("=" * 70)