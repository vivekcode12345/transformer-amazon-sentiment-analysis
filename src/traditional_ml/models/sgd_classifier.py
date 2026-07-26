"""
Stochastic Gradient Descent (SGD) Classifier for Sentiment Analysis
=====================================================================

This module implements an SGD Classifier for binary sentiment analysis
(Positive/Negative) using TF-IDF features.

WHAT IS SGD CLASSIFIER?
SGD Classifier is a linear model that uses Stochastic Gradient Descent for
optimization. It's extremely efficient for large-scale text classification
and supports multiple loss functions.

WHY USE SGD CLASSIFIER?
- Extremely fast training on large datasets
- Memory efficient (works well with sparse TF-IDF features)
- Supports multiple loss functions (hinge, log, modified_huber, etc.)
- Online learning capability (can update model incrementally)
- Great baseline for text classification
- Scalable to millions of samples

CONFIGURATION:
- loss="hinge": Linear SVM loss function (similar to LinearSVC)
- class_weight="balanced": Handles imbalanced classes
- max_iter=1000: Maximum training iterations
- tol=1e-3: Convergence tolerance

IMPORTANT NOTE ABOUT CONFIDENCE SCORES:
SGDClassifier with loss="hinge" does NOT support predict_proba(). Instead,
we use decision_function() which returns the distance from the decision boundary.
This is a confidence score, NOT a probability. It indicates how far the sample
is from the boundary, but doesn't sum to 1 like probabilities.

REUSES:
- Preprocessing from src.traditional_ml.preprocessing
- Metrics computation from sklearn
"""

from __future__ import annotations

import json  # For saving metrics to JSON file
from pathlib import Path  # For file path handling
from typing import Any, Dict, Tuple  # Type hints for better code documentation

# Add project root to path for imports when running as script
# This allows the script to be run directly: python src/traditional_ml/models/sgd_classifier.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# Import scikit-learn's SGDClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Import our custom preprocessing pipeline
# This REUSES the preprocessing module - no duplication!
from src.traditional_ml.preprocessing import prepare_data_for_training

# Import joblib for saving/loading models
# joblib is better than pickle for scikit-learn models (handles numpy arrays efficiently)
import joblib


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default hyperparameters for SGD Classifier
# Configured for binary text classification with hinge loss (linear SVM)

SGD_CLASSIFIER_CONFIG = {
    "loss": "hinge",  # Loss function (hinge = linear SVM)
    "penalty": "l2",  # Regularization penalty
    "alpha": 0.0001,  # Regularization strength
    "class_weight": "balanced",  # Handle imbalanced classes automatically
    "random_state": 42,  # For reproducibility
    "max_iter": 1000,  # Maximum number of iterations
    "tol": 1e-3,  # Convergence tolerance
    "early_stopping": False,  # Disable early stopping for simplicity
}

# Paths for saving model and results
MODEL_OUTPUT_DIR = Path("models/traditional_ml/sgd_classifier")
METRICS_OUTPUT_PATH = Path("reports/sgd_classifier_metrics.json")


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_sgd_classifier(
    sample_limit: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    clean: bool = True,
) -> Tuple[SGDClassifier, Any, Any, Any, Any, Dict[str, Any]]:
    """
    Train an SGD Classifier model for sentiment analysis.
    
    This function:
    1. Loads and preprocesses the data (using our preprocessing pipeline)
    2. Trains an SGD Classifier model
    3. Evaluates the model on test data
    4. Saves the model and metrics
    
    Args:
        sample_limit (int, optional): Limit samples for testing
        test_size (float): Proportion of data for testing (0.2 = 20%)
        random_state (int): Random seed for reproducibility
        clean (bool): Whether to clean text before vectorization
        
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test, metrics)
            - model: Trained SGDClassifier model
            - X_train: TF-IDF features for training
            - X_test: TF-IDF features for testing
            - y_train: Training labels
            - y_test: Test labels
            - metrics: Dictionary with evaluation metrics
    
    Example:
        >>> model, X_train, X_test, y_train, y_test, metrics = train_sgd_classifier()
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
    
    print("\n" + "=" * 70)
    print("SGD CLASSIFIER - TRAINING PIPELINE")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: Prepare data using our preprocessing pipeline
    # =========================================================================
    # This REUSES the preprocessing module - no code duplication!
    print("\n[Step 1/6] Preparing data with TF-IDF features...")
    
    X_train, X_test, y_train, y_test, vectorizer = prepare_data_for_training(
        sample_limit=sample_limit,
        test_size=test_size,
        random_state=random_state,
        clean=clean,
    )
    
    # =========================================================================
    # STEP 2: Initialize the SGD Classifier model
    # =========================================================================
    print("\n[Step 2/6] Initializing SGD Classifier model...")
    print(f"   Configuration: {SGD_CLASSIFIER_CONFIG}")
    
    # Create the model with our hyperparameters
    model = SGDClassifier(**SGD_CLASSIFIER_CONFIG)
    
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Loss function: {SGD_CLASSIFIER_CONFIG['loss']}")
    print(f"   Max iterations: {SGD_CLASSIFIER_CONFIG['max_iter']}")
    print(f"   Tolerance: {SGD_CLASSIFIER_CONFIG['tol']}")
    
    # =========================================================================
    # STEP 3: Train the model
    # =========================================================================
    print("\n[Step 3/6] Training model...")
    print("   This may take a moment...")
    
    # fit() trains the model on the training data
    # The model learns the optimal hyperplane using stochastic gradient descent
    model.fit(X_train, y_train)
    
    print("   ✓ Training complete!")
    
    # =========================================================================
    # STEP 4: Make predictions on test data
    # =========================================================================
    print("\n[Step 4/6] Making predictions on test data...")
    
    # predict() returns the predicted class labels (0 or 1)
    y_pred = model.predict(X_test)
    
    # decision_function() returns the distance from the decision boundary
    # This is a confidence score (NOT a probability)
    # Higher absolute values = more confident predictions
    decision_scores = model.decision_function(X_test)
    
    print(f"   Predicted {len(y_pred)} samples")
    
    # =========================================================================
    # STEP 5: Evaluate the model
    # =========================================================================
    print("\n[Step 5/6] Evaluating model performance...")
    
    # Calculate all the metrics
    metrics = evaluate_sgd_classifier(y_test, y_pred, decision_scores)
    
    # Print metrics to console
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']:.2%})")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']:.2%})")
    print(f"F1 Score:  {metrics['f1']:.4f} ({metrics['f1']:.2%})")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(metrics['classification_report'])
    print("=" * 70)
    
    # =========================================================================
    # STEP 6: Save model and metrics
    # =========================================================================
    print("\n[Step 6/6] Saving model and metrics...")
    
    # Create output directories if they don't exist
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the trained model using joblib
    # We save both the model AND the vectorizer because we need both for prediction
    model_path = MODEL_OUTPUT_DIR / "model.pkl"
    vectorizer_path = MODEL_OUTPUT_DIR / "vectorizer.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"   ✓ Model saved to: {model_path}")
    print(f"   ✓ Vectorizer saved to: {vectorizer_path}")
    
    # Save metrics to JSON file
    with METRICS_OUTPUT_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"   ✓ Metrics saved to: {METRICS_OUTPUT_PATH}")
    
    # =========================================================================
    # Done!
    # =========================================================================
    print("\n" + "=" * 70)
    print("✓ SGD CLASSIFIER TRAINING COMPLETE!")
    print("=" * 70)
    print("\nYou can now:")
    print("1. Load the model: model = joblib.load('models/traditional_ml/sgd_classifier/model.pkl')")
    print("2. Make predictions: model.predict(new_tfidf_features)")
    print("3. View metrics: reports/sgd_classifier_metrics.json")
    print("=" * 70)
    
    # Return everything for further analysis if needed
    return model, X_train, X_test, y_train, y_test, metrics


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_sgd_classifier(
    y_true: Any,
    y_pred: Any,
    decision_scores: Any,
) -> Dict[str, Any]:
    """
    Evaluate SGD Classifier model performance.
    
    This function computes comprehensive evaluation metrics for the model.
    
    Args:
        y_true: True labels (ground truth)
        y_pred: Predicted labels (0 or 1)
        decision_scores: Decision function values (confidence scores)
        
    Returns:
        Dictionary containing all evaluation metrics:
        - accuracy: Overall accuracy
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
        - confusion_matrix: Confusion matrix (2x2)
        - classification_report: Detailed classification report
        - mean_confidence: Average absolute decision score
    
    Example:
        >>> metrics = evaluate_sgd_classifier(y_test, y_pred, decision_scores)
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
    
    print("\nComputing evaluation metrics...")
    
    # =========================================================================
    # Calculate basic metrics
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
    # Calculate confidence statistics
    # =========================================================================
    # decision_function() returns signed distances from the decision boundary
    # Positive values = predicted as positive class
    # Negative values = predicted as negative class
    # Absolute values = confidence (distance from boundary)
    
    mean_confidence = float(abs(decision_scores).mean())
    
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
        "mean_confidence": mean_confidence,  # Average confidence score
        "note": "Confidence scores are from decision_function(), not probabilities"
    }
    
    return metrics


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_sentiment(
    review_text: str,
    model_path: Path | str = MODEL_OUTPUT_DIR / "model.pkl",
    vectorizer_path: Path | str = MODEL_OUTPUT_DIR / "vectorizer.pkl",
) -> Dict[str, Any]:
    """
    Predict sentiment for a single review text.
    
    This function loads the saved model and vectorizer, then makes a prediction.
    
    IMPORTANT: SGDClassifier with loss="hinge" does not support predict_proba().
    Instead, we use decision_function() which returns a confidence score (distance
    from the decision boundary). This is NOT a probability, but indicates how
    confident the model is in its prediction.
    
    Args:
        review_text (str): The review text to analyze
        model_path (Path): Path to the saved model
        vectorizer_path (Path): Path to the saved vectorizer
        
    Returns:
        Dictionary with prediction results:
        - text: Original review text
        - label: Predicted sentiment (Positive/Negative)
        - confidence: Confidence score (distance from decision boundary)
        - decision_score: Raw decision function value
        - probabilities: Dictionary with estimated probabilities (NOT from model)
    
    Example:
        >>> result = predict_sentiment("This product is amazing!")
        >>> print(result['label'])  # Positive
        >>> print(result['confidence'])  # 2.45 (confidence score, not probability)
    """
    
    # =========================================================================
    # Load the saved model and vectorizer
    # =========================================================================
    
    print("Loading model and vectorizer...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # =========================================================================
    # Vectorize the input text
    # =========================================================================
    # The vectorizer converts text to TF-IDF features (same as training)
    
    print("Vectorizing input text...")
    text_tfidf = vectorizer.transform([review_text])
    
    # =========================================================================
    # Make prediction
    # =========================================================================
    
    print("Making prediction...")
    
    # Get predicted class (0 or 1)
    predicted_label_id = model.predict(text_tfidf)[0]
    
    # Get decision function value (confidence score)
    # This is the distance from the decision boundary
    # Higher absolute values = more confident predictions
    decision_score = model.decision_function(text_tfidf)[0]
    
    # Convert to human-readable format
    predicted_label = "Positive" if predicted_label_id == 1 else "Negative"
    
    # Convert decision score to a confidence-like value (0-1 range)
    # We use sigmoid to map the decision score to (0, 1) range
    # This is NOT a true probability, but gives a comparable confidence metric
    import numpy as np
    confidence = float(1 / (1 + np.exp(-abs(decision_score))))
    
    # =========================================================================
    # Return results
    # =========================================================================
    
    result = {
        "text": review_text,
        "label": predicted_label,
        "label_id": int(predicted_label_id),
        "confidence": confidence,  # Confidence score (0-1, NOT probability)
        "decision_score": float(decision_score),  # Raw decision function value
        "probabilities": {
            "Negative": float(1 - confidence) if predicted_label_id == 0 else float(1 - confidence),
            "Positive": float(confidence) if predicted_label_id == 1 else float(1 - confidence),
        },
        "note": "Confidence is estimated from decision_function(), not true probability"
    }
    
    return result


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Train SGD Classifier model and make predictions.
    
    This will:
    1. Load and preprocess the data
    2. Train an SGD Classifier model
    3. Evaluate the model
    4. Save the model and metrics
    5. Make a sample prediction
    """
    
    print("\n" + "=" * 70)
    print("SGD CLASSIFIER - COMPLETE EXAMPLE")
    print("=" * 70)
    
    # Use a small sample for testing
    # Change to None for full dataset training (e.g., 50,000 samples)
    SAMPLE_LIMIT = 1000
    
    # Train the model
    model, X_train, X_test, y_train, y_test, metrics = train_sgd_classifier(
        sample_limit=SAMPLE_LIMIT,
        test_size=0.2,
        clean=True,
    )
    
    # Make a sample prediction
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTION")
    print("=" * 70)
    
    sample_reviews = [
        "This product exceeded my expectations! Amazing quality and fast shipping.",
        "Terrible product. Broke after one week. Complete waste of money.",
        "It's okay, does what it's supposed to do but nothing special.",
    ]
    
    for review in sample_reviews:
        print(f"\nReview: {review}")
        result = predict_sentiment(review)
        print(f"Prediction: {result['label']} (Confidence: {result['confidence']:.2%})")
        print(f"  Note: {result['note']}")
    
    print("\n" + "=" * 70)
    print("✓ Example complete!")
    print("=" * 70)