"""
Multinomial Naive Bayes Model for Sentiment Analysis
=====================================================

This module implements a Multinomial Naive Bayes classifier for binary sentiment
analysis (Positive/Negative) using TF-IDF features.

WHAT IS NAIVE BAYES?
Naive Bayes is a probabilistic classifier based on Bayes' theorem with an
assumption of feature independence. It's called "naive" because it assumes
that words are independent of each other (which is not true in reality, but
it works surprisingly well!).

WHAT IS MULTINOMIAL NAIVE BAYES?
Multinomial Naive Bayes is specifically designed for discrete features like
word counts or TF-IDF scores. It models the probability of each word appearing
in each class (positive/negative).

WHY MULTINOMIAL NAIVE BAYES?
- Fast to train and predict
- Works exceptionally well with TF-IDF features
- Handles high-dimensional data (thousands of features) efficiently
- Provides probability outputs
- Great baseline model for text classification
- Less prone to overfitting than complex models

REUSES:
- Preprocessing from src.traditional_ml.preprocessing
- Metrics computation from sklearn
"""

from __future__ import annotations

import json  # For saving metrics to JSON file
from pathlib import Path  # For file path handling
from typing import Any, Dict, Tuple  # Type hints for better code documentation

# Add project root to path for imports when running as script
# This allows the script to be run directly: python src/traditional_ml/models/multinomial_naive_bayes.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# Import scikit-learn's Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
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

# Default hyperparameters for Multinomial Naive Bayes
# You can tune these to improve performance

MULTINOMIAL_NAIVE_BAYES_CONFIG = {
    "alpha": 1.0,  # Laplace smoothing parameter (prevents zero probabilities)
    "fit_prior": True,  # Whether to learn class prior probabilities
    "class_prior": None,  # Custom class prior probabilities (None = learn from data)
}

# Paths for saving model and results
MODEL_OUTPUT_DIR = Path("models/traditional_ml/multinomial_naive_bayes")
METRICS_OUTPUT_PATH = Path("reports/multinomial_naive_bayes_metrics.json")


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_multinomial_naive_bayes(
    sample_limit: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    clean: bool = True,
) -> Tuple[MultinomialNB, Any, Any, Any, Any, Dict[str, Any]]:
    """
    Train a Multinomial Naive Bayes model for sentiment analysis.
    
    This function:
    1. Loads and preprocesses the data (using our preprocessing pipeline)
    2. Trains a Multinomial Naive Bayes model
    3. Evaluates the model on test data
    4. Saves the model and metrics
    
    Args:
        sample_limit (int, optional): Limit samples for testing
        test_size (float): Proportion of data for testing (0.2 = 20%)
        random_state (int): Random seed for reproducibility
        clean (bool): Whether to clean text before vectorization
        
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test, metrics)
            - model: Trained MultinomialNB model
            - X_train: TF-IDF features for training
            - X_test: TF-IDF features for testing
            - y_train: Training labels
            - y_test: Test labels
            - metrics: Dictionary with evaluation metrics
    
    Example:
        >>> model, X_train, X_test, y_train, y_test, metrics = train_multinomial_naive_bayes()
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
    
    print("\n" + "=" * 70)
    print("MULTINOMIAL NAIVE BAYES - TRAINING PIPELINE")
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
    # STEP 2: Initialize the Multinomial Naive Bayes model
    # =========================================================================
    print("\n[Step 2/6] Initializing Multinomial Naive Bayes model...")
    print(f"   Configuration: {MULTINOMIAL_NAIVE_BAYES_CONFIG}")
    
    # Create the model with our hyperparameters
    model = MultinomialNB(**MULTINOMIAL_NAIVE_BAYES_CONFIG)
    
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Alpha (smoothing): {MULTINOMIAL_NAIVE_BAYES_CONFIG['alpha']}")
    print(f"   Fit prior: {MULTINOMIAL_NAIVE_BAYES_CONFIG['fit_prior']}")
    
    # =========================================================================
    # STEP 3: Train the model
    # =========================================================================
    print("\n[Step 3/6] Training model...")
    print("   This may take a moment...")
    
    # fit() trains the model on the training data
    # The model learns the probability of each word appearing in each class
    model.fit(X_train, y_train)
    
    print("   ✓ Training complete!")
    
    # =========================================================================
    # STEP 4: Make predictions on test data
    # =========================================================================
    print("\n[Step 4/6] Making predictions on test data...")
    
    # predict() returns the predicted class labels (0 or 1)
    y_pred = model.predict(X_test)
    
    # predict_proba() returns the probability for each class
    # This gives us confidence scores for our predictions
    y_pred_proba = model.predict_proba(X_test)
    
    print(f"   Predicted {len(y_pred)} samples")
    
    # =========================================================================
    # STEP 5: Evaluate the model
    # =========================================================================
    print("\n[Step 5/6] Evaluating model performance...")
    
    # Calculate all the metrics
    metrics = evaluate_multinomial_naive_bayes(y_test, y_pred, y_pred_proba)
    
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
    print("✓ MULTINOMIAL NAIVE BAYES TRAINING COMPLETE!")
    print("=" * 70)
    print("\nYou can now:")
    print("1. Load the model: model = joblib.load('models/traditional_ml/multinomial_naive_bayes/model.pkl')")
    print("2. Make predictions: model.predict(new_tfidf_features)")
    print("3. View metrics: reports/multinomial_naive_bayes_metrics.json")
    print("=" * 70)
    
    # Return everything for further analysis if needed
    return model, X_train, X_test, y_train, y_test, metrics


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_multinomial_naive_bayes(
    y_true: Any,
    y_pred: Any,
    y_pred_proba: Any,
) -> Dict[str, Any]:
    """
    Evaluate Multinomial Naive Bayes model performance.
    
    This function computes comprehensive evaluation metrics for the model.
    
    Args:
        y_true: True labels (ground truth)
        y_pred: Predicted labels (0 or 1)
        y_pred_proba: Predicted probabilities (from predict_proba)
        
    Returns:
        Dictionary containing all evaluation metrics:
        - accuracy: Overall accuracy
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
        - confusion_matrix: Confusion matrix (2x2)
        - classification_report: Detailed classification report
        - confusion_matrix_list: Confusion matrix as list (JSON serializable)
    
    Example:
        >>> metrics = evaluate_multinomial_naive_bayes(y_test, y_pred, y_pred_proba)
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
    
    Args:
        review_text (str): The review text to analyze
        model_path (Path): Path to the saved model
        vectorizer_path (Path): Path to the saved vectorizer
        
    Returns:
        Dictionary with prediction results:
        - text: Original review text
        - label: Predicted sentiment (Positive/Negative)
        - confidence: Confidence score (0-1)
        - probabilities: Dictionary with probabilities for each class
    
    Example:
        >>> result = predict_sentiment("This product is amazing!")
        >>> print(result['label'])  # Positive
        >>> print(result['confidence'])  # 0.95
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
    
    # Get probabilities for each class
    probabilities = model.predict_proba(text_tfidf)[0]
    
    # Convert to human-readable format
    predicted_label = "Positive" if predicted_label_id == 1 else "Negative"
    confidence = probabilities[predicted_label_id]
    
    # =========================================================================
    # Return results
    # =========================================================================
    
    result = {
        "text": review_text,
        "label": predicted_label,
        "label_id": int(predicted_label_id),
        "confidence": float(confidence),
        "probabilities": {
            "Negative": float(probabilities[0]),
            "Positive": float(probabilities[1]),
        },
    }
    
    return result


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Train Multinomial Naive Bayes model and make predictions.
    
    This will:
    1. Load and preprocess the data
    2. Train a Multinomial Naive Bayes model
    3. Evaluate the model
    4. Save the model and metrics
    5. Make a sample prediction
    """
    
    print("\n" + "=" * 70)
    print("MULTINOMIAL NAIVE BAYES - COMPLETE EXAMPLE")
    print("=" * 70)
    
    # Use a small sample for testing
    # Change to None for full dataset training (e.g., 50,000 samples)
    SAMPLE_LIMIT = 1000
    
    # Train the model
    model, X_train, X_test, y_train, y_test, metrics = train_multinomial_naive_bayes(
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
    
    print("\n" + "=" * 70)
    print("✓ Example complete!")
    print("=" * 70)