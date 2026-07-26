"""
K-Nearest Neighbors (KNN) Classifier for Sentiment Analysis
============================================================

This module implements a K-Nearest Neighbors classifier for binary sentiment
analysis (Positive/Negative) using TF-IDF features.

WHAT IS K-NEAREST NEIGHBORS (KNN)?
KNN is a simple, instance-based learning algorithm that classifies new samples
by finding the K most similar training samples (neighbors) and taking a majority
vote. It's called "lazy learning" because it doesn't learn a model during training,
it just stores the training data.

HOW DOES IT WORK?
1. Store all training data (TF-IDF features)
2. For a new sample, find the K nearest neighbors (most similar samples)
3. Take a majority vote among the neighbors
4. Return the class with the most votes

WHY KNN FOR TEXT CLASSIFICATION?
- Simple and intuitive algorithm
- No training phase (just stores data)
- Works well with TF-IDF features
- Non-parametric (makes no assumptions about data distribution)
- Can capture complex decision boundaries
- Cosine similarity is ideal for text embeddings

CONFIGURATION:
- n_neighbors=5: Number of neighbors to consider
- weights="distance": Weight votes by distance (closer neighbors have more weight)
- metric="cosine": Cosine distance (better for text than Euclidean)
- algorithm="brute": Brute-force search (necessary for cosine metric)
- n_jobs=-1: Use all CPU cores

NOTE ON SPARSE MATRICES:
TF-IDF features are sparse matrices (mostly zeros). KNN is configured with
algorithm="brute" which works efficiently with sparse matrices and supports
cosine similarity, which is more suitable for text than Euclidean distance.

REUSES:
- Preprocessing from src.traditional_ml.preprocessing
- Metrics computation from sklearn
"""

from __future__ import annotations

import json  # For saving metrics to JSON file
from pathlib import Path  # For file path handling
from typing import Any, Dict, Tuple  # Type hints for better code documentation

# Add project root to path for imports when running as script
# This allows the script to be run directly: python src/traditional_ml/models/knn_classifier.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# Import scikit-learn's KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
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

# Default hyperparameters for KNN
# Optimized for text classification with TF-IDF features

KNN_CONFIG = {
    "n_neighbors": 5,  # Number of neighbors to consider
    "weights": "distance",  # Weight votes by distance (closer = more weight)
    "metric": "cosine",  # Cosine distance (better for text than Euclidean)
    "algorithm": "brute",  # Brute-force search (required for cosine metric)
    "n_jobs": -1,  # Use all CPU cores for parallel processing
}

# Paths for saving model and results
MODEL_OUTPUT_DIR = Path("models/traditional_ml/knn_classifier")
METRICS_OUTPUT_PATH = Path("reports/knn_classifier_metrics.json")


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_knn_classifier(
    sample_limit: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    clean: bool = True,
) -> Tuple[KNeighborsClassifier, Any, Any, Any, Any, Dict[str, Any]]:
    """
    Train a K-Nearest Neighbors model for sentiment analysis.
    
    This function:
    1. Loads and preprocesses the data (using our preprocessing pipeline)
    2. Trains a KNN model (stores training data)
    3. Evaluates the model on test data
    4. Saves the model and metrics
    
    Args:
        sample_limit (int, optional): Limit samples for testing
        test_size (float): Proportion of data for testing (0.2 = 20%)
        random_state (int): Random seed for reproducibility
        clean (bool): Whether to clean text before vectorization
        
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test, metrics)
            - model: Trained KNeighborsClassifier model
            - X_train: TF-IDF features for training
            - X_test: TF-IDF features for testing
            - y_train: Training labels
            - y_test: Test labels
            - metrics: Dictionary with evaluation metrics
    
    Example:
        >>> model, X_train, X_test, y_train, y_test, metrics = train_knn_classifier()
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
    
    print("\n" + "=" * 70)
    print("KNN CLASSIFIER - TRAINING PIPELINE")
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
    # STEP 2: Initialize the KNN model
    # =========================================================================
    print("\n[Step 2/6] Initializing KNN Classifier model...")
    print(f"   Configuration: {KNN_CONFIG}")
    
    # Create the model with our hyperparameters
    model = KNeighborsClassifier(**KNN_CONFIG)
    
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Number of neighbors: {KNN_CONFIG['n_neighbors']}")
    print(f"   Weights: {KNN_CONFIG['weights']}")
    print(f"   Metric: {KNN_CONFIG['metric']}")
    print(f"   Algorithm: {KNN_CONFIG['algorithm']}")
    
    # =========================================================================
    # STEP 3: Train the model
    # =========================================================================
    print("\n[Step 3/6] Training model...")
    print("   KNN doesn't 'learn' - it stores the training data")
    print("   This is fast, but prediction may be slower...")
    
    # fit() stores the training data (KNN is a lazy learner)
    # The model will use this data to find neighbors during prediction
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
    metrics = evaluate_knn_classifier(y_test, y_pred, y_pred_proba)
    
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
    print("✓ KNN CLASSIFIER TRAINING COMPLETE!")
    print("=" * 70)
    print("\nYou can now:")
    print("1. Load the model: model = joblib.load('models/traditional_ml/knn_classifier/model.pkl')")
    print("2. Make predictions: model.predict(new_tfidf_features)")
    print("3. View metrics: reports/knn_classifier_metrics.json")
    print("=" * 70)
    
    # Return everything for further analysis if needed
    return model, X_train, X_test, y_train, y_test, metrics


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_knn_classifier(
    y_true: Any,
    y_pred: Any,
    y_pred_proba: Any,
) -> Dict[str, Any]:
    """
    Evaluate KNN Classifier model performance.
    
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
    
    Example:
        >>> metrics = evaluate_knn_classifier(y_test, y_pred, y_pred_proba)
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
    KNN supports predict_proba(), so we can return true probabilities based on
    neighbor votes.
    
    Args:
        review_text (str): The review text to analyze
        model_path (Path): Path to the saved model
        vectorizer_path (Path): Path to the saved vectorizer
        
    Returns:
        Dictionary with prediction results:
        - text: Original review text
        - label: Predicted sentiment (Positive/Negative)
        - confidence: Confidence score (probability from neighbor votes)
        - probabilities: Dictionary with probabilities for each class
    
    Example:
        >>> result = predict_sentiment("This product is amazing!")
        >>> print(result['label'])  # Positive
        >>> print(result['confidence'])  # 0.80 (probability from neighbor votes)
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
    # KNN supports predict_proba() - returns proportion of neighbors in each class
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
        "confidence": float(confidence),  # Probability from neighbor votes
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
    Example: Train KNN Classifier model and make predictions.
    
    This will:
    1. Load and preprocess the data
    2. Train a KNN Classifier model
    3. Evaluate the model
    4. Save the model and metrics
    5. Make a sample prediction
    """
    
    print("\n" + "=" * 70)
    print("KNN CLASSIFIER - COMPLETE EXAMPLE")
    print("=" * 70)
    
    # Use a small sample for testing
    # Change to None for full dataset training (e.g., 50,000 samples)
    SAMPLE_LIMIT = 1000
    
    # Train the model
    model, X_train, X_test, y_train, y_test, metrics = train_knn_classifier(
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