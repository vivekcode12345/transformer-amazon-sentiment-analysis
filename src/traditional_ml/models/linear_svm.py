"""
Linear Support Vector Machine (LinearSVC) for Sentiment Analysis
=================================================================

This module implements a Linear Support Vector Machine classifier for binary
sentiment analysis (Positive/Negative) using TF-IDF features.

WHAT IS A SUPPORT VECTOR MACHINE (SVM)?
SVM is a powerful classifier that finds the optimal hyperplane (decision boundary)
to separate classes. It maximizes the margin between positive and negative samples.

WHAT IS LINEAR SVM (LinearSVC)?
LinearSVC is a fast, linear implementation of SVM specifically designed for
large-scale text classification. It uses a linear kernel which works well with
high-dimensional TF-IDF features.

WHY LINEAR SVM?
- Excellent for high-dimensional data (thousands of TF-IDF features)
- Fast training and prediction
- Robust to overfitting
- Works well with imbalanced datasets
- Memory efficient (uses sparse matrices)
- Great for text classification tasks

IMPORTANT NOTE ABOUT CONFIDENCE SCORES:
LinearSVC does NOT support predict_proba() by default. Instead, we use
decision_function() which returns the distance from the decision boundary.
This is a confidence score, NOT a probability. It indicates how far the sample
is from the boundary, but doesn't sum to 1 like probabilities.

REUSES:
- Preprocessing from src.traditional_ml.preprocessing
- Training pipeline from src.traditional_ml.trainer
- Evaluation from src.traditional_ml.evaluation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

# Add project root to path for imports when running as script
# This allows the script to be run directly: python src/traditional_ml/models/linear_svm.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# Import scikit-learn's LinearSVC
from sklearn.svm import LinearSVC

# Import our unified training pipeline
# This REUSES the trainer module - no duplication!
from src.traditional_ml.trainer import train_model


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default hyperparameters for Linear SVM
# You can tune these to improve performance

LINEAR_SVM_CONFIG = {
    "C": 1.0,  # Regularization parameter (smaller = more regularization)
    "loss": "squared_hinge",  # Loss function
    "penalty": "l2",  # Penalty norm
    "dual": True,  # Use dual formulation (better for n_samples < n_features)
    "class_weight": "balanced",  # Handle imbalanced classes automatically
    "random_state": 42,  # For reproducibility
    "max_iter": 1000,  # Maximum number of iterations
}

# Paths for saving model and results
MODEL_OUTPUT_DIR = Path("models/traditional_ml/linear_svm")
METRICS_OUTPUT_PATH = Path("reports/linear_svm_metrics.json")


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_linear_svm(
    sample_limit: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    clean: bool = True,
) -> tuple[LinearSVC, Any, Any, Dict[str, Any], Any, Any, Any]:
    """
    Train a Linear SVM model for sentiment analysis.
    
    This function:
    1. Creates a Linear SVM model with predefined configuration
    2. Trains it using the unified training pipeline
    3. Returns the trained model and all artifacts
    
    Args:
        sample_limit (int, optional): Limit samples for testing
        test_size (float): Proportion of data for testing (0.2 = 20%)
        random_state (int): Random seed for reproducibility
        clean (bool): Whether to clean text before vectorization
        
    Returns:
        tuple: (model, vectorizer, metrics, X_train, X_test, y_train, y_test)
            - model: Trained LinearSVC model
            - vectorizer: Fitted TF-IDF vectorizer
            - metrics: Dictionary with evaluation metrics
            - X_train: TF-IDF features for training
            - X_test: TF-IDF features for testing
            - y_train: Training labels
            - y_test: Test labels
    
    Example:
        >>> model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_linear_svm()
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
    
    # Create the model with our hyperparameters
    model = LinearSVC(**LINEAR_SVM_CONFIG)
    
    # Use the unified training pipeline
    # This handles: data loading, training, prediction, evaluation, and saving
    # LinearSVC uses decision_function() for confidence scores
    model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_model(
        model=model,
        model_name="Linear SVM",
        model_output_dir=MODEL_OUTPUT_DIR,
        metrics_output_path=METRICS_OUTPUT_PATH,
        sample_limit=sample_limit,
        test_size=test_size,
        random_state=random_state,
        clean=clean,
        confidence_type="decision_function",  # LinearSVC uses decision_function
    )
    
    return model, vectorizer, metrics, X_train, X_test, y_train, y_test


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
    IMPORTANT: LinearSVC does not support predict_proba(). Instead, we use
    decision_function() which returns a confidence score (distance from the
    decision boundary). This is NOT a probability, but indicates how confident
    the model is in its prediction.
    
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
    
    # Import here to avoid circular imports
    import joblib
    import numpy as np
    
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
    Example: Train Linear SVM model and make predictions.
    
    This will:
    1. Create and train a Linear SVM model
    2. Evaluate the model
    3. Save the model and metrics
    4. Make a sample prediction
    """
    
    print("\n" + "=" * 70)
    print("LINEAR SVM - COMPLETE EXAMPLE")
    print("=" * 70)
    
    # Use full dataset for training (set to None for full dataset, or specify a number for testing)
    SAMPLE_LIMIT = None
    
    # Train the model
    model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_linear_svm(
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