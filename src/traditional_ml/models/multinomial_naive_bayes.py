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
- Training pipeline from src.traditional_ml.trainer
- Evaluation from src.traditional_ml.evaluation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

# Add project root to path for imports when running as script
# This allows the script to be run directly: python src/traditional_ml/models/multinomial_naive_bayes.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# Import scikit-learn's Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB

# Import our unified training pipeline
# This REUSES the trainer module - no duplication!
from src.traditional_ml.trainer import train_model


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
) -> tuple[MultinomialNB, Any, Any, Dict[str, Any], Any, Any, Any]:
    """
    Train a Multinomial Naive Bayes model for sentiment analysis.
    
    This function:
    1. Creates a Multinomial Naive Bayes model with predefined configuration
    2. Trains it using the unified training pipeline
    3. Returns the trained model and all artifacts
    
    Args:
        sample_limit (int, optional): Limit samples for testing
        test_size (float): Proportion of data for testing (0.2 = 20%)
        random_state (int): Random seed for reproducibility
        clean (bool): Whether to clean text before vectorization
        
    Returns:
        tuple: (model, vectorizer, metrics, X_train, X_test, y_train, y_test)
            - model: Trained MultinomialNB model
            - vectorizer: Fitted TF-IDF vectorizer
            - metrics: Dictionary with evaluation metrics
            - X_train: TF-IDF features for training
            - X_test: TF-IDF features for testing
            - y_train: Training labels
            - y_test: Test labels
    
    Example:
        >>> model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_multinomial_naive_bayes()
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
    
    # Create the model with our hyperparameters
    model = MultinomialNB(**MULTINOMIAL_NAIVE_BAYES_CONFIG)
    
    # Use the unified training pipeline
    # This handles: data loading, training, prediction, evaluation, and saving
    model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_model(
        model=model,
        model_name="Multinomial Naive Bayes",
        model_output_dir=MODEL_OUTPUT_DIR,
        metrics_output_path=METRICS_OUTPUT_PATH,
        sample_limit=sample_limit,
        test_size=test_size,
        random_state=random_state,
        clean=clean,
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
    
    # Import here to avoid circular imports
    import joblib
    
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
    1. Create and train a Multinomial Naive Bayes model
    2. Evaluate the model
    3. Save the model and metrics
    4. Make a sample prediction
    """
    
    print("\n" + "=" * 70)
    print("MULTINOMIAL NAIVE BAYES - COMPLETE EXAMPLE")
    print("=" * 70)
    
    # Use a small sample for testing
    # Change to None for full dataset training (e.g., 50,000 samples)
    SAMPLE_LIMIT = 1000
    
    # Train the model
    model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_multinomial_naive_bayes(
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