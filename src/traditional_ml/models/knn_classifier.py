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
- Training pipeline from src.traditional_ml.trainer
- Evaluation from src.traditional_ml.evaluation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

# Add project root to path for imports when running as script
# This allows the script to be run directly: python src/traditional_ml/models/knn_classifier.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# Import scikit-learn's KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# Import our unified training pipeline
# This REUSES the trainer module - no duplication!
from src.traditional_ml.trainer import train_model


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
) -> tuple[KNeighborsClassifier, Any, Any, Dict[str, Any], Any, Any, Any]:
    """
    Train a K-Nearest Neighbors model for sentiment analysis.
    
    This function:
    1. Creates a KNN model with predefined configuration
    2. Trains it using the unified training pipeline
    3. Returns the trained model and all artifacts
    
    Args:
        sample_limit (int, optional): Limit samples for testing
        test_size (float): Proportion of data for testing (0.2 = 20%)
        random_state (int): Random seed for reproducibility
        clean (bool): Whether to clean text before vectorization
        
    Returns:
        tuple: (model, vectorizer, metrics, X_train, X_test, y_train, y_test)
            - model: Trained KNeighborsClassifier model
            - vectorizer: Fitted TF-IDF vectorizer
            - metrics: Dictionary with evaluation metrics
            - X_train: TF-IDF features for training
            - X_test: TF-IDF features for testing
            - y_train: Training labels
            - y_test: Test labels
    
    Example:
        >>> model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_knn_classifier()
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
    
    # Create the model with our hyperparameters
    model = KNeighborsClassifier(**KNN_CONFIG)
    
    # Use the unified training pipeline
    # This handles: data loading, training, prediction, evaluation, and saving
    model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_model(
        model=model,
        model_name="KNN Classifier",
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
    1. Create and train a KNN Classifier model
    2. Evaluate the model
    3. Save the model and metrics
    4. Make a sample prediction
    """
    
    print("\n" + "=" * 70)
    print("KNN CLASSIFIER - COMPLETE EXAMPLE")
    print("=" * 70)
    
    # Use full dataset for training (set to None for full dataset, or specify a number for testing)
    SAMPLE_LIMIT = None
    
    # Train the model
    model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_knn_classifier(
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