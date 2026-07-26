"""
Random Forest Classifier for Sentiment Analysis
================================================

This module implements a Random Forest classifier for binary sentiment
analysis (Positive/Negative) using TF-IDF features.

WHAT IS RANDOM FOREST?
Random Forest is an ensemble method that combines multiple decision trees
to improve accuracy and prevent overfitting. It works by:
1. Building multiple decision trees on different subsets of the data
2. Each tree votes on the final classification
3. The majority vote determines the final prediction

HOW DOES IT WORK?
- Creates N decision trees (n_estimators)
- Each tree is trained on a random subset of data (bootstrap)
- Each split considers a random subset of features
- Final prediction is the majority vote of all trees

WHY RANDOM FOREST?
- Reduces overfitting compared to single decision trees
- More accurate than individual trees
- Handles high-dimensional data well
- Provides feature importance
- Robust to noise and outliers
- Works well with imbalanced datasets

CONFIGURATION:
- n_estimators=200: Number of trees in the forest
- criterion="gini": Gini impurity for splitting
- max_depth=30: Maximum tree depth
- min_samples_split=5: Minimum samples to split a node
- min_samples_leaf=2: Minimum samples in a leaf node
- class_weight="balanced": Handles imbalanced classes
- n_jobs=-1: Use all CPU cores for parallel training

REUSES:
- Preprocessing from src.traditional_ml.preprocessing
- Training pipeline from src.traditional_ml.trainer
- Evaluation from src.traditional_ml.evaluation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

# Add project root to path for imports when running as script
# This allows the script to be run directly: python src/traditional_ml/models/random_forest.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# Import scikit-learn's RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Import our unified training pipeline
# This REUSES the trainer module - no duplication!
from src.traditional_ml.trainer import train_model


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default hyperparameters for Random Forest
# You can tune these to improve performance

RANDOM_FOREST_CONFIG = {
    "n_estimators": 200,  # Number of trees in the forest
    "criterion": "gini",  # Splitting criterion (gini or entropy)
    "max_depth": 30,  # Maximum depth of the tree
    "min_samples_split": 5,  # Minimum samples required to split a node
    "min_samples_leaf": 2,  # Minimum samples required in a leaf node
    "class_weight": "balanced",  # Handle imbalanced classes automatically
    "random_state": 42,  # For reproducibility
    "n_jobs": -1,  # Use all CPU cores for training
}

# Paths for saving model and results
MODEL_OUTPUT_DIR = Path("models/traditional_ml/random_forest")
METRICS_OUTPUT_PATH = Path("reports/random_forest_metrics.json")


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_random_forest(
    sample_limit: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    clean: bool = True,
) -> tuple[RandomForestClassifier, Any, Any, Dict[str, Any], Any, Any, Any]:
    """
    Train a Random Forest model for sentiment analysis.
    
    This function:
    1. Creates a Random Forest model with predefined configuration
    2. Trains it using the unified training pipeline
    3. Returns the trained model and all artifacts
    
    Args:
        sample_limit (int, optional): Limit samples for testing
        test_size (float): Proportion of data for testing (0.2 = 20%)
        random_state (int): Random seed for reproducibility
        clean (bool): Whether to clean text before vectorization
        
    Returns:
        tuple: (model, vectorizer, metrics, X_train, X_test, y_train, y_test)
            - model: Trained RandomForestClassifier model
            - vectorizer: Fitted TF-IDF vectorizer
            - metrics: Dictionary with evaluation metrics
            - X_train: TF-IDF features for training
            - X_test: TF-IDF features for testing
            - y_train: Training labels
            - y_test: Test labels
    
    Example:
        >>> model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_random_forest()
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
    
    # Create the model with our hyperparameters
    model = RandomForestClassifier(**RANDOM_FOREST_CONFIG)
    
    # Use the unified training pipeline
    # This handles: data loading, training, prediction, evaluation, and saving
    model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_model(
        model=model,
        model_name="Random Forest",
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
    Random Forest supports predict_proba(), so we can return true probabilities.
    
    Args:
        review_text (str): The review text to analyze
        model_path (Path): Path to the saved model
        vectorizer_path (Path): Path to the saved vectorizer
        
    Returns:
        Dictionary with prediction results:
        - text: Original review text
        - label: Predicted sentiment (Positive/Negative)
        - confidence: Confidence score (probability)
        - probabilities: Dictionary with probabilities for each class
    
    Example:
        >>> result = predict_sentiment("This product is amazing!")
        >>> print(result['label'])  # Positive
        >>> print(result['confidence'])  # 0.95 (true probability)
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
    # Random Forest supports predict_proba(), so we get true probabilities
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
        "confidence": float(confidence),  # True probability from predict_proba
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
    Example: Train Random Forest model and make predictions.
    
    This will:
    1. Create and train a Random Forest model
    2. Evaluate the model
    3. Save the model and metrics
    4. Make a sample prediction
    """
    
    print("\n" + "=" * 70)
    print("RANDOM FOREST - COMPLETE EXAMPLE")
    print("=" * 70)
    
    # Use full dataset for training (set to None for full dataset, or specify a number for testing)
    SAMPLE_LIMIT = None
    
    # Train the model
    model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_random_forest(
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