"""
Decision Tree Classifier for Sentiment Analysis
================================================

This module implements a Decision Tree classifier for binary sentiment
analysis (Positive/Negative) using TF-IDF features.

WHAT IS A DECISION TREE?
A Decision Tree is a flowchart-like structure where each internal node
represents a test on a feature (word), each branch represents the outcome
of the test, and each leaf node represents a class label (Positive/Negative).

HOW DOES IT WORK?
The tree splits the data recursively by selecting the feature that best
separates the classes. For text classification, it learns which words
are most indicative of positive or negative sentiment.

WHY DECISION TREE?
- Easy to interpret and visualize (can see the decision rules)
- Handles non-linear relationships
- Requires little data preprocessing
- Can capture complex patterns
- Fast for inference
- Great baseline model

CONFIGURATION:
- criterion="gini": Gini impurity for splitting
- max_depth=30: Maximum tree depth (prevents overfitting)
- min_samples_split=5: Minimum samples to split a node
- min_samples_leaf=2: Minimum samples in a leaf node
- class_weight="balanced": Handles imbalanced classes

REUSES:
- Preprocessing from src.traditional_ml.preprocessing
- Training pipeline from src.traditional_ml.trainer
- Evaluation from src.traditional_ml.evaluation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

# Add project root to path for imports when running as script
# This allows the script to be run directly: python src/traditional_ml/models/decision_tree.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# Import scikit-learn's DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import our unified training pipeline
# This REUSES the trainer module - no duplication!
from src.traditional_ml.trainer import train_model


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default hyperparameters for Decision Tree
# You can tune these to improve performance

DECISION_TREE_CONFIG = {
    "criterion": "gini",  # Splitting criterion (gini or entropy)
    "max_depth": 30,  # Maximum depth of the tree
    "min_samples_split": 5,  # Minimum samples required to split a node
    "min_samples_leaf": 2,  # Minimum samples required in a leaf node
    "class_weight": "balanced",  # Handle imbalanced classes automatically
    "random_state": 42,  # For reproducibility
}

# Paths for saving model and results
MODEL_OUTPUT_DIR = Path("models/traditional_ml/decision_tree")
METRICS_OUTPUT_PATH = Path("reports/decision_tree_metrics.json")


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_decision_tree(
    sample_limit: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    clean: bool = True,
) -> tuple[DecisionTreeClassifier, Any, Any, Dict[str, Any], Any, Any, Any]:
    """
    Train a Decision Tree model for sentiment analysis.
    
    This function:
    1. Creates a Decision Tree model with predefined configuration
    2. Trains it using the unified training pipeline
    3. Returns the trained model and all artifacts
    
    Args:
        sample_limit (int, optional): Limit samples for testing
        test_size (float): Proportion of data for testing (0.2 = 20%)
        random_state (int): Random seed for reproducibility
        clean (bool): Whether to clean text before vectorization
        
    Returns:
        tuple: (model, vectorizer, metrics, X_train, X_test, y_train, y_test)
            - model: Trained DecisionTreeClassifier model
            - vectorizer: Fitted TF-IDF vectorizer
            - metrics: Dictionary with evaluation metrics
            - X_train: TF-IDF features for training
            - X_test: TF-IDF features for testing
            - y_train: Training labels
            - y_test: Test labels
    
    Example:
        >>> model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_decision_tree()
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
    
    # Create the model with our hyperparameters
    model = DecisionTreeClassifier(**DECISION_TREE_CONFIG)
    
    # Use the unified training pipeline
    # This handles: data loading, training, prediction, evaluation, and saving
    model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_model(
        model=model,
        model_name="Decision Tree",
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
    Decision Tree supports predict_proba(), so we can return true probabilities.
    
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
    # Decision Tree supports predict_proba(), so we get true probabilities
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
    Example: Train Decision Tree model and make predictions.
    
    This will:
    1. Create and train a Decision Tree model
    2. Evaluate the model
    3. Save the model and metrics
    4. Make a sample prediction
    """
    
    print("\n" + "=" * 70)
    print("DECISION TREE - COMPLETE EXAMPLE")
    print("=" * 70)
    
    # Use a small sample for testing
    # Change to None for full dataset training (e.g., 50,000 samples)
    SAMPLE_LIMIT = 1000
    
    # Train the model
    model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_decision_tree(
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