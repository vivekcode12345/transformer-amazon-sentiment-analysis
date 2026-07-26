"""
Unified Training Pipeline for Traditional ML Models
====================================================

This module provides a single, reusable training pipeline for all traditional
ML models. It eliminates code duplication by centralizing the training logic
in one place.

WHY THIS MODULE?
Previously, each model (logistic_regression.py, linear_svm.py, etc.) had its
own train_*() function with duplicated code for:
- Data loading and preprocessing
- Model training
- Making predictions
- Evaluating the model
- Saving model, vectorizer, and metrics

This module consolidates all training logic into one reusable function,
following the DRY (Don't Repeat Yourself) principle.

WHAT DOES IT DO?
- Loads and preprocesses data using prepare_data_for_training()
- Trains any scikit-learn model
- Automatically uses predict_proba() or decision_function() based on model capabilities
- Evaluates the model using the unified evaluate_model() function
- Saves model, vectorizer, and metrics to specified paths
- Returns all artifacts for further analysis

REUSES:
- Preprocessing from src.traditional_ml.preprocessing
- Evaluation from src.traditional_ml.evaluation
- Used by all traditional ML models
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Add project root to path for imports when running as script
# This allows the script to be run directly: python src/traditional_ml/trainer.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

import joblib

# Import our custom modules
from src.traditional_ml.evaluation import evaluate_model, print_evaluation_results
from src.traditional_ml.preprocessing import prepare_data_for_training


def train_model(
    model: Any,
    model_name: str,
    model_output_dir: Union[str, Path],
    metrics_output_path: Union[str, Path],
    sample_limit: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    clean: bool = True,
    confidence_type: Optional[str] = None,
) -> Tuple[Any, Any, Dict[str, Any], Any, Any, Any, Any]:
    """
    Universal training pipeline for traditional ML models.
    
    This function handles the complete training workflow for any scikit-learn
    model, eliminating code duplication across model implementations.
    
    Args:
        model: Scikit-learn model instance (e.g., LogisticRegression(), RandomForestClassifier())
        model_name: Name of the model for logging (e.g., "Logistic Regression")
        model_output_dir: Directory path to save the trained model and vectorizer
        metrics_output_path: File path to save evaluation metrics (JSON)
        sample_limit (int, optional): Limit samples for testing. None = use all data
        test_size (float): Proportion of data for testing (0.2 = 20%)
        random_state (int): Random seed for reproducibility
        clean (bool): Whether to clean text before vectorization
        confidence_type (str, optional): Type of confidence scores:
            - None: Use predict_proba() if available
            - "decision_function": Use decision_function() for confidence scores
            - "proba": Explicitly use predict_proba()
            
    Returns:
        tuple: (model, vectorizer, metrics, X_train, X_test, y_train, y_test)
            - model: Trained scikit-learn model
            - vectorizer: Fitted TF-IDF vectorizer
            - metrics: Dictionary with evaluation metrics
            - X_train: TF-IDF features for training
            - X_test: TF-IDF features for testing
            - y_train: Training labels
            - y_test: Test labels
    
    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> 
        >>> # Create and train model
        >>> model = LogisticRegression()
        >>> model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_model(
        ...     model=model,
        ...     model_name="Logistic Regression",
        ...     model_output_dir="models/logistic_regression",
        ...     metrics_output_path="reports/logistic_regression_metrics.json",
        ...     sample_limit=1000
        ... )
        >>> 
        >>> # Use the results
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
        >>> predictions = model.predict(new_features)
    """
    
    print("\n" + "=" * 70)
    print(f"{model_name.upper()} - TRAINING PIPELINE")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: Prepare data using our preprocessing pipeline
    # =========================================================================
    print("\n[Step 1/6] Preparing data with TF-IDF features...")
    
    X_train, X_test, y_train, y_test, vectorizer = prepare_data_for_training(
        sample_limit=sample_limit,
        test_size=test_size,
        random_state=random_state,
        clean=clean,
    )
    
    # =========================================================================
    # STEP 2: Train the model
    # =========================================================================
    print(f"\n[Step 2/6] Training {model_name} model...")
    print("   This may take a moment...")
    
    # fit() trains the model on the training data
    model.fit(X_train, y_train)
    
    print("   ✓ Training complete!")
    
    # =========================================================================
    # STEP 3: Make predictions on test data
    # =========================================================================
    print("\n[Step 3/6] Making predictions on test data...")
    
    # predict() returns the predicted class labels (0 or 1)
    y_pred = model.predict(X_test)
    
    # Try to get probabilities or decision scores
    y_pred_proba = None
    decision_scores = None
    
    # Check if model supports predict_proba()
    if hasattr(model, 'predict_proba') and confidence_type != "decision_function":
        print("   Using predict_proba() for confidence scores")
        y_pred_proba = model.predict_proba(X_test)
    
    # Check if model supports decision_function()
    elif hasattr(model, 'decision_function') and confidence_type != "proba":
        print("   Using decision_function() for confidence scores")
        decision_scores = model.decision_function(X_test)
    
    else:
        print("   Model does not support probability or decision function predictions")
    
    print(f"   Predicted {len(y_pred)} samples")
    
    # =========================================================================
    # STEP 4: Evaluate the model
    # =========================================================================
    print("\n[Step 4/6] Evaluating model performance...")
    
    # Calculate all the metrics using the unified evaluation function
    metrics = evaluate_model(
        y_test,
        y_pred,
        y_pred_proba=y_pred_proba,
        decision_scores=decision_scores,
    )
    
    # Print metrics to console
    print_evaluation_results(metrics)
    
    # =========================================================================
    # STEP 5: Save model and metrics
    # =========================================================================
    print("\n[Step 5/6] Saving model and metrics...")
    
    # Convert paths to Path objects
    model_output_dir = Path(model_output_dir)
    metrics_output_path = Path(metrics_output_path)
    
    # Create output directories if they don't exist
    model_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the trained model using joblib
    model_path = model_output_dir / "model.pkl"
    vectorizer_path = model_output_dir / "vectorizer.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"   ✓ Model saved to: {model_path}")
    print(f"   ✓ Vectorizer saved to: {vectorizer_path}")
    
    # Save metrics to JSON file
    with metrics_output_path.open("w") as f:
        import json
        json.dump(metrics, f, indent=2)
    
    print(f"   ✓ Metrics saved to: {metrics_output_path}")
    
    # =========================================================================
    # Done!
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"✓ {model_name.upper()} TRAINING COMPLETE!")
    print("=" * 70)
    print("\nYou can now:")
    print(f"1. Load the model: model = joblib.load('{model_path}')")
    print(f"2. Make predictions: model.predict(new_tfidf_features)")
    print(f"3. View metrics: {metrics_output_path}")
    print("=" * 70)
    
    # Return everything for further analysis if needed
    return model, vectorizer, metrics, X_train, X_test, y_train, y_test


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Demonstrate the unified training pipeline.
    
    This shows how to use train_model() with different types of classifiers.
    """
    
    print("\n" + "=" * 70)
    print("UNIFIED TRAINING PIPELINE - EXAMPLE")
    print("=" * 70)
    
    # Example 1: Train Logistic Regression
    print("\n[Example 1] Training Logistic Regression")
    print("-" * 70)
    
    from sklearn.linear_model import LogisticRegression
    
    # Create model
    lr_model = LogisticRegression(class_weight='balanced', random_state=42)
    
    # Train using the unified pipeline
    model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_model(
        model=lr_model,
        model_name="Logistic Regression",
        model_output_dir="models/traditional_ml/logistic_regression_example",
        metrics_output_path="reports/logistic_regression_example_metrics.json",
        sample_limit=None,  # Use full dataset
        test_size=0.2,
        clean=True,
    )
    
    # Example 2: Train Linear SVM
    print("\n[Example 2] Training Linear SVM")
    print("-" * 70)
    
    from sklearn.svm import LinearSVC
    
    # Create model
    svm_model = LinearSVC(class_weight='balanced', random_state=42)
    
    # Train using the unified pipeline
    model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_model(
        model=svm_model,
        model_name="Linear SVM",
        model_output_dir="models/traditional_ml/linear_svm_example",
        metrics_output_path="reports/linear_svm_example_metrics.json",
        sample_limit=None,  # Use full dataset
        test_size=0.2,
        clean=True,
        confidence_type="decision_function",  # Explicitly use decision_function
    )
    
    # Example 3: Train Decision Tree
    print("\n[Example 3] Training Decision Tree")
    print("-" * 70)
    
    from sklearn.tree import DecisionTreeClassifier
    
    # Create model
    dt_model = DecisionTreeClassifier(
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    
    # Train using the unified pipeline
    model, vectorizer, metrics, X_train, X_test, y_train, y_test = train_model(
        model=dt_model,
        model_name="Decision Tree",
        model_output_dir="models/traditional_ml/decision_tree_example",
        metrics_output_path="reports/decision_tree_example_metrics.json",
        sample_limit=None,  # Use full dataset
        test_size=0.2,
        clean=True,
    )
    
    print("\n" + "=" * 70)
    print("✓ All examples complete!")
    print("=" * 70)