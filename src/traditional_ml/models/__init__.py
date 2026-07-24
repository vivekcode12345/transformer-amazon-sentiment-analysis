"""
Traditional ML Models Package
==============================

This package contains implementations of various traditional machine learning
algorithms for sentiment analysis.

Available Models:
- Logistic Regression
- Multinomial Naive Bayes (coming soon)
- Bernoulli Naive Bayes (coming soon)
- Linear SVM (coming soon)
- SGD Classifier (coming soon)
- Decision Tree (coming soon)
- Random Forest (coming soon)
- KNN (coming soon)

Quick Start:
-----------
>>> from src.traditional_ml.models.logistic_regression import train_logistic_regression
>>> model, X_train, X_test, y_train, y_test, metrics = train_logistic_regression()
>>> print(f"Accuracy: {metrics['accuracy']:.2%}")

Module Structure:
-----------------
Each model module contains:
- train_*(): Train the model
- evaluate_*(): Evaluate the model
- predict_sentiment(): Make predictions
- Configuration dictionary for hyperparameters
"""

__version__ = "1.0.0"

# Import Logistic Regression for easy access
from src.traditional_ml.models.logistic_regression import (
    train_logistic_regression,
    evaluate_logistic_regression,
    predict_sentiment,
    LOGISTIC_REGRESSION_CONFIG,
)

# Define what gets imported with "from src.traditional_ml.models import *"
__all__ = [
    "train_logistic_regression",
    "evaluate_logistic_regression",
    "predict_sentiment",
    "LOGISTIC_REGRESSION_CONFIG",
]