"""
Traditional ML Models Package
==============================

This package contains implementations of various traditional machine learning
algorithms for sentiment analysis.
"""

__version__ = "1.0.0"

from src.traditional_ml.models.logistic_regression import (
    train_logistic_regression,
    predict_sentiment,
    LOGISTIC_REGRESSION_CONFIG,
)

__all__ = [
    "train_logistic_regression",
    "predict_sentiment",
    "LOGISTIC_REGRESSION_CONFIG",
]