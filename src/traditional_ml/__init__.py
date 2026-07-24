"""
Traditional Machine Learning Module
====================================

This module provides a clean, modular architecture for implementing traditional
machine learning algorithms on the Amazon Polarity dataset.

Available Algorithms:
- Logistic Regression
- Multinomial Naive Bayes
- Bernoulli Naive Bayes
- Linear SVM
- SGD Classifier
- Decision Tree
- Random Forest
- KNN

Quick Start:
-----------
>>> from src.traditional_ml.preprocessing import prepare_data_for_training
>>> X_train, X_test, y_train, y_test, vectorizer = prepare_data_for_training()
>>>
>>> # Train a model
>>> from sklearn.linear_model import LogisticRegression
>>> model = LogisticRegression()
>>> model.fit(X_train, y_train)
>>>
>>> # Make predictions
>>> predictions = model.predict(X_test)

Module Structure:
-----------------
- preprocessing.py: Text cleaning, TF-IDF vectorization, data loading
- models/: Individual model implementations (coming soon)
- trainer.py: Training orchestration (coming soon)
- evaluator.py: Model evaluation (coming soon)
- metrics.py: Metrics computation (coming soon)
- utils.py: Helper utilities (coming soon)
- cli.py: Command-line interface (coming soon)
"""

__version__ = "1.0.0"
__author__ = "Vivek Verma"

# Import main preprocessing functions for easy access
from src.traditional_ml.preprocessing import (
    clean_text,
    clean_dataset,
    load_amazon_polarity_data,
    create_tfidf_features,
    prepare_data_for_training,
    get_feature_names,
    print_top_features,
)

# Define what gets imported with "from src.traditional_ml import *"
__all__ = [
    # Main pipeline
    "prepare_data_for_training",
    # Text cleaning
    "clean_text",
    "clean_dataset",
    # Data loading
    "load_amazon_polarity_data",
    # TF-IDF
    "create_tfidf_features",
    # Utilities
    "get_feature_names",
    "print_top_features",
]