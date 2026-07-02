"""Text-only BERT sentiment analysis module.

This subpackage contains all components for training, evaluation, and inference
on text-only Amazon product reviews using BERT.

Modules:
  - preprocessing: Data loading, tokenization, streaming utilities
  - model: BERT model loading with gradient checkpointing
  - train: Training pipeline with Trainer API
  - evaluation: Evaluation and metrics computation
  - inference: Single-text prediction
  - metrics: Detailed metric computation (accuracy, precision, recall, f1, confusion matrix)
"""
