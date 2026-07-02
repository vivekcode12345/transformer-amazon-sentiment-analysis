"""Model utilities for Amazon review sentiment classification.

This module provides a clean, reusable way to load the required transformer
model for binary sentiment prediction.
"""

from __future__ import annotations

from transformers import AutoModelForSequenceClassification, PreTrainedModel

from config import TRAINING_CONFIG


def load_model(model_name: str | None = None, num_labels: int | None = None) -> PreTrainedModel:
    """Load BERT sequence classification model.

    Args:
        model_name: Optional override for model name.
        num_labels: Optional override for number of classes.

    Returns:
        A Hugging Face PreTrainedModel ready for fine-tuning.
    """
    resolved_model_name = model_name or TRAINING_CONFIG.model_name
    resolved_num_labels = num_labels if num_labels is not None else TRAINING_CONFIG.num_labels

    if resolved_model_name != "bert-base-uncased":
        raise ValueError(
            "This project phase supports only bert-base-uncased. "
            f"Received: {resolved_model_name}"
        )

    if resolved_num_labels != 2:
        raise ValueError(
            "This project phase is configured for binary sentiment classification "
            f"with 2 labels. Received: {resolved_num_labels}"
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        resolved_model_name,
        num_labels=resolved_num_labels,
    )
    # Reduce peak memory by disabling cached key/value when training with
    # gradient checkpointing on memory-constrained devices.
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        # Some model wrappers or very old transformers versions may not
        # expose this method — ignore if unavailable.
        pass

    # Disable use_cache during training to avoid storing large KV caches.
    try:
        model.config.use_cache = False
    except Exception:
        pass

    return model


def print_model_summary(model: PreTrainedModel) -> None:
    """Print a minimal model summary for quick verification."""
    print("Model loaded successfully.")
    print(f"- Model class: {model.__class__.__name__}")
    print(f"- Num labels: {model.config.num_labels}")
    print(f"- Hidden size: {getattr(model.config, 'hidden_size', 'N/A')}")
