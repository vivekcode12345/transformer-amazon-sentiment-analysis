"""Inference utilities for Amazon review sentiment prediction.

This module loads the trained BERT checkpoint and predicts sentiment for custom
review text with probability and confidence score.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from config import PATH_CONFIG, TRAINING_CONFIG, create_project_dirs
from preprocessing import load_tokenizer


LABEL_MAP = {0: "Negative", 1: "Positive"}


def _runtime_device() -> torch.device:
    """Select the best available runtime device for inference."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_finetuned_model(model_dir: str | Path | None = None) -> PreTrainedModel:
    """Load the trained model from disk.

    Args:
        model_dir: Directory containing the saved fine-tuned checkpoint.

    Returns:
        A sequence classification model ready for inference.
    """
    resolved_model_dir = Path(model_dir) if model_dir is not None else PATH_CONFIG.best_model_dir

    if not resolved_model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {resolved_model_dir}. Run training first."
        )

    model = AutoModelForSequenceClassification.from_pretrained(str(resolved_model_dir))
    model.eval()
    return model


def predict_sentiment(
    review_text: str,
    model_dir: str | Path | None = None,
    max_length: int | None = None,
) -> dict[str, object]:
    """Predict sentiment for a custom Amazon review.

    Args:
        review_text: Free-form product review text.
        model_dir: Optional directory containing trained model artifacts.
        max_length: Optional override for tokenization length.

    Returns:
        Dictionary with predicted label, confidence score, and class probabilities.
    """
    if not review_text or not review_text.strip():
        raise ValueError("review_text must be a non-empty string.")

    create_project_dirs()
    resolved_max_length = max_length or TRAINING_CONFIG.max_length
    tokenizer = load_tokenizer(TRAINING_CONFIG.model_name)
    model = load_finetuned_model(model_dir=model_dir)

    device = _runtime_device()
    model = model.to(device)

    encoded_inputs = tokenizer(
        review_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=resolved_max_length,
    )
    encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}

    with torch.no_grad():
        outputs = model(**encoded_inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).squeeze(0)

    predicted_class_id = int(torch.argmax(probabilities).item())
    predicted_label = LABEL_MAP[predicted_class_id]
    confidence_score = float(probabilities[predicted_class_id].item())

    return {
        "review_text": review_text,
        "predicted_label_id": predicted_class_id,
        "predicted_label": predicted_label,
        "confidence_score": confidence_score,
        "probabilities": {
            "negative": float(probabilities[0].item()),
            "positive": float(probabilities[1].item()),
        },
    }


def print_prediction_result(result: dict[str, object]) -> None:
    """Pretty-print the sentiment prediction result."""
    print("Prediction Result")
    print("-" * 60)
    print(f"Review: {result['review_text']}")
    print(f"Predicted Sentiment: {result['predicted_label']}")
    print(f"Confidence Score: {result['confidence_score']:.4f}")
    print("Class Probabilities:")
    probabilities = result["probabilities"]
    print(f"- Negative: {probabilities['negative']:.4f}")
    print(f"- Positive: {probabilities['positive']:.4f}")


if __name__ == "__main__":
    sample_review = "This product exceeded my expectations and works perfectly."
    prediction = predict_sentiment(sample_review)
    print_prediction_result(prediction)
