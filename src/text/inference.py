"""Single-text sentiment prediction using fine-tuned text-only BERT."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel

from ..config import PATH_CONFIG, RUNTIME_CONFIG, TEXT_CONFIG


def _runtime_device() -> str:
    """Pick runtime device: MPS if available, else CPU."""
    if RUNTIME_CONFIG.use_mps_if_available and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_finetuned_model(model_dir: Path | str | None = None) -> PreTrainedModel:
    """Load fine-tuned BERT model from saved checkpoint."""
    resolved_dir = model_dir or PATH_CONFIG.text_best_model_dir

    if not Path(resolved_dir).exists():
        raise FileNotFoundError(
            f"Model directory not found: {resolved_dir}. Train model first."
        )

    return AutoModelForSequenceClassification.from_pretrained(str(resolved_dir))


def predict_sentiment(
    review_text: str,
    model_dir: Path | str | None = None,
    max_length: int | None = None,
) -> dict:
    """Predict sentiment (positive/negative) for a single review text.

    Args:
        review_text: Input text for prediction.
        model_dir: Optional override for saved model directory.
        max_length: Optional override for tokenization length.

    Returns:
        Dictionary with prediction results including probability distribution.
    """
    if not review_text or not review_text.strip():
        raise ValueError("review_text cannot be empty.")

    resolved_model_dir = model_dir or PATH_CONFIG.text_best_model_dir
    resolved_max_length = max_length or TEXT_CONFIG.max_length
    device = _runtime_device()

    model = load_finetuned_model(resolved_model_dir)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(TEXT_CONFIG.model_name)

    inputs = tokenizer(
        review_text,
        padding="max_length",
        truncation=True,
        max_length=resolved_max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probabilities = torch.softmax(logits, dim=0)

    predicted_label_id = torch.argmax(logits).item()
    predicted_label = "Positive" if predicted_label_id == 1 else "Negative"
    confidence_score = probabilities[predicted_label_id].item()

    return {
        "review_text": review_text,
        "predicted_label_id": predicted_label_id,
        "predicted_label": predicted_label,
        "confidence_score": float(confidence_score),
        "probabilities": {
            "negative": float(probabilities[0]),
            "positive": float(probabilities[1]),
        },
    }


def print_prediction_result(result: dict) -> None:
    """Pretty-print prediction result."""
    print("Prediction Result")
    print("-" * 60)
    print(f"Text: {result['review_text'][:100]}...")
    print(f"Predicted Label: {result['predicted_label']}")
    print(f"Confidence: {result['confidence_score']:.4f}")
    print(f"Probabilities: {result['probabilities']}")
