"""Evaluation of fine-tuned text-only BERT model on test split."""

from __future__ import annotations

from ..config import PATH_CONFIG, TEXT_CONFIG
from .metrics import compute_detailed_metrics, print_detailed_metrics
from .model import load_model
from .preprocessing import (
    prepare_tokenized_datasets,
    load_tokenizer,
    materialize_validation_from_test,
)
from transformers import Trainer, TrainingArguments


def evaluate_model(train_samples: int = 2000, test_samples: int = 500) -> dict:
    """Load saved model and run evaluation on test split.

    Returns:
        Dictionary with detailed test set metrics (accuracy, precision, recall,
        f1, confusion matrix, classification report).
    """
    if not PATH_CONFIG.text_best_model_dir.exists():
        raise FileNotFoundError(
            f"Saved model not found at {PATH_CONFIG.text_best_model_dir}. "
            "Run training first."
        )

    # Prepare test dataset
    tokenized_dataset, tokenizer = prepare_tokenized_datasets(
        train_samples=train_samples,
        test_samples=test_samples,
        validation_ratio=0.10,
        random_seed=TEXT_CONFIG.random_seed,
    )

    # Load fine-tuned model from checkpoint
    model = load_model()
    model.load_state_dict(
        __import__("torch").load(
            PATH_CONFIG.text_best_model_dir / "pytorch_model.bin",
            map_location="cpu",
        )
    )

    # Run evaluation
    training_args = TrainingArguments(
        output_dir=str(PATH_CONFIG.text_checkpoints_dir),
        per_device_eval_batch_size=TEXT_CONFIG.batch_size,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
    )

    predictions_output = trainer.predict(tokenized_dataset["test"])
    detailed_metrics = compute_detailed_metrics(
        predictions=predictions_output.predictions,
        labels=predictions_output.label_ids,
    )

    return detailed_metrics


if __name__ == "__main__":
    metrics = evaluate_model()
    print_detailed_metrics(metrics)
