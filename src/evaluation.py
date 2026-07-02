"""Model evaluation utilities for Amazon review sentiment classification.

This module loads the trained BERT model, evaluates it on the test split,
and prints research-grade metrics.
"""

from __future__ import annotations

from pathlib import Path

from transformers import Trainer, TrainingArguments

from config import PATH_CONFIG, TRAINING_CONFIG, create_project_dirs
from model import load_model
from preprocessing import prepare_tokenized_datasets, load_tokenizer


def evaluate_model(train_samples: int = 2000, test_samples: int = 500) -> dict:
    """Evaluate the saved model on the held-out test split.

    Args:
        train_samples: Number of streamed training samples to materialize.
        test_samples: Number of streamed test samples to materialize.

    Returns:
        Detailed metric dictionary for the test split.
    """
    # Avoid circular imports at module load time by importing metrics here.
    from metrics import compute_detailed_metrics, print_detailed_metrics

    create_project_dirs()

    tokenized_dataset, _ = prepare_tokenized_datasets(
        train_samples=train_samples,
        test_samples=test_samples,
        validation_ratio=0.10,
        random_seed=TRAINING_CONFIG.random_seed,
    )

    model_dir = Path(PATH_CONFIG.best_model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Saved model directory not found: {model_dir}. Run training first."
        )

    tokenizer = load_tokenizer(TRAINING_CONFIG.model_name)
    model = load_model()
    model = model.from_pretrained(str(model_dir))

    eval_args = TrainingArguments(
        output_dir=str(PATH_CONFIG.output_dir),
        per_device_eval_batch_size=TRAINING_CONFIG.batch_size,
        report_to="none",
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
    )

    predictions_output = trainer.predict(tokenized_dataset["test"])
    detailed_metrics = compute_detailed_metrics(
        predictions=predictions_output.predictions,
        labels=predictions_output.label_ids,
    )
    print_detailed_metrics(detailed_metrics)
    return detailed_metrics


if __name__ == "__main__":
    metrics = evaluate_model()
    print("\nReturned metrics dictionary:")
    print(metrics)
