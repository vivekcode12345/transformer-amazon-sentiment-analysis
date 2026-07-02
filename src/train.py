"""Training pipeline for BERT sentiment classification.

This module provides a research-grade training workflow using Hugging Face
Trainer, with early stopping, logging, checkpoints, and best-model export.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import DatasetDict

from config import PATH_CONFIG, RUNTIME_CONFIG, TRAINING_CONFIG, create_project_dirs
from metrics import compute_detailed_metrics, compute_trainer_metrics
from model import load_model
from preprocessing import (
    prepare_tokenized_datasets,
    stream_tokenized_train,
    materialize_validation_from_test,
    stream_tokenized_test,
    load_tokenizer,
)


def _runtime_device() -> str:
    """Pick runtime device based on project policy: MPS if available, else CPU."""
    if RUNTIME_CONFIG.use_mps_if_available and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_training_arguments(use_cpu: bool) -> TrainingArguments:
    """Create TrainingArguments with project-defined hyperparameters."""
    return TrainingArguments(
        output_dir=str(PATH_CONFIG.checkpoints_dir),
        logging_dir=str(PATH_CONFIG.logging_dir),
        learning_rate=TRAINING_CONFIG.learning_rate,
        num_train_epochs=TRAINING_CONFIG.epochs,
        per_device_train_batch_size=TRAINING_CONFIG.batch_size,
        per_device_eval_batch_size=TRAINING_CONFIG.batch_size,
        weight_decay=TRAINING_CONFIG.weight_decay,
        warmup_ratio=TRAINING_CONFIG.warmup_ratio,
        eval_strategy=TRAINING_CONFIG.evaluation_strategy,
        save_strategy=TRAINING_CONFIG.save_strategy,
        save_steps=getattr(TRAINING_CONFIG, "save_steps", None),
        load_best_model_at_end=TRAINING_CONFIG.load_best_model_at_end,
        metric_for_best_model=TRAINING_CONFIG.metric_for_best_model,
        greater_is_better=TRAINING_CONFIG.greater_is_better,
        save_total_limit=3,
        logging_strategy="epoch",
        report_to="none",
        seed=TRAINING_CONFIG.random_seed,
        data_seed=TRAINING_CONFIG.random_seed,
        use_cpu=use_cpu,
        remove_unused_columns=True,
    )


def _save_json_metrics(path: Path, metrics: dict) -> None:
    """Persist metrics to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def run_training(
    train_samples: int = 2000,
    test_samples: int = 500,
    full_dataset: bool = False,
    dataset_id: str = "fancyzhx/amazon_polarity",
) -> dict:
    """Run end-to-end fine-tuning and persist artifacts.

    Args:
        train_samples: Number of streamed training samples to materialize (used
            for quick/sampled runs).
        test_samples: Number of streamed test samples to materialize (used for
            quick/sampled runs).
        full_dataset: If True, train over the full training split using a
            streaming iterable (never materializes the full training set).
        dataset_id: HF dataset id to stream from.

    Returns:
        Dictionary containing train, validation, and test metrics.
    """
    create_project_dirs()
    set_seed(TRAINING_CONFIG.random_seed)

    selected_device = _runtime_device()
    use_cpu = selected_device == "cpu"
    print(f"Selected runtime device policy: {selected_device}")

    if full_dataset:
        # Streaming full-dataset mode: never materialize the train split.
        print("Running full-dataset streaming training (one pass over train)")

        tokenizer = load_tokenizer(TRAINING_CONFIG.model_name)

        # Streaming tokenized training iterable (datasets IterableDataset)
        train_iterable = stream_tokenized_train(
            dataset_id=dataset_id,
            tokenizer=tokenizer,
            max_length=TRAINING_CONFIG.max_length,
            batch_size=TRAINING_CONFIG.batch_size,
        )

        # Materialize a small validation set from the test split for evaluation.
        validation_ds = materialize_validation_from_test(
            dataset_id=dataset_id,
            validation_size=TRAINING_CONFIG.validation_materialize_size,
            seed=TRAINING_CONFIG.random_seed,
        )

        # Tokenize the validation dataset (materialized) in-memory.
        from preprocessing import tokenize_splits

        tokenized_validation = tokenize_splits(
            DatasetDict({"validation": validation_ds}),
            tokenizer=tokenizer,
            max_length=TRAINING_CONFIG.max_length,
        )["validation"]

        model = load_model()
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Build training arguments tuned for low-memory M1 machines.
        training_args = _build_training_arguments(use_cpu=use_cpu)
        # Override a few safe settings for streaming full-dataset runs.
        training_args.per_device_train_batch_size = TRAINING_CONFIG.batch_size
        training_args.gradient_accumulation_steps = TRAINING_CONFIG.gradient_accumulation_steps
        training_args.dataloader_num_workers = TRAINING_CONFIG.dataloader_num_workers
        # For streaming we'll run one pass over train (single epoch).
        training_args.num_train_epochs = 1

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_iterable,
            eval_dataset=tokenized_validation,
            data_collator=data_collator,
            compute_metrics=compute_trainer_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        print("Starting training (streaming)...")
        train_result = trainer.train()
    else:
        tokenized_dataset, tokenizer = prepare_tokenized_datasets(
            train_samples=train_samples,
            test_samples=test_samples,
            validation_ratio=0.10,
            random_seed=TRAINING_CONFIG.random_seed,
        )

        model = load_model()
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        training_args = _build_training_arguments(use_cpu=use_cpu)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            compute_metrics=compute_trainer_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        print("Starting training...")
        train_result = trainer.train()

    # Save trainer state, logs, and scalar metrics.
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    # Evaluate on validation split using trainer metric function.
    # Evaluate on the materialized validation set (if available in local scope)
    if full_dataset:
        validation_metrics = trainer.evaluate(eval_dataset=tokenized_validation)
    else:
        validation_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["validation"])
    trainer.log_metrics("validation", validation_metrics)
    trainer.save_metrics("validation", validation_metrics)

    # Run explicit test prediction for detailed research metrics.
    # Run explicit test prediction for detailed research metrics.
    if full_dataset:
        # Use streaming tokenized test iterable to avoid materializing the test split.
        test_iterable = stream_tokenized_test(
            dataset_id=dataset_id,
            tokenizer=tokenizer,
            max_length=TRAINING_CONFIG.max_length,
            batch_size=TRAINING_CONFIG.batch_size * 2,
        )
        test_predictions_output = trainer.predict(test_iterable)
    else:
        test_predictions_output = trainer.predict(tokenized_dataset["test"])
    test_detailed_metrics = compute_detailed_metrics(
        predictions=test_predictions_output.predictions,
        labels=test_predictions_output.label_ids,
    )

    # Persist final model artifacts.
    trainer.save_model(str(PATH_CONFIG.best_model_dir))
    tokenizer.save_pretrained(str(PATH_CONFIG.tokenizer_dir))

    # Persist custom metrics report files.
    _save_json_metrics(PATH_CONFIG.train_metrics_path, train_result.metrics)
    _save_json_metrics(PATH_CONFIG.eval_metrics_path, test_detailed_metrics)

    print("Training completed.")
    print(f"Best model saved to: {PATH_CONFIG.best_model_dir}")
    print(f"Tokenizer saved to: {PATH_CONFIG.tokenizer_dir}")

    return {
        "train": train_result.metrics,
        "validation": validation_metrics,
        "test": test_detailed_metrics,
    }


if __name__ == "__main__":
    results = run_training()
    print("Final summary metrics:")
    print(json.dumps(results["test"], indent=2))
