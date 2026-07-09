"""Training pipeline for text-only BERT sentiment classification."""

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

from configs.config import PATH_CONFIG, RUNTIME_CONFIG, TEXT_CONFIG, create_project_dirs
from .metrics import compute_detailed_metrics, compute_trainer_metrics
from .model import load_model
from .preprocessing import (
    prepare_tokenized_datasets,
    stream_tokenized_train,
    materialize_validation_from_test,
    stream_tokenized_test,
    load_tokenizer,
    tokenize_splits,
)


def _runtime_device() -> str:
    """Pick runtime device: MPS if available, else CPU."""
    if RUNTIME_CONFIG.use_mps_if_available and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_training_arguments(use_cpu: bool) -> TrainingArguments:
    """Create TrainingArguments with project-defined hyperparameters."""
    return TrainingArguments(
        output_dir=str(PATH_CONFIG.text_checkpoints_dir),
        logging_dir=str(PATH_CONFIG.text_logging_dir),
        learning_rate=TEXT_CONFIG.learning_rate,
        num_train_epochs=TEXT_CONFIG.epochs,
        per_device_train_batch_size=TEXT_CONFIG.batch_size,
        per_device_eval_batch_size=TEXT_CONFIG.batch_size,
        weight_decay=TEXT_CONFIG.weight_decay,
        warmup_ratio=TEXT_CONFIG.warmup_ratio,
        eval_strategy=TEXT_CONFIG.evaluation_strategy,
        save_strategy=TEXT_CONFIG.save_strategy,
        save_steps=getattr(TEXT_CONFIG, "save_steps", None),
        load_best_model_at_end=TEXT_CONFIG.load_best_model_at_end,
        metric_for_best_model=TEXT_CONFIG.metric_for_best_model,
        greater_is_better=TEXT_CONFIG.greater_is_better,
        save_total_limit=3,
        logging_strategy="epoch",
        report_to="none",
        seed=TEXT_CONFIG.random_seed,
        data_seed=TEXT_CONFIG.random_seed,
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
    """Run end-to-end BERT fine-tuning and persist artifacts.

    Args:
        train_samples: Number of training samples (for sampled runs).
        test_samples: Number of test samples (for sampled runs).
        full_dataset: If True, stream full training split without materialization.
        dataset_id: Hugging Face dataset identifier.

    Returns:
        Dictionary with train, validation, test metrics.
    """
    create_project_dirs()
    set_seed(TEXT_CONFIG.random_seed)

    selected_device = _runtime_device()
    use_cpu = selected_device == "cpu"
    print(f"Selected runtime device: {selected_device}")

    if full_dataset:
        print("Running full-dataset streaming training (one pass over train)")

        tokenizer = load_tokenizer(TEXT_CONFIG.model_name)

        train_iterable = stream_tokenized_train(
            dataset_id=dataset_id,
            tokenizer=tokenizer,
            max_length=TEXT_CONFIG.max_length,
            batch_size=TEXT_CONFIG.batch_size,
        )

        validation_ds = materialize_validation_from_test(
            dataset_id=dataset_id,
            validation_size=TEXT_CONFIG.validation_materialize_size,
            seed=TEXT_CONFIG.random_seed,
        )

        tokenized_validation = tokenize_splits(
            DatasetDict({"validation": validation_ds}),
            tokenizer=tokenizer,
            max_length=TEXT_CONFIG.max_length,
        )["validation"]

        model = load_model()
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = _build_training_arguments(use_cpu=use_cpu)
        training_args.per_device_train_batch_size = TEXT_CONFIG.batch_size
        training_args.gradient_accumulation_steps = TEXT_CONFIG.gradient_accumulation_steps
        training_args.dataloader_num_workers = TEXT_CONFIG.dataloader_num_workers
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
            random_seed=TEXT_CONFIG.random_seed,
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

    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    if full_dataset:
        validation_metrics = trainer.evaluate(eval_dataset=tokenized_validation)
    else:
        validation_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["validation"])

    trainer.log_metrics("validation", validation_metrics)
    trainer.save_metrics("validation", validation_metrics)

    if full_dataset:
        test_iterable = stream_tokenized_test(
            dataset_id=dataset_id,
            tokenizer=tokenizer,
            max_length=TEXT_CONFIG.max_length,
            batch_size=TEXT_CONFIG.batch_size * 2,
        )
        test_predictions_output = trainer.predict(test_iterable)
    else:
        test_predictions_output = trainer.predict(tokenized_dataset["test"])

    test_detailed_metrics = compute_detailed_metrics(
        predictions=test_predictions_output.predictions,
        labels=test_predictions_output.label_ids,
    )

    trainer.save_model(str(PATH_CONFIG.text_best_model_dir))
    tokenizer.save_pretrained(str(PATH_CONFIG.text_tokenizer_dir))

    _save_json_metrics(PATH_CONFIG.text_train_metrics_path, train_result.metrics)
    _save_json_metrics(PATH_CONFIG.text_eval_metrics_path, test_detailed_metrics)

    print("Training completed.")
    print(f"Best model saved to: {PATH_CONFIG.text_best_model_dir}")
    print(f"Tokenizer saved to: {PATH_CONFIG.text_tokenizer_dir}")

    return {
        "train": train_result.metrics,
        "validation": validation_metrics,
        "test": test_detailed_metrics,
    }


if __name__ == "__main__":
    results = run_training()
    print("Final summary metrics:")
    print(json.dumps(results["test"], indent=2))
