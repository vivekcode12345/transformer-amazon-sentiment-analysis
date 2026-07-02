"""CLI orchestrator for Phase-3: training, evaluation, and inference.

This script provides a small interactive menu to run the research-grade
training pipeline, evaluate a saved checkpoint, run a quick tokenization
preview, or perform single-text inference using the fine-tuned model.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from config import create_project_dirs, TRAINING_CONFIG
from preprocessing import (
    prepare_tokenized_datasets,
    print_dataset_sizes,
    print_tokenization_preview,
)
from train import run_training
from evaluation import evaluate_model
from inference import predict_sentiment, print_prediction_result


def _quick_tokenization_preview() -> None:
    print("Quick tokenization preview (small sampled subset)")
    tokenized_dataset, tokenizer = prepare_tokenized_datasets(
        train_samples=200, test_samples=50, validation_ratio=0.10
    )
    print_dataset_sizes(tokenized_dataset)
    print_tokenization_preview(tokenized_dataset, tokenizer)


def _train_flow() -> None:
    print("Training flow: this will fine-tune BERT and save artifacts.")
    confirm = input("Proceed with training? (y/N): ").strip().lower()
    if confirm != "y":
        print("Training cancelled by user.")
        return

    results = run_training()
    print("Training finished. Test metrics summary:")
    print(json.dumps(results.get("test", {}), indent=2))


def _evaluate_flow() -> None:
    print("Evaluate saved model on the held-out test split")
    metrics = evaluate_model()
    print("Returned metrics:")
    print(json.dumps(metrics, indent=2))


def _predict_flow() -> None:
    print("Interactive single-text prediction")
    review = input("Enter review text (or empty to cancel): ").strip()
    if not review:
        print("Prediction cancelled.")
        return
    result = predict_sentiment(review)
    print_prediction_result(result)


def _print_menu() -> None:
    print("\n== Amazon Polarity — Phase 3 Orchestrator ==")
    print("1) Quick tokenization preview (small sample)")
    print("2) Train model (fine-tune BERT)")
    print("3) Evaluate saved model")
    print("4) Predict single review")
    print("5) Exit")


def main(argv: list[str] | None = None) -> int:
    argv = argv or []
    create_project_dirs()

    # Allow non-interactive shortcuts: e.g., `python main.py train`
    if argv:
        cmd = argv[0].lower()
        if cmd in {"train", "run"}:
            _train_flow()
            return 0
        if cmd in {"eval", "evaluate"}:
            _evaluate_flow()
            return 0
        if cmd in {"preview", "tokenize"}:
            _quick_tokenization_preview()
            return 0
        if cmd in {"predict", "infer"}:
            if len(argv) >= 2:
                review = " ".join(argv[1:])
                print_prediction_result(predict_sentiment(review))
                return 0
            print("Usage: python main.py predict <your review text>")
            return 2

    # Interactive loop
    while True:
        _print_menu()
        choice = input("Select an option [1-5]: ").strip()
        if choice == "1":
            _quick_tokenization_preview()
        elif choice == "2":
            _train_flow()
        elif choice == "3":
            _evaluate_flow()
        elif choice == "4":
            _predict_flow()
        elif choice == "5":
            print("Exiting.")
            return 0
        else:
            print("Invalid selection. Please choose 1-5.")


if __name__ == "__main__":
    try:
        exit_code = main(sys.argv[1:])
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        exit_code = 1
    sys.exit(exit_code)
