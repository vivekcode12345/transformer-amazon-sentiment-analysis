"""CLI orchestrator for text-only BERT sentiment analysis pipeline.

This script provides an interactive menu to:
- Run text-only BERT training (Amazon Reviews)
- Evaluate saved models
- Perform inference on single samples
- Preview tokenization

Supports both interactive (menu-driven) and non-interactive (CLI) modes.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from configs.config import create_project_dirs, RUNTIME_CONFIG
from .text.preprocessing import (
    prepare_tokenized_datasets,
    print_dataset_sizes,
    print_tokenization_preview,
)
from .text.train import run_training as run_text_training
from .text.evaluation import evaluate_model as evaluate_text_model
from .text.inference import predict_sentiment, print_prediction_result


# ============================================================================
# TEXT-ONLY PIPELINE
# ============================================================================

def _quick_tokenization_preview() -> None:
    """Preview tokenization on small sampled subset."""
    print("Quick tokenization preview (small sampled subset)")
    tokenized_dataset, tokenizer = prepare_tokenized_datasets(
        train_samples=200, test_samples=50, validation_ratio=0.10
    )
    print_dataset_sizes(tokenized_dataset)
    print_tokenization_preview(tokenized_dataset, tokenizer)


def _train_text_flow() -> None:
    """Train text-only BERT model."""
    print("Text-only training: fine-tune BERT on Amazon Polarity")
    print("Proceeding with training...")

    results = run_text_training(
        train_samples=50000,
        test_samples=10000,
    )
    print("Training finished. Test metrics summary:")
    print(json.dumps(results.get("test", {}), indent=2))


def _evaluate_text_flow() -> None:
    """Evaluate saved text-only model."""
    print("Evaluate saved text-only model on test split")
    metrics = evaluate_text_model()
    print("Returned metrics:")
    print(json.dumps(metrics, indent=2))


def _predict_text_flow() -> None:
    """Predict sentiment for single review text."""
    print("Interactive single-text prediction (text-only model)")
    review = input("Enter review text (or empty to cancel): ").strip()
    if not review:
        print("Prediction cancelled.")
        return
    result = predict_sentiment(review)
    print_prediction_result(result)




# ============================================================================
# MENU & ORCHESTRATION
# ============================================================================

def _print_menu() -> None:
    """Display main menu."""
    print("\n" + "=" * 60)
    print("BERT Sentiment Analysis - Amazon Reviews")
    print("=" * 60)
    print("\nText-Only Pipeline (BERT):")
    print("  1) Quick tokenization preview (small sample)")
    print("  2) Train BERT on Amazon Polarity")
    print("  3) Evaluate saved BERT model")
    print("  4) Predict single review (text-only)")
    print("\nUtilities:")
    print("  5) View configuration")
    print("  6) Exit")


def _print_config() -> None:
    """Display current configuration."""
    print("\n" + "=" * 60)
    print("Current Configuration")
    print("=" * 60)
    print(f"MPS available: {RUNTIME_CONFIG.use_mps_if_available}")


def main(argv: list[str] | None = None) -> int:
    """Main orchestrator: interactive menu or CLI shortcuts.
    
    CLI shortcuts:
      python main.py text-train      → Run BERT training
      python main.py text-eval       → Evaluate BERT model
      python main.py text-predict TEXT   → Predict on TEXT
      python main.py preview         → Quick preview
    """
    argv = argv or []
    create_project_dirs()

    # CLI shortcuts for non-interactive mode
    if argv:
        cmd = argv[0].lower()

        if cmd in {"text-train", "text-run"}:
            _train_text_flow()
            return 0

        if cmd in {"text-eval", "text-evaluate"}:
            _evaluate_text_flow()
            return 0

        if cmd in {"text-predict", "text-infer"}:
            if len(argv) >= 2:
                review = " ".join(argv[1:])
                print_prediction_result(predict_sentiment(review))
                return 0
            print("Usage: python main.py text-predict '<your review text>'")
            return 2

        if cmd in {"preview", "tokenize"}:
            _quick_tokenization_preview()
            return 0

        # Legacy shortcuts for backward compatibility
        if cmd in {"train", "run"}:
            print("ℹ️  Using legacy 'train' shortcut. Consider 'python main.py text-train' for clarity.")
            _train_text_flow()
            return 0

        if cmd in {"eval", "evaluate"}:
            _evaluate_text_flow()
            return 0

        if cmd in {"predict", "infer"}:
            if len(argv) >= 2:
                review = " ".join(argv[1:])
                print_prediction_result(predict_sentiment(review))
                return 0
            print("Usage: python main.py predict '<your review text>'")
            return 2

        # Help
        if cmd in {"--help", "-h", "help"}:
            print("Usage: python main.py [command] [args]")
            print("\nCommands:")
            print("  text-train              Train text-only BERT model")
            print("  text-eval               Evaluate text-only model")
            print("  text-predict <text>     Predict sentiment for text")
            print("  preview                 Quick tokenization preview")
            print("  (Interactive menu if no command provided)")
            return 0

    # Interactive menu loop
    while True:
        _print_menu()
        choice = input("\nSelect an option [1-6]: ").strip()

        if choice == "1":
            _quick_tokenization_preview()
        elif choice == "2":
            _train_text_flow()
        elif choice == "3":
            _evaluate_text_flow()
        elif choice == "4":
            _predict_text_flow()
        elif choice == "5":
            _print_config()
        elif choice == "6":
            print("Exiting. Goodbye!")
            return 0
        else:
            print("❌ Invalid selection. Please choose 1-6.")


if __name__ == "__main__":
    try:
        exit_code = main(sys.argv[1:])
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Error: {exc}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    sys.exit(exit_code)
