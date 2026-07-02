"""CLI orchestrator for text-only and multimodal sentiment analysis pipelines.

This script provides an interactive menu to:
- Run text-only BERT training (Amazon Reviews)
- Run multimodal training (image+text) [stub, awaiting image encoder]
- Evaluate saved models
- Perform inference on single samples
- Preview tokenization

Supports both interactive (menu-driven) and non-interactive (CLI) modes.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from config import create_project_dirs, RUNTIME_CONFIG
from text.preprocessing import (
    prepare_tokenized_datasets,
    print_dataset_sizes,
    print_tokenization_preview,
)
from text.train import run_training as run_text_training
from text.evaluation import evaluate_model as evaluate_text_model
from text.inference import predict_sentiment, print_prediction_result


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
    confirm = input("Proceed with text training? (y/N): ").strip().lower()
    if confirm != "y":
        print("Training cancelled by user.")
        return

    results = run_text_training()
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
# MULTIMODAL PIPELINE (Stubs)
# ============================================================================

def _train_multimodal_flow() -> None:
    """Train multimodal (image+text) model (stub)."""
    print("Multimodal training: image+text sentiment analysis")
    print("⚠️  Currently a stub - awaiting image encoder specification.")
    print("When implemented, will:")
    print("  1. Load image+text pairs from dataset")
    print("  2. Encode images using pluggable encoder (CLIP, BLIP, etc.)")
    print("  3. Fuse image + text embeddings")
    print("  4. Train classification head")
    print("\nPlease specify:")
    print("  - Image encoder (clip, blip, siglip)")
    print("  - Fusion strategy (concat, attention, cross_modal)")
    print("  - Multimodal dataset specification")


def _evaluate_multimodal_flow() -> None:
    """Evaluate multimodal model (stub)."""
    print("Multimodal evaluation (stub)")
    print("Not yet implemented. Please train a multimodal model first.")


def _predict_multimodal_flow() -> None:
    """Predict sentiment for image+text pair (stub)."""
    print("Multimodal prediction: image+text (stub)")
    print("Not yet implemented.")


# ============================================================================
# MENU & ORCHESTRATION
# ============================================================================

def _print_menu() -> None:
    """Display main menu."""
    print("\n" + "=" * 60)
    print("Health Multimodal AI - Sentiment Analysis Orchestrator")
    print("=" * 60)
    print("\nText-Only Pipeline (BERT):")
    print("  1) Quick tokenization preview (small sample)")
    print("  2) Train BERT on Amazon Polarity")
    print("  3) Evaluate saved BERT model")
    print("  4) Predict single review (text-only)")
    print("\nMultimodal Pipeline (Image+Text) [Coming Soon]:")
    print("  5) Train multimodal model (stub)")
    print("  6) Evaluate multimodal model (stub)")
    print("  7) Predict image+text sentiment (stub)")
    print("\nUtilities:")
    print("  8) View configuration")
    print("  9) Exit")


def _print_config() -> None:
    """Display current configuration."""
    print("\n" + "=" * 60)
    print("Current Configuration")
    print("=" * 60)
    print(f"Multimodal enabled: {RUNTIME_CONFIG.enable_multimodal}")
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

        if cmd in {"multimodal-train"}:
            _train_multimodal_flow()
            return 0

        if cmd in {"multimodal-eval", "multimodal-evaluate"}:
            _evaluate_multimodal_flow()
            return 0

        if cmd in {"multimodal-predict"}:
            _predict_multimodal_flow()
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
            print("  multimodal-train        Train multimodal model (stub)")
            print("  multimodal-eval         Evaluate multimodal model (stub)")
            print("  (Interactive menu if no command provided)")
            return 0

    # Interactive menu loop
    while True:
        _print_menu()
        choice = input("\nSelect an option [1-9]: ").strip()

        if choice == "1":
            _quick_tokenization_preview()
        elif choice == "2":
            _train_text_flow()
        elif choice == "3":
            _evaluate_text_flow()
        elif choice == "4":
            _predict_text_flow()
        elif choice == "5":
            _train_multimodal_flow()
        elif choice == "6":
            _evaluate_multimodal_flow()
        elif choice == "7":
            _predict_multimodal_flow()
        elif choice == "8":
            _print_config()
        elif choice == "9":
            print("Exiting. Goodbye!")
            return 0
        else:
            print("❌ Invalid selection. Please choose 1-9.")


if __name__ == "__main__":
    try:
        exit_code = main(sys.argv[1:])
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Error: {exc}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    sys.exit(exit_code)
