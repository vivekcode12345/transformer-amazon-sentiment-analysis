# Transformer-Based Sentiment Prediction of Amazon Reviews

A production-ready pipeline for fine-tuning Hugging Face transformer models (BERT, DistilBERT, RoBERTa, ALBERT, DeBERTa, etc.) on Amazon review data for binary sentiment classification.

## Project Description

This project applies state-of-the-art transformer-based models to perform sentiment analysis on Amazon review data. It provides a flexible, configurable pipeline for training, evaluating, and deploying sentiment classification models. The architecture supports multiple Hugging Face transformer models through a single configuration change, making it easy to experiment with different architectures without modifying the codebase.

## Features

- **Multi-Model Support**: Switch between BERT, DistilBERT, RoBERTa, ALBERT, DeBERTa, and other Hugging Face models by changing one config value
- **Memory Optimized**: Gradient checkpointing and cache disabling for efficient training on Apple Silicon and CPU
- **Streaming Dataset Loading**: Memory-efficient data loading with automatic fallback to cached mode
- **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, F1, confusion matrix, and classification reports
- **Interactive CLI**: User-friendly menu-driven interface for training, evaluation, and inference
- **Production Ready**: Proper error handling, logging, and artifact management

## Dataset

This project uses the [Amazon Polarity Dataset](https://huggingface.co/datasets/fancyzhx/amazon_polarity) from Hugging Face, which contains:
- **Training samples**: Product reviews labeled as positive or negative sentiment
- **Test samples**: Held-out test set for evaluation
- **Source**: Amazon product reviews across various categories

The dataset is loaded using streaming mode for memory efficiency, with automatic fallback to non-streaming mode if needed.

## Project Structure

```
Amazon Research/
├── configs/
│   └── config.py              # Central configuration (hyperparameters, paths, runtime settings)
├── data/
│   ├── raw/                   # Raw data directory
│   └── processed/             # Processed datasets (gitignored)
├── models/
│   └── bert_finetuned/        # Fine-tuned model artifacts (gitignored)
│       ├── best_model/        # Best model checkpoint
│       ├── checkpoints/       # Training checkpoints
│       └── tokenizer/         # Saved tokenizer
├── reports/                   # Evaluation metrics and results
├── scripts/                   # Utility scripts
├── src/
│   ├── dataset_loaders/       # Dataset loading utilities
│   │   └── amazon_reviews_2023.py
│   ├── text/                  # Text-only sentiment analysis pipeline
│   │   ├── preprocessing.py   # Tokenization and dataset preparation
│   │   ├── model.py           # Model loading and configuration
│   │   ├── train.py           # Training pipeline
│   │   ├── evaluation.py      # Model evaluation
│   │   ├── inference.py       # Single-text prediction
│   │   └── metrics.py         # Metrics computation
│   └── utils/                 # Shared utilities
│       └── metrics.py
├── .gitignore                 # Git ignore rules
├── LICENSE                    # MIT License
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda
- (Optional) Apple Silicon Mac for MPS acceleration

### Setup

1. Clone the repository:
```bash
git clone https://github.com/vivekcode12345/transformer-amazon-sentiment-analysis.git
cd transformer-amazon-sentiment-analysis
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode

Run the CLI orchestrator without arguments to launch the interactive menu:

```bash
python -m src.main
```

This presents a menu with options to:
1. Preview tokenization on a small sample
2. Train BERT on Amazon Polarity
3. Evaluate saved model
4. Predict sentiment for a single review
5. View configuration
6. Exit

### Command-Line Mode

#### Quick Tokenization Preview
```bash
python -m src.main preview
```

#### Train Model
```bash
python -m src.main text-train
```

#### Evaluate Model
```bash
python -m src.main text-eval
```

#### Predict Sentiment
```bash
python -m src.main text-predict "This product is amazing! Best purchase I've made."
```

#### View Help
```bash
python -m src.main --help
```

## Training

### Configuration

Edit `configs/config.py` to customize training:

```python
from configs.config import TEXT_CONFIG

# Model selection - change this to use different transformers
TEXT_CONFIG.model_name = "bert-base-uncased"  # Default

# Supported models:
# - "bert-base-uncased"
# - "distilbert-base-uncased"
# - "roberta-base"
# - "albert-base-v2"
# - "microsoft/deberta-v3-base"

# Training hyperparameters
TEXT_CONFIG.learning_rate = 2e-5
TEXT_CONFIG.epochs = 3
TEXT_CONFIG.batch_size = 8
TEXT_CONFIG.max_length = 128
```

### Run Training

```bash
python -m src.main text-train
```

The training pipeline will:
1. Load and tokenize the Amazon Polarity dataset
2. Fine-tune the specified transformer model
3. Save the best model checkpoint to `models/bert_finetuned/best_model/`
4. Save the tokenizer to `models/bert_finetuned/tokenizer/`
5. Save training metrics to `reports/train_metrics.json`

## Evaluation

Evaluate the fine-tuned model on the test set:

```bash
python -m src.main text-eval
```

This generates detailed metrics including:
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- Classification Report

Results are saved to `reports/eval_metrics.json`.

## Inference

Predict sentiment for a single review text:

```bash
python -m src.main text-predict "Your review text here"
```

Or use the Python API:

```python
from src.text.inference import predict_sentiment, print_prediction_result

result = predict_sentiment("This product exceeded my expectations!")
print_prediction_result(result)
```

Output:
```
Prediction Result
------------------------------------------------------------
Text: This product exceeded my expectations!
Predicted Label: Positive
Confidence: 0.9523
Probabilities: {'negative': 0.0477, 'positive': 0.9523}
```

## Switching Transformer Models

The pipeline supports multiple Hugging Face transformer models. To switch models, simply change one line in `configs/config.py`:

```python
# Default: BERT
model_name: str = "bert-base-uncased"

# Switch to DistilBERT (faster, smaller)
model_name: str = "distilbert-base-uncased"

# Switch to RoBERTa (improved BERT)
model_name: str = "roberta-base"

# Switch to ALBERT (parameter-efficient)
model_name: str = "albert-base-v2"

# Switch to DeBERTa (state-of-the-art)
model_name: str = "microsoft/deberta-v3-base"
```

The pipeline automatically adapts because:
- Tokenizer loading uses `AutoTokenizer.from_pretrained(TEXT_CONFIG.model_name)`
- Model loading uses `AutoModelForSequenceClassification.from_pretrained(TEXT_CONFIG.model_name)`
- No model-specific code exists in the pipeline

## Model Architecture

The pipeline uses Hugging Face's `AutoModelForSequenceClassification` which automatically selects the appropriate model architecture based on the model name. All models are fine-tuned with:

- Binary classification head (2 labels: positive/negative)
- Gradient checkpointing for memory efficiency
- Early stopping to prevent overfitting
- AdamW optimizer with weight decay
- Linear learning rate warmup

## Performance Tips

- **Apple Silicon**: Enable MPS acceleration (enabled by default in `RUNTIME_CONFIG.use_mps_if_available`)
- **CPU Only**: Set `use_cpu_fallback = True` for CPU training
- **Memory Issues**: Reduce `batch_size` or `max_length` in config
- **Speed**: Use DistilBERT for faster training with minimal accuracy loss

## Future Work

- Support for multi-class sentiment (1-5 star ratings)
- Model ensembling techniques
- Hyperparameter optimization with Optuna
- ONNX export for production deployment
- REST API wrapper for inference
- Experiment tracking with Weights & Biases or MLflow

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{transformer_amazon_sentiment,
  title = {Transformer-Based Sentiment Prediction of Amazon Reviews},
  author = {Vivek Verma},
  year = {2025},
  url = {https://github.com/vivekcode12345/transformer-amazon-sentiment-analysis}
}
```

## Contact

For questions or feedback, please open an issue on GitHub.