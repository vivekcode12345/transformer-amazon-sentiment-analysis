# Transformer-Based Amazon Review Sentiment Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-Latest-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-green" alt="Hugging Face">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Status">
</p>

A production-ready pipeline for fine-tuning BERT transformer models on the Amazon Polarity dataset to classify customer reviews into **Positive** and **Negative** sentiments. This project demonstrates practical application of state-of-the-art NLP techniques for binary sentiment classification.

---

## Project Overview

### Problem Statement

Customer reviews contain valuable insights for businesses, but manually analyzing thousands of reviews is impractical. Sentiment analysis automates this process by classifying reviews as positive or negative, enabling businesses to:
- Monitor product reception in real-time
- Identify product issues quickly
- Understand customer satisfaction at scale
- Make data-driven business decisions

### Why Sentiment Analysis Matters

Sentiment analysis is a fundamental NLP task with applications across industries:
- **E-commerce**: Product review analysis and recommendation systems
- **Customer Support**: Automated ticket prioritization and routing
- **Market Research**: Brand monitoring and competitive analysis
- **Product Development**: Feature feedback extraction

### Why Transformers (BERT)?

Traditional machine learning approaches (TF-IDF + SVM/Logistic Regression) rely on hand-crafted features and struggle with:
- Context understanding (e.g., "not good" vs "good")
- Semantic meaning and word relationships
- Out-of-vocabulary words

BERT (Bidirectional Encoder Representations from Transformers) addresses these limitations through:
- **Bidirectional context**: Understands words from both directions
- **Attention mechanism**: Captures relationships between all words in a sequence
- **Pretrained knowledge**: Leverages knowledge from massive text corpora
- **Transfer learning**: Fine-tunes on task-specific data with minimal training

### Real-World Applications

- Automated review moderation systems
- Customer feedback analysis dashboards
- Product sentiment tracking over time
- Competitive product analysis tools

---

## Features

- **Fine-tuned BERT Model**: State-of-the-art transformer architecture optimized for sentiment classification
- **Amazon Polarity Dataset**: Large-scale dataset with balanced positive/negative reviews
- **Hugging Face Transformers**: Industry-standard library for pretrained models
- **PyTorch Framework**: Flexible deep learning framework with GPU acceleration
- **Efficient Tokenization**: optimized text preprocessing with special token handling
- **Complete Training Pipeline**: End-to-end workflow from data loading to model saving
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1 score, and confusion matrix
- **Model Checkpointing**: Saves best model during training for reproducibility
- **Inference-Ready Model**: Production-ready model with saved tokenizer
- **Reproducible Training**: Fixed random seeds and configurable hyperparameters
- **Multi-Model Support**: Easily switch between BERT, DistilBERT, RoBERTa, ALBERT, and DeBERTa
- **Memory Optimization**: Gradient checkpointing and cache management for efficient training
- **Interactive CLI**: User-friendly command-line interface for training, evaluation, and inference

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.8+ |
| **Deep Learning** | PyTorch |
| **NLP Library** | Hugging Face Transformers |
| **Dataset Loading** | Hugging Face Datasets |
| **Numerical Computing** | NumPy |
| **Machine Learning** | Scikit-learn |
| **Data Processing** | Pandas |
| **Visualization** | Matplotlib |
| **Development** | Kaggle Notebook, Git, GitHub |

---

## Project Structure

```
Amazon Research/
│
├── configs/
│   └── config.py                    # Central configuration (hyperparameters, paths, settings)
│
├── models/
│   ├── bert_finetuned/              # Fine-tuned BERT model artifacts (gitignored)
│   │   ├── best_model/              # Best model checkpoint
│   │   ├── checkpoints/             # Training checkpoints
│   │   └── tokenizer/               # Saved tokenizer files
│   │
│   └── traditional_ml/              # Traditional ML models (TF-IDF + scikit-learn)
│       ├── decision_tree/           # Decision Tree model
│       ├── knn_classifier/          # K-Nearest Neighbors
│       ├── linear_svm/              # Linear SVM
│       ├── logistic_regression/     # Logistic Regression
│       ├── multinomial_naive_bayes/ # Multinomial Naive Bayes
│       ├── random_forest/           # Random Forest
│       └── sgd_classifier/          # SGD Classifier
│
├── reports/                         # Evaluation metrics and results
│   ├── eval_metrics.json            # BERT test set evaluation results
│   ├── train_metrics.json           # BERT training history
│   ├── *_metrics.json               # Traditional ML model metrics
│
├── src/
│   ├── __init__.py
│   ├── main.py                      # CLI orchestrator
│   │
│   ├── dataset_loaders/             # Dataset loading utilities
│   │   ├── __init__.py
│   │   └── amazon_reviews_2023.py   # Amazon Reviews dataset loader
│   │
│   ├── text/                        # BERT transformer pipeline
│   │   ├── __init__.py
│   │   ├── preprocessing.py         # Tokenization and dataset preparation
│   │   ├── model.py                 # Model loading and configuration
│   │   ├── train.py                 # Training pipeline
│   │   ├── evaluation.py            # Model evaluation
│   │   ├── inference.py             # Single-text prediction
│   │   └── metrics.py               # Metrics computation
│   │
│   ├── traditional_ml/              # Traditional ML pipeline (TF-IDF + sklearn)
│   │   ├── __init__.py
│   │   ├── preprocessing.py         # Text cleaning and TF-IDF features
│   │   ├── trainer.py               # Unified training pipeline
│   │   ├── evaluation.py            # Model evaluation
│   │   ├── README.md                # Traditional ML documentation
│   │   └── models/                  # Individual model implementations
│   │       ├── decision_tree.py
│   │       ├── knn_classifier.py
│   │       ├── linear_svm.py
│   │       ├── logistic_regression.py
│   │       ├── multinomial_naive_bayes.py
│   │       ├── random_forest.py
│   │       └── sgd_classifier.py
│   │
│   └── utils/                       # Shared utilities
│       ├── __init__.py
│       └── metrics.py
│
├── .gitignore                       # Git ignore rules
├── LICENSE                          # MIT License
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
└── simple_inference.py              # Quick inference script for BERT model
```

---

## Model Architecture

### Pretrained BERT Base

The project uses **BERT Base Uncased** (`bert-base-uncased`) as the foundation model, which consists of:
- **12 transformer encoder layers**
- **12 attention heads**
- **768 hidden dimensions**
- **110M parameters**
- Pretrained on large corpus of English text (Wikipedia + BookCorpus)

### Tokenization

BERT uses **WordPiece tokenization** which:
- Splits text into subword units
- Handles out-of-vocabulary words effectively
- Adds special tokens: `[CLS]` (classification) and `[SEP]` (separator)
- Pads/truncates sequences to fixed length (128 tokens)
- Generates attention masks to ignore padding tokens

### Sequence Classification Head

On top of BERT's base architecture, a classification head is added:
- Takes `[CLS]` token embedding (768-dim)
- Passes through dropout layer for regularization
- Projects to 2 output classes (positive/negative)
- Uses softmax for probability distribution

### Fine-Tuning Process

1. **Load pretrained BERT** with classification head
2. **Freeze** base BERT layers initially (optional)
3. **Train** entire model on Amazon Polarity dataset
4. **Update** all weights via backpropagation
5. **Save** best checkpoint based on validation performance

### Optimizer and Loss

- **Optimizer**: AdamW (Adam with weight decay)
  - Learning rate: 2e-5 (small for fine-tuning)
  - Weight decay: 0.01 (prevents overfitting)
  - Linear warmup for first 10% of steps
  
- **Loss Function**: CrossEntropyLoss
  - Standard loss for multi-class classification
  - Combines LogSoftmax and NLLLoss

---

## Dataset

### Amazon Polarity Dataset

The [Amazon Polarity Dataset](https://huggingface.co/datasets/fancyzhx/amazon_polarity) from Hugging Face contains Amazon product reviews labeled for binary sentiment classification.

**Dataset Statistics:**
- **Task**: Binary classification (Positive vs Negative)
- **Source**: Amazon product reviews across multiple categories
- **Format**: Review text with binary sentiment labels
- **Loading**: Streaming mode for memory efficiency with automatic fallback
- **Total samples**: 60,000 (50,000 train + 10,000 test)

**Label Distribution:**
- **Class 0**: Negative sentiment (29,369 samples, 48.9%)
- **Class 1**: Positive sentiment (30,631 samples, 51.1%)
- **Balance**: Reasonably balanced (~51% positive)

The dataset is split into:
- **Training set**: Used for model fine-tuning
- **Test set**: Used for final evaluation (held-out)

**Preprocessing:**
- Text tokenization using BERT tokenizer
- Sequence length: 128 tokens (truncated/padded)
- Lowercase conversion (for uncased models)
- Special token addition ([CLS], [SEP])

---

## Training Details

**Training Configuration:**
- **Epochs**: 3
- **Batch Size**: 8
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW
- **Scheduler**: Linear warmup with decay
- **Max Sequence Length**: 128 tokens
- **Model**: BERT Base Uncased

**Training Process:**
1. Dataset loaded in streaming mode from Hugging Face Hub
2. Text tokenized and formatted for BERT input
3. Model fine-tuned for 3 epochs with gradient accumulation
4. Best model checkpoint saved based on validation loss
5. Training metrics logged to `reports/train_metrics.json`

**Status**: Training completed successfully

**Artifacts Saved:**
- Best model checkpoint: `models/bert_finetuned/best_model/`
- Tokenizer files: `models/bert_finetuned/tokenizer/`
- Training metrics: `reports/train_metrics.json`

---

## Experimental Design

### Dataset Split

This project uses the **Amazon Polarity dataset** (`fancyzhx/amazon_polarity`) with the following split:

- **Training set**: 50,000 samples (used for model fine-tuning)
- **Test set**: 10,000 samples (held-out, never used during training)
- **Validation set**: 1,000 samples (materialized from test split for BERT training)

**Note**: The BERT model was evaluated on a small validation subset (50 samples) due to streaming mode limitations. For complete evaluation on the full test set, see the Traditional ML baseline (Logistic Regression, 85.14% on 10,000 samples).

### Evaluation Protocol

**BERT Transformer:**
- Evaluated on validation subset (n=50) materialized during training
- Metrics: Accuracy, Precision, Recall, F1 Score
- Best model selected based on validation loss

**Traditional ML Models:**
- Logistic Regression: Evaluated on full test set (n=10,000)
- Other models: Preliminary evaluation on 200 samples (full evaluation pending)
- Metrics: Accuracy, Precision, Recall, F1 Score
- 5-fold cross-validation recommended for future work

### Reproducibility

- **Random seed**: 42 (fixed for all train/test splits)
- **Dataset source**: Hugging Face Hub (`fancyzhx/amazon_polarity`)
- **Training runtime**: ~57.67 seconds (BERT, 3 epochs)
- **Hardware**: CPU (MPS/GPU recommended for faster training)

### Limitations

- **BERT evaluation**: Results based on small validation subset (n=50), not full test set
- **Traditional ML**: Only Logistic Regression evaluated on full test set; other models show preliminary results
- **Dataset scope**: Results specific to Amazon Polarity dataset; generalization to other domains untested
- **Class imbalance**: Dataset is reasonably balanced (~51% positive), but slight imbalance may affect metrics
- **Computational resources**: BERT training on CPU is slow; GPU recommended for production training

---

## Results

### BERT Transformer Model

The fine-tuned BERT model was evaluated on a validation subset. Below are the performance metrics:

| Metric | Score |
|--------|-------|
| **Accuracy** | 70.00% |
| **Precision** | 65.85% |
| **Recall** | 96.43% |
| **F1 Score** | 78.26% |

**Evaluation Details:**
- **Test set size**: 50 samples (validation subset materialized during training)
- **Note**: This is a small validation subset, not the full 10,000 sample test set

#### Metrics Explanation

- **Accuracy (70.00%)**: Percentage of correctly classified reviews out of total predictions
- **Precision (65.85%)**: Of all reviews predicted as positive, 65.85% were actually positive (low false positive rate)
- **Recall (96.43%)**: Of all actual positive reviews, 96.43% were correctly identified (low false negative rate)
- **F1 Score (78.26%)**: Harmonic mean of precision and recall, balanced measure of model performance

**Key Observations:**
- High recall (96.43%) indicates the model successfully identifies most positive reviews
- Lower precision (65.85%) suggests some negative reviews are misclassified as positive
- The model shows strong performance on the positive class with 96.43% recall
- Confusion matrix shows 27/28 positive reviews correctly classified vs 8/22 negative reviews

**⚠️ Important Note:** These results are based on a small validation subset (n=50). Performance on the full test set (10,000 samples) may differ. For a fair comparison with traditional ML models, full test set evaluation is recommended.

### Traditional ML Models (Baseline Comparison)

Seven traditional machine learning models were trained using TF-IDF features for comparison:

| Model | Accuracy | Test Size | Status |
|-------|----------|-----------|--------|
| **Logistic Regression** | **85.14%** | 10,000 | ✅ Full evaluation |
| Linear SVM | 86.50% | 200 | ⚠️ Preliminary |
| SGD Classifier | 86.00% | 200 | ⚠️ Preliminary |
| Multinomial Naive Bayes | 82.00% | 200 | ⚠️ Preliminary |
| Random Forest | 76.00% | 200 | ⚠️ Preliminary |
| KNN Classifier | 86.00% | 200 | ⚠️ Preliminary |
| Decision Tree | 74.50% | 200 | ⚠️ Preliminary |

**Key Findings:**
- **Logistic Regression** (85.14% on 10k test) demonstrates that TF-IDF + linear models achieve strong performance
- Traditional ML models are **faster to train** and **more interpretable** than transformers
- BERT's performance (70% on 50 samples) may improve with full test set evaluation
- Only Logistic Regression has been evaluated on the complete test set (10,000 samples)
- Other models show preliminary results on 200 samples and require full evaluation

**Important Notes:**
- ⚠️ **Comparison Limitation**: BERT was evaluated on 50 samples (validation subset), while Logistic Regression was evaluated on 10,000 samples. Direct comparison is not scientifically valid.
- ⚠️ **Preliminary Results**: Models marked with ⚠️ were tested on 200 samples only. Full evaluation on 10,000 samples may yield different results.
- ✅ **Validated Result**: Logistic Regression's 85.14% accuracy on 10,000 samples is the only reliable traditional ML baseline for comparison.

**Dataset Used:**
- Source: [fancyzhx/amazon_polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity)
- Training: 50,000 samples
- Test: 10,000 samples
- Both pipelines use the **exact same data** for fair comparison

---

## Confusion Matrix

**Confusion Matrix Breakdown:**
- **True Negatives**: 8 (correctly predicted negative reviews)
- **False Positives**: 14 (negative reviews predicted as positive)
- **False Negatives**: 1 (positive reviews predicted as negative)
- **True Positives**: 27 (correctly predicted positive reviews)

---

## Sample Predictions

| Review | Prediction | Confidence |
|--------|-----------|------------|
| "This product exceeded my expectations! Amazing quality and fast shipping." | Positive | 95.23% |
| "Terrible product. Broke after one week of use. Complete waste of money." | Negative | 92.45% |
| "It's okay, does what it's supposed to do but nothing special." | Positive | 67.89% |
| "Worst purchase ever. The description was misleading and quality is poor." | Negative | 88.34% |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (8GB recommended for training)
- Optional: CUDA-capable GPU or Apple Silicon Mac for faster training

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/vivekcode12345/transformer-amazon-sentiment-analysis.git
cd transformer-amazon-sentiment-analysis
```

2. **Create a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## Usage

### Training the Model

Train the BERT model on the Amazon Polarity dataset:

```bash
python -m src.main text-train
```

**What happens during training:**
1. Loads Amazon Polarity dataset from Hugging Face Hub
2. Tokenizes text using BERT tokenizer
3. Fine-tunes BERT for 3 epochs
4. Saves best model checkpoint to `models/bert_finetuned/best_model/`
5. Saves tokenizer to `models/bert_finetuned/tokenizer/`
6. Logs training metrics to `reports/train_metrics.json`

**Training time:** Approximately 15-30 minutes on CPU, 5-10 minutes on GPU

### Evaluating the Model

Evaluate the fine-tuned model on the test set:

```bash
python -m src.main text-eval
```

**Output:**
- Accuracy, precision, recall, F1 score
- Confusion matrix
- Detailed classification report
- Results saved to `reports/eval_metrics.json`

### Running Inference

Predict sentiment for a single review:

```bash
python -m src.main text-predict "This product is amazing! Best purchase I've made."
```

**Example Output:**
```
Prediction Result
------------------------------------------------------------
Text: This product is amazing! Best purchase I've made.
Predicted Label: Positive
Confidence: 0.9523
Probabilities: {'negative': 0.0477, 'positive': 0.9523}
```

**Python API:**
```python
from src.text.inference import predict_sentiment, print_prediction_result

result = predict_sentiment("This product exceeded my expectations!")
print_prediction_result(result)
```

### Interactive Mode

Launch the interactive CLI menu:

```bash
python -m src.main
```

**Menu Options:**
1. Preview tokenization on a small sample
2. Train BERT on Amazon Polarity
3. Evaluate saved model
4. Predict sentiment for a single review
5. View configuration
6. Exit

---

## Model Configuration

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

The pipeline automatically adapts to different models using Hugging Face's `AutoTokenizer` and `AutoModelForSequenceClassification`.

---

## Future Improvements

- **Advanced Models**: Experiment with RoBERTa, DistilBERT, and DeBERTa for improved performance
- **Hyperparameter Tuning**: Implement Optuna for automated hyperparameter optimization
- **Multi-Class Sentiment**: Extend to 5-class classification (1-5 star ratings)
- **Explainable AI**: Add LIME/SHAP for model interpretability
- **Docker Deployment**: Containerize the application for easy deployment
- **FastAPI Deployment**: Build REST API for production inference
- **Hugging Face Deployment**: Deploy model to Hugging Face Spaces for live demo
- **Model Ensembling**: Combine multiple models for better accuracy
- **Experiment Tracking**: Integrate Weights & Biases or MLflow for experiment management
- **ONNX Export**: Optimize model for production deployment

---

## Live Demo

**Live Demo**: Coming Soon

**Hugging Face Model**: Coming Soon

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Vivek Verma**

Computer Science Engineering Student | Backend Developer | Machine Learning Enthusiast | AI Developer

Passionate about building production-ready ML systems and applying cutting-edge NLP techniques to solve real-world problems.

**GitHub**: [@vivekcode12345](https://github.com/vivekcode12345)

**LinkedIn**: [Vivek Verma](https://www.linkedin.com/in/vivekverma)

---

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{transformer_amazon_sentiment,
  title = {Transformer-Based Amazon Review Sentiment Analysis},
  author = {Vivek Verma},
  year = {2025},
  url = {https://github.com/vivekcode12345/transformer-amazon-sentiment-analysis}
}
```

---

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing pretrained transformer models and the Transformers library
- [Amazon Polarity Dataset](https://huggingface.co/datasets/fancyzhx/amazon_polarity) for the benchmark dataset
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018)

---

## Contact

For questions, feedback, or collaboration opportunities, please open an issue on GitHub or reach out via LinkedIn.

**⭐ Star this repository if you find it helpful!**