# Health Multimodal AI: Sentiment Analysis from Text & Images

This project implements research-grade **text-only and multimodal sentiment analysis** on product reviews and health-related content using BERT and pluggable image encoders.

## Quick Summary

### Text-Only Pipeline (Production-Ready)
- **Model**: BERT (bert-base-uncased)
- **Data**: Amazon Reviews Polarity dataset (streaming mode for full-dataset training)
- **Capabilities**: Predict sentiment (positive/negative) from review text alone
- **Optimization**: Memory-efficient streaming for Apple Silicon (M1, 8GB RAM)

### Multimodal Pipeline (Architecture Ready, Implementation Pending)
- **Components**: BERT text encoder + pluggable image encoder + reusable fusion layer
- **Design**: Pluggable architecture—easily swap image encoders (CLIP, BLIP, SigLIP) and fusion strategies (concatenation, attention, cross-modal)
- **Use Case**: Future extension for health applications combining images and text
- **Status**: Stub implementations with clear interfaces; awaiting image encoder specification

## Project Structure

```
.
├── src/
│   ├── config.py                    # Unified config for text and multimodal
│   ├── main.py                      # CLI orchestrator (text + multimodal dispatch)
│   │
│   ├── text/                        # Text-only BERT pipeline (PRODUCTION)
│   │   ├── preprocessing.py         # Data loading, tokenization, streaming
│   │   ├── model.py                 # BERT loading with gradient checkpointing
│   │   ├── train.py                 # Training pipeline (Trainer API)
│   │   ├── evaluation.py            # Model evaluation
│   │   ├── inference.py             # Single-text prediction
│   │   └── metrics.py               # Detailed metric computation
│   │
│   ├── image/                       # Image encoding module (STUB)
│   │   ├── preprocessing.py         # Image loading, resizing, normalization
│   │   └── encoder.py               # Pluggable encoder interface (CLIP, BLIP, etc.)
│   │
│   ├── multimodal/                  # Image+text fusion module (STUB)
│   │   ├── fusion.py                # Pluggable fusion strategies
│   │   ├── model.py                 # Multimodal model architecture
│   │   ├── train.py                 # Multimodal training pipeline
│   │   ├── evaluate.py              # Multimodal evaluation
│   │   └── inference.py             # Image+text prediction
│   │
│   ├── dataset_loaders/             # Dataset utilities (PLACEHOLDER)
│   └── utils/                       # Common utilities (PLACEHOLDER)
│
├── data/
│   ├── README.md                    # Data folder documentation
│   ├── downloads/                   # Raw downloads (not committed)
│   ├── cache/                       # HF Datasets cache
│   └── processed/                   # Processed datasets
│
├── models/
│   ├── bert_finetuned/              # Text-only BERT artifacts
│   │   ├── best_model/              # Best checkpoint
│   │   ├── tokenizer/               # Saved tokenizer
│   │   └── checkpoints/             # Training checkpoints
│   │
│   ├── multimodal_finetuned/        # Multimodal model artifacts (future)
│   │   ├── best_model/
│   │   └── checkpoints/
│   │
│   └── README.md                    # Model folder documentation
│
├── logs/
│   ├── bert_trainer_logs/           # BERT training logs
│   └── multimodal_trainer_logs/     # Multimodal training logs (future)
│
├── reports/
│   ├── text_train_metrics.json      # BERT training metrics
│   ├── text_eval_metrics.json       # BERT evaluation metrics
│   ├── multimodal_train_metrics.json # Multimodal metrics (future)
│   └── multimodal_eval_metrics.json  # Multimodal evaluation (future)
│
├── requirements.txt
├── README.md (this file)
└── LICENSE
```

## Getting Started

### 1. Environment Setup

Install Python dependencies:

```bash
# Using the configured virtual environment
"/Users/vivekverma/MEGA downloads/Amazon Research/.venv/bin/python" -m pip install -r requirements.txt
```

Or create a new environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Text-Only Training (BERT on Amazon Reviews)

#### Quick Preview (200 training samples)
```bash
python src/main.py preview
```

#### Full Training (entire Amazon Polarity dataset, streaming mode)
```bash
python src/main.py text-train
```

Or interactively:
```bash
python src/main.py
# Select option "2) Train BERT on Amazon Polarity"
```

#### Evaluate Saved Model
```bash
python src/main.py text-eval
```

#### Predict on Custom Text
```bash
python src/main.py text-predict "This product is fantastic!"
```

### 3. Interactive Menu

```bash
python src/main.py
```

Provides options to:
- Preview tokenization
- Train text-only model
- Evaluate saved model
- Predict single review
- View configuration
- Multimodal options (stubs)

## Architecture Design

### Design Principle 1: Modular Separation
Each component (data, model, training, evaluation, inference) is independent and testable.

```
Text-Only Pipeline:
  Data → Tokenize → Model Load → Train → Evaluate → Inference

Multimodal Pipeline (extensible):
  Image + Text → Encode separately → Fuse → Model → Train → Evaluate → Inference
```

### Design Principle 2: Pluggable Encoders
Image encoders are abstract interfaces—swap between CLIP, BLIP, SigLIP without pipeline changes.

```python
# src/image/encoder.py
class ImageEncoder(ABC):
    def encode(self, images) -> Embeddings: ...
    def get_embedding_dim() -> int: ...

# Users can plug in any encoder
get_encoder("clip")      # OpenAI CLIP
get_encoder("blip")      # Salesforce BLIP
get_encoder("siglip")    # Google SigLIP
```

### Design Principle 3: Pluggable Fusion Strategies
Combine image + text embeddings flexibly.

```python
# src/multimodal/fusion.py
class FusionLayer(ABC):
    def forward(text_embed, image_embed) -> Fused: ...

# Easily switch strategies
get_fusion_layer("concat")      # Concatenate embeddings
get_fusion_layer("attention")   # Cross-modal attention
get_fusion_layer("cross_modal") # Gated cross-modal fusion
```

### Design Principle 4: Unified Configuration
Centralized hyperparameter management for both pipelines.

```python
# src/config.py
TEXT_CONFIG         # BERT-specific parameters
MULTIMODAL_CONFIG   # Multimodal training parameters
PATH_CONFIG         # Artifact paths (text + multimodal)
RUNTIME_CONFIG      # Device, MPS, logging settings
```

## Key Features

### ✅ Text-Only Pipeline (Implemented)
- **Streaming Dataset Loading**: Full Amazon Polarity dataset without materializing entire splits
- **Memory Optimization**: Gradient checkpointing, batch accumulation, KV cache disabled for M1 Macs (8GB RAM)
- **Research-Grade Metrics**: Accuracy, precision, recall, F1, confusion matrix, classification report
- **Best Model Selection**: Early stopping, metric tracking, automatic checkpoint management
- **Flexible Training Modes**: Sampled (quick testing) or full-dataset (research)

### 🔄 Multimodal Pipeline (Architecture Ready)
- **Pluggable Image Encoders**: Abstract interface for CLIP, BLIP, SigLIP, custom models
- **Reusable Fusion Strategies**: Concatenation, attention, cross-modal (easily extensible)
- **Consistent API**: Mirrors text-only pipeline for easy learning and maintenance
- **Prepared Paths**: Directory structure and configuration ready for image+text datasets

### 🎯 Future Extensions
- Image encoder implementation (CLIP, BLIP, or user-specified)
- Multimodal dataset loader
- Multimodal training pipeline
- Fine-tuned multimodal models for health applications

## Configuration

### Text Training Parameters
Edit `src/config.py`:

```python
@dataclass(frozen=True)
class TextTrainingConfig:
    model_name: str = "bert-base-uncased"
    batch_size: int = 8                    # M1 optimization
    learning_rate: float = 2e-5
    epochs: int = 3
    gradient_accumulation_steps: int = 2   # M1 optimization
    max_length: int = 128
    # ... more parameters
```

### Multimodal Configuration (Stub)
```python
@dataclass(frozen=True)
class MultimodalTrainingConfig:
    image_encoder_name: str = "clip"       # Pluggable: "clip", "blip", "siglip"
    fusion_strategy: str = "concat"        # Pluggable: "concat", "attention", "cross_modal"
    # ... more parameters
```

## Performance (Text-Only on Amazon Reviews)

**Smoke Test Results** (200 train / 50 test samples):
- Accuracy: 70%
- Precision: 65.85%
- Recall: 96.43%
- F1: 78.26%

**Full-Dataset Training** (1.4M samples, streaming mode):
- Successfully runs on M1 MacBook Air with 8GB RAM
- Gradient checkpointing reduces peak memory to ~6GB
- Training time: ~10-15 minutes per epoch

## Development Roadmap

### Phase 1-4: ✅ Complete
- Sentiment classification on Amazon Polarity (text-only)
- Memory-efficient M1 optimization
- Production-ready code structure

### Phase 5: 🟡 In Progress
- **Completed**: Modular architecture refactoring, text isolation, stub creation
- **Pending**: Image encoder selection and integration

### Phase 6: ⏳ Future
- Multimodal training pipeline implementation
- Health dataset integration
- Hyperparameter tuning for multimodal fusion
- Model serving and deployment

## Troubleshooting

### Memory Issues on M1
If you encounter OOM errors:
1. Reduce `batch_size` (current: 8, try: 4)
2. Reduce `gradient_accumulation_steps` (current: 2, try: 1)
3. Reduce `validation_materialize_size` in config

### Dataset Download Timeouts
The streaming mode gracefully handles network issues:
- Automatic retries configured
- Falls back between dataset sources if primary fails
- See `src/text/preprocessing.py` for details

### Model Not Found Error
Ensure you've run training first:
```bash
python src/main.py text-train
```

## Contributing

This is a research project. Contributions welcome:
- New image encoders in `src/image/encoder.py`
- Additional fusion strategies in `src/multimodal/fusion.py`
- New datasets in `src/datasets/`
- Performance improvements

## License

MIT License. See LICENSE file.

## Citation

If you use this code in research, please cite:

```bibtex
@software{health_multimodal_ai_2024,
  title = {Health Multimodal AI: Text and Image Sentiment Analysis},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-repo}
}
```

## Contact

For questions or suggestions: [your contact info]


## Current Starter File

Run the first working script with:

```bash
"/Users/vivekverma/MEGA downloads/Amazon Research/.venv/bin/python" src/main.py
```

This script creates a tiny demo dataset if no real dataset is present, so you can test the pipeline immediately.
