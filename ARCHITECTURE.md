# Health Multimodal AI - Architecture Documentation

## Overview

This document explains the modular architecture design for Health Multimodal AI sentiment analysis. The system is designed to support both text-only and multimodal (image+text) prediction while maintaining extensibility for future encoders and fusion strategies.

## Design Principles

### 1. **Modular Separation of Concerns**

Each component handles a single responsibility:

```
┌─────────────────────────────────────────────────────────┐
│  Main Orchestrator (main.py)                            │
│  - Routes between text-only and multimodal pipelines    │
│  - Dispatches CLI commands and interactive menus        │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
    ┌─────────┐      ┌──────────────┐
    │ TEXT    │      │ MULTIMODAL   │
    │ PIPELINE│      │ PIPELINE     │
    └─────────┘      │ (STUBS)      │
                     └──────────────┘
        │
        ├── Preprocessing (data loading, tokenization)
        ├── Model (encoder loading)
        ├── Training (Trainer API)
        ├── Evaluation (metrics)
        └── Inference (prediction)
```

### 2. **Pluggable Components**

Key components are abstract interfaces that can be replaced:

#### Image Encoders
```python
# Abstract interface
class ImageEncoder(ABC):
    def encode(images) -> Embeddings
    def get_embedding_dim() -> int

# Pluggable implementations
CLIPEncoder()      # OpenAI CLIP
BLIPEncoder()      # Salesforce BLIP
SigLIPEncoder()    # Google SigLIP
CustomEncoder()    # User-defined
```

#### Fusion Strategies
```python
# Abstract interface
class FusionLayer(ABC):
    def forward(text_embed, image_embed) -> Fused
    def get_output_dim() -> int

# Pluggable implementations
ConcatenationFusion()  # Simple concatenation
AttentionFusion()      # Cross-modal attention
CrossModalFusion()     # Gated cross-modal
```

### 3. **Unified Configuration**

Single source of truth for hyperparameters:

```python
# src/config.py contains:
- TextTrainingConfig      # BERT-specific parameters
- MultimodalTrainingConfig # Multimodal-specific parameters
- PathConfig              # Artifact paths (both pipelines)
- RuntimeConfig           # Device, logging settings
```

Avoids scattered magic numbers and enables easy experimentation.

### 4. **Clear Separation of Text and Multimodal**

```
src/text/          → Text-only BERT pipeline (STABLE, PRODUCTION)
src/image/         → Image encoding (STUB, awaiting implementation)
src/multimodal/    → Image+text fusion (STUB, awaiting implementation)
src/dataset_loaders/ → Dataset utilities (PLACEHOLDER)
src/utils/         → Shared utilities (PLACEHOLDER)
```

This structure allows:
- Independent development of each component
- Easy testing of individual modules
- Clear dependencies and import paths
- Future team collaboration

## Detailed Architecture

### Text-Only Pipeline (src/text/)

**Flow:**
```
Amazon Polarity Dataset
    ↓
Preprocessing (streaming)
    ↓
Tokenization (BERT)
    ↓
Training (Trainer API with early stopping)
    ↓
Evaluation (detailed metrics)
    ↓
Inference (single-text prediction)
```

**Files:**
- `preprocessing.py`: Data loading, streaming utilities, tokenization
- `model.py`: BERT loading with M1 optimization (gradient checkpointing)
- `train.py`: Training pipeline with Trainer, early stopping, checkpoints
- `evaluation.py`: Post-training model evaluation
- `inference.py`: Single-review prediction interface
- `metrics.py`: Detailed metric computation (sklearn-based)

**Key Features:**
- ✅ Streaming mode for full-dataset training without materialization
- ✅ Memory optimization for M1 Macs (gradient checkpointing, batch accumulation)
- ✅ Best-model selection with early stopping
- ✅ Research-grade metrics (confusion matrix, classification report)

### Image Module (src/image/) - STUB

**Designed Interface:**
```
Image → Load → Preprocess → Encode → Embeddings
                               ↑
                          [Pluggable Encoder]
```

**Files:**
- `preprocessing.py`: Image loading, resizing, normalization (STUB)
- `encoder.py`: Abstract ImageEncoder class with CLIP/BLIP/SigLIP stubs

**Extension Points:**
1. Implement `encode()` method for specific encoder
2. Return embeddings of known dimensionality
3. Update `get_embedding_dim()` to match

Example addition for CLIP:
```python
class CLIPEncoder(ImageEncoder):
    def encode(self, images):
        # Load CLIP model, process images, return embeddings
        return embeddings  # shape: [batch_size, 512]
    
    def get_embedding_dim(self):
        return 512
```

### Multimodal Module (src/multimodal/) - STUB

**Designed Architecture:**
```
      Text                Image
        ↓                  ↓
   [BERT Encoder]    [Image Encoder]
        ↓                  ↓
    Text Embeddings   Image Embeddings
         \                /
          [Fusion Layer]
              ↓
         Fused Embeddings
              ↓
        [Classification Head]
              ↓
          Prediction
```

**Files:**
- `fusion.py`: Abstract FusionLayer with concatenation/attention/cross-modal stubs
- `model.py`: MultimodalModel class combining encoders + fusion + classifier
- `train.py`: Training pipeline (mirrors src/text/train.py)
- `evaluate.py`: Evaluation utilities
- `inference.py`: Image+text prediction interface

**Extension Points:**
1. Choose/implement image encoder (CLIP, BLIP, etc.)
2. Choose/implement fusion strategy (concat, attention, cross-modal)
3. Specify multimodal dataset (image+text pairs)
4. Configure training parameters in MultimodalTrainingConfig
5. Run training pipeline

Example minimal multimodal setup:
```python
# 1. Select encoder
image_encoder = get_encoder("clip")

# 2. Select fusion
fusion_layer = get_fusion_layer("concat", 
    text_dim=768,      # BERT output
    image_dim=512,     # CLIP output
    output_dim=512
)

# 3. Create model
model = MultimodalModel(image_encoder, fusion_layer)

# 4. Train
run_multimodal_training(image_encoder_name="clip", fusion_strategy="concat")
```

## Data Flow

### Text-Only Training

```python
# src/main.py
python src/main.py text-train

# → main.py dispatches to:
from src.text.train import run_training

# → src/text/train.py:
1. load_tokenizer()              # Load BERT tokenizer
2. stream_tokenized_train()      # Stream full training data
3. materialize_validation_from_test()  # Small validation set
4. load_model()                  # Load BERT
5. Trainer.train()               # Fine-tune
6. Save best model + tokenizer
```

### Multimodal Training (Future)

```python
# src/main.py
python src/main.py multimodal-train

# → main.py dispatches to:
from src.multimodal.train import run_multimodal_training

# → src/multimodal/train.py (when implemented):
1. load_text_encoder()           # Reuse BERT from src/text/
2. load_image_encoder()          # From src/image/ (pluggable)
3. load_fusion_layer()           # From src/multimodal/ (pluggable)
4. load_multimodal_dataset()     # Image+text pairs
5. Trainer.train()               # Fine-tune multimodal model
6. Save best multimodal model
```

## Import Hierarchy

```
                    config.py
                       ↑
         ┌─────────────┼─────────────┐
         ↑             ↑             ↑
    text/          image/      multimodal/
    │              │           │
    ├─ preprocessing.py    encoder.py      fusion.py
    ├─ model.py                           model.py
    ├─ train.py                           train.py
    ├─ evaluation.py
    ├─ inference.py
    └─ metrics.py

main.py imports from:
  - config
  - text.preprocessing, text.train, text.evaluation, text.inference
  - (multimodal.train, multimodal.evaluate, etc. when implemented)
```

**Key Rule:** Lower-level modules NEVER import from higher-level modules.
- `text/train.py` → imports `config` ✓
- `main.py` → imports `text/train.py` ✓
- `text/train.py` → imports `main.py` ✗ (circular!)

## Memory Optimization Strategy (M1 MacBook, 8GB RAM)

### Problem
BERT fine-tuning requires large intermediate tensors during backpropagation.

### Solution
```python
# src/text/model.py
model.gradient_checkpointing_enable()  # Trade compute for memory
model.config.use_cache = False         # Don't cache KV matrices

# src/text/train.py
batch_size=8                           # Small batch
gradient_accumulation_steps=2          # Accumulate gradients
dataloader_num_workers=0               # No worker processes
```

### Result
Peak memory: ~6GB (within 8GB limit)
Training time: ~10-15 min/epoch (acceptable for research)

## Extensibility Checklist

To add a new image encoder:
- [ ] Inherit from `ImageEncoder` in `src/image/encoder.py`
- [ ] Implement `encode()` and `get_embedding_dim()`
- [ ] Add to `get_encoder()` factory function
- [ ] Update `MULTIMODAL_CONFIG.image_encoder_name` default (optional)

To add a new fusion strategy:
- [ ] Inherit from `FusionLayer` in `src/multimodal/fusion.py`
- [ ] Implement `forward()` and `get_output_dim()`
- [ ] Add to `get_fusion_layer()` factory function
- [ ] Update `MULTIMODAL_CONFIG.fusion_strategy` default (optional)

To add a new dataset:
- [ ] Create loader in `src/datasets/`
- [ ] Ensure compatibility with both pipelines
- [ ] Update dataset selection in config or CLI

## Testing Strategy

### Unit Tests (Future)
```python
# tests/test_text_preprocessing.py
def test_tokenization(): ...

# tests/test_image_encoder.py
def test_clip_encoder(): ...

# tests/test_multimodal_fusion.py
def test_attention_fusion(): ...
```

### Integration Tests (Future)
```python
# tests/test_end_to_end.py
def test_text_training_pipeline(): ...
def test_multimodal_training_pipeline(): ...
```

### Current Testing
- ✅ Smoke test: 200 train / 50 test samples verify full pipeline works
- ✅ Syntax validation: All Python files compile without errors
- ✅ Import validation: No circular imports

## Known Limitations & Future Work

### Current Limitations
- Image encoder not implemented (placeholder stubs)
- Multimodal training not implemented
- No distributed training support
- No GPU optimization (MPS-focused)
- No batch inference

### Future Enhancements
- [ ] Implement CLIP image encoder
- [ ] Implement multimodal training pipeline
- [ ] Add image augmentation
- [ ] Support additional models (DistilBERT, RoBERTa, etc.)
- [ ] Implement batch prediction
- [ ] Add model versioning and registry
- [ ] Create REST API for inference
- [ ] Add Docker containerization
- [ ] Implement cross-modal retrieval (image→text, text→image)

## Configuration Management

All settings are centralized in `src/config.py`:

```python
# Text-only hyperparameters
TEXT_CONFIG.learning_rate = 2e-5
TEXT_CONFIG.batch_size = 8
TEXT_CONFIG.epochs = 3

# Multimodal hyperparameters (future)
MULTIMODAL_CONFIG.image_encoder_name = "clip"
MULTIMODAL_CONFIG.fusion_strategy = "concat"

# Paths
PATH_CONFIG.text_best_model_dir
PATH_CONFIG.multimodal_best_model_dir

# Runtime
RUNTIME_CONFIG.use_mps_if_available = True
RUNTIME_CONFIG.enable_multimodal = False
```

This design allows:
- Easy hyperparameter tuning
- No magic numbers scattered in code
- Clear separation of concerns
- Version control friendly (changes visible at a glance)

## Backward Compatibility

Old imports still work via aliases in `config.py`:

```python
# Old code
from config import TRAINING_CONFIG, PATH_CONFIG
# → Resolves to: TEXT_CONFIG and PATH_CONFIG (via aliases)

# New code
from config import TEXT_CONFIG, MULTIMODAL_CONFIG, PATH_CONFIG
```

This ensures existing code doesn't break when refactoring.

## Deployment Considerations

### Development
```bash
python src/main.py text-train      # CLI
python src/main.py                 # Interactive menu
```

### Production (Future)
```bash
# API server
python -m src.serve:app --port 8000

# Batch processing
python -m src.batch_predict --input images/ --output results.json

# Model export
python -m src.export --format onnx --output model.onnx
```

## Conclusion

This architecture prioritizes:
1. **Clarity**: Each module has one responsibility
2. **Extensibility**: Pluggable encoders and fusion layers
3. **Memory Efficiency**: Optimized for M1 Macs
4. **Research Quality**: Detailed metrics, reproducibility
5. **Maintainability**: Unified configuration, clear paths

The design supports both current text-only research and future multimodal extensions without major refactoring.
