"""Central configuration for text-only BERT sentiment analysis pipeline.

This module organizes hyperparameters, paths, and runtime settings for:
- Text-only BERT sentiment analysis (Amazon Reviews use case)

All experiment settings are kept in one place so training/evaluation/inference
modules stay clean and configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ============================================================================
# TEXT-ONLY CONFIGURATION (BERT Sentiment Analysis)
# ============================================================================

@dataclass(frozen=True)
class TextTrainingConfig:
    """Hyperparameters for text-only BERT fine-tuning."""

    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    random_seed: int = 42

    # Training hyperparameters
    learning_rate: float = 2e-5
    epochs: int = 3
    batch_size: int = 8
    max_length: int = 128
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 2

    # Dataset materialization (for streaming with small validation)
    validation_materialize_size: int = 1000
    evaluation_materialize_size: int = 5000

    # Runtime
    dataloader_num_workers: int = 0

    # Trainer strategy
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_steps: int = 1000
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True




# ============================================================================
# SHARED PATH CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths for artifacts, logs, and intermediate outputs.
    
    Supports both text-only and multimodal pipelines.
    """

    project_root: Path = Path(__file__).resolve().parent.parent

    # Base directories
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"
    reports_dir: Path = project_root / "reports"
    logs_dir: Path = project_root / "logs"

    # Text-only BERT pipeline artifacts
    text_output_dir: Path = models_dir / "bert_finetuned"
    text_logging_dir: Path = logs_dir / "bert_trainer_logs"
    text_best_model_dir: Path = text_output_dir / "best_model"
    text_tokenizer_dir: Path = text_output_dir / "tokenizer"
    text_checkpoints_dir: Path = text_output_dir / "checkpoints"
    text_train_metrics_path: Path = reports_dir / "text_train_metrics.json"
    text_eval_metrics_path: Path = reports_dir / "text_eval_metrics.json"


# ============================================================================
# RUNTIME CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime toggles and device selection."""

    use_mps_if_available: bool = True
    use_cpu_fallback: bool = True



# ============================================================================
# SINGLETON INSTANCES
# ============================================================================

TEXT_CONFIG = TextTrainingConfig()
PATH_CONFIG = PathConfig()
RUNTIME_CONFIG = RuntimeConfig()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_project_dirs() -> None:
    """Create required directories if they do not exist yet."""
    text_directories = [
        PATH_CONFIG.data_dir,
        PATH_CONFIG.models_dir,
        PATH_CONFIG.reports_dir,
        PATH_CONFIG.logs_dir,
        PATH_CONFIG.text_output_dir,
        PATH_CONFIG.text_logging_dir,
        PATH_CONFIG.text_best_model_dir,
        PATH_CONFIG.text_tokenizer_dir,
        PATH_CONFIG.text_checkpoints_dir,
    ]

    for directory in text_directories:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# BACKWARD COMPATIBILITY (for existing code)
# ============================================================================
# Aliases for code that imports TRAINING_CONFIG, PATH_CONFIG, RUNTIME_CONFIG directly
TRAINING_CONFIG = TEXT_CONFIG  # Backward compatibility alias
