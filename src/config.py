"""Central configuration for BERT sentiment fine-tuning.

This module keeps all experiment and path settings in one place so the training,
evaluation, and inference modules stay clean and consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and runtime settings for BERT fine-tuning."""

    model_name: str = "bert-base-uncased"
    num_labels: int = 2
    random_seed: int = 42

    learning_rate: float = 2e-5
    epochs: int = 3
    batch_size: int = 8
    max_length: int = 128
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 2

    # When training on the full dataset we materialize only a small validation
    # subset to keep evaluation memory small. Default reduced to 1000 for safe runs.
    validation_materialize_size: int = 1000

    # For test/evaluation after training we may evaluate on a configurable
    # subset (or full test split in streaming batches). This controls a sane
    # default for materialized evaluation subsets to avoid OOM on low-RAM Macs.
    evaluation_materialize_size: int = 5000

    # DataLoader workers: keep low on macOS to avoid extra memory/process cost.
    dataloader_num_workers: int = 0

    evaluation_strategy: str = "epoch"
    save_strategy: str = "steps"
    save_steps: int = 1000
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True


@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths for artifacts, logs, and intermediate outputs."""

    project_root: Path = Path(__file__).resolve().parent.parent

    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"
    reports_dir: Path = project_root / "reports"
    logs_dir: Path = project_root / "logs"

    output_dir: Path = models_dir / "bert_finetuned"
    logging_dir: Path = logs_dir / "trainer_logs"
    best_model_dir: Path = output_dir / "best_model"
    tokenizer_dir: Path = output_dir / "tokenizer"
    checkpoints_dir: Path = output_dir / "checkpoints"

    train_metrics_path: Path = reports_dir / "train_metrics.json"
    eval_metrics_path: Path = reports_dir / "eval_metrics.json"


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime toggles and extension points for future phases."""

    use_mps_if_available: bool = True
    use_cpu_fallback: bool = True

    # Future extension: keep pipeline architecture open for multimodal sentiment.
    # Phase 3 stays text-only; this flag is intentionally off.
    enable_multimodal: bool = False


TRAINING_CONFIG = TrainingConfig()
PATH_CONFIG = PathConfig()
RUNTIME_CONFIG = RuntimeConfig()


def create_project_dirs() -> None:
    """Create required directories if they do not exist yet."""
    directories = [
        PATH_CONFIG.data_dir,
        PATH_CONFIG.models_dir,
        PATH_CONFIG.reports_dir,
        PATH_CONFIG.logs_dir,
        PATH_CONFIG.output_dir,
        PATH_CONFIG.logging_dir,
        PATH_CONFIG.best_model_dir,
        PATH_CONFIG.tokenizer_dir,
        PATH_CONFIG.checkpoints_dir,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
