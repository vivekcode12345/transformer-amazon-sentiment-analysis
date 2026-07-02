"""Multimodal training pipeline (stub).

Extends the text-only training logic to handle joint image+text data.
Reuses BERT tokenization and adds image encoder integration.

Currently a stub - to be filled when image encoder and fusion layer are ready.
"""

from __future__ import annotations


def run_multimodal_training(
    train_samples: int = 1000,
    test_samples: int = 200,
    image_encoder_name: str = "clip",
    fusion_strategy: str = "concat",
) -> dict:
    """Train multimodal model on image+text pairs.
    
    Args:
        train_samples: Number of training samples to use.
        test_samples: Number of test samples to use.
        image_encoder_name: Which image encoder to use.
        fusion_strategy: Which fusion strategy to use.
        
    Returns:
        Dictionary with training metrics.
    """
    raise NotImplementedError(
        "Multimodal training not yet implemented. "
        "Requires: image encoder implementation + fusion layer + multimodal dataset."
    )


if __name__ == "__main__":
    print("Multimodal training stub. Awaiting image encoder specification.")
