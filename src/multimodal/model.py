"""Multimodal model architecture (stub).

Combines BERT text encoder + pluggable image encoder + fusion layer
for joint image+text sentiment analysis.

Currently a stub - core components (image encoder, fusion) need implementation first.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig


class MultimodalConfig(PretrainedConfig):
    """Configuration for multimodal model."""

    model_type = "multimodal"

    def __init__(
        self,
        text_encoder_name: str = "bert-base-uncased",
        image_encoder_name: str = "clip",
        fusion_strategy: str = "concat",
        text_hidden_dim: int = 768,
        image_hidden_dim: int = 512,
        fusion_hidden_dim: int = 512,
        num_labels: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_encoder_name = text_encoder_name
        self.image_encoder_name = image_encoder_name
        self.fusion_strategy = fusion_strategy
        self.text_hidden_dim = text_hidden_dim
        self.image_hidden_dim = image_hidden_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.num_labels = num_labels


class MultimodalModel(PreTrainedModel):
    """Multimodal model combining text and image encoders (stub).
    
    When fully implemented, will:
    1. Encode text using BERT
    2. Encode image using pluggable encoder (CLIP, BLIP, etc.)
    3. Fuse embeddings using pluggable fusion layer
    4. Classify (sentiment prediction)
    """

    config_class = MultimodalConfig

    def __init__(self, config: MultimodalConfig):
        super().__init__(config)
        self.config = config
        # Stub: in real implementation, would initialize:
        # - self.text_encoder (BERT)
        # - self.image_encoder (pluggable)
        # - self.fusion_layer (pluggable)
        # - self.classifier (final linear layer)

    def forward(
        self,
        input_ids=None,  # Text token IDs
        attention_mask=None,  # Text attention mask
        pixel_values=None,  # Image pixel values
        labels=None,
    ):
        """Forward pass for multimodal model (stub)."""
        raise NotImplementedError(
            "Multimodal model forward pass not yet implemented. "
            "Awaiting image encoder and fusion layer implementation."
        )


if __name__ == "__main__":
    print("Multimodal model stub. To be implemented when components are ready.")
