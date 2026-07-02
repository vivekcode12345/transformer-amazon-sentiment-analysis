"""Pluggable image encoder interface (stub for multimodal phase).

This module defines a base class for image encoders to ensure consistent
interface across different encoder implementations (CLIP, BLIP, SigLIP, etc.).

Design principle: Encoders can be swapped without changing the rest of the
multimodal pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ImageEncoder(ABC):
    """Abstract base class for image encoders.
    
    Any concrete encoder (CLIP, BLIP, etc.) should implement this interface
    to ensure compatibility with the multimodal model.
    """

    @abstractmethod
    def encode(self, images: Any) -> Any:
        """Encode images to embedding vectors.
        
        Args:
            images: Input images (format depends on specific encoder).
            
        Returns:
            Image embeddings (typically shape [batch_size, embedding_dim]).
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the dimension of image embeddings."""
        pass


class CLIPEncoder(ImageEncoder):
    """CLIP image encoder (stub).
    
    When implemented, will use OpenAI's CLIP model or open-source variant
    (e.g., openai/clip-vit-base-patch32).
    """

    def encode(self, images: Any) -> Any:
        """Encode images using CLIP (stub)."""
        raise NotImplementedError("CLIP encoder not yet implemented.")

    def get_embedding_dim(self) -> int:
        """Return CLIP embedding dimension."""
        return 512  # Standard CLIP-ViT dimension


class BLIPEncoder(ImageEncoder):
    """BLIP image encoder (stub).
    
    When implemented, will use Salesforce's BLIP model for unified vision-language
    understanding.
    """

    def encode(self, images: Any) -> Any:
        """Encode images using BLIP (stub)."""
        raise NotImplementedError("BLIP encoder not yet implemented.")

    def get_embedding_dim(self) -> int:
        """Return BLIP embedding dimension."""
        return 256  # Standard BLIP dimension


class SigLIPEncoder(ImageEncoder):
    """SigLIP image encoder (stub).
    
    When implemented, will use Google's Sigmoid Loss for Language Image Pre-training.
    """

    def encode(self, images: Any) -> Any:
        """Encode images using SigLIP (stub)."""
        raise NotImplementedError("SigLIP encoder not yet implemented.")

    def get_embedding_dim(self) -> int:
        """Return SigLIP embedding dimension."""
        return 768  # Standard SigLIP dimension


def get_encoder(encoder_name: str) -> ImageEncoder:
    """Factory function to retrieve an encoder by name.
    
    Args:
        encoder_name: Name of the encoder ("clip", "blip", "siglip", etc.).
        
    Returns:
        An ImageEncoder instance.
        
    Raises:
        ValueError: If encoder name is not recognized.
    """
    encoders = {
        "clip": CLIPEncoder,
        "blip": BLIPEncoder,
        "siglip": SigLIPEncoder,
    }

    if encoder_name.lower() not in encoders:
        raise ValueError(
            f"Unknown encoder: {encoder_name}. "
            f"Supported: {list(encoders.keys())}"
        )

    return encoders[encoder_name.lower()]()


if __name__ == "__main__":
    print("Image encoder stub. To be implemented when encoder is selected.")
