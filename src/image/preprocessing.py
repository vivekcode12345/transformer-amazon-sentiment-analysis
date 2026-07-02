"""Image preprocessing module (stub for multimodal phase).

This module will handle image loading, resizing, normalization, and augmentation.

Currently a stub - to be implemented when image encoder is selected.
"""

from __future__ import annotations


class ImagePreprocessor:
    """Placeholder for image preprocessing pipeline.
    
    Will support resizing, normalization, and augmentation when implemented.
    """

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def preprocess(self, image):  # noqa: ARG002
        """Preprocess an image to model-ready format (stub)."""
        raise NotImplementedError(
            "Image preprocessing not yet implemented. "
            "Awaiting image encoder specification."
        )


if __name__ == "__main__":
    print("Image preprocessing stub. To be implemented in multimodal phase.")
