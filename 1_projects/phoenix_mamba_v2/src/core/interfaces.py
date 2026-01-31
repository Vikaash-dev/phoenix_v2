from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

class BasePreprocessor(ABC):
    """
    Interface for data cleaning and normalization strategies.

    Implementations should handle:
    - Normalization (Z-score, MinMax)
    - Bias field correction
    - Skull stripping
    - Intensity clipping
    """
    @abstractmethod
    def process(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process a single image.

        Args:
            image: Input image array (H, W, C) or (H, W)
            mask: Optional corresponding segmentation mask

        Returns:
            Processed image array
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration parameters for reproducibility."""
        pass

class BaseAugmentor(ABC):
    """
    Interface for data augmentation strategies.

    Implementations should handle:
    - Geometric transformations (rotate, flip, elastic)
    - Intensity transformations (noise, bias field)
    - Physics-informed MRI artifacts
    """
    @abstractmethod
    def augment(self, image: np.ndarray, label: Any = None) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
        """
        Apply augmentation to an image (and optionally label).

        Args:
            image: Input image array
            label: Optional label/mask to be augmented consistently

        Returns:
            Augmented image (and label if provided)
        """
        pass

class BaseModelAdapter(ABC):
    """
    Interface for Model adapters to ensure compatibility with the pipeline.
    Wraps TensorFlow/PyTorch/Mamba models.
    """
    @abstractmethod
    def build(self, input_shape: Tuple[int, ...], num_classes: int) -> Any:
        """Build and return the compiled model."""
        pass

    @abstractmethod
    def train(self, train_data, val_data, epochs: int, **kwargs) -> Dict[str, Any]:
        """Execute training loop."""
        pass
