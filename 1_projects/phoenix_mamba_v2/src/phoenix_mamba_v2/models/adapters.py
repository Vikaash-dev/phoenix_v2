from typing import Any, Dict, Tuple
import tensorflow as tf
from ..core.interfaces import BaseModelAdapter

# Import legacy model factories
# We use relative imports to point to the migrated location: src/models/legacy/
try:
    from .legacy.neurosnake_model import (
        create_neurosnake_model,
        create_neurosnake_with_coordinate_attention,
        create_baseline_model
    )
except ImportError:
    # Fallback if relative imports fail or structure is slightly different in runtime
    import sys
    import os
    # Add the legacy models directory to path as a fallback
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'legacy')))
    from neurosnake_model import (
        create_neurosnake_model,
        create_neurosnake_with_coordinate_attention,
        create_baseline_model
    )

class NeuroSnakeAdapter(BaseModelAdapter):
    """
    Adapter for the legacy NeuroSnake TensorFlow/Keras models.
    """

    def __init__(self, variant: str = "standard", dropout_rate: float = 0.3):
        """
        Args:
            variant: "standard", "coordinate_attention", or "baseline"
            dropout_rate: Dropout rate for regularization
        """
        self.variant = variant
        self.dropout_rate = dropout_rate
        self.model = None

    def build(self, input_shape: Tuple[int, ...], num_classes: int) -> tf.keras.Model:
        """Build and return the compiled model."""
        if self.variant == "standard":
            self.model = create_neurosnake_model(
                input_shape=input_shape,
                num_classes=num_classes,
                dropout_rate=self.dropout_rate
            )
        elif self.variant == "coordinate_attention":
            self.model = create_neurosnake_with_coordinate_attention(
                input_shape=input_shape,
                num_classes=num_classes,
                dropout_rate=self.dropout_rate
            )
        elif self.variant == "baseline":
            self.model = create_baseline_model(
                input_shape=input_shape,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown model variant: {self.variant}")

        return self.model

    def train(self, train_data, val_data, epochs: int, **kwargs) -> Dict[str, Any]:
        """
        Execute training loop.

        Args:
            train_data: tf.data.Dataset or similar
            val_data: tf.data.Dataset or similar
            epochs: Number of epochs
            **kwargs: Additional arguments for model.fit()
        """
        if self.model is None:
            raise RuntimeError("Model must be built before training. Call build() first.")

        # Default callbacks if not provided
        callbacks = kwargs.pop('callbacks', [])

        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            **kwargs
        )

        return history.history
