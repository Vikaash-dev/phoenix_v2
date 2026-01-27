"""
NeuroSnake-KAN Architecture.
Replaces the dense classification head of Spectral-Snake with a KAN (Kolmogorov-Arnold Network) Head.

Updates v4.1:
- Added explicit BatchNormalization/Clip before KAN layers to prevent grid extrapolation.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.models.neuro_mamba_spectral import NeuroSnakeSpectralModel, SpectralGatingBlock, SnakeConvBlock
from src.models.kan_layer import KANLinear

class NeuroSnakeKANModel:
    """
    NeuroSnake-KAN Model.
    """
    
    @staticmethod
    def create_model(
        input_shape=(224, 224, 3),
        num_classes=2,
        dropout_rate=0.3
    ):
        inputs = layers.Input(shape=input_shape)
        
        # --- Backbone (Same as NeuroSnake-Spectral) ---
        
        # Stage 1: Initial Conv
        x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Stage 2: Snake Block 1
        x = SnakeConvBlock(64, kernel_size=3)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        # Stage 3: Snake Block 2
        x = SnakeConvBlock(128, kernel_size=3)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        # Stage 4: Snake Block 3
        x = SnakeConvBlock(256, kernel_size=3)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        # Stage 5: Spectral Gating
        x = layers.MaxPooling2D(pool_size=2)(x) # 7x7
        
        x = layers.Conv2D(512, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = SpectralGatingBlock(filters=512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # --- Head (Replaced with KAN) ---
        
        x = layers.GlobalAveragePooling2D()(x) # (Batch, 512)
        x = layers.Dropout(dropout_rate)(x)
        
        # Instead of Dense -> ReLU -> Dense
        # We use KANLinear layers
        
        # PRE-KAN STABILIZATION
        # Critical Fix: Batch Norm enforces distribution centered at 0 with unit variance.
        # This keeps inputs within the effective range of B-splines [-1, 1] (mostly).
        x = layers.BatchNormalization(name='pre_kan_norm')(x)
        
        # Optional: Hard Clip for extreme safety (though BN is usually enough)
        # x = layers.Lambda(lambda t: tf.clip_by_value(t, -2.0, 2.0))(x)
        
        # KAN Layer 1
        x = KANLinear(64, grid_size=5, spline_order=3)(x)
        x = layers.LayerNormalization()(x) # KANs benefit from normalization
        
        # KAN Layer 2 (Output)
        # Note: KAN produces unbounded outputs usually, so we might not strictly need softmax in the layer itself
        # if using from_logits=True in loss. But for consistency:
        x = KANLinear(num_classes, grid_size=5, spline_order=3)(x)
        outputs = layers.Softmax()(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='NeuroSnake_KAN')
        return model

if __name__ == "__main__":
    model = NeuroSnakeKANModel.create_model()
    model.summary()
