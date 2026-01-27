"""
NeuroSnake-Hyper-Liquid Model.
Integrates HyperLiquidConv2D into the Phoenix backbone.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.models.dynamic_snake_conv import SnakeConvBlock
from src.models.hyper_liquid import HyperLiquidConv2D
from src.models.neuro_mamba_spectral import SpectralGatingBlock

class NeuroSnakeHyperLiquidModel:
    """
    NeuroSnake-Hyper-Liquid.
    """
    
    @staticmethod
    def create_model(
        input_shape=(224, 224, 3),
        num_classes=2,
        dropout_rate=0.3
    ):
        inputs = layers.Input(shape=input_shape)
        
        # Stage 1
        x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Stage 2: Snake + HyperLiquid
        x = SnakeConvBlock(64, kernel_size=3)(x)
        x = HyperLiquidConv2D(64, unfold_steps=2)(x) # Dynamic Dynamics!
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        # Stage 3
        x = SnakeConvBlock(128, kernel_size=3)(x)
        x = HyperLiquidConv2D(128, unfold_steps=2)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        # Stage 4
        x = SnakeConvBlock(256, kernel_size=3)(x)
        x = HyperLiquidConv2D(256, unfold_steps=2)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        # Stage 5: Spectral Global Context
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(512, 1, padding='same')(x)
        x = SpectralGatingBlock(512)(x)
        x = layers.Activation('relu')(x)
        
        # Head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='NeuroSnake_HyperLiquid')
        return model

if __name__ == "__main__":
    model = NeuroSnakeHyperLiquidModel.create_model()
    model.summary()
