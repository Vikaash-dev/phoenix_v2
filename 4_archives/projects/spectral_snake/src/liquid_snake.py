"""
Liquid-Snake Architecture.
Combines Dynamic Snake Convolutions with Liquid Time-Constant (LTC) layers.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.models.neuro_mamba_spectral import SpectralGatingBlock
from src.models.dynamic_snake_conv import SnakeConvBlock
from src.models.liquid_layer import LiquidConv2D
from src.models.efficient_liquid import EfficientLiquidConv2D

class NeuroSnakeLiquidModel:
    """
    NeuroSnake-Liquid Model.
    Supports both standard and efficient Liquid layers.
    """
    
    @staticmethod
    def create_model(
        input_shape=(224, 224, 3),
        num_classes=2,
        dropout_rate=0.3,
        efficient=True # Default to efficient implementation
    ):
        inputs = layers.Input(shape=input_shape)
        
        # Select Liquid Layer type
        LiquidLayer = EfficientLiquidConv2D if efficient else LiquidConv2D
        
        # Stage 1: Initial Conv
        x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Stage 2: Liquid-Snake Block 1
        x = SnakeConvBlock(64, kernel_size=3)(x)
        x = LiquidLayer(64, time_step=0.1, unfold_steps=2)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        # Stage 3: Liquid-Snake Block 2
        x = SnakeConvBlock(128, kernel_size=3)(x)
        x = LiquidLayer(128, time_step=0.1, unfold_steps=2)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        # Stage 4: Liquid-Snake Block 3
        x = SnakeConvBlock(256, kernel_size=3)(x)
        x = LiquidLayer(256, time_step=0.1, unfold_steps=2)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        # Stage 5: Spectral Gating (Keep Global Context)
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        x = layers.Conv2D(512, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = SpectralGatingBlock(filters=512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Classification Head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='NeuroSnake_Liquid')
        return model

if __name__ == "__main__":
    model = NeuroSnakeLiquidModel.create_model()
    model.summary()
