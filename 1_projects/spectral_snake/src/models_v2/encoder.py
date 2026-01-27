"""
Encoder component for the Disentangled VAE.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ..models.spectral_snake_conv import SpectralGatingBlock, SnakeConvBlock

def build_encoder(input_shape=(224, 224, 3), dropout_rate=0.3):
    """Builds the NeuroSnake backbone encoder."""
    inputs = layers.Input(shape=input_shape)

    # Stages from NeuroSnake-Spectral
    x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = SnakeConvBlock(64, kernel_size=3)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = SnakeConvBlock(128, kernel_size=3)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = SnakeConvBlock(256, kernel_size=3)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.MaxPooling2D(pool_size=2)(x) # 7x7
    x = layers.Conv2D(512, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = SpectralGatingBlock(filters=512)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    return keras.Model(inputs=inputs, outputs=x, name='encoder_backbone')
