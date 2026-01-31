"""
Spectral-Snake Architecture for Brain Tumor Detection.
Combines Dynamic Snake Convolutions (local geometry) with Spectral Gating (global frequency context).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from src.models.dynamic_snake_conv import SnakeConvBlock

class SpectralGatingBlock(layers.Layer):
    """
    Spectral Gating Block using Fourier Transform.
    Provides global receptive field via frequency domain mixing.
    """
    
    def __init__(self, filters, **kwargs):
        super(SpectralGatingBlock, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        # input_shape: (batch, height, width, channels)
        
        # We assume H, W are roughly fixed for the learnable weights
        # At Stage 5, H=W=7
        self.h_dim = 7
        self.w_dim = 7
        
        # Weights for (C, H, W_freq)
        # Transposed shape because we will process as (B, C, H, W)
        freq_w_dim = self.w_dim // 2 + 1
        
        w_init = tf.random_normal_initializer(mean=1.0, stddev=0.02)
        
        # Shape: (Filters, H, W_freq) to broadcast over Batch
        self.complex_weight_real = self.add_weight(
            name='complex_weight_real',
            shape=(self.filters, self.h_dim, freq_w_dim),
            initializer=w_init,
            trainable=True
        )
        self.complex_weight_imag = self.add_weight(
            name='complex_weight_imag',
            shape=(self.filters, self.h_dim, freq_w_dim),
            initializer='zeros',
            trainable=True
        )
        
        super(SpectralGatingBlock, self).build(input_shape)

    def call(self, x):
        # x: (B, H, W, C)
        
        # 1. Transpose to (B, C, H, W) for spatial FFT
        x_transposed = tf.transpose(x, perm=[0, 3, 1, 2])
        
        # Get dynamic shapes
        H = tf.shape(x_transposed)[2]
        W = tf.shape(x_transposed)[3]
        
        # 2. Real FFT2D -> (B, C, H, W/2+1)
        x_freq = tf.signal.rfft2d(x_transposed)
        
        # 3. Spectral Gating
        # Construct weight (C, H_fixed, W_fixed_freq)
        # Note: If input H,W vary, we should resize the weights or interpolation.
        # For this implementation we assume fixed size or broadcast if compatible.
        # But to be safe against small variations, let's assume valid broadcasting 
        # or strict 7x7 input. 
        # Since this is a prototype, strict matching is acceptable.
        
        weight = tf.complex(self.complex_weight_real, self.complex_weight_imag)
        
        # Expand dims for batch: (1, C, H, W_freq)
        weight = tf.expand_dims(weight, 0)
        
        # Element-wise multiplication
        # x_freq is (B, C, H, W_freq)
        # weight is (1, C, 7, 4)
        # TensorFlow will broadcast if H=7.
        x_gated_freq = x_freq * weight
        
        # 4. Inverse Real FFT2D -> (B, C, H, W)
        x_out_transposed = tf.signal.irfft2d(x_gated_freq, fft_length=[H, W])
        
        # 5. Transpose back -> (B, H, W, C)
        x_out = tf.transpose(x_out_transposed, perm=[0, 2, 3, 1])
        
        return x_out + x  # Residual connection

class NeuroSnakeSpectralModel:
    """
    NeuroSnake-Spectral Model.
    Replaces MobileViT with Spectral Gating Block.
    """
    
    @staticmethod
    def create_model(
        input_shape=(224, 224, 3),
        num_classes=2,
        dropout_rate=0.3
    ):
        inputs = layers.Input(shape=input_shape)
        
        # Stage 1: Initial Conv (Standard)
        x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # Shape: 112x112x32
        
        # Stage 2: Snake Block 1
        x = SnakeConvBlock(64, kernel_size=3)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        # Shape: 56x56x64
        
        # Stage 3: Snake Block 2
        x = SnakeConvBlock(128, kernel_size=3)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        # Shape: 28x28x128
        
        # Stage 4: Snake Block 3
        x = SnakeConvBlock(256, kernel_size=3)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        # Shape: 14x14x256
        
        # Stage 5: Deepest Layer - Spectral Gating
        # First reduce spatial dim to 7x7
        x = layers.MaxPooling2D(pool_size=2)(x)
        # Shape: 7x7x256
        
        # Project to higher dim
        x = layers.Conv2D(512, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # Shape: 7x7x512
        
        # Spectral Gating Block (Replaces MobileViT)
        x = SpectralGatingBlock(filters=512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Classification Head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='NeuroSnake_Spectral')
        return model

if __name__ == "__main__":
    model = NeuroSnakeSpectralModel.create_model()
    model.summary()
