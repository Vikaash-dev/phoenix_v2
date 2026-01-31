"""
NeuroSnake-Spectral: The Research Hypothesis Model
Combines Dynamic Snake Convolutions (Local Deformable) with Spectral Gating (Global Frequency).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.dynamic_snake_conv import SnakeConvBlock
from models.coordinate_attention import CoordinateAttentionBlock
from models.kan_layer import KANDense
from src.models.spectral_gating import SpectralGatingBlock
import config

class NeuroSnakeSpectralModel:
    
    @staticmethod
    def create_model(
        input_shape=(224, 224, 3),
        num_classes=4,
        dropout_rate=0.3,
        spectral_stage=3  # 3 or 4
    ):
        inputs = layers.Input(shape=input_shape, name='input')
        
        # Stem
        x = layers.Conv2D(32, 3, strides=2, padding='same', name='stem_conv')(inputs) # 112x112
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Stage 1: Snake (Local)
        x = SnakeConvBlock(64, kernel_size=3, dropout_rate=dropout_rate, name='snake1')(x)
        x = layers.MaxPooling2D(2, name='pool1')(x) # 56x56
        
        # Stage 2: Snake (Local)
        x = SnakeConvBlock(128, kernel_size=3, dropout_rate=dropout_rate, name='snake2')(x)
        x = layers.MaxPooling2D(2, name='pool2')(x) # 28x28
        
        # Stage 3
        if spectral_stage == 3:
            # Spectral Mixing at 28x28
            x = SpectralGatingBlock(channels=128, name='spectral_gating_s3')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(256, 1, padding='same')(x) # Upsample channels
        else:
            # Standard Snake
            x = SnakeConvBlock(256, kernel_size=3, dropout_rate=dropout_rate, name='snake3')(x)
            
        x = layers.MaxPooling2D(2, name='pool3')(x) # 14x14
        
        # Stage 4
        if spectral_stage == 4:
            # Spectral Mixing at 14x14
            # Input channels here are 256.
            # If stage 3 was Snake, output is 256 channels.
            # We process at 14x14.
            x = SpectralGatingBlock(channels=256, name='spectral_gating_s4')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(512, 1, padding='same')(x) # Upsample channels
        else:
            # Standard Snake or Refinement
            x = SnakeConvBlock(512, kernel_size=3, dropout_rate=dropout_rate, name='snake4')(x)
            
        x = layers.MaxPooling2D(2, name='pool4')(x) # 7x7
        
        # Head (NeuroKAN style)
        x = layers.GlobalAveragePooling2D()(x)
        
        # KAN Head
        x = KANDense(256, name='kan_fc1')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = KANDense(128, name='kan_fc2')(x)
        
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='NeuroSnake_Spectral')
        return model

def create_neurosnake_spectral(**kwargs):
    return NeuroSnakeSpectralModel.create_model(**kwargs)
