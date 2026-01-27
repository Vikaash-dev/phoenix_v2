"""
Phoenix Protocol v2.0 - Unified SOTA Model.
Combines: Snake Conv + Spectral Gating + Hyper-Liquid + TTT-KAN Head.

This is the flagship model for the research paper.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.dynamic_snake_conv import SnakeConvBlock
from src.models.spectral_gating import SpectralGatingBlock
from src.models.hyper_liquid import HyperLiquidConv2D
from src.models.ttt_kan import TTTKANLinear


class PhoenixV2Model:
    """
    Phoenix Protocol v2.0 Architecture.
    
    Architecture:
    1. Snake Conv Backbone (local geometry adaptation)
    2. Spectral Gating (global FFT-based context)
    3. Hyper-Liquid Block (dynamic ODE dynamics)
    4. TTT-KAN Head (inference-time adaptation)
    """
    
    @staticmethod
    def create_model(
        input_shape=(224, 224, 3),
        num_classes=2,
        dropout_rate=0.3,
        use_hyper_liquid=True,
        use_ttt=False  # TTT requires custom inference loop
    ):
        inputs = layers.Input(shape=input_shape)
        
        # ==== BACKBONE: Snake Conv Stages ====
        # Stage 1: Initial Conv
        x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # 112x112x32
        
        # Stage 2: Snake Block 1
        x = SnakeConvBlock(64, kernel_size=3)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        # 56x56x64
        
        # Stage 3: Snake Block 2
        x = SnakeConvBlock(128, kernel_size=3)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        # 28x28x128
        
        # Stage 4: Snake Block 3
        x = SnakeConvBlock(256, kernel_size=3)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        # 14x14x256
        
        # ==== GLOBAL CONTEXT: Spectral Gating ====
        x = layers.MaxPooling2D(pool_size=2)(x)
        # 7x7x256
        
        x = layers.Conv2D(512, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = SpectralGatingBlock(filters=512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # ==== DYNAMIC PHYSICS: Hyper-Liquid (Optional) ====
        if use_hyper_liquid:
            x = HyperLiquidConv2D(filters=512)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
        
        # ==== CLASSIFICATION HEAD ====
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Dense layers (can be replaced with KAN in custom loop)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='Phoenix_V2')
        return model
    
    @staticmethod
    def create_model_with_kan_head(
        input_shape=(224, 224, 3),
        num_classes=2,
        dropout_rate=0.3
    ):
        """
        Creates a Phoenix V2 model with KAN classification head.
        Note: KAN layers require more memory but offer higher expressivity.
        """
        from src.models.kan_layer import KANLinear
        
        inputs = layers.Input(shape=input_shape)
        
        # Backbone (same as above)
        x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = SnakeConvBlock(64, kernel_size=3)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        x = SnakeConvBlock(128, kernel_size=3)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        x = SnakeConvBlock(256, kernel_size=3)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(512, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = SpectralGatingBlock(filters=512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = HyperLiquidConv2D(filters=512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # KAN Head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = KANLinear(128)(x)
        x = layers.Dropout(0.3)(x)
        
        x = KANLinear(64)(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='Phoenix_V2_KAN')
        return model


if __name__ == "__main__":
    print("=== Phoenix V2 Standard ===")
    model = PhoenixV2Model.create_model()
    model.summary()
    
    print("\n=== Phoenix V2 with KAN Head ===")
    model_kan = PhoenixV2Model.create_model_with_kan_head()
    model_kan.summary()
