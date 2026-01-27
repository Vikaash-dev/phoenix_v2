"""
NeuroKAN Architecture for Phoenix Protocol (v2.0)
Reinvents the classification head using Kolmogorov-Arnold Networks (KAN).
Combines Dynamic Snake Convolutions (Backbone) with KAN Layers (Head).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.dynamic_snake_conv import SnakeConvBlock
from models.coordinate_attention import CoordinateAttentionBlock
from models.sevector_attention import SEVectorBlock
from models.kan_layer import KANDense
import config

class NeuroKANModel:
    """
    NeuroKAN: Hybrid architecture combining Snake Conv backbone with KAN classification head.
    
    Backbone: Dynamic Snake Convolutions + Coordinate/SEVector Attention
    Head: Kolmogorov-Arnold Network (Learnable Activations)
    
    Why KAN Head?
    - Standard MLPs use fixed activations (ReLU) on nodes.
    - KANs use learnable activations (B-Splines) on edges.
    - Result: Higher expressivity with fewer parameters, critical for medical decision making.
    """
    
    @staticmethod
    def create_model(
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
        num_classes=config.NUM_CLASSES,
        dropout_rate=0.3,
        use_kan=True,
        attention_type='coordinate' # 'coordinate' or 'se'
    ):
        inputs = layers.Input(shape=input_shape, name='input')
        
        # --- Backbone (Feature Extraction) ---
        
        # Stem
        x = layers.Conv2D(32, 3, strides=2, padding='same', name='stem_conv')(inputs)
        x = layers.BatchNormalization(name='stem_bn')(x)
        x = layers.Activation('relu', name='stem_act')(x)
        
        # Stage 1
        x = SnakeConvBlock(64, kernel_size=3, dropout_rate=dropout_rate, name='snake1')(x)
        if attention_type == 'coordinate':
            x = CoordinateAttentionBlock(64, name='ca1')(x)
        else:
            x = SEVectorBlock(64, name='se1')(x)
        x = layers.MaxPooling2D(2, name='pool1')(x)
        
        # Stage 2
        x = SnakeConvBlock(128, kernel_size=3, dropout_rate=dropout_rate, name='snake2')(x)
        if attention_type == 'coordinate':
            x = CoordinateAttentionBlock(128, name='ca2')(x)
        else:
            x = SEVectorBlock(128, name='se2')(x)
        x = layers.MaxPooling2D(2, name='pool2')(x)
        
        # Stage 3
        x = SnakeConvBlock(256, kernel_size=3, dropout_rate=dropout_rate, name='snake3')(x)
        if attention_type == 'coordinate':
            x = CoordinateAttentionBlock(256, name='ca3')(x)
        else:
            x = SEVectorBlock(256, name='se3')(x)
        x = layers.MaxPooling2D(2, name='pool3')(x)
        
        # Stage 4 (Deepest)
        x = SnakeConvBlock(512, kernel_size=3, dropout_rate=dropout_rate, name='snake4')(x)
        if attention_type == 'coordinate':
            x = CoordinateAttentionBlock(512, name='ca4')(x)
        else:
            x = SEVectorBlock(512, name='se4')(x)
        x = layers.MaxPooling2D(2, name='pool4')(x)
        
        # --- Head (Classification) ---
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        
        if use_kan:
            # NeuroKAN Head
            # Replace standard Dense layers with KAN layers
            x = KANDense(256, grid_size=5, spline_order=3, name='kan_fc1')(x)
            x = layers.Dropout(dropout_rate, name='kan_drop1')(x)
            
            x = KANDense(128, grid_size=5, spline_order=3, name='kan_fc2')(x)
            x = layers.Dropout(dropout_rate, name='kan_drop2')(x)
        else:
            # Standard MLP Head (NeuroSnake v1)
            x = layers.Dense(256, activation='relu', name='fc1')(x)
            x = layers.BatchNormalization(name='bn1')(x)
            x = layers.Dropout(dropout_rate)(x)
            
            x = layers.Dense(128, activation='relu', name='fc2')(x)
            x = layers.BatchNormalization(name='bn2')(x)
            x = layers.Dropout(dropout_rate)(x)
            
        # Output
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        
        model_name = f"NeuroKAN_{attention_type}" if use_kan else f"NeuroSnake_{attention_type}"
        model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
        
        return model

def create_neurokan_model(**kwargs):
    """Factory for NeuroKAN model."""
    return NeuroKANModel.create_model(use_kan=True, **kwargs)

if __name__ == "__main__":
    model = create_neurokan_model()
    model.summary()
    print(f"NeuroKAN created with {model.count_params():,} parameters.")
