"""
SEVector (Squeeze-and-Excitation Vector) Attention Module
Lightweight channel attention mechanism for efficient feature recalibration.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SEVectorBlock(layers.Layer):
    """
    Squeeze-and-Excitation Vector (SEVector) attention block.
    
    Provides adaptive channel-wise feature recalibration with minimal overhead.
    More efficient than standard SE blocks for medical imaging.
    
    Key features:
    - Global average pooling for spatial squeeze
    - Two FC layers with bottleneck for channel excitation
    - Sigmoid activation for channel-wise weights
    - Element-wise multiplication for recalibration
    
    Reference: 
    - Squeeze-and-Excitation Networks (SE-Net)
    - Adapted for lightweight medical imaging (SEVector variant)
    """
    
    def __init__(
        self,
        filters: int,
        reduction_ratio: int = 16,
        name: str = 'sevector',
        **kwargs
    ):
        """
        Initialize SEVector block.
        
        Args:
            filters: Number of input/output channels
            reduction_ratio: Channel reduction factor for bottleneck (default: 16)
            name: Layer name
        """
        super(SEVectorBlock, self).__init__(name=name, **kwargs)
        
        self.filters = filters
        self.reduction_ratio = reduction_ratio
        self.reduced_filters = max(filters // reduction_ratio, 8)  # Minimum 8 filters
        
        # Global average pooling (spatial squeeze)
        self.global_avg_pool = layers.GlobalAveragePooling2D(name=f'{name}_gap')
        
        # Bottleneck FC layers (channel excitation)
        self.fc1 = layers.Dense(
            self.reduced_filters,
            activation='relu',
            name=f'{name}_fc1',
            kernel_initializer='he_normal'
        )
        
        self.fc2 = layers.Dense(
            filters,
            activation='sigmoid',
            name=f'{name}_fc2',
            kernel_initializer='he_normal'
        )
        
        # Reshape layer to broadcast channel weights
        self.reshape = layers.Reshape((1, 1, filters), name=f'{name}_reshape')
        
        # Multiply layer for recalibration
        self.multiply = layers.Multiply(name=f'{name}_multiply')
    
    def call(self, inputs, training=None):
        """
        Forward pass through SEVector block.
        
        Args:
            inputs: Input feature map (batch, height, width, channels)
            training: Training mode flag
            
        Returns:
            Recalibrated feature map with same shape as input
        """
        # Squeeze: Global spatial information into channel descriptor
        squeeze = self.global_avg_pool(inputs)  # (batch, channels)
        
        # Excitation: Generate channel-wise attention weights
        excitation = self.fc1(squeeze)  # (batch, reduced_channels)
        excitation = self.fc2(excitation)  # (batch, channels)
        
        # Reshape to enable broadcasting
        attention_weights = self.reshape(excitation)  # (batch, 1, 1, channels)
        
        # Recalibration: Scale input features by attention weights
        output = self.multiply([inputs, attention_weights])
        
        return output
    
    def get_config(self):
        """Return layer configuration for serialization."""
        config = {
            'filters': self.filters,
            'reduction_ratio': self.reduction_ratio,
        }
        base_config = super(SEVectorBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        """Compute output shape (same as input)."""
        return input_shape


class SEVectorConvBlock(layers.Layer):
    """
    Convolutional block with integrated SEVector attention.
    
    Combines:
    - Standard/Depthwise Convolution
    - Batch Normalization
    - Activation
    - SEVector Attention
    - Optional Dropout
    """
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1,
        use_depthwise: bool = True,
        reduction_ratio: int = 16,
        dropout_rate: float = 0.0,
        name: str = 'sevector_conv_block',
        **kwargs
    ):
        """
        Initialize SEVector Conv Block.
        
        Args:
            filters: Number of output filters
            kernel_size: Convolution kernel size
            strides: Stride for convolution
            use_depthwise: Use depthwise separable conv (more efficient)
            reduction_ratio: SE reduction ratio
            dropout_rate: Dropout rate (0 = no dropout)
            name: Layer name
        """
        super(SEVectorConvBlock, self).__init__(name=name, **kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_depthwise = use_depthwise
        self.reduction_ratio = reduction_ratio
        self.dropout_rate = dropout_rate
        
        if use_depthwise:
            # Depthwise separable convolution (MobileNet-style)
            self.conv1 = layers.DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                name=f'{name}_dwconv'
            )
            self.conv2 = layers.Conv2D(
                filters=filters,
                kernel_size=1,
                padding='same',
                name=f'{name}_pwconv'
            )
        else:
            # Standard convolution
            self.conv1 = layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                name=f'{name}_conv'
            )
            self.conv2 = None
        
        # Batch normalization
        self.bn = layers.BatchNormalization(name=f'{name}_bn')
        
        # Activation
        self.activation = layers.Activation('relu', name=f'{name}_relu')
        
        # SEVector attention
        self.se_block = SEVectorBlock(
            filters=filters,
            reduction_ratio=reduction_ratio,
            name=f'{name}_sevector'
        )
        
        # Optional dropout
        if dropout_rate > 0:
            self.dropout = layers.Dropout(dropout_rate, name=f'{name}_dropout')
        else:
            self.dropout = None
    
    def call(self, inputs, training=None):
        """Forward pass."""
        # Convolution
        x = self.conv1(inputs)
        if self.conv2 is not None:
            x = self.conv2(x)
        
        # Normalization and activation
        x = self.bn(x, training=training)
        x = self.activation(x)
        
        # SEVector attention
        x = self.se_block(x, training=training)
        
        # Optional dropout
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        
        return x
    
    def get_config(self):
        """Return layer configuration."""
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'use_depthwise': self.use_depthwise,
            'reduction_ratio': self.reduction_ratio,
            'dropout_rate': self.dropout_rate,
        }
        base_config = super(SEVectorConvBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def create_sevector_model_example():
    """
    Example: Create a simple CNN with SEVector attention.
    Demonstrates usage for brain tumor classification.
    """
    inputs = layers.Input(shape=(224, 224, 3), name='input')
    
    # Initial convolution
    x = layers.Conv2D(32, 3, strides=2, padding='same', name='stem')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation('relu', name='stem_relu')(x)
    
    # SEVector Conv Blocks
    x = SEVectorConvBlock(64, strides=1, name='sevector_block1')(x)
    x = layers.MaxPooling2D(2, name='pool1')(x)
    
    x = SEVectorConvBlock(128, strides=1, name='sevector_block2')(x)
    x = layers.MaxPooling2D(2, name='pool2')(x)
    
    x = SEVectorConvBlock(256, strides=1, name='sevector_block3')(x)
    x = layers.MaxPooling2D(2, name='pool3')(x)
    
    # Global pooling and classification
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(2, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='SEVector_CNN')
    return model


if __name__ == "__main__":
    # Test SEVector block
    print("Testing SEVector Block...")
    sevector = SEVectorBlock(filters=128, reduction_ratio=16)
    test_input = tf.random.normal((2, 56, 56, 128))
    output = sevector(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ SEVector block working correctly")
    
    # Test SEVector Conv Block
    print("\nTesting SEVector Conv Block...")
    sevector_conv = SEVectorConvBlock(filters=256, kernel_size=3, strides=1)
    test_input2 = tf.random.normal((2, 56, 56, 128))
    output2 = sevector_conv(test_input2)
    print(f"Input shape: {test_input2.shape}")
    print(f"Output shape: {output2.shape}")
    print("✓ SEVector Conv block working correctly")
    
    # Test example model
    print("\nCreating example SEVector model...")
    model = create_sevector_model_example()
    print(f"Model parameters: {model.count_params():,}")
    print("✓ Example model created successfully")
