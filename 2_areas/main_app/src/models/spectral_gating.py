"""
Spectral Gating Block for global context mixing via FFT.

Source: Consolidated from jules_session_7226522193343951134
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.models.dynamic_snake_conv import SnakeConvBlock


class SpectralGatingBlock(layers.Layer):
    """
    Spectral Gating using Fourier Transform.
    Provides O(N log N) global receptive field.
    """
    
    def __init__(self, filters, **kwargs):
        super(SpectralGatingBlock, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        # Assume 7x7 spatial at deepest layer
        self.h_dim = 7
        self.w_dim = 7
        freq_w_dim = self.w_dim // 2 + 1
        
        w_init = tf.random_normal_initializer(mean=1.0, stddev=0.02)
        
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
        # x: (B, H, W, C) -> (B, C, H, W)
        x_transposed = tf.transpose(x, perm=[0, 3, 1, 2])
        
        # FFT
        x_complex = tf.cast(x_transposed, tf.complex64)
        x_fft = tf.signal.rfft2d(tf.math.real(x_complex))
        
        # Complex weight multiplication
        weight_complex = tf.complex(self.complex_weight_real, self.complex_weight_imag)
        
        # Resize weight to match FFT output
        target_h = tf.shape(x_fft)[2]
        target_w = tf.shape(x_fft)[3]
        weight_resized = tf.image.resize(
            tf.expand_dims(weight_complex, 0),
            [target_h, target_w],
            method='nearest'
        )[0]
        
        x_fft_weighted = x_fft * tf.cast(weight_resized, tf.complex64)
        
        # Inverse FFT
        x_ifft = tf.signal.irfft2d(x_fft_weighted)
        
        # Transpose back
        output = tf.transpose(x_ifft, perm=[0, 2, 3, 1])
        
        return output

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config


class NeuroSnakeSpectralModel:
    """NeuroSnake model with Spectral Gating instead of ViT."""
    
    @staticmethod
    def create_model(
        input_shape=(224, 224, 3),
        num_classes=2,
        dropout_rate=0.3
    ):
        inputs = layers.Input(shape=input_shape)
        
        # Stage 1: Initial Conv
        x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Stage 2-4: Snake Blocks
        x = SnakeConvBlock(64, kernel_size=3)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        x = SnakeConvBlock(128, kernel_size=3)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        x = SnakeConvBlock(256, kernel_size=3)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        # Stage 5: Spectral Gating
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(512, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = SpectralGatingBlock(filters=512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        return keras.Model(inputs=inputs, outputs=outputs, name='NeuroSnake_Spectral')


if __name__ == "__main__":
    model = NeuroSnakeSpectralModel.create_model()
    model.summary()
