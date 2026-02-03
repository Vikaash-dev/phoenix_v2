"""
Kolmogorov-Arnold Network (KAN) Layer for TensorFlow.
Based on the principle that multivariate functions can be represented as sums of univariate non-linear functions.

Source: Consolidated from jules_session_7226522193343951134
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class KANLinear(layers.Layer):
    """
    KAN Linear Layer.
    Replaces matrix multiplication w*x with learnable non-linear functions phi(x).
    phi(x) is modeled as a basis of B-splines.
    """
    
    def __init__(
        self,
        units,
        grid_size=5,
        spline_order=3,
        base_activation='silu',
        grid_range=[-1, 1],
        **kwargs
    ):
        super(KANLinear, self).__init__(**kwargs)
        self.units = units
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = keras.activations.get(base_activation)
        self.grid_range = grid_range

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Base weight (like residual connection w*x)
        self.base_weight = self.add_weight(
            name='base_weight',
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Spline weights
        num_coeffs = self.grid_size + self.spline_order
        
        self.spline_weight = self.add_weight(
            name='spline_weight',
            shape=(input_dim, self.units, num_coeffs),
            initializer=tf.random_normal_initializer(stddev=0.1),
            trainable=True
        )
        
        # Spline scaler
        self.spline_scaler = self.add_weight(
            name='spline_scaler',
            shape=(input_dim, self.units),
            initializer='ones',
            trainable=True
        )
        
        super(KANLinear, self).build(input_shape)

    def _compute_b_spline_basis(self, x):
        """Compute B-spline basis functions."""
        # Normalize input to grid range
        x_normalized = (x - self.grid_range[0]) / (self.grid_range[1] - self.grid_range[0])
        x_clamped = tf.clip_by_value(x_normalized, 0.0, 1.0)
        
        # RBF approximation of B-splines for efficiency
        num_coeffs = self.grid_size + self.spline_order
        centers = tf.linspace(0.0, 1.0, num_coeffs)
        width = 1.0 / self.grid_size
        
        # Expand dims for broadcasting: x (B, I) -> (B, I, 1), centers (C,) -> (1, 1, C)
        x_exp = tf.expand_dims(x_clamped, -1)
        centers_exp = tf.reshape(centers, (1, 1, -1))
        
        # RBF kernel
        basis = tf.exp(-0.5 * ((x_exp - centers_exp) / width) ** 2)
        
        return basis

    def call(self, x):
        # Base output (residual path)
        base_output = self.base_activation(tf.matmul(x, self.base_weight))
        
        # Spline basis
        spline_basis = self._compute_b_spline_basis(x)
        
        # Apply scaler to spline weights
        weighted_spline_weights = self.spline_weight * self.spline_scaler[:, :, tf.newaxis]
        spline_output = tf.einsum('bic,ioc->bo', spline_basis, weighted_spline_weights)
        
        return base_output + spline_output

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'grid_size': self.grid_size,
            'spline_order': self.spline_order,
            'grid_range': self.grid_range,
        })
        return config


if __name__ == "__main__":
    layer = KANLinear(32)
    x = tf.random.normal((10, 16))
    y = layer(x)
    print("Input:", x.shape)
    print("Output:", y.shape)
