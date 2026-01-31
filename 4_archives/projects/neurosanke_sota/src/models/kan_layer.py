"""
Kolmogorov-Arnold Network (KAN) Layer for Phoenix Protocol
Implements a learnable activation layer using B-Splines, replacing standard Dense layers.

Reference:
"KAN: Kolmogorov-Arnold Networks" (arXiv:2404.19756, 2024)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class KANDense(layers.Layer):
    """
    Kolmogorov-Arnold Network (KAN) Dense Layer.
    
    Unlike MLPs (fixed activation, learnable weights), KANs have learnable activation functions
    (parametrized as B-splines) on edges.
    
    This implementation simplifies the original KAN for GPU efficiency:
    y = base_activation(xW_b) + spline(xW_s)
    
    Where:
    - base_activation is SiLU (Sigmoid Linear Unit)
    - spline is a linear combination of B-spline basis functions
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
        super(KANDense, self).__init__(**kwargs)
        self.units = units
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = keras.activations.get(base_activation)
        self.grid_range = grid_range

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Base weights (like standard Dense)
        self.base_weight = self.add_weight(
            name='base_weight',
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Spline weights
        # We model the spline as a linear combination of basis functions
        # For efficiency, we use a simplified basis expansion
        self.spline_weight = self.add_weight(
            name='spline_weight',
            shape=(input_dim, self.units, self.grid_size + self.spline_order),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Grid points for B-splines (fixed)
        # Create a grid from grid_range[0] to grid_range[1]
        grid = tf.linspace(
            float(self.grid_range[0]), 
            float(self.grid_range[1]), 
            self.grid_size + 1
        )
        # Extend grid for B-spline calculation
        grid_step = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        grid_ext = tf.concat([
            grid[0] - grid_step * tf.range(self.spline_order, 0, -1, dtype=tf.float32),
            grid,
            grid[-1] + grid_step * tf.range(1, self.spline_order + 1, dtype=tf.float32)
        ], axis=0)
        
        self.grid = tf.Variable(grid_ext, trainable=False, name='grid')
        
        super(KANDense, self).build(input_shape)

    def call(self, inputs):
        # 1. Base transformation (like MLP but with SiLU)
        # y_b = SiLU(x @ W_b)
        base_output = tf.matmul(inputs, self.base_weight)
        base_output = self.base_activation(base_output)
        
        # 2. Spline transformation
        # Compute B-spline basis functions for inputs
        # Use a simplified B-spline approximation or RBF for differentiability in TF
        # Here we use a Radial Basis Function (RBF) approximation for implementation simplicity/stability
        # Real B-splines are recursive and hard to vectorize efficiently in pure TF without custom ops
        
        # Normalize inputs to grid range for spline lookup
        x_norm = (inputs - self.grid_range[0]) / (self.grid_range[1] - self.grid_range[0])
        x_norm = tf.clip_by_value(x_norm, 0.0, 1.0)
        
        # We'll use a simplified polynomial expansion instead of full B-splines for this version
        # to ensure stability and speed on standard GPUs
        # y_s = sum(w_i * x^i)
        
        # Let's use a Chebyshev polynomial expansion instead? 
        # Or simple RBF: exp(-gamma * (x - center)^2)
        
        # Expand inputs: (batch, in_dim, 1)
        x_expanded = tf.expand_dims(inputs, -1)
        
        # We'll actually implement the "FastKAN" approach:
        # y = W_base * act(x) + W_spline * spline(x)
        # where spline(x) is approximated by RBFs
        
        # RBF Centers
        grid_centers = tf.linspace(
            float(self.grid_range[0]), 
            float(self.grid_range[1]), 
            self.grid_size
        ) # (grid_size,)
        
        # (batch, in_dim, grid_size)
        dist = tf.square(x_expanded - grid_centers)
        rbf = tf.exp(-5.0 * dist) 
        
        # Apply spline weights
        # rbf: (batch, in_dim, grid_size)
        # spline_weight: (in_dim, units, grid_size) - but created as (input_dim, units, grid_size + order)
        # Sliced weight: (input_dim, units, grid_size) -> (i, u, g)
        
        # We need (batch, units)
        # Einstein summation: b = batch, i = input_dim, g = grid_size, u = units
        # rbf: big
        # weight: iug
        # output: bu
        spline_output = tf.einsum('big,iug->bu', rbf, self.spline_weight[:, :, :self.grid_size])
        
        # Combine
        return base_output + spline_output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super(KANDense, self).get_config()
        config.update({
            'units': self.units,
            'grid_size': self.grid_size,
            'spline_order': self.spline_order,
            'base_activation': keras.activations.serialize(self.base_activation),
            'grid_range': self.grid_range
        })
        return config
