"""
Hyper-Liquid Layer.
A Liquid Time-Constant (LTC) layer where the time constant 'tau' is dynamically generated
by a Hypernetwork (Meta-Learner) based on the input features.

Updates v4.1:
- Added Spectral Normalization to Hyper-Controller to prevent collapse.
- Improved Tau initialization for stability.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class SpectralNormalization(layers.Wrapper):
    """
    Simple Spectral Normalization Wrapper.
    Restrains the Lipschitz constant of the layer to stabilize Hypernetwork training.
    """
    def __init__(self, layer, iteration=1, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.iteration = iteration

    def build(self, input_shape):
        self.layer.build(input_shape)
        # Get the kernel from the wrapped layer
        if hasattr(self.layer, 'kernel'):
            self.w = self.layer.kernel
            self.w_shape = self.w.shape.as_list()
            self.u = self.add_weight(
                shape=(1, self.w_shape[-1]),
                initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                trainable=False,
                name='sn_u',
                dtype=self.w.dtype
            )
        super(SpectralNormalization, self).build(input_shape)

    def call(self, inputs):
        self._update_weights()
        output = self.layer(inputs)
        return output

    def _update_weights(self):
        # Power iteration to estimate spectral norm
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u_hat = self.u
        v_hat = None
        
        for _ in range(self.iteration):
            v_ = tf.matmul(u_hat, w_reshaped, transpose_b=True)
            v_hat = tf.nn.l2_normalize(v_)
            u_ = tf.matmul(v_hat, w_reshaped)
            u_hat = tf.nn.l2_normalize(u_)
            
        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), u_hat, transpose_b=True)
        self.u.assign(u_hat)
        self.layer.kernel.assign(self.w / sigma)

class HyperLiquidConv2D(layers.Layer):
    """
    Hyper-Liquid Convolutional Layer.
    
    Structure:
    1. Input X -> Hyper-Controller (GAP + SpectralNorm(MLP)) -> Tau (Batch, Filters)
    2. Input X -> Liquid ODE Solver (using dynamic Tau) -> Output Y
    """
    
    def __init__(
        self,
        filters,
        kernel_size=3,
        strides=1,
        time_step=0.1,
        unfold_steps=3,
        **kwargs
    ):
        super(HyperLiquidConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.time_step = time_step
        self.unfold_steps = unfold_steps
        
    def build(self, input_shape):
        input_channels = input_shape[-1]
        
        # 1. Hyper-Controller (Meta-Network)
        # Generates 'tau' for each sample in the batch
        # UPDATED: Wrapped in SpectralNormalization to prevent "Hyper-Collapse"
        self.hyper_pool = layers.GlobalAveragePooling2D()
        
        self.hyper_dense1 = SpectralNormalization(
            layers.Dense(16, activation='relu', name='hyper_dense1')
        )
        
        # Initialize dense2 to output values near 0 initially to start with safe Tau
        self.hyper_dense2 = SpectralNormalization(
            layers.Dense(
                self.filters, 
                activation='sigmoid', 
                name='hyper_tau_gen',
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)
            )
        )
        
        # 2. Main Synaptic Weights (Standard Convolution)
        self.input_conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            name='input_conv'
        )
        
        # 3. Recurrent Weights (Lateral connectivity)
        self.recurrent_conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            name='recurrent_conv'
        )
        
        # Bias
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer='zeros',
            trainable=True
        )
        
        super(HyperLiquidConv2D, self).build(input_shape)

    def call(self, inputs):
        # inputs: (Batch, H, W, C)
        
        # A. Generate Dynamic Tau
        # (Batch, C) -> (Batch, 16) -> (Batch, Filters)
        context = self.hyper_pool(inputs)
        tau_raw = self.hyper_dense2(self.hyper_dense1(context))
        
        # Scale Tau to avoid extreme values (0.1 to 1.1)
        # UPDATED: Centered initialization via bias addition
        # If sigmoid output is ~0.5, tau is ~0.6. Safe range.
        tau = tau_raw + 0.1 
        
        # Reshape Tau for broadcasting: (Batch, 1, 1, Filters)
        tau = tf.reshape(tau, (-1, 1, 1, self.filters))
        
        # B. Compute Synaptic Input
        synaptic_input = self.input_conv(inputs)
        
        # C. ODE Solver (Euler) using Dynamic Tau
        h = tf.zeros_like(synaptic_input)
        
        for _ in range(self.unfold_steps):
            # -(h - bias) / tau
            # Note: tau is now (Batch, 1, 1, Filters), specific to each sample!
            leakage = -(h - self.bias) / tau
            
            recurrent_input = self.recurrent_conv(h)
            nonlinear_drive = tf.nn.tanh(synaptic_input + recurrent_input)
            
            dh_dt = leakage + nonlinear_drive
            
            h = h + self.time_step * dh_dt
            
        return h

if __name__ == "__main__":
    # Quick verify
    layer = HyperLiquidConv2D(32)
    x = tf.random.normal((4, 64, 64, 16))
    y = layer(x)
    print("Input:", x.shape)
    print("Output:", y.shape)
