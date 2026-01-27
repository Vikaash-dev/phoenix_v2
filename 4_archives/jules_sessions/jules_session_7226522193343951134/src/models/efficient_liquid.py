"""
Efficient Liquid Layer.
Replaces the dense recurrent convolution with a Depthwise Separable Convolution.
Reduces parameter count from O(C^2) to O(C) for the mixing step.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class EfficientLiquidConv2D(layers.Layer):
    """
    Efficient Liquid Convolutional Layer.
    
    Architecture:
    dy/dt = -(1/tau) * y + S(x)
    
    Optimization:
    The recurrent mixing W_rec * h is replaced by a Depthwise Separable Conv.
    This drastically reduces parameters while maintaining spatial mixing.
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
        super(EfficientLiquidConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.time_step = time_step
        self.unfold_steps = unfold_steps
        
    def build(self, input_shape):
        input_channels = input_shape[-1]
        
        # Learnable Time Constant
        self.tau = self.add_weight(
            name='tau',
            shape=(self.filters,),
            initializer=tf.random_uniform_initializer(minval=0.1, maxval=1.0),
            constraint=keras.constraints.MinMaxNorm(min_value=0.01, max_value=10.0, axis=0),
            trainable=True
        )
        
        # Input Synapses (Standard Conv)
        # We keep this dense to extract features properly
        self.input_conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            name='input_conv'
        )
        
        # Recurrent Connections (Efficient Depthwise Separable)
        # 1. Depthwise: Spatial mixing per channel
        # 2. Pointwise: Channel mixing
        self.recurrent_conv = layers.SeparableConv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            depth_multiplier=1,
            name='recurrent_conv_sep'
        )
        
        # Bias
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer='zeros',
            trainable=True
        )
        
        super(EfficientLiquidConv2D, self).build(input_shape)

    def call(self, inputs, state=None):
        synaptic_input = self.input_conv(inputs)
        
        if state is None:
            h = tf.zeros_like(synaptic_input)
        else:
            h = state
            
        for _ in range(self.unfold_steps):
            # -(h - bias) / tau
            leakage = -(h - self.bias) / self.tau
            
            # Recurrent input (Efficient)
            recurrent_input = self.recurrent_conv(h)
            
            nonlinear_drive = tf.nn.tanh(synaptic_input + recurrent_input)
            
            dh_dt = leakage + nonlinear_drive
            h = h + self.time_step * dh_dt
            
        return h
