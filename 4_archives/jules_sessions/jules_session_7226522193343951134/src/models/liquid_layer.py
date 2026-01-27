"""
Liquid Neural Network Layer for Computer Vision.
Implements a spatially-aware Liquid Time-Constant (LTC) unit using an ODE solver.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class LiquidConv2D(layers.Layer):
    """
    Liquid Convolutional Layer.
    Models the feature map evolution as a system of Ordinary Differential Equations (ODEs).
    dy/dt = -(1/tau) * y + S(x)
    
    Discretized using a semi-implicit Euler method for stability.
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
        super(LiquidConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.time_step = time_step # delta t
        self.unfold_steps = unfold_steps # How many ODE steps to simulate
        
    def build(self, input_shape):
        input_channels = input_shape[-1]
        
        # Tau (Time constant) - Learnable parameter per filter
        # Constraints: Tau > 0
        self.tau = self.add_weight(
            name='tau',
            shape=(self.filters,),
            initializer=tf.random_uniform_initializer(minval=0.1, maxval=1.0),
            constraint=keras.constraints.MinMaxNorm(min_value=0.01, max_value=10.0, axis=0),
            trainable=True
        )
        
        # Input processing convolution (Synaptic inputs)
        self.input_conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            name='input_conv'
        )
        
        # Recurrent weight (Internal connectivity)
        # In a strict "Liquid" sense, this models lateral inhibition/excitation
        self.recurrent_conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1, # Recurrent steps don't downsample
            padding='same',
            name='recurrent_conv',
            kernel_initializer='orthogonal'
        )
        
        # Bias/Leakage
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer='zeros',
            trainable=True
        )
        
        super(LiquidConv2D, self).build(input_shape)

    def call(self, inputs, state=None):
        # Input current I(t) is static for the duration of the "thinking" steps in this layer
        # In a video setting, I(t) would change. Here, we treat the static image as a constant forcing function.
        
        synaptic_input = self.input_conv(inputs)
        
        # Initialize hidden state h_0
        if state is None:
            # Initialize with 0 or the synaptic input itself
            # Using synaptic input helps convergence
            h = tf.zeros_like(synaptic_input)
        else:
            h = state
            
        # ODE Solver Loop (Euler Method)
        # dh/dt = -(h - bias)/tau + sigmoid(synaptic + recurrent(h))
        # h_{t+1} = h_t + dt * dh/dt
        
        for _ in range(self.unfold_steps):
            # 1. Compute derivative components
            # Leakage term: -(h - bias) / tau
            # We broadcast tau and bias across spatial dims
            leakage = -(h - self.bias) / self.tau
            
            # Synaptic + Recurrent term
            # Nonlinear interaction
            recurrent_input = self.recurrent_conv(h)
            total_input = synaptic_input + recurrent_input
            nonlinear_drive = tf.nn.tanh(total_input) # Sigmoid or Tanh
            
            # 2. ODE Update
            # dh/dt = leakage + A * nonlinear_drive? 
            # Simplified Liquid Model: dh/dt = -h/tau + S(x)
            
            dh_dt = leakage + nonlinear_drive
            
            # 3. Step
            h = h + self.time_step * dh_dt
            
        return h

if __name__ == "__main__":
    layer = LiquidConv2D(32)
    x = tf.random.normal((2, 64, 64, 16))
    y = layer(x)
    print("Input:", x.shape)
    print("Output:", y.shape)
