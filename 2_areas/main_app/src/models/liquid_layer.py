"""
Liquid Neural Network Layer for Computer Vision.
Implements a spatially-aware Liquid Time-Constant (LTC) unit using an ODE solver.

Source: Consolidated from jules_session_7226522193343951134
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class LiquidConv2D(layers.Layer):
    """
    Liquid Convolutional Layer.
    Models feature map evolution as ODEs: dy/dt = -(1/tau)*y + S(x)
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
        self.time_step = time_step
        self.unfold_steps = unfold_steps
        
    def build(self, input_shape):
        # Tau (Time constant)
        self.tau = self.add_weight(
            name='tau',
            shape=(self.filters,),
            initializer=tf.random_uniform_initializer(minval=0.1, maxval=1.0),
            constraint=keras.constraints.MinMaxNorm(min_value=0.01, max_value=10.0),
            trainable=True
        )
        
        self.input_conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same'
        )
        
        self.recurrent_conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            kernel_initializer='orthogonal'
        )
        
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer='zeros',
            trainable=True
        )
        
        super(LiquidConv2D, self).build(input_shape)

    def call(self, inputs, state=None):
        synaptic_input = self.input_conv(inputs)
        h = tf.zeros_like(synaptic_input) if state is None else state
        
        for _ in range(self.unfold_steps):
            leakage = -(h - self.bias) / self.tau
            recurrent_input = self.recurrent_conv(h)
            nonlinear_drive = tf.nn.tanh(synaptic_input + recurrent_input)
            dh_dt = leakage + nonlinear_drive
            h = h + self.time_step * dh_dt
            
        return h

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'time_step': self.time_step,
            'unfold_steps': self.unfold_steps,
        })
        return config


if __name__ == "__main__":
    layer = LiquidConv2D(32)
    x = tf.random.normal((2, 64, 64, 16))
    y = layer(x)
    print("Input:", x.shape, "Output:", y.shape)
