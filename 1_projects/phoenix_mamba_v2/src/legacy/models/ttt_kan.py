"""
Test-Time Training (TTT) KAN Layer.
Extends KANLinear to support self-supervised adaptation during inference.

Source: Consolidated from jules_session_7226522193343951134
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .kan_layer import KANLinear


class TTTKANLinear(KANLinear):
    """
    Test-Time Training KAN Layer.
    Adds a reconstruction head to allow self-supervised updates at inference.
    """
    
    def __init__(self, units, **kwargs):
        super(TTTKANLinear, self).__init__(units, **kwargs)
        
    def build(self, input_shape):
        super(TTTKANLinear, self).build(input_shape)
        input_dim = input_shape[-1]
        
        # Auxiliary Reconstruction Head
        self.decoder_weight = self.add_weight(
            name='decoder_weight',
            shape=(self.units, input_dim),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, x, training=False, ttt_step=False):
        y = super(TTTKANLinear, self).call(x)
        
        if ttt_step:
            x_recon = tf.matmul(y, self.decoder_weight)
            return y, x_recon
            
        return y


def ttt_inference(model, x_test, kan_layer, steps=1, lr=0.001):
    """
    Performs Test-Time Training inference.
    1. Optimize KAN parameters to minimize reconstruction loss on x_test
    2. Predict label using updated parameters
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    
    for _ in range(steps):
        with tf.GradientTape() as tape:
            y, recon = kan_layer(x_test, ttt_step=True)
            loss = tf.reduce_mean(tf.square(x_test - recon))
        
        grads = tape.gradient(loss, kan_layer.trainable_variables)
        optimizer.apply_gradients(zip(grads, kan_layer.trainable_variables))
    
    return model(x_test)


if __name__ == "__main__":
    layer = TTTKANLinear(32)
    x = tf.random.normal((1, 16))
    y, recon = layer(x, ttt_step=True)
    print("Forward + Recon shape:", y.shape, recon.shape)
