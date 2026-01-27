"""
Spectral Gating Block for Phoenix Protocol (NeuroSnakeSpectral)
Implements frequency-domain global mixing to complement local Snake convolutions.

Based on: "Global Filter Networks for Image Classification" (Rao et al., NeurIPS 2021)
and "Spectral Gating Networks" ideas.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class SpectralGatingBlock(layers.Layer):
    """
    Spectral Gating Block (SGB).
    
    Computes global features via 2D FFT, applies a learnable weight matrix in the frequency domain,
    and transforms back via IFFT. This provides a global receptive field with O(N log N) complexity.
    """
    
    def __init__(self, channels, **kwargs):
        super(SpectralGatingBlock, self).__init__(**kwargs)
        self.channels = channels
        
    def build(self, input_shape):
        # Input shape: (B, H, W, C)
        # FFT2D over H, W.
        # Learnable complex weights: (H, W//2 + 1, C)
        # Note: We need fixed size H, W for simple implementation or dynamic handling.
        # For simplicity in this research phase, we assume fixed resolution 224x224 or adapt dynamically.
        # To be fully dynamic, we learn a smaller weight map and resize it, but let's assume standard resize.
        
        # Actually, to be safe with varying sizes, we can use a fixed-size spectral filter 
        # and interpolate, OR just learn weights for the specific training resolution.
        # Let's go with the standard "Global Filter" approach: 
        # The weights usually match the FFT size.
        
        # But wait, Keras layers must handle `None` dimensions. 
        # If H, W are None, we can't define static weights matching exact freq bins.
        # Solution: Use a fixed size parameter matrix and bilinear upsample it to match FFT size.
        
        self.h_param = 112 # Half of 224
        self.w_param = 112
        
        # Real and Imaginary parts of the spectral filter
        self.complex_weight_real = self.add_weight(
            name='complex_weight_real',
            shape=(self.h_param, self.w_param // 2 + 1, self.channels),
            initializer='glorot_uniform',
            trainable=True
        )
        self.complex_weight_imag = self.add_weight(
            name='complex_weight_imag',
            shape=(self.h_param, self.w_param // 2 + 1, self.channels),
            initializer='glorot_uniform',
            trainable=True
        )
        
        super(SpectralGatingBlock, self).build(input_shape)
        
    def call(self, inputs):
        # inputs: (B, H, W, C)
        dtype = inputs.dtype
        x = tf.cast(inputs, tf.float32)
        
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = x.shape[-1]
        
        # 1. 2D Real FFT
        # Permute to (B, C, H, W) for standard FFT ops if needed, but TF supports (B, H, W, C) usually on last dims?
        # tf.signal.rfft2d computes over the inner-most 2 dimensions.
        # So we need to transpose to (B, C, H, W) to fft over H, W?
        # No, tf.signal.rfft2d takes `input_tensor`. "The inner-most 2 dimensions are used."
        # So if input is (B, H, W, C), inner-most are (W, C). That's wrong.
        # We need (B, C, H, W) -> FFT over (H, W).
        
        x_transposed = tf.transpose(x, [0, 3, 1, 2]) # (B, C, H, W)
        
        # FFT
        x_fft = tf.signal.rfft2d(x_transposed) # (B, C, H, W//2 + 1)
        
        # 2. Spectral Gating (Multiplication)
        # We need to broadcast our learnable weights to this size.
        # Current FFT size: (H, W//2 + 1)
        # Weight size: (h_param, w_param//2 + 1)
        
        # Dynamic Resizing of Weights to match Input FFT size
        # This makes the block resolution-agnostic (essential for SOTA)
        
        # x_fft shape: (B, C, H, W_freq) where W_freq = W // 2 + 1
        fft_h = tf.shape(x_fft)[2]
        fft_w = tf.shape(x_fft)[3]
        
        # Resize Real and Imag parts separately using Bilinear Interpolation
        # weights are (H_param, W_param, C) -> Need (C, H, W_freq)
        
        # 1. Transpose to (C, H_param, W_param) -> (H_param, W_param, C) is default
        # TF Resize expects (Batch, H, W, Channels). Let's treat C as Channels? No, we need to resize H,W.
        # Treat H,W as the spatial dims. 
        # complex_weight_real: (112, 57, C)
        
        # Resize to (fft_h, fft_w)
        real_resized = tf.image.resize(self.complex_weight_real, size=(fft_h, fft_w))
        imag_resized = tf.image.resize(self.complex_weight_imag, size=(fft_h, fft_w))
        
        # Construct complex weight
        weight = tf.complex(real_resized, imag_resized) # (H, W_freq, C)
        
        # Transpose to (C, H, W_freq) for broadcasting with (B, C, H, W_freq)
        # Wait, x_fft is (B, C, H, W_freq). 
        # weight is (H, W_freq, C).
        # We need (1, C, H, W_freq) or (1, H, W_freq, C) if we change x_fft.
        
        # Let's align weight to x_fft.
        # x_fft: (B, C, H, W_freq)
        # weight: (H, W_freq, C) -> Transpose to (C, H, W_freq)
        weight = tf.transpose(weight, [2, 0, 1])
        
        # Element-wise product
        Y_fft = x_fft * weight
        
        # 3. Inverse FFT
        Y = tf.signal.irfft2d(Y_fft, fft_length=[H, W]) # (B, C, H, W)
        
        # Transpose back
        Y = tf.transpose(Y, [0, 2, 3, 1]) # (B, H, W, C)
        
        return tf.cast(Y, dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(SpectralGatingBlock, self).get_config()
        config.update({'channels': self.channels})
        return config
