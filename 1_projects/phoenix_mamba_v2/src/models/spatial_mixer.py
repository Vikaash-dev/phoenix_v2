"""
SpatialMixer: Topology-Aware Preprocessing for SSM Flattening

Addresses the "Zombie Topology" issue identified by the Multi-Agent Scientific Conference.

Problem:
    Naive 2D→1D flattening destroys spatial adjacency:
    - Pixel at (i, j+1) is 1 token away
    - Pixel at (i+1, j) is W tokens away (e.g., 56 positions)
    - SSM must learn basic 2D topology instead of semantic patterns

Solution:
    Depthwise Separable Convolution captures 3×3 local neighborhoods BEFORE flattening:
    - Each flattened token now contains information from its 8 adjacent pixels
    - SSM processes "spatially-aware tokens" with neighborhood context
    - Preserves gradient flow via residual connection

Architecture:
    1. Depthwise Conv2D (3×3): Spatial mixing per channel
    2. BatchNorm + SiLU activation
    3. Pointwise Conv2D (1×1): Cross-channel mixing
    4. BatchNorm + SiLU activation
    5. Residual addition with input

Parameters:
    - Stage 3 (128 channels): ~18.5k params
    - Stage 4 (256 channels): ~70k params
    - Total: ~88.5k params (negligible vs 5M budget)

References:
    - Agent 2 (Negative Reviewer): Identified topology loss as P0 blocker
    - Agent 5 (Synthesis): Recommended depthwise conv approach
    - Plan: /home/shadow_garden/.claude/plans/quiet-dancing-quokka.md
"""

import tensorflow as tf
from tensorflow.keras import layers


class SpatialMixer(layers.Layer):
    """
    Preserves 2D topology before 1D flattening via depthwise separable convolution.

    Args:
        channels (int): Number of channels (must match input)
        kernel_size (int): Spatial kernel size (default: 3 for 3×3 neighborhood)
        name (str): Layer name for debugging

    Input:
        x: (B, H, W, C) spatial feature map

    Output:
        (B, H, W, C) spatially-mixed feature map with neighborhood context

    Example:
        >>> mixer = SpatialMixer(128, name="stage3_spatial_mix")
        >>> x = tf.random.normal((2, 56, 56, 128))
        >>> out = mixer(x)
        >>> assert out.shape == (2, 56, 56, 128)  # Shape preserved
    """

    def __init__(self, channels, kernel_size=3, name="spatial_mixer", **kwargs):
        super(SpatialMixer, self).__init__(name=name, **kwargs)
        self.channels = channels
        self.kernel_size = kernel_size

        # Depthwise Spatial Mixing (captures 3×3 neighborhoods per channel)
        self.depthwise = layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            padding='same',
            use_bias=False,
            name=f"{name}_dw"
        )
        self.bn_dw = layers.BatchNormalization(name=f"{name}_bn_dw")
        self.act_dw = layers.Activation('silu', name=f"{name}_act_dw")

        # Pointwise Channel Mixing (1×1 conv for cross-channel communication)
        self.pointwise = layers.Conv2D(
            channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=f"{name}_pw"
        )
        self.bn_pw = layers.BatchNormalization(name=f"{name}_bn_pw")
        self.act_pw = layers.Activation('silu', name=f"{name}_act_pw")

        # Residual Addition (preserves gradient flow and input information)
        self.add = layers.Add(name=f"{name}_add")

    def call(self, x, training=None):
        """
        Forward pass with depthwise-pointwise mixing and residual connection.

        Args:
            x: (B, H, W, C) input tensor
            training: Boolean flag for batch normalization

        Returns:
            (B, H, W, C) spatially-mixed tensor
        """
        shortcut = x

        # 1. Spatial Mix (depthwise conv captures local neighborhoods)
        x = self.depthwise(x)
        x = self.bn_dw(x, training=training)
        x = self.act_dw(x)

        # 2. Channel Mix (pointwise conv for cross-channel interaction)
        x = self.pointwise(x)
        x = self.bn_pw(x, training=training)
        x = self.act_pw(x)

        # 3. Residual (allows network to learn "how much" mixing is needed)
        return self.add([x, shortcut])

    def compute_output_shape(self, input_shape):
        """
        Ensures Keras 3 compatibility for static shape inference.

        Args:
            input_shape: (B, H, W, C)

        Returns:
            Same shape (B, H, W, C) - spatial dimensions preserved
        """
        return input_shape

    def get_config(self):
        """
        Serialization support for model saving/loading.
        """
        config = super(SpatialMixer, self).get_config()
        config.update({
            "channels": self.channels,
            "kernel_size": self.kernel_size
        })
        return config
