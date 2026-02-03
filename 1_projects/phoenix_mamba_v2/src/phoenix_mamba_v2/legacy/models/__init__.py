"""
src/models - Phoenix Protocol SOTA Architectures

Consolidated from jules sessions (7226, 1322, 2878).
"""

from .kan_layer import KANLinear
from .ttt_kan import TTTKANLinear, ttt_inference
from .dynamic_snake_conv import DynamicSnakeConv2D, SnakeConvBlock
from .liquid_layer import LiquidConv2D
from .hyper_liquid import HyperLiquidConv2D, SpectralNormalization
from .spectral_gating import SpectralGatingBlock, NeuroSnakeSpectralModel

__all__ = [
    'KANLinear',
    'TTTKANLinear',
    'ttt_inference',
    'DynamicSnakeConv2D',
    'SnakeConvBlock',
    'LiquidConv2D',
    'HyperLiquidConv2D',
    'SpectralNormalization',
    'SpectralGatingBlock',
    'NeuroSnakeSpectralModel',
]
