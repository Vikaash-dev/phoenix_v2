"""
src/models - Phoenix Protocol SOTA Architectures

Consolidated from jules sessions (7226, 1322, 2878).
"""

from src.models.kan_layer import KANLinear
from src.models.ttt_kan import TTTKANLinear, ttt_inference
from src.models.dynamic_snake_conv import DynamicSnakeConv2D, SnakeConvBlock
from src.models.liquid_layer import LiquidConv2D
from src.models.hyper_liquid import HyperLiquidConv2D, SpectralNormalization
from src.models.spectral_gating import SpectralGatingBlock, NeuroSnakeSpectralModel

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
