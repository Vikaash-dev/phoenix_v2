"""
Decoder component for the Disentangled VAE.
"""

import tensorflow as tf
from tensorflow import keras
from ..models.kan_decoder import KANLiquidDecoder

def build_decoder(output_shape=(224, 224, 3), latent_dim=256):
    """Builds the KAN-Liquid decoder."""
    return KANLiquidDecoder(output_shape=output_shape, latent_dim=latent_dim)
