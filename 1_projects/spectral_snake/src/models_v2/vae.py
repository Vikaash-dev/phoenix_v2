"""
VAE model for the Disentangled VAE.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .encoder import build_encoder
from .decoder import build_decoder
from ..models.latent_partition import LatentPartition

class VAE(keras.Model):
    """
    A VAE model that is built from modular components.
    """
    def __init__(
        self,
        input_shape=(224, 224, 3),
        partition_sizes={'tumor': 32, 'edema': 32, 'cavity': 32, 'background': 128},
        **kwargs
    ):
        super(VAE, self).__init__(**kwargs)
        self.encoder = build_encoder(input_shape=input_shape)
        self.decoder = build_decoder(output_shape=input_shape, latent_dim=sum(partition_sizes.values()))
        self.latent_proj = layers.Dense(2 * sum(partition_sizes.values()), name='latent_projection')
        self.partition_layer = LatentPartition(partition_sizes)

    def call(self, inputs, training=False):
        encoded = self.encoder(inputs, training=training)
        latent_params_raw = self.latent_proj(encoded)
        z_combined, samples, params = self.partition_layer(latent_params_raw)
        reconstruction = self.decoder(z_combined, training=training)
        return {
            'reconstruction': reconstruction,
            'latent_samples': samples,
            'latent_params': params
        }
