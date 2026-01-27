"""
Classifier component for the Disentangled VAE.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ..models.kan_layer import KANLinear

def build_classifier(num_classes=2):
    """Builds the KAN-based classifier."""
    return keras.Sequential([
        layers.BatchNormalization(),
        KANLinear(64, grid_size=5, spline_order=3),
        layers.LayerNormalization(),
        KANLinear(num_classes, grid_size=5, spline_order=3),
        layers.Softmax()
    ], name='diagnostic_classifier')
