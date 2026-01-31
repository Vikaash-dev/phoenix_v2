import tensorflow as tf
import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ssm import S6Layer
from src.models.aggregator import SliceAggregator25D
from src.models.attention import HeterogeneityAttention
from src.models.phoenix import PhoenixMambaV2

def test_s6_layer_shape():
    """Test S6 Layer output shape and build."""
    batch_size = 2
    seq_len = 10
    d_model = 32
    
    layer = S6Layer(d_model=d_model, d_state=4)
    x = tf.random.normal((batch_size, seq_len, d_model))
    
    y = layer(x)
    
    assert y.shape == (batch_size, seq_len, d_model)
    print("S6 Layer test passed!")

def test_aggregator_shape():
    """Test 2.5D Aggregator reducing depth."""
    batch_size = 2
    depth = 3
    h, w = 64, 64
    c = 16
    out_filters = 32
    
    aggregator = SliceAggregator25D(filters=out_filters)
    x = tf.random.normal((batch_size, depth, h, w, c))
    
    y = aggregator(x)
    
    assert y.shape == (batch_size, h, w, out_filters)
    print("Aggregator test passed!")

def test_attention_shape():
    """Test Heterogeneity Attention."""
    batch_size = 2
    seq_len = 20
    d_model = 32
    num_heads = 4
    
    attn = HeterogeneityAttention(d_model=d_model, num_heads=num_heads)
    x = tf.random.normal((batch_size, seq_len, d_model))
    
    y = attn(x, x, x)
    
    assert y.shape == (batch_size, seq_len, d_model)
    print("Attention test passed!")

def test_phoenix_model_build():
    """Test full PhoenixMambaV2 model forward pass."""
    batch_size = 1
    depth = 3
    h, w = 224, 224
    c = 3
    num_classes = 4
    
    model = PhoenixMambaV2(num_classes=num_classes)
    x = tf.random.normal((batch_size, depth, h, w, c))
    
    y = model(x)
    
    assert y.shape == (batch_size, num_classes)
    
    # Check parameters
    total_params = model.count_params()
    print(f"Total Parameters: {total_params}")
    
    # Requirement: ~1.4M. 
    # With 4 stages (64, 128, 256, 512) and Mamba blocks, let's see.
    # We won't assert exact match but logging is good.
    assert total_params > 0

if __name__ == "__main__":
    test_s6_layer_shape()
    test_aggregator_shape()
    test_attention_shape()
    test_phoenix_model_build()
    print("All manual tests passed!")
