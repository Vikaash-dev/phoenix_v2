import tensorflow as tf
import pytest
import numpy as np
import sys
import os

# Add src to path
# Assuming we are in tests/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from phoenix_mamba_v2.models.phoenix import PhoenixMambaV2
    from phoenix_mamba_v2.visualization.explainability import MambaCAM
except ImportError:
    # Fallback if running from root or different structure
    from src.phoenix_mamba_v2.models.phoenix import PhoenixMambaV2
    from src.phoenix_mamba_v2.visualization.explainability import MambaCAM

def test_mambacam_shapes():
    """Test MambaCAM generation shapes and value ranges."""
    # Setup
    # Input shape: (Batch, Depth, H, W, C)
    # We use small spatial dims for speed
    # Input image needs to be compatible with model downsampling (divide by 8)
    H, W = 64, 64
    B, D, C = 1, 3, 3
    input_tensor = tf.random.normal((B, D, H, W, C))

    # Initialize model with 4 classes
    # We need to ensure config is valid. Using default.
    model = PhoenixMambaV2(num_classes=4)

    # Build model by calling it once
    # This triggers build() of layers
    _ = model(input_tensor)

    # Initialize CAM
    cam_generator = MambaCAM(model)

    # Test 1: Gating Focus
    print("\nTesting Gating Focus...")
    heatmap = cam_generator.generate_map(input_tensor, layer_name='stage4', focus='gating')

    # Check shape: Should be (H_feat, W_feat)
    # Stage 1: /1 -> /2 (pool)
    # Stage 2: /2 -> /4 (pool)
    # Stage 3: /4 -> /8 (pool)
    # Stage 4: /8

    expected_dim = H // 8
    assert heatmap.shape == (expected_dim, expected_dim), f"Expected shape {(expected_dim, expected_dim)}, got {heatmap.shape}"
    assert np.min(heatmap) >= 0.0, "Heatmap contains negative values"
    assert np.max(heatmap) <= 1.0 + 1e-6, "Heatmap values exceed 1.0"

    print("Gating test passed")

    # Test 2: State Write Focus
    print("\nTesting State Write Focus...")
    heatmap = cam_generator.generate_map(input_tensor, layer_name='stage4', focus='state_write')
    assert heatmap.shape == (expected_dim, expected_dim)
    print("State write test passed")

    # Test 3: Timescale Focus
    print("\nTesting Timescale Focus...")
    heatmap = cam_generator.generate_map(input_tensor, layer_name='stage4', focus='timescale')
    assert heatmap.shape == (expected_dim, expected_dim)
    print("Timescale test passed")

    # Test 4: With Target Class (GradCAM style)
    print("\nTesting Class-Targeted Focus...")
    # We need to ensure gradients flow.
    # TF GradientTape requires watching or variables.
    # The input to generate_map is a tensor.
    # In generate_map, we do model(input).
    # model(input) depends on model variables.
    # We differentiate score w.r.t internals.
    # Internals depend on input.
    # So gradient should flow back to internals.

    heatmap = cam_generator.generate_map(input_tensor, target_class_index=0, layer_name='stage4', focus='gating')
    assert heatmap.shape == (expected_dim, expected_dim)
    print("Class-targeted test passed")

    # Test 5: Overlay
    print("\nTesting Overlay...")
    # Create a dummy image
    img = np.zeros((H, W, 3), dtype=np.uint8)
    overlay = cam_generator.overlay_heatmap(img, heatmap)
    assert overlay.shape == (H, W, 3)
    print("Overlay test passed")

if __name__ == "__main__":
    test_mambacam_shapes()
