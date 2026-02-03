import os
import sys

# Ensure we can import the local package
sys.path.append(os.path.abspath("1_projects/phoenix_mamba_v2/src"))

import tensorflow as tf
from phoenix_mamba_v2.models.phoenix import PhoenixMambaV2

def test_model():
    print("Testing PhoenixMambaV2 initialization...")
    try:
        model = PhoenixMambaV2(num_classes=4)
        
        # Create dummy input (Batch, H, W, 3)
        # Using 128x128 as in the Kaggle script
        x = tf.random.normal((1, 128, 128, 3))
        
        print("Running forward pass...")
        y = model(x)
        
        print(f"Output shape: {y.shape}")
        print("Test PASSED")
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
