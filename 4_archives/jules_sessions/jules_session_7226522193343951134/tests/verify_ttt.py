import tensorflow as tf
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.ttt_kan import TTTKANLinear

def verify_ttt():
    print("Verifying TTT-KAN Layer...")
    
    # 1. Instantiation
    try:
        layer = TTTKANLinear(units=32, grid_size=5)
        # Build layer
        layer.build((None, 64))
        print("✅ Layer instantiation successful.")
    except Exception as e:
        print(f"❌ Layer instantiation failed: {e}")
        return False

    # 2. Forward Pass (Standard)
    try:
        x = tf.random.normal((1, 64))
        y = layer(x)
        print(f"✅ Standard forward pass successful. Output: {y.shape}")
    except Exception as e:
        print(f"❌ Standard pass failed: {e}")
        return False

    # 3. TTT Pass (Reconstruction)
    try:
        y, recon = layer(x, ttt_step=True)
        print(f"✅ TTT forward pass successful. Recon: {recon.shape}")
        
        if recon.shape != x.shape:
             print(f"❌ Recon shape mismatch. Expected {x.shape}, got {recon.shape}")
             return False
    except Exception as e:
        print(f"❌ TTT pass failed: {e}")
        return False
        
    print("\nVerification Complete: TTT-KAN is ready.")
    return True

if __name__ == "__main__":
    verify_ttt()
