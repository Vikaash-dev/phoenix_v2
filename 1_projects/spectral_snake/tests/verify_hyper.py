import tensorflow as tf
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.neuro_snake_hyper import NeuroSnakeHyperLiquidModel

def verify_hyper():
    print("Verifying NeuroSnake-Hyper-Liquid Model...")
    
    # 1. Instantiation
    try:
        model = NeuroSnakeHyperLiquidModel.create_model(input_shape=(224, 224, 3))
        print("✅ Model instantiation successful.")
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        return False

    # 2. Forward Pass
    try:
        # Use a batch > 1 to verify dynamic broadcasting
        dummy_input = tf.random.normal((4, 224, 224, 3))
        output = model(dummy_input)
        print(f"✅ Forward pass successful. Output shape: {output.shape}")
        
        if output.shape != (4, 2):
            print(f"❌ Expected output shape (4, 2), got {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
        
    print("\nVerification Complete: Hyper-Liquid is ready.")
    return True

if __name__ == "__main__":
    verify_hyper()
