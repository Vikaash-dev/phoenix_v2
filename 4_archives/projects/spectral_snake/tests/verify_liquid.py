import tensorflow as tf
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.neuro_snake_liquid import NeuroSnakeLiquidModel

def verify_liquid():
    print("Verifying NeuroSnake-Liquid Model...")
    
    # 1. Instantiation
    try:
        model = NeuroSnakeLiquidModel.create_model(input_shape=(224, 224, 3))
        print("✅ Model instantiation successful.")
        model.summary()
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. Forward Pass
    try:
        dummy_input = tf.random.normal((2, 224, 224, 3))
        output = model(dummy_input)
        print(f"✅ Forward pass successful. Output shape: {output.shape}")
        
        if output.shape != (2, 2):
            print(f"❌ Expected output shape (2, 2), got {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
        
    print("\nVerification Complete: NeuroSnake-Liquid is ready.")
    return True

if __name__ == "__main__":
    verify_liquid()
