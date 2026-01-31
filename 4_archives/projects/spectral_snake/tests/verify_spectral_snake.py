import tensorflow as tf
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.neuro_mamba_spectral import NeuroSnakeSpectralModel

def verify_spectral_snake():
    print("Verifying Spectral-Snake Model...")
    
    # 1. Instantiation
    try:
        model = NeuroSnakeSpectralModel.create_model(input_shape=(224, 224, 3))
        print("✅ Model instantiation successful.")
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
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

    # 3. Compilation check
    try:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("✅ Compilation successful.")
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        return False
        
    print("\nVerification Complete: Spectral-Snake is ready for experiments.")
    return True

if __name__ == "__main__":
    verify_spectral_snake()
