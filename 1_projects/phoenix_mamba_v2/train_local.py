import tensorflow as tf
import os
import sys

# Ensure src is in pythonpath
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.append(src_path)

from phoenix_mamba_v2.models.phoenix import PhoenixMambaV2

def test_model_compilation():
    print("Initializing model...")
    try:
        model = PhoenixMambaV2(num_classes=4)
        print("Building model...")
        # Testing with 4D input (standard image)
        # We run a forward pass first to ensure everything is built
        print("\nTesting forward pass...")
        dummy_input = tf.random.normal((1, 224, 224, 3))
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")

        print("\nModel Summary:")
        model.summary()

        return model
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    test_model_compilation()
    print("Model compiled successfully")
