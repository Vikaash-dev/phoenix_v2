"""
ONNX Export Module for Phoenix Protocol
Exports trained NeuroSnake models to ONNX format for cross-platform edge deployment.
"""

import os
import sys
import argparse
import tensorflow as tf
import tf2onnx
import onnx
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

def export_to_onnx(model_path, output_path, opset=13):
    """
    Convert a Keras model to ONNX format.
    
    Args:
        model_path: Path to the trained Keras model (.h5 or .keras)
        output_path: Path to save the ONNX model (.onnx)
        opset: ONNX opset version (default: 13)
    """
    print(f"Loading model from {model_path}...")
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
        
    print(f"Model loaded. Input shape: {model.input_shape}")
    
    # Define input signature
    input_signature = [
        tf.TensorSpec(
            shape=(None, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS), 
            dtype=tf.float32, 
            name="input"
        )
    ]
    
    print(f"Converting to ONNX (opset {opset})...")
    
    try:
        # Convert using tf2onnx
        model_proto, _ = tf2onnx.convert.from_keras(
            model, 
            input_signature=input_signature, 
            opset=opset
        )
        
        # Save model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        onnx.save_model(model_proto, output_path)
        
        print(f"✓ ONNX model saved to {output_path}")
        
        # Verify model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verified successfully")
        
        return True
        
    except Exception as e:
        print(f"Error during ONNX conversion: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export NeuroSnake model to ONNX')
    parser.add_argument('--model-path', type=str, required=True, help='Path to input Keras model')
    parser.add_argument('--output-path', type=str, required=True, help='Path to output ONNX model')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version')
    
    args = parser.parse_args()
    
    success = export_to_onnx(args.model_path, args.output_path, args.opset)
    sys.exit(0 if success else 1)
