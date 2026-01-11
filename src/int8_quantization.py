"""
Real INT8 Post-Training Quantization for Phoenix Protocol
Implements true INT8 quantization (not fake quantization) for edge deployment.
Reduces memory bandwidth by 4× and energy consumption by ~20×.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Callable, Iterator, Optional
import json


class INT8Quantizer:
    """
    Real INT8 Post-Training Quantization framework.
    Maps FP32 activations to [-128, 127] integer range using calibration dataset.
    """
    
    def __init__(
        self,
        model: keras.Model,
        calibration_dataset: Optional[tf.data.Dataset] = None,
        num_calibration_samples: int = 100
    ):
        """
        Initialize INT8 quantizer.
        
        Args:
            model: Trained FP32 Keras model
            calibration_dataset: Dataset for calibration (representative samples)
            num_calibration_samples: Number of samples to use for calibration
        """
        self.model = model
        self.calibration_dataset = calibration_dataset
        self.num_calibration_samples = num_calibration_samples
    
    def representative_dataset_gen(self) -> Iterator:
        """
        Generator for representative dataset used in calibration.
        
        Yields:
            Input samples for calibration
        """
        if self.calibration_dataset is None:
            raise ValueError("Calibration dataset not provided")
        
        count = 0
        for images, _ in self.calibration_dataset:
            for image in images:
                if count >= self.num_calibration_samples:
                    return
                yield [tf.expand_dims(image, 0)]
                count += 1
    
    def quantize_to_tflite(
        self,
        output_path: str,
        optimization_mode: str = 'default',
        enable_int8_io: bool = True
    ) -> str:
        """
        Quantize model to TensorFlow Lite INT8 format.
        
        Args:
            output_path: Path to save quantized model
            optimization_mode: 'default' or 'full' quantization
            enable_int8_io: Whether to use INT8 for input/output
            
        Returns:
            Path to saved quantized model
        """
        print("="*80)
        print("PHOENIX PROTOCOL: INT8 QUANTIZATION")
        print("="*80)
        
        # Create TFLite converter
        print("\n1. Creating TFLite converter...")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Set optimization flags
        print("2. Configuring quantization...")
        print(f"   - Optimization mode: {optimization_mode}")
        print(f"   - INT8 input/output: {enable_int8_io}")
        
        if optimization_mode == 'default':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif optimization_mode == 'full':
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        
        # Set representative dataset for calibration
        if self.calibration_dataset is not None:
            print(f"3. Running calibration with {self.num_calibration_samples} samples...")
            converter.representative_dataset = self.representative_dataset_gen
            
            # Enforce INT8 quantization for weights and activations
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            
            if enable_int8_io:
                converter.inference_input_type = tf.uint8  # or tf.int8
                converter.inference_output_type = tf.uint8  # or tf.int8
        else:
            print("⚠ Warning: No calibration dataset provided. Using dynamic range quantization.")
        
        # Convert model
        print("4. Converting model to TFLite INT8...")
        tflite_model = converter.convert()
        
        # Save quantized model
        print(f"5. Saving quantized model to: {output_path}")
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get model statistics
        fp32_size = self._get_model_size_mb(self.model)
        int8_size = os.path.getsize(output_path) / (1024 * 1024)
        compression_ratio = fp32_size / int8_size if int8_size > 0 else 0
        
        print("\n" + "="*80)
        print("QUANTIZATION RESULTS")
        print("="*80)
        print(f"FP32 model size: {fp32_size:.2f} MB")
        print(f"INT8 model size: {int8_size:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}×")
        print(f"Memory reduction: {(1 - int8_size/fp32_size)*100:.1f}%")
        print("="*80 + "\n")
        
        return output_path
    
    def _get_model_size_mb(self, model: keras.Model) -> float:
        """
        Estimate FP32 model size in MB.
        
        Args:
            model: Keras model
            
        Returns:
            Model size in MB
        """
        import tempfile
        # Save temporary file to get actual size
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            model.save(temp_path)
            size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return size_mb
    
    def validate_quantized_model(
        self,
        tflite_model_path: str,
        test_dataset: tf.data.Dataset,
        num_samples: int = 100
    ) -> dict:
        """
        Validate quantized model performance.
        
        Args:
            tflite_model_path: Path to quantized TFLite model
            test_dataset: Test dataset
            num_samples: Number of samples to test
            
        Returns:
            Dictionary with validation metrics
        """
        print("="*80)
        print("VALIDATING QUANTIZED MODEL")
        print("="*80)
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\nInput details: {input_details[0]['shape']}, {input_details[0]['dtype']}")
        print(f"Output details: {output_details[0]['shape']}, {output_details[0]['dtype']}")
        
        # Run inference on test samples
        correct = 0
        total = 0
        
        for images, labels in test_dataset:
            for i in range(len(images)):
                if total >= num_samples:
                    break
                
                # Prepare input
                input_data = np.expand_dims(images[i].numpy(), axis=0)
                
                # Convert to INT8 if needed
                if input_details[0]['dtype'] == np.uint8:
                    input_scale, input_zero_point = input_details[0]['quantization']
                    input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)
                elif input_details[0]['dtype'] == np.int8:
                    input_scale, input_zero_point = input_details[0]['quantization']
                    input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
                
                # Run inference
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                
                # Get output
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                # Dequantize output if needed
                if output_details[0]['dtype'] == np.uint8 or output_details[0]['dtype'] == np.int8:
                    output_scale, output_zero_point = output_details[0]['quantization']
                    output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
                
                # Get prediction
                pred_class = np.argmax(output_data)
                true_class = np.argmax(labels[i].numpy())
                
                if pred_class == true_class:
                    correct += 1
                total += 1
            
            if total >= num_samples:
                break
        
        accuracy = correct / total if total > 0 else 0
        
        print(f"\nQuantized Model Accuracy: {accuracy:.4f} ({correct}/{total})")
        print("="*80 + "\n")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def compare_fp32_vs_int8(
        self,
        tflite_model_path: str,
        test_dataset: tf.data.Dataset,
        num_samples: int = 100
    ) -> dict:
        """
        Compare FP32 and INT8 model performance.
        
        Args:
            tflite_model_path: Path to quantized model
            test_dataset: Test dataset
            num_samples: Number of samples for comparison
            
        Returns:
            Dictionary with comparison metrics
        """
        print("="*80)
        print("FP32 vs INT8 COMPARISON")
        print("="*80)
        
        # Validate FP32 model
        print("\n1. Evaluating FP32 model...")
        fp32_results = self.model.evaluate(
            test_dataset.take(num_samples // 32),
            verbose=0
        )
        fp32_accuracy = fp32_results[1] if len(fp32_results) > 1 else fp32_results[0]
        
        # Validate INT8 model
        print("2. Evaluating INT8 model...")
        int8_results = self.validate_quantized_model(
            tflite_model_path,
            test_dataset,
            num_samples
        )
        int8_accuracy = int8_results['accuracy']
        
        # Compute accuracy degradation
        accuracy_drop = fp32_accuracy - int8_accuracy
        accuracy_drop_pct = (accuracy_drop / fp32_accuracy) * 100 if fp32_accuracy > 0 else 0
        
        # Print comparison
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        print(f"FP32 Accuracy: {fp32_accuracy:.4f}")
        print(f"INT8 Accuracy: {int8_accuracy:.4f}")
        print(f"Accuracy Drop: {accuracy_drop:.4f} ({accuracy_drop_pct:.2f}%)")
        
        if accuracy_drop_pct < 1.0:
            print("✓ Excellent quantization: <1% accuracy drop")
        elif accuracy_drop_pct < 2.0:
            print("✓ Good quantization: <2% accuracy drop")
        else:
            print("⚠ Warning: >2% accuracy drop. Consider more calibration samples.")
        
        print("="*80 + "\n")
        
        return {
            'fp32_accuracy': float(fp32_accuracy),
            'int8_accuracy': float(int8_accuracy),
            'accuracy_drop': float(accuracy_drop),
            'accuracy_drop_percentage': float(accuracy_drop_pct)
        }


def quantize_model(
    model_path: str,
    calibration_dataset: tf.data.Dataset,
    output_path: str,
    test_dataset: Optional[tf.data.Dataset] = None,
    num_calibration_samples: int = 100,
    num_test_samples: int = 100
) -> dict:
    """
    Main function to quantize a trained model.
    
    Args:
        model_path: Path to trained FP32 model
        calibration_dataset: Dataset for calibration
        output_path: Path to save quantized model
        test_dataset: Optional test dataset for validation
        num_calibration_samples: Number of calibration samples
        num_test_samples: Number of test samples
        
    Returns:
        Dictionary with quantization results
    """
    # Load model
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    
    # Create quantizer
    quantizer = INT8Quantizer(
        model=model,
        calibration_dataset=calibration_dataset,
        num_calibration_samples=num_calibration_samples
    )
    
    # Quantize model
    quantized_path = quantizer.quantize_to_tflite(
        output_path=output_path,
        optimization_mode='default',
        enable_int8_io=True
    )
    
    # Validate if test dataset provided
    results = {'quantized_model_path': quantized_path}
    
    if test_dataset is not None:
        comparison = quantizer.compare_fp32_vs_int8(
            tflite_model_path=quantized_path,
            test_dataset=test_dataset,
            num_samples=num_test_samples
        )
        results.update(comparison)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Phoenix Protocol: INT8 Quantization'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained FP32 model (.h5)'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='./quantized_model.tflite',
        help='Path to save quantized model'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Data directory for calibration and testing'
    )
    parser.add_argument(
        '--num-calibration-samples',
        type=int,
        default=100,
        help='Number of samples for calibration'
    )
    parser.add_argument(
        '--num-test-samples',
        type=int,
        default=100,
        help='Number of samples for validation'
    )
    
    args = parser.parse_args()
    
    # Load datasets (simplified example)
    print("Loading datasets...")
    # This is a placeholder - implement proper data loading
    # based on your actual data structure
    
    print("\nQuantization process will be implemented when datasets are available.")
    print("Use the INT8Quantizer class directly for custom implementations.")
