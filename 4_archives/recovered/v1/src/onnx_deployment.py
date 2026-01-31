"""
ONNX Export and Cross-Platform Deployment
Critical P0 feature identified in cross-analysis

Enables deployment on:
- ONNX Runtime (CPU, GPU, NPU)
- TensorRT (NVIDIA)
- OpenVINO (Intel)
- CoreML (Apple)
- Mobile devices (Android, iOS)
"""

import tensorflow as tf
import numpy as np
import os


class ONNXExporter:
    """
    Export TensorFlow/Keras models to ONNX format
    Based on best practices from TensorFlow and ONNX Runtime
    """
    
    def __init__(self, model_path, output_path='model.onnx'):
        """
        Initialize ONNX exporter
        
        Args:
            model_path: Path to saved Keras model (.h5 or SavedModel)
            output_path: Output ONNX file path
        """
        self.model_path = model_path
        self.output_path = output_path
        self.model = None
    
    def load_model(self):
        """Load Keras model"""
        print(f"Loading model from: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        print(f"✅ Model loaded: {self.model.name}")
        print(f"   Input shape: {self.model.input_shape}")
        print(f"   Output shape: {self.model.output_shape}")
        return self.model
    
    def export_to_onnx(self, opset_version=13, optimize=True):
        """
        Export model to ONNX format
        
        Args:
            opset_version: ONNX opset version (13 recommended for compatibility)
            optimize: Whether to optimize the ONNX graph
        
        Returns:
            Path to exported ONNX model
        """
        try:
            import tf2onnx
        except ImportError:
            print("❌ tf2onnx not installed. Install with: pip install tf2onnx")
            return None
        
        if self.model is None:
            self.load_model()
        
        print(f"\nExporting to ONNX (opset {opset_version})...")
        
        # Convert to ONNX
        spec = (tf.TensorSpec(self.model.input_shape, tf.float32, name="input"),)
        
        try:
            model_proto, _ = tf2onnx.convert.from_keras(
                self.model,
                input_signature=spec,
                opset=opset_version,
                output_path=self.output_path
            )
            
            print(f"✅ ONNX export successful: {self.output_path}")
            
            # Get model size
            size_mb = os.path.getsize(self.output_path) / (1024 * 1024)
            print(f"   Model size: {size_mb:.2f} MB")
            
            # Optimize if requested
            if optimize:
                self.optimize_onnx()
            
            return self.output_path
            
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            return None
    
    def optimize_onnx(self):
        """
        Optimize ONNX model for inference
        Applies graph optimizations (operator fusion, constant folding, etc.)
        """
        try:
            import onnx
            from onnxruntime.transformers import optimizer
        except ImportError:
            print("⚠️  ONNX optimization tools not available")
            return
        
        print("\nOptimizing ONNX model...")
        
        try:
            # Load ONNX model
            onnx_model = onnx.load(self.output_path)
            
            # Basic optimizations
            from onnx import optimizer as onnx_optimizer
            passes = [
                'eliminate_nop_transpose',
                'eliminate_nop_pad',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
                'fuse_add_bias_into_conv',
                'fuse_bn_into_conv',
            ]
            
            optimized_model = onnx_optimizer.optimize(onnx_model, passes)
            
            # Save optimized model
            optimized_path = self.output_path.replace('.onnx', '_optimized.onnx')
            onnx.save(optimized_model, optimized_path)
            
            # Compare sizes
            original_size = os.path.getsize(self.output_path) / (1024 * 1024)
            optimized_size = os.path.getsize(optimized_path) / (1024 * 1024)
            reduction = (1 - optimized_size / original_size) * 100
            
            print(f"✅ Optimization complete:")
            print(f"   Original: {original_size:.2f} MB")
            print(f"   Optimized: {optimized_size:.2f} MB")
            print(f"   Reduction: {reduction:.1f}%")
            
            return optimized_path
            
        except Exception as e:
            print(f"⚠️  Optimization failed: {e}")
            return None
    
    def validate_onnx(self, test_input=None):
        """
        Validate ONNX model by comparing with original Keras model
        
        Args:
            test_input: Test input array (optional)
        
        Returns:
            True if outputs match within tolerance
        """
        try:
            import onnxruntime as ort
        except ImportError:
            print("❌ onnxruntime not installed. Install with: pip install onnxruntime")
            return False
        
        print("\nValidating ONNX model...")
        
        # Create test input if not provided
        if test_input is None:
            input_shape = list(self.model.input_shape)
            input_shape[0] = 1  # Batch size
            test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Keras prediction
        keras_output = self.model.predict(test_input, verbose=0)
        
        # ONNX prediction
        ort_session = ort.InferenceSession(self.output_path)
        onnx_output = ort_session.run(
            None,
            {ort_session.get_inputs()[0].name: test_input}
        )[0]
        
        # Compare outputs
        max_diff = np.max(np.abs(keras_output - onnx_output))
        mean_diff = np.mean(np.abs(keras_output - onnx_output))
        
        print(f"✅ Validation results:")
        print(f"   Max difference: {max_diff:.6f}")
        print(f"   Mean difference: {mean_diff:.6f}")
        
        tolerance = 1e-4
        if max_diff < tolerance:
            print(f"   ✅ PASS (difference < {tolerance})")
            return True
        else:
            print(f"   ⚠️  FAIL (difference >= {tolerance})")
            return False


class ONNXBenchmark:
    """
    Benchmark ONNX model inference performance
    Compares TensorFlow vs ONNX Runtime
    """
    
    def __init__(self, onnx_path, keras_model_path=None):
        """
        Initialize benchmarking
        
        Args:
            onnx_path: Path to ONNX model
            keras_model_path: Path to original Keras model (optional)
        """
        self.onnx_path = onnx_path
        self.keras_model_path = keras_model_path
        self.ort_session = None
        self.keras_model = None
    
    def load_models(self):
        """Load both ONNX and Keras models"""
        import onnxruntime as ort
        
        # Load ONNX
        self.ort_session = ort.InferenceSession(self.onnx_path)
        print(f"✅ ONNX model loaded: {self.onnx_path}")
        
        # Load Keras if provided
        if self.keras_model_path:
            self.keras_model = tf.keras.models.load_model(
                self.keras_model_path, compile=False
            )
            print(f"✅ Keras model loaded: {self.keras_model_path}")
    
    def benchmark(self, input_shape, num_iterations=100, warmup=10):
        """
        Run inference benchmark
        
        Args:
            input_shape: Input tensor shape (including batch)
            num_iterations: Number of inference iterations
            warmup: Number of warmup iterations
        
        Returns:
            Dictionary with benchmark results
        """
        import time
        
        if self.ort_session is None:
            self.load_models()
        
        # Create test input
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        print(f"\nBenchmarking with input shape: {input_shape}")
        print(f"Iterations: {num_iterations} (after {warmup} warmup)")
        
        results = {}
        
        # Benchmark ONNX Runtime
        print("\nONNX Runtime:")
        input_name = self.ort_session.get_inputs()[0].name
        
        # Warmup
        for _ in range(warmup):
            _ = self.ort_session.run(None, {input_name: test_input})
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            _ = self.ort_session.run(None, {input_name: test_input})
        onnx_time = time.time() - start_time
        
        onnx_latency = (onnx_time / num_iterations) * 1000  # ms
        onnx_throughput = num_iterations / onnx_time
        
        print(f"  Latency: {onnx_latency:.2f} ms/sample")
        print(f"  Throughput: {onnx_throughput:.2f} samples/sec")
        
        results['onnx'] = {
            'latency_ms': onnx_latency,
            'throughput': onnx_throughput
        }
        
        # Benchmark Keras if available
        if self.keras_model is not None:
            print("\nKeras/TensorFlow:")
            
            # Warmup
            for _ in range(warmup):
                _ = self.keras_model.predict(test_input, verbose=0)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                _ = self.keras_model.predict(test_input, verbose=0)
            keras_time = time.time() - start_time
            
            keras_latency = (keras_time / num_iterations) * 1000  # ms
            keras_throughput = num_iterations / keras_time
            
            print(f"  Latency: {keras_latency:.2f} ms/sample")
            print(f"  Throughput: {keras_throughput:.2f} samples/sec")
            
            results['keras'] = {
                'latency_ms': keras_latency,
                'throughput': keras_throughput
            }
            
            # Speedup
            speedup = keras_latency / onnx_latency
            print(f"\n✅ ONNX Runtime speedup: {speedup:.2f}x")
            results['speedup'] = speedup
        
        return results


def export_to_tflite(model_path, output_path='model.tflite', quantize=False):
    """
    Export to TensorFlow Lite for mobile deployment
    
    Args:
        model_path: Path to Keras model
        output_path: Output TFLite file path
        quantize: Whether to quantize to INT8
    
    Returns:
        Path to TFLite model
    """
    print(f"\nExporting to TensorFlow Lite...")
    
    # Load model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        print("  Applying INT8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ TFLite export successful: {output_path}")
    print(f"   Model size: {size_mb:.2f} MB")
    
    return output_path


def create_deployment_package(
    model_path,
    output_dir='deployment_package',
    include_onnx=True,
    include_tflite=True,
    include_quantized=True
):
    """
    Create complete deployment package with multiple formats
    
    Args:
        model_path: Path to Keras model
        output_dir: Output directory for package
        include_onnx: Include ONNX export
        include_tflite: Include TFLite export
        include_quantized: Include quantized versions
    
    Returns:
        Dictionary with paths to all exported models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("CREATING DEPLOYMENT PACKAGE")
    print("="*60)
    
    exports = {}
    
    # 1. ONNX Export
    if include_onnx:
        onnx_path = os.path.join(output_dir, 'model.onnx')
        exporter = ONNXExporter(model_path, onnx_path)
        exporter.load_model()
        onnx_result = exporter.export_to_onnx(optimize=True)
        if onnx_result:
            exports['onnx'] = onnx_result
            exporter.validate_onnx()
    
    # 2. TFLite Export
    if include_tflite:
        tflite_path = os.path.join(output_dir, 'model.tflite')
        tflite_result = export_to_tflite(model_path, tflite_path, quantize=False)
        exports['tflite'] = tflite_result
        
        # 3. Quantized TFLite
        if include_quantized:
            tflite_quant_path = os.path.join(output_dir, 'model_int8.tflite')
            tflite_quant_result = export_to_tflite(
                model_path, tflite_quant_path, quantize=True
            )
            exports['tflite_quantized'] = tflite_quant_result
    
    # 4. Create README
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# Model Deployment Package\n\n")
        f.write("This package contains the trained model in multiple formats ")
        f.write("for cross-platform deployment.\n\n")
        f.write("## Available Formats\n\n")
        
        for format_name, path in exports.items():
            size_mb = os.path.getsize(path) / (1024 * 1024)
            f.write(f"- **{format_name}**: `{os.path.basename(path)}` ")
            f.write(f"({size_mb:.2f} MB)\n")
        
        f.write("\n## Deployment Instructions\n\n")
        f.write("### ONNX Runtime (Python)\n")
        f.write("```python\n")
        f.write("import onnxruntime as ort\n")
        f.write("session = ort.InferenceSession('model.onnx')\n")
        f.write("output = session.run(None, {'input': input_data})\n")
        f.write("```\n\n")
        
        f.write("### TensorFlow Lite (Python)\n")
        f.write("```python\n")
        f.write("import tensorflow as tf\n")
        f.write("interpreter = tf.lite.Interpreter(model_path='model.tflite')\n")
        f.write("interpreter.allocate_tensors()\n")
        f.write("```\n\n")
    
    exports['readme'] = readme_path
    
    print("\n" + "="*60)
    print("✅ DEPLOYMENT PACKAGE CREATED")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Total files: {len(exports)}")
    
    return exports


# Example usage
if __name__ == "__main__":
    print("Phoenix Protocol - ONNX Export & Deployment")
    print("Critical P0 feature for cross-platform deployment\n")
    
    # Example: Export a model
    model_path = "models/neurosnake_best.h5"
    
    if os.path.exists(model_path):
        # Create deployment package
        exports = create_deployment_package(
            model_path=model_path,
            output_dir='deployment',
            include_onnx=True,
            include_tflite=True,
            include_quantized=True
        )
        
        # Benchmark if ONNX export succeeded
        if 'onnx' in exports:
            print("\n" + "="*60)
            print("BENCHMARKING")
            print("="*60)
            
            benchmark = ONNXBenchmark(
                onnx_path=exports['onnx'],
                keras_model_path=model_path
            )
            
            results = benchmark.benchmark(
                input_shape=(1, 224, 224, 3),
                num_iterations=100
            )
            
            print("\n✅ Deployment package ready for production")
    else:
        print(f"⚠️  Model not found: {model_path}")
        print("   Train a model first, then run this script")
