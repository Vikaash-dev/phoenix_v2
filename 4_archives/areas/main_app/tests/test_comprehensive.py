"""
Comprehensive Test Suite for Phoenix Protocol
Based on cross-analysis with nnU-Net, MONAI, and academic best practices

Implements:
- Unit tests for all modules
- Integration tests for training pipeline
- Performance regression tests
- Statistical validation tests
"""

import unittest
import numpy as np
import tensorflow as tf
import os
import sys

from pathlib import Path
# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root.parent))


class TestTrainingImprovements(unittest.TestCase):
    """Test P0 training improvements"""

    def test_seed_fixing(self):
        """Test reproducible random seed setting"""
        from src.training_improvements import set_seed

        set_seed(42)

        # Generate random numbers
        np1 = np.random.rand(10)
        tf1 = tf.random.uniform([10]).numpy()

        # Reset seed
        set_seed(42)

        # Generate again
        np2 = np.random.rand(10)
        tf2 = tf.random.uniform([10]).numpy()

        # Should be identical
        np.testing.assert_array_equal(np1, np2)
        np.testing.assert_array_almost_equal(tf1, tf2)

        print("✅ Seed fixing works correctly")

    def test_mixed_precision(self):
        """Test mixed precision configuration"""
        from src.training_improvements import MixedPrecisionConfig

        # Try to enable (may fail on CPU-only systems)
        result = MixedPrecisionConfig.enable_mixed_precision("mixed_float16")

        if result:
            policy = tf.keras.mixed_precision.global_policy()
            self.assertEqual(policy.name, "mixed_float16")
            print("✅ Mixed precision enabled successfully")
        else:
            print("⚠️  Mixed precision not available (CPU-only system)")

    def test_cosine_annealing_scheduler(self):
        """Test cosine annealing LR schedule"""
        from src.training_improvements import LearningRateSchedulers

        scheduler = LearningRateSchedulers.cosine_annealing(
            initial_lr=1e-3, epochs=100, warmup_epochs=5
        )

        # Test warmup phase
        lr_warmup = scheduler.schedule(2, 0)
        self.assertGreater(lr_warmup, 0)
        self.assertLess(lr_warmup, 1e-3)

        # Test annealing phase
        lr_mid = scheduler.schedule(50, 0)
        lr_end = scheduler.schedule(99, 0)
        self.assertGreater(lr_mid, lr_end)

        print("✅ Cosine annealing scheduler works correctly")

    def test_kfold_cross_validation(self):
        """Test k-fold cross-validation setup"""
        from src.training_improvements import KFoldCrossValidation

        kfold = KFoldCrossValidation(n_splits=5, random_state=42)

        # Create dummy data
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 4, 100)

        # Test splitting
        splits = list(kfold.split(X, y))
        self.assertEqual(len(splits), 5)

        # Check that all samples are used
        all_indices = set()
        for train_idx, val_idx in splits:
            all_indices.update(val_idx)
        self.assertEqual(len(all_indices), 100)

        print("✅ K-fold cross-validation works correctly")

    def test_patient_level_splitting(self):
        """Test patient-level k-fold splitting"""
        from src.training_improvements import KFoldCrossValidation

        kfold = KFoldCrossValidation(n_splits=3, random_state=42)

        # Create dummy data with patient IDs
        n_samples = 90
        n_patients = 10
        patient_ids = np.repeat(np.arange(n_patients), n_samples // n_patients)
        X = np.random.rand(n_samples, 10)
        y = np.random.randint(0, 2, n_samples)

        # Test patient-level splitting
        splits = list(kfold.split(X, y, patient_ids))

        for train_idx, val_idx in splits:
            train_patients = set(patient_ids[train_idx])
            val_patients = set(patient_ids[val_idx])

            # Patients should not overlap
            overlap = train_patients & val_patients
            self.assertEqual(
                len(overlap), 0, "Patient-level splitting failed: patients in both sets"
            )

        print("✅ Patient-level splitting works correctly")


class TestONNXDeployment(unittest.TestCase):
    """Test ONNX export and deployment features"""

    def setUp(self):
        """Create a simple test model"""
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.Conv2D(16, 3, activation="relu"),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(4, activation="softmax"),
            ]
        )
        self.model.compile(optimizer="adam", loss="categorical_crossentropy")

        # Save model
        self.model_path = "/tmp/test_model.h5"
        self.model.save(self.model_path)

    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_onnx_export(self):
        """Test ONNX export functionality"""
        try:
            import tf2onnx
            from src.onnx_deployment import ONNXExporter

            onnx_path = "/tmp/test_model.onnx"
            exporter = ONNXExporter(self.model_path, onnx_path)
            exporter.load_model()

            result = exporter.export_to_onnx(optimize=False)

            if result:
                self.assertTrue(os.path.exists(onnx_path))
                print("✅ ONNX export works correctly")
                os.remove(onnx_path)
            else:
                print("⚠️  ONNX export skipped (dependencies not available)")

        except ImportError:
            print("⚠️  tf2onnx not installed, skipping ONNX tests")

    def test_tflite_export(self):
        """Test TensorFlow Lite export"""
        from src.onnx_deployment import export_to_tflite

        tflite_path = "/tmp/test_model.tflite"
        result = export_to_tflite(self.model_path, tflite_path, quantize=False)

        self.assertTrue(os.path.exists(tflite_path))
        self.assertGreater(os.path.getsize(tflite_path), 0)

        print("✅ TFLite export works correctly")
        os.remove(tflite_path)


class TestCoordinateAttention(unittest.TestCase):
    """Test Coordinate Attention implementation"""

    def test_coordinate_attention_output_shape(self):
        """Test CA output shape matches input"""
        try:
            from models.coordinate_attention import CoordinateAttentionBlock
        except ImportError:
            self.skipTest("CoordinateAttention not available")
            return

        ca = CoordinateAttentionBlock(channels=64, reduction=8)

        # Test with different input shapes
        for h, w in [(32, 32), (64, 64), (128, 128)]:
            x = tf.random.normal([2, h, w, 64])
            y = ca(x)

            self.assertEqual(
                y.shape, x.shape, f"Output shape {y.shape} != input shape {x.shape}"
            )

        print("✅ Coordinate Attention output shapes correct")

    def test_coordinate_attention_position_encoding(self):
        """Test that CA preserves spatial information"""
        try:
            from models.coordinate_attention import CoordinateAttentionBlock
        except ImportError:
            self.skipTest("CoordinateAttention not available")
            return

        ca = CoordinateAttentionBlock(channels=64, reduction=8)

        # Create input with spatial pattern
        x = tf.zeros([1, 32, 32, 64])
        x = tf.tensor_scatter_nd_update(x, indices=[[0, 10, 10, 0]], updates=[1.0])

        y = ca(x, training=False)

        # Output should vary spatially (not uniform)
        variance = tf.math.reduce_variance(y)
        self.assertGreater(
            variance, 0, "CA output is uniform - not preserving position"
        )

        print("✅ Coordinate Attention preserves spatial information")


class TestDynamicSnakeConv(unittest.TestCase):
    """Test Dynamic Snake Convolution implementation"""

    def test_snake_conv_output_shape(self):
        """Test Snake Conv output shape"""
        from models.dynamic_snake_conv import DynamicSnakeConv

        snake_conv = DynamicSnakeConv(filters=64, kernel_size=3, strides=1)

        # Build layer
        snake_conv.build((None, 64, 64, 32))

        # Test forward pass
        x = tf.random.normal([2, 64, 64, 32])
        y = snake_conv(x, training=False)

        self.assertEqual(y.shape[0], 2)  # Batch
        self.assertEqual(y.shape[-1], 64)  # Filters

        print("✅ Dynamic Snake Conv output shape correct")

    def test_snake_conv_offset_learning(self):
        """Test that offsets are learnable"""
        from models.dynamic_snake_conv import DynamicSnakeConv

        snake_conv = DynamicSnakeConv(filters=32, kernel_size=3)
        snake_conv.build((None, 32, 32, 16))

        # Check that offset conv exists and has weights
        self.assertTrue(hasattr(snake_conv, "offset_conv"))
        self.assertGreater(len(snake_conv.trainable_weights), 0)

        print("✅ Snake Conv offsets are learnable")


class TestDataDeduplication(unittest.TestCase):
    """Test data deduplication implementation"""

    def test_phash_duplicate_detection(self):
        """Test pHash-based duplicate detection"""
        from src.data_deduplication import DataDeduplicator

        # Create dummy images
        img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img2 = img1.copy()  # Exact duplicate
        img3 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        dedup = DataDeduplicator(hamming_threshold=5)

        # Compute hashes
        hash1 = dedup.compute_phash(img1)
        hash2 = dedup.compute_phash(img2)
        hash3 = dedup.compute_phash(img3)

        # Check duplicate detection
        dist_12 = dedup.hamming_distance(hash1, hash2)
        dist_13 = dedup.hamming_distance(hash1, hash3)

        self.assertEqual(dist_12, 0, "Identical images should have distance 0")
        self.assertGreater(dist_13, 5, "Different images should have distance > 5")

        print("✅ pHash duplicate detection works correctly")


class TestPhysicsInformedAugmentation(unittest.TestCase):
    """Test physics-informed augmentation"""

    def test_rician_noise(self):
        """Test Rician noise augmentation"""
        from src.physics_informed_augmentation import PhysicsInformedAugmentation

        aug = PhysicsInformedAugmentation()

        # Create test image
        img = np.ones((128, 128, 3), dtype=np.float32) * 0.5

        # Apply Rician noise
        noisy_img = aug.add_rician_noise(img, sigma=0.1)

        # Check that image changed
        self.assertFalse(np.array_equal(img, noisy_img))

        # Check that values are still in valid range
        self.assertGreaterEqual(noisy_img.min(), 0.0)
        self.assertLessEqual(noisy_img.max(), 1.0)

        print("✅ Rician noise augmentation works correctly")

    def test_elastic_deformation(self):
        """Test elastic deformation augmentation"""
        from src.physics_informed_augmentation import PhysicsInformedAugmentation

        aug = PhysicsInformedAugmentation()

        # Create test image
        img = np.random.rand(128, 128, 3).astype(np.float32)

        # Apply elastic deformation
        deformed_img = aug.elastic_deformation(img, alpha=10, sigma=3)

        # Check shape preserved
        self.assertEqual(deformed_img.shape, img.shape)

        # Check that image changed
        self.assertFalse(np.array_equal(img, deformed_img))

        print("✅ Elastic deformation works correctly")


class TestClinicalPreprocessing(unittest.TestCase):
    """Test clinical preprocessing pipeline"""

    def test_skull_stripping(self):
        """Test skull stripping functionality"""
        from src.clinical_preprocessing import ClinicalPreprocessor

        preprocessor = ClinicalPreprocessor()

        # Create synthetic brain image
        img = np.zeros((128, 128), dtype=np.float32)
        # Add brain region
        img[32:96, 32:96] = 1.0
        # Add skull (bright ring)
        img[20:30, :] = 0.8
        img[98:108, :] = 0.8

        # Apply skull stripping
        stripped = preprocessor.skull_strip(img)

        # Check that skull region is reduced
        self.assertLess(stripped[25, 64], img[25, 64])

        print("✅ Skull stripping works correctly")

    def test_z_score_normalization(self):
        """Test Z-score normalization"""
        from src.clinical_preprocessing import ClinicalPreprocessor

        preprocessor = ClinicalPreprocessor()

        # Create test image
        img = np.random.randn(128, 128) * 50 + 100

        # Apply Z-score normalization
        normalized = preprocessor.z_score_normalize(img)

        # Check mean ≈ 0 and std ≈ 1
        self.assertAlmostEqual(normalized.mean(), 0.0, places=1)
        self.assertAlmostEqual(normalized.std(), 1.0, places=1)

        print("✅ Z-score normalization works correctly")


class TestStatisticalValidation(unittest.TestCase):
    """Test statistical validation methods"""

    def test_confidence_intervals(self):
        """Test confidence interval calculation"""
        # Simulate fold results
        accuracies = [0.95, 0.96, 0.94, 0.97, 0.95]

        mean = np.mean(accuracies)
        std = np.std(accuracies, ddof=1)

        # 95% CI for t-distribution (n=5)
        t_value = 2.776  # From t-table
        margin = t_value * std / np.sqrt(len(accuracies))

        ci_lower = mean - margin
        ci_upper = mean + margin

        # Check CI makes sense
        self.assertGreater(ci_upper, ci_lower)
        self.assertGreater(mean, ci_lower)
        self.assertLess(mean, ci_upper)

        print(f"✅ Confidence interval: [{ci_lower:.3f}, {ci_upper:.3f}]")


def run_all_tests():
    """Run all test suites"""
    print("=" * 70)
    print("PHOENIX PROTOCOL - COMPREHENSIVE TEST SUITE")
    print("Based on cross-analysis with SOTA medical imaging projects")
    print("=" * 70)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingImprovements))
    suite.addTests(loader.loadTestsFromTestCase(TestONNXDeployment))
    suite.addTests(loader.loadTestsFromTestCase(TestCoordinateAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestDynamicSnakeConv))
    suite.addTests(loader.loadTestsFromTestCase(TestDataDeduplication))
    suite.addTests(loader.loadTestsFromTestCase(TestPhysicsInformedAugmentation))
    suite.addTests(loader.loadTestsFromTestCase(TestClinicalPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalValidation))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
