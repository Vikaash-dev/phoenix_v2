"""
Comprehensive Test Suite for Phoenix Protocol Implementation
Tests all major components to ensure correctness and functionality.
"""

import sys
import os
from pathlib import Path
# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root.parent))

import numpy as np

# Test results storage
test_results = []

def run_test(test_name, test_func):
    """Run a test and record the result."""
    try:
        test_func()
        test_results.append((test_name, "PASS", None))
        print(f"✓ {test_name}")
        return True
    except Exception as e:
        test_results.append((test_name, "FAIL", str(e)))
        print(f"✗ {test_name}: {e}")
        return False

# Test 1: Module Imports
def test_imports():
    """Test that all modules can be imported."""
    from models.neurosnake_model import create_neurosnake_model, create_baseline_model
    from src.phoenix_optimizer import create_adan_optimizer, create_focal_loss
    from src.physics_informed_augmentation import PhysicsInformedAugmentation
    from src.data_deduplication import ImageDeduplicator
    import config

# Test 2: Physics-Informed Augmentation
def test_physics_augmentation():
    """Test physics-informed augmentation functions."""
    from src.physics_informed_augmentation import PhysicsInformedAugmentation
    
    augmentor = PhysicsInformedAugmentation(
        elastic_alpha_range=(30, 40),
        elastic_sigma=5.0,
        rician_noise_sigma_range=(0.01, 0.05),
        apply_probability=0.5
    )
    
    # Test with dummy image
    test_image = np.random.rand(224, 224, 3).astype(np.float32)
    
    # Test elastic deformation
    deformed = augmentor.elastic_deformation(test_image, alpha=35, sigma=5.0)
    assert deformed.shape == test_image.shape, "Elastic deformation changed image shape"
    assert 0 <= deformed.min() <= 1, "Elastic deformation values out of range"
    
    # Test Rician noise
    noisy = augmentor.rician_noise(test_image, sigma=0.02)
    assert noisy.shape == test_image.shape, "Rician noise changed image shape"
    assert 0 <= noisy.min() <= 1, "Rician noise values out of range"
    
    # Test full augmentation pipeline
    augmented = augmentor.augment(test_image)
    assert augmented.shape == test_image.shape, "Augmentation changed image shape"

# Test 3: Data Deduplication
def test_deduplication():
    """Test pHash-based deduplication."""
    from src.data_deduplication import ImageDeduplicator
    
    deduplicator = ImageDeduplicator(hamming_threshold=5)
    
    # Test hash computation (requires actual image, so we skip if PIL not available)
    try:
        from PIL import Image
        # Create a test image
        test_img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        temp_path = "/tmp/test_image.png"
        test_img.save(temp_path)
        
        phash = deduplicator.compute_phash(temp_path)
        assert phash is not None, "pHash computation failed"
        
        # Clean up
        os.remove(temp_path)
    except ImportError:
        print("  (Skipping pHash test - PIL not available)")

# Test 4: Phoenix Optimizer
def test_phoenix_optimizer():
    """Test Adan optimizer and Focal Loss."""
    from src.phoenix_optimizer import create_adan_optimizer, create_focal_loss
    
    # Test Adan optimizer creation
    optimizer = create_adan_optimizer(
        learning_rate=0.001,
        beta1=0.98,
        beta2=0.92,
        beta3=0.99
    )
    assert optimizer is not None, "Adan optimizer creation failed"
    
    # Test Focal Loss creation
    loss = create_focal_loss(alpha=0.25, gamma=2.0)
    assert loss is not None, "Focal Loss creation failed"
    
    # Test loss computation (if tensorflow available)
    try:
        import tensorflow as tf
        y_true = tf.constant([[1, 0], [0, 1]], dtype=tf.float32)
        y_pred = tf.constant([[0.9, 0.1], [0.2, 0.8]], dtype=tf.float32)
        loss_value = loss(y_true, y_pred)
        assert loss_value is not None, "Focal Loss computation failed"
    except ImportError:
        print("  (Skipping TensorFlow-based tests)")

# Test 5: Configuration
def test_config():
    """Test configuration parameters."""
    import config
    
    # Check essential parameters exist
    assert hasattr(config, 'IMG_HEIGHT'), "IMG_HEIGHT not defined"
    assert hasattr(config, 'IMG_WIDTH'), "IMG_WIDTH not defined"
    assert hasattr(config, 'BATCH_SIZE'), "BATCH_SIZE not defined"
    assert hasattr(config, 'LEARNING_RATE'), "LEARNING_RATE not defined"
    
    # Check reasonable values
    assert config.IMG_HEIGHT == 224, "IMG_HEIGHT should be 224"
    assert config.IMG_WIDTH == 224, "IMG_WIDTH should be 224"
    assert config.BATCH_SIZE > 0, "BATCH_SIZE should be positive"
    assert config.LEARNING_RATE > 0, "LEARNING_RATE should be positive"

# Test 6: Model Architecture (requires TensorFlow)
def test_model_creation():
    """Test NeuroSnake and baseline model creation."""
    try:
        import tensorflow as tf
        from models.neurosnake_model import create_neurosnake_model, create_baseline_model
        
        # Test NeuroSnake creation
        neurosnake = create_neurosnake_model(
            input_shape=(224, 224, 3),
            num_classes=2,
            use_mobilevit=True
        )
        assert neurosnake is not None, "NeuroSnake creation failed"
        
        # Test baseline creation
        baseline = create_baseline_model(
            input_shape=(224, 224, 3),
            num_classes=2
        )
        assert baseline is not None, "Baseline model creation failed"
        
        print(f"  NeuroSnake parameters: {neurosnake.count_params():,}")
        print(f"  Baseline parameters: {baseline.count_params():,}")
        
    except ImportError:
        print("  (Skipping model tests - TensorFlow not available)")

# Test 7: Documentation Check
def test_documentation():
    """Test that all documentation files exist."""
    docs = [
        'README.md',
        'PHOENIX_PROTOCOL.md',
        'SECURITY_ANALYSIS.md',
        'requirements.txt',
        'config.py'
    ]
    
    for doc in docs:
        assert os.path.exists(doc), f"{doc} not found"

# Test 8: Code Structure
def test_code_structure():
    """Test that all essential files exist."""
    essential_files = [
        'models/dynamic_snake_conv.py',
        'models/neurosnake_model.py',
        'src/phoenix_optimizer.py',
        'src/physics_informed_augmentation.py',
        'src/data_deduplication.py',
        'src/train_phoenix.py',
        'src/int8_quantization.py',
        'src/comparative_analysis.py'
    ]
    
    for file in essential_files:
        assert os.path.exists(file), f"{file} not found"

# Run all tests
def main():
    print("="*80)
    print("PHOENIX PROTOCOL - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print()
    
    tests = [
        ("Module Imports", test_imports),
        ("Physics-Informed Augmentation", test_physics_augmentation),
        ("Data Deduplication", test_deduplication),
        ("Phoenix Optimizer", test_phoenix_optimizer),
        ("Configuration", test_config),
        ("Model Architecture", test_model_creation),
        ("Documentation", test_documentation),
        ("Code Structure", test_code_structure),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        if run_test(test_name, test_func):
            passed += 1
        else:
            failed += 1
        print()
    
    print("="*80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*80)
    
    # Print detailed failures
    if failed > 0:
        print("\nFailed Tests:")
        for name, status, error in test_results:
            if status == "FAIL":
                print(f"  - {name}: {error}")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
