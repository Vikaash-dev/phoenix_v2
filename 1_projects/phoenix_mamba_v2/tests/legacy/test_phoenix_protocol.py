"""
Comprehensive Test Suite for Phoenix Protocol Implementation
Tests all major components to ensure correctness and functionality.
"""

import sys
import os
import pytest
import numpy as np

# Add project root to path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Test 1: Module Imports
def test_imports():
    """Test that all modules can be imported."""
    from src.models.legacy.neurosnake_model import create_neurosnake_model, create_baseline_model
    from src.legacy.phoenix_optimizer import create_adan_optimizer, create_focal_loss
    from src.legacy.physics_informed_augmentation import PhysicsInformedAugmentation
    from src.legacy.data_deduplication import ImageDeduplicator
    import src.legacy.config as config

# Test 2: Physics-Informed Augmentation
def test_physics_augmentation():
    """Test physics-informed augmentation functions."""
    from src.legacy.physics_informed_augmentation import PhysicsInformedAugmentation
    
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
    
    # Test Rician noise
    noisy = augmentor.rician_noise(test_image, sigma=0.02)
    assert noisy.shape == test_image.shape, "Rician noise changed image shape"
    
    # Test full augmentation pipeline
    augmented = augmentor.augment(test_image)
    assert augmented.shape == test_image.shape, "Augmentation changed image shape"

# Test 3: Data Deduplication
def test_deduplication():
    """Test pHash-based deduplication."""
    from src.legacy.data_deduplication import ImageDeduplicator
    
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
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except ImportError:
        pytest.skip("PIL not available")

# Test 4: Phoenix Optimizer
@pytest.mark.skip("Incompatible with Keras 3")
def test_phoenix_optimizer():
    """Test Adan optimizer and Focal Loss."""
    from src.legacy.phoenix_optimizer import create_adan_optimizer, create_focal_loss
    
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
        pytest.skip("TensorFlow not available")

# Test 5: Configuration
def test_config():
    """Test configuration parameters."""
    import src.legacy.config as config
    
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
        from src.models.legacy.neurosnake_model import create_neurosnake_model, create_baseline_model
        
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
        
    except ImportError:
        pytest.skip("TensorFlow not available")

# Test 8: Code Structure
def test_code_structure():
    """Test that all essential files exist."""
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    essential_files = [
        'src/models/legacy/dynamic_snake_conv.py',
        'src/models/legacy/neurosnake_model.py',
        'src/legacy/phoenix_optimizer.py',
        'src/legacy/physics_informed_augmentation.py',
        'src/legacy/data_deduplication.py',
        'src/legacy/train_phoenix.py',
        'src/legacy/int8_quantization.py',
        'src/legacy/comparative_analysis.py'
    ]
    
    for file in essential_files:
        full_path = os.path.join(base_path, file)
        assert os.path.exists(full_path), f"{file} not found at {full_path}"
