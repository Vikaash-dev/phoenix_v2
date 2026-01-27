
import unittest
import numpy as np
import tensorflow as tf
from src.physics_informed_augmentation import PhysicsInformedAugmentation

class TestPhysicsInformedAugmentation(unittest.TestCase):
    def setUp(self):
        self.augmentor = PhysicsInformedAugmentation(apply_probability=1.0)
        self.image = np.zeros((224, 224, 3), dtype=np.float32)

    def test_reproducibility(self):
        """Test that augmentation is reproducible when global seed is set."""
        # Run 1
        np.random.seed(42)
        aug1 = self.augmentor.augment(self.image)
        
        # Run 2
        np.random.seed(42)
        aug2 = self.augmentor.augment(self.image)
        
        np.testing.assert_array_equal(aug1, aug2, err_msg="Augmentation should be reproducible with global seed")

    def test_random_state_override(self):
        """Test that passing a RandomState overrides global seed."""
        rs1 = np.random.RandomState(42)
        aug1 = self.augmentor.augment(self.image, random_state=rs1)
        
        rs2 = np.random.RandomState(42)
        aug2 = self.augmentor.augment(self.image, random_state=rs2)
        
        np.testing.assert_array_equal(aug1, aug2, err_msg="Augmentation should be reproducible with explicit RandomState")

    def test_divergence(self):
        """Test that different seeds produce different results."""
        np.random.seed(42)
        aug1 = self.augmentor.augment(self.image)
        
        np.random.seed(43)
        aug2 = self.augmentor.augment(self.image)
        
        # It's statistically possible but highly unlikely they are identical
        self.assertFalse(np.array_equal(aug1, aug2), "Different seeds should produce different augmentations")

    def test_output_shape(self):
        aug = self.augmentor.augment(self.image)
        self.assertEqual(aug.shape, self.image.shape)

if __name__ == '__main__':
    unittest.main()
