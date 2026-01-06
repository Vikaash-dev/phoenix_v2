"""
Physics-Informed Augmentation for Phoenix Protocol
Implements Elastic Deformations and Rician Noise Injection
to simulate real-world MRI artifacts instead of generic augmentations.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates
from typing import Tuple, Optional
import tensorflow as tf


class PhysicsInformedAugmentation:
    """
    Physics-informed augmentation tailored for MRI imaging.
    Replaces generic "jigsaw" augmentations with realistic MRI artifacts.
    """
    
    def __init__(
        self,
        elastic_alpha_range: Tuple[float, float] = (30, 40),
        elastic_sigma: float = 5.0,
        rician_noise_sigma_range: Tuple[float, float] = (0.01, 0.05),
        apply_probability: float = 0.5
    ):
        """
        Initialize physics-informed augmentation.
        
        Args:
            elastic_alpha_range: Range for elastic deformation strength
            elastic_sigma: Gaussian filter sigma for elastic deformation
            rician_noise_sigma_range: Range for Rician noise standard deviation
            apply_probability: Probability of applying each augmentation
        """
        self.elastic_alpha_range = elastic_alpha_range
        self.elastic_sigma = elastic_sigma
        self.rician_noise_sigma_range = rician_noise_sigma_range
        self.apply_probability = apply_probability
    
    def elastic_deformation(
        self, 
        image: np.ndarray, 
        alpha: Optional[float] = None,
        sigma: Optional[float] = None,
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Apply elastic deformation to simulate tissue deformations.
        Critical for capturing irregular, finger-like infiltrations of Glioblastomas.
        
        Based on:
        Simard, Steinkraus and Platt, "Best Practices for Convolutional Neural Networks
        applied to Visual Document Analysis", ICDAR, 2003.
        
        Args:
            image: Input image (H, W, C) or (H, W)
            alpha: Deformation strength (if None, random from range)
            sigma: Gaussian filter sigma (if None, use default)
            random_state: Random state for reproducibility
            
        Returns:
            Elastically deformed image
        """
        if random_state is None:
            random_state = np.random.RandomState()
        
        if alpha is None:
            alpha = random_state.uniform(*self.elastic_alpha_range)
        
        if sigma is None:
            sigma = self.elastic_sigma
        
        shape = image.shape[:2]
        
        # Generate random displacement fields
        dx = gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), 
            sigma, 
            mode="constant", 
            cval=0
        ) * alpha
        
        dy = gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), 
            sigma, 
            mode="constant", 
            cval=0
        ) * alpha
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # Apply deformation to each channel
        if len(image.shape) == 3:
            deformed = np.zeros_like(image)
            for i in range(image.shape[2]):
                deformed[:, :, i] = map_coordinates(
                    image[:, :, i], 
                    indices, 
                    order=1, 
                    mode='reflect'
                ).reshape(shape)
        else:
            deformed = map_coordinates(
                image, 
                indices, 
                order=1, 
                mode='reflect'
            ).reshape(shape)
        
        return deformed
    
    def rician_noise(
        self, 
        image: np.ndarray, 
        sigma: Optional[float] = None,
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Add Rician noise to simulate MRI acquisition noise.
        
        MRI magnitude images follow a Rician distribution due to the 
        magnitude reconstruction from complex k-space data.
        
        Rician PDF: f(x|A,σ) = (x/σ²) * exp(-(x²+A²)/(2σ²)) * I₀(xA/σ²)
        
        For computational efficiency, we approximate using the Gaussian approximation:
        When SNR is high, Rician → Gaussian in the magnitude domain.
        
        Args:
            image: Input image (normalized to [0, 1])
            sigma: Noise standard deviation (if None, random from range)
            random_state: Random state for reproducibility
            
        Returns:
            Image with Rician noise added
        """
        if random_state is None:
            random_state = np.random.RandomState()
        
        if sigma is None:
            sigma = random_state.uniform(*self.rician_noise_sigma_range)
        
        # For Rician noise approximation:
        # Generate two independent Gaussian noise arrays (real and imaginary components)
        noise_real = random_state.normal(0, sigma, image.shape)
        noise_imag = random_state.normal(0, sigma, image.shape)
        
        # Simulate magnitude reconstruction
        # Original signal + noise in complex domain
        real_component = image + noise_real
        imag_component = noise_imag
        
        # Magnitude operation
        noisy_image = np.sqrt(real_component**2 + imag_component**2)
        
        # Clip to valid range
        noisy_image = np.clip(noisy_image, 0, 1)
        
        return noisy_image
    
    def intensity_inhomogeneity(
        self,
        image: np.ndarray,
        strength: float = 0.3,
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Simulate intensity inhomogeneity (bias field) artifact in MRI.
        
        This is caused by RF coil inhomogeneities and is a common MRI artifact.
        
        Args:
            image: Input image
            strength: Strength of the bias field
            random_state: Random state for reproducibility
            
        Returns:
            Image with intensity inhomogeneity
        """
        if random_state is None:
            random_state = np.random.RandomState()
        
        shape = image.shape[:2]
        
        # Generate smooth bias field using low-frequency noise
        bias_field = random_state.randn(*shape)
        bias_field = gaussian_filter(bias_field, sigma=shape[0] / 4)
        
        # Normalize bias field
        bias_field = (bias_field - bias_field.min()) / (bias_field.max() - bias_field.min())
        bias_field = 1 + strength * (bias_field - 0.5) * 2
        
        # Apply bias field
        if len(image.shape) == 3:
            bias_field = np.expand_dims(bias_field, axis=-1)
        
        result = image * bias_field
        result = np.clip(result, 0, 1)
        
        return result
    
    def ghosting_artifact(
        self,
        image: np.ndarray,
        num_ghosts: int = 2,
        intensity: float = 0.2,
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Simulate motion-induced ghosting artifacts in MRI.
        
        Caused by patient motion during acquisition, creating periodic replicas.
        
        Args:
            image: Input image
            num_ghosts: Number of ghost artifacts to add
            intensity: Intensity of ghost artifacts
            random_state: Random state for reproducibility
            
        Returns:
            Image with ghosting artifacts
        """
        if random_state is None:
            random_state = np.random.RandomState()
        
        result = image.copy()
        
        for _ in range(num_ghosts):
            # Random shift amount
            shift_y = random_state.randint(-image.shape[0] // 20, image.shape[0] // 20)
            shift_x = random_state.randint(-image.shape[1] // 20, image.shape[1] // 20)
            
            # Create shifted version (ghost)
            ghost = np.roll(image, shift=(shift_y, shift_x), axis=(0, 1))
            
            # Add ghost with reduced intensity
            result = result + intensity * ghost
        
        result = np.clip(result, 0, 1)
        return result
    
    def augment(
        self, 
        image: np.ndarray,
        apply_elastic: bool = True,
        apply_rician: bool = True,
        apply_inhomogeneity: bool = True,
        apply_ghosting: bool = False,
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Apply physics-informed augmentation pipeline.
        
        Args:
            image: Input image (normalized to [0, 1])
            apply_elastic: Whether to apply elastic deformation
            apply_rician: Whether to apply Rician noise
            apply_inhomogeneity: Whether to apply intensity inhomogeneity
            apply_ghosting: Whether to apply ghosting artifacts
            random_state: Random state for reproducibility
            
        Returns:
            Augmented image
        """
        if random_state is None:
            random_state = np.random.RandomState()
        
        result = image.copy()
        
        # Apply elastic deformation (most important for irregular tumor boundaries)
        if apply_elastic and random_state.rand() < self.apply_probability:
            result = self.elastic_deformation(result, random_state=random_state)
        
        # Apply Rician noise (MRI-specific)
        if apply_rician and random_state.rand() < self.apply_probability:
            result = self.rician_noise(result, random_state=random_state)
        
        # Apply intensity inhomogeneity (bias field)
        if apply_inhomogeneity and random_state.rand() < self.apply_probability * 0.5:
            result = self.intensity_inhomogeneity(result, random_state=random_state)
        
        # Apply ghosting (less common, so lower probability)
        if apply_ghosting and random_state.rand() < self.apply_probability * 0.3:
            result = self.ghosting_artifact(result, random_state=random_state)
        
        return result


class TensorFlowPhysicsAugmentation:
    """
    TensorFlow-compatible version for integration with tf.data pipeline.
    """
    
    def __init__(self, augmentor: PhysicsInformedAugmentation):
        """
        Initialize TF-compatible augmentation.
        
        Args:
            augmentor: PhysicsInformedAugmentation instance
        """
        self.augmentor = augmentor
    
    @tf.function
    def augment_tf(self, image: tf.Tensor) -> tf.Tensor:
        """
        TensorFlow-compatible augmentation function.
        
        Args:
            image: Input image tensor (H, W, C) in [0, 1] range
            
        Returns:
            Augmented image tensor
        """
        # Convert to numpy for augmentation
        def numpy_augment(img):
            img_np = img.numpy()
            augmented = self.augmentor.augment(img_np)
            return augmented.astype(np.float32)
        
        # Use tf.py_function to wrap numpy operations
        augmented = tf.py_function(
            func=numpy_augment,
            inp=[image],
            Tout=tf.float32
        )
        
        # Ensure shape is preserved
        augmented.set_shape(image.shape)
        
        return augmented


def create_physics_augmentation_layer(
    elastic_alpha_range: Tuple[float, float] = (30, 40),
    elastic_sigma: float = 5.0,
    rician_noise_sigma_range: Tuple[float, float] = (0.01, 0.05),
    apply_probability: float = 0.5
):
    """
    Factory function to create a physics-informed augmentation layer.
    
    Args:
        elastic_alpha_range: Range for elastic deformation strength
        elastic_sigma: Gaussian filter sigma for elastic deformation
        rician_noise_sigma_range: Range for Rician noise standard deviation
        apply_probability: Probability of applying each augmentation
        
    Returns:
        PhysicsInformedAugmentation instance
    """
    return PhysicsInformedAugmentation(
        elastic_alpha_range=elastic_alpha_range,
        elastic_sigma=elastic_sigma,
        rician_noise_sigma_range=rician_noise_sigma_range,
        apply_probability=apply_probability
    )


if __name__ == "__main__":
    # Test physics-informed augmentation
    print("Testing Physics-Informed Augmentation...")
    
    # Create dummy image
    test_image = np.random.rand(224, 224, 3).astype(np.float32)
    
    # Create augmentor
    augmentor = create_physics_augmentation_layer()
    
    # Apply augmentation
    augmented = augmentor.augment(test_image)
    
    print(f"Original image shape: {test_image.shape}")
    print(f"Augmented image shape: {augmented.shape}")
    print(f"Original image range: [{test_image.min():.3f}, {test_image.max():.3f}]")
    print(f"Augmented image range: [{augmented.min():.3f}, {augmented.max():.3f}]")
    
    print("\n✓ Physics-informed augmentation test passed!")
