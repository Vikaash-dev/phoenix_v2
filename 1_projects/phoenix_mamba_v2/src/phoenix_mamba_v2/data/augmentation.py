from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from ..core.interfaces import BaseAugmentor

class PhysicsInformedAugmentor(BaseAugmentor):
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
        self.elastic_alpha_range = elastic_alpha_range
        self.elastic_sigma = elastic_sigma
        self.rician_noise_sigma_range = rician_noise_sigma_range
        self.apply_probability = apply_probability
        self._rng = np.random.RandomState()

    def augment(self, image: np.ndarray, label: Any = None) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
        """Apply augmentation pipeline."""
        augmented_image = image.copy()

        # 1. Elastic Deformation (simulates tissue deformation)
        if self._rng.rand() < self.apply_probability:
            augmented_image = self._elastic_deformation(augmented_image)
            # If label is a mask, we should ideally deform it too, but omitting for now or implementing if label is provided

        # 2. Rician Noise (simulates MRI acquisition noise)
        if self._rng.rand() < self.apply_probability:
            augmented_image = self._rician_noise(augmented_image)

        # 3. Intensity Inhomogeneity (Bias Field)
        if self._rng.rand() < (self.apply_probability * 0.5):
            augmented_image = self._intensity_inhomogeneity(augmented_image)

        # 4. Ghosting Artifact (Motion)
        if self._rng.rand() < (self.apply_probability * 0.3):
            augmented_image = self._ghosting_artifact(augmented_image)

        if label is not None:
            return augmented_image, label
        return augmented_image

    def _elastic_deformation(self, image: np.ndarray) -> np.ndarray:
        """Apply elastic deformation."""
        alpha = self._rng.uniform(*self.elastic_alpha_range)
        sigma = self.elastic_sigma
        shape = image.shape[:2]

        dx = gaussian_filter((self._rng.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((self._rng.rand(*shape) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        if len(image.shape) == 3:
            deformed = np.zeros_like(image)
            for i in range(image.shape[2]):
                deformed[:, :, i] = map_coordinates(
                    image[:, :, i], indices, order=1, mode='reflect'
                ).reshape(shape)
        else:
            deformed = map_coordinates(
                image, indices, order=1, mode='reflect'
            ).reshape(shape)

        return deformed

    def _rician_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Rician noise (Gaussian approx for high SNR)."""
        sigma = self._rng.uniform(*self.rician_noise_sigma_range)
        noise_real = self._rng.normal(0, sigma, image.shape)
        noise_imag = self._rng.normal(0, sigma, image.shape)

        real = image + noise_real
        imag = noise_imag
        noisy = np.sqrt(real**2 + imag**2)
        return np.clip(noisy, 0, 1)

    def _intensity_inhomogeneity(self, image: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """Simulate RF coil inhomogeneity."""
        shape = image.shape[:2]
        bias_field = self._rng.randn(*shape)
        bias_field = gaussian_filter(bias_field, sigma=shape[0] / 4)

        # Normalize bias field
        bias_min, bias_max = bias_field.min(), bias_field.max()
        bias_field = (bias_field - bias_min) / (bias_max - bias_min + 1e-8)
        bias_field = 1 + strength * (bias_field - 0.5) * 2

        if len(image.shape) == 3:
            bias_field = np.expand_dims(bias_field, axis=-1)

        return np.clip(image * bias_field, 0, 1)

    def _ghosting_artifact(self, image: np.ndarray, num_ghosts: int = 2, intensity: float = 0.2) -> np.ndarray:
        """Simulate motion ghosting."""
        result = image.copy()
        h, w = image.shape[:2]

        for _ in range(num_ghosts):
            shift_y = self._rng.randint(-h // 20, h // 20)
            shift_x = self._rng.randint(-w // 20, w // 20)
            ghost = np.roll(image, shift=(shift_y, shift_x), axis=(0, 1))
            result += intensity * ghost

        return np.clip(result, 0, 1)
