from typing import Any, Dict, Optional, Tuple
import numpy as np
import cv2
from scipy import ndimage
from ..core.interfaces import BasePreprocessor

class ClinicalPreprocessor(BasePreprocessor):
    """
    Clinical-grade MRI preprocessing pipeline.

    Includes:
    - Skull stripping (threshold-based)
    - N4 bias field correction (simplified)
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Z-score normalization
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        apply_skull_stripping: bool = True,
        apply_bias_correction: bool = True,
        apply_clahe: bool = True,
        apply_znorm: bool = True,
        clip_percentile: float = 99.5
    ):
        self.target_size = target_size
        self.apply_skull_stripping = apply_skull_stripping
        self.apply_bias_correction = apply_bias_correction
        self.apply_clahe = apply_clahe
        self.apply_znorm = apply_znorm
        self.clip_percentile = clip_percentile

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "ClinicalPreprocessor",
            "target_size": self.target_size,
            "skull_strip": self.apply_skull_stripping,
            "bias_correction": self.apply_bias_correction,
            "clahe": self.apply_clahe,
            "znorm": self.apply_znorm,
            "clip_percentile": self.clip_percentile
        }

    def process(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply the clinical preprocessing pipeline."""
        # Ensure image is float32 [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        processed = image.copy()
        brain_mask = None

        # 1. Skull Stripping
        if self.apply_skull_stripping:
            processed, brain_mask = self._skull_strip(processed)

        # 2. Bias Field Correction
        if self.apply_bias_correction:
            processed = self._bias_field_correction(processed)

        # 3. Intensity Clipping
        processed = self._intensity_clipping(processed)

        # 4. CLAHE
        if self.apply_clahe:
            processed = self._apply_clahe(processed)

        # 5. Z-Score Normalization
        if self.apply_znorm:
            processed = self._z_score_normalize(processed, brain_mask)
            # Re-scale to [0, 1] for typical DL input
            processed = (processed - processed.min()) / (processed.max() - processed.min() + 1e-8)

        # 6. Resize
        if processed.shape[:2] != self.target_size:
            processed = cv2.resize(processed, self.target_size, interpolation=cv2.INTER_LINEAR)

        return np.clip(processed, 0, 1).astype(np.float32)

    def _skull_strip(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple skull stripping using thresholding and morphology."""
        # Convert to grayscale for mask generation
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Keep largest component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            brain_mask = (labels == largest_label).astype(np.uint8)
        else:
            brain_mask = binary

        # Apply mask
        if len(image.shape) == 3:
            brain_mask_3d = np.stack([brain_mask] * image.shape[2], axis=2)
            stripped = image * brain_mask_3d
        else:
            stripped = image * brain_mask

        return stripped, brain_mask

    def _bias_field_correction(self, image: np.ndarray) -> np.ndarray:
        """Simplified N4 bias correction using low-pass filtering."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        else:
            gray = image.copy()

        # Estimate bias field (low frequency)
        bias_field = ndimage.gaussian_filter(gray, sigma=20)
        bias_field = np.maximum(bias_field, 0.01) # Avoid div by zero

        # Correct
        if len(image.shape) == 3:
            corrected = np.zeros_like(image)
            for c in range(image.shape[2]):
                channel = image[:, :, c]
                channel_bias = ndimage.gaussian_filter(channel, sigma=20)
                channel_bias = np.maximum(channel_bias, 0.01)
                corrected[:, :, c] = channel / channel_bias
        else:
            corrected = gray / bias_field

        return np.clip(corrected, 0, 1)

    def _apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        img_uint8 = (image * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

        if len(image.shape) == 3:
            enhanced = np.zeros_like(img_uint8)
            for c in range(image.shape[2]):
                enhanced[:, :, c] = clahe.apply(img_uint8[:, :, c])
        else:
            enhanced = clahe.apply(img_uint8)

        return enhanced.astype(np.float32) / 255.0

    def _intensity_clipping(self, image: np.ndarray) -> np.ndarray:
        upper_bound = np.percentile(image, self.clip_percentile)
        if upper_bound > 0:
            return np.clip(image, 0, upper_bound) / upper_bound
        return image

    def _z_score_normalize(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if mask is not None:
            masked_values = image[mask > 0]
            if len(masked_values) > 0:
                mean, std = masked_values.mean(), masked_values.std()
            else:
                mean, std = image.mean(), image.std()
        else:
            mean, std = image.mean(), image.std()

        if std < 1e-8: std = 1.0
        return (image - mean) / std


class FastPreprocessor(BasePreprocessor):
    """
    Fast preprocessing for rapid experimentation.
    Only resizes and normalizes intensity to [0,1].
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "FastPreprocessor",
            "target_size": self.target_size
        }

    def process(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Resize
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

        # Simple Min-Max Normalization
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        return np.clip(image, 0, 1)
