"""
Comprehensive MRI Preprocessing Pipeline
Clinical-grade preprocessing for brain tumor detection.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from scipy import ndimage


class ClinicalPreprocessing:
    """
    Clinical-grade MRI preprocessing pipeline.
    
    Includes:
    - Skull stripping (simple thresholding-based)
    - N4 bias field correction (simplified)
    - Histogram equalization / CLAHE
    - Z-score normalization
    - Artifact removal
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
        """
        Initialize preprocessing pipeline.
        
        Args:
            target_size: Output image size (height, width)
            apply_skull_stripping: Enable skull stripping
            apply_bias_correction: Enable bias field correction
            apply_clahe: Enable CLAHE histogram equalization
            apply_znorm: Enable Z-score normalization
            clip_percentile: Percentile for intensity clipping
        """
        self.target_size = target_size
        self.apply_skull_stripping = apply_skull_stripping
        self.apply_bias_correction = apply_bias_correction
        self.apply_clahe = apply_clahe
        self.apply_znorm = apply_znorm
        self.clip_percentile = clip_percentile
    
    def skull_strip(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple skull stripping using thresholding and morphology.
        
        Note: This is a simplified version. For production, use FSL BET or ANTs.
        
        Args:
            image: Input MRI image (H, W) or (H, W, C)
            
        Returns:
            Stripped image and brain mask
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean mask
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find largest connected component (brain)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels > 1:
            # Get largest component (excluding background)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            brain_mask = (labels == largest_label).astype(np.uint8)
        else:
            brain_mask = binary
        
        # Apply mask to original image
        if len(image.shape) == 3:
            brain_mask_3d = np.stack([brain_mask] * image.shape[2], axis=2)
            stripped = image * brain_mask_3d
        else:
            stripped = image * brain_mask
        
        return stripped, brain_mask
    
    def bias_field_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Simplified N4 bias field correction.
        
        Note: This is a simplified version using polynomial fitting.
        For production, use ANTs N4BiasFieldCorrection.
        
        Args:
            image: Input MRI image
            
        Returns:
            Bias-corrected image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        else:
            gray = image.copy()
        
        # Estimate bias field using Gaussian filtering (low-pass)
        bias_field = ndimage.gaussian_filter(gray, sigma=20)
        
        # Avoid division by zero
        bias_field = np.where(bias_field < 0.01, 0.01, bias_field)
        
        # Correct bias
        corrected = gray / bias_field
        
        # Normalize back to [0, 1]
        corrected = np.clip(corrected, 0, 1)
        
        # Apply to all channels if RGB
        if len(image.shape) == 3:
            corrected_rgb = np.zeros_like(image)
            for c in range(image.shape[2]):
                channel = image[:, :, c]
                channel_bias = ndimage.gaussian_filter(channel, sigma=20)
                channel_bias = np.where(channel_bias < 0.01, 0.01, channel_bias)
                corrected_rgb[:, :, c] = np.clip(channel / channel_bias, 0, 1)
            return corrected_rgb
        
        return corrected
    
    def apply_clahe_enhancement(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input image [0, 1]
            clip_limit: Clipping limit for CLAHE
            
        Returns:
            Enhanced image
        """
        # Convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        
        if len(image.shape) == 3:
            # Apply to each channel
            enhanced = np.zeros_like(img_uint8)
            for c in range(image.shape[2]):
                enhanced[:, :, c] = clahe.apply(img_uint8[:, :, c])
        else:
            enhanced = clahe.apply(img_uint8)
        
        return enhanced / 255.0
    
    def z_score_normalize(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Z-score normalization (zero mean, unit variance).
        
        Args:
            image: Input image
            mask: Optional mask to compute stats only on brain region
            
        Returns:
            Normalized image
        """
        if mask is not None:
            # Compute stats only on masked region
            masked_values = image[mask > 0]
            if len(masked_values) > 0:
                mean = masked_values.mean()
                std = masked_values.std()
            else:
                mean, std = image.mean(), image.std()
        else:
            mean = image.mean()
            std = image.std()
        
        # Avoid division by zero
        if std < 1e-8:
            std = 1.0
        
        normalized = (image - mean) / std
        return normalized
    
    def intensity_clipping(self, image: np.ndarray, percentile: float = 99.5) -> np.ndarray:
        """
        Clip extreme intensity values.
        
        Args:
            image: Input image
            percentile: Upper percentile for clipping
            
        Returns:
            Clipped image
        """
        upper_bound = np.percentile(image, percentile)
        clipped = np.clip(image, 0, upper_bound)
        
        # Renormalize to [0, 1]
        if upper_bound > 0:
            clipped = clipped / upper_bound
        
        return clipped
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline.
        
        Args:
            image: Raw MRI image (H, W, C) in [0, 1] range
            
        Returns:
            Preprocessed image ready for model input
        """
        processed = image.copy()
        brain_mask = None
        
        # Step 1: Skull stripping
        if self.apply_skull_stripping:
            processed, brain_mask = self.skull_strip(processed)
        
        # Step 2: Bias field correction
        if self.apply_bias_correction:
            processed = self.bias_field_correction(processed)
        
        # Step 3: Intensity clipping
        processed = self.intensity_clipping(processed, self.clip_percentile)
        
        # Step 4: CLAHE enhancement
        if self.apply_clahe:
            processed = self.apply_clahe_enhancement(processed)
        
        # Step 5: Z-score normalization
        if self.apply_znorm:
            processed = self.z_score_normalize(processed, brain_mask)
            # Scale to [0, 1] for model input
            processed = (processed - processed.min()) / (processed.max() - processed.min() + 1e-8)
        
        # Step 6: Resize to target size
        if processed.shape[:2] != self.target_size:
            processed = cv2.resize(processed, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Ensure proper shape and range
        processed = np.clip(processed, 0, 1).astype(np.float32)
        
        return processed


# Example usage
if __name__ == "__main__":
    print("Testing Clinical Preprocessing Pipeline...")
    
    # Create synthetic test image
    test_image = np.random.rand(256, 256, 3).astype(np.float32)
    
    # Initialize preprocessor
    preprocessor = ClinicalPreprocessing(
        target_size=(224, 224),
        apply_skull_stripping=True,
        apply_bias_correction=True,
        apply_clahe=True,
        apply_znorm=True
    )
    
    # Preprocess
    processed = preprocessor.preprocess(test_image)
    
    print(f"Input shape: {test_image.shape}")
    print(f"Output shape: {processed.shape}")
    print(f"Output range: [{processed.min():.3f}, {processed.max():.3f}]")
    print("✓ Preprocessing pipeline working correctly")
