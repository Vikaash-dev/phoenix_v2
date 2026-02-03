import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from phoenix_mamba_v2.data.processors import ClinicalPreprocessor, FastPreprocessor

class TestProcessorsSecurity:

    @pytest.fixture
    def clinical_preprocessor(self):
        return ClinicalPreprocessor(target_size=(224, 224))

    @pytest.fixture
    def fast_preprocessor(self):
        return FastPreprocessor(target_size=(224, 224))

    def test_clinical_preprocessor_none_input(self, clinical_preprocessor):
        """Test rejection of None input."""
        with pytest.raises(ValueError, match="Input image cannot be None"):
            clinical_preprocessor.process(None)

    def test_clinical_preprocessor_wrong_type(self, clinical_preprocessor):
        """Test rejection of non-numpy input."""
        with pytest.raises(TypeError, match="Input must be a numpy array"):
            clinical_preprocessor.process([1, 2, 3])

    def test_clinical_preprocessor_empty_array(self, clinical_preprocessor):
        """Test rejection of empty array."""
        with pytest.raises(ValueError, match="Input image is empty"):
            clinical_preprocessor.process(np.array([]))

    def test_clinical_preprocessor_wrong_dimensions(self, clinical_preprocessor):
        """Test rejection of wrong dimensions (1D or 4D)."""
        # 1D
        with pytest.raises(ValueError, match="must have 2 or 3 dimensions"):
            clinical_preprocessor.process(np.zeros((100,)))

        # 4D
        with pytest.raises(ValueError, match="must have 2 or 3 dimensions"):
            clinical_preprocessor.process(np.zeros((1, 224, 224, 3)))

    def test_clinical_preprocessor_oversized_image(self, clinical_preprocessor):
        """Test rejection of oversized images (DoS prevention)."""
        # Create a "large" image metadata without allocating full memory if possible,
        # but here we just check the logic with a shape that triggers it
        # Note: We need to pass an array with that shape.
        # Using a sparse array or just a large index logic check?
        # The validation checks shape directly.
        # We can simulate a large array with stride tricks or just create a 4097x1 array to trigger it cheaply
        large_img = np.zeros((4097, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="exceed maximum allowed size"):
            clinical_preprocessor.process(large_img)

    def test_clinical_preprocessor_nan_values(self, clinical_preprocessor):
        """Test rejection of NaN values."""
        img = np.zeros((224, 224), dtype=np.float32)
        img[0, 0] = np.nan
        with pytest.raises(ValueError, match="Input image contains NaN or Inf values"):
            clinical_preprocessor.process(img)

    def test_clinical_preprocessor_inf_values(self, clinical_preprocessor):
        """Test rejection of Inf values."""
        img = np.zeros((224, 224), dtype=np.float32)
        img[0, 0] = np.inf
        with pytest.raises(ValueError, match="Input image contains NaN or Inf values"):
            clinical_preprocessor.process(img)

    def test_fast_preprocessor_security(self, fast_preprocessor):
        """Verify FastPreprocessor has similar protections."""
        with pytest.raises(ValueError, match="Input image cannot be None"):
            fast_preprocessor.process(None)

        with pytest.raises(ValueError, match="exceed maximum allowed size"):
            fast_preprocessor.process(np.zeros((4097, 10)))

        img_nan = np.zeros((100, 100), dtype=np.float32)
        img_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="Input image contains NaN or Inf values"):
            fast_preprocessor.process(img_nan)

if __name__ == "__main__":
    pytest.main([__file__])
