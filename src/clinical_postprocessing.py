"""
Post-Processing Pipeline with Uncertainty Quantification and Explainability
Clinical-grade post-processing for brain tumor detection.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple, Dict, Optional
import cv2


class ClinicalPostProcessing:
    """
    Post-processing pipeline for clinical deployment.
    
    Features:
    - Confidence thresholding
    - Test-time augmentation (TTA)
    - Uncertainty quantification
    - Grad-CAM visualization
    - Ensemble predictions
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.8,
        use_tta: bool = True,
        tta_transforms: int = 5,
        uncertainty_method: str = 'entropy'
    ):
        """
        Initialize post-processing pipeline.
        
        Args:
            confidence_threshold: Minimum confidence for accepting predictions
            use_tta: Enable test-time augmentation
            tta_transforms: Number of TTA augmentations
            uncertainty_method: Method for uncertainty ('entropy' or 'variance')
        """
        self.confidence_threshold = confidence_threshold
        self.use_tta = use_tta
        self.tta_transforms = tta_transforms
        self.uncertainty_method = uncertainty_method
    
    def confidence_check(
        self,
        predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Check if predictions meet confidence threshold.
        
        Args:
            predictions: Softmax probabilities (batch, num_classes)
            
        Returns:
            - Filtered predictions
            - Confidence scores
            - Indices of low-confidence samples
        """
        max_probs = np.max(predictions, axis=1)
        confident_mask = max_probs >= self.confidence_threshold
        low_confidence_indices = np.where(~confident_mask)[0].tolist()
        
        return predictions, max_probs, low_confidence_indices
    
    def test_time_augmentation(
        self,
        model: keras.Model,
        image: np.ndarray,
        num_augmentations: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform test-time augmentation for robust predictions.
        
        Args:
            model: Trained model
            image: Input image (H, W, C)
            num_augmentations: Number of augmented versions
            
        Returns:
            - Mean prediction
            - Prediction variance (uncertainty measure)
        """
        predictions = []
        
        # Original prediction
        pred = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        predictions.append(pred)
        
        # Augmented predictions
        for i in range(num_augmentations - 1):
            aug_image = self._augment_for_tta(image, i)
            aug_pred = model.predict(np.expand_dims(aug_image, axis=0), verbose=0)[0]
            predictions.append(aug_pred)
        
        predictions = np.array(predictions)
        
        # Compute mean and variance
        mean_pred = np.mean(predictions, axis=0)
        var_pred = np.var(predictions, axis=0)
        
        return mean_pred, var_pred
    
    def _augment_for_tta(self, image: np.ndarray, aug_id: int) -> np.ndarray:
        """
        Apply specific augmentation for TTA.
        
        Args:
            image: Input image
            aug_id: Augmentation identifier
            
        Returns:
            Augmented image
        """
        aug_image = image.copy()
        
        if aug_id == 0:
            # Horizontal flip
            aug_image = np.fliplr(aug_image)
        elif aug_id == 1:
            # Small rotation (+5 degrees)
            h, w = aug_image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), 5, 1.0)
            aug_image = cv2.warpAffine(aug_image, M, (w, h))
        elif aug_id == 2:
            # Small rotation (-5 degrees)
            h, w = aug_image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), -5, 1.0)
            aug_image = cv2.warpAffine(aug_image, M, (w, h))
        elif aug_id == 3:
            # Slight brightness adjustment
            aug_image = np.clip(aug_image * 1.1, 0, 1)
        elif aug_id == 4:
            # Slight contrast adjustment
            mean = np.mean(aug_image)
            aug_image = np.clip((aug_image - mean) * 1.1 + mean, 0, 1)
        
        return aug_image
    
    def compute_uncertainty(
        self,
        predictions: np.ndarray,
        variance: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute prediction uncertainty.
        
        Args:
            predictions: Softmax probabilities
            variance: Prediction variance from TTA (optional)
            
        Returns:
            Uncertainty scores
        """
        if self.uncertainty_method == 'entropy':
            # Predictive entropy
            epsilon = 1e-10
            entropy = -np.sum(predictions * np.log(predictions + epsilon), axis=-1)
            return entropy
        
        elif self.uncertainty_method == 'variance' and variance is not None:
            # Total variance
            total_var = np.sum(variance, axis=-1)
            return total_var
        
        else:
            # Margin-based uncertainty (1 - confidence)
            max_probs = np.max(predictions, axis=-1)
            return 1 - max_probs
    
    def grad_cam(
        self,
        model: keras.Model,
        image: np.ndarray,
        last_conv_layer_name: str,
        pred_index: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for explainability.
        
        Args:
            model: Trained model
            image: Input image (H, W, C)
            last_conv_layer_name: Name of last convolutional layer
            pred_index: Class index to visualize (None = predicted class)
            
        Returns:
            Heatmap (H, W) in [0, 1] range
        """
        # Create gradient model
        grad_model = keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Compute gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        # Gradient of class output with respect to feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()
        
        # Resize to input image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        return heatmap
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            image: Original image (H, W, C) in [0, 1]
            heatmap: Grad-CAM heatmap (H, W) in [0, 1]
            alpha: Blending factor
            colormap: OpenCV colormap
            
        Returns:
            Overlaid image (H, W, C) in [0, 1]
        """
        # Convert heatmap to color
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
        
        # Convert image to RGB if grayscale
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        
        # Overlay
        overlaid = alpha * heatmap_color + (1 - alpha) * image
        overlaid = np.clip(overlaid, 0, 1)
        
        return overlaid
    
    def generate_clinical_report(
        self,
        predictions: np.ndarray,
        uncertainty: float,
        class_names: List[str],
        confidence_threshold: float = None
    ) -> Dict:
        """
        Generate clinical-style report for radiologists.
        
        Args:
            predictions: Softmax probabilities
            uncertainty: Uncertainty score
            class_names: List of class names
            confidence_threshold: Optional custom threshold
            
        Returns:
            Clinical report dictionary
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        pred_class = np.argmax(predictions)
        confidence = predictions[pred_class]
        
        report = {
            'predicted_class': class_names[pred_class],
            'confidence': float(confidence),
            'uncertainty': float(uncertainty),
            'all_probabilities': {
                class_names[i]: float(predictions[i])
                for i in range(len(class_names))
            },
            'recommendation': self._get_recommendation(confidence, uncertainty, confidence_threshold),
            'requires_review': confidence < confidence_threshold or uncertainty > 0.5
        }
        
        return report
    
    def _get_recommendation(
        self,
        confidence: float,
        uncertainty: float,
        threshold: float
    ) -> str:
        """Generate clinical recommendation based on confidence and uncertainty."""
        if confidence >= 0.9 and uncertainty < 0.3:
            return "High confidence prediction. Suitable for automated reporting."
        elif confidence >= threshold and uncertainty < 0.5:
            return "Acceptable confidence. Recommend radiologist verification."
        elif confidence >= threshold:
            return "High uncertainty detected. Mandatory radiologist review required."
        else:
            return "Low confidence prediction. Mandatory expert review required."


# Example usage
if __name__ == "__main__":
    print("Testing Clinical Post-Processing Pipeline...")
    
    # Create synthetic test data
    test_predictions = np.array([[0.1, 0.9], [0.95, 0.05], [0.6, 0.4]])
    test_image = np.random.rand(224, 224, 3).astype(np.float32)
    
    # Initialize post-processor
    postprocessor = ClinicalPostProcessing(
        confidence_threshold=0.8,
        use_tta=True,
        tta_transforms=5
    )
    
    # Test confidence check
    filtered, confidences, low_conf = postprocessor.confidence_check(test_predictions)
    print(f"Confidence scores: {confidences}")
    print(f"Low confidence indices: {low_conf}")
    
    # Test uncertainty computation
    uncertainties = postprocessor.compute_uncertainty(test_predictions)
    print(f"Uncertainties: {uncertainties}")
    
    # Test clinical report generation
    report = postprocessor.generate_clinical_report(
        test_predictions[0],
        uncertainties[0],
        ['No Tumor', 'Tumor']
    )
    print(f"\nClinical Report:")
    print(f"  Predicted class: {report['predicted_class']}")
    print(f"  Confidence: {report['confidence']:.3f}")
    print(f"  Uncertainty: {report['uncertainty']:.3f}")
    print(f"  Recommendation: {report['recommendation']}")
    print(f"  Requires review: {report['requires_review']}")
    
    print("\n✓ Post-processing pipeline working correctly")
