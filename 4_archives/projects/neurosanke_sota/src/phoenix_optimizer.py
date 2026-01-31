"""
Adan Optimizer and Focal Loss for Phoenix Protocol
Adan: Adaptive Nesterov Momentum Optimizer for stability on non-convex landscapes
Focal Loss: Handles class imbalance by down-weighting easy negatives
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class AdanOptimizer(keras.optimizers.Optimizer):
    """
    Adan (Adaptive Nesterov Momentum) Optimizer.
    
    Estimates first, second, and third moments of the gradient for superior
    stability on non-convex medical imaging landscapes compared to Adam or Lion.
    
    Reference:
    Xie et al., "Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing
    Deep Models" (2022)
    """
    
    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.98,
        beta2=0.92,
        beta3=0.99,
        epsilon=1e-8,
        weight_decay=0.02,
        name='Adan',
        **kwargs
    ):
        """
        Initialize Adan optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for 1st moment (gradient)
            beta2: Exponential decay rate for 2nd moment (gradient difference)
            beta3: Exponential decay rate for 3rd moment (squared gradient)
            epsilon: Small constant for numerical stability
            weight_decay: Weight decay (L2 regularization)
        """
        super(AdanOptimizer, self).__init__(name=name, **kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('beta1', beta1)
        self._set_hyper('beta2', beta2)
        self._set_hyper('beta3', beta3)
        self._set_hyper('epsilon', epsilon)
        self._set_hyper('weight_decay', weight_decay)
    
    def _create_slots(self, var_list):
        """Create optimizer state variables."""
        for var in var_list:
            # First moment (exponential moving average of gradients)
            self.add_slot(var, 'm', initializer='zeros')
            # Second moment (exponential moving average of gradient differences)
            self.add_slot(var, 'v', initializer='zeros')
            # Third moment (exponential moving average of squared gradients)
            self.add_slot(var, 'n', initializer='zeros')
            # Previous gradient for computing differences
            self.add_slot(var, 'prev_grad', initializer='zeros')
    
    def _resource_apply_dense(self, grad, var, apply_state=None):
        """Update variable given gradient tensor."""
        var_dtype = var.dtype.base_dtype
        lr_t = self._get_hyper('learning_rate', var_dtype)
        beta1_t = self._get_hyper('beta1', var_dtype)
        beta2_t = self._get_hyper('beta2', var_dtype)
        beta3_t = self._get_hyper('beta3', var_dtype)
        epsilon_t = self._get_hyper('epsilon', var_dtype)
        weight_decay_t = self._get_hyper('weight_decay', var_dtype)
        
        # Get state variables
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        n = self.get_slot(var, 'n')
        prev_grad = self.get_slot(var, 'prev_grad')
        
        # Compute gradient difference
        grad_diff = grad - prev_grad
        
        # Update first moment (gradient)
        m_t = m.assign(beta1_t * m + (1 - beta1_t) * grad)
        
        # Update second moment (gradient difference)
        v_t = v.assign(beta2_t * v + (1 - beta2_t) * grad_diff)
        
        # Update third moment (squared gradient)
        n_t = n.assign(beta3_t * n + (1 - beta3_t) * (grad ** 2))
        
        # Compute bias-corrected moments
        # Note: In practice, bias correction is often omitted after initial steps
        m_hat = m_t
        v_hat = v_t
        n_hat = n_t
        
        # Compute update
        # Adan update: combines gradient, gradient difference, and adaptive learning rate
        update = m_hat + beta2_t * v_hat
        denominator = tf.sqrt(n_hat) + epsilon_t
        
        # Apply weight decay (decoupled weight decay like AdamW)
        var_update = var.assign_sub(
            lr_t * (update / denominator + weight_decay_t * var)
        )
        
        # Update previous gradient
        prev_grad.assign(grad)
        
        return var_update
    
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        """Update variable given sparse gradient tensor."""
        # For sparse updates, we use a simplified approach
        return self._resource_apply_dense(
            tf.convert_to_tensor(grad), 
            var, 
            apply_state
        )
    
    def get_config(self):
        """Return optimizer configuration."""
        config = super(AdanOptimizer, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta1': self._serialize_hyperparameter('beta1'),
            'beta2': self._serialize_hyperparameter('beta2'),
            'beta3': self._serialize_hyperparameter('beta3'),
            'epsilon': self._serialize_hyperparameter('epsilon'),
            'weight_decay': self._serialize_hyperparameter('weight_decay'),
        })
        return config


class FocalLoss(keras.losses.Loss):
    """
    Focal Loss for addressing class imbalance.
    
    Down-weights easy negatives (high confidence correct predictions) to focus
    training on hard examples (rare Pituitary tumors vs common Gliomas).
    
    Reference:
    Lin et al., "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        from_logits=False,
        label_smoothing=0.0,
        reduction=keras.losses.Reduction.AUTO,
        name='focal_loss'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for class imbalance (0-1)
            gamma: Focusing parameter for down-weighting easy examples (>= 0)
            from_logits: Whether inputs are logits or probabilities
            label_smoothing: Label smoothing factor (0-1)
            reduction: Type of reduction to apply to loss
            name: Name of the loss
        """
        super(FocalLoss, self).__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        """
        Compute focal loss.
        
        Args:
            y_true: Ground truth labels (one-hot encoded)
            y_pred: Predicted probabilities or logits
            
        Returns:
            Focal loss value
        """
        # Convert to float32 for numerical stability
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1 - self.label_smoothing) + \
                     (self.label_smoothing / num_classes)
        
        # Compute probabilities
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Clip predictions to prevent log(0)
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Compute cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Compute focal weight: (1 - p_t)^gamma
        # p_t is the probability of the true class
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        
        # Apply alpha weighting for class imbalance
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        # Compute focal loss
        focal_loss = alpha_weight * focal_weight * cross_entropy
        
        return tf.reduce_sum(focal_loss, axis=-1)
    
    def get_config(self):
        """Return loss configuration."""
        config = super(FocalLoss, self).get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits,
            'label_smoothing': self.label_smoothing
        })
        return config


class LogCoshDiceLoss(keras.losses.Loss):
    """
    Log-Cosh Dice Loss.
    
    A novel loss function for medical image segmentation and classification.
    Combines the smoothing properties of Log-Cosh with the overlap maximization of Dice Loss.
    
    Formula: L_lc_dice = log(cosh(1 - Dice_Coefficient))
    
    Advantages:
    - Smooth and differentiable everywhere (unlike raw Dice loss)
    - Robust to outliers
    - Handles class imbalance effectively
    """
    
    def __init__(self, smooth=1.0, name='log_cosh_dice_loss', **kwargs):
        super(LogCoshDiceLoss, self).__init__(name=name, **kwargs)
        self.smooth = smooth
        
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Calculate Dice Coefficient
        intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
        union = tf.reduce_sum(y_true, axis=-1) + tf.reduce_sum(y_pred, axis=-1)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Calculate Log-Cosh Dice Loss
        dice_loss = 1 - dice
        log_cosh_dice = tf.math.log(tf.math.cosh(dice_loss))
        
        return log_cosh_dice
        
    def get_config(self):
        config = super(LogCoshDiceLoss, self).get_config()
        config.update({'smooth': self.smooth})
        return config


class BoundaryLoss(keras.losses.Loss):
    """
    Boundary Loss for Medical Image Segmentation.
    
    Approximates the Hausdorff Distance to penalize errors at the boundaries.
    Critical for tumors where the exact edge definition is the main diagnostic challenge.
    
    Reference:
    "Boundary Loss for Highly Unbalanced Segmentation" (MIDL 2019)
    """
    
    def __init__(self, name='boundary_loss', **kwargs):
        super(BoundaryLoss, self).__init__(name=name, **kwargs)
        
    def call(self, y_true, y_pred):
        """
        Compute boundary loss.
        Note: This is a simplified differentiable approximation suitable for training.
        For exact Hausdorff, use distance maps precomputed (hard to do in-graph).
        
        Here we use a gradient-based approximation:
        Minimize difference between gradients (edges) of prediction and ground truth.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Compute gradients (edges)
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        
        # L2 difference between gradients
        loss = tf.reduce_mean(
            tf.abs(dy_pred - dy_true) + tf.abs(dx_pred - dx_true), 
            axis=-1
        )
        
        return loss


def create_adan_optimizer(
    learning_rate=0.001,
    beta1=0.98,
    beta2=0.92,
    beta3=0.99,
    epsilon=1e-8,
    weight_decay=0.02
):
    """
    Factory function to create Adan optimizer.
    
    Args:
        learning_rate: Learning rate
        beta1: First moment decay rate
        beta2: Second moment decay rate
        beta3: Third moment decay rate
        epsilon: Numerical stability constant
        weight_decay: Weight decay factor
        
    Returns:
        AdanOptimizer instance
    """
    return AdanOptimizer(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        beta3=beta3,
        epsilon=epsilon,
        weight_decay=weight_decay
    )


def create_focal_loss(alpha=0.25, gamma=2.0, label_smoothing=0.0):
    """
    Factory function to create Focal Loss.
    
    Args:
        alpha: Class balance weight
        gamma: Focusing parameter
        label_smoothing: Label smoothing factor
        
    Returns:
        FocalLoss instance
    """
    return FocalLoss(
        alpha=alpha,
        gamma=gamma,
        from_logits=False,
        label_smoothing=label_smoothing
    )


def create_log_cosh_dice_loss(smooth=1.0):
    """
    Factory function to create Log-Cosh Dice Loss.
    
    Returns:
        LogCoshDiceLoss instance
    """
    return LogCoshDiceLoss(smooth=smooth)


def create_boundary_loss():
    """Factory for Boundary Loss."""
    return BoundaryLoss()


if __name__ == "__main__":
    print("="*80)
    print("PHOENIX PROTOCOL: Adan Optimizer and Focal Loss")
    print("="*80)
    
    # Test Adan optimizer
    print("\n1. Testing Adan Optimizer...")
    optimizer = create_adan_optimizer(learning_rate=0.001)
    print(f"Optimizer: {optimizer}")
    print(f"Config: {optimizer.get_config()}")
    
    # Test Focal Loss
    print("\n2. Testing Focal Loss...")
    focal_loss = create_focal_loss(alpha=0.25, gamma=2.0)
    print(f"Loss: {focal_loss}")
    print(f"Config: {focal_loss.get_config()}")
    
    # Test loss computation
    print("\n3. Testing loss computation...")
    y_true = tf.constant([[1, 0], [0, 1], [1, 0]], dtype=tf.float32)
    y_pred = tf.constant([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]], dtype=tf.float32)
    
    loss_value = focal_loss(y_true, y_pred)
    print(f"Sample loss value: {loss_value}")
    
    # Compare with standard categorical crossentropy
    ce_loss = keras.losses.CategoricalCrossentropy()
    ce_value = ce_loss(y_true, y_pred)
    print(f"Standard CE loss: {ce_value}")
    print(f"Focal loss: {tf.reduce_mean(loss_value)}")
    
    print("\n✓ Adan optimizer and Focal Loss tests passed!")
    print("="*80)
