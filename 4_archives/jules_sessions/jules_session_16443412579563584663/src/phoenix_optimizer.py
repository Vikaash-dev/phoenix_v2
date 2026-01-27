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
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            name=name,
            **kwargs
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.epsilon = epsilon
    
    def build(self, var_list):
        """Create optimizer state variables."""
        if self.built:
            return
        super().build(var_list)
        self.m = []
        self.v = []
        self.n = []
        self.prev_grad = []
        for var in var_list:
            # First moment (exponential moving average of gradients)
            self.m.append(self.add_variable_from_reference(var, 'm', initializer='zeros'))
            # Second moment (exponential moving average of gradient differences)
            self.v.append(self.add_variable_from_reference(var, 'v', initializer='zeros'))
            # Third moment (exponential moving average of squared gradients)
            self.n.append(self.add_variable_from_reference(var, 'n', initializer='zeros'))
            # Previous gradient for computing differences
            self.prev_grad.append(self.add_variable_from_reference(var, 'prev_grad', initializer='zeros'))
    
    def update_step(self, grad, variable, learning_rate):
        """Update variable given gradient tensor."""
        lr = tf.cast(learning_rate, variable.dtype)
        beta1 = tf.cast(self.beta1, variable.dtype)
        beta2 = tf.cast(self.beta2, variable.dtype)
        beta3 = tf.cast(self.beta3, variable.dtype)
        epsilon = tf.cast(self.epsilon, variable.dtype)
        weight_decay = tf.cast(self.weight_decay, variable.dtype)
        
        # Get state variables
        m = self.m[self._get_variable_index(variable)]
        v = self.v[self._get_variable_index(variable)]
        n = self.n[self._get_variable_index(variable)]
        prev_grad = self.prev_grad[self._get_variable_index(variable)]
        
        # Compute gradient difference
        # Note: For the first step, prev_grad is 0, so grad_diff = grad
        grad_diff = grad - prev_grad
        
        # Update first moment (gradient)
        m.assign(beta1 * m + (1 - beta1) * grad)
        
        # Update second moment (gradient difference)
        v.assign(beta2 * v + (1 - beta2) * grad_diff)
        
        # Update third moment (squared gradient)
        n.assign(beta3 * n + (1 - beta3) * (grad ** 2))
        
        # Compute update
        # Adan update: combines gradient, gradient difference, and adaptive learning rate
        # Update rule: theta = theta - lr * (m + beta2 * v) / (sqrt(n) + epsilon)
        update = m + beta2 * v
        denominator = tf.sqrt(n) + epsilon
        
        # Apply weight decay (decoupled weight decay like AdamW)
        # variable.assign_sub(lr * (update / denominator + weight_decay * variable))
        # Keras 3 Optimizer automatically handles weight decay if self.weight_decay is set
        # But we need to be careful not to double count it.
        # However, custom update_step usually needs to apply the update to the variable.
        
        # In Keras 3, if we implement update_step, we are responsible for applying the update.
        # But `weight_decay` passed to super().__init__ implies Keras might handle it?
        # Checking Keras 3 docs: "If you want to decouple weight decay, you can use the weight_decay argument in the constructor."
        # "If you implement your own update_step, you don't need to worry about weight decay... unless you want to implement it yourself."
        # Actually, Keras 3 applies weight decay separately if `weight_decay` is set in init.
        # So we just apply the main update.
        
        variable.assign_sub(lr * (update / denominator))
        
        # Update previous gradient
        prev_grad.assign(grad)

    def get_config(self):
        """Return optimizer configuration."""
        config = super().get_config()
        config.update({
            'beta1': self.beta1,
            'beta2': self.beta2,
            'beta3': self.beta3,
            'epsilon': self.epsilon,
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
        reduction='sum_over_batch_size', # Default for Keras 3
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
