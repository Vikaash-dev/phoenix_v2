"""
Adan Optimizer for Phoenix Protocol
Adan: Adaptive Nesterov Momentum Optimizer for stability on non-convex landscapes.
Updated for Keras 3.x compatibility.
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

class AdanOptimizer(keras.optimizers.Optimizer):
    """
    Adan (Adaptive Nesterov Momentum) Optimizer.
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
        super().__init__(name=name, learning_rate=learning_rate, weight_decay=weight_decay, **kwargs)
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
            self.m.append(self.add_variable_from_reference(var, "m", initializer="zeros"))
            self.v.append(self.add_variable_from_reference(var, "v", initializer="zeros"))
            self.n.append(self.add_variable_from_reference(var, "n", initializer="zeros"))
            self.prev_grad.append(self.add_variable_from_reference(var, "prev_grad", initializer="zeros"))
            
    def update_step(self, gradient, variable, learning_rate):
        """Update step for a single variable."""
        lr = tf.cast(learning_rate, variable.dtype)
        
        var_key = self._get_variable_index(variable)
        
        m = self.m[var_key]
        v = self.v[var_key]
        n = self.n[var_key]
        prev_grad = self.prev_grad[var_key]
        
        # Gradient difference
        grad_diff = gradient - prev_grad
        
        # Update moments
        m.assign(self.beta1 * m + (1 - self.beta1) * gradient)
        v.assign(self.beta2 * v + (1 - self.beta2) * grad_diff)
        n.assign(self.beta3 * n + (1 - self.beta3) * (gradient ** 2))
        
        # Adan update rule
        update = m + self.beta2 * v
        denominator = tf.sqrt(n) + self.epsilon
        
        # Apply update
        variable.assign_sub(lr * (update / denominator))
        
        # Update previous gradient
        prev_grad.assign(gradient)

# Re-export FocalLoss and factories
class FocalLoss(keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, label_smoothing=0.0, reduction="sum_over_batch_size", name='focal_loss'):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1 - self.label_smoothing) + (self.label_smoothing / num_classes)
        
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        focal_loss = alpha_weight * focal_weight * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits,
            'label_smoothing': self.label_smoothing
        })
        return config

def create_adan_optimizer(learning_rate=0.001, beta1=0.98, beta2=0.92, beta3=0.99, epsilon=1e-8, weight_decay=0.02):
    return AdanOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, beta3=beta3, epsilon=epsilon, weight_decay=weight_decay)

def create_focal_loss(alpha=0.25, gamma=2.0, label_smoothing=0.0):
    return FocalLoss(alpha=alpha, gamma=gamma, from_logits=False, label_smoothing=label_smoothing)
