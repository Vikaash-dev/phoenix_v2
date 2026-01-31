"""
Disentanglement Loss Functions for Phase 6 (D-VAE-KAN).
Includes Beta-VAE and Total Correlation (TC) loss components.
"""

import tensorflow as tf
import numpy as np

def kl_divergence_loss(mu, log_var):
    """Standard KL Divergence loss."""
    return -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)

def total_correlation_loss(z, mu, log_var):
    """
    Approximation of Total Correlation for disentanglement.
    Encourages independence between latent dimensions.
    """
    # Simple approximation: Batch-wise variance of the latent dimensions
    # In a full TCVAE, this involves estimating the density q(z)
    # For this implementation, we use a contrastive penalty on latent correlations

    # Calculate correlation matrix of z (Batch, Latent)
    # We want this to be close to Identity
    z_centered = z - tf.reduce_mean(z, axis=0)
    cov = tf.matmul(z_centered, z_centered, transpose_a=True) / tf.cast(tf.shape(z)[0], z.dtype)

    # Penalty: Sum of squared off-diagonal elements
    off_diag = cov - tf.linalg.diag(tf.linalg.diag_part(cov))
    tc_penalty = tf.reduce_sum(tf.square(off_diag))

    return tc_penalty

class DisentangledLoss(tf.keras.losses.Loss):
    """
    Combined loss for D-VAE-KAN.
    L = Recon + Beta*KL + Gamma*TC + Lambda*Classification
    """

    def __init__(
        self,
        beta=4.0,      # KL weight (standard Beta-VAE)
        gamma=10.0,    # TC weight (for disentanglement)
        alpha_cls=1.0, # Classification weight
        name='disentangled_loss'
    ):
        super(DisentangledLoss, self).__init__(name=name)
        self.beta = beta
        self.gamma = gamma
        self.alpha_cls = alpha_cls
        self.mse = tf.keras.losses.MeanSquaredError()
        self.ce = tf.keras.losses.CategoricalCrossentropy()

    def call(self, y_true, y_pred_dict):
        """
        y_true: Original image or dictionary containing (image, label)
        y_pred_dict: Dictionary from DisentangledVAE.call()
        """
        # In a real training loop, we'd handle the dictionary structure carefully.
        # This is a functional implementation of the logic.

        recon = y_pred_dict['reconstruction']
        classification = y_pred_dict['classification']
        params = y_pred_dict['latent_params']
        samples = y_pred_dict['latent_samples']

        # 1. Reconstruction Loss
        recon_loss = self.mse(y_true['image'], recon)

        # 2. KL Divergence (summed across partitions)
        total_kl = 0.0
        for key in params:
            mu, log_var = params[key]
            total_kl += tf.reduce_mean(kl_divergence_loss(mu, log_var))

        # 3. Total Correlation (between clinical factors)
        # We concatenate tumor, edema, and cavity factors for TC penalty
        clinical_z = tf.concat([samples['tumor'], samples['edema'], samples['cavity']], axis=-1)
        tc_loss = total_correlation_loss(clinical_z, None, None)

        # 4. Classification Loss
        cls_loss = self.ce(y_true['label'], classification)

        # Total Weighted Loss
        total_loss = (
            recon_loss +
            (self.beta * total_kl) +
            (self.gamma * tc_loss) +
            (self.alpha_cls * cls_loss)
        )

        return total_loss

if __name__ == "__main__":
    # Test TC loss
    z = tf.random.normal((32, 64))
    tc = total_correlation_loss(z, None, None)
    print("TC Loss (Random):", tc.numpy())

    # Highly correlated z should have higher TC
    z_corr = tf.repeat(tf.random.normal((32, 1)), 64, axis=1)
    tc_corr = total_correlation_loss(z_corr, None, None)
    print("TC Loss (Correlated):", tc_corr.numpy())
