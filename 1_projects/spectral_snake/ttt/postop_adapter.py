"""
Post-Operative Adapter (Phase 6: TTT-Disentanglement).
Enables patient-specific adaptation to surgical cavities and artifacts.
"""

import tensorflow as tf
import numpy as np

class PostOpAdapter:
    """
    Handles Test-Time Training (TTT) for a single patient scan.
    Focuses adaptation on the 'cavity' and 'background' partitions.
    """

    def __init__(self, model, steps=5, lr=0.0001):
        self.model = model
        self.steps = steps
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.mse = tf.keras.losses.MeanSquaredError()

    def adapt(self, x_patient):
        """
        Performs TTT on a single patient scan.
        Only updates weights involved in latent partitioning and decoding.
        """
        # Ensure input has batch dimension
        if len(x_patient.shape) == 3:
            x_patient = tf.expand_dims(x_patient, 0)

        # Variables to optimize: We focus on the latent projection and decoder
        # This allows the model to "explain" the new scan's structure
        trainable_vars = (
            self.model.latent_proj.trainable_variables +
            self.model.decoder_net.trainable_variables
        )

        print(f"Adapting to patient scan over {self.steps} steps...")

        for i in range(self.steps):
            with tf.GradientTape() as tape:
                outputs = self.model(x_patient, training=True)
                recon = outputs['reconstruction']

                # We only minimize reconstruction loss during TTT
                # (Self-supervised, no labels needed)
                loss = self.mse(x_patient, recon)

            grads = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(grads, trainable_vars))

            if (i + 1) % 1 == 0:
                print(f"  Step {i+1}: Adaptation Loss = {loss.numpy():.6f}")

        # Final prediction after adaptation
        final_outputs = self.model(x_patient, training=False)
        return final_outputs

if __name__ == "__main__":
    # Test adaptation logic
    from src.models.disentangled_vae import DisentangledVAE

    model = DisentangledVAE()
    adapter = PostOpAdapter(model, steps=3)

    dummy_scan = tf.random.normal((1, 224, 224, 3))
    results = adapter.adapt(dummy_scan)

    print("Adaptation complete.")
    print("Diagnosis Confidence:", results['classification'].numpy())
