"""
Phase 6: Disentangled Training Runner.
Trains the D-VAE-KAN model using multi-objective disentanglement loss.
"""

import os
import sys
import argparse
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.disentangled_vae import DisentangledVAE
from src.losses.disentanglement import DisentangledLoss
from src.phoenix_optimizer import create_adan_optimizer
import config

class Phase6Trainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self.model(x, training=True)

            # Prepare y_true dict for loss function
            y_true = {'image': x, 'label': y}

            # Calculate multi-objective loss
            total_loss = self.loss_fn(y_true, outputs)

        # Apply gradients
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return total_loss

def run_training(args):
    print("="*80)
    print("PHOENIX PROTOCOL PHASE 6: DISENTANGLED TRAINING")
    print("="*80)

    # 1. Initialize Model
    print("\n1. Initializing D-VAE-KAN Model...")
    model = DisentangledVAE(
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3),
        num_classes=config.NUM_CLASSES
    )

    # 2. Configure Loss and Optimizer
    print("2. Configuring Disentangled Loss (Beta-TCVAE)...")
    loss_fn = DisentangledLoss(beta=args.beta, gamma=args.gamma)
    optimizer = create_adan_optimizer(learning_rate=args.lr)

    trainer = Phase6Trainer(model, loss_fn, optimizer)

    # 3. Data Preparation (Simplified for Phase 6 Prototype)
    # In a real run, this would load the BraTS or clinical dataset
    print("3. Preparing Data (Synthetic for Prototype)...")
    def synthetic_gen():
        while True:
            img = tf.random.normal((config.IMG_HEIGHT, config.IMG_WIDTH, 3))
            label = tf.one_hot(tf.random.uniform((), maxval=2, dtype=tf.int32), 2)
            yield img, label

    train_ds = tf.data.Dataset.from_generator(
        synthetic_gen,
        output_signature=(
            tf.TensorSpec(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(2,), dtype=tf.float32)
        )
    ).batch(args.batch_size).take(args.steps_per_epoch)

    # 4. Training Loop
    print(f"\n4. Starting Training ({args.epochs} epochs)...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        print(f"Epoch {epoch+1}/{args.epochs}")

        for step, (x_batch, y_batch) in enumerate(train_ds):
            loss = trainer.train_step(x_batch, y_batch)
            epoch_loss += loss
            if step % 10 == 0:
                print(f"  Step {step}: Loss = {loss.numpy():.4f}")

        avg_loss = epoch_loss / args.steps_per_epoch
        print(f"End of Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

    # 5. Save Model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"./results/d_vae_kan_phase6_{timestamp}.h5"
    os.makedirs("./results", exist_ok=True)
    # Note: Keras Model.save might need special handling for custom layers,
    # but weights save will work.
    model.save_weights(save_path)
    print(f"\n✓ Phase 6 Model weights saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 6 Training Runner')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--steps-per-epoch', type=int, default=20, help='Steps per epoch')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--beta', type=float, default=4.0, help='KL weight')
    parser.add_argument('--gamma', type=float, default=10.0, help='TC weight')

    args = parser.parse_args()
    run_training(args)
