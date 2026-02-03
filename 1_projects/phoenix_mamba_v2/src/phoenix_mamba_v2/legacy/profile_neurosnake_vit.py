"""
Performance profiling for the NeuroSnake-ViT model using the TensorFlow Profiler.
"""

import os
import sys
import argparse
import tensorflow as tf
from tensorflow import keras

# Import from local modules
try:
    from phoenix_mamba_v2.legacy.models.legacy.neurosnake_vit_model import NeuroSnakeViTModel
    from phoenix_mamba_v2.legacy.train_neurosnake_vit import create_data_generators_with_physics_augmentation
    import config
except ImportError:
    # Fallback: Add parent directory to path if imports fail
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from phoenix_mamba_v2.legacy.models.legacy.neurosnake_vit_model import NeuroSnakeViTModel
    from phoenix_mamba_v2.legacy.train_neurosnake_vit import create_data_generators_with_physics_augmentation
    import config


def profile_model(
    data_dir,
    output_dir,
    batch_size=32,
    num_steps=100,
):
    """
    Profile the NeuroSnake-ViT model.
    """
    print("=" * 80)
    print("NeuroSnake-ViT PERFORMANCE PROFILING")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create model
    model = NeuroSnakeViTModel.create_model()

    # Create data generators
    train_ds, _, _ = create_data_generators_with_physics_augmentation(
        train_dir=os.path.join(data_dir, 'train'),
        val_dir=os.path.join(data_dir, 'validation'),
        test_dir=os.path.join(data_dir, 'test'),
        batch_size=batch_size,
    )

    # Create optimizer and loss
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.CategoricalCrossentropy()

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )

    # Start profiling
    tf.profiler.experimental.start(output_dir)

    # Train for a few steps
    model.fit(
        train_ds,
        epochs=1,
        steps_per_epoch=num_steps,
        verbose=1,
    )

    # Stop profiling
    tf.profiler.experimental.stop()

    print("=" * 80)
    print(f"Profiling data saved to: {output_dir}")
    print("To view the profiling results, run:")
    print(f"tensorboard --logdir={output_dir}")
    print("=" * 80)


def main():
    """Main profiling function."""
    parser = argparse.ArgumentParser(
        description="NeuroSnake-ViT Performance Profiling Script"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Base data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/profiling",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for profiling",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of steps to profile",
    )

    args = parser.parse_args()

    # Profile model
    profile_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
    )


if __name__ == "__main__":
    main()
