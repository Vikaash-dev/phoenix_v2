"""
Hyperparameter tuning for the NeuroSnake-ViT model using Optuna.
"""

import os
import sys
import argparse
import optuna
import tensorflow as tf
from tensorflow import keras

# Import from local modules
try:
    from src.legacy.models.legacy.neurosnake_vit_model import NeuroSnakeViTModel
    from src.legacy.train_neurosnake_vit import create_data_generators_with_physics_augmentation, create_callbacks
    import config
except ImportError:
    # Fallback: Add parent directory to path if imports fail
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.legacy.models.legacy.neurosnake_vit_model import NeuroSnakeViTModel
    from src.legacy.train_neurosnake_vit import create_data_generators_with_physics_augmentation, create_callbacks
    import config


def objective(trial):
    """
    Optuna objective function for hyperparameter tuning.
    """
    # Define hyperparameter search space
    num_transformer_blocks = trial.suggest_int("num_transformer_blocks", 2, 6)
    num_heads = trial.suggest_categorical("num_heads", [4, 8, 16])
    mlp_dim = trial.suggest_int("mlp_dim", 256, 1024)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Create model with suggested hyperparameters
    model = NeuroSnakeViTModel.create_model(
        num_transformer_blocks=num_transformer_blocks,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout_rate=dropout_rate,
    )

    # Create data generators
    train_ds, val_ds, _ = create_data_generators_with_physics_augmentation(
        train_dir=os.path.join(args.data_dir, 'train'),
        val_dir=os.path.join(args.data_dir, 'validation'),
        test_dir=os.path.join(args.data_dir, 'test'),
        batch_size=batch_size,
    )

    # Create optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy()

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )

    # Create callbacks
    model_save_path = os.path.join(args.output_dir, f"trial_{trial.number}.h5")
    callbacks = create_callbacks(model_save_path)

    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=0,
    )

    # Return validation accuracy
    return max(history.history["val_accuracy"])


def main():
    """Main tuning function."""
    parser = argparse.ArgumentParser(
        description="NeuroSnake-ViT Hyperparameter Tuning Script"
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
        default="./results/tuning",
        help="Output directory for tuning results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs per trial",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials",
    )

    global args
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    # Print best trial
    print("=" * 80)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("=" * 80)


if __name__ == "__main__":
    main()
