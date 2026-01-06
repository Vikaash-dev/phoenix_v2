"""
Phoenix Protocol Training Script
Trains NeuroSnake model with Adan optimizer, Focal Loss, and physics-informed augmentation.
"""

import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.neurosnake_model import create_neurosnake_model, create_baseline_model
from src.phoenix_optimizer import create_adan_optimizer, create_focal_loss
from src.physics_informed_augmentation import PhysicsInformedAugmentation
import config


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Random seeds set to {seed}")


def create_data_generators_with_physics_augmentation(
    train_dir,
    val_dir,
    test_dir,
    batch_size=32,
    img_size=(224, 224),
    use_physics_augmentation=True
):
    """
    Create data generators with optional physics-informed augmentation.
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        test_dir: Test data directory
        batch_size: Batch size
        img_size: Target image size
        use_physics_augmentation: Whether to use physics-informed augmentation
        
    Returns:
        Tuple of (train_gen, val_gen, test_gen)
    """
    # Base preprocessing
    def preprocess_image(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    # Physics-informed augmentation function
    if use_physics_augmentation:
        augmentor = PhysicsInformedAugmentation(
            elastic_alpha_range=(30, 40),
            elastic_sigma=5.0,
            rician_noise_sigma_range=(0.01, 0.05),
            apply_probability=0.5
        )
        
        def augment_fn(image, label):
            # Apply physics-informed augmentation
            image_np = image.numpy()
            augmented = augmentor.augment(image_np)
            return tf.convert_to_tensor(augmented, dtype=tf.float32), label
        
        def apply_physics_augmentation(image, label):
            image, label = tf.py_function(
                func=augment_fn,
                inp=[image, label],
                Tout=[tf.float32, tf.int32]
            )
            image.set_shape([img_size[0], img_size[1], 3])
            return image, label
    
    # Training dataset with augmentation
    train_ds = keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    )
    train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if use_physics_augmentation:
        train_ds = train_ds.unbatch()
        train_ds = train_ds.map(apply_physics_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(batch_size)
    
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    # Validation dataset (no augmentation)
    val_ds = keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    # Test dataset (no augmentation)
    test_ds = keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )
    test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds


def create_callbacks(model_save_path, log_dir='./logs'):
    """
    Create training callbacks.
    
    Args:
        model_save_path: Path to save best model
        log_dir: Directory for TensorBoard logs
        
    Returns:
        List of callbacks
    """
    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        
        # CSV logger
        keras.callbacks.CSVLogger(
            os.path.join(log_dir, 'training_log.csv'),
            append=True
        )
    ]
    
    return callbacks


def train_phoenix_protocol(
    data_dir,
    model_type='neurosnake',
    use_physics_augmentation=True,
    batch_size=32,
    epochs=100,
    learning_rate=0.001,
    output_dir='./results',
    resume_from=None
):
    """
    Train model using Phoenix Protocol.
    
    Args:
        data_dir: Base data directory
        model_type: 'neurosnake' or 'baseline'
        use_physics_augmentation: Whether to use physics-informed augmentation
        batch_size: Batch size
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        output_dir: Output directory for results
        resume_from: Path to model to resume training from
        
    Returns:
        Training history
    """
    print("="*80)
    print("PHOENIX PROTOCOL: TRAINING PIPELINE")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seeds
    set_random_seeds(config.RANDOM_SEED)
    
    # GPU configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✓ Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\n⚠ No GPU found, using CPU")
    
    # Data paths
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')
    test_dir = os.path.join(data_dir, 'test')
    
    # Create data generators
    print("\n1. Creating data generators...")
    print(f"   - Physics-informed augmentation: {use_physics_augmentation}")
    print(f"   - Batch size: {batch_size}")
    
    train_ds, val_ds, test_ds = create_data_generators_with_physics_augmentation(
        train_dir, val_dir, test_dir,
        batch_size=batch_size,
        img_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        use_physics_augmentation=use_physics_augmentation
    )
    
    # Create or load model
    print("\n2. Creating model...")
    print(f"   - Model type: {model_type}")
    
    if resume_from and os.path.exists(resume_from):
        print(f"   - Resuming from: {resume_from}")
        model = keras.models.load_model(resume_from)
    else:
        if model_type == 'neurosnake':
            model = create_neurosnake_model(
                input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
                num_classes=config.NUM_CLASSES,
                use_mobilevit=True,
                dropout_rate=0.3
            )
        elif model_type == 'baseline':
            model = create_baseline_model(
                input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
                num_classes=config.NUM_CLASSES
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"   - Total parameters: {model.count_params():,}")
    
    # Create optimizer and loss
    print("\n3. Configuring training...")
    print(f"   - Optimizer: Adan")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Loss: Focal Loss")
    
    optimizer = create_adan_optimizer(
        learning_rate=learning_rate,
        beta1=0.98,
        beta2=0.92,
        beta3=0.99,
        weight_decay=0.02
    )
    
    loss = create_focal_loss(
        alpha=0.25,
        gamma=2.0,
        label_smoothing=0.1
    )
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    # Create callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(output_dir, f'{model_type}_best_{timestamp}.h5')
    log_dir = os.path.join(output_dir, 'logs', timestamp)
    
    callbacks = create_callbacks(model_save_path, log_dir)
    
    # Train model
    print("\n4. Training model...")
    print(f"   - Epochs: {epochs}")
    print(f"   - Model will be saved to: {model_save_path}")
    print("="*80 + "\n")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("5. Evaluating on test set...")
    
    test_results = model.evaluate(test_ds, verbose=1)
    
    print("\nTest Results:")
    for metric_name, value in zip(model.metrics_names, test_results):
        print(f"   - {metric_name}: {value:.4f}")
    
    # Save training history
    history_path = os.path.join(output_dir, f'{model_type}_history_{timestamp}.json')
    with open(history_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        history_dict = {
            key: [float(val) for val in values]
            for key, values in history.history.items()
        }
        json.dump(history_dict, f, indent=2)
    
    print(f"\n✓ Training history saved to: {history_path}")
    
    # Save test results
    results_path = os.path.join(output_dir, f'{model_type}_test_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        test_results_dict = {
            metric_name: float(value)
            for metric_name, value in zip(model.metrics_names, test_results)
        }
        json.dump(test_results_dict, f, indent=2)
    
    print(f"✓ Test results saved to: {results_path}")
    
    print("\n" + "="*80)
    print("PHOENIX PROTOCOL: TRAINING COMPLETE")
    print("="*80)
    
    return history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Phoenix Protocol Training Script'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Base data directory containing train/validation/test splits'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='neurosnake',
        choices=['neurosnake', 'baseline'],
        help='Model architecture to train'
    )
    parser.add_argument(
        '--no-physics-augmentation',
        action='store_true',
        help='Disable physics-informed augmentation'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Path to model checkpoint to resume training from'
    )
    
    args = parser.parse_args()
    
    # Train model
    history = train_phoenix_protocol(
        data_dir=args.data_dir,
        model_type=args.model_type,
        use_physics_augmentation=not args.no_physics_augmentation,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        resume_from=args.resume_from
    )


if __name__ == "__main__":
    main()
