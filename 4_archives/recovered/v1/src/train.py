"""
Training Script for Brain Tumor Detection Model
Trains the CNN model on brain MRI images
"""

import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.cnn_model import create_cnn_model, compile_model, create_callbacks, get_model_summary
from src.data_preprocessing import create_data_generators, load_dataset_from_directory, split_dataset, create_tf_dataset


def set_seeds(seed=config.RANDOM_SEED):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seeds set to {seed}")


def train_model_with_generators():
    """
    Train model using data generators
    Best for when data is organized in directories
    """
    print("\n" + "="*80)
    print("TRAINING BRAIN TUMOR DETECTION MODEL WITH DATA GENERATORS")
    print("="*80 + "\n")
    
    # Set random seeds
    set_seeds()
    
    # Create data generators
    print("Creating data generators...")
    try:
        train_gen, val_gen, test_gen = create_data_generators()
        
        print(f"\nTraining samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Test samples: {test_gen.samples}")
        print(f"Number of classes: {train_gen.num_classes}")
        print(f"Class indices: {train_gen.class_indices}")
        
    except Exception as e:
        print(f"Error creating data generators: {e}")
        print("\nPlease ensure your data is organized as:")
        print("  data/train/tumor/")
        print("  data/train/no_tumor/")
        print("  data/validation/tumor/")
        print("  data/validation/no_tumor/")
        print("  data/test/tumor/")
        print("  data/test/no_tumor/")
        return None
    
    # Create model
    print("\nCreating model...")
    model = create_cnn_model()
    model = compile_model(model)
    get_model_summary(model)
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Calculate steps per epoch
    steps_per_epoch = train_gen.samples // config.BATCH_SIZE
    validation_steps = val_gen.samples // config.BATCH_SIZE
    
    # Train model
    print("\nStarting training...")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}\n")
    
    start_time = datetime.now()
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=config.EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    print(f"\nTraining completed in {training_time}")
    print(f"Model saved to {config.MODEL_PATH}")
    
    return history, model


def train_model_with_arrays():
    """
    Train model using numpy arrays
    Best for when you want to load all data into memory
    """
    print("\n" + "="*80)
    print("TRAINING BRAIN TUMOR DETECTION MODEL WITH ARRAYS")
    print("="*80 + "\n")
    
    # Set random seeds
    set_seeds()
    
    # Load dataset
    print("Loading dataset from directory...")
    # Try to load from a unified 'all' directory if it exists (optional structure)
    data_dir = os.path.join(config.DATA_DIR, 'all')
    
    if not os.path.exists(data_dir):
        print(f"Note: Optional unified data directory {data_dir} not found.")
        print("This is expected if using train/val/test split structure.")
        print("Switching to generator-based training (recommended)...")
        return train_model_with_generators()
    
    images, labels = load_dataset_from_directory(data_dir)
    
    if len(images) == 0:
        print("No images found! Please check your data directory.")
        return None
    
    # Split dataset
    print("\nSplitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(images, labels)
    
    # Create TensorFlow datasets
    print("\nCreating TensorFlow datasets...")
    train_dataset = create_tf_dataset(X_train, y_train, augment=True)
    val_dataset = create_tf_dataset(X_val, y_val, shuffle=False, augment=False)
    test_dataset = create_tf_dataset(X_test, y_test, shuffle=False, augment=False)
    
    # Create model
    print("\nCreating model...")
    model = create_cnn_model()
    model = compile_model(model)
    get_model_summary(model)
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Train model
    print("\nStarting training...")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}\n")
    
    start_time = datetime.now()
    
    history = model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    print(f"\nTraining completed in {training_time}")
    print(f"Model saved to {config.MODEL_PATH}")
    
    return history, model


def save_training_history(history, filepath=None):
    """
    Save training history to file
    
    Args:
        history: Keras History object
        filepath: Path to save the history
    """
    if filepath is None:
        filepath = os.path.join(config.RESULTS_DIR, 'training_history.npy')
    
    np.save(filepath, history.history)
    print(f"\nTraining history saved to {filepath}")


def main():
    """Main training function"""
    
    # Check TensorFlow GPU availability
    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices('GPU'))
    print()
    
    # Train model
    # Try generator-based training first (most common setup)
    result = train_model_with_generators()
    
    if result is None:
        print("\nGenerator-based training failed. Trying array-based training...")
        result = train_model_with_arrays()
    
    if result is not None:
        history, model = result
        
        # Save training history
        save_training_history(history)
        
        # Print final metrics
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_val_loss:.4f}")
        
        print("\nNext steps:")
        print("1. Run evaluate.py to evaluate the model on test set")
        print("2. Run predict.py to make predictions on new images")
        print("3. Check results/ directory for training visualizations")
        
    else:
        print("\nTraining failed! Please check your data setup and try again.")


if __name__ == "__main__":
    main()
