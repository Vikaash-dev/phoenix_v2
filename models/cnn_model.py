"""
CNN Model Architecture for Brain Tumor Detection
Implements a deep convolutional neural network with multiple blocks
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, 
    Dropout, BatchNormalization, Input
)
import config


def create_cnn_model(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
                     num_classes=config.NUM_CLASSES):
    """
    Create a CNN model for brain tumor detection
    
    Args:
        input_shape: Tuple of (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    
    model = Sequential([
        # Input layer
        Input(shape=input_shape),
        
        # Convolutional Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'),
        Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
        MaxPooling2D((2, 2), name='pool1'),
        BatchNormalization(name='bn1'),
        Dropout(0.25, name='dropout1'),
        
        # Convolutional Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
        MaxPooling2D((2, 2), name='pool2'),
        BatchNormalization(name='bn2'),
        Dropout(0.25, name='dropout2'),
        
        # Convolutional Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
        MaxPooling2D((2, 2), name='pool3'),
        BatchNormalization(name='bn3'),
        Dropout(0.25, name='dropout3'),
        
        # Convolutional Block 4
        Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1'),
        Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2'),
        MaxPooling2D((2, 2), name='pool4'),
        BatchNormalization(name='bn4'),
        Dropout(0.25, name='dropout4'),
        
        # Flatten and Dense layers
        Flatten(name='flatten'),
        Dense(512, activation='relu', name='fc1'),
        BatchNormalization(name='bn5'),
        Dropout(config.DROPOUT_RATE, name='dropout5'),
        
        Dense(256, activation='relu', name='fc2'),
        BatchNormalization(name='bn6'),
        Dropout(config.DROPOUT_RATE, name='dropout6'),
        
        # Output layer
        Dense(num_classes, activation='softmax', name='output')
    ], name='BrainTumorCNN')
    
    return model


def compile_model(model, learning_rate=config.LEARNING_RATE):
    """
    Compile the model with optimizer, loss, and metrics
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled model
    """
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def create_callbacks(model_path=config.MODEL_PATH):
    """
    Create training callbacks
    
    Args:
        model_path: Path to save the best model
        
    Returns:
        List of callbacks
    """
    
    # Model checkpoint - save best model
    checkpoint = keras.callbacks.ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.MIN_LR,
        verbose=1
    )
    
    # TensorBoard logging (optional)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        write_graph=True
    )
    
    return [checkpoint, early_stop, reduce_lr, tensorboard]


def get_model_summary(model):
    """
    Print model summary
    
    Args:
        model: Keras model
    """
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*80)
    model.summary()
    print("="*80)
    
    # Count parameters
    total_params = model.count_params()
    
    # For more detailed breakdown, we can use trainable_weights
    # Note: This is kept for completeness but model.count_params() is more efficient
    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_count = total_params - trainable_count
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_count:,}")
    print(f"Non-trainable Parameters: {non_trainable_count:,}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Test model creation
    print("Creating Brain Tumor Detection CNN Model...")
    
    model = create_cnn_model()
    model = compile_model(model)
    
    get_model_summary(model)
    
    print("Model created successfully!")
    print(f"Input shape: {config.IMG_HEIGHT}x{config.IMG_WIDTH}x{config.IMG_CHANNELS}")
    print(f"Number of classes: {config.NUM_CLASSES}")
    print(f"Class names: {config.CLASS_NAMES}")
