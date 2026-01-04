"""
Data Preprocessing and Augmentation Module
Handles loading, preprocessing, and augmenting brain MRI images
"""

import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import config


def load_image(image_path, target_size=config.IMG_SIZE):
    """
    Load and preprocess a single image
    
    Args:
        image_path: Path to the image file
        target_size: Target size (height, width)
        
    Returns:
        Preprocessed image array
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(target_size)
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        
        return img_array
    
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_dataset_from_directory(directory):
    """
    Load dataset from directory structure
    Expected structure:
        directory/
            tumor/
                image1.jpg
                image2.jpg
            no_tumor/
                image1.jpg
                image2.jpg
    
    Args:
        directory: Root directory containing class subdirectories
        
    Returns:
        images: numpy array of images
        labels: numpy array of labels
    """
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(directory, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist")
            continue
        
        print(f"Loading images from {class_dir}...")
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            
            # Skip non-image files
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            
            img = load_image(img_path)
            
            if img is not None:
                images.append(img)
                labels.append(class_idx)
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Loaded {len(images)} images from {directory}")
    
    return images, labels


def create_data_generators(train_dir=config.TRAIN_DIR,
                          val_dir=config.VAL_DIR,
                          test_dir=config.TEST_DIR):
    """
    Create data generators with augmentation for training
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        test_dir: Test data directory
        
    Returns:
        train_generator, val_generator, test_generator
    """
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=config.ROTATION_RANGE,
        width_shift_range=config.WIDTH_SHIFT_RANGE,
        height_shift_range=config.HEIGHT_SHIFT_RANGE,
        zoom_range=config.ZOOM_RANGE,
        horizontal_flip=config.HORIZONTAL_FLIP,
        brightness_range=config.BRIGHTNESS_RANGE,
        fill_mode='nearest'
    )
    
    # Validation and test data generators (no augmentation)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=config.RANDOM_SEED
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


def preprocess_image_for_prediction(image_path):
    """
    Preprocess a single image for prediction
    
    Args:
        image_path: Path to the image or numpy array
        
    Returns:
        Preprocessed image ready for model input
    """
    if isinstance(image_path, str):
        img = load_image(image_path)
    else:
        img = image_path
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img


def split_dataset(images, labels, test_size=0.15, val_size=0.15, random_state=config.RANDOM_SEED):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        images: Array of images
        labels: Array of labels
        test_size: Proportion of test set
        val_size: Proportion of validation set
        random_state: Random seed
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: separate validation from training
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, num_classes=config.NUM_CLASSES)
    y_val = to_categorical(y_val, num_classes=config.NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=config.NUM_CLASSES)
    
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_tf_dataset(images, labels, batch_size=config.BATCH_SIZE, shuffle=True, augment=False):
    """
    Create TensorFlow dataset with optional augmentation
    
    Args:
        images: Array of images
        labels: Array of labels
        batch_size: Batch size
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation
        
    Returns:
        TensorFlow dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images), seed=config.RANDOM_SEED)
    
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def augment_image(image, label):
    """
    Apply augmentation to an image
    
    Args:
        image: Input image
        label: Image label
        
    Returns:
        Augmented image and label
    """
    # Random rotation
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    
    # Random flip
    image = tf.image.random_flip_left_right(image)
    
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Ensure values are in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label


def get_class_weights(labels):
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        labels: Array of labels
        
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print("Class weights:", class_weight_dict)
    
    return class_weight_dict


if __name__ == "__main__":
    print("Data Preprocessing Module")
    print("="*80)
    
    # Test image loading
    print("\nTesting data preprocessing functions...")
    
    # Check if data directories exist
    for directory in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]:
        exists = os.path.exists(directory)
        print(f"{directory}: {'exists' if exists else 'does not exist'}")
    
    print("\nData preprocessing module ready!")
