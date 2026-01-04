"""
Configuration file for Brain Tumor Detection Model
Contains all hyperparameters and settings
"""

import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
for directory in [DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Model parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.5

# Class names
CLASS_NAMES = ['no_tumor', 'tumor']
NUM_CLASSES = len(CLASS_NAMES)

# Data augmentation parameters
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
BRIGHTNESS_RANGE = [0.8, 1.2]

# Training parameters
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# Random seed for reproducibility
RANDOM_SEED = 42

# Model checkpoint
MODEL_NAME = 'brain_tumor_detection_model.h5'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Results
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
ROC_CURVE_PATH = os.path.join(RESULTS_DIR, 'roc_curve.png')
TRAINING_HISTORY_PATH = os.path.join(RESULTS_DIR, 'training_history.png')
CLASSIFICATION_REPORT_PATH = os.path.join(RESULTS_DIR, 'classification_report.txt')
