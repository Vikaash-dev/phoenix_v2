"""
Configuration file for Legacy NeuroSnake models.
Recovered from v1 archives.
"""

# Image Dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (224, 224)

# Model Parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.5
NUM_CLASSES = 2  # ['no_tumor', 'tumor']

# Data Augmentation
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
