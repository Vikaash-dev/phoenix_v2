"""
Legacy25DLoader: Adapter for Local 2D Data Training

This loader adapts standard 2D image folders (JPEG/PNG) to the 3D input format
required by PHOENIX v3.1. It simulates 2.5D context by duplicating the same slice.

Input: Single 2D Image (H, W, C)
Output: Fake 2.5D Stack (3, H, W, C) -> [Image, Image, Image]

Use Case: Local testing/debugging when 3D NIfTI volumes are unavailable.
"""

import tensorflow as tf
import os

class Legacy25DLoader:
    def __init__(self, data_dir, batch_size=32, img_size=224, num_classes=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes

    def get_dataset(self, subset="training"):
        """
        Creates a tf.data.Dataset from a directory of 2D images.
        """
        if subset.lower() == "training":
            shuffle = True
            directory = os.path.join(self.data_dir, "Training")
        else:
            shuffle = False
            directory = os.path.join(self.data_dir, "Testing")

        print(f"📂 LegacyLoader: Loading {subset} data from {directory}")

        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels='inferred',
            label_mode='categorical',
            class_names=['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'],
            batch_size=self.batch_size,
            image_size=(self.img_size, self.img_size),
            shuffle=shuffle,
            seed=42,
        )

        # Optimization: Pre-fetch and map
        ds = ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    def _process_path(self, image, label):
        """
        Adapts 2D image to 3D model input.
        Input: (B, H, W, C)
        Output: ((B, 3, H, W, C), Label)
        """
        # 1. Normalize [0, 255] -> [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        # 2. Fake 2.5D Stacking: (B, H, W, C) -> (B, 3, H, W, C)
        # Duplicate the slice 3 times to simulate [z-1, z, z+1]
        image_stack = tf.stack([image, image, image], axis=1)

        return image_stack, label

def create_legacy_loader(data_dir, batch_size=32, img_size=224):
    """Factory function for legacy loader."""
    loader = Legacy25DLoader(data_dir, batch_size, img_size)

    # Return tuple of (train, val) for convenience
    # Note: Using 'Testing' folder as validation for local dev
    train_ds = loader.get_dataset("training")
    val_ds = loader.get_dataset("validation")

    return train_ds, val_ds
