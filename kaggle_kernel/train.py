import os
import sys
import subprocess
import glob
import argparse
import gc

# 1. Install Package
def install_package():
    """Install the custom library from the attached dataset."""
    print("Listing files in /kaggle/input/phoenix-mamba-v2-lib/:")
    try:
        print(os.listdir("/kaggle/input/phoenix-mamba-v2-lib/"))
    except Exception as e:
        print(f"Could not list directory: {e}")

    wheel_files = glob.glob("/kaggle/input/phoenix-mamba-v2-lib/*.whl")
    if wheel_files:
        # Sort to get the latest if multiple exist (though dataset should only have one)
        wheel_files.sort(reverse=True) 
        wheel_path = wheel_files[0]
        print(f"Found library wheel: {wheel_path}")
        print("Installing package...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_path, "--no-deps"])
            print("Successfully installed phoenix-mamba-v2")
        except subprocess.CalledProcessError as e:
            print(f"Error installing package: {e}")
            sys.exit(1)
    else:
        print("WARNING: No wheel file found in /kaggle/input/phoenix-mamba-v2-lib/")

install_package()

# 2. Imports & Config
import tensorflow as tf
from tensorflow import keras

# Enable Mixed Precision
try:
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")
except Exception as e:
    print(f"Could not enable mixed precision: {e}")

# Configure GPU Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPUs")
    except RuntimeError as e:
        print(e)

from phoenix_mamba_v2.models.phoenix import PhoenixMambaV2

def train(data_dir, output_dir, epochs=20, batch_size=4): 
    print(f"Starting training with PHOENIX-MAMBA v2 (Low Memory Mode)")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {batch_size}")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist.")
        return
    
    # REDUCED IMAGE SIZE for Memory Efficiency
    img_height, img_width = 128, 128
    print(f"Image Size: {img_height}x{img_width}")
    
    # 3. Data Loading
    print("Creating dataset from directory...")
    train_ds = keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'Training'),
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical',
        seed=42
    )
    
    val_ds = keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'Testing'),
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical',
        seed=42
    )
    
    # Normalization
    normalization_layer = keras.layers.Rescaling(1./255)
    
    def preprocess(x, y):
        return normalization_layer(x), y

    # Reduced prefetch buffer to save RAM
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(2)
    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(2)
    
    # 4. Create Model
    keras.backend.clear_session()
    gc.collect()
    
    model = PhoenixMambaV2(num_classes=4)
    
    # Build explicitly to lock shapes
    # IMPORTANT: We use (None, H, W, 3) because we adapted the code to handle dynamic shapes or fixed shapes
    model.build((None, img_height, img_width, 3))
    
    optimizer = keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=0.01)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 5. Train
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'phoenix_best.keras'),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.CSVLogger(os.path.join(output_dir, 'training_log.csv'))
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/kaggle/input/brain-tumor-mri-dataset")
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()
    
    if os.path.exists("/kaggle/working"):
        args.output_dir = "/kaggle/working"
        
    train(args.data_dir, args.output_dir)
