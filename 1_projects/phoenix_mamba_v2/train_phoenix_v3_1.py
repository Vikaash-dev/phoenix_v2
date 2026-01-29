"""
PHOENIX v3.1 Adaptive Training Node

"Write Once, Train Anywhere"
- Automatically detects Kaggle vs Local environment
- Selects the correct Data Loader strategy (Real 3D vs Fake 2.5D)
- Builds and trains PHOENIX v3.1 (Topology Fixed)

Usage:
    # Local (uses Legacy Loader with fake stacking)
    python train_phoenix_v3_1.py --data-dir ./data

    # Kaggle (uses True 2.5D Loader with NIfTI volumes)
    python train_phoenix_v3_1.py --data-dir /kaggle/input/brats2023
"""

import os
import sys
import argparse
import tensorflow as tf
from pathlib import Path

# --- 1. Environment Detection & Import Handling ---
# Detect if we are running on Kaggle or Locally
IS_KAGGLE = os.path.exists('/kaggle/input') or os.path.exists('/kaggle/working')

# If local, add project root to path to allow absolute imports
if not IS_KAGGLE:
    PROJECT_ROOT = Path(__file__).resolve().parent
    sys.path.append(str(PROJECT_ROOT))

# Imports (Conditional based on bundling)
try:
    # Local Development Imports
    from src.models.model_v3_1 import create_phoenix_v3_1
    from src.data.loader_legacy import create_legacy_loader
    # Note: True25DLoader import is conditional to avoid dependency errors locally
except ImportError:
    # Fallback for Bundled Script (where classes are defined in the same file)
    # The bundler will paste classes above this execution block
    print("⚠ Using bundled classes or resolving imports failed. Assuming bundled mode.")
    pass

def get_data_loaders(args):
    """
    Factory function to return the correct data loader based on environment.
    """
    if IS_KAGGLE and not args.force_legacy:
        print("🚀 Environment: KAGGLE (Using True 2.5D NIfTI Loader)")

        # Check if True25DLoader is available (it might be bundled or imported)
        try:
            from src.data.loader_true_2_5d import True25DLoader
            print("✅ True25DLoader imported successfully")

            # Initialize True 2.5D Loader
            loader = True25DLoader(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                img_size=(args.img_size, args.img_size),
                num_classes=4
            )
            return loader.get_dataset("train"), loader.get_dataset("validation")

        except ImportError:
            print("⚠ True25DLoader not found/bundled. Fallback to Legacy.")
            return create_legacy_loader(args.data_dir, args.batch_size, args.img_size)
        except Exception as e:
            print(f"⚠ Error initializing True25DLoader: {e}")
            print("Fallback to Legacy Loader.")
            return create_legacy_loader(args.data_dir, args.batch_size, args.img_size)

    else:
        print("💻 Environment: LOCAL (Using Legacy 2.5D JPEG Loader)")
        # This duplicates 2D images to fake a 3D stack: (B, H, W, C) -> (B, 3, H, W, C)
        return create_legacy_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            img_size=args.img_size
        )

def main():
    parser = argparse.ArgumentParser(description="Phoenix v3.1 Training Node")
    parser.add_argument('--data-dir', type=str, default='/home/shadow_garden/brain-tumor-detection/data', help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--force-legacy', action='store_true', help='Force use of 2D loader on Kaggle')
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()

    # 1. Setup Data
    print(f"Initializing Data Pipeline from: {args.data_dir}")
    try:
        train_ds, val_ds = get_data_loaders(args)
    except Exception as e:
        print(f"❌ Critical Error in Data Loading: {e}")
        return

    # 2. Build Model v3.1 (Topology Fixed)
    print("Initializing PHOENIX v3.1 (Hybrid Pyramid + SpatialMixer)...")

    # Strategy scope for Multi-GPU (Kaggle T4 x2)
    strategy = tf.distribute.MirroredStrategy()
    print(f"✅ Number of devices: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        # Note: Input shape has 3 frames (temporal/depth dimension)
        # (3, H, W, C) -> 3 slices context
        model = create_phoenix_v3_1(
            input_shape=(3, args.img_size, args.img_size, 3),
            num_classes=4
        )

        # 3. Compile
        print(f"Compiling with AdamW (lr={args.lr})...")
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=args.lr,
            weight_decay=1e-4
        )

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )

    model.summary()

    # 4. Train
    print(f"Starting Training Loop for {args.epochs} epochs...")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_phoenix_v3_1.keras', save_best_only=True, monitor='val_accuracy', mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.CSVLogger('training_log.csv')
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    print("✅ Training Complete. Model saved to best_phoenix_v3_1.keras")

if __name__ == "__main__":
    main()
