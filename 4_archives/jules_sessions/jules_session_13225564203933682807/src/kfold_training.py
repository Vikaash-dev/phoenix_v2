"""
k-Fold Cross-Validation for Phoenix Protocol
Implements stratified 5-fold cross-validation for reliable performance estimation.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.neurosnake_model import (
        create_neurosnake_model, 
        create_neurosnake_with_coordinate_attention,
        create_neurosnake_with_sevector_attention,
        create_baseline_model
    )
    from src.phoenix_optimizer import create_adan_optimizer, create_focal_loss
    from src.physics_informed_augmentation import PhysicsInformedAugmentation
    import config
except ImportError:
    print("Error importing Phoenix Protocol modules. Ensure you are in the project root.")
    sys.exit(1)


class KFoldTrainer:
    """
    Manages k-fold cross-validation training.
    """
    
    def __init__(
        self,
        data_dir,
        n_splits=5,
        model_type='neurosnake_ca',
        batch_size=32,
        epochs=50,
        output_dir='./results/kfold'
    ):
        self.data_dir = data_dir
        self.n_splits = n_splits
        self.model_type = model_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_dir = output_dir
        
        # Setup output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def _get_all_filepaths_and_labels(self):
        """
        Scan train directory to get all filepaths and labels for splitting.
        """
        train_dir = os.path.join(self.data_dir, 'train')
        class_names = sorted(os.listdir(train_dir))
        class_indices = {name: i for i, name in enumerate(class_names)}
        
        filepaths = []
        labels = []
        
        for class_name in class_names:
            class_dir = os.path.join(train_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    filepaths.append(os.path.join(class_dir, filename))
                    labels.append(class_indices[class_name])
                    
        return np.array(filepaths), np.array(labels), class_names
    
    def _create_dataset(self, filepaths, labels, class_names, is_training=True):
        """
        Create tf.data.Dataset from filepaths.
        """
        def process_path(filepath, label):
            # Read image
            img = tf.io.read_file(filepath)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, [config.IMG_HEIGHT, config.IMG_WIDTH])
            img = tf.cast(img, tf.float32) / 255.0
            
            # One-hot encode label
            label = tf.one_hot(label, len(class_names))
            return img, label
            
        dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
        dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training:
            # Physics-informed augmentation
            augmentor = PhysicsInformedAugmentation()
            
            def augment_fn(image, label):
                image_np = image.numpy()
                augmented = augmentor.augment(image_np)
                return tf.convert_to_tensor(augmented, dtype=tf.float32), label
            
            def apply_augmentation(image, label):
                image, label = tf.py_function(augment_fn, [image, label], [tf.float32, tf.float32])
                image.set_shape([config.IMG_HEIGHT, config.IMG_WIDTH, 3])
                label.set_shape([len(class_names)])
                return image, label
            
            dataset = dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.shuffle(buffer_size=1000)
            
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _create_model(self):
        """Create a fresh model instance."""
        if self.model_type == 'neurosnake':
            return create_neurosnake_model()
        elif self.model_type == 'neurosnake_ca':
            return create_neurosnake_with_coordinate_attention()
        elif self.model_type == 'neurosnake_se':
            return create_neurosnake_with_sevector_attention()
        elif self.model_type == 'baseline':
            return create_baseline_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
    def run(self):
        """Execute k-fold cross-validation."""
        print(f"Starting {self.n_splits}-Fold Cross-Validation for {self.model_type}")
        print("="*80)
        
        filepaths, labels, class_names = self._get_all_filepaths_and_labels()
        
        if len(filepaths) == 0:
            print("Error: No training data found!")
            return
            
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=config.RANDOM_SEED)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(filepaths, labels)):
            print(f"\nFold {fold+1}/{self.n_splits}")
            print("-" * 40)
            
            # Split data
            X_train, X_val = filepaths[train_idx], filepaths[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # Create datasets
            train_ds = self._create_dataset(X_train, y_train, class_names, is_training=True)
            val_ds = self._create_dataset(X_val, y_val, class_names, is_training=False)
            
            # Create and compile model
            model = self._create_model()
            
            optimizer = create_adan_optimizer(learning_rate=config.LEARNING_RATE)
            loss = create_focal_loss()
            
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=['accuracy', 'precision', 'recall']
            )
            
            # Callbacks
            fold_dir = os.path.join(self.output_dir, f'fold_{fold+1}')
            os.makedirs(fold_dir, exist_ok=True)
            
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    os.path.join(fold_dir, 'best_model.h5'),
                    save_best_only=True,
                    monitor='val_accuracy'
                ),
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.CSVLogger(os.path.join(fold_dir, 'training_log.csv'))
            ]
            
            # Train
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            scores = model.evaluate(val_ds, verbose=0)
            results = {k: v for k, v in zip(model.metrics_names, scores)}
            fold_results.append(results)
            
            print(f"Fold {fold+1} Results: {results}")
            
            # Clear session to free memory
            keras.backend.clear_session()
            
        # Aggregate results
        print("\n" + "="*80)
        print("CROSS-VALIDATION SUMMARY")
        print("="*80)
        
        avg_results = {}
        for metric in fold_results[0].keys():
            values = [res[metric] for res in fold_results]
            mean = np.mean(values)
            std = np.std(values)
            avg_results[metric] = {'mean': mean, 'std': std}
            print(f"{metric}: {mean:.4f} ± {std:.4f}")
            
        # Save summary
        summary_path = os.path.join(self.output_dir, 'cv_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'fold_results': fold_results,
                'average_results': avg_results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
            
        print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='k-Fold Cross-Validation')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--model', type=str, default='neurosnake_ca', 
                      choices=['neurosnake', 'neurosnake_ca', 'neurosnake_se', 'baseline'])
    
    args = parser.parse_args()
    
    trainer = KFoldTrainer(
        data_dir=args.data_dir,
        n_splits=args.folds,
        model_type=args.model
    )
    
    trainer.run()
