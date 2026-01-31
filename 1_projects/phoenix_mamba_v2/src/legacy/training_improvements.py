"""
P0 Critical Training Improvements
Based on cross-analysis with nnU-Net, MONAI, and academic best practices
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import os


def set_seed(seed=42):
    """
    Fix all random seeds for reproducibility
    Critical P0 feature identified in cross-analysis
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Additional TensorFlow determinism
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    print(f"✅ All random seeds fixed to {seed} for reproducibility")


class MixedPrecisionConfig:
    """
    Mixed Precision Training Configuration
    Critical P0 feature - provides 2-3x speedup
    
    Based on TensorFlow AMP best practices
    """
    
    @staticmethod
    def enable_mixed_precision(policy='mixed_float16'):
        """
        Enable automatic mixed precision training
        
        Args:
            policy: 'mixed_float16' (GPU) or 'mixed_bfloat16' (TPU)
        
        Returns:
            True if enabled successfully
        """
        try:
            policy = keras.mixed_precision.Policy(policy)
            keras.mixed_precision.set_global_policy(policy)
            print(f"✅ Mixed precision enabled: {policy.name}")
            print(f"   Compute dtype: {policy.compute_dtype}")
            print(f"   Variable dtype: {policy.variable_dtype}")
            return True
        except Exception as e:
            print(f"⚠️  Mixed precision not available: {e}")
            return False
    
    @staticmethod
    def wrap_optimizer(optimizer, loss_scale='dynamic'):
        """
        Wrap optimizer with loss scaling for mixed precision
        
        Args:
            optimizer: Base optimizer
            loss_scale: 'dynamic' or fixed float value
        
        Returns:
            Wrapped optimizer
        """
        if loss_scale == 'dynamic':
            return keras.mixed_precision.LossScaleOptimizer(optimizer)
        elif isinstance(loss_scale, (int, float)):
            return keras.mixed_precision.LossScaleOptimizer(
                optimizer, dynamic=False, initial_scale=loss_scale
            )
        return optimizer


class LearningRateSchedulers:
    """
    Advanced Learning Rate Schedulers
    Critical P0 feature identified in cross-analysis
    
    Implements state-of-the-art scheduling strategies
    """
    
    @staticmethod
    def cosine_annealing(initial_lr, epochs, warmup_epochs=5, min_lr=1e-6):
        """
        Cosine annealing with warmup
        Used in: ResNet, EfficientNet, ViT papers
        
        Args:
            initial_lr: Starting learning rate
            epochs: Total training epochs
            warmup_epochs: Linear warmup epochs
            min_lr: Minimum learning rate
        
        Returns:
            Learning rate scheduler callback
        """
        def schedule(epoch, lr):
            if epoch < warmup_epochs:
                # Linear warmup
                return initial_lr * (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return min_lr + (initial_lr - min_lr) * 0.5 * (
                    1 + np.cos(np.pi * progress)
                )
        
        return keras.callbacks.LearningRateScheduler(schedule, verbose=1)
    
    @staticmethod
    def reduce_on_plateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6):
        """
        Reduce learning rate when metric plateaus
        Used in: Medical imaging papers
        
        Args:
            monitor: Metric to monitor
            factor: Factor to reduce LR
            patience: Number of epochs with no improvement
            min_lr: Minimum learning rate
        
        Returns:
            ReduceLROnPlateau callback
        """
        return keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=1,
            mode='min' if 'loss' in monitor else 'max'
        )
    
    @staticmethod
    def one_cycle(max_lr, epochs, steps_per_epoch):
        """
        One Cycle Learning Rate Policy
        Reference: "Super-Convergence" paper (arxiv:1708.07120)
        
        Args:
            max_lr: Maximum learning rate
            epochs: Total epochs
            steps_per_epoch: Training steps per epoch
        
        Returns:
            OneCycleLR schedule function
        """
        total_steps = epochs * steps_per_epoch
        
        def schedule(step):
            # Phase 1: Increase (45% of training)
            if step < total_steps * 0.45:
                return max_lr * step / (total_steps * 0.45)
            # Phase 2: Decrease (45% of training)
            elif step < total_steps * 0.9:
                progress = (step - total_steps * 0.45) / (total_steps * 0.45)
                return max_lr * (1 - progress)
            # Phase 3: Final decrease (10% of training)
            else:
                return max_lr * 0.01
        
        return schedule


class EarlyStoppingWithRestore(keras.callbacks.EarlyStopping):
    """
    Early stopping with model restoration
    Critical P0 feature - prevents overfitting
    
    Extends EarlyStopping to restore best weights
    """
    
    def __init__(self, monitor='val_loss', patience=10, min_delta=0.001, 
                 restore_best_weights=True, verbose=1, **kwargs):
        super().__init__(
            monitor=monitor,
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=restore_best_weights,
            verbose=verbose,
            **kwargs
        )
        self.mode = 'min' if 'loss' in monitor else 'max'
    
    def on_train_end(self, logs=None):
        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose > 0:
                print(f"\n✅ Restoring best weights from epoch {self.best_epoch + 1}")
                print(f"   Best {self.monitor}: {self.best:.4f}")
        super().on_train_end(logs)


class GradientClipping(keras.callbacks.Callback):
    """
    Gradient clipping callback
    Critical P0 feature - prevents gradient explosion
    
    Implements global norm clipping
    """
    
    def __init__(self, clip_norm=1.0):
        super().__init__()
        self.clip_norm = clip_norm
        self.gradient_norms = []
    
    def on_train_begin(self, logs=None):
        # Wrap optimizer with gradient clipping
        if hasattr(self.model.optimizer, 'clipnorm'):
            self.model.optimizer.clipnorm = self.clip_norm
            print(f"✅ Gradient clipping enabled: norm <= {self.clip_norm}")
        else:
            print(f"⚠️  Optimizer doesn't support clipnorm, using clipvalue")
            self.model.optimizer.clipvalue = self.clip_norm
    
    def on_batch_end(self, batch, logs=None):
        # Track gradient norms for monitoring
        if hasattr(self.model.optimizer, 'get_gradients'):
            try:
                gradients = self.model.optimizer.get_gradients(
                    self.model.total_loss, self.model.trainable_weights
                )
                norm = tf.linalg.global_norm(gradients)
                self.gradient_norms.append(float(norm))
            except:
                pass


class AdvancedModelCheckpoint(keras.callbacks.ModelCheckpoint):
    """
    Enhanced model checkpoint with best model tracking
    Critical P0 feature - saves best model, not last
    
    Saves both full model and weights
    """
    
    def __init__(self, filepath, monitor='val_loss', save_best_only=True,
                 save_weights_only=False, mode='auto', verbose=1, **kwargs):
        super().__init__(
            filepath=filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            verbose=verbose,
            **kwargs
        )
        self.best_metric_value = None
    
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        
        # Track best metric
        current = logs.get(self.monitor)
        if current is not None:
            if self.best_metric_value is None:
                self.best_metric_value = current
            else:
                if self.monitor_op(current, self.best_metric_value):
                    self.best_metric_value = current
                    if self.verbose > 0:
                        print(f"\n✅ New best {self.monitor}: {current:.4f}")


class KFoldCrossValidation:
    """
    K-Fold Cross-Validation for Medical Imaging
    Critical P0 feature identified in cross-analysis
    
    Implements patient-level stratified k-fold
    """
    
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        """
        Initialize k-fold cross-validation
        
        Args:
            n_splits: Number of folds (typically 5 for medical data)
            shuffle: Whether to shuffle before splitting
            random_state: Random seed for reproducibility
        """
        from sklearn.model_selection import StratifiedKFold
        self.kfold = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=shuffle, 
            random_state=random_state
        )
        self.n_splits = n_splits
        self.fold_results = []
    
    def split(self, X, y, patient_ids=None):
        """
        Generate train/val splits
        
        Args:
            X: Features
            y: Labels
            patient_ids: Optional patient IDs for patient-level splitting
        
        Yields:
            (train_idx, val_idx) tuples
        """
        if patient_ids is not None:
            # Patient-level splitting (critical for medical data)
            unique_patients = np.unique(patient_ids)
            patient_labels = np.array([
                y[patient_ids == p][0] for p in unique_patients
            ])
            
            for train_patients_idx, val_patients_idx in self.kfold.split(
                unique_patients, patient_labels
            ):
                train_patients = unique_patients[train_patients_idx]
                val_patients = unique_patients[val_patients_idx]
                
                train_idx = np.where(np.isin(patient_ids, train_patients))[0]
                val_idx = np.where(np.isin(patient_ids, val_patients))[0]
                
                yield train_idx, val_idx
        else:
            # Standard splitting
            for train_idx, val_idx in self.kfold.split(X, y):
                yield train_idx, val_idx
    
    def record_fold_result(self, fold_idx, metrics):
        """
        Record results for a fold
        
        Args:
            fold_idx: Fold index (0 to n_splits-1)
            metrics: Dictionary of metrics
        """
        self.fold_results.append({
            'fold': fold_idx,
            **metrics
        })
    
    def get_summary_statistics(self):
        """
        Calculate summary statistics across folds
        
        Returns:
            Dictionary with mean and std for each metric
        """
        import pandas as pd
        
        df = pd.DataFrame(self.fold_results)
        summary = {}
        
        for col in df.columns:
            if col != 'fold':
                summary[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
        
        return summary
    
    def print_summary(self):
        """Print formatted summary statistics"""
        summary = self.get_summary_statistics()
        
        print("\n" + "="*60)
        print(f"K-FOLD CROSS-VALIDATION SUMMARY ({self.n_splits} folds)")
        print("="*60)
        
        for metric, stats in summary.items():
            print(f"\n{metric.upper()}:")
            print(f"  Mean:  {stats['mean']:.4f}")
            print(f"  Std:   {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        print("="*60)


def create_production_callbacks(
    model_path='models/best_model.h5',
    monitor='val_loss',
    patience=10,
    lr_schedule='cosine',
    initial_lr=1e-3,
    epochs=100,
    clip_norm=1.0
):
    """
    Create production-ready callback suite
    Implements all P0 features from cross-analysis
    
    Args:
        model_path: Path to save best model
        monitor: Metric to monitor
        patience: Early stopping patience
        lr_schedule: 'cosine', 'plateau', or 'onecycle'
        initial_lr: Initial learning rate
        epochs: Total epochs
        clip_norm: Gradient clipping norm
    
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # 1. Model checkpoint (best model)
    callbacks.append(AdvancedModelCheckpoint(
        filepath=model_path,
        monitor=monitor,
        save_best_only=True,
        verbose=1
    ))
    
    # 2. Early stopping with restore
    callbacks.append(EarlyStoppingWithRestore(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        verbose=1
    ))
    
    # 3. Learning rate scheduler
    if lr_schedule == 'cosine':
        callbacks.append(LearningRateSchedulers.cosine_annealing(
            initial_lr=initial_lr,
            epochs=epochs,
            warmup_epochs=5
        ))
    elif lr_schedule == 'plateau':
        callbacks.append(LearningRateSchedulers.reduce_on_plateau(
            monitor=monitor,
            patience=patience//2
        ))
    # onecycle requires steps_per_epoch, added during training
    
    # 4. Gradient clipping
    callbacks.append(GradientClipping(clip_norm=clip_norm))
    
    # 5. TensorBoard
    callbacks.append(keras.callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    ))
    
    # 6. CSV Logger
    callbacks.append(keras.callbacks.CSVLogger(
        'training_log.csv',
        append=True
    ))
    
    print(f"\n✅ Created {len(callbacks)} production callbacks:")
    print(f"   - Best model checkpoint ({monitor})")
    print(f"   - Early stopping (patience={patience})")
    print(f"   - LR scheduler ({lr_schedule})")
    print(f"   - Gradient clipping (norm<={clip_norm})")
    print(f"   - TensorBoard logging")
    print(f"   - CSV logging")
    
    return callbacks


# Example usage
if __name__ == "__main__":
    print("Phoenix Protocol - P0 Training Improvements")
    print("Based on cross-analysis with SOTA projects\n")
    
    # 1. Set seed for reproducibility
    set_seed(42)
    
    # 2. Enable mixed precision
    MixedPrecisionConfig.enable_mixed_precision()
    
    # 3. Create production callbacks
    callbacks = create_production_callbacks(
        model_path='models/best_neurosnake.h5',
        monitor='val_accuracy',
        patience=15,
        lr_schedule='cosine',
        initial_lr=1e-3,
        epochs=100,
        clip_norm=1.0
    )
    
    # 4. Setup k-fold cross-validation
    kfold = KFoldCrossValidation(n_splits=5, random_state=42)
    print(f"\n✅ K-Fold CV initialized ({kfold.n_splits} folds)")
    
    print("\n✅ All P0 improvements ready for production training")
