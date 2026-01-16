"""
Experimental Framework for Research-Grade Validation
Implements K-fold cross-validation, statistical testing, and comprehensive metrics.

Key Features:
- Stratified K-fold cross-validation with patient-level splitting
- Statistical significance testing (paired t-test, Wilcoxon signed-rank)
- Bootstrapped 95% confidence intervals
- Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC, sensitivity, specificity)
- Experiment logging and reproducibility
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Callable, Optional, Tuple, Any
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from scipy import stats
import config


class ExperimentalFramework:
    """
    Complete experimental framework for research-grade validation.
    
    Features:
    - K-fold cross-validation (5-fold default) with stratification
    - Patient-level splitting to prevent data leakage
    - Statistical significance testing (paired t-test, Wilcoxon signed-rank)
    - 95% confidence interval estimation via bootstrapping
    - Comprehensive metric computation
    - Experiment logging and reproducibility (seed management)
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str,
        n_folds: int = 5,
        random_seed: int = 42
    ):
        """
        Initialize experimental framework.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save results
            n_folds: Number of folds for cross-validation
            random_seed: Random seed for reproducibility
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.n_folds = n_folds
        self.random_seed = random_seed
        
        # Create directory structure
        os.makedirs(output_dir, exist_ok=True)
        self.experiment_dir = os.path.join(
            output_dir, 
            f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Set all random seeds for reproducibility
        self._set_all_seeds(random_seed)
        
        self.logger.info(f"Initialized experiment: {experiment_name}")
        self.logger.info(f"Output directory: {self.experiment_dir}")
        self.logger.info(f"Number of folds: {n_folds}")
        self.logger.info(f"Random seed: {random_seed}")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_file = os.path.join(self.experiment_dir, 'experiment.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(self.experiment_name)
    
    def _set_all_seeds(self, seed: int):
        """
        Set all random seeds for reproducibility.
        
        Args:
            seed: Random seed value
        """
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        
        self.logger.info(f"Set all random seeds to {seed}")
    
    def run_kfold_experiment(
        self,
        model_factory: Callable,
        X: np.ndarray,
        y: np.ndarray,
        patient_ids: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.15,
        callbacks: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Run complete k-fold cross-validation.
        
        Args:
            model_factory: Function that returns a fresh compiled model
            X: Input data (images)
            y: Labels (one-hot encoded or class indices)
            patient_ids: Optional patient IDs for group stratification
            epochs: Training epochs per fold
            batch_size: Batch size
            validation_split: Validation split within training data
            callbacks: Optional list of Keras callbacks
        
        Returns:
            Dictionary with:
            - fold_results: Per-fold metrics
            - aggregated: Mean ± std across folds with 95% CI
            - predictions: Predictions for each fold
        """
        self.logger.info("Starting k-fold cross-validation")
        self.logger.info(f"Data shape: X={X.shape}, y={y.shape}")
        
        # Convert one-hot to class indices if needed
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_classes = np.argmax(y, axis=1)
        else:
            y_classes = y
        
        # Choose appropriate cross-validation strategy
        if patient_ids is not None:
            self.logger.info("Using GroupKFold (patient-level splitting)")
            kfold = GroupKFold(n_splits=self.n_folds)
            splits = kfold.split(X, y_classes, groups=patient_ids)
        else:
            self.logger.info("Using StratifiedKFold")
            kfold = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_seed
            )
            splits = kfold.split(X, y_classes)
        
        fold_results = []
        all_predictions = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Fold {fold_idx + 1}/{self.n_folds}")
            self.logger.info(f"{'='*60}")
            
            # Split data
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            self.logger.info(f"Train samples: {len(X_train_fold)}")
            self.logger.info(f"Test samples: {len(X_test_fold)}")
            
            # Reset seeds for this fold
            self._set_all_seeds(self.random_seed + fold_idx)
            
            # Create fresh model
            model = model_factory()
            
            # Train model
            self.logger.info("Training model...")
            history = model.fit(
                X_train_fold,
                y_train_fold,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate on test fold
            self.logger.info("Evaluating model...")
            y_pred_proba = model.predict(X_test_fold, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            if len(y_test_fold.shape) > 1:
                y_test_classes = np.argmax(y_test_fold, axis=1)
            else:
                y_test_classes = y_test_fold
            
            # Compute metrics
            metrics = self._compute_metrics(y_test_classes, y_pred, y_pred_proba)
            metrics['fold'] = fold_idx + 1
            metrics['train_samples'] = len(X_train_fold)
            metrics['test_samples'] = len(X_test_fold)
            
            fold_results.append(metrics)
            all_predictions.append({
                'y_true': y_test_classes,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'test_idx': test_idx
            })
            
            # Log fold results
            self.logger.info(f"Fold {fold_idx + 1} Results:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and metric_name != 'fold':
                    self.logger.info(f"  {metric_name}: {value:.4f}")
            
            # Clear model to free memory
            del model
            tf.keras.backend.clear_session()
        
        # Aggregate results across folds
        self.logger.info("\n" + "="*60)
        self.logger.info("Aggregating results across folds")
        self.logger.info("="*60)
        
        aggregated = self._aggregate_results(fold_results)
        
        # Pre-convert numpy types for efficient JSON serialization
        serializable_fold_results = []
        for fold_result in fold_results:
            serializable_result = {}
            for key, value in fold_result.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_result[key] = float(value) if isinstance(value, np.floating) else int(value)
                elif isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                else:
                    serializable_result[key] = value
            serializable_fold_results.append(serializable_result)
        
        # Save results
        results = {
            'experiment_name': self.experiment_name,
            'n_folds': self.n_folds,
            'random_seed': self.random_seed,
            'fold_results': serializable_fold_results,
            'aggregated': aggregated,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = os.path.join(self.experiment_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"\nResults saved to: {results_file}")
        
        return {
            'fold_results': fold_results,
            'aggregated': aggregated,
            'predictions': all_predictions,
            'experiment_dir': self.experiment_dir
        }
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics.
        
        Args:
            y_true: True labels (class indices)
            y_pred: Predicted labels (class indices)
            y_pred_proba: Predicted probabilities (optional)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # Sensitivity and Specificity (for binary classification)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # ROC-AUC (if probabilities available)
        if y_pred_proba is not None:
            try:
                if y_pred_proba.shape[1] == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    # Multi-class (use OVR strategy)
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_pred_proba, multi_class='ovr', average='macro'
                    )
            except Exception as e:
                self.logger.warning(f"Could not compute ROC-AUC: {e}")
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def _aggregate_results(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """
        Compute mean, std, and 95% confidence intervals via bootstrapping.
        
        Args:
            fold_results: List of per-fold metric dictionaries
        
        Returns:
            Dictionary with aggregated statistics
        """
        aggregated = {}
        
        # Get all metric names (excluding non-numeric fields)
        metric_names = [
            key for key in fold_results[0].keys()
            if isinstance(fold_results[0][key], (int, float))
            and key not in ['fold', 'train_samples', 'test_samples']
        ]
        
        for metric_name in metric_names:
            values = [result[metric_name] for result in fold_results]
            
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)  # Sample std
            
            # Compute 95% CI via bootstrapping
            ci_lower, ci_upper = self._bootstrap_ci(values)
            
            aggregated[metric_name] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'ci_95_lower': float(ci_lower),
                'ci_95_upper': float(ci_upper),
                'values': [float(v) for v in values]
            }
            
            self.logger.info(
                f"{metric_name}: {mean_val:.4f} ± {std_val:.4f} "
                f"[95% CI: {ci_lower:.4f}, {ci_upper:.4f}]"
            )
        
        return aggregated
    
    def _bootstrap_ci(
        self,
        values: List[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute confidence interval using bootstrap resampling.
        
        Args:
            values: List of values
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (default 0.95 for 95% CI)
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        values = np.array(values)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return ci_lower, ci_upper
    
    def compare_models(
        self,
        results_a: Dict[str, Any],
        results_b: Dict[str, Any],
        name_a: str = "Model A",
        name_b: str = "Model B",
        metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Statistical comparison using paired t-test and Wilcoxon signed-rank test.
        
        Args:
            results_a: Results from model A (from run_kfold_experiment)
            results_b: Results from model B (from run_kfold_experiment)
            name_a: Name of model A
            name_b: Name of model B
            metric: Metric to compare (default: 'accuracy')
        
        Returns:
            Dictionary with statistical test results
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Statistical Comparison: {name_a} vs {name_b}")
        self.logger.info(f"Metric: {metric}")
        self.logger.info(f"{'='*60}")
        
        # Extract values for the specified metric
        values_a = results_a['aggregated'][metric]['values']
        values_b = results_b['aggregated'][metric]['values']
        
        if len(values_a) != len(values_b):
            raise ValueError("Both models must have the same number of folds")
        
        # Paired t-test (assumes normality)
        t_statistic, t_pvalue = stats.ttest_rel(values_a, values_b)
        
        # Wilcoxon signed-rank test (non-parametric)
        try:
            w_statistic, w_pvalue = stats.wilcoxon(values_a, values_b)
        except ValueError:
            # If differences are all zero
            w_statistic, w_pvalue = 0, 1.0
        
        # Effect size (Cohen's d for paired samples)
        differences = np.array(values_a) - np.array(values_b)
        cohen_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences) > 0 else 0
        
        # Determine significance
        alpha = 0.05
        t_significant = t_pvalue < alpha
        w_significant = w_pvalue < alpha
        
        results = {
            'model_a': name_a,
            'model_b': name_b,
            'metric': metric,
            'values_a': values_a,
            'values_b': values_b,
            'mean_a': float(np.mean(values_a)),
            'mean_b': float(np.mean(values_b)),
            'mean_difference': float(np.mean(differences)),
            'paired_t_test': {
                'statistic': float(t_statistic),
                'p_value': float(t_pvalue),
                'significant': bool(t_significant),
                'alpha': alpha
            },
            'wilcoxon_test': {
                'statistic': float(w_statistic),
                'p_value': float(w_pvalue),
                'significant': bool(w_significant),
                'alpha': alpha
            },
            'effect_size': {
                'cohen_d': float(cohen_d),
                'interpretation': self._interpret_cohen_d(cohen_d)
            }
        }
        
        # Log results
        self.logger.info(f"\n{name_a}: {results['mean_a']:.4f}")
        self.logger.info(f"{name_b}: {results['mean_b']:.4f}")
        self.logger.info(f"Mean difference: {results['mean_difference']:.4f}")
        self.logger.info(f"\nPaired t-test:")
        self.logger.info(f"  t-statistic: {t_statistic:.4f}")
        self.logger.info(f"  p-value: {t_pvalue:.4f}")
        self.logger.info(f"  Significant: {t_significant}")
        self.logger.info(f"\nWilcoxon signed-rank test:")
        self.logger.info(f"  W-statistic: {w_statistic:.4f}")
        self.logger.info(f"  p-value: {w_pvalue:.4f}")
        self.logger.info(f"  Significant: {w_significant}")
        self.logger.info(f"\nEffect size (Cohen's d): {cohen_d:.4f} ({self._interpret_cohen_d(cohen_d)})")
        
        # Save comparison results
        comparison_file = os.path.join(
            self.experiment_dir,
            f"comparison_{name_a}_vs_{name_b}_{metric}.json"
        )
        with open(comparison_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"\nComparison saved to: {comparison_file}")
        
        return results
    
    def _interpret_cohen_d(self, d: float) -> str:
        """
        Interpret Cohen's d effect size.
        
        Args:
            d: Cohen's d value
        
        Returns:
            Interpretation string
        """
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _numpy_to_python(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


class AblationStudyFramework:
    """Framework for systematic ablation studies."""
    
    def __init__(self, base_config: Dict[str, Any], output_dir: str):
        """
        Initialize ablation study framework.
        
        Args:
            base_config: Base configuration dictionary
            output_dir: Directory to save results
        """
        self.base_config = base_config
        self.output_dir = output_dir
        self.ablation_results = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AblationStudy')
    
    def run_ablation(
        self,
        ablation_name: str,
        config_modifications: Dict[str, Any],
        model_factory: Callable,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Run single ablation experiment with k-fold CV.
        
        Args:
            ablation_name: Name of the ablation
            config_modifications: Dictionary of config changes
            model_factory: Function to create model with config
            X: Input data
            y: Labels
            n_folds: Number of folds
            epochs: Training epochs
            batch_size: Batch size
        
        Returns:
            Dictionary with ablation results
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Running ablation: {ablation_name}")
        self.logger.info(f"Modifications: {config_modifications}")
        self.logger.info(f"{'='*60}")
        
        # Create experiment framework
        exp_framework = ExperimentalFramework(
            experiment_name=f"ablation_{ablation_name}",
            output_dir=self.output_dir,
            n_folds=n_folds
        )
        
        # Run k-fold experiment
        results = exp_framework.run_kfold_experiment(
            model_factory=model_factory,
            X=X,
            y=y,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Store ablation results
        ablation_result = {
            'name': ablation_name,
            'modifications': config_modifications,
            'results': results['aggregated']
        }
        
        self.ablation_results.append(ablation_result)
        
        return ablation_result
    
    def generate_ablation_table(self, baseline_name: str = "full_model") -> str:
        """
        Generate markdown table showing Δ accuracy for each ablation.
        
        Args:
            baseline_name: Name of the baseline configuration
        
        Returns:
            Markdown formatted table
        """
        if not self.ablation_results:
            return "No ablation results available"
        
        # Find baseline results
        baseline_result = None
        for result in self.ablation_results:
            if result['name'] == baseline_name:
                baseline_result = result
                break
        
        if baseline_result is None:
            self.logger.warning(f"Baseline '{baseline_name}' not found, using first result")
            baseline_result = self.ablation_results[0]
        
        baseline_acc = baseline_result['results']['accuracy']['mean']
        
        # Generate table
        table = "| Configuration | Accuracy (%) | Δ Accuracy (%) | 95% CI |\n"
        table += "|---------------|--------------|----------------|--------|\n"
        
        for result in self.ablation_results:
            name = result['name']
            acc = result['results']['accuracy']['mean']
            ci_lower = result['results']['accuracy']['ci_95_lower']
            ci_upper = result['results']['accuracy']['ci_95_upper']
            delta = acc - baseline_acc
            
            table += f"| {name} | {acc*100:.2f} | "
            if delta >= 0:
                table += f"+{delta*100:.2f} | "
            else:
                table += f"{delta*100:.2f} | "
            table += f"[{ci_lower*100:.2f}, {ci_upper*100:.2f}] |\n"
        
        self.logger.info("\nAblation Study Results:")
        self.logger.info(table)
        
        # Save table
        table_file = os.path.join(self.output_dir, 'ablation_table.md')
        with open(table_file, 'w') as f:
            f.write(table)
        
        self.logger.info(f"Ablation table saved to: {table_file}")
        
        return table
    
    def save_results(self):
        """Save all ablation results to JSON."""
        results_file = os.path.join(self.output_dir, 'ablation_results.json')
        
        with open(results_file, 'w') as f:
            json.dump(self.ablation_results, f, indent=2)
        
        self.logger.info(f"Ablation results saved to: {results_file}")
