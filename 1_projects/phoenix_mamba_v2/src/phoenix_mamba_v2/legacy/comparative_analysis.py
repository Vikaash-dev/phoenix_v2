"""
Comparative Analysis Framework for Phoenix Protocol
Compares NeuroSnake vs Baseline performance with comprehensive metrics.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score
)
import pandas as pd


class PhoenixComparator:
    """
    Comparative analysis between NeuroSnake and Baseline models.
    """
    
    def __init__(self, output_dir: str = './results/comparison'):
        """
        Initialize comparator.
        
        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_model(
        self,
        model: keras.Model,
        test_dataset: tf.data.Dataset,
        model_name: str
    ) -> Dict:
        """
        Comprehensive evaluation of a model.
        
        Args:
            model: Keras model to evaluate
            test_dataset: Test dataset
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating {model_name}...")
        
        # Collect predictions and ground truth
        y_true = []
        y_pred = []
        y_pred_proba = []
        
        for images, labels in test_dataset:
            predictions = model.predict(images, verbose=0)
            y_pred_proba.extend(predictions)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(np.argmax(labels.numpy(), axis=1))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)
        
        # Compute metrics
        accuracy = np.mean(y_true == y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=['no_tumor', 'tumor'],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Specificity and Sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # False rates
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        
        # ROC AUC
        if len(np.unique(y_true)) > 1:
            fpr_roc, tpr_roc, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr_roc, tpr_roc)
        else:
            roc_auc = 0.0
        
        # F1 Score
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        results = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'precision': float(report['weighted avg']['precision']),
            'recall': float(report['weighted avg']['recall']),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'sensitivity': float(sensitivity),
            'false_negative_rate': float(fnr),
            'false_positive_rate': float(fpr),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'classification_report': report
        }
        
        return results
    
    def compare_models(
        self,
        neurosnake_results: Dict,
        baseline_results: Dict
    ) -> Dict:
        """
        Compare NeuroSnake vs Baseline results.
        
        Args:
            neurosnake_results: NeuroSnake evaluation results
            baseline_results: Baseline evaluation results
            
        Returns:
            Comparison dictionary
        """
        print("\n" + "="*80)
        print("PHOENIX PROTOCOL: COMPARATIVE ANALYSIS")
        print("="*80)
        
        metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'specificity', 'sensitivity', 'false_negative_rate',
            'false_positive_rate', 'roc_auc'
        ]
        
        comparison = {
            'neurosnake': {},
            'baseline': {},
            'improvement': {}
        }
        
        print("\n{:<30} {:>15} {:>15} {:>15}".format(
            "Metric", "NeuroSnake", "Baseline", "Improvement"
        ))
        print("-" * 80)
        
        for metric in metrics:
            ns_value = neurosnake_results.get(metric, 0)
            bl_value = baseline_results.get(metric, 0)
            
            # For error rates, lower is better
            if 'false' in metric or 'rate' in metric:
                improvement = bl_value - ns_value  # Positive = NeuroSnake better
                improvement_pct = (improvement / bl_value * 100) if bl_value != 0 else 0
            else:
                improvement = ns_value - bl_value
                improvement_pct = (improvement / bl_value * 100) if bl_value != 0 else 0
            
            comparison['neurosnake'][metric] = ns_value
            comparison['baseline'][metric] = bl_value
            comparison['improvement'][metric] = {
                'absolute': float(improvement),
                'percentage': float(improvement_pct)
            }
            
            print("{:<30} {:>15.4f} {:>15.4f} {:>14.2f}%".format(
                metric.replace('_', ' ').title(),
                ns_value,
                bl_value,
                improvement_pct
            ))
        
        print("="*80)
        
        # Clinical significance analysis
        print("\nClinical Significance Analysis:")
        print("-" * 80)
        
        # False Negative Rate (Critical for medical applications)
        ns_fnr = neurosnake_results['false_negative_rate']
        bl_fnr = baseline_results['false_negative_rate']
        fnr_improvement = (bl_fnr - ns_fnr) / bl_fnr * 100 if bl_fnr > 0 else 0
        
        print(f"False Negative Rate Reduction: {fnr_improvement:.2f}%")
        if fnr_improvement > 10:
            print("✓ Significant improvement in detecting tumors (>10% FNR reduction)")
        elif fnr_improvement > 0:
            print("✓ Modest improvement in detecting tumors")
        else:
            print("⚠ No improvement in False Negative Rate")
        
        # False Positive Rate
        ns_fpr = neurosnake_results['false_positive_rate']
        bl_fpr = baseline_results['false_positive_rate']
        fpr_improvement = (bl_fpr - ns_fpr) / bl_fpr * 100 if bl_fpr > 0 else 0
        
        print(f"False Positive Rate Reduction: {fpr_improvement:.2f}%")
        if fpr_improvement > 10:
            print("✓ Significant reduction in false alarms (>10% FPR reduction)")
        elif fpr_improvement > 0:
            print("✓ Modest reduction in false alarms")
        else:
            print("⚠ No improvement in False Positive Rate")
        
        print("="*80 + "\n")
        
        return comparison
    
    def plot_comparison(
        self,
        comparison: Dict,
        save_path: Optional[str] = None
    ):
        """
        Create visualization of comparison results.
        
        Args:
            comparison: Comparison dictionary
            save_path: Path to save plot
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        ns_values = [comparison['neurosnake'][m] for m in metrics]
        bl_values = [comparison['baseline'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, ns_values, width, label='NeuroSnake', color='#2ecc71')
        bars2 = ax.bar(x + width/2, bl_values, width, label='Baseline', color='#3498db')
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Phoenix Protocol: NeuroSnake vs Baseline Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0.85, 1.0])
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=9)
        
        autolabel(bars1)
        autolabel(bars2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison plot saved to: {save_path}")
        
        plt.close()
    
    def plot_confusion_matrices(
        self,
        neurosnake_results: Dict,
        baseline_results: Dict,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrices side by side.
        
        Args:
            neurosnake_results: NeuroSnake results
            baseline_results: Baseline results
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        cm_ns = np.array(neurosnake_results['confusion_matrix'])
        cm_bl = np.array(baseline_results['confusion_matrix'])
        
        # NeuroSnake
        sns.heatmap(cm_ns, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['No Tumor', 'Tumor'],
                   yticklabels=['No Tumor', 'Tumor'])
        axes[0].set_title('NeuroSnake', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        
        # Baseline
        sns.heatmap(cm_bl, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                   xticklabels=['No Tumor', 'Tumor'],
                   yticklabels=['No Tumor', 'Tumor'])
        axes[1].set_title('Baseline', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrices saved to: {save_path}")
        
        plt.close()
    
    def generate_report(
        self,
        comparison: Dict,
        neurosnake_results: Dict,
        baseline_results: Dict
    ):
        """
        Generate comprehensive comparison report.
        
        Args:
            comparison: Comparison dictionary
            neurosnake_results: NeuroSnake results
            baseline_results: Baseline results
        """
        report_path = os.path.join(self.output_dir, 'comparison_report.json')
        
        full_report = {
            'comparison': comparison,
            'neurosnake_detailed': neurosnake_results,
            'baseline_detailed': baseline_results
        }
        
        with open(report_path, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        print(f"✓ Full comparison report saved to: {report_path}")
        
        # Generate markdown report
        md_path = os.path.join(self.output_dir, 'comparison_report.md')
        with open(md_path, 'w') as f:
            f.write("# Phoenix Protocol: Comparative Analysis Report\n\n")
            f.write("## Performance Comparison\n\n")
            
            f.write("| Metric | NeuroSnake | Baseline | Improvement |\n")
            f.write("|--------|------------|----------|-------------|\n")
            
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                ns_val = comparison['neurosnake'][metric]
                bl_val = comparison['baseline'][metric]
                imp = comparison['improvement'][metric]['percentage']
                f.write(f"| {metric.replace('_', ' ').title()} | {ns_val:.4f} | {bl_val:.4f} | {imp:+.2f}% |\n")
            
            f.write("\n## Clinical Metrics\n\n")
            f.write("| Metric | NeuroSnake | Baseline |\n")
            f.write("|--------|------------|----------|\n")
            f.write(f"| False Negative Rate | {neurosnake_results['false_negative_rate']:.4f} | {baseline_results['false_negative_rate']:.4f} |\n")
            f.write(f"| False Positive Rate | {neurosnake_results['false_positive_rate']:.4f} | {baseline_results['false_positive_rate']:.4f} |\n")
            f.write(f"| Sensitivity | {neurosnake_results['sensitivity']:.4f} | {baseline_results['sensitivity']:.4f} |\n")
            f.write(f"| Specificity | {neurosnake_results['specificity']:.4f} | {baseline_results['specificity']:.4f} |\n")
        
        print(f"✓ Markdown report saved to: {md_path}")


if __name__ == "__main__":
    print("Phoenix Protocol: Comparative Analysis Framework")
    print("Use this module to compare NeuroSnake and Baseline models.")
    print("\nExample usage:")
    print("""
    from phoenix_mamba_v2.legacy.comparative_analysis import PhoenixComparator
    
    comparator = PhoenixComparator(output_dir='./results/comparison')
    
    # Evaluate models
    ns_results = comparator.evaluate_model(neurosnake_model, test_data, 'NeuroSnake')
    bl_results = comparator.evaluate_model(baseline_model, test_data, 'Baseline')
    
    # Compare
    comparison = comparator.compare_models(ns_results, bl_results)
    
    # Generate visualizations and reports
    comparator.plot_comparison(comparison, 'comparison.png')
    comparator.plot_confusion_matrices(ns_results, bl_results, 'confusion_matrices.png')
    comparator.generate_report(comparison, ns_results, bl_results)
    """)
