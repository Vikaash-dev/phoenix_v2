"""
Research Visualizations for Publication-Quality Figures
Generates high-resolution (300 DPI) figures in PDF and PNG formats.

Features:
- SOTA model comparison bar charts with error bars
- Multi-metric radar charts
- Ablation study heatmaps
- Training curves
- ROC curves
- Confusion matrices
- GradCAM visualizations
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import logging


class ResearchVisualizer:
    """Generate publication-quality visualizations (300 DPI, PDF/PNG)."""
    
    def __init__(self, output_dir: str = './research_results/figures'):
        """
        Initialize research visualizer.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set matplotlib parameters for publication quality
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.format': 'png',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 1.2,
            'grid.alpha': 0.3,
            'legend.frameon': False
        })
        
        # Define consistent color palette
        self.colors = {
            'NeuroSnake': '#2ecc71',  # Green (highlight)
            'NeuroSnake_CoordAttn': '#27ae60',  # Dark green variant
            'ResNet50': '#3498db',  # Blue
            'EfficientNetB0': '#e74c3c',  # Red
            'VGG16': '#9b59b6',  # Purple
            'Baseline_CNN': '#f39c12',  # Orange
            'MobileNetV2': '#1abc9c',  # Teal
            'DenseNet121': '#34495e'  # Dark gray
        }
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('ResearchVisualizer')
        
        self.logger.info(f"Initialized ResearchVisualizer")
        self.logger.info(f"Output directory: {output_dir}")
    
    def plot_sota_comparison_bar(
        self,
        results: Dict[str, Dict],
        metric: str = 'accuracy',
        title: Optional[str] = None,
        filename: str = 'sota_comparison_bar'
    ):
        """
        Figure 1: Bar chart with error bars comparing all models.
        
        Args:
            results: Dictionary of {model_name: results_dict}
            metric: Metric to plot (default: 'accuracy')
            title: Optional custom title
            filename: Output filename (without extension)
        """
        self.logger.info(f"Creating SOTA comparison bar chart for {metric}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_names = []
        means = []
        stds = []
        colors = []
        
        for model_name, result in results.items():
            model_names.append(model_name.replace('_', ' '))
            
            if 'aggregated' in result:
                means.append(result['aggregated'][metric]['mean'] * 100)
                stds.append(result['aggregated'][metric]['std'] * 100)
            else:
                means.append(result[metric]['mean'] * 100)
                stds.append(result[metric]['std'] * 100)
            
            colors.append(self.colors.get(model_name, '#95a5a6'))
        
        x_pos = np.arange(len(model_names))
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Model', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)', 
                      fontsize=14, fontweight='bold')
        
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        else:
            ax.set_title(f'SOTA Model Comparison - {metric.replace("_", " ").title()}',
                        fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.5, f'{mean:.2f}%', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save in multiple formats
        for ext in ['png', 'pdf']:
            filepath = os.path.join(self.output_dir, f'{filename}.{ext}')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_radar_chart(
        self,
        results: Dict[str, Dict],
        metrics: List[str] = None,
        title: str = 'Multi-Metric Model Comparison',
        filename: str = 'radar_chart'
    ):
        """
        Figure 2: Radar/spider chart for multi-metric comparison.
        
        Args:
            results: Dictionary of {model_name: results_dict}
            metrics: List of metrics to include
            title: Chart title
            filename: Output filename
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        self.logger.info(f"Creating radar chart with metrics: {metrics}")
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Number of variables
        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each model
        for model_name, result in results.items():
            values = []
            for metric in metrics:
                if 'aggregated' in result:
                    val = result['aggregated'][metric]['mean']
                else:
                    val = result[metric]['mean']
                values.append(val * 100)  # Convert to percentage
            
            values += values[:1]  # Complete the circle
            
            color = self.colors.get(model_name, '#95a5a6')
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name.replace('_', ' '), 
                   color=color, markersize=6)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        
        plt.tight_layout()
        
        for ext in ['png', 'pdf']:
            filepath = os.path.join(self.output_dir, f'{filename}.{ext}')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_ablation_heatmap(
        self,
        ablation_results: List[Dict],
        metrics: List[str] = None,
        title: str = 'Ablation Study Results',
        filename: str = 'ablation_heatmap'
    ):
        """
        Figure 3: Heatmap showing ablation study results.
        
        Args:
            ablation_results: List of ablation result dictionaries
            metrics: List of metrics to display
            title: Chart title
            filename: Output filename
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        self.logger.info("Creating ablation study heatmap")
        
        # Prepare data matrix
        config_names = [result['name'].replace('_', ' ').title() 
                       for result in ablation_results]
        
        data_matrix = []
        for result in ablation_results:
            row = []
            for metric in metrics:
                if metric in result['results']:
                    val = result['results'][metric]['mean'] * 100
                    row.append(val)
                else:
                    row.append(0)
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, len(config_names) * 0.5 + 2))
        
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=80, vmax=100)
        
        # Set ticks
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(config_names)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], 
                          fontsize=11, rotation=45, ha='right')
        ax.set_yticklabels(config_names, fontsize=11)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Score (%)', rotation=270, labelpad=20, fontsize=12)
        
        # Add text annotations
        for i in range(len(config_names)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.1f}',
                             ha='center', va='center', color='black', fontsize=9)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        for ext in ['png', 'pdf']:
            filepath = os.path.join(self.output_dir, f'{filename}.{ext}')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_training_curves(
        self,
        histories: Dict[str, Any],
        title: str = 'Training and Validation Curves',
        filename: str = 'training_curves'
    ):
        """
        Figure 4: Training/validation accuracy and loss curves.
        
        Args:
            histories: Dictionary of {model_name: history_dict}
            title: Chart title
            filename: Output filename
        """
        self.logger.info("Creating training curves")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        for model_name, history in histories.items():
            color = self.colors.get(model_name, '#95a5a6')
            label = model_name.replace('_', ' ')
            
            # Plot accuracy
            if 'accuracy' in history:
                epochs = range(1, len(history['accuracy']) + 1)
                ax1.plot(epochs, history['accuracy'], '-', 
                        label=f'{label} (Train)', color=color, linewidth=2)
                if 'val_accuracy' in history:
                    ax1.plot(epochs, history['val_accuracy'], '--',
                           label=f'{label} (Val)', color=color, linewidth=2, alpha=0.7)
            
            # Plot loss
            if 'loss' in history:
                epochs = range(1, len(history['loss']) + 1)
                ax2.plot(epochs, history['loss'], '-',
                        label=f'{label} (Train)', color=color, linewidth=2)
                if 'val_loss' in history:
                    ax2.plot(epochs, history['val_loss'], '--',
                           label=f'{label} (Val)', color=color, linewidth=2, alpha=0.7)
        
        # Configure accuracy plot
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Configure loss plot
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        for ext in ['png', 'pdf']:
            filepath = os.path.join(self.output_dir, f'{filename}.{ext}')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_roc_curves(
        self,
        roc_data: Dict[str, Dict],
        title: str = 'ROC Curves Comparison',
        filename: str = 'roc_curves'
    ):
        """
        Figure 5: ROC curves with AUC values for all models.
        
        Args:
            roc_data: Dictionary of {model_name: {'fpr': fpr, 'tpr': tpr, 'auc': auc}}
            title: Chart title
            filename: Output filename
        """
        self.logger.info("Creating ROC curves")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.5)
        
        # Plot ROC curve for each model
        for model_name, data in roc_data.items():
            color = self.colors.get(model_name, '#95a5a6')
            label = f"{model_name.replace('_', ' ')} (AUC = {data['auc']:.3f})"
            
            ax.plot(data['fpr'], data['tpr'], color=color, 
                   linewidth=2.5, label=label)
        
        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        
        for ext in ['png', 'pdf']:
            filepath = os.path.join(self.output_dir, f'{filename}.{ext}')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_confusion_matrices(
        self,
        cm_data: Dict[str, np.ndarray],
        class_names: List[str] = None,
        title: str = 'Confusion Matrices',
        filename: str = 'confusion_matrices'
    ):
        """
        Figure 6: Side-by-side confusion matrices.
        
        Args:
            cm_data: Dictionary of {model_name: confusion_matrix}
            class_names: List of class names
            title: Chart title
            filename: Output filename
        """
        if class_names is None:
            class_names = ['No Tumor', 'Tumor']
        
        self.logger.info("Creating confusion matrices")
        
        n_models = len(cm_data)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else axes
        
        for idx, (model_name, cm) in enumerate(cm_data.items()):
            ax = axes[idx] if n_models > 1 else axes[0]
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot heatmap
            im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names, fontsize=10)
            ax.set_yticklabels(class_names, fontsize=10)
            
            # Add text annotations
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    text = ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2%})',
                                 ha='center', va='center', 
                                 color='white' if cm_normalized[i, j] > 0.5 else 'black',
                                 fontsize=9)
            
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
            ax.set_title(model_name.replace('_', ' '), fontsize=12, fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        for ext in ['png', 'pdf']:
            filepath = os.path.join(self.output_dir, f'{filename}.{ext}')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_metric_comparison_table(
        self,
        results: Dict[str, Dict],
        metrics: List[str] = None,
        title: str = 'Model Performance Metrics',
        filename: str = 'metrics_table'
    ):
        """
        Create a visual table of all metrics for all models.
        
        Args:
            results: Dictionary of {model_name: results_dict}
            metrics: List of metrics to include
            title: Chart title
            filename: Output filename
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 
                      'sensitivity', 'specificity']
        
        self.logger.info("Creating metrics comparison table")
        
        # Prepare data
        model_names = list(results.keys())
        data_matrix = []
        
        for model_name in model_names:
            row = []
            result = results[model_name]
            for metric in metrics:
                if 'aggregated' in result and metric in result['aggregated']:
                    mean = result['aggregated'][metric]['mean']
                    std = result['aggregated'][metric]['std']
                    row.append(f"{mean*100:.2f} ± {std*100:.2f}")
                elif metric in result:
                    mean = result[metric]['mean']
                    std = result[metric]['std']
                    row.append(f"{mean*100:.2f} ± {std*100:.2f}")
                else:
                    row.append("N/A")
            data_matrix.append(row)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, len(model_names) * 0.6 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=data_matrix,
            rowLabels=[m.replace('_', ' ') for m in model_names],
            colLabels=[m.replace('_', ' ').title() for m in metrics],
            cellLoc='center',
            loc='center',
            colWidths=[0.12] * len(metrics)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(metrics)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style rows
        for i in range(len(model_names)):
            table[(i+1, -1)].set_facecolor('#ecf0f1')
            table[(i+1, -1)].set_text_props(weight='bold')
            
            # Alternate row colors
            if i % 2 == 0:
                for j in range(len(metrics)):
                    table[(i+1, j)].set_facecolor('#f8f9fa')
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        for ext in ['png', 'pdf']:
            filepath = os.path.join(self.output_dir, f'{filename}.{ext}')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_gradcam_examples(
        self,
        model,
        images: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        layer_name: str = None,
        n_examples: int = 6,
        title: str = 'GradCAM Visualizations',
        filename: str = 'gradcam_examples'
    ):
        """
        Figure 7: GradCAM visualization showing model attention.
        
        Note: This is a placeholder for GradCAM visualization.
        Full implementation requires model-specific layer selection.
        
        Args:
            model: Keras model
            images: Input images
            labels: True labels
            predictions: Predicted labels
            layer_name: Name of layer for GradCAM
            n_examples: Number of examples to show
            title: Chart title
            filename: Output filename
        """
        self.logger.info("GradCAM visualization placeholder")
        self.logger.info("Full GradCAM implementation requires model-specific configuration")
        
        # Create placeholder figure
        fig, axes = plt.subplots(2, n_examples//2, figsize=(15, 6))
        axes = axes.flatten()
        
        for i in range(min(n_examples, len(images))):
            axes[i].imshow(images[i])
            axes[i].axis('off')
            axes[i].set_title(f'True: {labels[i]}, Pred: {predictions[i]}', 
                            fontsize=10)
        
        plt.suptitle(f'{title}\n(Placeholder - Full implementation pending)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        for ext in ['png', 'pdf']:
            filepath = os.path.join(self.output_dir, f'{filename}.{ext}')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved: {filepath}")
        
        plt.close()
    
    def create_figure_summary(self):
        """Create a summary document of all generated figures."""
        summary_file = os.path.join(self.output_dir, 'FIGURES_SUMMARY.md')
        
        figures = [f for f in os.listdir(self.output_dir) if f.endswith('.png')]
        
        with open(summary_file, 'w') as f:
            f.write("# Research Figures Summary\n\n")
            f.write(f"Generated: {os.listdir(self.output_dir)}\n\n")
            f.write(f"Total figures: {len(figures)}\n\n")
            
            f.write("## Available Figures\n\n")
            for fig in sorted(figures):
                f.write(f"- `{fig}`\n")
        
        self.logger.info(f"Figure summary saved to: {summary_file}")
