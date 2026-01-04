"""
Visualization Utilities for Brain Tumor Detection
Helper functions for creating visualizations and plots
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
import tensorflow as tf


def plot_sample_images(image_dir, class_name, num_samples=5, save_path=None):
    """
    Plot sample images from a class
    
    Args:
        image_dir: Directory containing class images
        class_name: Name of the class
        num_samples: Number of samples to display
        save_path: Path to save the plot
    """
    images = []
    class_dir = os.path.join(image_dir, class_name)
    
    if not os.path.exists(class_dir):
        print(f"Directory not found: {class_dir}")
        return
    
    # Get image files
    image_files = [f for f in os.listdir(class_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # Select random samples
    if len(image_files) > num_samples:
        image_files = np.random.choice(image_files, num_samples, replace=False)
    elif len(image_files) < num_samples:
        num_samples = len(image_files)  # Adjust to available samples
    
    # Load images
    for img_file in image_files[:num_samples]:
        img_path = os.path.join(class_dir, img_file)
        img = plt.imread(img_path)
        images.append(img)
    
    # Plot
    fig, axes = plt.subplots(1, len(images), figsize=(4*len(images), 4))
    
    if len(images) == 1:
        axes = [axes]
    
    for idx, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax.axis('off')
        ax.set_title(f'Sample {idx+1}')
    
    plt.suptitle(f'Sample Images - {class_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample images saved to {save_path}")
    
    plt.show()


def plot_class_distribution(labels, class_names, save_path=None):
    """
    Plot class distribution
    
    Args:
        labels: Array of labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar([class_names[i] for i in unique], counts, 
                   color=['#4ECDC4', '#FF6B6B'], alpha=0.8)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Class Distribution', fontsize=16, fontweight='bold')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    plt.show()


def plot_model_comparison(models_data, save_path=None):
    """
    Plot comparison of multiple models
    
    Args:
        models_data: List of dictionaries with model info
                    [{'name': 'Model1', 'accuracy': 0.95, 'f1': 0.94}, ...]
        save_path: Path to save the plot
    """
    model_names = [m['name'] for m in models_data]
    accuracies = [m['accuracy'] for m in models_data]
    
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(model_names, accuracies, color='steelblue', alpha=0.8)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.ylim([0, 1])
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.2f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
    
    plt.show()


def visualize_predictions_grid(model, images, true_labels, class_names, 
                               num_samples=9, save_path=None):
    """
    Visualize predictions in a grid
    
    Args:
        model: Trained model
        images: Array of images
        true_labels: True labels
        class_names: List of class names
        num_samples: Number of samples to display
        save_path: Path to save the plot
    """
    # Select random samples
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    # Make predictions
    predictions = model.predict(images[indices], verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels[indices], axis=1)
    
    # Create grid
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()
    
    for idx, (ax, img_idx) in enumerate(zip(axes, indices)):
        ax.imshow(images[img_idx])
        
        pred_class = pred_classes[idx]
        true_class = true_classes[idx]
        confidence = predictions[idx][pred_class]
        
        # Color: green if correct, red if wrong
        color = 'green' if pred_class == true_class else 'red'
        
        title = f'True: {class_names[true_class]}\n'
        title += f'Pred: {class_names[pred_class]} ({confidence*100:.1f}%)'
        
        ax.set_title(title, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Prediction Examples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions grid saved to {save_path}")
    
    plt.show()


def plot_feature_maps(model, image, layer_name, save_path=None):
    """
    Visualize feature maps from a convolutional layer
    
    Args:
        model: Trained model
        image: Input image
        layer_name: Name of layer to visualize
        save_path: Path to save the plot
    """
    # Create a model that outputs the feature maps
    layer_output = model.get_layer(layer_name).output
    feature_model = tf.keras.Model(inputs=model.input, outputs=layer_output)
    
    # Get feature maps
    feature_maps = feature_model.predict(np.expand_dims(image, axis=0), verbose=0)
    
    # Plot
    num_features = min(feature_maps.shape[-1], 16)  # Show max 16 feature maps
    rows = 4
    cols = 4
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flatten()):
        if i < num_features:
            ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
            ax.set_title(f'Feature {i}')
        ax.axis('off')
    
    plt.suptitle(f'Feature Maps - Layer: {layer_name}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature maps saved to {save_path}")
    
    plt.show()


def create_summary_report(metrics, save_path=None):
    """
    Create a visual summary report
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save the report
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Metrics bar chart
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [metrics['accuracy'], metrics['precision'], 
                    metrics['recall'], metrics['f1_score']]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = axes[0, 0].bar(metric_names, metric_values, color=colors, alpha=0.8)
    axes[0, 0].set_ylabel('Score', fontsize=12)
    axes[0, 0].set_title('Performance Metrics', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim([0, 1])
    
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontweight='bold')
    
    # Confusion matrix
    cm = metrics.get('confusion_matrix', np.array([[0, 0], [0, 0]]))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'])
    axes[0, 1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('True Label')
    axes[0, 1].set_xlabel('Predicted Label')
    
    # Training history
    if 'history' in metrics:
        history = metrics['history']
        axes[1, 0].plot(history['accuracy'], label='Training', linewidth=2)
        axes[1, 0].plot(history['val_accuracy'], label='Validation', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Training History', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
    
    # ROC curve
    if 'fpr' in metrics and 'tpr' in metrics:
        fpr, tpr = metrics['fpr'], metrics['tpr']
        auc_score = metrics.get('auc', 0)
        axes[1, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.3f}')
        axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('Brain Tumor Detection - Summary Report', 
                fontsize=18, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary report saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Visualization utilities module loaded successfully!")
    print("\nAvailable functions:")
    print("  - plot_sample_images()")
    print("  - plot_class_distribution()")
    print("  - plot_model_comparison()")
    print("  - visualize_predictions_grid()")
    print("  - plot_feature_maps()")
    print("  - create_summary_report()")
