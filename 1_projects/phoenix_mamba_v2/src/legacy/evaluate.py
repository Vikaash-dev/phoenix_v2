"""
Model Evaluation Script
Evaluates the trained model on test set and generates performance metrics
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score
)
import tensorflow as tf
from tensorflow import keras

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.legacy.data_preprocessing import create_data_generators, load_dataset_from_directory, split_dataset


def load_trained_model(model_path=config.MODEL_PATH):
    """
    Load trained model from file
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        print("Please train the model first using train.py")
        return None
    
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    return model


def evaluate_with_generator(model, test_generator):
    """
    Evaluate model using data generator
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print("\n" + "="*80)
    print("EVALUATING MODEL ON TEST SET")
    print("="*80 + "\n")
    
    # Get predictions
    print("Generating predictions...")
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    
    # Get true labels
    y_true = test_generator.classes
    y_pred = np.argmax(predictions, axis=1)
    y_pred_proba = predictions[:, 1]  # Probability for tumor class
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Classification report
    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT")
    print("-"*80)
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES)
    print(report)
    
    # Save classification report
    report_path = config.CLASSIFICATION_REPORT_PATH
    with open(report_path, 'w') as f:
        f.write("BRAIN TUMOR DETECTION - CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n\n")
        f.write("Per-Class Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(report)
    
    print(f"\nClassification report saved to {report_path}")
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def plot_confusion_matrix(y_true, y_pred, save_path=config.CONFUSION_MATRIX_PATH):
    """
    Plot and save confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    print("\nGenerating confusion matrix...")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Brain Tumor Detection', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    
    plt.close()
    
    # Print confusion matrix details
    print("\nConfusion Matrix:")
    print(cm)
    print("\nInterpretation:")
    print(f"True Negatives (TN):  {cm[0, 0]} - Correctly identified no tumor")
    print(f"False Positives (FP): {cm[0, 1]} - Incorrectly identified as tumor")
    print(f"False Negatives (FN): {cm[1, 0]} - Missed tumor cases")
    print(f"True Positives (TP):  {cm[1, 1]} - Correctly identified tumor")


def plot_roc_curve(y_true, y_pred_proba, save_path=config.ROC_CURVE_PATH):
    """
    Plot and save ROC curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    print("\nGenerating ROC curve...")
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Brain Tumor Detection', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")
    
    plt.close()
    
    print(f"\nROC AUC Score: {roc_auc:.4f}")


def plot_training_history(history_path=None):
    """
    Plot training history from saved file
    
    Args:
        history_path: Path to saved training history
    """
    if history_path is None:
        history_path = os.path.join(config.RESULTS_DIR, 'training_history.npy')
    
    if not os.path.exists(history_path):
        print(f"\nTraining history not found at {history_path}")
        return
    
    print("\nLoading training history...")
    history = np.load(history_path, allow_pickle=True).item()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot accuracy
    axes[0, 0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot loss
    axes[0, 1].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[0, 1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot precision
    if 'precision' in history:
        axes[1, 0].plot(history['precision'], label='Training Precision', linewidth=2)
        axes[1, 0].plot(history['val_precision'], label='Validation Precision', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
    
    # Plot recall
    if 'recall' in history:
        axes[1, 1].plot(history['recall'], label='Training Recall', linewidth=2)
        axes[1, 1].plot(history['val_recall'], label='Validation Recall', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('Training History - Brain Tumor Detection', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    save_path = config.TRAINING_HISTORY_PATH
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    
    plt.close()


def main():
    """Main evaluation function"""
    
    # Load model
    model = load_trained_model()
    
    if model is None:
        return
    
    # Create test generator
    print("\nCreating test data generator...")
    try:
        _, _, test_gen = create_data_generators()
        print(f"Test samples: {test_gen.samples}")
    except Exception as e:
        print(f"Error creating test generator: {e}")
        return
    
    # Evaluate model
    results = evaluate_with_generator(model, test_gen)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_confusion_matrix(results['y_true'], results['y_pred'])
    plot_roc_curve(results['y_true'], results['y_pred_proba'])
    plot_training_history()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to {config.RESULTS_DIR}")
    print("\nGenerated files:")
    print(f"  - {config.CONFUSION_MATRIX_PATH}")
    print(f"  - {config.ROC_CURVE_PATH}")
    print(f"  - {config.TRAINING_HISTORY_PATH}")
    print(f"  - {config.CLASSIFICATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
