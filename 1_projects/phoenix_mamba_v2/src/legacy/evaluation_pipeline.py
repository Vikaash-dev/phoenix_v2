"""
Standardized evaluation pipeline for brain tumor detection models.
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import from local modules
try:
    from src.legacy.train_neurosnake_vit import create_data_generators_with_physics_augmentation
    import config
except ImportError:
    # Fallback: Add parent directory to path if imports fail
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.legacy.train_neurosnake_vit import create_data_generators_with_physics_augmentation
    import config


def evaluate_model(
    model_path,
    data_dir,
    output_dir,
    batch_size=32,
):
    """
    Evaluate a trained brain tumor detection model.
    """
    print("=" * 80)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"1. Loading model from: {model_path}")
    model = keras.models.load_model(model_path)

    # Create data generator
    print("2. Creating data generator...")
    _, _, test_ds = create_data_generators_with_physics_augmentation(
        train_dir=os.path.join(data_dir, 'train'),
        val_dir=os.path.join(data_dir, 'validation'),
        test_dir=os.path.join(data_dir, 'test'),
        batch_size=batch_size,
        use_physics_augmentation=False,
    )

    # Get true labels and predictions
    print("3. Generating predictions...")
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred = model.predict(test_ds)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)

    # Generate classification report
    print("4. Generating classification report...")
    report = classification_report(y_true_classes, y_pred_classes, target_names=['no_tumor', 'tumor'], output_dict=True)
    report_path = os.path.join(output_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"   - Classification report saved to: {report_path}")

    # Calculate ROC-AUC score
    roc_auc = roc_auc_score(y_true, y_pred)
    print(f"   - ROC-AUC Score: {roc_auc:.4f}")


    # Generate confusion matrix
    print("5. Generating confusion matrix...")
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"   - Confusion matrix saved to: {cm_path}")

    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Model Evaluation Pipeline Script"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model file (.h5)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Base data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/evaluation",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )

    args = parser.parse_args()

    # Evaluate model
    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
