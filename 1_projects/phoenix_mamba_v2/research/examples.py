"""
Example Usage Script
Demonstrates how to use the brain tumor detection system
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def example_1_train_model():
    """Example 1: Train a model from scratch"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Training a Model")
    print("="*80 + "\n")
    
    print("Code:")
    print("""
    from models.cnn_model import create_cnn_model, compile_model, create_callbacks
    from src.data_preprocessing import create_data_generators
    
    # Create model
    model = create_cnn_model()
    model = compile_model(model, learning_rate=0.0001)
    
    # Load data
    train_gen, val_gen, test_gen = create_data_generators()
    
    # Train
    history = model.fit(
        train_gen,
        epochs=50,
        validation_data=val_gen,
        callbacks=create_callbacks()
    )
    """)
    
    print("\nTo run: python src/train.py")


def example_2_make_prediction():
    """Example 2: Make a prediction on a single image"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Making Predictions")
    print("="*80 + "\n")
    
    print("Code:")
    print("""
    from src.predict import load_model, predict_single_image
    
    # Load trained model
    model = load_model()
    
    # Predict on a single image
    result = predict_single_image(
        model, 
        'path/to/brain_mri.jpg',
        display=True
    )
    
    # Print results
    print(f"Prediction: {result['class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    """)
    
    print("\nTo run: python src/predict.py")


def example_3_batch_prediction():
    """Example 3: Batch prediction on multiple images"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Batch Predictions")
    print("="*80 + "\n")
    
    print("Code:")
    print("""
    from src.predict import load_model, predict_batch
    
    # Load model
    model = load_model()
    
    # Predict on all images in a directory
    results = predict_batch(
        model,
        'path/to/images/directory',
        save_results=True
    )
    
    # Process results
    for result in results:
        print(f"{result['filename']}: {result['class']} ({result['confidence']:.2%})")
    """)


def example_4_evaluate_model():
    """Example 4: Evaluate model performance"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Model Evaluation")
    print("="*80 + "\n")
    
    print("Code:")
    print("""
    from src.evaluate import load_trained_model, evaluate_with_generator
    from src.data_preprocessing import create_data_generators
    
    # Load model
    model = load_trained_model()
    
    # Get test data
    _, _, test_gen = create_data_generators()
    
    # Evaluate
    results = evaluate_with_generator(model, test_gen)
    
    # Results include:
    # - Accuracy, Precision, Recall, F1-Score
    # - Confusion Matrix
    # - ROC Curve
    # - Classification Report
    """)
    
    print("\nTo run: python src/evaluate.py")


def example_5_custom_training():
    """Example 5: Custom training with your own parameters"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Training Configuration")
    print("="*80 + "\n")
    
    print("Code:")
    print("""
    # Modify config.py:
    
    IMG_HEIGHT = 256  # Change image size
    IMG_WIDTH = 256
    BATCH_SIZE = 16   # Smaller batch size
    EPOCHS = 100      # More epochs
    LEARNING_RATE = 0.00001  # Lower learning rate
    
    # Then run training normally:
    python src/train.py
    """)


def example_6_visualization():
    """Example 6: Create visualizations"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Visualizations")
    print("="*80 + "\n")
    
    print("Code:")
    print("""
    from src.visualize import (
        plot_sample_images,
        plot_class_distribution,
        visualize_predictions_grid
    )
    
    # Plot sample images from dataset
    plot_sample_images('data/train', 'tumor', num_samples=5)
    
    # Plot class distribution
    plot_class_distribution(labels, ['no_tumor', 'tumor'])
    
    # Visualize predictions
    visualize_predictions_grid(
        model, 
        test_images, 
        test_labels, 
        ['no_tumor', 'tumor'],
        num_samples=9
    )
    """)


def example_7_data_preparation():
    """Example 7: Prepare your dataset"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Dataset Preparation")
    print("="*80 + "\n")
    
    print("Steps:")
    print("""
    1. Create directory structure:
       python setup_data.py --create
    
    2. Organize your images:
       data/
         train/
           tumor/      <- Place tumor MRI images here
           no_tumor/   <- Place non-tumor MRI images here
         validation/
           tumor/
           no_tumor/
         test/
           tumor/
           no_tumor/
    
    3. Verify setup:
       python setup_data.py --check
    
    4. Count images:
       python setup_data.py --count
    """)


def main():
    """Run all examples"""
    print("\n" + "#"*80)
    print("#" + " "*28 + "USAGE EXAMPLES" + " "*37 + "#")
    print("#"*80)
    
    examples = [
        example_1_train_model,
        example_2_make_prediction,
        example_3_batch_prediction,
        example_4_evaluate_model,
        example_5_custom_training,
        example_6_visualization,
        example_7_data_preparation
    ]
    
    for example in examples:
        example()
        print()
    
    print("\n" + "="*80)
    print("For more information, see:")
    print("  - README.md: Complete documentation")
    print("  - QUICKSTART.md: Quick start guide")
    print("  - Research_Paper_Brain_Tumor_Detection.md: Technical details")
    print("  - CONTRIBUTING.md: Contribution guidelines")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
