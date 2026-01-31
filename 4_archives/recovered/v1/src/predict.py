"""
Prediction Script for Brain Tumor Detection
Make predictions on new brain MRI images
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_preprocessing import preprocess_image_for_prediction


def load_model(model_path=config.MODEL_PATH):
    """
    Load trained model
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using train.py")
        return None
    
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    return model


def predict_single_image(model, image_path, display=True):
    """
    Predict tumor presence in a single brain MRI image
    
    Args:
        model: Trained Keras model
        image_path: Path to the image file
        display: Whether to display the image and prediction
        
    Returns:
        Dictionary with prediction results
    """
    # Preprocess image
    img = preprocess_image_for_prediction(image_path)
    
    # Make prediction
    prediction = model.predict(img, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    class_name = config.CLASS_NAMES[predicted_class]
    
    # Prepare results
    results = {
        'class': class_name,
        'confidence': float(confidence),
        'probabilities': {
            config.CLASS_NAMES[0]: float(prediction[0][0]),
            config.CLASS_NAMES[1]: float(prediction[0][1])
        }
    }
    
    # Display results
    if display:
        display_prediction(image_path, results)
    
    return results


def display_prediction(image_path, results):
    """
    Display image with prediction results
    
    Args:
        image_path: Path to the image
        results: Prediction results dictionary
    """
    # Load and display image
    img = Image.open(image_path)
    img_rgb = img.convert('RGB')
    
    plt.figure(figsize=(12, 5))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title('Brain MRI Scan', fontsize=14, fontweight='bold')
    
    # Display prediction
    plt.subplot(1, 2, 2)
    classes = list(results['probabilities'].keys())
    probabilities = list(results['probabilities'].values())
    
    colors = ['green' if c == results['class'] else 'gray' for c in classes]
    bars = plt.barh(classes, probabilities, color=colors)
    
    plt.xlabel('Probability', fontsize=12)
    plt.title('Prediction Results', fontsize=14, fontweight='bold')
    plt.xlim([0, 1])
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        plt.text(prob + 0.02, i, f'{prob*100:.2f}%', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Predicted Class: {results['class'].upper()}")
    print(f"Confidence: {results['confidence']*100:.2f}%")
    print("\nClass Probabilities:")
    for class_name, prob in results['probabilities'].items():
        print(f"  {class_name}: {prob*100:.2f}%")
    print("="*80 + "\n")


def predict_batch(model, image_dir, save_results=True):
    """
    Predict on a batch of images
    
    Args:
        model: Trained model
        image_dir: Directory containing images
        save_results: Whether to save results to file
        
    Returns:
        List of prediction results
    """
    print(f"\nProcessing images from {image_dir}...")
    
    results_list = []
    
    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        
        img_path = os.path.join(image_dir, img_name)
        
        try:
            result = predict_single_image(model, img_path, display=False)
            result['filename'] = img_name
            results_list.append(result)
            
            print(f"{img_name}: {result['class']} ({result['confidence']*100:.2f}%)")
        
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
    
    # Save results
    if save_results and results_list:
        save_path = os.path.join(config.RESULTS_DIR, 'batch_predictions.txt')
        with open(save_path, 'w') as f:
            f.write("BRAIN TUMOR DETECTION - BATCH PREDICTIONS\n")
            f.write("="*80 + "\n\n")
            
            for result in results_list:
                f.write(f"Image: {result['filename']}\n")
                f.write(f"Prediction: {result['class']}\n")
                f.write(f"Confidence: {result['confidence']*100:.2f}%\n")
                f.write(f"Probabilities: {result['probabilities']}\n")
                f.write("-"*80 + "\n")
        
        print(f"\nBatch results saved to {save_path}")
    
    return results_list


def predict_with_visualization(model, image_path, save_path=None):
    """
    Predict and create detailed visualization
    
    Args:
        model: Trained model
        image_path: Path to image
        save_path: Path to save visualization
    """
    # Load image
    original_img = Image.open(image_path).convert('RGB')
    
    # Preprocess for prediction
    preprocessed_img = preprocess_image_for_prediction(image_path)
    
    # Make prediction
    prediction = model.predict(preprocessed_img, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    class_name = config.CLASS_NAMES[predicted_class]
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].axis('off')
    axes[0].set_title('Original MRI Scan', fontsize=14, fontweight='bold')
    
    # Preprocessed image
    axes[1].imshow(preprocessed_img[0])
    axes[1].axis('off')
    axes[1].set_title('Preprocessed Image', fontsize=14, fontweight='bold')
    
    # Prediction bars
    classes = config.CLASS_NAMES
    probabilities = prediction[0]
    colors = ['#FF6B6B' if classes[i] == 'tumor' else '#4ECDC4' for i in range(len(classes))]
    
    bars = axes[2].barh(classes, probabilities, color=colors, alpha=0.8)
    axes[2].set_xlabel('Probability', fontsize=12)
    axes[2].set_title('Prediction Confidence', fontsize=14, fontweight='bold')
    axes[2].set_xlim([0, 1])
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        axes[2].text(prob + 0.02, i, f'{prob*100:.1f}%', 
                    va='center', fontsize=11, fontweight='bold')
    
    # Overall title with prediction
    fig.suptitle(f'Prediction: {class_name.upper()} (Confidence: {confidence*100:.2f}%)',
                fontsize=16, fontweight='bold', 
                color='red' if class_name == 'tumor' else 'green')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def main():
    """Main prediction function"""
    
    print("\n" + "="*80)
    print("BRAIN TUMOR DETECTION - PREDICTION SCRIPT")
    print("="*80 + "\n")
    
    # Load model
    model = load_model()
    
    if model is None:
        return
    
    # Example usage
    print("\nUsage Examples:")
    print("-"*80)
    print("\n1. Single Image Prediction:")
    print("   results = predict_single_image(model, 'path/to/image.jpg')")
    print("\n2. Batch Prediction:")
    print("   results = predict_batch(model, 'path/to/image/directory')")
    print("\n3. Prediction with Visualization:")
    print("   predict_with_visualization(model, 'path/to/image.jpg', 'output.png')")
    print("\n" + "-"*80)
    
    # Interactive mode
    print("\nInteractive Prediction Mode")
    print("Enter image path to predict (or 'quit' to exit):")
    
    while True:
        user_input = input("\nImage path: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not os.path.exists(user_input):
            print(f"File not found: {user_input}")
            continue
        
        try:
            predict_single_image(model, user_input, display=True)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
