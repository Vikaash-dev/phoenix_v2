"""
Data Setup Helper Script
Helps you prepare your dataset for training
"""

import os
import shutil
from pathlib import Path
import config


def create_directory_structure():
    """Create the required directory structure for the project"""
    directories = [
        config.TRAIN_DIR,
        config.VAL_DIR,
        config.TEST_DIR,
        os.path.join(config.TRAIN_DIR, 'tumor'),
        os.path.join(config.TRAIN_DIR, 'no_tumor'),
        os.path.join(config.VAL_DIR, 'tumor'),
        os.path.join(config.VAL_DIR, 'no_tumor'),
        os.path.join(config.TEST_DIR, 'tumor'),
        os.path.join(config.TEST_DIR, 'no_tumor'),
        config.MODEL_DIR,
        config.RESULTS_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print("\n" + "="*80)
    print("Directory structure created successfully!")
    print("="*80)


def check_dataset_structure():
    """Check if the dataset is properly structured"""
    print("\n" + "="*80)
    print("CHECKING DATASET STRUCTURE")
    print("="*80 + "\n")
    
    required_dirs = [
        (config.TRAIN_DIR, 'tumor'),
        (config.TRAIN_DIR, 'no_tumor'),
        (config.VAL_DIR, 'tumor'),
        (config.VAL_DIR, 'no_tumor'),
        (config.TEST_DIR, 'tumor'),
        (config.TEST_DIR, 'no_tumor')
    ]
    
    all_good = True
    
    for base_dir, class_name in required_dirs:
        class_dir = os.path.join(base_dir, class_name)
        
        if os.path.exists(class_dir):
            # Count images
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            count = len(image_files)
            
            status = "✓" if count > 0 else "✗"
            print(f"{status} {class_dir}: {count} images")
            
            if count == 0:
                all_good = False
        else:
            print(f"✗ {class_dir}: Directory not found")
            all_good = False
    
    print("\n" + "-"*80)
    if all_good:
        print("✓ Dataset structure is correct and contains images!")
        print("\nYou're ready to train. Run: python src/train.py")
    else:
        print("✗ Dataset structure is incomplete or empty.")
        print("\nPlease add images to the appropriate directories.")
    print("-"*80 + "\n")


def count_images():
    """Count total images in the dataset"""
    total_train = 0
    total_val = 0
    total_test = 0
    
    print("\n" + "="*80)
    print("IMAGE COUNT SUMMARY")
    print("="*80 + "\n")
    
    # Training set
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.TRAIN_DIR, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            total_train += count
            print(f"Training - {class_name}: {count} images")
    
    print(f"Training Total: {total_train} images\n")
    
    # Validation set
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.VAL_DIR, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            total_val += count
            print(f"Validation - {class_name}: {count} images")
    
    print(f"Validation Total: {total_val} images\n")
    
    # Test set
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.TEST_DIR, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            total_test += count
            print(f"Test - {class_name}: {count} images")
    
    print(f"Test Total: {total_test} images\n")
    
    print("="*80)
    print(f"GRAND TOTAL: {total_train + total_val + total_test} images")
    print("="*80 + "\n")


def print_instructions():
    """Print instructions for setting up the dataset"""
    print("\n" + "="*80)
    print("BRAIN TUMOR DETECTION - DATASET SETUP INSTRUCTIONS")
    print("="*80 + "\n")
    
    print("Step 1: Download a brain tumor dataset")
    print("  - Recommended: Br35H Dataset from Kaggle")
    print("  - URL: https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection")
    print()
    
    print("Step 2: Organize images into the following structure:")
    print("  data/")
    print("    train/")
    print("      tumor/        <- Place tumor images here")
    print("      no_tumor/     <- Place no tumor images here")
    print("    validation/")
    print("      tumor/")
    print("      no_tumor/")
    print("    test/")
    print("      tumor/")
    print("      no_tumor/")
    print()
    
    print("Step 3: Verify your setup")
    print("  python setup_data.py --check")
    print()
    
    print("Step 4: Train the model")
    print("  python src/train.py")
    print()
    
    print("="*80 + "\n")


def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--create':
            create_directory_structure()
        elif sys.argv[1] == '--check':
            check_dataset_structure()
            count_images()
        elif sys.argv[1] == '--count':
            count_images()
        else:
            print("Unknown option. Use --create, --check, or --count")
    else:
        print_instructions()
        print("\nOptions:")
        print("  python setup_data.py --create    Create directory structure")
        print("  python setup_data.py --check     Check dataset structure")
        print("  python setup_data.py --count     Count images in dataset")


if __name__ == "__main__":
    main()
