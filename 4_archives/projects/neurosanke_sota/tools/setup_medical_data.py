"""
Medical Data Setup Tool
Downloads and prepares the Br35H / Brain Tumor Classification dataset from Kaggle.
Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

def setup_kaggle_credentials():
    """Ensure Kaggle credentials are set up for the API."""
    # Check environment variables
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    
    if not username or not key:
        print("❌ Error: KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set.")
        print("Please set them in your cloud environment (Deepnote/Camber) secrets.")
        return False
    
    print(f"✅ Kaggle credentials found for user: {username}")
    return True

def download_dataset(dataset_name, output_dir):
    """Download and unzip dataset using Kaggle API."""
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle...")
        os.system("pip install kaggle")
        import kaggle

    print(f"⬇️ Downloading {dataset_name}...")
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
        print("✅ Download complete.")
        return True
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        return False

def organize_dataset(source_dir):
    """
    Organize the dataset into standard train/val/test structure if not already.
    The 'sartajbhuvaji/brain-tumor-classification-mri' dataset usually comes as Training/Testing folders.
    """
    source_path = Path(source_dir)
    
    # Check structure
    if (source_path / "Training").exists() and (source_path / "Testing").exists():
        print("📦 Dataset structure detected: Training/Testing splits.")
        # Create standard structure
        base_dir = source_path / "standardized"
        if base_dir.exists():
            print("Standardized directory already exists. Skipping reorganization.")
            return str(base_dir)
            
        base_dir.mkdir(parents=True)
        
        # We need a Validation split. Let's take it from Training.
        # Structure: data/standardized/{train,validation,test}/{class_name}/images
        
        print("Reorganizing into Train/Validation/Test...")
        
        # Move Testing -> Test
        shutil.copytree(source_path / "Testing", base_dir / "test", dirs_exist_ok=True)
        
        # Process Training
        # We'll rely on our data_deduplication and split logic elsewhere? 
        # Or just do a simple split here.
        
        import random
        
        for class_dir in (source_path / "Training").iterdir():
            if not class_dir.is_dir(): continue
            
            class_name = class_dir.name
            images = list(class_dir.glob("*"))
            random.shuffle(images)
            
            # 80% Train, 20% Val from the original Training set
            split_idx = int(len(images) * 0.8)
            train_imgs = images[:split_idx]
            val_imgs = images[split_idx:]
            
            # Copy to dest
            (base_dir / "train" / class_name).mkdir(parents=True, exist_ok=True)
            (base_dir / "validation" / class_name).mkdir(parents=True, exist_ok=True)
            
            for img in train_imgs:
                shutil.copy2(img, base_dir / "train" / class_name / img.name)
            for img in val_imgs:
                shutil.copy2(img, base_dir / "validation" / class_name / img.name)
                
        print("✅ Reorganization complete.")
        return str(base_dir)
        
    else:
        print("⚠️ Unknown dataset structure. Please check manually.")
        return source_dir

def main():
    parser = argparse.ArgumentParser(description="Download Medical Datasets")
    parser.add_argument("--dataset", type=str, default="sartajbhuvaji/brain-tumor-classification-mri", help="Kaggle dataset ID")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Download location")
    args = parser.parse_args()
    
    if not setup_kaggle_credentials():
        sys.exit(1)
        
    output_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if download_dataset(args.dataset, str(output_dir)):
        final_data_path = organize_dataset(output_dir)
        print(f"\n🎉 Data is ready at: {final_data_path}")
        print("Next: Run 'python src/data_deduplication.py' to clean it.")

if __name__ == "__main__":
    main()
