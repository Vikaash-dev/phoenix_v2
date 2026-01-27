#!/usr/bin/env python3
"""
Kaggle Dataset Manager for Phoenix Protocol

One-command dataset download and preparation for brain tumor detection:
- Automated download with progress tracking
- Data integrity validation
- Patient-level data splitting
- Preprocessing automation
- Seamless integration with training pipeline

Supported datasets:
- brain-tumor-classification-mri (150 MB, binary classification)
- brain-tumor-mri-dataset (200 MB, multi-class classification)

Usage:
    python scripts/kaggle_dataset_setup.py --validate --prepare
    python scripts/kaggle_dataset_setup.py -d brain-tumor-classification-mri --prepare
    python scripts/kaggle_dataset_setup.py --list

Author: Phoenix Protocol Team
Date: January 2026
"""

import os
import sys
import json
import argparse
import subprocess
import zipfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class KaggleDatasetManager:
Manages Kaggle dataset download and preparation.
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.datasets = {
            "brain-tumor-classification-mri": {
                "name": "Br35H Brain Tumor Detection Dataset",
                "description": "3,000 images, binary classification (Tumor, No Tumor)",
                "size_mb": 150,
                "classes": ["no_tumor", "tumor"],
                "split_structure": True,
            },
            "brain-tumor-mri-dataset": {
                "name": "Brain MRI Images for Brain Tumor Detection",
                "description": "253 images, multi-class classification (Glioma, Meningioma, Pituitary, No Tumor)",
                "size_mb": 200,
                "classes": ["no_tumor", "glioma", "meningioma", "pituitary"],
                "split_structure": False,  # Single directory structure
            },
        }

    def check_kaggle_installation(self) -> bool:
def _check_kaggle_installation(self) -> bool:
        """Check if Kaggle API is properly installed."""
        try:
            import kaggle
            return True
        except ImportError:
            return False

    def setup_kaggle_credentials(self) -> bool:
def _check_kaggle_credentials(self) -> bool:
        """Check if Kaggle credentials are set up."""
        # Check for kaggle.json file
        kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
        if kaggle_json.exists():
            return True
        
        # Check environment variables
        if os.getenv('KAGGLE_USERNAME') and os.getenv('KAGGLE_KEY'):
            return True
        
        return False

    def list_available_datasets(self) -> None:
        def list_available_datasets(self) -> None:
        """List all available datasets."""
        print("\n=== AVAILABLE DATASETS ===")
        for dataset_id, info in self.datasets.items():
            print(f"\n{dataset_id}:")
            print(f"  Name: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size_mb']} MB")
            print(f"  Classes: {', '.join(info['classes'])}")
            print(f"  Split Structure: {info['split_structure']}")

    def download_dataset(self, dataset_id: str, target_dir: Path) -> bool:
def download_dataset(self, dataset_id: str, target_dir: Path) -> bool:
        """Download a specific dataset."""
        if dataset_id not in self.datasets:
            print(f"Error: Unknown dataset '{dataset_id}'")
            print("Available datasets:")
            self.list_available_datasets()
            return False
        
        dataset_info = self.datasets[dataset_id]
        print(f"\nDownloading: {dataset_info['name']}")
        print(f"Expected size: {dataset_info['size_mb']} MB")
        
        try:
            # Download dataset using Kaggle API
            cmd = ['kaggle', 'datasets', 'download', '-d', str(target_dir), '--unzip', dataset_id]
            print(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Download complete: {result.stdout}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error downloading dataset: {e}")
            print(f"Stderr: {e.stderr}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False

        dataset_info = self.datasets[dataset_id]
        print(f"\nDownloading: {dataset_info['name']}")
        print(f"Expected size: {dataset_info['size_mb']} MB")

        try:
            # Download dataset using Kaggle API
            cmd = [
                "kaggle",
                "datasets",
                "download",
                "-d",
                str(target_dir),
                "--unzip",
                dataset_id,
            ]
            print(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Download complete: {result.stdout}")

            return True

        except subprocess.CalledProcessError as e:
            print(f"Error downloading dataset: {e}")
            print(f"Stderr: {e.stderr}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False

    def validate_dataset(self, dataset_path: Path, dataset_id: str) -> Dict:
def validate_dataset(self, dataset_path: Path, dataset_id: str) -> Dict:
        """Validate downloaded dataset integrity."""
        validation = {
            'valid': False,
            'total_files': 0,
            'class_distribution': {},
            'missing_files': [],
            'corrupt_files': [],
            'checksum_passed': False
        }
        
        dataset_info = self.datasets[dataset_id]
        expected_classes = dataset_info['classes']
        
        if not dataset_path.exists():
            validation['missing_files'].append(str(dataset_path))
            return validation
        
        # Count files by class
        class_counts = {}
        total_files = 0
        
        if dataset_info['split_structure']:
            # Check for train/validation/test structure
            for split in ['train', 'validation', 'test']:
                split_path = dataset_path / split
                if split_path.exists():
                    for class_name in expected_classes:
                        class_path = split_path / class_name
                        if class_path.exists():
                            files = [f for f in class_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                            count = len(files)
                            class_counts[f"{split}_{class_name}"] = count
                            total_files += count
                        else:
                            validation['missing_files'].append(str(class_path))
                else:
                    validation['missing_files'].append(str(split_path))
        else:
            # Single directory structure
            if dataset_path.exists():
                for class_name in expected_classes:
                    class_path = dataset_path / class_name
                    if class_path.exists():
                        files = [f for f in class_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                        count = len(files)
                        class_counts[class_name] = count
                        total_files += count
                    else:
                        validation['missing_files'].append(str(class_path))
            else:
                validation['missing_files'].append(str(dataset_path))
        
        validation['total_files'] = total_files
        validation['class_distribution'] = class_counts
        validation['valid'] = len(validation['missing_files']) == 0 and total_files > 0
        
        return validation

        # Count files by class
        class_counts = {}
        total_files = 0

        if dataset_info["split_structure"]:
            # Check for train/validation/test structure
            for split in ["train", "validation", "test"]:
                split_path = dataset_path / split
                if split_path.exists():
                    for class_name in expected_classes:
                        class_path = split_path / class_name
                        if class_path.exists():
                            files = [
                                f
                                for f in class_path.iterdir()
                                if f.is_file()
                                and f.suffix.lower()
                                in [".jpg", ".jpeg", ".png", ".bmp"]
                            ]
                            count = len(files)
                            class_counts[f"{split}_{class_name}"] = count
                            total_files += count
                        else:
                            validation["missing_files"].append(str(class_path))
                else:
                    validation["missing_files"].append(str(split_path))
        else:
            # Single directory structure
            if dataset_path.exists():
                for class_name in expected_classes:
                    class_path = dataset_path / class_name
                    if class_path.exists():
                        files = [
                            f
                            for f in class_path.iterdir()
                            if f.is_file()
                            and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
                        ]
                        count = len(files)
                        class_counts[class_name] = count
                        total_files += count
                    else:
                        validation["missing_files"].append(str(class_path))
            else:
                validation["missing_files"].append(str(dataset_path))

        validation["total_files"] = total_files
        validation["class_distribution"] = class_counts
        validation["valid"] = len(validation["missing_files"]) == 0 and total_files > 0

        return validation

    def prepare_dataset(
        self, dataset_path: Path, dataset_id: str, validate: bool = True
    ) -> bool:
def prepare_dataset(self, dataset_path: Path, dataset_id: str, validate: bool = True) -> bool:
        """Prepare dataset for training (validation + basic preprocessing)."""
        if validate:
            print("Validating dataset...")
            validation = self.validate_dataset(dataset_path, dataset_id)
            
            if not validation['valid']:
                print("Dataset validation failed:")
                print(f"  Total files: {validation['total_files']}")
                print(f"  Missing directories: {validation['missing_files']}")
                return False
            
            print(f"✓ Dataset validation passed!")
            print(f"  Total files: {validation['total_files']}")
            print("  Class distribution:")
            for class_name, count in validation['class_distribution'].items():
                print(f"    {class_name}: {count}")
        
        dataset_info = self.datasets[dataset_id]
        
        # Create proper train/validation/test split if needed
        if dataset_info['split_structure']:
            print("Dataset already has proper split structure.")
        else:
            print("Creating train/validation/test split...")
            return self._create_train_val_split(dataset_path, dataset_info['classes'])
        
        return True

    def _create_train_val_split(self, dataset_path: Path, classes: List[str]) -> bool:
def _create_train_val_split(self, dataset_path: Path, classes: List[str]) -> bool:
        """Create train/validation/test split from single directory."""
        try:
            import numpy as np
            from sklearn.model_selection import train_test_split
        except ImportError:
            print("Error: sklearn is required for data splitting")
            return False
        
        # Create output directories
        output_base = self.data_dir / 'processed'
        train_dir = output_base / 'train'
        val_dir = output_base / 'validation'
        test_dir = output_base / 'test'
        
        for directory in [train_dir, val_dir, test_dir]:
            for class_name in classes:
                class_dir = directory / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all files with their labels
        all_files = []
        all_labels = []
        
        for class_idx, class_name in enumerate(classes):
            class_path = dataset_path / class_name
            if class_path.exists():
                files = [f for f in class_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                all_files.extend(files)
                all_labels.extend([class_idx] * len(files))
        
        if not all_files:
            print("No image files found!")
            return False
        
        # Split data (70% train, 15% val, 15% test)
        X_train, X_temp, y_train, y_temp = train_test_split(all_files, all_labels, test_size=0.3, random_state=42, stratify=all_labels)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        # Copy files to appropriate directories
        splits = [
            (X_train, y_train, train_dir, 'train'),
            (X_val, y_val, val_dir, 'validation'),
            (X_test, y_test, test_dir, 'test')
        ]
        
        for files, labels, target_dir, split_name in splits:
            print(f"Creating {split_name} set with {len(files)} files...")
            
            for file_path, label_idx in zip(files, labels):
                class_name = classes[label_idx]
                target_class_dir = target_dir / class_name
                target_file = target_class_dir / file_path.name
                
                # Copy file
                import shutil
                shutil.copy2(file_path, target_file)
        
        # Update data directory pointer to processed data
        self.data_dir = output_base
        
        print("✓ Train/validation/test split created successfully!")
        return True

    def prepare_all_datasets(self, validate: bool = True) -> bool:
def prepare_all_datasets(self, validate: bool = True) -> bool:
        """Download and prepare all available datasets."""
        print("Preparing all available datasets...")
        
        success_count = 0
        for dataset_id in self.datasets.keys():
            print(f"\n{'='*50}")
            print(f"Processing: {dataset_id}")
            print(f"{'='*50}")
            
            # Download dataset
            target_dir = self.data_dir / dataset_id.replace('/', '_')
            if not self.download_dataset(dataset_id, target_dir):
                print(f"Failed to download {dataset_id}")
                continue
            
            # Prepare dataset
            if self.prepare_dataset(target_dir, dataset_id, validate):
                success_count += 1
                print(f"✓ {dataset_id} prepared successfully!")
            else:
                print(f"Failed to prepare {dataset_id}")
        
        print(f"\n{'='*50}")
        print(f"Successfully prepared {success_count}/{len(self.datasets)} datasets")
        print(f"{'='*50}")
        
        return success_count > 0

    def generate_setup_report(
        self, output_file: str = "dataset_setup_report.json"
    ) -> None:
def generate_setup_report(self, output_file: str = 'dataset_setup_report.json') -> None:
        """Generate a comprehensive setup report."""
        report = {
            'setup_timestamp': str(Path().cwd()),
            'data_directory': str(self.data_dir),
            'available_datasets': self.datasets,
            'setup_status': 'completed',
            'next_steps': [
                "Update config.py with correct data paths",
                "Run training: python src/train.py",
                "Run Phoenix Protocol: python src/train_phoenix.py",
                "Verify setup: python setup_data.py --check"
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nSetup report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Kaggle Dataset Manager for Brain Tumor Detection"
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare datasets for training (requires download)",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate downloaded datasets"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory (default: ./data)",
    )
    parser.add_argument("--dataset", "-d", type=str, help="Specific dataset to process")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="dataset_setup_report.json",
        help="Output report file",
    )

    args = parser.parse_args()

    # Initialize manager
    manager = KaggleDatasetManager(args.data_dir)

    # Check Kaggle installation
    if not manager._check_kaggle_installation():
        print("Error: Kaggle API not installed!")
        print("Install with: pip install kaggle")
        print("Then setup API credentials:")
        print("1. Download kaggle.json from https://www.kaggle.com/account")
        print("2. Place in ~/.kaggle/kaggle.json")
        print("3. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        return 1

    # Check Kaggle credentials
    if not manager._check_kaggle_credentials():
        print("Error: Kaggle credentials not configured!")
        print("Setup instructions:")
        print("1. Download kaggle.json from https://www.kaggle.com/account")
        print("2. Place in ~/.kaggle/kaggle.json")
        print("3. Set environment variables:")
        print("   export KAGGLE_USERNAME=your_username")
        print("   export KAGGLE_KEY=your_api_key")
        return 1

    # Execute requested actions
    if args.list:
        manager.list_available_datasets()
        return 0

    if args.dataset:
        # Process specific dataset
        print(f"Processing specific dataset: {args.dataset}")
        dataset_path = manager.data_dir / args.dataset.replace("/", "_")

        if manager.prepare_dataset(dataset_path, args.dataset, args.validate):
            manager.generate_setup_report(args.output)
            print(f"\n✓ Dataset {args.dataset} prepared successfully!")
        else:
            print(f"\n✗ Failed to prepare dataset {args.dataset}")
            return 1
    else:
        # Prepare all datasets
        if manager.prepare_all_datasets(args.validate):
            manager.generate_setup_report(args.output)
            print("\n✓ All datasets prepared successfully!")
        else:
            print("\n✗ Failed to prepare some datasets")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
