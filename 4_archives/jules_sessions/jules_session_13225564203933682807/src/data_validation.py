"""
Data Validation Tool for Phoenix Protocol
Ensures dataset quality, format compliance, and class balance before training.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates medical image datasets for:
    1. Corruption (unreadable files)
    2. Format compliance (supported extensions)
    3. Resolution consistency
    4. Class balance
    """
    
    def __init__(self, data_dir, valid_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
        self.data_dir = Path(data_dir)
        self.valid_extensions = valid_extensions
        self.report = {
            'total_files': 0,
            'valid_files': 0,
            'corrupt_files': [],
            'invalid_format': [],
            'class_distribution': {},
            'resolutions': {}
        }

    def validate_dataset(self):
        """Run full validation suite."""
        logger.info(f"Starting validation on {self.data_dir}...")
        
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                file_path = Path(root) / file
                self.report['total_files'] += 1
                
                # Check extension
                if file_path.suffix.lower() not in self.valid_extensions:
                    self.report['invalid_format'].append(str(file_path))
                    continue
                
                # Check class (assume parent folder is class name)
                class_name = file_path.parent.name
                if class_name not in self.report['class_distribution']:
                    self.report['class_distribution'][class_name] = 0
                self.report['class_distribution'][class_name] += 1
                
                # Check image integrity
                try:
                    with Image.open(file_path) as img:
                        img.verify() # Verify file structure
                        
                    # Re-open to check dimensions (verify closes file)
                    with Image.open(file_path) as img:
                        res = img.size # (W, H)
                        res_str = f"{res[0]}x{res[1]}"
                        if res_str not in self.report['resolutions']:
                            self.report['resolutions'][res_str] = 0
                        self.report['resolutions'][res_str] += 1
                        
                    self.report['valid_files'] += 1
                    
                except Exception as e:
                    logger.error(f"Corrupt file {file_path}: {e}")
                    self.report['corrupt_files'].append(str(file_path))
                    
        self._print_summary()
        return self.report

    def _print_summary(self):
        """Print validation summary."""
        print("\n" + "="*50)
        print("DATA VALIDATION REPORT")
        print("="*50)
        print(f"Total Files Scanned: {self.report['total_files']}")
        print(f"Valid Images:        {self.report['valid_files']}")
        print(f"Corrupt Images:      {len(self.report['corrupt_files'])}")
        print(f"Invalid Formats:     {len(self.report['invalid_format'])}")
        print("-" * 50)
        print("Class Distribution:")
        for cls, count in self.report['class_distribution'].items():
            print(f"  {cls}: {count}")
        print("-" * 50)
        print("Resolutions (Top 5):")
        sorted_res = sorted(self.report['resolutions'].items(), key=lambda x: x[1], reverse=True)
        for res, count in sorted_res[:5]:
            print(f"  {res}: {count}")
        print("="*50 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phoenix Protocol Data Validator")
    parser.add_argument("--data-dir", type=str, default="./data", help="Root data directory")
    args = parser.parse_args()
    
    if os.path.exists(args.data_dir):
        validator = DataValidator(args.data_dir)
        validator.validate_dataset()
    else:
        print(f"Error: Directory {args.data_dir} not found.")
