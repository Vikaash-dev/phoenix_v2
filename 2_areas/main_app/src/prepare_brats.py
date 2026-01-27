"""
Script for downloading and preparing the BraTS dataset.
"""

import os
import argparse
import tarfile
import requests
from tqdm import tqdm

def download_file(url, target_path):
    """
    Download a file from a URL with a progress bar.
    """
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(target_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

def extract_tar(tar_path, extract_path):
    """
    Extract a tar file.
    """
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_path)

def main():
    """
    Main function to download and prepare the BraTS dataset.
    """
    parser = argparse.ArgumentParser(
        description="BraTS Dataset Preparation Script"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/brats",
        help="Output directory for the BraTS dataset",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Download BraTS 2020 dataset
    print("Downloading BraTS 2020 dataset...")
    url = "https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS_2020_Data_Training.zip" # This is a placeholder URL, a real one would be needed
    target_path = os.path.join(args.output_dir, "BraTS2020_TrainingData.tar")
    download_file(url, target_path)

    # Extract the dataset
    print("Extracting dataset...")
    extract_path = os.path.join(args.output_dir, "BraTS2020_TrainingData")
    extract_tar(target_path, extract_path)

    print("BraTS dataset preparation complete.")

if __name__ == "__main__":
    main()
