#!/usr/bin/env python3
"""
Phoenix Mamba V2 - Unified Training & Testing Pipeline
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path so we can import src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.processors import ClinicalPreprocessor, FastPreprocessor
from src.data.augmentation import PhysicsInformedAugmentor
from src.models.adapters import NeuroSnakeAdapter
# We will assume a training loop exists or implement a basic one here for the adapter to use
# For now, we'll implement a basic integration in main

def parse_args():
    parser = argparse.ArgumentParser(description="Phoenix Mamba V2 Pipeline")

    # Mode
    parser.add_argument("--mode", type=str, choices=["train", "test", "demo"], default="train", help="Execution mode")

    # Data
    parser.add_argument("--data-dir", type=str, default="data", help="Path to dataset directory")
    parser.add_argument("--preprocessor", type=str, choices=["clinical", "fast"], default="clinical", help="Preprocessing strategy")

    # Model
    parser.add_argument("--model-variant", type=str, choices=["standard", "coordinate_attention", "baseline"], default="coordinate_attention", help="Model architecture variant")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")

    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    # Augmentation
    parser.add_argument("--augment", action="store_true", help="Enable physics-informed augmentation")

    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Phoenix Mamba V2 Pipeline | Mode: {args.mode}")
    print("=" * 60)

    # 1. Initialize Preprocessor
    if args.preprocessor == "clinical":
        print("Initializing Clinical Preprocessor (Skull Strip + Bias Correct + CLAHE)...")
        preprocessor = ClinicalPreprocessor()
    else:
        print("Initializing Fast Preprocessor (Resize + Normalize)...")
        preprocessor = FastPreprocessor()

    # 2. Initialize Augmentor
    augmentor = None
    if args.augment:
        print("Initializing Physics-Informed Augmentor (Elastic + Rician + Ghosting)...")
        augmentor = PhysicsInformedAugmentor()

    # 3. Initialize Model Adapter
    print(f"Initializing Model: NeuroSnake ({args.model_variant})...")
    model_adapter = NeuroSnakeAdapter(variant=args.model_variant, dropout_rate=args.dropout)

    if args.mode == "demo":
        print("\n[DEMO MODE] Running component verification...")

        # Verify Preprocessor config
        print(f"Preprocessor Config: {preprocessor.get_config()}")

        # Build Model to verify architecture
        print("Building model...")
        model = model_adapter.build(input_shape=(224, 224, 3), num_classes=2)
        model.summary()
        print("\n✓ Demo check passed: Components initialized successfully.")
        return 0

    # 4. Data Loading (Stub for now)
    # In a real implementation, we would use the PipelineContext here to create tf.data.Datasets
    print(f"\nScanning data directory: {args.data_dir}")
    if not os.path.exists(args.data_dir):
        print(f"⚠ Warning: Data directory '{args.data_dir}' does not exist.")
        if args.mode == "train":
            print("Cannot train without data. Exiting.")
            return 1

    if args.mode == "train":
        print(f"Starting training for {args.epochs} epochs...")
        # Placeholder for actual training call
        # history = model_adapter.train(train_ds, val_ds, epochs=args.epochs)
        print("⚠ Training loop requires configured DataLoaders (implement PipelineContext next)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
