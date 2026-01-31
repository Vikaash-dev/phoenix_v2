#!/bin/bash

# Phoenix Protocol Cloud Training Launcher
# Usage: ./start_cloud_training.sh [API_KEY]

echo "================================================================"
echo "   PHOENIX PROTOCOL: CLOUD TRAINING LAUNCHER"
echo "   NeuroSnake-Series (Spectral, KAN, TTT, Liquid)"
echo "================================================================"

# 1. Setup Environment Variables
if [ -z "$1" ]; then
    echo "Info: No Tavily API Key provided. Research agent features will be disabled."
else
    export TAVILY_API_KEY="$1"
    echo "✓ API Key set."
fi

# 2. Install Dependencies
echo -e "\n[1/4] Installing dependencies..."
pip install -r requirements.txt
pip install opencv-python-headless scipy tensorflow-addons

# 3. Check Data
echo -e "\n[2/4] Checking dataset..."
if [ -d "./data/train" ]; then
    echo "✓ Data directory found."
else
    echo "⚠ Data directory not found at ./data"
    echo "  The training script will run in 'Dry Run' mode using synthetic data."
    echo "  To train on real data, please upload/mount your dataset to ./data"
fi

# 4. Select Model
echo -e "\n[3/4] Select Model Architecture:"
echo "1) NeuroSnake-Spectral (Frequency Domain Gating)"
echo "2) NeuroSnake-KAN (Kolmogorov-Arnold Networks)"
echo "3) NeuroSnake-Liquid (Continuous-Time Dynamics)"
echo "4) Hyper-Liquid (Adaptive Dynamics)"
echo "5) Baseline (Standard CNN)"

read -p "Enter choice [3]: " choice
choice=${choice:-3}

case $choice in
    1) model="neurosnake_spectral";;
    2) model="neurosnake_kan";;
    3) model="neurosnake_liquid";;
    4) model="neurosnake_hyper";;
    5) model="baseline";;
    *) model="neurosnake_liquid";;
esac

echo "Selected: $model"

# 5. Launch Training
echo -e "\n[4/4] Launching Training..."
echo "Command: python src/train_phoenix.py --model-type $model --epochs 50 --batch-size 32"

python src/train_phoenix.py --model-type $model --epochs 50 --batch-size 32

echo -e "\n================================================================"
echo "Training Complete. Check ./results for logs and weights."
echo "================================================================"
