# Quick Start Guide

## Getting Started in 5 Minutes

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/Vikaash-dev/Ai-research-paper-and-implementation-of-brain-tumor-detection-.git
cd Ai-research-paper-and-implementation-of-brain-tumor-detection-

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Create directory structure
python setup_data.py --create

# Organize your images:
# - Place tumor MRI images in: data/train/tumor/
# - Place non-tumor MRI images in: data/train/no_tumor/
# - Do the same for validation/ and test/ directories

# Verify setup
python setup_data.py --check
```

### 3. Train Model

```bash
# Train with default settings
python src/train.py

# This will:
# - Train for 50 epochs (with early stopping)
# - Save best model to models/saved_models/
# - Take ~2 hours on GPU, longer on CPU
```

### 4. Evaluate Model

```bash
# Evaluate on test set
python src/evaluate.py

# Results saved to results/ directory
```

### 5. Make Predictions

```bash
# Interactive prediction mode
python src/predict.py

# Or use in Python:
# from src.predict import load_model, predict_single_image
# model = load_model()
# result = predict_single_image(model, 'path/to/image.jpg')
```

## Troubleshooting

### No GPU Available
- Install CUDA and cuDNN for GPU acceleration
- Or train on CPU (will be slower)

### Out of Memory
- Reduce `BATCH_SIZE` in config.py (try 16 or 8)
- Reduce image size (try 128x128 instead of 224x224)

### Low Accuracy
- Need more training data (aim for 1000+ images per class)
- Increase training epochs
- Try data augmentation (already enabled by default)

### Import Errors
- Ensure you're in the project root directory
- Activate virtual environment
- Reinstall requirements: `pip install -r requirements.txt`

## Tips

- **Start Small**: Test with a small dataset first (~100 images)
- **Use GPU**: Training is much faster with GPU (hours vs. days)
- **Monitor Training**: Check TensorBoard: `tensorboard --logdir=./logs`
- **Backup Models**: Save your trained models regularly
- **Document Results**: Keep track of experiments and hyperparameters

## Next Steps

1. Read the full [README.md](README.md)
2. Review the [Research Paper](Research_Paper_Brain_Tumor_Detection.md)
3. Experiment with hyperparameters in [config.py](config.py)
4. Try different architectures in [models/cnn_model.py](models/cnn_model.py)

## Need Help?

- Check GitHub Issues
- Review documentation
- Read research paper for background
- Contact: [@Vikaash-dev](https://github.com/Vikaash-dev)
