# Reproducibility Package

This directory contains all necessary files to reproduce the research experiments and results.

## Contents

- `Dockerfile` - Complete environment setup with all dependencies
- `environment.yml` - Conda environment specification
- `seeds.json` - Random seeds used in all experiments
- `training_config.json` - Complete hyperparameter configurations
- `README.md` - This file

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build the Docker image
docker build -t neurosnake-research .

# Run experiments
docker run --gpus all -v $(pwd)/research_results:/app/research_results neurosnake-research

# Results will be saved to ./research_results/
```

### Option 2: Conda Environment

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate neurosnake-research

# Run experiments
python run_research_experiments.py
```

### Option 3: Virtual Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run experiments
python run_research_experiments.py
```

## Reproducibility Guarantees

### Random Seeds
All random seeds are fixed (see `seeds.json`):
- Python hash seed: 42
- NumPy seed: 42
- TensorFlow seed: 42
- Random module seed: 42

### Deterministic Operations
The following environment variables are set:
```bash
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
export PYTHONHASHSEED=42
```

### Hardware Requirements

**Minimum:**
- CPU: 4+ cores
- RAM: 16 GB
- Storage: 20 GB

**Recommended:**
- GPU: NVIDIA GPU with 8+ GB VRAM (RTX 2060 or better)
- CPU: 8+ cores
- RAM: 32 GB
- Storage: 50 GB

### Training Time Estimates

On NVIDIA RTX 3090:
- Single fold: ~30 minutes
- 5-fold CV (one model): ~2.5 hours
- Full SOTA comparison (5 models): ~12 hours
- Ablation studies (4 configs): ~10 hours
- **Total**: ~22 hours

On CPU (16-core):
- Single fold: ~4 hours
- 5-fold CV (one model): ~20 hours
- **Not recommended for full experiments**

## Verification

After running experiments, verify outputs:

```bash
# Check experiment logs
ls -lh research_results/sota_comparison/*/experiment.log

# Check figures
ls -lh research_results/figures/

# Check results
cat research_results/sota_comparison/all_models_results.json
```

## Dataset

### Data Preparation
Place your brain MRI dataset in the following structure:
```
data/
├── train/
│   ├── no_tumor/
│   └── tumor/
├── validation/
│   ├── no_tumor/
│   └── tumor/
└── test/
    ├── no_tumor/
    └── tumor/
```

### Data Format
- Images: PNG or JPEG
- Size: 224x224 pixels (will be resized if different)
- Color: RGB (3 channels)

### Data Preprocessing
All preprocessing is handled automatically:
1. Resize to 224x224
2. Normalize to [0, 1]
3. CLAHE contrast enhancement
4. pHash-based deduplication

## Troubleshooting

### GPU Out of Memory
Reduce batch size in `run_research_experiments.py`:
```python
batch_size = 16  # Default: 32
```

### Slow Training
Enable mixed precision:
```python
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

### CUDA Errors
Ensure CUDA and cuDNN versions match TensorFlow requirements:
- TensorFlow 2.10: CUDA 11.2, cuDNN 8.1

### Dependency Conflicts
Use exact versions from `requirements.txt`:
```bash
pip install -r requirements.txt --no-deps
```

## Citation

If you use this code or reproduce our results, please cite:

```bibtex
@article{neurosnake2024,
  title={NeuroSnake: A Hybrid Dynamic Snake Convolution Architecture with Coordinate Attention for Brain Tumor Detection},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## Contact

For questions or issues:
- Open an issue on GitHub: [repository URL]
- Email: [contact email]

## License

[License information]
