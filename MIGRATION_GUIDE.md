# Migration Guide

This guide helps you migrate between different versions of the brain tumor detection system.

## Quick Reference

```bash
# Use v1 (baseline)
cd v1 && pip install -r requirements.txt

# Use v2 (SOTA upgrade)
cd v2 && pip install -r requirements.txt

# Use v3 (Spectral-Snake)
cd v3 && pip install -r requirements.txt
```

## Migrating from v1 to v2

### What Changes
- **Training Infrastructure**: Adds AMP (Mixed Precision) and K-Fold validation
- **Model Architecture**: Adds SEVector attention and KAN layers
- **Loss Function**: Introduces Log-Cosh Dice Loss
- **Deployment**: Adds ONNX export and serving capabilities

### Step-by-Step Migration

1. **Install Additional Dependencies**
   ```bash
   cd v2
   pip install -r requirements.txt
   # Additional packages: tf2onnx, onnxruntime
   ```

2. **Update Training Code**
   ```python
   # Old (v1)
   python one_click_train_test.py --mode train --model-type neurosnake_ca
   
   # New (v2) - with k-fold validation
   python src/kfold_training.py --model-type neurosnake_ca --k-folds 5
   ```

3. **Enable Mixed Precision** (Optional)
   ```python
   # Add to your training script
   from tensorflow.keras import mixed_precision
   mixed_precision.set_global_policy('mixed_float16')
   ```

4. **Export to ONNX** (New in v2)
   ```bash
   python src/export_onnx.py --model-path models/best.h5 --output model.onnx
   ```

5. **Serve the Model** (New in v2)
   ```bash
   python src/serve.py --model-path model.onnx --port 8000
   ```

### Compatibility Notes
- v2 models are backward compatible with v1 data pipelines
- Mixed precision requires NVIDIA GPU with compute capability >= 7.0
- ONNX export works with CPU and GPU models

### Configuration Changes
```python
# v1 config
config = {
    'model_type': 'neurosnake_ca',
    'optimizer': 'adan',
    'loss': 'focal'
}

# v2 config (enhanced)
config = {
    'model_type': 'neurosnake_ca',
    'optimizer': 'adan',
    'loss': 'log_cosh_dice',  # NEW
    'mixed_precision': True,   # NEW
    'k_folds': 5              # NEW
}
```

## Migrating from v2 to v3

### What Changes
- **Architecture**: Replaces MobileViT-v2 with Spectral Gating Blocks (FFT-based)
- **Performance**: Significantly faster inference (35ms vs 42ms)
- **Parameters**: Fewer parameters (1.8M vs 2.3M)
- **Accuracy**: Improved accuracy (96.8% vs 95.8%)

### Step-by-Step Migration

1. **Install Research Dependencies**
   ```bash
   cd v3
   pip install -r requirements.txt
   # Additional packages for spectral analysis and research framework
   ```

2. **Update Model Type**
   ```python
   # Old (v2)
   python one_click_train_test.py --model-type neurosnake_ca
   
   # New (v3) - Spectral-Snake
   python one_click_train_test.py --model-type neurosnake_spectral
   ```

3. **Use Enhanced Research Framework**
   ```bash
   # Run comprehensive experiments
   python run_research_experiments.py --architecture spectral
   
   # Generate research paper
   python src/research/generate_paper.py
   ```

4. **Leverage FFT Optimizations**
   ```python
   # v3 models automatically use FFT-based gating
   # No code changes needed - just specify model type
   ```

### Model Conversion
```python
# Convert v2 model to v3 architecture
from models.neurosnake_spectral import convert_from_v2

# Load v2 weights
v2_model = load_model('v2/models/best.h5')

# Convert to v3 (transfers compatible weights)
v3_model = convert_from_v2(v2_model)

# Fine-tune spectral gating blocks
# (New blocks are randomly initialized)
```

### Compatibility Notes
- v3 requires the same input preprocessing as v1/v2
- Spectral gating works on both CPU and GPU
- FFT operations are optimized with TensorFlow's native FFT

## Migrating from v1 to v3 (Direct)

If you want to skip v2 and go directly to v3:

1. **Prerequisites**
   ```bash
   pip install -r v3/requirements.txt
   ```

2. **Retrain from Scratch**
   ```bash
   cd v3
   python one_click_train_test.py --mode train --model-type neurosnake_spectral
   ```

3. **Or Transfer Weights**
   ```python
   # Load v1 model
   v1_model = load_model('../v1/models/best.h5')
   
   # Initialize v3 model
   v3_model = NeuroSnakeSpectral(input_shape=(224, 224, 3))
   
   # Transfer snake convolution and attention weights
   transfer_compatible_weights(v1_model, v3_model)
   
   # Fine-tune on your data
   ```

## Data Compatibility

All versions use the same data format:
- Input: 224×224×3 MRI images
- Output: 4-class classification (Glioma, Meningioma, Pituitary, No Tumor)

```python
# Data pipeline works across all versions
from src.data_preprocessing import preprocess_data

# Works for v1, v2, and v3
train_data, val_data = preprocess_data('data/')
```

## Configuration Files

### v1 Configuration
```yaml
model: neurosnake_ca
optimizer: adan
loss: focal
batch_size: 32
epochs: 50
```

### v2 Configuration
```yaml
model: neurosnake_ca
optimizer: adan
loss: log_cosh_dice
batch_size: 32
epochs: 50
mixed_precision: true
k_folds: 5
onnx_export: true
```

### v3 Configuration
```yaml
model: neurosnake_spectral
optimizer: adan
loss: log_cosh_dice
batch_size: 32
epochs: 50
mixed_precision: true
spectral_layers: 4
fft_optimization: true
```

## Performance Comparison

| Operation | v1 | v2 | v3 |
|-----------|----|----|-----|
| Training (1 epoch) | 10 min | 7 min (AMP) | 9 min |
| Inference | 45ms | 42ms | **35ms** |
| Memory | 120MB | 115MB | **95MB** |
| Accuracy | 95.2% | 95.8% | **96.8%** |

## Rollback Strategy

If you need to rollback:

```bash
# From v3 to v2
cd ../v2
python one_click_train_test.py --model-type neurosnake_ca

# From v2 to v1
cd ../v1
python one_click_train_test.py --model-type neurosnake_ca
```

## Troubleshooting

### v2 Mixed Precision Issues
```python
# If you encounter errors with mixed_precision
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('float32')
```

### v3 FFT Performance
```python
# For CPU optimization
import os
os.environ['TF_NUM_INTEROP_THREADS'] = '4'
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
```

### Model Loading Errors
```python
# Use custom_objects when loading
from models.neurosnake_model import NeuroSnake
from models.dynamic_snake_conv import DynamicSnakeConv2D

custom_objects = {
    'NeuroSnake': NeuroSnake,
    'DynamicSnakeConv2D': DynamicSnakeConv2D
}

model = load_model('model.h5', custom_objects=custom_objects)
```

## Best Practices

1. **Always test with a small dataset first** before full migration
2. **Keep v1 models as baseline** for comparison
3. **Use k-fold validation in v2/v3** for robust evaluation
4. **Monitor memory usage** when enabling mixed precision
5. **Benchmark inference time** before deploying to production

## Support

For migration issues:
- Check [VERSION_GUIDE.md](VERSION_GUIDE.md) for version details
- See [PR_REFERENCES.md](PR_REFERENCES.md) for source PRs
- Review version-specific READMEs in each directory

## Next Steps

After successful migration:
1. Run comprehensive tests
2. Validate on your datasets
3. Benchmark performance
4. Update deployment pipelines
5. Document any custom changes
