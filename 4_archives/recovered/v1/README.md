# Version 1 - Phoenix Protocol Baseline Implementation

## Overview

This is the baseline implementation of the brain tumor detection system featuring the **Phoenix Protocol** with the **NeuroSnake architecture**.

## Key Features

- **NeuroSnake Architecture**: Hybrid model combining Dynamic Snake Convolutions with MobileViT-v2 blocks
- **Phoenix Protocol**: Data deduplication, physics-informed augmentation, Adan optimizer
- **EfficientQuant**: Hybrid quantization for edge deployment
- **Clinical Robustness**: Security-hardened for real-world deployment
- **Complete Pipeline**: From data preprocessing to model deployment

## Architecture Highlights

1. **Dynamic Snake Convolutions (DSC)**: Adaptively trace irregular tumor boundaries
2. **MobileViT-v2 Blocks**: Capture global context with lightweight design
3. **Coordinate Attention**: Enhanced spatial attention mechanism
4. **Adan Optimizer**: Superior stability on medical imaging tasks
5. **Focal Loss**: Handle class imbalance effectively

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Setup data
python setup_data.py

# Train the model
python one_click_train_test.py --mode train --model-type neurosnake_ca

# Run validation
python validate_implementation.py
```

## Key Components

- `src/` - Core source code
  - `train.py` - Training script
  - `evaluate.py` - Evaluation utilities
  - `data_preprocessing.py` - Data preprocessing pipeline
  - `efficient_quant.py` - Quantization for edge deployment
- `models/` - Model architectures
  - `neurosnake_model.py` - NeuroSnake implementation
  - `dynamic_snake_conv.py` - Dynamic Snake Convolution layer
  - `coordinate_attention.py` - Coordinate Attention mechanism
- `test_*.py` - Comprehensive test suite

## Performance

- Baseline CNN: ~96% accuracy
- NeuroSnake: Enhanced performance with better boundary detection
- INT8 Quantized: 2.5-8.7× speedup with <1% accuracy loss

## Reference

This version corresponds to the main branch as of the consolidation and represents the foundational implementation of the Phoenix Protocol.

For more details, see the root documentation files:
- `../PHOENIX_PROTOCOL.md`
- `../RESEARCH_PAPER_FINAL.md`
- `../TECHNICAL_SPECS.md`

## Additional Resources

- [VERSION_GUIDE.md](../VERSION_GUIDE.md) - Compare all versions
- [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md) - Migrate to v2 or v3
- [PR_REFERENCES.md](../PR_REFERENCES.md) - Source pull requests
