# SRC/MODELS MODULE KNOWLEDGE BASE

**Context**: Consolidated SOTA model components from all jules sessions.

## OVERVIEW

Contains 7 key architecture files implementing the Phoenix-2.0 protocol:

- **Backbone**: Dynamic Snake Convolutions
- **Global Context**: Spectral Gating (FFT)
- **Dynamics**: Liquid Neural Nets, Hyper-Liquid
- **Head**: KAN, TTT-KAN
- **Unified**: Phoenix V2 Model

## WHERE TO LOOK

| Component | File | Key Class |
| :--- | :--- | :--- |
| **KAN Layer** | `kan_layer.py` | `KANLinear` |
| **TTT-KAN** | `ttt_kan.py` | `TTTKANLinear` |
| **Snake Conv** | `dynamic_snake_conv.py` | `SnakeConvBlock` |
| **Liquid Layer** | `liquid_layer.py` | `LiquidConv2D` |
| **Hyper-Liquid** | `hyper_liquid.py` | `HyperLiquidConv2D` |
| **Spectral Gating** | `spectral_gating.py` | `SpectralGatingBlock` |
| **Phoenix V2** | `phoenix_v2.py` | `PhoenixV2Model` |

## USAGE

```python
from src.models.phoenix_v2 import PhoenixV2Model

# Standard model
model = PhoenixV2Model.create_model(num_classes=4)

# With KAN head (higher expressivity)
model_kan = PhoenixV2Model.create_model_with_kan_head(num_classes=4)
```

## CONVENTIONS

- All models use `create_model()` factory pattern
- TensorFlow/Keras only (no PyTorch)
- Layers are serializable via `get_config()`
