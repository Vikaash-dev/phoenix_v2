# SRC MODULE KNOWLEDGE BASE

**Context**: Core implementation of Brain Tumor Detection (Phoenix Protocol)

## OVERVIEW

Contains the operational logic: training loops, data pipelines, evaluation, and enterprise feature implementations (P1/P2).
**Complexity**: High (21 files, mix of research & production code).

## STRUCTURE (Functional Groups)

- **Entry Points**: `train.py`, `train_phoenix.py`, `predict.py`
- **Data**: `data_preprocessing.py`, `data_deduplication.py`, `physics_informed_augmentation.py`
- **Optimization**: `phoenix_optimizer.py`, `efficient_quant.py`, `int8_quantization.py`
- **Enterprise**: `p1_features.py` (MultiGPU/QAT), `p2_features.py` (MLFlow/Docker), `onnx_deployment.py`
- **Analysis**: `evaluate.py`, `comparative_analysis.py`, `visualize.py`

## WHERE TO LOOK

| Task | File | Key Functions/Classes |
|------|------|-----------------------|
| **Standard Training** | `train.py` | `train_model`, basic Keras loops |
| **Phoenix Training** | `train_phoenix.py` | `train_phoenix_protocol` (Advanced) |
| **Data Loading** | `data_preprocessing.py` | `BrainTumorDataset`, `load_data` |
| **Custom Optimization** | `phoenix_optimizer.py` | `PhoenixOptimizer` (Adaptive LR) |
| **Quantization** | `efficient_quant.py` | `hybrid_quantization` |
| **Enterprise P1** | `p1_features.py` | `EnterpriseFeatures` (Distributed training) |
| **Enterprise P2** | `p2_features.py` | `ProductionSystem` (Model Registry) |

## CODE MAP (Key Symbols)

| Symbol | Type | File | Role |
|--------|------|------|------|
| `PhoenixOptimizer` | Class | `phoenix_optimizer.py` | Custom optimizer with gradient centralization |
| `BrainTumorDataset` | Class | `data_preprocessing.py` | Main data loader (handles MRI formats) |
| `PhysicsInformedAugmentation` | Class | `physics_informed_augmentation.py` | MRI-specific augmentations (bias field) |
| `EnterpriseFeatures` | Class | `p1_features.py` | Monolith class for advanced training features |

## CONVENTIONS

- **Argument Parsing**: Most scripts use `argparse` but rely on defaults if unspecified.
- **Enterprise Flags**: Features in `p1`/`p2` often need explicit instantiation; not always hooked into `train.py` automatically.
- **Logging**: formatted string printing prevalent over standard `logging` in older files.

## ANTI-PATTERNS

- **Ghost Features**: `p1_features.py` and `p2_features.py` contain sophisticated code that is rarely called by the main content. Check references before assuming active use.
- **Hardcoded Paths**: Some data loaders might default to `./data` relative to execution root.
