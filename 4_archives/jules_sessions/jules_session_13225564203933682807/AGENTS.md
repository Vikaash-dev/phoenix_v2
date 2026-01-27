# SESSION 1322 - NEUROKAN PROTOCOL KNOWLEDGE BASE

**Context**: Introduces KAN (Kolmogorov-Arnold Networks) and NeuroKAN architecture.

## OVERVIEW

First implementation of replacing MLP heads with learnable activation functions (KAN). Introduces Boundary Loss for tumor edge sharpening.

## STRUCTURE

```text
jules_session_1322.../
├── models/               # KAN layer, NeuroKAN model, Dynamic Snake Conv
├── src/                  # Training, evaluation, kfold, Mamba-Spectral
├── tools/                # Simulation data generators, ablation
└── results/              # Simulated metrics and figures
```

## WHERE TO LOOK

| Task | File | Notes |
| :--- | :--- | :--- |
| **KAN Layer** | `models/kan_layer.py` | FastKAN with RBF approximation |
| **NeuroKAN Model** | `models/neurokan_model.py` | Snake Conv + KAN Head |
| **Spectral-Mamba** | `src/models/neuro_mamba_spectral.py` | Alternative SSM-based approach |
| **Research Paper** | `RESEARCH_PAPER_2_0.md` | Full NeuroKAN protocol thesis |
| **Training** | `src/train_phoenix.py` | Phoenix Protocol training loop |

## KEY INNOVATIONS

- **FastKAN**: RBF-based approximation of B-splines for GPU efficiency
- **Log-Cosh Boundary Loss**: Gradient-based edge sharpening
- **Adan Optimizer**: 3-moment gradient estimation

## CONVENTIONS

- Models use `@staticmethod create_model()` pattern
- Compound Loss = Focal + LogCoshDice + Boundary
