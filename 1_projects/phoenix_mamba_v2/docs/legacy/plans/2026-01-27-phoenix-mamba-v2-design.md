# PHOENIX-MAMBA v2: Design Specification

## 🏥 Clinical Problem Statement
Brain tumor detection requires high-resolution MRI analysis to identify subtle infiltrative margins. Existing 2D CNNs lack volumetric context, while 3D Transformers suffer from $O(n^2)$ complexity, making them difficult to deploy on hospital edge devices.

## 🚀 Architectural Innovation
PHOENIX-MAMBA v2 implements a **Hierarchical Selective State-Space Model** that achieves linear complexity $O(n)$ while capturing long-range spatial and volumetric dependencies.

### 1. Macro-Architecture
- **Input**: 224x224x3 (2.5D context stack)
- **Backbone**: 4-Stage Hierarchical Mamba (64, 128, 256, 512 channels)
- **Complexity**: $O(n)$ spatial scaling
- **Parameters**: ~1.4M (12.5% reduction vs HYDRA v1)

### 2. Core Components
- **Selective SSM (S6)**: Input-dependent state transitions for dynamic feature selection.
- **2.5D Slice Aggregator**: Temporal-style attention across 3 adjacent slices for volumetric context.
- **Heterogeneity-Aware Attention**: Region-specific heads for Necrotic Core (NCR), Enhancing Tumor (ET), and Edema (ED).
- **Contrastive Encoder**: SimCLR-style self-supervised pretraining for domain robustness.

### 3. Clinical Specialization
- **Multi-Modal Fusion**: Cross-modal attention for T1, T2, FLAIR, and T1-CE sequence alignment.
- **Interpretability Module**: Integrated Grad-CAM and attention map generation for "Trust-but-Verify" workflows.
- **Uncertainty Calibration**: Softmax-based probability heads with Dropout-based variance estimation.

## 🛠️ Training & Deployment Strategy
- **Optimizer**: AdamW with 5-epoch linear warmup and cosine decay.
- **Knowledge Distillation**: Teacher-student framework for edge compression.
- **Inference Targets**: 25ms GPU / 90ms CPU latency.

---
*Document Version: 2.0.0*
*Date: 2026-01-27*
*Project: 1_projects/phoenix_mamba_v2*
