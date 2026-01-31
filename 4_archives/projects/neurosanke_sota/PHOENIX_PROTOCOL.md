# The Phoenix Protocol: Comprehensive Implementation Guide

**Date:** January 06, 2026
**Project:** Lightweight Neuro-Oncology AI Optimization
**Architecture:** NeuroKAN (Dynamic Snake Convolutions + Kolmogorov-Arnold Networks)
**Version:** 3.2 (Production SOTA)

---

## Executive Summary

The Phoenix Protocol is a production-grade system for detecting brain tumors from MRI scans. It implements the **NeuroKAN** architecture described in `RESEARCH_PAPER_2_0.md`, achieving state-of-the-art performance through geometric adaptability and learnable activations.

**Status**: Production-Ready.
**Key Features**:
- **NeuroKAN Architecture**: Snake Conv backbone + KAN classification head.
- **Advanced Attention**: Configurable Coordinate (Spatial) or SEVector (Channel) attention.
- **Robust Training**: Adan Optimizer, Log-Cosh Dice + Boundary Loss, Mixed Precision (AMP).
- **AutoML**: Automated Hyperparameter Optimization via Optuna.
- **Production Support**: Docker, API Serving, ONNX Export, INT8 Quantization.

---

## 1. Documentation Map

*   **[RESEARCH_PAPER_2_0.md](RESEARCH_PAPER_2_0.md)**: The theoretical foundation.
*   **[NEGATIVE_ANALYSIS.md](NEGATIVE_ANALYSIS.md)**: **CRITICAL READ**. A deep analysis of the system's limitations compared to cutting-edge 2025 research (MedKAN/FunKAN).
*   **[CROSS_ANALYSIS_REPORT.md](CROSS_ANALYSIS_REPORT.md)**: Comparison against nnU-Net, MONAI, and other benchmarks.
*   **[LLM_CONTEXT.md](LLM_CONTEXT.md)**: Technical context for AI agents.

---

## 2. Usage Guide

### 2.1 Installation
```bash
pip install -r requirements.txt
```

### 2.2 Data Preparation
Clean the dataset to prevent leakage:
```bash
python src/data_deduplication.py --data-dir ./data --remove-duplicates
```

### 2.3 Hyperparameter Optimization
Find the best configuration for your specific dataset:
```bash
python src/hyperparameter_tuning.py --trials 20
```

### 2.4 Training (SOTA Mode)
Train the NeuroKAN model with Mixed Precision:
```bash
python src/train_phoenix.py \
    --model-type neurokan \
    --mixed-precision \
    --optimizer adan \
    --epochs 100 \
    --data-dir ./data
```

### 2.5 Production Deployment
**Option A: API Server (Docker)**
```bash
docker build -t phoenix:latest .
docker run -p 8000:8000 phoenix:latest
```

**Option B: Edge Deployment (INT8)**
```bash
python src/int8_quantization.py \
    --model-path results/neurokan_best.h5 \
    --output-path deploy/model_int8.tflite
```

---

## 3. Reproducing Paper Results

To achieve the results cited in `RESEARCH_PAPER_2_0.md`:

1.  **Deduplicate**: Ensure cross-split duplicates are removed.
2.  **Cross-Validation**: Run 5-Fold CV to get robust metrics.
    ```bash
    python src/kfold_training.py --folds 5 --model neurokan
    ```
3.  **Evaluate**: Generate the full Classification Report (with Specificity/MCC).
    ```bash
    python one_click_train_test.py --mode test --model-path results/neurokan_best.h5
    ```

---

## 4. Architecture Details

### 4.1 NeuroKAN
Combines:
- **SnakeConvBlock**: Dynamic Snake Convolutions to trace tumor edges.
- **KANDense**: FastKAN layers (RBF-based) for the classification head.
- **Attention**: Coordinate Attention (default) for spatial awareness.

### 4.2 Compound Loss
$$ L = L_{Focal} + L_{Dice} + 0.5 \cdot L_{Boundary} $$
Optimizes for:
- Class Imbalance (Focal)
- Area Overlap (Dice)
- Edge Sharpness (Boundary)

---

## 5. Ethical Considerations & Error Analysis

### 5.1 Ethics
- **Bias**: Ensure training data covers diverse demographics.
- **Usage**: This is a decision support tool, NOT a replacement for a radiologist.

### 5.2 Error Analysis
- **False Negatives**: Primarily occur in low-contrast, motion-blurred scans.
- **Mitigation**: The system uses Uncertainty Quantification (Entropy) to flag these low-confidence predictions for manual review.

---

## 6. File Structure

```
project/
├── src/
│   ├── train_phoenix.py          # Main training with AMP/NeuroKAN
│   ├── kfold_training.py         # 5-Fold CV
│   ├── hyperparameter_tuning.py  # AutoML (Optuna)
│   ├── serve.py                  # FastAPI Server
│   ├── export_onnx.py            # ONNX Export
│   ├── phoenix_optimizer.py      # Adan, Boundary Loss, etc.
│   ├── clinical_postprocessing.py # TTA, Grad-CAM, Uncertainty
│   └── ...
├── models/
│   ├── neurokan_model.py         # NeuroKAN Architecture
│   ├── kan_layer.py              # FastKAN Layer
│   ├── dynamic_snake_conv.py     # Deformable Conv
│   └── ...
├── Dockerfile                    # Production container
└── RESEARCH_PAPER_2_0.md         # Full Academic Paper
```

---

**Medical Disclaimer**: This system is for research purposes only.
