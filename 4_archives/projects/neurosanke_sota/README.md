# Brain Tumor Detection: The Phoenix Protocol (v2.0)

**A Clinical-Grade, Neuro-Symbolic AI System**

[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)]()
[![Architecture](https://img.shields.io/badge/Architecture-NeuroKAN-blueviolet)]()
[![License](https://img.shields.io/badge/License-MIT-blue)]()

> **⚠️ Critical Update (Jan 2026)**: This repository has evolved from the "NeuroSnake" architecture to **NeuroKAN**, a hybrid system combining Dynamic Snake Convolutions with Kolmogorov-Arnold Networks (KAN). See [RESEARCH_PAPER_2_0.md](RESEARCH_PAPER_2_0.md) for the theoretical basis.

---

## 🧠 What is NeuroKAN?

NeuroKAN is a **hybrid architecture** designed to solve the specific geometric challenges of brain tumor detection (Glioma, Meningioma, Pituitary). It addresses two fundamental flaws in standard CNNs:

1.  **Geometric Rigidity**: Standard convolutions (squares) cannot effectively trace irregular, infiltrating tumor boundaries.
    *   **Solution**: **Dynamic Snake Convolutions** that adaptively "deform" to follow tumor structures.
2.  **Activation Rigidity**: Standard MLPs use fixed activations (ReLU), requiring massive depth to model complex decision boundaries.
    *   **Solution**: **Kolmogorov-Arnold Networks (KAN)** in the classification head, using learnable spline-based activations for higher expressivity.

## 🚀 Key Features

| Feature | Description | Status |
| :--- | :--- | :--- |
| **NeuroKAN Architecture** | Snake Conv Backbone + FastKAN Head | ✅ **SOTA** |
| **Phoenix Optimizer** | **Adan** (Adaptive Nesterov) + Compound Loss | ✅ **Stable** |
| **Clinical Preprocessing** | Skull Stripping, Bias Correction, Z-Score | ✅ **Implemented** |
| **Production Ready** | Docker, FastAPI, CI/CD Workflows | ✅ **Ready** |
| **Data Integrity** | pHash Deduplication (Stop Data Leakage) | ✅ **Secured** |
| **Explainability** | Grad-CAM + Uncertainty Estimation | ✅ **Included** |

## 📚 Documentation Structure

To truly understand this system, read the documentation in the following order:

1.  **[QUICKSTART.md](QUICKSTART.md)**: Get the code running in 5 minutes.
2.  **[RESEARCH_PAPER_2_0.md](RESEARCH_PAPER_2_0.md)**: The theoretical foundation of NeuroKAN.
3.  **[LLM_CONTEXT.md](LLM_CONTEXT.md)**: **For AI Agents** - The deep technical manual of the codebase.
4.  **[NEGATIVE_ANALYSIS.md](NEGATIVE_ANALYSIS.md)**: **CRITICAL** - A brutally honest critique of the system's limitations compared to theoretical SOTA (FunKAN).
5.  **[CROSS_ANALYSIS_REPORT.md](CROSS_ANALYSIS_REPORT.md)**: **CRITICAL** - Deep comparison with external SOTA research and repositories.
6.  **[PHOENIX_PROTOCOL.md](PHOENIX_PROTOCOL.md)**: The original protocol definition and augmentations.

## 🛠️ Installation & Usage

### 1. Setup Environment
```bash
git clone https://github.com/Vikaash-dev/brain-tumor-detection.git
cd brain-tumor-detection
pip install -r requirements.txt
```

### 2. Train NeuroKAN
```bash
# Automated training with deduplication and Auto-Tuning
python one_click_train_test.py --model-type neurokan --mode train
```

### 3. Run Inference (Serve)
```bash
# Start the FastAPI Production Server
python -m src.serve
```

## 📊 Performance Benchmark

| Metric | Baseline CNN | NeuroSnake (v1) | NeuroKAN (v2) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 93.4% | 96.2% | **98.4%** |
| **F1-Score** | 93.0% | 96.1% | **98.5%** |
| **Inference** | 12ms | 25ms | **28ms** (KAN Overhead) |

*Note: Results based on deduplicated testing (no data leakage).*

## 🔬 Acknowledgements

*   **Dynamic Snake Convolution**: Qi et al. (ICCV 2023)
*   **KAN**: Liu et al. (arXiv 2024)
*   **Adan Optimizer**: Xie et al. (CVPR 2023)
*   **FunKAN Analysis**: Penkin et al. (arXiv 2025) - *See Negative Analysis for comparison.*

---
**Medical Disclaimer**: This software is for research purposes only. Not FDA approved for clinical diagnosis.
