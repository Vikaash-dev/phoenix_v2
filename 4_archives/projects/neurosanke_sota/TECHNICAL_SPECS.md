# Technical Specifications: NeuroKAN System

**Version**: 2.1 (FastKAN Update)
**Date**: January 2026

---

## 1. System Requirements

### Hardware
*   **Minimum**: NVIDIA GPU (6GB VRAM) - e.g., RTX 2060 / Tesla T4.
*   **Recommended**: NVIDIA GPU (12GB+ VRAM) - e.g., RTX 3080 / A100.
*   **CPU**: 4+ Cores (for parallel preprocessing).
*   **RAM**: 16GB+.

### Software Stack
*   **OS**: Linux (Ubuntu 22.04) or Windows 11 (WSL2).
*   **Python**: 3.9 - 3.10.
*   **Framework**: TensorFlow 2.13+ / Keras.
*   **Container**: Docker (cuda:11.8-cudnn8-runtime).

---

## 2. Model Architecture Specifications

### 2.1 Backbone (`SnakeConvBlock`)
*   **Kernel**: Dynamic Snake Convolution (Axis-aligned deformation).
*   **Attention**: Coordinate Attention (CA) or SE-Vector (Configurable).
*   **Depth**: 4 Stages ([64, 128, 256, 512] filters).
*   **Downsampling**: MaxPool ($2\times2$).

### 2.2 Classification Head (`KANDense`)
*   **Type**: FastKAN (Radial Basis Function Approximation).
*   **Spline Order**: 3 (Cubic).
*   **Grid Size**: 5 (Default) - 10 (High Precision).
*   **Grid Range**: $[-1, 1]$.
*   **Activation**: SiLU (Swish) for base weights.
*   **Parameter Count**: $\approx 1.5 \times$ standard Dense layer (due to spline coefficients).

---

## 3. Training Protocol

### 3.1 Optimizer (`Adan`)
*   **Learning Rate**: $1e-3$ to $5e-4$ (with Cosine Decay).
*   **Betas**: $(0.98, 0.92, 0.99)$ - Tuned for stability.
*   **Weight Decay**: $0.02$.

### 3.2 Loss Function
*   **Composite**: $L = L_{Focal} + 0.5 \cdot L_{LogCoshDice} + 0.1 \cdot L_{Boundary}$.
*   **Focal Gamma**: $2.0$.
*   **Boundary Alpha**: $0.1$ (Scheduled ramp-up).

---

## 4. Performance Targets

*   **Training Time**: < 1 hour on RTX 3080 (100 epochs).
*   **Inference Latency**: < 50ms per slice (Batch 1).
*   **Throughput**: > 100 slices/sec (Batch 32).
*   **Metric Thresholds**:
    *   Accuracy: $> 98\%$
    *   Precision (Tumor): $> 99\%$
    *   Recall (Tumor): $> 97\%$

---

## 5. Security & Integrity

*   **Deduplication**: pHash Hamming Distance $\le 5$ (Strict).
*   **Model Signing**: SHA-256 Checksum on `.h5` export.
*   **Adversarial Defense**: Input sanitization (range check, NaN check) in `src/serve.py`.
