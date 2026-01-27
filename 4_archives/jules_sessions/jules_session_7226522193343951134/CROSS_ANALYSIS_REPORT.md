# Cross-Analysis Report: NeuroSnake vs. SOTA Literature

**Date:** January 25, 2026  
**Benchmarks:** Medical Image Analysis (2024-2025)

---

## 1. NeuroSnake vs. "MedKAN" (arXiv:2502.18416)

**MedKAN** proposes a full-stack KAN architecture for medical imaging.
*   **Their Approach**: Replaces *all* convolutions with KAN-Convolutions.
*   **Our Approach (NeuroSnake-KAN)**: Hybrid backbone (Snake Conv) + KAN Head.
*   **Analysis**: MedKAN reports extreme parameter efficiency but significant training instability. Our hybrid approach stabilizes training by relying on robust convolutions for feature extraction, using KANs only for the final decision boundary. This aligns with the "Critical Assessment" paper (arXiv:2407.11075) suggesting KANs struggle with high-frequency texture.

## 2. NeuroSnake vs. "HyDA" (arXiv:2503.04979)

**HyDA** uses hypernetworks for domain adaptation.
*   **Their Approach**: Generates weights for the entire network based on domain labels.
*   **Our Approach (Hyper-Liquid)**: Generates *only* the time-constant ($\tau$) based on *input entropy*.
*   **Analysis**: HyDA requires domain labels. Our Hyper-Liquid is "Zero-Shot" (input-dependent). By restricting the hypernetwork to modulate only the *dynamics* (time-constant), we reduce the search space, likely leading to faster convergence than generating full weight matrices.

## 3. NeuroSnake vs. "Liquid-S4" (arXiv:2510.07578)

**Liquid-S4** combines LNNs with Structured State Spaces.
*   **Their Approach**: Focuses on long-range sequence modeling (time-series/video).
*   **Our Approach (Liquid-Snake)**: Adapts LNNs to *spatial* data via `EfficientLiquidConv2D`.
*   **Analysis**: We innovated by replacing the dense recurrent matrix $W_{rec}$ with a Depthwise Separable Convolution. This addresses the $O(C^2)$ parameter explosion criticized in standard LNN papers, making our `EfficientLiquid` layer feasible for high-channel vision tasks.

---

## 4. Fundamental Differentiation

| Feature | Competitors (MedKAN, HyDA) | NeuroSnake-Series |
|---------|----------------------------|-------------------|
| **Backbone** | Pure KAN or Standard CNN | **Dynamic Snake Convolution** (Geometric) |
| **Dynamics** | Static | **Adaptive ODEs** (Hyper-Liquid) |
| **Deployment** | Often ignored | **Synthetic Fallback & Cloud Ready** |

NeuroSnake wins on **Geometric Adaptability** (Snake Convs) and **Causal Robustness** (Hyper-Liquid dynamics).
