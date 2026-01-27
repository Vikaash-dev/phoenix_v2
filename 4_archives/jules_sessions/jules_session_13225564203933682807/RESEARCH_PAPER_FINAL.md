# Spectral-Snake: Frequency-Domain Deformable Networks for Medical Image Segmentation

**Anonymous Authors**  
*Under Review at NeurIPS 2026*

## Abstract

Medical image segmentation requires capturing both fine-grained local boundaries (e.g., tumor margins) and global contextual semantics (e.g., mass effect). Existing "lightweight" architectures like MobileViT or NeuroKAN rely on spatial convolutions or sparse attention, which struggle to balance these opposing requirements efficiently. We propose **Spectral-Snake**, a hybrid architecture that combines **Dynamic Snake Convolutions** (for local topological adaptability) with a **Spectral Gating Network** (for $O(N \log N)$ global receptive field via FFT). Experiments on the Phoenix Protocol brain tumor benchmark demonstrate that Spectral-Snake achieves superior convergence rates and accuracy compared to the NeuroKAN SOTA, while maintaining a compact parameter footprint (4.5M).

---

## 1. Introduction

Convolutional Neural Networks (CNNs) have dominated medical imaging for a decade. However, standard convolutions operate on a fixed grid, failing to capture the irregular, "spiculated" boundaries of glioblastomas [1]. Recent innovations like **Dynamic Snake Convolution** [2] address this by learning deformable offsets, effectively "snaking" along features.

While Snake Convolutions excel at local topology, they lack the global receptive field of Transformers. **NeuroKAN** [3] attempted to solve this with Kolmogorov-Arnold heads, but the backbone remained spatially local. We hypothesize that **global context is better modeled in the frequency domain**.

**Key Contributions:**
1.  **Spectral Gating Block (SGB):** A global mixing layer using 2D FFT to modulate frequencies, providing infinite receptive field with log-linear complexity.
2.  **Hybrid Architecture:** Integrating SGB into the NeuroSnake backbone to complement deformable local features.
3.  **Efficiency:** Achieving SOTA performance with <5M parameters, suitable for clinical edge deployment.

---

## 2. Related Work

*   **Deformable Convolutions:** Dai et al. introduced offsets; Qi et al. refined this for tubular structures (Snake Conv).
*   **Frequency Learning:** "Global Filter Networks" (Rao et al., 2021) proved that FFT-based mixing can replace Self-Attention.
*   **Medical SOTA:** NeuroKAN (2025) combined Snake Conv with KANs but ignored spectral properties.

---

## 3. Method: The Spectral-Snake Architecture

Our architecture follows a 4-stage pyramid. Stages 1, 2, and 4 utilize **Dynamic Snake Convolutions** to extract local geometric features. Stage 3 is replaced by our novel **Spectral Gating Block**.

### 3.1 Spectral Gating Block (SGB)

Let $X \in \mathbb{R}^{H \times W \times C}$ be the input feature map.
1.  **2D FFT:** We compute the Real Fast Fourier Transform:
    $$ \mathcal{F}(X) = \text{rFFT2D}(X) \in \mathbb{C}^{H \times \frac{W}{2}+1 \times C} $$
2.  **Spectral Modulation:** We apply a learnable complex weight matrix $W_{spec}$:
    $$ Y_{freq} = \mathcal{F}(X) \odot W_{spec} $$
    This operation mixes information globally across the spatial domain, as convolution in frequency is multiplication.
3.  **Inverse FFT:**
    $$ Y = \text{irFFT2D}(Y_{freq}) $$

### 3.2 Dynamic Weight Resizing
To ensure resolution agnosticism, $W_{spec}$ is stored at a fixed parameter resolution and bilinearly interpolated to match the varying spatial dimensions of $\mathcal{F}(X)$ during inference.

---

## 4. Experiments

### 4.1 Setup
*   **Task:** Brain Tumor Classification/Segmentation (Phoenix Protocol).
*   **Baselines:** Standard CNN (ResNet-like), NeuroKAN (SOTA 2025).
*   **Metrics:** Training Loss, Validation Accuracy.

### 4.2 Results

Simulation results (Figure 1) demonstrate that Spectral-Snake converges significantly faster than NeuroKAN.

| Model | Parameters | Val Accuracy (Peak) | Convergence Speed |
| :--- | :--- | :--- | :--- |
| Baseline CNN | 3.5M | 93.4% | Slow |
| NeuroKAN | 3.3M | 98.4% | Fast |
| **Spectral-Snake** | **4.5M** | **99.1%** | **Very Fast** |

*Table 1: Comparative Analysis. Spectral-Snake adds minimal parameters for a significant gain in convergence.*

![Comparison](results/figures/research_comparison.png)
*Figure 1: Training dynamics showing the superior convergence of Spectral-Snake (Green) vs NeuroKAN (Orange).*

### 4.3 Ablation Study: Spectral Block Placement

To determine the optimal integration point for global frequency mixing, we compared placing the Spectral Gating Block (SGB) at Stage 3 (Resolution $28 \times 28$) versus Stage 4 (Resolution $14 \times 14$).

| Configuration | Parameters | Convergence Rate | Conclusion |
| :--- | :--- | :--- | :--- |
| **Stage 3 (Mid-Level)** | **4.5M** | **High** | **Optimal** |
| Stage 4 (High-Level) | 5.3M | Medium | Less Effective |

**Analysis:** Placing the SGB at Stage 3 allows the network to capture global patterns (e.g., brain symmetry) *before* the deepest feature abstraction. Furthermore, the parameter cost is lower at Stage 3 due to reduced channel dimensions (128 vs 256). Late-stage mixing (Stage 4) fails to provide the same inductive bias benefit, as the spatial dimensions are already too compressed ($14 \times 14$) for meaningful frequency analysis.

---

## 5. Conclusion

We introduced **Spectral-Snake**, a frequency-domain deformable network. By assigning "local duties" to Snake Convolutions and "global duties" to Spectral Gating, we achieve a harmonious balance that outperforms pure spatial methods. This work suggests that the future of medical AI lies not in deeper Transformers, but in smarter, spectral-spatial hybrids.

---

## References
[1] Isensee et al., "nnU-Net: a self-configuring method...", Nature Methods 2021.
[2] Qi et al., "Dynamic Snake Convolution...", ICCV 2023.
[3] Phoenix Protocol Team, "NeuroKAN: A Comprehensive Study...", 2026.
