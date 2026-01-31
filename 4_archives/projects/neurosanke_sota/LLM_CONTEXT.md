# LLM Context: The Phoenix Protocol (v2.0)

**Role**: You are a Senior Research Engineer and Systems Architect analyzing the "Phoenix Protocol" codebase.
**Objective**: Understand the *exact* implementation details, architectural decisions, and data flows to assist in future development, debugging, or optimization.

---

## 1. System Identity: "NeuroKAN"

The system is a **Hybrid Neuro-Oncology Classifier**.
*   **Marketing Name**: NeuroKAN
*   **True Architecture**: `DynamicSnakeConv` Backbone + `FastKAN` (RBF) Classification Head.
*   **Task**: Binary/Multi-class Classification of Brain Tumors from MRI (e.g., Glioma, Meningioma, Pituitary).

## 2. Codebase Topology

### 2.1 Core Model (`models/neurokan_model.py`)
*   **Input**: `(224, 224, 3)` (Configurable in `config.py`)
*   **Backbone**: 4-Stage Hierarchical Feature Extractor.
    *   **Stem**: Standard Conv2D (32 filters).
    *   **Stages 1-4**: `SnakeConvBlock` $\to$ `Attention` (Coordinate or SEVector) $\to$ `MaxPooling`.
    *   **Filters**: [64, 128, 256, 512].
*   **Neck**: `GlobalAveragePooling2D`. Flattens 3D feature maps to 1D vectors. **CRITICAL**: This destroys spatial topology before the KAN head.
*   **Head**: `KANDense` Layers (FastKAN implementation).
    *   Structure: `KAN(256)` $\to$ `Dropout` $\to$ `KAN(128)` $\to$ `Dropout` $\to$ `Softmax`.

### 2.2 The KAN Layer (`models/kan_layer.py`)
*   **Class**: `KANDense`
*   **Type**: **FastKAN** (Radial Basis Function approximation).
*   **Math**:
    $$ y = \text{SiLU}(x W_{base}) + \sum_{i} w_i \cdot \text{RBF}_i(x) $$
    $$ \text{RBF}(x) = \exp(-5.0 \cdot (x - \mu)^2) $$
*   **Key Variables**:
    *   `base_weight`: Standard MLP weights.
    *   `spline_weight`: Weights for the RBF basis functions.
    *   `grid`: Fixed centers for RBFs (linear space [-1, 1]).
*   **Constraint**: Inputs must be normalized to $[-1, 1]$ (or $[0, 1]$ mapped) for the RBFs to activate correctly.

### 2.3 The Snake Convolution (`models/dynamic_snake_conv.py`)
*   **Class**: `DynamicSnakeConv2D`
*   **Mechanism**: Deformable convolution that constrains offsets to x-axis or y-axis to simulate "snaking" tubular structures.
*   **Purpose**: Capture irregular tumor boundaries (spiculations) that standard $3\times3$ kernels miss.

### 2.4 Optimization (`src/phoenix_optimizer.py`)
*   **Optimizer**: **Adan** (Adaptive Nesterov Momentum).
    *   State: 1st moment ($m$), 2nd moment ($v$), 3rd moment ($n$).
    *   Stability: Superior for non-convex medical loss landscapes compared to Adam.
*   **Loss Function**: `CompoundLoss`.
    *   $L = \lambda_1 L_{Focal} + \lambda_2 L_{LogCoshDice} + \lambda_3 L_{Boundary}$
    *   Note: `BoundaryLoss` is typically for segmentation. Its use in classification (via auxiliary output or feature regularization) is implied but needs verification in `src/train_phoenix.py`.

## 3. Data Flow & Constraints

1.  **Preprocessing** (`src/clinical_preprocessing.py`):
    *   Images are **NOT** just resized.
    *   **Skull Stripping**: OpenCV morphological ops.
    *   **Bias Correction**: Low-pass filtering (simulated N4).
    *   **Normalization**: Z-score (standardization) per image.

2.  **Training** (`src/train_phoenix.py`):
    *   **Deduplication**: `src/data_deduplication.py` uses pHash to remove test-set leaks (Threshold $\le 5$).
    *   **Augmentation**: Physics-informed (Elastic deformation, Rician noise) to simulate MRI artifacts.

## 4. Key Implementation Limitations (The "Real" Context)

1.  **Input Size Sensitivity**: Snake Convolutions learn offsets based on feature map size. Changing input resolution (e.g., $224 \to 512$) might require retraining offset layers.
2.  **Global Pooling Bottleneck**: The transition from Snake Backbone (Spatial) to KAN Head (Spectral) goes through a standard Global Average Pool. This discards the "functional" spatial information that a true *FunKAN* would preserve.
3.  **RBF vs B-Spline**: The `KANDense` layer approximates B-splines with Gaussians (RBF). This is faster and more stable on GPUs but mathematically distinct from the original Kolmogorov-Arnold theorem.

## 5. Usage for LLM Agents

*   **When debugging accuracy**: Check `src/clinical_preprocessing.py`. Medical images are sensitive to intensity scaling.
*   **When debugging OOM**: Reduce `grid_size` in `KANDense` or `batch_size`. Adan uses 3x memory for optimizer states compared to SGD.
*   **When extending**: If adding Segmentation, attach a decoder *before* the `GlobalAveragePooling` layer in `NeuroKANModel`.
