# PHOENIX-MAMBA v2: Research Index

This directory contains the PARA-structured research for the next-generation Brain Tumor Detection system.

## 🔬 Innovation Streams

### [1. S6 Backbone (State-Space Models)](./s6_backbone/)
- **Hypothesis**: Replacing Multi-Head Attention with Selective SSMs will reduce inference time by 3-5x without losing diagnostic accuracy.
- **Key Reference**: Albert Gu and Tri Dao (2023), "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".

### [2. Volumetric Context (2.5D)](./volumetric_context/)
- **Hypothesis**: Temporal-style attention across 3 slices captures mass effect and infiltrative growth patterns missed by 2D-only models.
- **Key Reference**: "2.5D Convolutional Neural Networks for Medical Image Segmentation".

### [3. Contrastive Learning](./contrastive_learning/)
- **Hypothesis**: Self-supervised pretraining on unlabeled MRI datasets will improve domain generalization across different hospital scanner types.

### [4. Heterogeneity-Aware Attention](./heterogeneity_attention/)
- **Hypothesis**: Dedicated attention heads for NCR (Necrotic Core), ET (Enhancing Tumor), and ED (Edema) will improve sensitivity to subtle tumor margins.

### [5. Clinical Interpretability](./interpretability/)
- **Goal**: Built-in Grad-CAM and attention visualizations to enable "Trust-but-Verify" clinical adoption.

---
*Created on 2026-01-27 as part of PARA Project: PHOENIX-MAMBA v2*
