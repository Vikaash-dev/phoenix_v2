# Reflection and Refinement: Phoenix Mamba V2 Migration

## 1. PARA Structure Adoption
The repository has been successfully reorganized into the **PARA** framework (Projects, Areas, Resources, Archives) to ensure long-term scalability and maintainability.
- **Projects**: Active development (`1_projects/phoenix_mamba_v2`) isolates the experimental V2 architecture.
- **Areas**: Long-term responsibilities (`2_areas/main_app`) house the stable legacy code.
- **Resources**: Shared documentation and assets (`3_resources`) provide a centralized knowledge base.
- **Archives**: Deprecated or superseded code (`4_archives`) keeps the workspace clean.

## 2. Adapters & Model Architecture
The core architecture has been modernized with **Mamba-based adapters** and specialized components:
- **PhoenixMambaV2**: A hierarchical Selective State-Space Model (SSM) replacing pure CNNs.
- **S6 Layer Integration**: Incorporated `S6Layer` for efficient long-range dependency modeling, handling 2D medical imagery via flattening/scanning sequences.
- **2.5D Aggregation**: `SliceAggregator25D` adapts 2D backbones to process volumetric data by aggregating adjacent slices.
- **Heterogeneity Attention**: Specialized attention mechanism added to the bottleneck to capture tumor heterogeneity.

## 3. Legacy Integration
A careful migration strategy was employed to transition from the legacy `main_app` to the new `phoenix_mamba_v2` project:
- **Component Extraction**: Critical logic (preprocessing, evaluation metrics) was extracted from legacy scripts and modularized into `src/`.
- **Pipeline Consolidation**: Disparate training and testing scripts were unified into a single CLI tool (`scripts/run_pipeline.py`) managed by `uv`.
- **Backward Compatibility**: Legacy models and weights are preserved in `2_areas` while the new experimentation happens in `1_projects`.

## 4. Next Steps
- **PipelineContext**: Implement robust data loading to replace the current placeholder in the training loop.
- **Full Training Run**: Execute a complete training cycle on the Kaggle dataset.
- **Evaluation**: Compare PhoenixMambaV2 performance against the legacy benchmarks.
