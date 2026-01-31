# Phoenix Protocol Repository

This repository follows the **PARA** method (Projects, Areas, Resources, Archives) for organization.

## 📂 Structure

### 1_projects/
Active development projects.
*   **[phoenix_mamba_v2](1_projects/phoenix_mamba_v2/)**: The current main focus. Next-generation brain tumor detection using Mamba and NeuroSnake architectures.
    *   **Docs**: [Documentation](1_projects/phoenix_mamba_v2/docs/)
    *   **Source**: [Source Code](1_projects/phoenix_mamba_v2/src/)
    *   **Run**: `python 1_projects/phoenix_mamba_v2/scripts/run_pipeline.py`

### 3_resources/
Shared tools and documentation.
*   **claude-skills-mcp**: Reusable skills for the agent.

### 4_archives/
Inactive, completed, or superseded work.
*   **neurosanke_sota**: Legacy SOTA implementation.
*   **spectral_snake**: Legacy spectral analysis.
*   **kaggle**: Kaggle-specific integration files.
*   **legacy_systems**: Old agent systems and scripts.

## 🚀 Quick Start (Phoenix Mamba V2)

1.  **Install Dependencies:**
    ```bash
    uv pip install -e 1_projects/phoenix_mamba_v2/
    ```

2.  **Run Pipeline Demo:**
    ```bash
    python 1_projects/phoenix_mamba_v2/scripts/run_pipeline.py --mode demo
    ```

3.  **Train Model:**
    ```bash
    python 1_projects/phoenix_mamba_v2/scripts/run_pipeline.py --mode train --data-dir /path/to/dataset
    ```
