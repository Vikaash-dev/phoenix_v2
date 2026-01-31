# PARA Refactoring Plan for Brain Tumor Detection Project

This document outlines the plan to refactor the project repository into the PARA (Projects, Areas, Resources, Archives) structure.

## 1. Analysis of Architectures and Approaches

The repository contains a multitude of deep learning architectures and research ideas, often fragmented across different directories. The key concepts identified are:

* **Core Architectures:**
  * **NeuroSnake:** A primary architecture, appearing in several variations.
  * **CNN:** Standard convolutional neural networks.
  * **ViT (Vision Transformer):** Used in conjunction with NeuroSnake.
* **Experimental Architectures:**
  * **Spectral-Snake:** An evolution of NeuroSnake, found in the `v3` directory.
  * **KAN (Kolmogorov-Arnold Networks):** Integrated with NeuroSnake.
  * **Liquid Networks:** Explored as a potential enhancement.
  * **HyperNetworks:** Used to dynamically generate model weights.
* **Techniques and Proposals:**
  * **Coordinate Attention:** A specific type of attention mechanism.
  * **Physics-Informed Augmentation:** A data augmentation strategy.
  * **Test-Time Training (TTT):** An adaptation technique.
  * **Disentangled Representations:** A research proposal.

## 2. PARA Categorization

Here is the breakdown of the existing file structure into the PARA categories:

### Projects

* `v2/`: SOTA Upgrade (incomplete project).
* `v3/`: Spectral-Snake version (minimal project).
* `reproducibility/`: Scripts to reproduce research paper results.
* `camber_sample/`: A small, self-contained CNN training example.
* `one_click_train_test.py`, `setup_data.py`: Scripts that represent the main project entry points.

### Areas

* `src/`: The core logic and active codebase for the main project.
* `models/`: Definitions of the primary neural network architectures.
* `tests/`: Tests for the core project.
* `config.py`: The main configuration file.
* `requirements.txt`, `pyproject.toml`, `uv.lock`: Project dependencies and environment.

### Resources

* All markdown files (`.md`): These are documentation, research papers, analysis reports, and proposals.
* `tavily_*.csv`: Results from search queries.
* `jules_session_*/research_artifacts/`: Figures, papers, and data from research iterations.
* `examples.py`: Code examples.
* `scripts/`: General-purpose scripts.

### Archives

* `jules_session_*`: These appear to be snapshots of research and development sessions and should be archived.

## 3. Proposed New Directory Structure

```
.
├── 1_projects/
│   ├── neurosnake_sota/
│   ├── spectral_snake/
│   └── reproducibility/
├── 2_areas/
│   ├── main_app/
│   │   ├── src/
│   │   ├── models/
│   │   └── tests/
│   └── research/
│       ├── kan/
│       ├── liquid_networks/
│       ├── hypernetworks/
│       └── ttt/
├── 3_resources/
│   ├── documentation/
│   ├── papers/
│   ├── analysis_reports/
│   ├── data/
│   └── scripts/
└── 4_archives/
    ├── jules_sessions/
    └── camber_sample/
```

## 4. Migration Plan

This plan will be executed by a separate agent.

### Phase 1: Create New Directory Structure

1. Create `1_projects/`, `2_areas/`, `3_resources/`, `4_archives/`.
2. Create subdirectories within each of the main PARA directories as specified in the proposed structure.

### Phase 2: Move Files and Directories

* **Archives:**
  * Move all `jules_session_*` directories into `4_archives/jules_sessions/`.
  * Move `camber_sample/` into `4_archives/`.

* **Projects:**
  * Move the contents of `v2/` to `1_projects/neurosanke_sota/`.
  * Move the contents of `v3/` to `1_projects/spectral_snake/`.
  * Move `reproducibility/` to `1_projects/`.

* **Areas:**
  * Move `src/` to `2_areas/main_app/`.
  * Move `models/` to `2_areas/main_app/`.
  * Move `tests/` to `2_areas/main_app/`.
  * Create `2_areas/research/` and the subdirectories for `kan`, `liquid_networks`, etc. The code for these will be extracted from the archived `jules_sessions` later if needed.

* **Resources:**
  * Move all `.md` files into `3_resources/documentation/`, `3_resources/papers/`, or `3_resources/analysis_reports/` based on their content.
  * Move `tavily_*.csv` files to `3_resources/data/`.
  * Move `scripts/` to `3_resources/`.

### Phase 3: Root Directory Cleanup

* Delete the now-empty directories (`v2`, `v3`).
* Verify that the root directory is clean and organized according to the new structure.

This migration will consolidate the fragmented project into a clear, maintainable structure, separating active projects, ongoing areas of responsibility, valuable resources, and historical archives.
