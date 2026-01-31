# Phoenix Mamba V2

## Next-Generation Brain Tumor Detection
**Powered by Selective State-Space Models and Physics-Informed Augmentation**

### Overview
Phoenix Mamba V2 is a clinical-grade deep learning framework for brain tumor detection. It leverages:
1.  **NeuroSnake Architecture**: A hybrid model combining Dynamic Snake Convolutions (for geometric adaptability) with MobileViT-v2 (for global context).
2.  **Physics-Informed Augmentation**: Simulates real MRI artifacts (Rician noise, bias fields, ghosting) rather than generic image distortions.
3.  **Modular Pipeline**: Swappable "Clinical" vs "Fast" preprocessing strategies.

### Directory Structure
- `src/core`: Base interfaces and utilities.
- `src/data`: Preprocessing (Clinical/Fast) and Augmentation logic.
- `src/models`: NeuroSnake and Mamba architectures.
- `scripts/`: CLI entry points.
- `docs/`: Architecture diagrams and research papers.

### Quick Start

**Installation:**
```bash
pip install -e .
```

**Run Demo:**
```bash
python scripts/run_pipeline.py --mode demo
```

**Train:**
```bash
python scripts/run_pipeline.py --mode train --data-dir /path/to/mri/data
```
