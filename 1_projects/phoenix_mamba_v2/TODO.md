# Action Plan: Phoenix Mamba V2 Refinement

Based on the automated LLM analysis of the codebase, the following tasks have been identified to improve performance, security, and reproducibility.

## 🔴 High Priority (Security & Stability)

- [ ] **Input Validation (`src/data/processors.py`)**
    - [ ] Add checks for image dimensions to prevent OOM/DoS attacks.
    - [ ] Add checks for `NaN`/`Inf` values in floating-point inputs.
    - [ ] Validate data types explicitly before processing.

- [ ] **Dependency Pinning (`pyproject.toml`)**
    - [ ] Pin `tensorflow` to a specific stable version (e.g., `2.16.1`) to ensure reproducibility.
    - [ ] Lock all dependencies with `uv pip compile` to generate a `requirements.lock`.

## 🟡 Medium Priority (Performance)

- [ ] **Optimize S6 Layer (`src/models/ssm.py`)**
    - [ ] Replace `tf.scan` (symbolic loop) with a custom TensorFlow CUDA kernel or a more efficient vectorization strategy.
    - [ ] *Impact*: `tf.scan` is significantly slower than standard RNN/LSTM implementations and orders of magnitude slower than optimized Mamba kernels.

- [ ] **Configuration Management**
    - [ ] Refactor `PhoenixMambaV2` to accept a config object instead of hardcoded filter sizes (`64`, `128`...).
    - [ ] Centralize all hyperparameters (learning rate, dropout, shapes) into `config/training_config.yaml`.

## 🟢 Low Priority (Features & Docs)

- [ ] **Explainability**
    - [ ] Adapt Grad-CAM or implement "Mamba-CAM" to visualize the contributions of the S6 state-space features.

- [ ] **3D Validation**
    - [ ] Benchmark `SliceAggregator25D` against true 3D ConvNets on the BraTS dataset.

- [ ] **Type Hinting**
    - [ ] Add full type hints to `src/models/phoenix.py` and `src/models/ssm.py`.
