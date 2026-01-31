# Analysis of V3: Spectral-Snake & Sukuna Agent

**Date:** 2026-01-30
**Subject:** Technical Analysis of "Phoenix Protocol v3" (Spectral-Snake) and "Sukuna Agent"

## 1. System Overview
The "v3" codebase (`start-v3.sh`, `scientific_agent_system/v3/`, `1_projects/spectral_snake/`) represents a distinct development branch focused on two key technologies:
1.  **Spectral-Snake Architecture:** A hybrid deep learning model combining Dynamic Snake Convolutions with FFT-based Spectral Gating.
2.  **Sukuna Agent (v3):** An autonomous "Lead Data Scientist" agent framework designed for rigorous falsification and self-evolving research.

This system appears to be a parallel or predecessor branch to the "Phoenix Mamba v2" architecture (which uses State-Space Models).

## 2. Model Architecture: Spectral-Snake
**Definition:** `1_projects/spectral_snake/src/spectral_snake_conv.py`

The model is designed to tackle the "Local vs Global" trade-off in medical imaging (Brain Tumor Detection) without the computational cost of Transformers ($O(N^2)$).

### Key Components
*   **Dynamic Snake Convolutions (`SnakeConvBlock`):**
    *   Used in stages 2-4 (Resolutions: 56x56 -> 14x14).
    *   Specialized kernels that adapt to tubular structures (vessels, nerves) by "slithering" along features.
*   **Spectral Gating Block (`SpectralGatingBlock`):**
    *   Used in the deepest stage (Stage 5, 7x7 resolution).
    *   **Mechanism:**
        1.  **FFT:** Converts spatial features to the frequency domain (`tf.signal.rfft2d`).
        2.  **Gating:** Multiplies frequencies by a learnable complex weight matrix (`complex_weight_real` + `1j * complex_weight_imag`).
        3.  **IFFT:** Converts back to spatial domain.
    *   **Benefit:** Achieves a global receptive field (all pixels affect all pixels via frequency mixing) with $O(N \log N)$ complexity.

### Comparison to State-of-the-Art
*   **vs. ViT:** Lower complexity ($N \log N$ vs $N^2$). Better inductive bias for continuous structures (via Snake Conv).
*   **vs. CNN:** Explicit global context via Spectral Gating, which standard CNNs lack.

## 3. Agent System: The Sukuna Protocol
**Location:** `scientific_agent_system/v3/`

This is a sophisticated, autonomous agent framework that differs significantly from standard "Chat with Code" assistants.

### Core Philosophy: "The Triad of Analysis"
Defined in `prompts.py`, the agent is programmed to be **adversarial** to its own findings:
1.  **Reverse Analysis:** Deconstruct claims to raw hyperparameters/hardware.
2.  **Cross Analysis:** Triangulate facts against independent sources (Code vs Paper).
3.  **Negative Analysis (Falsification):** Actively search for failure cases and counter-examples.

### Architecture
*   **State-Based Memory (`agent.md`):** The agent persists long-term memory in a markdown file. It reads this "Brain" at the start of every turn and updates it at the end.
*   **Docker Runtime (`aci.py` / `runtime.py`):**
    *   Uses a `DockerRuntime` class to execute code in isolated containers (`agent_v3_...`).
    *   Wraps `subprocess` to manage Docker CLI.
    *   Uses `heredoc` syntax for safe file writing.
*   **Regex-Based Control (`agent_loop.py`):**
    *   Parses actions like `ACTION: run_shell(...)` from the LLM's raw text stream.

## 4. Comparison with Phoenix Mamba (v2)

| Feature | Spectral-Snake (v3) | Phoenix Mamba (v2) |
| :--- | :--- | :--- |
| **Core Tech** | Snake Conv + FFT (Spectral) | State-Space Models (Mamba/S6) |
| **Math Basis** | Fourier Transforms ($N \log N$) | Linear Recurrent / ODEs ($N$) |
| **Focus** | 2.5D Global Context & Geometry | Volumetric (3D) Efficiency |
| **Agent** | Sukuna v3 (Adversarial Researcher) | (Not explicitly defined in v2 spec) |

## 5. Status
*   **Code Quality:** High. Clean separation of concerns in the Agent system.
*   **Dependencies:** Requires `tensorflow`, `numpy`.
*   **Operational:** `start-v3.sh` provides a direct entry point.

