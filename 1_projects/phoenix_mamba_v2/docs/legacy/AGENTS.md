# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-25
**Scope:** Root Level (Index)

## OVERVIEW

AI-powered brain tumor detection system implementing "Phoenix Protocol" (NeuroSnake architecture + Dynamic Snake Convolutions).
Acts as a monorepo-style project with multiple versions (`v2`, `v3`) and a core research implementation (`src`).

## STRUCTURE

```text
brain-tumor-detection/
├── src/                    # CORE LOGIC - See src/AGENTS.md
├── models/                 # Neural Architecture definitions (CNN, NeuroSnake)
├── v2/                     # "SOTA Upgrade" version (Incomplete/WIP)
├── v3/                     # "Spectral-Snake" version (Minimal)
├── reproducibility/        # Research paper replication scripts
└── jules_session_*/        # ARCHIVED SNAPSHOTS - See SESSIONS.md
```

## WHERE TO LOOK

| Task | Location | Notes |
| :--- | :--- | :--- |
| **Core Logic & Training** | [`src/`](src/AGENTS.md) | **MAIN ACTIVE CODEBASE**. See `src/AGENTS.md`. |
| **Model Architectures** | `models/` | `cnn_model.py`, `neurosnake_model.py` |
| **Quick Start** | Root Scripts | `one_click_train_test.py`, `setup_data.py` |
| **Research Paper** | `Research_Paper_*.md` | Theoretical basis for implementation |

## GLOBAL CONVENTIONS

- **Medical Focus**: Variable names and logic assume MRI inputs (T1, T2, FLAIR).
- **Configuration**: Centralized in `config.py` (root) or implicitly handled in `src`.
- **Dual-Mode**: Code often supports "Standard" vs "Phoenix" (Advanced) modes.

## COMMANDS (Common)

```bash
# Data Setup
python setup_data.py

# Quick Train (NeuroSnake)
python one_click_train_test.py --mode train --model-type neurosnake_ca

# Run Comparison
python src/compare_analysis.py
```

## NAVIGATION

- **For Deep Dive into Logic**: Go to [src/AGENTS.md](src/AGENTS.md)
- **For Model Definitions**: Check `models/` directory directly.
