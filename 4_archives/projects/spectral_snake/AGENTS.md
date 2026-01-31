# SESSION 7226 - GRAND BENCHMARK KNOWLEDGE BASE

**Context**: Most comprehensive architecture exploration branch. Contains 5 evolutionary model variants and ablation tooling.

## OVERVIEW

This session implements the "AI Scientist" evolutionary discovery process, exploring TTT-KAN, Hyper-Liquid, Spectral Gating, and Liquid Neural Networks.

## STRUCTURE

```text
jules_session_7226.../
├── src/models/           # 10 model variants (KAN, Liquid, Hyper, Spectral)
├── research_artifacts/   # Benchmarks, peer reviews, iteration logs
├── tools/                # Simulation generators, plotting, ablation runners
└── tests/                # Verification scripts for each architecture
```

## WHERE TO LOOK

| Task | File | Notes |
| :--- | :--- | :--- |
| **TTT-KAN Model** | `src/models/ttt_kan.py` | Test-Time Training + KAN |
| **Hyper-Liquid** | `src/models/hyper_liquid.py` | Hypernetwork + Liquid ODE |
| **Spectral Gating** | `src/models/neuro_mamba_spectral.py` | FFT-based global context |
| **KAN Layer** | `src/models/kan_layer.py` | Base Kolmogorov-Arnold implementation |
| **Liquid Layer** | `src/models/liquid_layer.py` | LTC ODE dynamics |
| **Grand Benchmark** | `research_artifacts/GRAND_SUMMARY.md` | Full comparison of all 5 branches |
| **Run Ablation** | `tools/run_grand_benchmark.py` | Execute comparative experiments |

## KEY INNOVATIONS

- **TTT-KAN**: Self-supervised reconstruction head for inference-time adaptation
- **Hyper-Liquid**: Spectral-normalized hypernetwork predicts ODE time-constants
- **Spectral Gating**: $O(N \log N)$ global mixing via FFT

## CONVENTIONS

- All models follow `NeuroSnake*Model.create_model()` factory pattern
- Simulated metrics stored as JSON in `research_artifacts/`
- Verification tests in `tests/verify_*.py`

## ANTI-PATTERNS

- **Simulated metrics only**: No real trained weights. Metrics are placeholders.
- **TensorFlow-specific**: Uses `tf.keras` patterns, not PyTorch.
