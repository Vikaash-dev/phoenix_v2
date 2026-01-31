# Research Experimental Framework - Quick Start Guide

This guide explains how to use the newly added research experimental validation framework.

## Overview

The research framework adds rigorous experimental validation capabilities to the brain tumor detection project, making it publication-ready. It includes:

- ✅ K-fold cross-validation with statistical testing
- ✅ SOTA baseline comparisons (ResNet50, EfficientNetB0, VGG16)
- ✅ Ablation studies
- ✅ Publication-quality visualizations (300 DPI)
- ✅ Complete reproducibility package

## Quick Start

### 1. Basic Usage

Run the complete research experiment pipeline:

```bash
python run_research_experiments.py
```

This will:
1. Load and prepare data (or generate sample data for testing)
2. Run 5-fold cross-validation for all SOTA models
3. Conduct ablation studies
4. Generate all figures and tables
5. Create a comprehensive research report

### 2. Using the Experimental Framework

```python
from src.experimental_framework import ExperimentalFramework

# Create experiment
experiment = ExperimentalFramework(
    experiment_name='my_model',
    output_dir='./my_results',
    n_folds=5,
    random_seed=42
)

# Define model factory
def my_model_factory():
    model = create_my_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Run k-fold cross-validation
results = experiment.run_kfold_experiment(
    model_factory=my_model_factory,
    X=X_train,
    y=y_train,
    epochs=50,
    batch_size=32
)

# Access results
print(f"Accuracy: {results['aggregated']['accuracy']['mean']:.4f}")
print(f"95% CI: [{results['aggregated']['accuracy']['ci_95_lower']:.4f}, "
      f"{results['aggregated']['accuracy']['ci_95_upper']:.4f}]")
```

### 3. Statistical Comparison

```python
# Compare two models
comparison = experiment.compare_models(
    results_a=model_a_results,
    results_b=model_b_results,
    name_a='NeuroSnake',
    name_b='ResNet50',
    metric='accuracy'
)

print(f"p-value: {comparison['paired_t_test']['p_value']:.4f}")
print(f"Significant: {comparison['paired_t_test']['significant']}")
```

### 4. Ablation Studies

```python
from src.experimental_framework import AblationStudyFramework

ablation = AblationStudyFramework(
    base_config={},
    output_dir='./ablation_results'
)

# Run ablation
ablation.run_ablation(
    ablation_name='without_attention',
    config_modifications={'use_attention': False},
    model_factory=lambda: create_model(use_attention=False),
    X=X_train,
    y=y_train
)

# Generate comparison table
table = ablation.generate_ablation_table()
```

### 5. Visualization

```python
from src.research_visualizations import ResearchVisualizer

viz = ResearchVisualizer(output_dir='./figures')

# SOTA comparison bar chart
viz.plot_sota_comparison_bar(
    results=all_model_results,
    metric='accuracy',
    filename='sota_comparison'
)

# Radar chart
viz.plot_radar_chart(
    results=all_model_results,
    metrics=['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
)

# Ablation heatmap
viz.plot_ablation_heatmap(
    ablation_results=ablation_results
)
```

## Output Structure

After running experiments, you'll get:

```
research_results/
├── figures/                          # Publication-quality figures (300 DPI)
│   ├── figure1_sota_comparison_accuracy.png
│   ├── figure1_sota_comparison_accuracy.pdf
│   ├── figure2_radar_chart.png
│   ├── figure3_ablation_heatmap.png
│   └── FIGURES_SUMMARY.md
├── sota_comparison/                  # SOTA model results
│   ├── NeuroSnake_*/
│   │   ├── experiment.log
│   │   └── results.json
│   ├── ResNet50_*/
│   ├── ...
│   └── all_models_results.json
├── ablation_studies/                 # Ablation results
│   ├── ablation_results.json
│   └── ablation_table.md
├── tables/                           # Markdown tables
│   └── table1_sota_comparison.md
└── EXPERIMENT_REPORT.md              # Comprehensive report
```

## Customization

### Custom Models

Add your own model to the comparison:

```python
def create_my_custom_model():
    model = keras.Sequential([
        # Your architecture here
    ])
    model.compile(...)
    return model

# Add to model factories
model_factories = create_baseline_models()
model_factories['My_Custom_Model'] = create_my_custom_model
```

### Custom Metrics

The framework computes these metrics by default:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Sensitivity
- Specificity
- FPR (False Positive Rate)
- FNR (False Negative Rate)

All metrics are automatically computed from predictions.

### Adjusting Experiment Parameters

Edit `run_research_experiments.py`:

```python
# In main()
n_folds = 10  # More folds for smaller datasets
epochs = 100  # More epochs for production
batch_size = 16  # Smaller if GPU memory limited
```

## Reproducibility

All experiments are fully reproducible:

1. **Fixed Seeds**: All random seeds documented in `reproducibility/seeds.json`
2. **Deterministic Ops**: TensorFlow configured for deterministic execution
3. **Docker Support**: Complete environment in `reproducibility/Dockerfile`
4. **Config Tracking**: All hyperparameters in `reproducibility/training_config.json`

To reproduce results:

```bash
# Option 1: Docker
docker build -t neurosnake-research -f reproducibility/Dockerfile .
docker run --gpus all neurosnake-research

# Option 2: Conda
conda env create -f reproducibility/environment.yml
conda activate neurosnake-research
python run_research_experiments.py
```

## Testing

Validate the framework works correctly:

```bash
python test_research_framework.py
```

This runs minimal tests with synthetic data to verify all components.

## Integration with Existing Code

The framework integrates seamlessly:

```python
# Use existing model from models/
from models.neurosnake_model import create_neurosnake_model

experiment = ExperimentalFramework(...)
results = experiment.run_kfold_experiment(
    model_factory=lambda: create_neurosnake_model(use_mobilevit=True),
    X=X,
    y=y
)
```

## Common Issues

### GPU Out of Memory

Reduce batch size:
```python
batch_size = 16  # or 8
```

### Slow Training

- Use fewer epochs for testing: `epochs=5`
- Use fewer folds: `n_folds=3`
- Enable mixed precision (if supported):
```python
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

### Missing Dependencies

```bash
pip install -r requirements.txt
```

Required packages (already in requirements.txt):
- scipy >= 1.9.0
- seaborn >= 0.12.0

## Citation

If you use this framework in your research:

```bibtex
@software{neurosnake_framework,
  title={NeuroSnake Research Experimental Framework},
  author={...},
  year={2024},
  url={https://github.com/...}
}
```

## Support

- Documentation: See `RESEARCH_PAPER_FINAL.md` for methodology
- Reproducibility: See `reproducibility/README.md`
- Issues: Open a GitHub issue

## Next Steps

1. **Replace sample data** with your actual dataset
2. **Run full experiments** with production settings (epochs=50+)
3. **Fill in paper template** with generated results
4. **Submit for publication**

The framework is designed to generate all necessary data for academic publication!
