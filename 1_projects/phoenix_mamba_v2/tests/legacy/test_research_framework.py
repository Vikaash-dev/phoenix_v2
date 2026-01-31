"""
Test script for research experimental framework
Validates that all components work correctly with minimal data
"""

# Suppress TF warnings BEFORE imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import tensorflow as tf
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.legacy.experimental_framework import ExperimentalFramework, AblationStudyFramework
from src.legacy.research_visualizations import ResearchVisualizer
import src.legacy.config as config


@pytest.fixture
def sample_data():
    n_samples = 50
    X = np.random.rand(n_samples, 28, 28, 3).astype(np.float32)
    y = np.zeros((n_samples, 2))
    y[:n_samples//2, 0] = 1
    y[n_samples//2:, 1] = 1
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]

@pytest.fixture
def simple_model_factory():
    def _factory():
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 3)),
            tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    return _factory

def test_full_framework_flow(sample_data, simple_model_factory):
    """Test the experimental framework, visualizations, and statistical comparison in one flow."""
    print("\n" + "="*60)
    print("Testing Full Research Framework Flow")
    print("="*60)
    
    X, y = sample_data
    
    # 1. Experimental Framework
    output_dir = '/tmp/test_experiments'
    exp_framework = ExperimentalFramework(
        experiment_name='test_model',
        output_dir=output_dir,
        n_folds=2,
        random_seed=42
    )
    
    results = exp_framework.run_kfold_experiment(
        model_factory=simple_model_factory,
        X=X,
        y=y,
        epochs=2,
        batch_size=16
    )
    
    assert results is not None
    assert 'aggregated' in results
    
    # 2. Visualizations
    visualizer = ResearchVisualizer(output_dir=output_dir)
    mock_results = {'Model_A': results, 'Model_B': results}
    
    try:
        visualizer.plot_sota_comparison_bar(results=mock_results, metric='accuracy', filename='test_sota')
        visualizer.plot_radar_chart(results=mock_results, metrics=['accuracy'], filename='test_radar')
    except Exception as e:
        pytest.fail(f"Visualization failed: {str(e)}")

    # 3. Statistical Comparison
    results_b = {
        'aggregated': {
            'accuracy': {
                'mean': results['aggregated']['accuracy']['mean'] + 0.01,
                'std': results['aggregated']['accuracy']['std'],
                'values': [v + 0.01 for v in results['aggregated']['accuracy']['values']]
            }
        }
    }
    
    comparison = exp_framework.compare_models(
        results_a=results,
        results_b=results_b,
        name_a='Model_A',
        name_b='Model_B',
        metric='accuracy'
    )
    
    assert 'paired_t_test' in comparison

def test_ablation_framework(sample_data, simple_model_factory):
    """Test ablation study framework."""
    X, y = sample_data
    output_dir = '/tmp/test_ablation'
    
    ablation_framework = AblationStudyFramework(
        base_config={},
        output_dir=output_dir
    )
    
    ablation_framework.run_ablation(
        ablation_name='test_config',
        config_modifications={'dropout': 0.3},
        model_factory=simple_model_factory,
        X=X,
        y=y,
        n_folds=2,
        epochs=2,
        batch_size=16
    )
    
    table = ablation_framework.generate_ablation_table(baseline_name='test_config')
    assert table is not None
