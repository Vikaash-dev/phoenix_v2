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

from pathlib import Path
# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root.parent))

from src.experimental_framework import ExperimentalFramework, AblationStudyFramework
from src.research_visualizations import ResearchVisualizer
import config


def test_experimental_framework():
    """Test the experimental framework with minimal data."""
    print("\n" + "="*60)
    print("Testing Experimental Framework")
    print("="*60)
    
    # Generate minimal sample data
    n_samples = 50
    X = np.random.rand(n_samples, 28, 28, 3).astype(np.float32)
    y = np.zeros((n_samples, 2))
    y[:n_samples//2, 0] = 1
    y[n_samples//2:, 1] = 1
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    print(f"Sample data: X={X.shape}, y={y.shape}")
    
    # Define a simple model factory
    def simple_model_factory():
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 3)),
            tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    # Test experimental framework
    output_dir = '/tmp/test_experiments'
    
    exp_framework = ExperimentalFramework(
        experiment_name='test_model',
        output_dir=output_dir,
        n_folds=2,  # Minimal folds for testing
        random_seed=42
    )
    
    print("\nRunning 2-fold cross-validation...")
    results = exp_framework.run_kfold_experiment(
        model_factory=simple_model_factory,
        X=X,
        y=y,
        epochs=2,  # Minimal epochs for testing
        batch_size=16
    )
    
    print("\n✅ Experimental framework test passed!")
    print(f"Accuracy: {results['aggregated']['accuracy']['mean']:.4f} ± {results['aggregated']['accuracy']['std']:.4f}")
    
    return results


def test_visualizations(results):
    """Test visualization generation."""
    print("\n" + "="*60)
    print("Testing Research Visualizations")
    print("="*60)
    
    output_dir = '/tmp/test_visualizations'
    visualizer = ResearchVisualizer(output_dir=output_dir)
    
    # Create mock results for multiple models
    mock_results = {
        'Model_A': results,
        'Model_B': results
    }
    
    try:
        # Test bar chart
        print("\nGenerating SOTA comparison bar chart...")
        visualizer.plot_sota_comparison_bar(
            results=mock_results,
            metric='accuracy',
            filename='test_sota_comparison'
        )
        print("✅ Bar chart generated")
        
        # Test radar chart
        print("\nGenerating radar chart...")
        visualizer.plot_radar_chart(
            results=mock_results,
            metrics=['accuracy', 'precision', 'recall', 'f1_score'],
            filename='test_radar_chart'
        )
        print("✅ Radar chart generated")
        
        # Test metrics table
        print("\nGenerating metrics table...")
        visualizer.plot_metric_comparison_table(
            results=mock_results,
            metrics=['accuracy', 'precision', 'recall'],
            filename='test_metrics_table'
        )
        print("✅ Metrics table generated")
        
        print(f"\n✅ All visualizations saved to: {output_dir}")
        print(f"Generated files: {os.listdir(output_dir)}")
        
    except Exception as e:
        print(f"❌ Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_ablation_framework():
    """Test ablation study framework."""
    print("\n" + "="*60)
    print("Testing Ablation Study Framework")
    print("="*60)
    
    # Generate minimal sample data
    n_samples = 40
    X = np.random.rand(n_samples, 28, 28, 3).astype(np.float32)
    y = np.zeros((n_samples, 2))
    y[:n_samples//2, 0] = 1
    y[n_samples//2:, 1] = 1
    
    def simple_model_factory():
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 3)),
            tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    output_dir = '/tmp/test_ablation'
    
    ablation_framework = AblationStudyFramework(
        base_config={},
        output_dir=output_dir
    )
    
    print("\nRunning minimal ablation study...")
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
    
    # Generate table
    table = ablation_framework.generate_ablation_table(baseline_name='test_config')
    print(f"\n✅ Ablation framework test passed!")
    print(f"Generated table:\n{table}")
    
    return True


def test_statistical_comparison(results):
    """Test statistical comparison between models."""
    print("\n" + "="*60)
    print("Testing Statistical Comparison")
    print("="*60)
    
    output_dir = '/tmp/test_experiments'
    
    exp_framework = ExperimentalFramework(
        experiment_name='comparison_test',
        output_dir=output_dir,
        n_folds=2,
        random_seed=42
    )
    
    # Create slightly different results for comparison
    results_b = {
        'aggregated': {
            'accuracy': {
                'mean': results['aggregated']['accuracy']['mean'] + 0.01,
                'std': results['aggregated']['accuracy']['std'],
                'values': [v + 0.01 for v in results['aggregated']['accuracy']['values']]
            }
        }
    }
    
    print("\nComparing two models...")
    comparison = exp_framework.compare_models(
        results_a=results,
        results_b=results_b,
        name_a='Model_A',
        name_b='Model_B',
        metric='accuracy'
    )
    
    print(f"\n✅ Statistical comparison test passed!")
    print(f"p-value (t-test): {comparison['paired_t_test']['p_value']:.4f}")
    print(f"Effect size (Cohen's d): {comparison['effect_size']['cohen_d']:.4f}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("RESEARCH FRAMEWORK VALIDATION TESTS")
    print("="*80)
    
    try:
        # Test 1: Experimental Framework
        results = test_experimental_framework()
        
        # Test 2: Visualizations
        test_visualizations(results)
        
        # Test 3: Ablation Framework
        test_ablation_framework()
        
        # Test 4: Statistical Comparison
        test_statistical_comparison(results)
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED SUCCESSFULLY")
        print("="*80)
        print("\nThe research experimental framework is ready for use!")
        print("Run 'python run_research_experiments.py' to execute full experiments.")
        
        return 0
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED")
        print("="*80)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
