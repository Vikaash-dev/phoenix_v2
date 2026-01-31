"""
Complete Research Experiment Runner
Generates all data needed for paper publication.

This script:
1. Loads and prepares data
2. Runs SOTA comparison experiments (5-fold CV)
3. Runs ablation studies
4. Generates all publication-quality figures
5. Creates comparison tables
6. Generates a comprehensive research report
"""

import os
import sys
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, EfficientNetB0, VGG16
from typing import Dict, List, Callable, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.legacy.experimental_framework import ExperimentalFramework, AblationStudyFramework
from src.legacy.research_visualizations import ResearchVisualizer
from src.legacy.models.legacy.neurosnake_model import NeuroSnakeModel, create_neurosnake_model
from src.legacy.models.legacy.cnn_model import create_cnn_model


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ResearchExperiments')


def create_resnet50_classifier(
    input_shape=(224, 224, 3),
    num_classes=2,
    learning_rate=0.0001
):
    """
    Create ResNet50-based classifier for brain tumor detection.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model
    """
    # Load pre-trained ResNet50
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build classifier
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='ResNet50_Classifier')
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_efficientnet_classifier(
    input_shape=(224, 224, 3),
    num_classes=2,
    learning_rate=0.0001
):
    """
    Create EfficientNetB0-based classifier for brain tumor detection.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model
    """
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build classifier
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='EfficientNetB0_Classifier')
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_vgg16_classifier(
    input_shape=(224, 224, 3),
    num_classes=2,
    learning_rate=0.0001
):
    """
    Create VGG16-based classifier for brain tumor detection.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model
    """
    # Load pre-trained VGG16
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build classifier
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='VGG16_Classifier')
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_neurosnake_classifier(
    input_shape=(224, 224, 3),
    num_classes=2,
    use_mobilevit=True,
    dropout_rate=0.3,
    learning_rate=0.0001
):
    """
    Create NeuroSnake classifier.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        use_mobilevit: Whether to use MobileViT blocks
        dropout_rate: Dropout rate
        learning_rate: Learning rate
    
    Returns:
        Compiled Keras model
    """
    model = NeuroSnakeModel.create_model(
        input_shape=input_shape,
        num_classes=num_classes,
        use_mobilevit=use_mobilevit,
        dropout_rate=dropout_rate
    )
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_baseline_cnn_classifier(
    input_shape=(224, 224, 3),
    num_classes=2,
    learning_rate=0.0001
):
    """
    Create baseline CNN classifier.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        learning_rate: Learning rate
    
    Returns:
        Compiled Keras model
    """
    model = NeuroSnakeModel.create_baseline_comparison_model(
        input_shape=input_shape,
        num_classes=num_classes
    )
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_baseline_models() -> Dict[str, Callable]:
    """
    Create model factories for SOTA baselines.
    
    Returns:
        Dictionary of {model_name: model_factory_function}
    """
    return {
        'NeuroSnake': lambda: create_neurosnake_classifier(use_mobilevit=True),
        'ResNet50': create_resnet50_classifier,
        'EfficientNetB0': create_efficientnet_classifier,
        'VGG16': create_vgg16_classifier,
        'Baseline_CNN': create_baseline_cnn_classifier
    }


def run_sota_comparison(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: str,
    n_folds: int = 5,
    epochs: int = 50,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Run 5-fold CV for all SOTA models and save results.
    
    Args:
        X: Input data
        y: Labels (one-hot encoded)
        output_dir: Directory to save results
        n_folds: Number of folds
        epochs: Training epochs per fold
        batch_size: Batch size
    
    Returns:
        Dictionary of results for all models
    """
    logger.info("\n" + "="*80)
    logger.info("STARTING SOTA MODEL COMPARISON")
    logger.info("="*80)
    
    sota_dir = os.path.join(output_dir, 'sota_comparison')
    os.makedirs(sota_dir, exist_ok=True)
    
    model_factories = create_baseline_models()
    all_results = {}
    
    for model_name, model_factory in model_factories.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*80}")
        
        try:
            # Create experiment framework
            exp_framework = ExperimentalFramework(
                experiment_name=model_name,
                output_dir=sota_dir,
                n_folds=n_folds,
                random_seed=config.RANDOM_SEED
            )
            
            # Run k-fold cross-validation
            results = exp_framework.run_kfold_experiment(
                model_factory=model_factory,
                X=X,
                y=y,
                epochs=epochs,
                batch_size=batch_size
            )
            
            all_results[model_name] = results
            
            logger.info(f"\n✅ Completed: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save combined results
    combined_file = os.path.join(sota_dir, 'all_models_results.json')
    with open(combined_file, 'w') as f:
        # Extract only serializable data
        serializable_results = {}
        for model_name, result in all_results.items():
            serializable_results[model_name] = {
                'fold_results': result['fold_results'],
                'aggregated': result['aggregated']
            }
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"\n✅ All SOTA comparison results saved to: {combined_file}")
    
    return all_results


def run_ablation_studies(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: str,
    n_folds: int = 5,
    epochs: int = 50,
    batch_size: int = 32
) -> List[Dict[str, Any]]:
    """
    Run ablation studies to evaluate component contributions.
    
    Ablation configurations:
    - full_model (baseline)
    - without_mobilevit
    - dropout_0.1
    - dropout_0.5
    
    Args:
        X: Input data
        y: Labels
        output_dir: Directory to save results
        n_folds: Number of folds
        epochs: Training epochs
        batch_size: Batch size
    
    Returns:
        List of ablation results
    """
    logger.info("\n" + "="*80)
    logger.info("STARTING ABLATION STUDIES")
    logger.info("="*80)
    
    ablation_dir = os.path.join(output_dir, 'ablation_studies')
    os.makedirs(ablation_dir, exist_ok=True)
    
    # Initialize ablation framework
    ablation_framework = AblationStudyFramework(
        base_config={},
        output_dir=ablation_dir
    )
    
    # Define ablation configurations
    ablations = [
        {
            'name': 'full_model',
            'modifications': {'use_mobilevit': True, 'dropout_rate': 0.3},
            'factory': lambda: create_neurosnake_classifier(
                use_mobilevit=True, dropout_rate=0.3
            )
        },
        {
            'name': 'without_mobilevit',
            'modifications': {'use_mobilevit': False, 'dropout_rate': 0.3},
            'factory': lambda: create_neurosnake_classifier(
                use_mobilevit=False, dropout_rate=0.3
            )
        },
        {
            'name': 'dropout_0_1',
            'modifications': {'use_mobilevit': True, 'dropout_rate': 0.1},
            'factory': lambda: create_neurosnake_classifier(
                use_mobilevit=True, dropout_rate=0.1
            )
        },
        {
            'name': 'dropout_0_5',
            'modifications': {'use_mobilevit': True, 'dropout_rate': 0.5},
            'factory': lambda: create_neurosnake_classifier(
                use_mobilevit=True, dropout_rate=0.5
            )
        }
    ]
    
    # Run each ablation
    for ablation_config in ablations:
        logger.info(f"\nRunning ablation: {ablation_config['name']}")
        
        try:
            ablation_framework.run_ablation(
                ablation_name=ablation_config['name'],
                config_modifications=ablation_config['modifications'],
                model_factory=ablation_config['factory'],
                X=X,
                y=y,
                n_folds=n_folds,
                epochs=epochs,
                batch_size=batch_size
            )
            
            logger.info(f"✅ Completed ablation: {ablation_config['name']}")
            
        except Exception as e:
            logger.error(f"❌ Error in ablation {ablation_config['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Generate ablation table
    table = ablation_framework.generate_ablation_table(baseline_name='full_model')
    
    # Save results
    ablation_framework.save_results()
    
    logger.info("\n✅ Ablation studies completed")
    
    return ablation_framework.ablation_results


def generate_all_figures(
    sota_results: Dict[str, Any],
    ablation_results: List[Dict[str, Any]],
    output_dir: str
):
    """
    Generate all publication-quality figures.
    
    Args:
        sota_results: SOTA comparison results
        ablation_results: Ablation study results
        output_dir: Directory to save figures
    """
    logger.info("\n" + "="*80)
    logger.info("GENERATING PUBLICATION FIGURES")
    logger.info("="*80)
    
    figures_dir = os.path.join(output_dir, 'figures')
    visualizer = ResearchVisualizer(output_dir=figures_dir)
    
    try:
        # Figure 1: SOTA Comparison Bar Chart
        logger.info("Creating Figure 1: SOTA comparison bar chart")
        visualizer.plot_sota_comparison_bar(
            results=sota_results,
            metric='accuracy',
            filename='figure1_sota_comparison_accuracy'
        )
        
        # Figure 2: Radar Chart
        logger.info("Creating Figure 2: Multi-metric radar chart")
        visualizer.plot_radar_chart(
            results=sota_results,
            metrics=['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
            filename='figure2_radar_chart'
        )
        
        # Figure 3: Ablation Heatmap
        if ablation_results:
            logger.info("Creating Figure 3: Ablation study heatmap")
            visualizer.plot_ablation_heatmap(
                ablation_results=ablation_results,
                filename='figure3_ablation_heatmap'
            )
        
        # Figure 4: Metrics Table
        logger.info("Creating Figure 4: Metrics comparison table")
        visualizer.plot_metric_comparison_table(
            results=sota_results,
            filename='figure4_metrics_table'
        )
        
        # Create summary
        visualizer.create_figure_summary()
        
        logger.info(f"\n✅ All figures saved to: {figures_dir}")
        
    except Exception as e:
        logger.error(f"❌ Error generating figures: {str(e)}")
        import traceback
        traceback.print_exc()


def generate_comparison_tables(
    sota_results: Dict[str, Any],
    output_dir: str
):
    """
    Generate markdown tables for paper.
    
    Args:
        sota_results: SOTA comparison results
        output_dir: Directory to save tables
    """
    logger.info("\n" + "="*80)
    logger.info("GENERATING COMPARISON TABLES")
    logger.info("="*80)
    
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    # Table 1: Main Results
    table1 = "# Table 1: SOTA Model Comparison\n\n"
    table1 += "| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC |\n"
    table1 += "|-------|--------------|---------------|------------|--------------|----------|\n"
    
    for model_name, result in sota_results.items():
        agg = result['aggregated']
        table1 += f"| {model_name} | "
        table1 += f"{agg['accuracy']['mean']*100:.2f} ± {agg['accuracy']['std']*100:.2f} | "
        table1 += f"{agg['precision']['mean']*100:.2f} ± {agg['precision']['std']*100:.2f} | "
        table1 += f"{agg['recall']['mean']*100:.2f} ± {agg['recall']['std']*100:.2f} | "
        table1 += f"{agg['f1_score']['mean']*100:.2f} ± {agg['f1_score']['std']*100:.2f} | "
        
        if 'roc_auc' in agg:
            table1 += f"{agg['roc_auc']['mean']:.3f} ± {agg['roc_auc']['std']:.3f} |\n"
        else:
            table1 += "N/A |\n"
    
    table1_file = os.path.join(tables_dir, 'table1_sota_comparison.md')
    with open(table1_file, 'w') as f:
        f.write(table1)
    
    logger.info(f"✅ Table 1 saved to: {table1_file}")
    logger.info("\n" + table1)


def generate_research_report(
    sota_results: Dict[str, Any],
    ablation_results: List[Dict[str, Any]],
    output_dir: str
):
    """
    Generate comprehensive research report.
    
    Args:
        sota_results: SOTA comparison results
        ablation_results: Ablation study results
        output_dir: Output directory
    """
    logger.info("\n" + "="*80)
    logger.info("GENERATING RESEARCH REPORT")
    logger.info("="*80)
    
    report_file = os.path.join(output_dir, 'EXPERIMENT_REPORT.md')
    
    with open(report_file, 'w') as f:
        f.write("# Research Experiment Report\n\n")
        f.write(f"Generated: {np.datetime64('now')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This report presents the results of comprehensive experiments comparing ")
        f.write("the NeuroSnake architecture against SOTA baselines and ablation studies.\n\n")
        
        f.write("## SOTA Model Comparison\n\n")
        f.write("### Results Summary\n\n")
        
        for model_name, result in sorted(sota_results.items(), 
                                        key=lambda x: x[1]['aggregated']['accuracy']['mean'],
                                        reverse=True):
            agg = result['aggregated']
            f.write(f"**{model_name}**\n")
            f.write(f"- Accuracy: {agg['accuracy']['mean']*100:.2f}% ± {agg['accuracy']['std']*100:.2f}%\n")
            f.write(f"- Precision: {agg['precision']['mean']*100:.2f}% ± {agg['precision']['std']*100:.2f}%\n")
            f.write(f"- Recall: {agg['recall']['mean']*100:.2f}% ± {agg['recall']['std']*100:.2f}%\n")
            f.write(f"- F1-Score: {agg['f1_score']['mean']*100:.2f}% ± {agg['f1_score']['std']*100:.2f}%\n")
            f.write("\n")
        
        f.write("## Ablation Studies\n\n")
        if ablation_results:
            for ablation in ablation_results:
                f.write(f"### {ablation['name']}\n")
                f.write(f"Accuracy: {ablation['results']['accuracy']['mean']*100:.2f}% ")
                f.write(f"± {ablation['results']['accuracy']['std']*100:.2f}%\n\n")
        
        f.write("## Figures\n\n")
        f.write("All publication-quality figures are available in the `figures/` directory.\n\n")
        
        f.write("## Statistical Significance\n\n")
        f.write("Statistical comparisons between models can be performed using the ")
        f.write("`ExperimentalFramework.compare_models()` method.\n\n")
    
    logger.info(f"✅ Research report saved to: {report_file}")


def load_sample_data(n_samples: int = 100) -> tuple:
    """
    Load or generate sample data for testing.
    
    Args:
        n_samples: Number of samples to generate
    
    Returns:
        Tuple of (X, y)
    """
    logger.info(f"Generating {n_samples} sample images for testing...")
    
    # Generate random sample data (for testing without real data)
    X = np.random.rand(n_samples, 224, 224, 3).astype(np.float32)
    
    # Generate balanced labels
    y = np.zeros((n_samples, 2))
    y[:n_samples//2, 0] = 1  # No tumor
    y[n_samples//2:, 1] = 1  # Tumor
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    logger.info(f"Sample data shape: X={X.shape}, y={y.shape}")
    
    return X, y


def main():
    """Main experiment runner."""
    logger.info("\n" + "="*80)
    logger.info("BRAIN TUMOR DETECTION - RESEARCH EXPERIMENTS")
    logger.info("="*80)
    
    # Create output directory
    output_dir = os.path.join(config.BASE_DIR, 'research_results')
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Configuration
    n_folds = 5
    epochs = 5  # Reduced for testing; increase to 50+ for real experiments
    batch_size = 32
    
    # Step 1: Load data
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOADING DATA")
    logger.info("="*80)
    
    # TODO: Replace with actual data loading
    # from src.legacy.data_preprocessing import load_and_preprocess_data
    # X, y = load_and_preprocess_data()
    
    X, y = load_sample_data(n_samples=100)  # Sample data for testing
    
    # Step 2: Run SOTA comparison
    logger.info("\n" + "="*80)
    logger.info("STEP 2: SOTA MODEL COMPARISON")
    logger.info("="*80)
    
    sota_results = run_sota_comparison(
        X=X,
        y=y,
        output_dir=output_dir,
        n_folds=n_folds,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Step 3: Run ablation studies
    logger.info("\n" + "="*80)
    logger.info("STEP 3: ABLATION STUDIES")
    logger.info("="*80)
    
    ablation_results = run_ablation_studies(
        X=X,
        y=y,
        output_dir=output_dir,
        n_folds=n_folds,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Step 4: Generate figures
    logger.info("\n" + "="*80)
    logger.info("STEP 4: GENERATING FIGURES")
    logger.info("="*80)
    
    generate_all_figures(
        sota_results=sota_results,
        ablation_results=ablation_results,
        output_dir=output_dir
    )
    
    # Step 5: Generate tables
    logger.info("\n" + "="*80)
    logger.info("STEP 5: GENERATING TABLES")
    logger.info("="*80)
    
    generate_comparison_tables(
        sota_results=sota_results,
        output_dir=output_dir
    )
    
    # Step 6: Generate report
    logger.info("\n" + "="*80)
    logger.info("STEP 6: GENERATING REPORT")
    logger.info("="*80)
    
    generate_research_report(
        sota_results=sota_results,
        ablation_results=ablation_results,
        output_dir=output_dir
    )
    
    logger.info("\n" + "="*80)
    logger.info("✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"\nResults directory: {output_dir}")
    logger.info(f"Figures: {os.path.join(output_dir, 'figures')}")
    logger.info(f"Tables: {os.path.join(output_dir, 'tables')}")
    logger.info(f"Report: {os.path.join(output_dir, 'EXPERIMENT_REPORT.md')}")


if __name__ == '__main__':
    main()
