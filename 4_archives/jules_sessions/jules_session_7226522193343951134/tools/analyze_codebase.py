"""
Codebase Analyzer for NeuroSnake Project.
Extracts model statistics and architecture details for LLM context generation.
"""

import os
import sys
import json
import inspect
import tensorflow as tf
from tensorflow import keras

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.neuro_snake_hyper import NeuroSnakeHyperLiquidModel
from src.models.neuro_snake_kan import NeuroSnakeKANModel
from src.models.neuro_snake_liquid import NeuroSnakeLiquidModel
from src.models.neuro_mamba_spectral import NeuroSnakeSpectralModel

def get_model_stats(model_factory, name):
    try:
        model = model_factory(input_shape=(224, 224, 3))
        
        # Count layers by type
        layer_types = {}
        for layer in model.layers:
            t = type(layer).__name__
            layer_types[t] = layer_types.get(t, 0) + 1
            
        return {
            "name": name,
            "total_params": model.count_params(),
            "layers": layer_types,
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "status": "Operational"
        }
    except Exception as e:
        return {
            "name": name,
            "status": "Failed",
            "error": str(e)
        }

def analyze_codebase():
    stats = []
    
    models_to_analyze = [
        ("Hyper-Liquid Snake", NeuroSnakeHyperLiquidModel.create_model),
        ("NeuroSnake-KAN", NeuroSnakeKANModel.create_model),
        ("Liquid-Snake", NeuroSnakeLiquidModel.create_model),
        ("NeuroSnake-Spectral", NeuroSnakeSpectralModel.create_model)
    ]
    
    for name, factory in models_to_analyze:
        stats.append(get_model_stats(factory, name))
        
    # Output JSON for LLM ingestion
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    analyze_codebase()
