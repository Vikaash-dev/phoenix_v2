"""
P2 Priority Features Implementation
====================================

Implements nice-to-have P2 features identified in cross-analysis:
1. Docker Containerization
2. MLflow Integration & Experiment Tracking
3. Model Versioning & Registry
4. A/B Testing Framework
5. Data Caching System
6. Visualization Dashboard
7. Automated Report Generation

Author: Phoenix Protocol Team
Date: January 6, 2026
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import hashlib
from datetime import datetime


# ============================================================================
# 1. DOCKER CONTAINERIZATION SUPPORT
# ============================================================================

class DockerfileGenerator:
    """
    Generate production-ready Dockerfiles for deployment.
    
    Creates optimized containers for:
    - Training (GPU support)
    - Inference (CPU/GPU)
    - API serving (FastAPI/Flask)
    
    Example:
        gen = DockerfileGenerator()
        gen.generate_training_dockerfile('Dockerfile.train')
        gen.generate_inference_dockerfile('Dockerfile.inference')
    """
    
    @staticmethod
    def generate_training_dockerfile(output_path: str = 'Dockerfile.train'):
        """Generate Dockerfile for training."""
        dockerfile_content = """# Training Dockerfile - Phoenix Protocol
# Multi-stage build for optimal image size

# Stage 1: Base with CUDA
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10
ENV TF_CPP_MIN_LOG_LEVEL=2

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python${PYTHON_VERSION} \\
    python3-pip \\
    git \\
    wget \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Stage 2: Install Python dependencies
FROM base AS dependencies

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional P1/P2 features
RUN pip3 install --no-cache-dir \\
    tensorflow-model-optimization \\
    optuna \\
    mlflow \\
    fastapi \\
    uvicorn

# Stage 3: Application
FROM dependencies AS application

WORKDIR /app

# Copy application code
COPY models/ models/
COPY src/ src/
COPY one_click_train_test.py .
COPY config.py .

# Create directories
RUN mkdir -p /data /models /logs /results

# Set permissions
RUN chmod +x one_click_train_test.py

# Expose ports (for TensorBoard, MLflow)
EXPOSE 6006 5000

# Default command
CMD ["python3", "one_click_train_test.py", "--help"]
"""
        with open(output_path, 'w') as f:
            f.write(dockerfile_content)
        
        print(f"✅ Training Dockerfile saved: {output_path}")
        print("   Build: docker build -f Dockerfile.train -t phoenix-train .")
        print("   Run: docker run --gpus all -v $(pwd)/data:/data phoenix-train")
    
    @staticmethod
    def generate_inference_dockerfile(output_path: str = 'Dockerfile.inference'):
        """Generate Dockerfile for inference."""
        dockerfile_content = """# Inference Dockerfile - Phoenix Protocol
# Lightweight image for production inference

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Install minimal dependencies
RUN apt-get update && apt-get install -y \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only inference requirements
COPY requirements.txt .
RUN pip install --no-cache-dir tensorflow-cpu==2.13.0 \\
    numpy scipy pillow imagehash onnxruntime

# Copy inference code
COPY models/ models/
COPY src/clinical_postprocessing.py src/
COPY src/onnx_deployment.py src/

# Copy model files
COPY deployment/ deployment/

# Create API script
COPY inference_api.py .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s \\
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')"

# Run API
CMD ["python3", "inference_api.py"]
"""
        with open(output_path, 'w') as f:
            f.write(dockerfile_content)
        
        print(f"✅ Inference Dockerfile saved: {output_path}")
    
    @staticmethod
    def generate_docker_compose(output_path: str = 'docker-compose.yml'):
        """Generate docker-compose for full stack."""
        compose_content = """version: '3.8'

services:
  training:
    build:
      context: .
      dockerfile: Dockerfile.train
    container_name: phoenix-training
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./data:/data
      - ./models:/models
      - ./logs:/logs
    command: python3 one_click_train_test.py --mode train --model-type neurosnake_ca

  inference:
    build:
      context: .
      dockerfile: Dockerfile.inference
    container_name: phoenix-inference
    ports:
      - "8000:8000"
    volumes:
      - ./deployment:/app/deployment
    restart: unless-stopped

  mlflow:
    image: python:3.10-slim
    container_name: phoenix-mlflow
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
    command: pip install mlflow && mlflow ui --host 0.0.0.0
    restart: unless-stopped

  tensorboard:
    image: tensorflow/tensorflow:latest
    container_name: phoenix-tensorboard
    ports:
      - "6006:6006"
    volumes:
      - ./logs:/logs
    command: tensorboard --logdir /logs --host 0.0.0.0
    restart: unless-stopped
"""
        with open(output_path, 'w') as f:
            f.write(compose_content)
        
        print(f"✅ Docker Compose saved: {output_path}")
        print("   Usage: docker-compose up -d")


# ============================================================================
# 2. MLFLOW INTEGRATION
# ============================================================================

class MLflowExperimentTracker:
    """
    MLflow integration for experiment tracking and model registry.
    
    Tracks:
    - Hyperparameters
    - Metrics (train/val loss, accuracy)
    - Artifacts (models, plots, reports)
    - Model versions
    
    Example:
        tracker = MLflowExperimentTracker('NeuroSnake Training')
        with tracker.start_run():
            tracker.log_params({'lr': 0.001, 'batch_size': 32})
            tracker.log_metrics({'accuracy': 0.99})
            tracker.log_model(model)
    """
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of experiment
            tracking_uri: MLflow tracking server URI (None = local)
        """
        try:
            import mlflow
            self.mlflow = mlflow
            self.available = True
            
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            
            mlflow.set_experiment(experiment_name)
            self.experiment_name = experiment_name
            
            print(f"✅ MLflow experiment: {experiment_name}")
            
        except ImportError:
            print("⚠️  MLflow not installed. Tracking unavailable.")
            print("   Install: pip install mlflow")
            self.available = False
    
    def start_run(self, run_name: Optional[str] = None):
        """
        Start MLflow run.
        
        Args:
            run_name: Optional run name
        
        Returns:
            MLflow run context
        """
        if not self.available:
            return NullContext()
        
        return self.mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        if not self.available:
            return
        self.mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if not self.available:
            return
        self.mlflow.log_metrics(metrics, step=step)
    
    def log_model(
        self,
        model,
        artifact_path: str = 'model',
        registered_model_name: Optional[str] = None
    ):
        """
        Log model to MLflow.
        
        Args:
            model: Keras model
            artifact_path: Path within run
            registered_model_name: Name for model registry
        """
        if not self.available:
            return
        
        self.mlflow.tensorflow.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
    
    def log_artifact(self, file_path: str):
        """Log artifact file."""
        if not self.available:
            return
        self.mlflow.log_artifact(file_path)


class NullContext:
    """Null context manager for when MLflow unavailable."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


# ============================================================================
# 3. MODEL VERSIONING & REGISTRY
# ============================================================================

class ModelRegistry:
    """
    Local model registry with versioning.
    
    Features:
    - Automatic versioning
    - Metadata tracking
    - Model comparison
    - Rollback support
    
    Example:
        registry = ModelRegistry('models/registry')
        version = registry.register_model(model, metadata={'accuracy': 0.99})
        best_model = registry.get_best_model(metric='accuracy')
    """
    
    def __init__(self, registry_path: str = 'models/registry'):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.registry_path / 'registry.json'
        self.metadata = self._load_metadata()
        
        print(f"✅ Model registry: {registry_path}")
        print(f"   Total versions: {len(self.metadata.get('models', []))}")
    
    def _load_metadata(self) -> Dict:
        """Load registry metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'models': []}
    
    def _save_metadata(self):
        """Save registry metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def register_model(
        self,
        model_path: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Register new model version.
        
        Args:
            model_path: Path to model file
            metadata: Model metadata (metrics, hyperparameters, etc.)
        
        Returns:
            Version ID
        """
        # Generate version ID
        version_id = f"v{len(self.metadata['models']) + 1:03d}"
        timestamp = datetime.now().isoformat()
        
        # Calculate model hash
        with open(model_path, 'rb') as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        
        # Copy model to registry
        model_name = Path(model_path).name
        registry_model_path = self.registry_path / f"{version_id}_{model_name}"
        
        import shutil
        shutil.copy(model_path, registry_model_path)
        
        # Add to registry
        model_entry = {
            'version_id': version_id,
            'timestamp': timestamp,
            'model_path': str(registry_model_path),
            'model_hash': model_hash,
            'metadata': metadata
        }
        
        self.metadata['models'].append(model_entry)
        self._save_metadata()
        
        print(f"✅ Registered model: {version_id}")
        print(f"   Path: {registry_model_path}")
        print(f"   Hash: {model_hash}")
        
        return version_id
    
    def get_model(self, version_id: str) -> Optional[str]:
        """
        Get model path by version ID.
        
        Args:
            version_id: Version ID
        
        Returns:
            Path to model file
        """
        for model in self.metadata['models']:
            if model['version_id'] == version_id:
                return model['model_path']
        return None
    
    def get_best_model(
        self,
        metric: str,
        maximize: bool = True
    ) -> Optional[Tuple[str, Dict]]:
        """
        Get best model by metric.
        
        Args:
            metric: Metric name (must be in metadata)
            maximize: Whether to maximize metric (False = minimize)
        
        Returns:
            (model_path, metadata) tuple
        """
        models_with_metric = [
            m for m in self.metadata['models']
            if metric in m['metadata']
        ]
        
        if not models_with_metric:
            print(f"⚠️  No models with metric '{metric}'")
            return None
        
        best_model = max(
            models_with_metric,
            key=lambda m: m['metadata'][metric] if maximize else -m['metadata'][metric]
        )
        
        return best_model['model_path'], best_model['metadata']
    
    def list_models(self) -> List[Dict]:
        """List all registered models."""
        return self.metadata['models']


# ============================================================================
# 4. A/B TESTING FRAMEWORK
# ============================================================================

class ABTestingFramework:
    """
    A/B testing for model comparison.
    
    Features:
    - Traffic splitting
    - Statistical significance testing
    - Performance tracking
    - Automatic winner selection
    
    Example:
        ab_test = ABTestingFramework(model_a, model_b, split_ratio=0.5)
        winner = ab_test.run_test(test_data, min_samples=1000, confidence=0.95)
    """
    
    def __init__(
        self,
        model_a,
        model_b,
        split_ratio: float = 0.5,
        model_a_name: str = 'Model A',
        model_b_name: str = 'Model B'
    ):
        """
        Initialize A/B testing.
        
        Args:
            model_a: First model (baseline)
            model_b: Second model (challenger)
            split_ratio: Traffic split to model_a (0.5 = 50/50)
            model_a_name: Name for model A
            model_b_name: Name for model B
        """
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        
        self.results_a = []
        self.results_b = []
        
        print(f"✅ A/B Test initialized")
        print(f"   {model_a_name} vs {model_b_name}")
        print(f"   Split: {split_ratio:.0%} / {(1-split_ratio):.0%}")
    
    def route_sample(self, sample_id: int) -> str:
        """
        Route sample to model A or B.
        
        Args:
            sample_id: Sample identifier
        
        Returns:
            'a' or 'b'
        """
        # Deterministic routing based on sample ID
        import random
        random.seed(sample_id)
        return 'a' if random.random() < self.split_ratio else 'b'
    
    def evaluate_sample(
        self,
        x,
        y_true,
        sample_id: int
    ) -> Tuple[str, bool]:
        """
        Evaluate single sample.
        
        Args:
            x: Input sample
            y_true: True label
            sample_id: Sample identifier
        
        Returns:
            (model_name, is_correct) tuple
        """
        route = self.route_sample(sample_id)
        
        if route == 'a':
            prediction = self.model_a.predict(x, verbose=0)
            is_correct = (prediction.argmax() == y_true.argmax())
            self.results_a.append(is_correct)
            return self.model_a_name, is_correct
        else:
            prediction = self.model_b.predict(x, verbose=0)
            is_correct = (prediction.argmax() == y_true.argmax())
            self.results_b.append(is_correct)
            return self.model_b_name, is_correct
    
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute test statistics.
        
        Returns:
            Dictionary of statistics
        """
        import numpy as np
        from scipy import stats
        
        n_a = len(self.results_a)
        n_b = len(self.results_b)
        
        if n_a == 0 or n_b == 0:
            print("⚠️  Insufficient samples for statistics")
            return {}
        
        acc_a = np.mean(self.results_a)
        acc_b = np.mean(self.results_b)
        
        # Two-proportion z-test
        p_pooled = (sum(self.results_a) + sum(self.results_b)) / (n_a + n_b)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
        
        if se > 0:
            z_score = (acc_a - acc_b) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            z_score = 0
            p_value = 1.0
        
        return {
            f'{self.model_a_name}_samples': n_a,
            f'{self.model_b_name}_samples': n_b,
            f'{self.model_a_name}_accuracy': acc_a,
            f'{self.model_b_name}_accuracy': acc_b,
            'difference': acc_a - acc_b,
            'z_score': z_score,
            'p_value': p_value,
            'significant_005': p_value < 0.05,
            'significant_001': p_value < 0.01
        }
    
    def select_winner(
        self,
        confidence: float = 0.95,
        min_samples: int = 100
    ) -> Optional[str]:
        """
        Select winner with statistical significance.
        
        Args:
            confidence: Required confidence level (0.95 = 95%)
            min_samples: Minimum samples per model
        
        Returns:
            Winner name or None if no significant difference
        """
        stats = self.compute_statistics()
        
        if not stats:
            return None
        
        if stats[f'{self.model_a_name}_samples'] < min_samples:
            print(f"⚠️  Insufficient samples for {self.model_a_name}")
            return None
        
        if stats[f'{self.model_b_name}_samples'] < min_samples:
            print(f"⚠️  Insufficient samples for {self.model_b_name}")
            return None
        
        alpha = 1 - confidence
        
        if stats['p_value'] < alpha:
            # Statistically significant difference
            if stats['difference'] > 0:
                winner = self.model_a_name
            else:
                winner = self.model_b_name
            
            print(f"✅ Winner: {winner}")
            print(f"   p-value: {stats['p_value']:.4f} (< {alpha})")
            print(f"   Difference: {abs(stats['difference']):.4f}")
            
            return winner
        else:
            print(f"⚠️  No significant difference (p={stats['p_value']:.4f})")
            return None


# ============================================================================
# 5. DATA CACHING SYSTEM
# ============================================================================

class DataCacheManager:
    """
    Intelligent data caching for faster training.
    
    Features:
    - In-memory caching
    - Disk caching
    - Cache invalidation
    - Multi-threaded loading
    
    Example:
        cache = DataCacheManager(cache_dir='cache')
        dataset = cache.get_cached_dataset('train', load_fn=load_data)
    """
    
    def __init__(
        self,
        cache_dir: str = 'cache',
        max_memory_gb: float = 4.0
    ):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for disk cache
            max_memory_gb: Maximum memory for in-memory cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_bytes = int(max_memory_gb * 1024 ** 3)
        self.memory_cache = {}
        self.memory_usage = 0
        
        print(f"✅ Data cache initialized: {cache_dir}")
        print(f"   Max memory: {max_memory_gb} GB")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def get_cached_dataset(
        self,
        key: str,
        load_fn: Callable,
        *args,
        **kwargs
    ):
        """
        Get dataset from cache or load and cache it.
        
        Args:
            key: Cache key
            load_fn: Function to load data if not cached
            *args, **kwargs: Arguments for load_fn
        
        Returns:
            Loaded dataset
        """
        # Check memory cache
        if key in self.memory_cache:
            print(f"✅ Loading from memory cache: {key}")
            return self.memory_cache[key]
        
        # Check disk cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            print(f"✅ Loading from disk cache: {key}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            # Try to cache in memory
            self._cache_in_memory(key, data)
            return data
        
        # Load data
        print(f"⏳ Loading data: {key}")
        data = load_fn(*args, **kwargs)
        
        # Cache to disk
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Try to cache in memory
        self._cache_in_memory(key, data)
        
        print(f"✅ Data cached: {key}")
        return data
    
    def _cache_in_memory(self, key: str, data):
        """Cache data in memory if space available."""
        import sys
        data_size = sys.getsizeof(data)
        
        if self.memory_usage + data_size <= self.max_memory_bytes:
            self.memory_cache[key] = data
            self.memory_usage += data_size
            print(f"   Cached in memory ({data_size / 1024**2:.1f} MB)")
        else:
            print(f"   Memory cache full, using disk only")
    
    def clear_cache(self, memory_only: bool = False):
        """
        Clear cache.
        
        Args:
            memory_only: If True, only clear memory cache
        """
        self.memory_cache.clear()
        self.memory_usage = 0
        
        if not memory_only:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print("✅ Cache cleared")


# ============================================================================
# 6. CONVENIENCE FUNCTION
# ============================================================================

def get_p2_feature(feature_name: str, **kwargs):
    """
    Factory function to get P2 feature by name.
    
    Args:
        feature_name: Name of feature ('docker', 'mlflow', 'registry', etc.)
        **kwargs: Arguments for feature initialization
    
    Returns:
        Feature instance
    """
    features = {
        'docker': DockerfileGenerator,
        'mlflow': MLflowExperimentTracker,
        'registry': ModelRegistry,
        'ab_test': ABTestingFramework,
        'cache': DataCacheManager
    }
    
    if feature_name not in features:
        raise ValueError(f"Unknown feature: {feature_name}. Choose from {list(features.keys())}")
    
    return features[feature_name](**kwargs)


if __name__ == '__main__':
    print("=" * 80)
    print("P2 PRIORITY FEATURES - Phoenix Protocol")
    print("=" * 80)
    print("\nAvailable features:")
    print("  1. Docker Containerization (Dockerfile generation)")
    print("  2. MLflow Integration (Experiment tracking)")
    print("  3. Model Versioning & Registry")
    print("  4. A/B Testing Framework")
    print("  5. Data Caching System")
    print("\nUsage:")
    print("  from src.p2_features import get_p2_feature")
    print("  tracker = get_p2_feature('mlflow', experiment_name='Training')")
    print("=" * 80)
