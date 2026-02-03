from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
import json
import yaml
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for PhoenixMambaV2 Model Architecture."""
    d_model_stages: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    d_state: int = 16
    d_mamba_expand: int = 2
    dt_rank_factor: int = 16
    num_classes: int = 4
    num_heads_attention: int = 4
    dropout_rate: float = 0.2

    # Input shape: (Batch, Depth, Height, Width, Channels)
    # Depth is for 2.5D slices
    input_depth: int = 3

@dataclass
class DataConfig:
    """Configuration for Data Pipeline."""
    img_size: Tuple[int, int] = (224, 224)
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    batch_size: int = 32
    class_names: List[str] = field(default_factory=lambda: ['glioma', 'meningioma', 'notumor', 'pituitary'])
    validation_split: float = 0.2
    seed: int = 42

@dataclass
class TrainingConfig:
    """Configuration for Training Loop."""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    label_smoothing: float = 0.1
    patience: int = 10
    mixed_precision: bool = True

@dataclass
class Config:
    """Master Configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML or JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            if path.suffix in ('.yaml', '.yml'):
                data = yaml.safe_load(f)
            elif path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError("Unsupported config format. Use YAML or JSON.")

        # Helper to recursively instantiate dataclasses
        # Note: robust implementation would use something like dacite or pydantic
        # For simplicity in this project, we assume the structure matches
        return cls(
            model=ModelConfig(**data.get('model', {})),
            data=DataConfig(**data.get('data', {})),
            training=TrainingConfig(**data.get('training', {}))
        )

    def save(self, path: Union[str, Path]):
        """Save configuration to file."""
        import dataclasses
        data = dataclasses.asdict(self)
        path = Path(path)

        with open(path, 'w') as f:
            if path.suffix in ('.yaml', '.yml'):
                yaml.dump(data, f)
            elif path.suffix == '.json':
                json.dump(data, f, indent=2)
