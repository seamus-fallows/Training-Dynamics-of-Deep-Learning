from dataclasses import dataclass, field
from typing import Literal, Dict, Any, List


@dataclass
class ModelConfig:
    in_dim: int
    out_dim: int
    num_hidden: int
    hidden_size: int
    gamma: float
    bias: bool = False


@dataclass
class DataConfig:
    type: Literal["diagonal_teacher", "random_teacher"]
    num_samples: int
    test_split: float | None
    data_seed: int
    params: Dict[str, Any] | None  # Dictionary for dataset-specific parameters.


@dataclass
class ModelTrainingConfig:
    """Per-model training parameters"""

    lr: float
    batch_size: int
    optimizer: str
    optimizer_params: Dict[str, Any] | None
    criterion: str
    model_seed: int
    preload_data: bool


@dataclass
class TrainingConfig(ModelTrainingConfig):
    """Full training config for single-model experiments."""

    max_steps: int
    evaluate_every: int


@dataclass
class ExperimentMeta:
    name: str


@dataclass
class ExperimentConfig:
    experiment: ExperimentMeta
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig


@dataclass
class ComparativeExperimentConfig:
    experiment: ExperimentMeta
    model_a: ModelConfig
    model_b: ModelConfig
    data: DataConfig
    training_a: ModelTrainingConfig
    training_b: ModelTrainingConfig
    max_steps: int
    evaluate_every: int
    metrics: List[str] = field(default_factory=lambda: ["param_distance"])
    shared: Dict[str, Any] = field(default_factory=dict)
