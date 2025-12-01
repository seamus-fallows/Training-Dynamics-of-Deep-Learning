from dataclasses import dataclass, field
from typing import Any


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
    type: str
    num_samples: int
    test_split: float | None
    data_seed: int
    params: dict[str, Any] | None  # Dictionary for dataset-specific parameters.


@dataclass
class TrainingConfig:
    """Per-model training parameters"""

    lr: float
    batch_size: int
    optimizer: str
    optimizer_params: dict[str, Any] | None
    criterion: str
    model_seed: int


@dataclass
class ExperimentMeta:
    name: str


@dataclass
class SwitchConfig:
    """Optional batch size switching during training."""

    step: int | None
    batch_size: int | None


@dataclass
class ExperimentConfig:
    experiment: ExperimentMeta
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    max_steps: int
    evaluate_every: int
    metrics: list[str] | None = None
    switch: SwitchConfig | None = None


@dataclass
class ComparativeExperimentConfig:
    experiment: ExperimentMeta
    model_a: ModelConfig
    model_b: ModelConfig
    data: DataConfig
    training_a: TrainingConfig
    training_b: TrainingConfig
    max_steps: int
    evaluate_every: int
    model_metrics: list[str] = field(default_factory=list)
    comparative_metrics: list[str] = field(default_factory=list)
    shared: dict[str, Any] = field(default_factory=dict)
