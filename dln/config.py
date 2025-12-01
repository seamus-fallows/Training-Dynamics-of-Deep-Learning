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
    # Dictionary for dataset-specific parameters e.g. mean, std.
    params: dict[str, Any] | None


@dataclass
class TrainingConfig:
    lr: float
    batch_size: int | None
    optimizer: str
    optimizer_params: dict[str, Any] | None
    criterion: str
    model_seed: int


@dataclass
class ExperimentMeta:
    name: str


@dataclass
class SwitchConfig:
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
    switch: SwitchConfig | None = None
    metrics: list[str] = field(default_factory=list)


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
